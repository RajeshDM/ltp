#!/usr/bin/env python3
"""The 2026-11 checkpoint rule: best Val-OOD coverage, ties -> earlier (C1).

    python tools/select_checkpoint.py --config configs/p2_sd_jc_gripper.yaml [--val-domains]
                                      [--write] [--print-epoch]

For one config, reads the selection sweep main.py wrote to
cache/results/<expid>_sweep/results_*.json (periodic snapshots tested on each
TRAINING domain's @val_ood split) and picks the snapshot with the highest MEAN
Val-OOD coverage over the training domains. Only epochs every training domain
has a number for are eligible. Ties go to the earlier epoch. Test coverage
never enters, so the reported number is test-blind; for a leave-one-out
config the held-out domain is not a training domain, so zero-shot selection
uses no data from it at all.

  --val-domains  print the main.py --test-domains value for the sweep and exit
  --write        record the choice in cache/results/<expid>/selected_checkpoint.json
  --print-epoch  print only the chosen epoch (for scripts); exit 1 if none
Pinned by tests/test_select_checkpoint.py.
"""
import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_config(path):
    import yaml
    cfg = {}
    here = path
    while here:
        with open(here) as f:
            c = yaml.safe_load(f) or {}
        base = c.pop("base", None)
        cfg = {**c, **cfg}
        here = os.path.join(os.path.dirname(here), base) if base else None
    return cfg


def train_domains(cfg):
    return [d.split(":")[0].capitalize() for d in cfg["domains"].split(",") if d]


def val_coverage(dumps, domains):
    """{epoch: {domain: coverage}} on each domain's @val_ood; newest dump wins."""
    out = {}
    for dump in sorted(dumps, key=lambda d: d.get("timestamp", "")):
        for key, entries in dump.get("results", {}).items():
            dom = next((d for d in domains if f"{d}@val_ood" in key), None)
            if dom is None:
                continue
            for e in entries:
                learned = next((m for p, m in e.get("metrics", {}).items() if "LEARNED" in p), None)
                if learned is not None:
                    out.setdefault(int(e["epoch"]), {})[dom] = learned["success_rate_with_monitor"]
    return out


def choose(cov, domains, loss=None):
    """(epoch, mean) maximizing the mean over `domains`. Ties -> the lowest
    validation (Val-OOD) loss when `loss` ({epoch: loss}) has every tied epoch,
    then the earliest epoch."""
    full = {e: sum(v[d] for d in domains) / len(domains)
            for e, v in cov.items() if all(d in v for d in domains)}
    if not full:
        return None, None
    best = max(full.values())
    tied = sorted(e for e, m in full.items() if m == best)
    if loss and len(tied) > 1 and all(e in loss for e in tied):
        return min(tied, key=lambda e: (loss[e], e)), best
    return tied[0], best


def domain_val_losses(root, cfg_name, domains, every=50):
    """{epoch: {domain: mean Val-OOD loss}} at snapshot epochs (multiples of
    `every`, > 0) from the config's newest training log: the `v` value of each
    domain on the `domain:` line printed after every `Epoch N/M loss:` line
    (teacher-forced cross-entropy on the expert path of the <dom>_val_ood
    problems, averaged per state)."""
    logs = sorted(glob.glob(os.path.join(root, "logs", f"{cfg_name}_mode_train*.log")),
                  key=os.path.getmtime)
    out, epoch = {}, None
    if not logs:
        return out
    with open(logs[-1], errors="replace") as f:
        for line in f:
            m = re.match(r"Epoch (\d+)/\d+ loss:", line)
            if m:
                epoch = int(m.group(1))
                continue
            if epoch is not None and line.lstrip().startswith("domain:"):
                if epoch > 0 and epoch % every == 0:
                    vals = dict(re.findall(r"([A-Za-z0-9_]+):t[\d.eE+-]+/v([\d.eE+-]+)", line))
                    if all(d in vals for d in domains):
                        out[epoch] = {d: float(vals[d]) for d in domains}
                epoch = None
    return out


def choose_by_loss(losses, domains):
    """(epoch, score) minimizing the mean over domains of each domain's Val-OOD
    loss RELATIVE TO ITS OWN BEST snapshot (L_d(e) / min_e' L_d(e')), ties ->
    earliest. Scale-free, so every training domain weighs the same: raw losses
    differ by domain (Rovers ~3.4 vs Gripper ~0.2 per state at epoch 50 of a
    seven-domain run - more parameter slots, more object-choice terms), and a
    plain mean would let the largest one decide. Per-domain losses are already
    per-state means, so domains with more or longer plans get no extra weight.
    For one domain this is plain argmin."""
    if not losses:
        return None, None
    best_d = {d: min(v[d] for v in losses.values()) for d in domains}
    score = {e: sum(v[d] / best_d[d] if best_d[d] > 0 else 1.0 for d in domains) / len(domains)
             for e, v in losses.items()}
    best = min(score.values())
    return min(e for e, m in score.items() if m == best), best


def relative_loss_scores(losses, domains):
    """{epoch: mean over domains of L_d(e) / min_e' L_d(e')} - the scale-free
    score choose_by_loss minimizes, for use as choose()'s tie-break."""
    if not losses:
        return {}
    best_d = {d: min(v[d] for v in losses.values()) for d in domains}
    return {e: sum(v[d] / best_d[d] if best_d[d] > 0 else 1.0 for d in domains) / len(domains)
            for e, v in losses.items()}


def active_rule():
    """The campaign-wide rule: the first word of campaign/SELECTION_RULE."""
    path = os.path.join(ROOT, "campaign", "SELECTION_RULE")
    try:
        with open(path) as f:
            rule = f.read().split()[0]
        return rule if rule in ("loss", "coverage", "hybrid") else "loss"
    except (OSError, IndexError):
        return "loss"


def val_losses(root, cfg_name):
    """{epoch: Val-OOD loss} from the config's newest training log (the 'val'
    entry of each `Epoch N/M loss: {...}` line; with --val-protocol split it is
    the loss on the <dom>_val_ood problems, summed over batches)."""
    logs = sorted(glob.glob(os.path.join(root, "logs", f"{cfg_name}_mode_train*.log")),
                  key=os.path.getmtime)
    out = {}
    if logs:
        with open(logs[-1], errors="replace") as f:
            for line in f:
                m = re.match(r"Epoch (\d+)/\d+ loss: \{[^}]*'val': ([\d.eE+-]+)", line)
                if m:
                    out[int(m.group(1))] = float(m.group(2))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--results", default=os.path.join(ROOT, "cache", "results"))
    ap.add_argument("--expid", default="", help="override the config's expid (smoke runs)")
    ap.add_argument("--val-domains", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--print-epoch", action="store_true")
    ap.add_argument("--rule", choices=["loss", "coverage", "hybrid"], default=active_rule(),
                    help="default: campaign/SELECTION_RULE (one switch for every tool). "
                         "loss: lowest relative Val-OOD loss over the training domains, from "
                         "the training log - no sweep; coverage: best mean Val-OOD coverage from "
                         "the <expid>_sweep dumps, ties -> earlier; hybrid: best coverage, ties "
                         "-> lowest relative Val-OOD loss, then earlier"),
    ap.add_argument("--tiebreak", choices=["loss", "earliest"], default="earliest",
                    help="among snapshots with equal best coverage: lowest Val-OOD loss, "
                         "or earliest epoch")
    ap.add_argument("--min-epochs", type=int, default=10,
                    help="snapshots that must have a number on every domain")
    args = ap.parse_args()
    cfg = read_config(args.config)
    doms = train_domains(cfg)
    if args.val_domains:
        print(",".join(f"{d.lower()}@val_ood" for d in doms))
        return 0
    expid = args.expid or cfg["expid"]
    if args.rule == "loss":
        cfg_name = os.path.basename(args.config)[:-5]
        every = int(cfg.get("checkpoint_every") or 50)
        losses = domain_val_losses(ROOT, cfg_name, doms, every)
        epoch, mean = choose_by_loss(losses, doms)
        n = len(losses)
        if args.print_epoch:
            if epoch is None or n < args.min_epochs:
                print(f"{expid}: {n} snapshot epochs with Val-OOD loss (need {args.min_epochs})",
                      file=sys.stderr)
                return 1
            print(epoch)
        else:
            print(f"{expid}: {n} snapshot epochs with Val-OOD loss on all {len(doms)} training domain(s)")
            for e in sorted(losses):
                row = " ".join(f"{losses[e][d]:6.3f}" for d in doms)
                print(f"  epoch {e:4d}  {row}" + ("  <- selected" if e == epoch else ""))
            print(f"selected epoch {epoch} (mean Val-OOD loss {mean})")
        if args.write and epoch is not None:
            out = os.path.join(args.results, expid)
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "selected_checkpoint.json"), "w") as f:
                json.dump(dict(expid=expid, epoch=epoch, mean_val_ood_loss=mean, domains=doms,
                               per_domain_loss=losses[epoch], snapshots=n,
                               rule="min mean per-domain Val-OOD loss (teacher-forced) over "
                                    "training domains, ties -> earlier"), f, indent=1)
        return 0
    dumps = []
    for p in glob.glob(os.path.join(args.results, f"{expid}_sweep", "results_*.json")):
        with open(p) as f:
            dumps.append(json.load(f))
    cov = val_coverage(dumps, doms)
    cfg_name = os.path.basename(args.config)[:-5]
    loss = val_losses(ROOT, cfg_name) if args.tiebreak == "loss" else None
    if args.rule == "hybrid":
        loss = relative_loss_scores(domain_val_losses(ROOT, cfg_name, doms,
                                                      int(cfg.get("checkpoint_every") or 50)), doms)
    epoch, mean = choose(cov, doms, loss)
    complete = sum(all(d in v for d in doms) for v in cov.values())
    if args.print_epoch:
        if epoch is None or complete < args.min_epochs:
            print(f"{expid}: {complete} complete snapshots (need {args.min_epochs})", file=sys.stderr)
            return 1
        print(epoch)
    else:
        print(f"{expid}: {complete} snapshots with Val-OOD coverage on all {len(doms)} training domain(s)")
        for e in sorted(cov):
            row = " ".join(f"{cov[e].get(d, float('nan')):5.1f}" for d in doms)
            if loss:
                row += f"   val loss {loss.get(e, float('nan')):8.2f}"
            mark = "  <- selected" if e == epoch else ""
            print(f"  epoch {e:4d}  {row}{mark}")
        print(f"selected epoch {epoch} (mean Val-OOD coverage {mean})")
    if args.write and epoch is not None:
        out = os.path.join(args.results, expid)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "selected_checkpoint.json"), "w") as f:
            json.dump(dict(expid=expid, epoch=epoch, mean_val_ood=mean, domains=doms,
                           per_domain=cov[epoch], snapshots=complete,
                           rule=("max mean Val-OOD coverage over training domains, ties -> "
                                 + ("lowest relative Val-OOD loss, then earlier" if loss else "earlier"))),
                      f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
