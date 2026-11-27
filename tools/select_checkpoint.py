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


def choose(cov, domains):
    """(epoch, mean) maximizing the mean over `domains`, ties -> earliest."""
    full = {e: sum(v[d] for d in domains) / len(domains)
            for e, v in cov.items() if all(d in v for d in domains)}
    if not full:
        return None, None
    best = max(full.values())
    epoch = min(e for e, m in full.items() if m == best)
    return epoch, best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--results", default=os.path.join(ROOT, "cache", "results"))
    ap.add_argument("--expid", default="", help="override the config's expid (smoke runs)")
    ap.add_argument("--val-domains", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--print-epoch", action="store_true")
    ap.add_argument("--min-epochs", type=int, default=10,
                    help="snapshots that must have a number on every domain")
    args = ap.parse_args()
    cfg = read_config(args.config)
    doms = train_domains(cfg)
    if args.val_domains:
        print(",".join(f"{d.lower()}@val_ood" for d in doms))
        return 0
    expid = args.expid or cfg["expid"]
    dumps = []
    for p in glob.glob(os.path.join(args.results, f"{expid}_sweep", "results_*.json")):
        with open(p) as f:
            dumps.append(json.load(f))
    cov = val_coverage(dumps, doms)
    epoch, mean = choose(cov, doms)
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
            mark = "  <- selected" if e == epoch else ""
            print(f"  epoch {e:4d}  {row}{mark}")
        print(f"selected epoch {epoch} (mean Val-OOD coverage {mean})")
    if args.write and epoch is not None:
        out = os.path.join(args.results, expid)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "selected_checkpoint.json"), "w") as f:
            json.dump(dict(expid=expid, epoch=epoch, mean_val_ood=mean, domains=doms,
                           per_domain=cov[epoch], snapshots=complete,
                           rule="max mean Val-OOD coverage over training domains, ties -> earlier"),
                      f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
