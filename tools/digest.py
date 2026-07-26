#!/usr/bin/env python3
"""One compact, committable summary of every result so far.

Purpose: replace copy-pasting terminal logs. Run this after any batch of
runs, commit the output file, push. The digest is small and diffable, so
the whole result history lives in git and can be read without re-running
anything.

    python tools/digest.py                  # writes cache/results/DIGEST.md
    python tools/digest.py --print          # also dump it to stdout
    python tools/digest.py --domains visitall_ipcc,gripper_ipcc   # filter

Columns per row: coverage with/without monitor, plan quality, V1 (top-1
proposal validity), and the random floor for the same domain/split when
tools/random_policy_baseline.py has been run. ZS marks zero-shot rows -
those are the ones the paper's claims rest on, so they are listed first
and every tested checkpoint is shown (zero-shot is checkpoint-spiky).
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_dumps(results_dir):
    dumps = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*", "results_*.json"))):
        try:
            with open(p) as f:
                d = json.load(f)
            d["_path"] = p
            dumps.append(d)
        except Exception as e:
            print(f"WARN: unreadable {p}: {e}", file=sys.stderr)
    # keep the newest dump per (experiment, train_domain): re-tests supersede
    latest = {}
    for d in dumps:
        key = (d.get("experiment"), d.get("train_domain"))
        if key not in latest or d.get("timestamp", "") > latest[key].get("timestamp", ""):
            latest[key] = d
    return list(latest.values())


def load_floor(results_dir):
    path = os.path.join(results_dir, "results_random_policy.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    # keys are '<Domain>@<split>' -> coverage; normalize to our row labels
    out = {}
    for key, rec in raw.items():
        dom, split = key.rsplit("@", 1)
        label = dom if split == "test" else f"{dom}@train"
        out[label] = rec.get("coverage")
    return out


def rows_from(dump, floor):
    """(zero_shot, domain_label, metric, epoch, cov_m, cov_nm, pq, v1, floor)."""
    rows = []
    for key, entries in dump.get("results", {}).items():
        zs = key.startswith("zeroshot_")
        rest = key[len("zeroshot_"):] if zs else key
        # key = <display_name>_<metric> ; metric is one of these three
        metric = next((m for m in ("validation", "training", "combined")
                       if rest.endswith(m)), "?")
        domain = rest[:-len(metric)].rstrip("_") if metric != "?" else rest
        for e in entries:
            learned = None
            for ptype, m in e.get("metrics", {}).items():
                if "LEARNED" in ptype:
                    learned = m
            if learned is None:
                continue
            rows.append((
                zs, domain, metric, e.get("epoch"),
                100.0 * (learned.get("success_rate_with_monitor") or 0),
                100.0 * (learned.get("success_rate_without_monitor") or 0),
                learned.get("plan_quality"),
                learned.get("top1_valid_rate"),
                floor.get(domain),
            ))
    return rows


def fmt(v, spec="6.2f", dash="-"):
    return dash.rjust(int(spec.split(".")[0])) if v is None else format(v, spec)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="cache/results")
    ap.add_argument("--out", default="cache/results/DIGEST.md")
    ap.add_argument("--domains", default="",
                    help="Comma-separated substrings to keep (default: all)")
    ap.add_argument("--print", dest="do_print", action="store_true")
    args = ap.parse_args()

    dumps = load_dumps(args.dir)
    if not dumps:
        sys.exit(f"No results_*.json under {args.dir}/*/")
    floor = load_floor(args.dir)
    keep = [s.strip().lower() for s in args.domains.split(",") if s.strip()]

    lines = ["# Results digest", "",
             f"{len(dumps)} run dump(s); newest per (experiment, train set).",
             "cov = coverage with/without monitor | PQ = plan quality vs "
             "the satisficing planner | V1 = top-1 proposal validity |",
             "floor = random-policy coverage, same domain+split "
             "(`-` = not measured). ZS rows first.", ""]

    for zs_wanted, title in ((True, "## Zero-shot"), (False, "## In-domain")):
        lines += [title, "",
                  "| experiment | feat | domain | metric | epoch | cov m/nm | "
                  "PQ | V1 | floor |",
                  "|---|---|---|---|---|---|---|---|---|"]
        any_row = False
        for d in sorted(dumps, key=lambda x: (x.get("featurization", ""),
                                              x.get("experiment", ""))):
            for (zs, dom, metric, epoch, cm, cnm, pq, v1, fl) in sorted(
                    rows_from(d, floor), key=lambda r: (r[1], r[2], r[3] or 0)):
                if zs != zs_wanted:
                    continue
                if keep and not any(k in dom.lower() for k in keep):
                    continue
                any_row = True
                lines.append(
                    f"| {d.get('experiment','?')} | {d.get('featurization','?')} "
                    f"| {dom} | {metric} | {epoch} "
                    f"| {cm:.1f}/{cnm:.1f} | {fmt(pq, '5.3f')} "
                    f"| {fmt(100*v1 if (v1 or -1) >= 0 else None, '5.1f')} "
                    f"| {fmt(fl, '5.2f')} |")
        if not any_row:
            lines.append("| _(none yet)_ | | | | | | | | |")
        lines.append("")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {args.out} ({sum(1 for l in lines if l.startswith('| loo') or l.startswith('| all'))} rows)")
    if args.do_print:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
