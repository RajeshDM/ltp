#!/usr/bin/env python3
"""Aggregate results_*.json dumps from all experiments into one table.

Usage:
    python tools/analyze_results.py                    # scan cache/results/
    python tools/analyze_results.py --dir other/path
    python tools/analyze_results.py --csv results.csv  # also write CSV
    python tools/analyze_results.py --all              # every dump, not just latest

Reads the JSON files written at the end of each main.py test run
(one per invocation, in cache/results/<expid>/). For each experiment it
shows, per tested domain and checkpoint-selection metric, the learned
model's success rates and plan quality next to the non-optimal planner.
Zero-shot rows are marked ZS. By default only the LATEST dump per
(experiment, train_domain) is used; --all shows every dump.
"""
import argparse
import csv
import glob
import json
import os
import sys


def load_dumps(results_dir, keep_all):
    paths = sorted(glob.glob(os.path.join(results_dir, "*", "results_*.json")))
    if not paths:
        print(f"No results_*.json found under {results_dir}/*/")
        sys.exit(1)
    dumps = []
    for p in paths:
        try:
            with open(p) as f:
                d = json.load(f)
            d["_path"] = p
            dumps.append(d)
        except Exception as e:
            print(f"WARN: skipping unreadable {p}: {e}")
    if keep_all:
        return dumps
    latest = {}
    for d in dumps:
        key = (d.get("experiment"), d.get("train_domain"))
        if key not in latest or d.get("timestamp", "") > latest[key].get("timestamp", ""):
            latest[key] = d
    return list(latest.values())


def extract_rows(dumps):
    rows = []
    for d in dumps:
        zs_domains = {e["domain"] for e in d.get("eval_plan", []) if e.get("zero_shot")}
        for result_key, entries in d.get("results", {}).items():
            for entry in entries:
                learned = non_opt = None
                for ptype, m in entry.get("metrics", {}).items():
                    if "LEARNED" in ptype:
                        learned = m
                    elif "NON_OPTIMAL" in ptype:
                        non_opt = m
                if learned is None:
                    continue
                # Longest label first so 'X_ipcc' never shadows 'X_ipcc@train'
                domain = next((z for z in sorted(zs_domains, key=len, reverse=True)
                               if z in result_key), None)
                zero_shot = domain is not None
                if domain is None:
                    _cands = sorted((e["domain"] for e in d.get("eval_plan", [])),
                                    key=len, reverse=True)
                    domain = next((c for c in _cands if c in result_key),
                                  result_key)
                metric = next((m for m in ("validation", "training", "combined")
                               if result_key.endswith(m)), "?")
                rows.append({
                    "experiment": d.get("experiment", "?"),
                    "featurization": d.get("featurization", "?"),
                    "domain": domain,
                    "zero_shot": zero_shot,
                    "sel_metric": metric,
                    "epoch": entry.get("epoch"),
                    "succ_monitor": learned.get("success_rate_with_monitor", 0.0),
                    "succ_no_monitor": learned.get("success_rate_without_monitor", 0.0),
                    "plan_quality": learned.get("plan_quality", 0.0),
                    "avg_plan_len": learned.get("avg_plan_length", 0.0),
                    "avg_time": learned.get("avg_time_taken", 0.0),
                    "nonopt_succ": (non_opt or {}).get("success_rate_with_monitor", None),
                })
    return rows


def print_table(rows):
    if not rows:
        print("No learned-model results found in the dumps.")
        return
    rows.sort(key=lambda r: (r["experiment"], r["zero_shot"], r["domain"], r["sel_metric"]))
    hdr = (f'{"experiment":<26} {"domain":<22} {"ZS":<3} {"sel":<11} '
           f'{"ep":>4} {"succ":>7} {"succ-noM":>8} {"quality":>8} '
           f'{"plan":>7} {"time":>6} {"nonopt":>7}')
    print(hdr)
    print("-" * len(hdr))
    last_exp = None
    for r in rows:
        if last_exp is not None and r["experiment"] != last_exp:
            print()
        last_exp = r["experiment"]
        nonopt = f'{r["nonopt_succ"]:.0%}' if r["nonopt_succ"] is not None else "-"
        print(f'{r["experiment"]:<26} {r["domain"]:<22} '
              f'{"ZS" if r["zero_shot"] else "":<3} {r["sel_metric"]:<11} '
              f'{r["epoch"]:>4} {r["succ_monitor"]:>7.1%} {r["succ_no_monitor"]:>8.1%} '
              f'{r["plan_quality"]:>8.3f} {r["avg_plan_len"]:>7.1f} '
              f'{r["avg_time"]:>6.2f} {nonopt:>7}')

    zs = [r for r in rows if r["zero_shot"]]
    if zs:
        print("\n=== Zero-shot summary (C1) ===")
        for r in zs:
            print(f'  {r["experiment"]:<26} -> {r["domain"]:<22} '
                  f'succ={r["succ_monitor"]:.1%}  quality={r["plan_quality"]:.3f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="cache/results", help="results root to scan")
    ap.add_argument("--csv", default="", help="also write rows to this CSV file")
    ap.add_argument("--all", action="store_true",
                    help="show every dump, not just the latest per experiment")
    args = ap.parse_args()

    dumps = load_dumps(args.dir, args.all)
    print(f"Loaded {len(dumps)} result dump(s)\n")
    rows = extract_rows(dumps)
    print_table(rows)

    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
