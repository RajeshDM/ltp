#!/usr/bin/env python3
"""In-domain coverage: models trained on several domains, tested on those
same domains' held-out test problems (Table 2 of the paper).

    python tools/indomain_table.py                 # combined-loss checkpoint
    python tools/indomain_table.py --metric validation --since 20260905

Columns (test split, success with monitor, %):
    LOO-<rung>  mean over the 7 leave-one-out models that TRAINED on the row's
                domain (n shown), from loo8_<rung>_no_<fold>
    ALL-<rung>  the single model trained on all 8 (all8_<rung>)
    SD          single-domain GADAR trained on the row's domain only
                (sd_joint_chain_<domain>)

Zero-shot dumps (ZERO_SHOT_ONLY passes) hold no in-domain cells, so each
(experiment, cell) is taken from the NEWEST dump that has it rather than
from the newest dump overall. Dumps older than --since are ignored: before
the mixed-arity fix (20260905) the numbers are wrong, not merely old.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

RUNGS = [("union", "UNION"), ("joint_lite", "BIND"), ("joint_chain", "GADAR")]


def learned_cov(entry):
    for ptype, m in entry.get("metrics", {}).items():
        if "LEARNED" in ptype:
            return m["success_rate_with_monitor"]
    return None


def in_domain_cells(dump, metric):
    """{domain_label: (coverage, epoch)} for in-domain TEST cells at `metric`."""
    plan = dump.get("eval_plan") or []
    labels = sorted((e["domain"] for e in plan), key=len, reverse=True)
    wanted = {e["domain"] for e in plan
              if not e.get("zero_shot") and e.get("split", "test") == "test"}
    out = {}
    for key, entries in (dump.get("results") or {}).items():
        if key.startswith("zeroshot_") or not key.endswith(metric) or not entries:
            continue
        label = next((l for l in labels if l in key), None)
        if label not in wanted:
            continue
        cov = learned_cov(entries[0])
        if cov is not None:
            out[label.lower()] = (cov, entries[0].get("epoch"))
    return out


def collect(results_dir, metric, since):
    """{experiment: {domain: (cov, epoch, timestamp)}}, newest dump per cell."""
    cells = defaultdict(dict)
    for p in glob.glob(os.path.join(results_dir, "*", "results_*.json")):
        try:
            with open(p) as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        ts = str(d.get("timestamp", ""))
        if ts[:8] < since:
            continue
        exp = d.get("experiment", "")
        for dom, (cov, ep) in in_domain_cells(d, metric).items():
            if dom not in cells[exp] or ts > cells[exp][dom][2]:
                cells[exp][dom] = (cov, ep, ts)
    return cells


def short(dom):
    return re.sub(r"_ipcc.*$", "", dom)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="cache/results")
    ap.add_argument("--metric", default="combined")
    ap.add_argument("--since", default="20260905")
    args = ap.parse_args()

    cells = collect(args.dir, args.metric, args.since)
    domains = sorted({d for exp in cells.values() for d in exp})
    if not domains:
        print(f"No in-domain '{args.metric}' cells in dumps since {args.since}.")
        return

    cols = [f"LOO-{n}" for _, n in RUNGS] + [f"ALL-{n}" for _, n in RUNGS] + ["SD"]
    print(f"in-domain test coverage %, checkpoint = {args.metric}, dumps >= {args.since}")
    print(f"{'domain':<12}" + "".join(f"{c:>14}" for c in cols))
    col_vals = defaultdict(list)
    for dom in domains:
        row = []
        for rung, name in RUNGS:
            vals = [c[dom][0] for exp, c in cells.items()
                    if re.fullmatch(rf"loo8_{rung}_no_[a-z]+", exp) and dom in c]
            if vals:
                m = 100 * sum(vals) / len(vals)
                col_vals[f"LOO-{name}"].append(m)
                row.append(f"{m:6.1f} (n={len(vals)})")
            else:
                row.append("--")
        for exp, name in [(f"all8_{r}", f"ALL-{n}") for r, n in RUNGS] + \
                         [(f"sd_joint_chain_{short(dom)}", "SD")]:
            v = cells.get(exp, {}).get(dom)
            if v:
                col_vals[name].append(100 * v[0])
                row.append(f"{100 * v[0]:6.1f} E{v[1]}")
            else:
                row.append("--")
        print(f"{short(dom):<12}" + "".join(f"{x:>14}" for x in row))
    print(f"{'mean':<12}" + "".join(
        f"{(sum(col_vals[c]) / len(col_vals[c])):>9.1f} ({len(col_vals[c])})" if col_vals[c]
        else f"{'--':>14}" for c in cols))
    print("\n'--' = no post-fix dump with that cell; mean is over the rows present,"
          " so compare columns only where both have all rows.")


if __name__ == "__main__":
    main()
