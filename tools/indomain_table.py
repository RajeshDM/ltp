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
import sys
from collections import defaultdict

RUNGS = [("union", "UNION"), ("joint_lite", "BIND"), ("joint_chain", "GADAR")]


def learned_cov(entry):
    for ptype, m in entry.get("metrics", {}).items():
        if "LEARNED" in ptype:
            return m["success_rate_with_monitor"]
    return None


def in_domain_cells(dump, metric):
    """({domain_label: (coverage, epoch, how)}, [ambiguous labels]) for the
    in-domain TEST cells at `metric`.

    Dumps written before 2026-11-13 leave a metric's cell EMPTY when its best
    checkpoint had already been tested under an earlier metric of the same run
    (run_tests skipped it as "already tested"). That checkpoint's result is in
    the dump under the earlier metric, so it is recovered here: `how` names
    the metric it was read from. If the earlier metrics tested different
    epochs the cell cannot be attributed and the label is returned as
    ambiguous instead of guessed.
    """
    plan = dump.get("eval_plan") or []
    labels = sorted((e["domain"] for e in plan), key=len, reverse=True)
    wanted = {e["domain"] for e in plan
              if not e.get("zero_shot") and e.get("split", "test") == "test"}
    order = list(dump.get("test_model_metrics") or [])
    names = order if metric in order else order + [metric]
    by_label = defaultdict(dict)          # label -> {metric: entries}
    for key, entries in (dump.get("results") or {}).items():
        if key.startswith("zeroshot_"):
            continue
        label = next((l for l in labels if l in key), None)
        m = next((m for m in names if key.endswith(m)), None)
        if label in wanted and m is not None:
            by_label[label][m] = entries
    out, ambiguous = {}, []
    for label, cells in by_label.items():
        entries, how = cells.get(metric), metric
        if not entries and metric in order and metric in cells:
            earlier = [(m, cells[m]) for m in order[:order.index(metric)] if cells.get(m)]
            if len({e[0].get("epoch") for _, e in earlier}) == 1:
                how, entries = earlier[0]
            elif earlier:
                ambiguous.append(label.lower())
        if entries:
            cov = learned_cov(entries[0])
            if cov is not None:
                out[label.lower()] = (cov, entries[0].get("epoch"), how)
    return out, ambiguous


def collect(results_dir, metric, since, notes=None):
    """{experiment: {domain: (cov, epoch, timestamp)}}, newest dump per cell.

    `notes` (a list) receives what a reader of the table must know: cells
    recovered from another metric of the same dump, and cells that had to
    come from an OLDER dump although a newer one evaluated that domain.
    """
    cells = defaultdict(dict)
    newest_unresolved = defaultdict(dict)      # exp -> {domain: timestamp}
    recovered = 0
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
        found, ambiguous = in_domain_cells(d, metric)
        for dom in ambiguous:
            if ts > newest_unresolved[exp].get(dom, ""):
                newest_unresolved[exp][dom] = ts
        for dom, (cov, ep, how) in found.items():
            if dom not in cells[exp] or ts > cells[exp][dom][2]:
                cells[exp][dom] = (cov, ep, ts, how)
    if notes is not None:
        for exp, doms in cells.items():
            for dom, (_, ep, ts, how) in doms.items():
                if how != metric:
                    recovered += 1
                newer = newest_unresolved[exp].get(dom, "")
                if newer > ts:
                    notes.append(
                        f"STALE {exp} / {short(dom)}: '{metric}' shown from {ts} (E{ep}), but the "
                        f"run of {newer} tested this domain and left the cell empty with "
                        f"no single earlier metric to read it from - re-evaluate this config")
        if recovered:
            notes.append(f"{recovered} cell(s) read from another metric of the same dump: that "
                         f"checkpoint was also the best '{metric}' one and was tested once")
    return {exp: {dom: v[:3] for dom, v in doms.items()} for exp, doms in cells.items()}


def short(dom):
    return re.sub(r"_ipcc.*$", "", dom)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="cache/results")
    ap.add_argument("--metric", default="combined")
    ap.add_argument("--since", default="20260905")
    ap.add_argument("--rungs", default="",
                    help="comma-separated featurizations to tabulate instead "
                         "of the paper ladder, e.g. "
                         "joint_chain,jc_no_bind,jc_no_goal,joint (C2 "
                         "ablations). Shows only the LOO columns.")
    ap.add_argument("--folds", default="",
                    help="with --rungs: average only these held-out folds, "
                         "e.g. manyblocks,visitall,grid, so every column is "
                         "over the SAME folds (the ablations train three)")
    ap.add_argument("--missing", action="store_true",
                    help="print the loo8 union/joint_lite configs with no in-domain "
                         "cells yet, as config paths, and exit")
    args = ap.parse_args()

    notes = []
    cells = collect(args.dir, args.metric, args.since, notes)
    if args.missing:
        # Untrained cells (e.g. loo8_union_no_gripper) have nothing to evaluate.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from eval_status import trained_counts, evaluating
        from grid_status import MIN_CKPTS
        trained = trained_counts("models", 10)
        for rung in ("union", "joint_lite"):
            for cfg in sorted(glob.glob(f"configs/loo8_{rung}_no_*.yaml")):
                name = os.path.basename(cfg)[:-len(".yaml")]
                if (not cells.get(name) and trained.get(name, 0) >= MIN_CKPTS
                        and not evaluating(name)):
                    print(cfg)
        return
    domains = sorted({d for exp in cells.values() for d in exp})
    if not domains:
        print(f"No in-domain '{args.metric}' cells in dumps since {args.since}.")
        return

    rungs, extra = RUNGS, True
    folds = [f for f in args.folds.split(",") if f]
    fold_pat = "(?:" + "|".join(folds) + ")" if folds else "[a-z]+"
    if args.rungs:
        rungs = [(r, r) for r in args.rungs.split(",") if r]
        extra = False
    cols = [f"LOO-{n}" for _, n in rungs]
    if extra:
        cols += [f"ALL-{n}" for _, n in RUNGS] + ["SD"]
    if folds:
        print(f"held-out folds: {', '.join(folds)}")
    print(f"in-domain test coverage %, checkpoint = {args.metric}, dumps >= {args.since}")
    width = max(14, max(len(c) for c in cols) + 2)
    print(f"{'domain':<12}" + "".join(f"{c:>{width}}" for c in cols))
    col_vals = defaultdict(list)
    for dom in domains:
        row = []
        for rung, name in rungs:
            vals = [c[dom][0] for exp, c in cells.items()
                    if re.fullmatch(rf"loo8_{rung}_no_{fold_pat}", exp) and dom in c]
            if vals:
                m = 100 * sum(vals) / len(vals)
                col_vals[f"LOO-{name}"].append(m)
                row.append(f"{m:6.1f} (n={len(vals)})")
            else:
                row.append("--")
        for exp, name in ([(f"all8_{r}", f"ALL-{n}") for r, n in RUNGS] +
                          [(f"sd_joint_chain_{short(dom)}", "SD")]) if extra else []:
            v = cells.get(exp, {}).get(dom)
            if v:
                col_vals[name].append(100 * v[0])
                row.append(f"{100 * v[0]:6.1f} E{v[1]}")
            else:
                row.append("--")
        print(f"{short(dom):<12}" + "".join(f"{x:>{width}}" for x in row))
    print(f"{'mean':<12}" + "".join(
        f"{f'{sum(col_vals[c]) / len(col_vals[c]):.1f} ({len(col_vals[c])})':>{width}}"
        if col_vals[c] else f"{'--':>{width}}" for c in cols))
    print("\n'--' = no post-fix dump with that cell; mean is over the rows present,"
          " so compare columns only where both have all rows.")
    for note in notes:
        print("NOTE " + note)


if __name__ == "__main__":
    main()
