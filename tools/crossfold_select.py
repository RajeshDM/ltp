#!/usr/bin/env python3
"""Cross-fold (nested leave-one-out) epoch selection for zero-shot numbers.

    python tools/crossfold_select.py                  # reads *_epochs dumps
    python tools/crossfold_select.py --suffix _epochs --csv crossfold.csv

The paper's zero-shot rule (gadar.tex, "Checkpoint selection"): for held-out
fold d, use the epoch e*(d) that maximizes MEAN zero-shot coverage on the
hard (test) split of the OTHER seven folds of the same rung. Fold d's own
test coverage never enters its choice, so the reported number is test-blind.
Ties go to the earliest epoch.

Input is the epoch sweep written by
    ZERO_SHOT_ONLY=1 ZS_SPLITS=test METRICS=periodic NMODELS=12 \\
    EXPID_SUFFIX=_epochs ./train_test_scripts/eval_worker.sh
i.e. cache/results/loo8_<rung>_no_<fold>_epochs/results_*.json. Every dump
of an experiment is read (a relaunch can split the sweep across dumps); for
a repeated epoch the latest dump wins.

Prints, per rung and fold: e*(d), coverage at e*(d) (REPORTABLE), and for
reference the latest epoch and the best-on-test epoch (ORACLE - not
reportable, it selects on the number it reports).
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

EXP_RE = re.compile(r"^loo\d+_(?P<rung>.+)_no_(?P<fold>[a-z]+)$")


def zero_shot_test_coverage(dump):
    """{epoch: success_rate_with_monitor} on the held-out TEST split."""
    zs = [e for e in dump.get("eval_plan", []) if e.get("zero_shot")]
    labels = sorted((e["domain"] for e in zs), key=len, reverse=True)
    test_labels = {e["domain"] for e in zs if e.get("split") == "test"}
    out = {}
    for key, entries in dump.get("results", {}).items():
        # Longest label first so 'X' never shadows 'X@train'
        label = next((l for l in labels if l in key), None)
        if label not in test_labels:
            continue
        for entry in entries:
            learned = next((m for p, m in entry.get("metrics", {}).items()
                            if "LEARNED" in p), None)
            if learned is not None:
                out[int(entry["epoch"])] = learned["success_rate_with_monitor"]
    return out


def load_sweeps(results_dir, suffix):
    """{rung: {fold: {epoch: coverage}}} from every matching dump."""
    paths = sorted(glob.glob(os.path.join(results_dir, f"*{suffix}", "results_*.json")))
    dumps = []
    for p in paths:
        try:
            with open(p) as f:
                dumps.append(json.load(f))
        except Exception as e:
            print(f"WARN: skipping unreadable {p}: {e}")
    dumps.sort(key=lambda d: d.get("timestamp", ""))  # later dumps overwrite
    sweeps = defaultdict(lambda: defaultdict(dict))
    for d in dumps:
        exp = d.get("experiment", "")
        if suffix and exp.endswith(suffix):
            exp = exp[:-len(suffix)]
        m = EXP_RE.match(exp)
        if not m:
            continue
        sweeps[m["rung"]][m["fold"]].update(zero_shot_test_coverage(d))
    return sweeps


def select(folds):
    """{fold: (e*, cov at e*, #other folds, epochs considered)} for one rung.

    Only epochs present in EVERY other fold are candidates, so each mean is
    over the same set of folds; a fold missing an epoch shrinks the
    candidate set rather than silently averaging over fewer folds.
    """
    out = {}
    for d, own in folds.items():
        others = [f for f in folds if f != d and folds[f]]
        if not others or not own:
            continue
        common = set(own)
        for f in others:
            common &= set(folds[f])
        if not common:
            continue
        mean = {e: sum(folds[f][e] for f in others) / len(others) for e in common}
        best = max(mean.values())
        e_star = min(e for e in common if mean[e] == best)
        out[d] = (e_star, own[e_star], len(others), sorted(common))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="cache/results")
    ap.add_argument("--suffix", default="_epochs",
                    help="EXPID_SUFFIX of the sweep pass (default _epochs)")
    ap.add_argument("--csv", help="also write the table here")
    args = ap.parse_args()

    sweeps = load_sweeps(args.dir, args.suffix)
    if not sweeps:
        print(f"No loo*_<rung>_no_<fold>{args.suffix} dumps under {args.dir}/")
        sys.exit(1)

    rows = []
    for rung in sorted(sweeps):
        folds = sweeps[rung]
        chosen = select(folds)
        print(f"\n== {rung}: {len(folds)} folds swept ==")
        print(f"{'fold':<11} {'e*':>5} {'cov@e*':>7} {'n':>2}   "
              f"{'latest':>11}   {'oracle(NOT reportable)':>22}  epochs")
        for d in sorted(folds):
            own = folds[d]
            if not own:
                continue
            e_last = max(own)
            e_or = min(e for e in own if own[e] == max(own.values()))
            if d in chosen:
                e_star, cov, n, common = chosen[d]
                sel = f"{e_star:>5} {100 * cov:>6.1f}% {n:>2}"
            else:
                e_star = cov = n = None
                common = []
                sel = f"{'--':>5} {'--':>7} {'':>2}"
            print(f"{d:<11} {sel}   E{e_last:<4}{100 * own[e_last]:>5.1f}%   "
                  f"E{e_or:<4}{100 * own[e_or]:>16.1f}%  "
                  f"{len(own)} ({len(common)} common)")
            rows.append({"rung": rung, "fold": d, "e_star": e_star,
                         "cov_at_e_star": cov, "other_folds": n,
                         "latest_epoch": e_last, "cov_latest": own[e_last],
                         "oracle_epoch": e_or, "cov_oracle": own[e_or],
                         "epochs_swept": len(own)})
        missing = 8 - len(folds)
        if missing > 0:
            print(f"   NOTE: {missing} fold(s) not swept yet - every e* above "
                  f"will move when they land")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
