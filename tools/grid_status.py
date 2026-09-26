#!/usr/bin/env python3
"""The leave-one-out grid: which (held-out fold x rung) cells have a model.

    python tools/grid_status.py            # the 8x3 grid + what is missing
    python tools/grid_status.py --seed 11  # a different seed family

Reads models/ only - no training state, no cluster access - so it answers
"how much of the paper exists" from any node, including the login node.

The held-out fold is inferred from the directory name, which lists the
TRAINING domains: the one of the eight that is absent is the fold. That is
the same reading done by hand every time this question came up, and getting
it wrong (miscounting which domain is missing) silently reports a cell as
done when it is not.

A cell counts as present only with MIN_CKPTS checkpoints: a directory is
created by ModelManager the moment a run starts, so an empty or nearly-empty
one is a launch that died in data collection, not a trained model.
"""

import argparse
import glob
import os
import re

DOMAINS = ["Manyblocks_ipcc_big", "Gripper_ipcc", "Miconic_ipcc", "Visitall_ipcc",
           "Grid_ipcc", "Logistics_ipcc", "Spanner_ipcc", "Rovers_ipcc"]
# paper column order: control, BIND, full
RUNGS = [("union", "UNION"), ("joint_lite", "BIND"), ("joint_chain", "GADAR")]
# A cell needs this many checkpoints to count. Every genuine 500-epoch run in
# this suite has 34-62; the runs killed early had 7 and 13. A threshold of 5
# would call `union_no_rovers` (killed at epoch 60, 7 checkpoints) done, which
# is exactly the mistake this script exists to prevent.
MIN_CKPTS = 20


def fold_short(domain):
    """Manyblocks_ipcc_big -> manyblocks: the fold name used in config files."""
    return domain.split("_")[0].lower()


def checkpoint_counts(models="models", seed=10):
    """{domain: {rung: n_checkpoints}} for every leave-one-out cell.

    Shared with tools/eval_status.py, which uses it to refuse evaluating a
    cell this table would not count as trained - so the two tools cannot
    disagree about whether a model is ready.
    """
    grid = {d: {r: 0 for r, _ in RUNGS} for d in DOMAINS}
    for p in sorted(glob.glob(os.path.join(models, "MULTI-*feat*/"))):
        b = os.path.basename(p.rstrip("/"))
        if f"_seed{seed}_" not in b:
            continue
        m = re.search(r"feat(joint_chain|joint_lite|union)", b)
        if not m:
            continue
        # A domain is in the training set if it appears followed by '-' (mid
        # list) or '_seed' (last). Substring alone would match nothing else
        # here, but the anchors keep it honest if a domain name ever becomes
        # a prefix of another.
        missing = [d for d in DOMAINS
                   if f"{d}-" not in b and f"{d}_seed" not in b]
        if len(missing) != 1:
            continue                      # all-8 or 2-domain runs: not a fold
        n = len(glob.glob(os.path.join(p, "*.pt")))
        rung = m.group(1)
        grid[missing[0]][rung] = max(grid[missing[0]][rung], n)
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--min-ckpts", type=int, default=MIN_CKPTS)
    a = ap.parse_args()

    grid = checkpoint_counts(a.models, a.seed)

    w = max(len(d) for d in DOMAINS) + 2
    print(f"{'held-out fold':<{w}}" + "".join(f"{lbl:>10}" for _, lbl in RUNGS)
          + "   ladder")
    print("-" * (w + 10 * len(RUNGS) + 11))
    cells = complete = 0
    for d in DOMAINS:
        counts = [grid[d][r] for r, _ in RUNGS]
        have = sum(1 for c in counts if c >= a.min_ckpts)
        cells += have
        complete += have == len(RUNGS)
        cols = "".join(f"{(c if c else '-'):>10}" for c in counts)
        print(f"{d:<{w}}{cols}   "
              + ("COMPLETE" if have == len(RUNGS) else f"{have}/{len(RUNGS)}"))

    total = len(DOMAINS) * len(RUNGS)
    print(f"\n{cells}/{total} cells, {complete}/{len(DOMAINS)} complete ladders"
          f"  (a cell needs >= {a.min_ckpts} checkpoints)")

    missing = [f"loo8_{r}_no_{fold_short(d)}"
               for d in DOMAINS for r, _ in RUNGS
               if grid[d][r] < a.min_ckpts]
    if missing:
        print("\nstill to run:")
        for c in missing:
            print(f"  configs/{c}.yaml")


if __name__ == "__main__":
    main()
