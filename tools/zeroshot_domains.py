#!/usr/bin/env python3
"""Print the zero-shot --test-domains for a leave-one-out config.

    python tools/zeroshot_domains.py configs/loo8_union_no_grid.yaml
    -> grid_ipcc:48,grid_ipcc@train:192
    python tools/zeroshot_domains.py configs/loo8_union_no_grid.yaml --test-only
    -> grid_ipcc:48

A LOO config's `test_domains` lists all eight domains; the zero-shot cells
are the entries whose domain is NOT in `domains` (the training set). Those
are what C1 and C2 are measured on, and evaluating only them is ~1/5 of a
full eval, whose in-domain cells cost the other 4/5.

Prints nothing (exit 1) if the config holds out no domain, so a caller can
fall back to the full evaluation rather than silently evaluating nothing.
"""
import re
import sys


def read_key(path, key):
    with open(path) as f:
        for line in f:
            m = re.match(rf"^{key}:\s*(.+?)\s*$", line)
            if m:
                return m.group(1)
    return ""


def main():
    cfg = sys.argv[1]
    train = {d.strip().lower() for d in read_key(cfg, "domains").split(",") if d.strip()}
    cells = [c.strip() for c in read_key(cfg, "test_domains").split(",") if c.strip()]
    held = [c for c in cells
            if re.split(r"[:@]", c, 1)[0].lower() not in train]
    # --in-domain prints the complement: the training domains' test splits,
    # i.e. an in-domain-only evaluation (Table 2) without the zero-shot cells.
    if "--in-domain" in sys.argv[2:]:
        held = [c for c in cells
                if re.split(r"[:@]", c, 1)[0].lower() in train and "@" not in c]
    # --test-only drops the @train split. Its random floor is 63-99% on
    # several folds, so it cannot discriminate there, and it is the larger
    # split - dropping it more than halves an epoch sweep.
    if "--test-only" in sys.argv[2:]:
        held = [c for c in held if "@" not in c]
    if not held:
        sys.exit(1)
    print(",".join(held))


if __name__ == "__main__":
    main()
