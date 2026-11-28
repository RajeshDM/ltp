#!/usr/bin/env python3
"""Re-queue every p2 evaluation whose tested epoch is not the one the current
selection rule picks (C1).

    python tools/requalify_selection.py [--dry-run]

The 2026-11-27 rule (tools/select_checkpoint.py --rule loss: lowest mean
per-domain Val-OOD loss) replaced the coverage rule after evaluations had
already run. For each configs/p2_*.yaml with a cache/results/<cfg>_final/
dump: if its epoch differs from the loss rule's, move that directory aside
(<cfg>_final.coverage_rule_<stamp> - never deleted) and reset the p2eval job,
so a worker re-runs campaign/jobs/protocol_eval.sh (no sweep, just the test at
the new epoch). Evaluations still running are left alone; run this again
after they finish.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tested_epoch(cfg):
    dumps = sorted(glob.glob(os.path.join(ROOT, "cache", "results", f"{cfg}_final", "results_*.json")),
                   key=os.path.getmtime)
    if not dumps:
        return None
    with open(dumps[-1]) as f:
        d = json.load(f)
    eps = {int(e["epoch"]) for v in (d.get("results") or {}).values() for e in v}
    return eps.pop() if len(eps) == 1 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    st = subprocess.run([sys.executable, "campaign/worker.py", "status", "--all", "--json"], cwd=ROOT,
                        capture_output=True, text=True, timeout=600).stdout
    state = {r["id"]: r["state"] for r in json.loads(st)}
    for path in sorted(glob.glob(os.path.join(ROOT, "configs", "p2_*.yaml"))):
        cfg = os.path.basename(path)[:-5]
        have = tested_epoch(cfg)
        if have is None:
            continue
        want = subprocess.run([sys.executable, "tools/select_checkpoint.py", "--config", path,
                               "--print-epoch"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        if not want.isdigit() or int(want) == have:
            continue
        job = f"p2eval_{cfg}"
        if state.get(job) == "running":
            print(f"{cfg}: tested epoch {have}, rule picks {want} - eval still running, later")
            continue
        print(f"{cfg}: tested epoch {have}, rule picks {want} -> re-queue")
        if args.dry_run:
            continue
        src = os.path.join(ROOT, "cache", "results", f"{cfg}_final")
        os.rename(src, f"{src}.coverage_rule_{time.strftime('%Y%m%d_%H%M%S')}")
        subprocess.run([sys.executable, "campaign/worker.py", "reset", job], cwd=ROOT)


if __name__ == "__main__":
    main()
