#!/usr/bin/env python3
"""Which planned runs are live / done / missing on THIS filesystem.

Run on each machine; the union across machines is the true picture.
Checks three independent signals per config:

  proc   - a live `main.py --config <cfg>` process (DataLoader workers make
           one run look like several; deduped here)
  ckpt   - models/<MULTI-...>/ exists for the config's training set
  result - cache/results/<expid>/results_*.json exists

    python tools/run_audit.py              # the planned set (see PLANNED)
    python tools/run_audit.py --all        # every configs/*.yaml
"""
import argparse
import glob
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# The paper's current cell list (four held-out targets: visitall, miconic,
# grid, logistics; ladder UNION -> BIND(joint) -> GADAR(joint_chain)).
PLANNED = [
    "loo8_union_no_visitall", "loo8_union_no_miconic",
    "loo8_union_no_grid", "loo8_union_no_logistics",
    "loo8_joint_no_visitall", "loo8_joint_no_miconic",
    "loo8_joint_no_grid", "loo8_joint_no_logistics",
    "loo8_joint_chain_no_visitall", "loo8_joint_chain_no_miconic",
    "loo8_joint_chain_no_grid", "loo8_joint_chain_no_logistics",
    # measured but out of the paper's table; keep visible
    "loo8_joint_chain_no_manyblocks", "loo8_joint_chain_no_gripper",
]


def live_configs():
    """Config basenames of running main.py processes (deduped)."""
    try:
        out = subprocess.run(["pgrep", "-u", os.environ.get("USER", ""), "-af",
                              "main.py"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return {}
    seen = {}
    for line in out.splitlines():
        m = re.search(r"configs/([\w.]+)\.yaml", line)
        if not m:
            continue
        name = m.group(1)
        seen.setdefault(name, {"n": 0, "cmd": line.split(None, 1)[-1]})
        seen[name]["n"] += 1
    return seen


def expected_model_dir(cfg_name):
    """MULTI-<domains> directory name from the config's `domains:` list."""
    path = os.path.join("configs", cfg_name + ".yaml")
    if not os.path.exists(path):
        return None
    doms = None
    for line in open(path):
        if line.startswith("domains:"):
            doms = line.split(":", 1)[1].strip()
            break
    if not doms:
        return None
    return "MULTI-" + "-".join(d.strip().capitalize() for d in doms.split(","))


def has_checkpoints(mdir):
    """ModelManager dirs live under cache/results/<save_prefix>/<env>_seed<n>_...
    (see get_filenames/main.py), NOT under models/. Match on the MULTI- name
    anywhere at those depths and require a tracking file."""
    if not mdir:
        return False
    patterns = [
        os.path.join("models", mdir + "*", "model_tracking.json"),
        os.path.join("cache", "results", "*", mdir + "*", "model_tracking.json"),
    ]
    return any(glob.glob(p) for p in patterns)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    names = PLANNED
    if args.all:
        names = sorted(os.path.basename(p)[:-5]
                       for p in glob.glob("configs/*.yaml")
                       if not os.path.basename(p).startswith("_"))

    live = live_configs()
    print(f"{'config':34s} {'proc':>5s} {'ckpt':>5s} {'result':>7s}  verdict")
    print("-" * 72)
    missing = []
    for name in names:
        n_proc = live.get(name, {}).get("n", 0)
        mdir = expected_model_dir(name)
        has_ckpt = has_checkpoints(mdir)
        results = glob.glob(os.path.join("cache", "results", "*", "results_*.json"))
        # expid defaults to the config name in our suite
        has_res = any(f"/{name}/" in p for p in results)
        if n_proc:
            verdict = "RUNNING"
        elif has_res:
            verdict = "done (results written)"
        elif has_ckpt:
            verdict = "trained, NOT tested"
        else:
            verdict = "*** MISSING ***"
            missing.append(name)
        print(f"{name:34s} {n_proc:5d} {'yes' if has_ckpt else '-':>5s} "
              f"{'yes' if has_res else '-':>7s}  {verdict}")

    print()
    if missing:
        print("Not started on this filesystem (check the other machine first):")
        for m in missing:
            print(f"  ./train_test_scripts/run_config.sh configs/{m}.yaml cuda:0")
    else:
        print("Every planned config is running, trained, or done here.")


if __name__ == "__main__":
    main()
