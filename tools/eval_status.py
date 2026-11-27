#!/usr/bin/env python3
"""Which configs have been EVALUATED, and which still need it.

    python tools/eval_status.py                    # the table
    python tools/eval_status.py --list-missing     # config paths, one per line
    python tools/eval_status.py --metrics combined # what counts as complete

The eval analogue of tools/grid_status.py, and needed for the same reason:
allocations die mid-queue, and without this a relaunch re-runs everything
that already finished - hours per config.

"Evaluated" is decided from the dump main.py writes to
cache/results/<expid>/results_*.json, not from the file existing:

  * `results` must be non-empty. A run whose checkpoint key did not resolve
    logs "No models found to test" as a WARNING, exits 0, and still writes a
    dump - with nothing in it. File presence alone calls that a success.
  * `test_model_metrics` must cover the metrics asked for. A config
    evaluated under `combined` is not done for `training,validation`.
  * every cell of `eval_plan` (each test domain, plus the @train split and
    the zero-shot held-out domain) must appear among the result keys, so a
    run killed halfway through its domains reads as PARTIAL, not done.

The newest dump per expid wins; older ones are history.
"""

import argparse
import glob
import json
import os
import re
import sys
import time

# Same directory, not a package: make grid_status importable so both tools
# judge "trained" by one rule.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_status import (DOMAINS, MIN_CKPTS, checkpoint_counts,  # noqa: E402
                         fold_short)

DEFAULT_METRICS = "training,combined,validation"
# The paper's ladder is UNION -> GADAR-BIND -> GADAR. `joint` and `structural`
# are internal rungs with no paper column (RUNBOOK P3, CUT), so a bare
# configs/loo8_*.yaml glob pulls in 16 configs nobody will report.
PAPER_RUNGS = ("union", "joint_lite", "joint_chain")
# Evaluations written before the batched decoder's mixed-arity fix are wrong,
# not merely old: the row->graph map mis-assigned objects whenever a test
# domain's max arity differed from the model's cap (REVISION_PLAN §9). A dump
# older than this is reported `stale` and counted as still to run.
FIX_DATE = "20260905"


def config_expid(path):
    """`expid:` out of the YAML. Every suite config sets its own, so this
    needs no yaml import and no base: inheritance walk."""
    try:
        with open(path) as f:
            for line in f:
                m = re.match(r"^expid:\s*(\S+)", line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return os.path.splitext(os.path.basename(path))[0]


def still_training(name, logs_dir="logs", stale_min=90):
    """True if a training run for this config is alive right now.

    Evaluating a config mid-training picks the best checkpoint SO FAR, and
    because `done` is sticky that half-trained number becomes the reported
    one and is never re-run. So a live run excludes the config from the
    to-do list.

    Alive = a `.running` marker AND a log written within `stale_min`. The
    marker alone is not enough: a run killed by its allocation leaves the
    marker behind, and those partially-trained models (e.g. 360/500 epochs
    with 30+ checkpoints) are exactly the ones worth evaluating.
    `_mode_train` anchors the match so `..._no_grid` cannot claim
    `..._no_gripper`'s marker.
    """
    now = time.time()
    for marker in glob.glob(os.path.join(logs_dir, f"{name}_mode_train*.running")):
        log = marker[:-len(".running")] + ".log"
        try:
            if now - os.path.getmtime(log) <= stale_min * 60:
                return True
        except OSError:
            continue
    return False


def evaluating(name, logs_dir="logs", quiet_min=30):
    """True if an evaluation of this config is running right now.

    eval_queue.sh writes each config's output to logs/eval_<name>[_<tag>].log
    and main.py's last line is `Results written to ...`. A log that is still
    being written and has no such line is an eval in flight. Without this a
    second eval_all.sh launch sees the config as `missing` and evaluates it
    twice, concurrently.

    Decided from the log's CONTENT and mtime, not a marker, because the
    eval_queue.sh already running cannot be edited safely (bash reads
    scripts lazily). A log quiet for `quiet_min` without that line is a
    crashed eval and the config is offered again. The `_` / `.log` anchors
    keep `..._no_grid` from matching `..._no_gripper`.
    """
    now = time.time()
    logs = (glob.glob(os.path.join(logs_dir, f"eval_{name}.log"))
            + glob.glob(os.path.join(logs_dir, f"eval_{name}_*.log")))
    for log in logs:
        try:
            if now - os.path.getmtime(log) > quiet_min * 60:
                continue
            with open(log, "rb") as f:
                if b"Results written to" not in f.read():
                    return True
        except OSError:
            continue
    return False


def trained_counts(models, seed):
    """{config_name: n_checkpoints} for every loo8_<rung>_no_<fold> cell."""
    grid = checkpoint_counts(models, seed)
    return {f"loo8_{rung}_no_{fold_short(d)}": n
            for d in DOMAINS for rung, n in grid[d].items()}


def newest_dump(results_dir, expid):
    files = glob.glob(os.path.join(results_dir, expid, "results_*.json"))
    if not files:
        return None
    for p in sorted(files, key=os.path.getmtime, reverse=True):
        try:
            with open(p) as f:
                return json.load(f)
        except (OSError, ValueError):
            continue          # truncated dump: fall through to an older one
    return None


def assess(dump, want_metrics, since):
    """-> (state, detail). state in {'done','stale','partial','empty','missing'}
    (the caller may override with 'training')"""
    if dump is None:
        return "missing", "never evaluated"

    results = dump.get("results") or {}
    # "No models found to test" does NOT produce an empty dict: run_tests
    # returns [] for every cell, so the dump is {cell: [], ...} - keys present,
    # nothing in them. Testing the dict alone called 14 such configs `done`
    # (the union and BIND rungs, all written by one run on a node where the
    # key did not resolve) and they were never evaluated.
    if not any(results.values()):
        return "empty", "dump written but no models resolved"

    ts = str(dump.get("timestamp", ""))
    if since and ts[:8] < since:
        return "stale", f"{ts} predates the mixed-arity fix ({since})"

    raw = dump.get("test_model_metrics")
    have = set(raw or [])
    missing_m = [m for m in want_metrics if m not in have]

    # Only cells that actually hold a result count as covered.
    keys = " ".join(k for k, v in results.items() if v).lower()
    missing_cells = []
    for cell in dump.get("eval_plan") or []:
        dom = str(cell.get("domain", "")).split("@")[0].lower()
        if not dom:
            continue
        zs = cell.get("zero_shot")
        # Key shapes vary (`Visitall_ipcc_combined` vs `Visitall_ipcccombined`
        # depending on how many cells there are), so match on the domain name
        # and the zeroshot_ prefix rather than reconstructing the exact key.
        hit = dom in keys and (("zeroshot_" + dom) in keys if zs else True)
        if not hit:
            missing_cells.append(str(cell.get("domain")))

    bits = []
    if raw is None:
        # Old dumps have no such field, so which metrics ran is unknowable -
        # say that rather than listing every metric as "missing".
        bits.append("no metrics recorded (older dump format)")
    elif missing_m:
        bits.append("metrics: " + ",".join(missing_m))
    if missing_cells:
        bits.append(f"{len(missing_cells)} cell(s): "
                    + ",".join(missing_cells[:3])
                    + ("..." if len(missing_cells) > 3 else ""))
    if bits:
        return "partial", f"{ts}  " + "; ".join(bits)
    return "done", f"{len(results)} keys, {ts}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", default=None,
                    help="default: the three paper rungs of the LOO grid")
    ap.add_argument("--results-dir", default="cache/results")
    ap.add_argument("--metrics", default=DEFAULT_METRICS)
    ap.add_argument("--since", default=FIX_DATE,
                    help=f"dumps older than this are 'stale' (default "
                         f"{FIX_DATE}, the mixed-arity fix); 0 to disable")
    ap.add_argument("--all-rungs", action="store_true",
                    help="include the cut `joint` and `structural` rungs")
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--stale-min", type=int, default=90,
                    help="a training log quiet this long is a dead run, "
                         "not a live one")
    ap.add_argument("--eval-quiet-min", type=int, default=30,
                    help="an eval log quiet this long without 'Results "
                         "written to' is a crashed eval, not a running one")
    ap.add_argument("--expid-suffix", default="",
                    help="judge the results in cache/results/<config><suffix>/ "
                         "(a control pass, e.g. _untrained) instead of the "
                         "config's own")
    ap.add_argument("--models", default="models")
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--min-ckpts", type=int, default=MIN_CKPTS,
                    help="fewer checkpoints than this = not trained yet "
                         "(the same threshold as tools/grid_status.py)")
    ap.add_argument("--list-missing", action="store_true",
                    help="print only the config paths still needing eval")
    a = ap.parse_args()

    want = [m.strip() for m in a.metrics.split(",") if m.strip()]
    since = "" if a.since in ("0", "") else a.since
    if a.configs:
        configs = a.configs
    elif a.all_rungs:
        configs = sorted(glob.glob("configs/loo8_*.yaml"))
    else:
        configs = sorted(c for r in PAPER_RUNGS
                         for c in glob.glob(f"configs/loo8_{r}_no_*.yaml"))

    counts = trained_counts(a.models, a.seed)
    rows, todo = [], []
    for c in configs:
        name = os.path.basename(c)[:-5]
        expid = config_expid(c)
        state, detail = assess(newest_dump(a.results_dir, expid + a.expid_suffix),
                               want, since)
        if state != "done":
            n_ckpt = counts.get(name)
            if evaluating(name + a.expid_suffix, a.logs_dir, a.eval_quiet_min):
                state, detail = "evaluating", "eval in flight - leave it to finish"
            elif still_training(name, a.logs_dir, a.stale_min):
                state, detail = "training", "run in flight - evaluate once it finishes"
            elif n_ckpt is not None and n_ckpt < a.min_ckpts:
                # Not trained by grid_status's standard. Covers configs QUEUED
                # behind a live run (no marker yet) and stubs left by killed
                # runs: a 1-checkpoint directory would evaluate, write a
                # complete dump, read as `done`, and never be re-run.
                state = "untrained"
                detail = (f"{n_ckpt} checkpoint(s) < {a.min_ckpts} - "
                          "train it first (tools/grid_status.py)")
        rows.append((name, state, detail))
        if state not in ("done", "training", "untrained", "evaluating"):
            todo.append(c)

    if a.list_missing:
        print("\n".join(todo))
        return

    w = max((len(r[0]) for r in rows), default=20) + 2
    for name, state, detail in rows:
        print(f"{name:<{w}}{state:<11}{detail}")
    n_done = sum(1 for r in rows if r[1] == "done")
    n_train = sum(1 for r in rows if r[1] in ("training", "untrained"))
    n_eval = sum(1 for r in rows if r[1] == "evaluating")
    print(f"\n{n_done}/{len(rows)} evaluated for metrics={','.join(want)}"
          f"  ({len(todo)} to run"
          + (f", {n_eval} evaluating now" if n_eval else "")
          + (f", {n_train} held back: training or untrained" if n_train else "")
          + ")")
    if todo:
        print("\nrerun just these:")
        print("  ./train_test_scripts/eval_all.sh "
              "$(python tools/eval_status.py --list-missing)")


if __name__ == "__main__":
    main()
