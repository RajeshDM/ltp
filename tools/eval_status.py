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
    """-> (state, detail). state in {'done','stale','partial','empty','missing'}"""
    if dump is None:
        return "missing", "never evaluated"

    results = dump.get("results") or {}
    if not results:
        return "empty", "dump written but no models resolved"

    ts = str(dump.get("timestamp", ""))
    if since and ts[:8] < since:
        return "stale", f"{ts} predates the mixed-arity fix ({since})"

    raw = dump.get("test_model_metrics")
    have = set(raw or [])
    missing_m = [m for m in want_metrics if m not in have]

    keys = " ".join(results.keys()).lower()
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

    rows, todo = [], []
    for c in configs:
        expid = config_expid(c)
        state, detail = assess(newest_dump(a.results_dir, expid), want, since)
        rows.append((os.path.basename(c)[:-5], state, detail))
        if state != "done":
            todo.append(c)

    if a.list_missing:
        print("\n".join(todo))
        return

    w = max((len(r[0]) for r in rows), default=20) + 2
    for name, state, detail in rows:
        print(f"{name:<{w}}{state:<9}{detail}")
    n_done = sum(1 for r in rows if r[1] == "done")
    print(f"\n{n_done}/{len(rows)} evaluated for metrics={','.join(want)}"
          f"  ({len(todo)} to run)")
    if todo:
        print("\nrerun just these:")
        print("  ./train_test_scripts/eval_all.sh "
              "$(python tools/eval_status.py --list-missing)")


if __name__ == "__main__":
    main()
