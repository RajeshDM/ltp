#!/usr/bin/env python3
"""Result checks for campaign jobs: exit 0 only if the RESULT is real.

    python tools/campaign_checks.py trained  <config> [--since YYYYMMDD] [--min-frac 0.9]
    python tools/campaign_checks.py evaluated <expid> [--since YYYYMMDD] [--metrics combined]
                                              [--suffix _epochs] [--min-keys 1]
    python tools/campaign_checks.py globcount '<glob>' <N> [--since YYYYMMDD] [--min-keys K]
    python tools/campaign_checks.py file <path> [--since YYYYMMDD]

Each prints one line saying why it passed or failed. `--since` rejects results
older than a date, so a result from an earlier campaign (or an earlier,
pre-fix model) cannot satisfy a job that must be redone.

Why these exist: the previous campaign counted a training run whose python
crashed as finished (train_queue.sh exits 0 either way), and an evaluation
whose checkpoint key did not resolve as done (the dump exists, every cell is
an empty list). Both checks read the content, not the exit code.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _since_epoch(since):
    return time.mktime(time.strptime(since, "%Y%m%d")) if since else 0.0


def _say(ok, msg):
    print(("OK   " if ok else "NO   ") + msg)
    sys.exit(0 if ok else 1)


def check_trained(config, since, min_frac, logs):
    name = os.path.basename(config)
    name = name[:-5] if name.endswith(".yaml") else name
    exits = sorted(glob.glob(os.path.join(logs, f"{name}_mode_train*.exit")),
                   key=os.path.getmtime)
    exits = [e for e in exits if os.path.getmtime(e) >= _since_epoch(since)]
    if not exits:
        return False, f"{name}: no training exit file since {since or 'ever'}"
    exit_file = exits[-1]
    with open(exit_file) as f:
        rc = f.read().strip()
    if rc != "0":
        return False, f"{name}: training exited rc={rc} ({os.path.basename(exit_file)})"
    log = exit_file[:-len(".exit")] + ".log"
    try:
        with open(log, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 400_000))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return False, f"{name}: exit file but no log {os.path.basename(log)}"
    epochs = re.findall(r"Epoch (\d+)/(\d+)", tail)
    if not epochs:
        return False, f"{name}: rc=0 but no 'Epoch x/y' line in the log"
    last, total = (int(v) for v in epochs[-1])
    if last < min_frac * total:
        return False, f"{name}: rc=0 but only epoch {last}/{total}"
    return True, f"{name}: trained, epoch {last}/{total}"


def check_evaluated(expid, since, metrics, suffix, min_keys, results_dir):
    d = os.path.join(results_dir, expid + suffix)
    files = sorted(glob.glob(os.path.join(d, "results_*.json")), key=os.path.getmtime)
    best = None
    for path in reversed(files):
        try:
            with open(path) as f:
                dump = json.load(f)
        except (OSError, ValueError):
            continue
        if str(dump.get("timestamp", ""))[:8] < (since or ""):
            continue
        best = dump
        break
    if best is None:
        return False, f"{expid}{suffix}: no results dump since {since or 'ever'}"
    results = best.get("results") or {}
    filled = [k for k, v in results.items() if v]
    if not filled:
        return False, f"{expid}{suffix}: dump has no results (checkpoint key did not resolve?)"
    have = set(best.get("test_model_metrics") or [])
    missing = [m for m in metrics if m not in have]
    if missing:
        return False, f"{expid}{suffix}: metrics missing {missing}"
    if len(filled) < min_keys:
        return False, f"{expid}{suffix}: {len(filled)} filled cells < {min_keys}"
    return True, f"{expid}{suffix}: {len(filled)} cells, {best.get('timestamp')}"


def check_globcount(pattern, n, since, min_keys, root):
    paths = [p for p in glob.glob(os.path.join(root, pattern))
             if os.path.getmtime(p) >= _since_epoch(since)]
    good = 0
    for p in paths:
        try:
            with open(p) as f:
                data = json.load(f)
            good += isinstance(data, dict) and len(data) >= min_keys
        except (OSError, ValueError):
            pass
    return good >= n, f"{pattern}: {good}/{n} files with >= {min_keys} keys"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT)
    sub = ap.add_subparsers(dest="cmd")
    sub.required = True          # the required= keyword needs Python 3.7
    t = sub.add_parser("trained")
    t.add_argument("config")
    t.add_argument("--since", default="")
    t.add_argument("--min-frac", type=float, default=0.9)
    e = sub.add_parser("evaluated")
    e.add_argument("expid")
    e.add_argument("--since", default="")
    e.add_argument("--metrics", default="combined")
    e.add_argument("--suffix", default="")
    e.add_argument("--min-keys", type=int, default=1)
    g = sub.add_parser("globcount")
    g.add_argument("pattern")
    g.add_argument("n", type=int)
    g.add_argument("--since", default="")
    g.add_argument("--min-keys", type=int, default=1)
    f = sub.add_parser("file")
    f.add_argument("path")
    f.add_argument("--since", default="")
    a = ap.parse_args()
    root = a.root
    if a.cmd == "trained":
        _say(*check_trained(a.config, a.since, a.min_frac, os.path.join(root, "logs")))
    elif a.cmd == "evaluated":
        _say(*check_evaluated(a.expid, a.since,
                              [m for m in a.metrics.split(",") if m], a.suffix,
                              a.min_keys, os.path.join(root, "cache", "results")))
    elif a.cmd == "globcount":
        _say(*check_globcount(a.pattern, a.n, a.since, a.min_keys, root))
    else:
        p = os.path.join(root, a.path)
        ok = os.path.isfile(p) and os.path.getsize(p) > 0 and \
            os.path.getmtime(p) >= _since_epoch(a.since)
        _say(ok, f"{a.path}: {'present' if ok else 'missing, empty or older than --since'}")


if __name__ == "__main__":
    main()
