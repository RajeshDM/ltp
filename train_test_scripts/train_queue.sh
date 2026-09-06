#!/bin/bash
# train_queue.sh — train several configs on ONE GPU, one at a time.
#
# Usage:
#   nohup ./train_test_scripts/train_queue.sh cuda:0 \
#       configs/a.yaml configs/b.yaml ... > logs/train_queue_A.log 2>&1 &
#
# run_config.sh backgrounds each launch, so a `for` loop over it puts every
# config on the GPU at once. Concurrent runs time-slice a single GPU rather
# than sharing it (PERFORMANCE.md), so N at once finishes no sooner than N
# in sequence and each one's log is N times slower to become useful. This is
# the queue.
#
# TRAINING ONLY: every run gets `--mode train`, so no evaluation happens
# here. Evaluation is host-CPU bound and belongs on a CPU-rich machine
# afterwards (eval_queue.sh), where it costs no GPU time at all.
#
# Env:
#   EXTRA="..."   extra main.py flags for every config (e.g. EXTRA="--seed 12").
#                 They are folded into each per-config log name automatically.
#   MODE=train    override the mode (train_test if you really want testing).
#   DEV           alternative to the first positional argument.
#
# Per-config logs are logs/<config>_mode_train[_<extra>].log, written by
# run_config.sh's derived naming - nothing to remember and nothing to pass.
# This script's own stdout is just the queue's progress.
set -u

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <device> configs/<name>.yaml [configs/<name>.yaml ...]"
    echo "   e.g. $0 cuda:0 configs/loo8_union_no_gripper.yaml"
    exit 1
fi

DEV=${1:-cuda:0}
shift
MODE="${MODE:-train}"
EXTRA="${EXTRA:-}"
HERE=$(dirname "$0")

mkdir -p logs
echo "train_queue: $# configs on $DEV, mode=$MODE ${EXTRA:+extra=$EXTRA}"
echo "             started $(date '+%Y-%m-%d %H:%M:%S')"
echo

QUEUE_START=$SECONDS
for CFG in "$@"; do
    if [ ! -f "$CFG" ]; then
        echo "SKIP (not found): $CFG"
        continue
    fi
    NAME=$(basename "$CFG" .yaml)
    printf '%-36s %s  ' "$NAME" "$(date '+%H:%M:%S')"
    T0=$SECONDS

    # run_config.sh backgrounds the run and refuses a duplicate name; wait
    # for the one it started rather than launching the next config on top.
    OUT=$("$HERE/run_config.sh" "$CFG" "$DEV" --mode "$MODE" $EXTRA 2>&1)
    if ! printf '%s' "$OUT" | grep -q '^Started '; then
        echo "LAUNCH REFUSED"
        printf '%s\n' "$OUT" | sed 's/^/    /'
        continue
    fi
    RUNFILE=$(printf '%s' "$OUT" | sed -n 's/.*tail -f \(logs\/.*\)\.log/\1.running/p')
    LOG="${RUNFILE%.running}.log"

    # Poll the run file: it is written by the wrapper on start and removed on
    # exit, so its disappearance is the completion signal. Nothing here has
    # to know the pid.
    while [ -f "$RUNFILE" ]; do sleep 30; done

    ELAPSED=$((SECONDS - T0))
    RC=$(cat "${LOG%.log}.exit" 2>/dev/null || echo "?")
    if [ "$RC" = "0" ]; then
        printf 'ok    %5dm\n' "$((ELAPSED / 60))"
    else
        # 137 = SIGKILL (OOM killer or scheduler), 139 = SIGSEGV,
        # 134 = abort (often CUDA/driver), 1 = a Python exception.
        printf 'FAIL  %5dm  rc=%s  %s\n' "$((ELAPSED / 60))" "$RC" "$LOG"
        tail -3 "$LOG" 2>/dev/null | sed 's/^/    /'
    fi
done

echo
echo "queue done in $(( (SECONDS - QUEUE_START) / 60 ))m at $(date '+%Y-%m-%d %H:%M:%S')"
echo "checkpoints: models/   |   evaluate later with eval_queue.sh on a CPU box"
