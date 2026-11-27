#!/bin/bash
# run_config.sh — launch one experiment config in the background with logging.
#
# Usage: ./train_test_scripts/run_config.sh configs/<name>.yaml [device] [extra flags...]
#   device is ALWAYS cuda:0 on this cluster (one visible GPU per allocation;
#   cuda:1+ selects a device that is not there). Extra flags go to main.py
#   (e.g. --continue-training True).
#
# Survives ssh disconnect (nohup).
#
# THE LOG NAME IS DERIVED, NOT REMEMBERED. It is the config name plus any
# extra flags you passed:
#
#   run_config.sh configs/x.yaml cuda:0                  -> logs/x.log
#   run_config.sh configs/x.yaml cuda:0 --seed 12        -> logs/x_seed_12.log
#   run_config.sh configs/x.yaml cuda:0 --num-train-problems 25
#                                                        -> logs/x_num-train-problems_25.log
#
# so the seed replicates and data-fraction arms that used to need RUN_TAG
# separate themselves. This exists because RUN_TAG was a memory dependency
# and memory lost: two sweep arms once ran concurrently under one name.
# RUN_TAG still works and is appended after the derived suffix; it is now
# only for distinguishing runs that differ in NO flag at all.
#
# Launching a run whose derived name is already alive is REFUSED - identical
# config and identical flags means an identical checkpoint key too, so the
# two would evict each other's checkpoints, not merely share a log.
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 configs/<name>.yaml [device] [extra main.py flags...]"
    exit 1
fi

CFG=$1
DEV=${2:-cuda:0}
shift $(( $# >= 2 ? 2 : 1 ))

if [ ! -f "$CFG" ]; then
    echo "Config not found: $CFG"
    exit 1
fi

NAME=$(basename "$CFG" .yaml)
# The extra flags ARE the difference between two launches of one config, so
# they are the name. `--seed 12` -> `seed_12`; anything not safe in a
# filename collapses to '_'.
if [ "$#" -gt 0 ]; then
    SUFFIX=$(printf '%s_' "$@" \
             | sed -e 's/--//g' -e 's|[^A-Za-z0-9_.-]|_|g' \
                   -e 's/__*/_/g' -e 's/^_//' -e 's/_$//')
    [ -n "$SUFFIX" ] && NAME="${NAME}_${SUFFIX}"
fi
[ -n "${RUN_TAG:-}" ] && NAME="${NAME}_${RUN_TAG}"
mkdir -p logs

# A live run under this exact name means the same config AND the same flags,
# hence the same checkpoint key: the two would compete for the same
# loss-ranked slots and each would delete the other's best models. Refuse
# rather than let that happen silently, which is how the sweep_jc_base /
# sweep_jc_l2 arms destroyed each other.
RUNFILE="logs/${NAME}.running"
if [ -f "$RUNFILE" ] && kill -0 -- -"$(cat "$RUNFILE" 2>/dev/null)" 2>/dev/null; then
    echo "REFUSING: ${NAME} is already running (process group $(cat "$RUNFILE"))."
    echo "  Same config and same flags means the same checkpoint key, so the"
    echo "  two runs would evict each other's checkpoints - not just share a log."
    echo "  For a variant, pass the flag that differs (e.g. --seed 12); the"
    echo "  log name follows the flags. To run it anyway, RUN_TAG=<suffix>."
    exit 1
fi
rm -f "$RUNFILE"
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
# print() is block-buffered when stdout is a file, so progress lines lag the
# logger's by many minutes and a healthy run looks stalled. Logging is the
# only window into a nohup'd job; keep it in order.
export PYTHONUNBUFFERED=1

# Never clobber a previous run's log: the same config is often re-run with
# different flags (--representation-size, --gnn-rounds, --seed ...), and the
# stdout log is the ONLY place the loss curve lives (metrics survive in the
# results JSON, checkpoints in models/). Rotate instead of truncating.
if [ -f "logs/${NAME}.log" ]; then
    # One `date`, not two: called twice these can straddle a second and the
    # name echoed is not the name written.
    ROTATED="logs/${NAME}.$(date +%Y%m%d_%H%M%S).log"
    mv "logs/${NAME}.log" "$ROTATED"
    echo "  rotated previous log -> $ROTATED"
fi

# Record the exit status. A run that is killed by a signal writes no
# traceback anywhere, and the status is the only evidence of what happened:
# 137 = SIGKILL (OOM killer or scheduler), 139 = SIGSEGV, 134 = abort (often
# a CUDA/driver failure), 1 = an ordinary Python exception. Without this the
# post-mortem is guesswork.
rm -f "logs/${NAME}.exit"
# setsid puts the wrapper in its own process group so `kill -- -PID` stops
# the python child with it. Without it the wrapper is in the launching
# shell's group: killing the recorded pid signals only bash (which is
# sitting in wait) and leaves training running, invisibly, holding the GPU.
SETSID=$(command -v setsid || true)
nohup ${SETSID} bash -c '
    # The process GROUP id, not $$: setsid does not always make this shell
    # the group leader, and `kill -- -PID` on a non-leader either fails or
    # signals the wrong group. The group is what stops the run.
    ps -o pgid= -p $$ | tr -d " " > "logs/'"${NAME}"'.running"
    python main.py --config "$1" --device "$2" "${@:3}"
    echo $? > "logs/'"${NAME}"'.exit"
    rm -f "logs/'"${NAME}"'.running"
' _ "$CFG" "$DEV" "$@" > "logs/${NAME}.log" 2>&1 &
PID=$!
echo "Started ${NAME} on ${DEV} (pid ${PID})"
echo "  log:    tail -f logs/${NAME}.log"
# Kill THIS run, not every variant of the config: with seed replicates or
# data-fraction arms in flight, `pkill -f configs/<name>.yaml` takes them all
# down. The .running file names the one wrapper to signal, and killing the
# wrapper's process group stops the python child with it.
echo "  stop:   kill -- -\$(cat logs/${NAME}.running)   # the '-' is the process GROUP"
