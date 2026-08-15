#!/bin/bash
# run_config.sh — launch one experiment config in the background with logging.
#
# Usage: ./train_test_scripts/run_config.sh configs/<name>.yaml [device] [extra flags...]
#   device is ALWAYS cuda:0 on this cluster (one visible GPU per allocation;
#   cuda:1+ selects a device that is not there). Extra flags go to main.py
#   (e.g. --continue-training True).
#
# Survives ssh disconnect (nohup). Log: logs/<config-name>.log
#
# RUN_TAG=<suffix> makes the log logs/<config-name>_<suffix>.log instead.
# REQUIRED when launching the same config several times concurrently (seed
# replicates, data-fraction arms): without it every launch rotates the
# previous one's log and then all of them append to the same path, so the
# loss curves interleave into garbage and the only record of training is
# lost. It changes the log name ONLY - nothing about the run, the checkpoint
# key or the results JSON.
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
[ -n "${RUN_TAG:-}" ] && NAME="${NAME}_${RUN_TAG}"
mkdir -p logs
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
nohup bash -c '
    python main.py --config "$1" --device "$2" "${@:3}"
    echo $? > "logs/'"${NAME}"'.exit"
' _ "$CFG" "$DEV" "$@" > "logs/${NAME}.log" 2>&1 &
PID=$!
echo "Started ${NAME} on ${DEV} (pid ${PID})"
echo "  log:    tail -f logs/${NAME}.log"
# ${PID} is the wrapper that records the exit status, not python itself, so
# killing it would orphan the run. Match on the config instead.
echo "  stop:   pkill -f \"configs/$(basename "$CFG")\""
