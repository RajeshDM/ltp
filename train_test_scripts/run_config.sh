#!/bin/bash
# run_config.sh — launch one experiment config in the background with logging.
#
# Usage: ./train_test_scripts/run_config.sh configs/<name>.yaml [device] [extra flags...]
#   device is ALWAYS cuda:0 on this cluster (one visible GPU per allocation;
#   cuda:1+ selects a device that is not there). Extra flags go to main.py
#   (e.g. --continue-training True).
#
# Survives ssh disconnect (nohup). Log: logs/<config-name>.log
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
    mv "logs/${NAME}.log" "logs/${NAME}.$(date +%Y%m%d_%H%M%S).log"
    echo "  rotated previous log -> logs/${NAME}.$(date +%Y%m%d_%H%M%S).log"
fi

nohup python main.py --config "$CFG" --device "$DEV" "$@" \
    > "logs/${NAME}.log" 2>&1 &
PID=$!
echo "Started ${NAME} on ${DEV} (pid ${PID})"
echo "  log:    tail -f logs/${NAME}.log"
echo "  stop:   kill ${PID}"
