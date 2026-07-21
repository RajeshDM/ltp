#!/bin/bash
# run_config.sh — launch one experiment config in the background with logging.
#
# Usage: ./train_test_scripts/run_config.sh configs/<name>.yaml [device] [extra flags...]
#   device defaults to cuda:0. Extra flags are passed verbatim to main.py
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

nohup python main.py --config "$CFG" --device "$DEV" "$@" \
    > "logs/${NAME}.log" 2>&1 &
PID=$!
echo "Started ${NAME} on ${DEV} (pid ${PID})"
echo "  log:    tail -f logs/${NAME}.log"
echo "  stop:   kill ${PID}"
