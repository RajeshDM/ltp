#!/bin/bash
# Read out a STILL-TRAINING run's periodic checkpoints after a delay.
#
#   bash tools/delayed_test.sh <config.yaml> <device> <delay_hours> \
#        <test-domains> [num_models]
#
# The five train_test jobs test themselves when training ends. This is the
# EARLY probe: it tests whatever periodic checkpoints exist at wake-up time,
# so a dead result is known by morning instead of tomorrow afternoon.
#
# Must run on the SAME filesystem as the training job -- it reads that job's
# models/ directory. Checkpoint selection order is latest-first (main.py
# reverses the list), so a small num_models still sees the newest epochs.
#
# SEED must match the training run (default 30, override with SEED=n).
set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <config.yaml> <device> <delay_hours> <test-domains> [num_models]"
    echo "  e.g. $0 configs/loo8_joint_chain_no_gripper.yaml cuda:5 5 'gripper_ipcc@train:80' 3"
    exit 1
fi

CFG=$1; DEV=$2; HOURS=$3; DOMAINS=$4; NMODELS=${5:-3}
SEED=${SEED:-30}

[ -f "$CFG" ] || { echo "Config not found: $CFG"; exit 1; }

NAME=$(basename "$CFG" .yaml)
LOG="logs/${NAME}.earlytest.log"
SECS=$(awk "BEGIN{printf \"%d\", $HOURS*3600}")
mkdir -p logs
[ -f "$LOG" ] && mv "$LOG" "logs/${NAME}.earlytest.$(date +%Y%m%d_%H%M%S).log"

export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export PYTHONUNBUFFERED=1

nohup bash -c "
    sleep $SECS
    echo \"=== early readout started \$(date) ===\"
    python main.py --config '$CFG' --mode test --seed $SEED --device '$DEV' \
        --test-domains '$DOMAINS' \
        --test-model-metrics periodic --num-models-to-test $NMODELS
" > "$LOG" 2>&1 &

echo "Scheduled early test for ${NAME}"
echo "  fires:  $(date -d "+${SECS} seconds" 2>/dev/null || echo "in ${HOURS}h")"
echo "  device: ${DEV}   models: ${NMODELS} (latest periodic first)   seed: ${SEED}"
echo "  domains:${DOMAINS}"
echo "  log:    tail -f ${LOG}"
echo "  cancel: kill $!"
