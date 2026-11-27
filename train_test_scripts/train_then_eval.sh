#!/bin/bash
# train_then_eval.sh — for each config: evaluate it if its models exist,
# otherwise train it (train_queue.sh) and then evaluate. One at a time.
#
#   nohup ./train_test_scripts/train_then_eval.sh configs/baseline_*.yaml \
#       > logs/tte_<machine>.log 2>&1 &
#
# For short single-domain runs (the GABAR baselines) where it is unknown
# whether an earlier campaign already trained them: an existing checkpoint
# key is reused, a missing one costs one training run. Evaluation device is
# the GPU when CUDA works, else the CPU. Env: METRICS (combined), NMODELS (1).
set -u
cd "$(dirname "$0")/.." || exit 1
if ! python -c "import torch" 2>/dev/null; then
    echo "REFUSING: 'import torch' fails - conda activate di_ltp_1 first."
    exit 1
fi
DEV=$(python -c "import torch; print('cuda:0' if torch.cuda.is_available() else 'cpu')")
export METRICS="${METRICS:-combined}" NMODELS="${NMODELS:-1}" DEV
echo "train_then_eval: $# configs, device=$DEV, metrics=$METRICS"
for CFG in "$@"; do
    NAME=$(basename "$CFG" .yaml)
    OUT=$(./train_test_scripts/eval_queue.sh "$CFG" | grep -aE " (ok|FAIL|NO MODELS)")
    echo "$(date +%T) eval  $OUT"
    case "$OUT" in
        *"NO MODELS"*)
            ./train_test_scripts/train_queue.sh cuda:0 "$CFG" | grep -aE "^$NAME|FAIL|REFUS"
            echo "$(date +%T) eval  $(./train_test_scripts/eval_queue.sh "$CFG" | grep -aE " (ok|FAIL|NO MODELS)")" ;;
    esac
done
echo "$(date +%T) done"
