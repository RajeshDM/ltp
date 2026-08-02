#!/bin/bash
# eval_queue.sh — run several configs through `--mode test`, ONE AT A TIME.
#
# Usage: ./train_test_scripts/eval_queue.sh configs/a.yaml configs/b.yaml ...
#
# Evaluations must not overlap. Each wants GABAR_FEATURIZE_WORKERS=16 (the
# plateau, PERFORMANCE.md) and `configured_workers()` clamps against the
# affinity mask, which every concurrent run sees as the full node - so two
# at once oversubscribe rather than share, and both get slower. This script
# is the queue that keeps that from happening by accident.
#
# Runs on --device cpu on purpose. Evaluation is host-CPU bound: 137s on GPU
# against 135s on CPU for the same 20 visitall problems (H100 node, batched).
# Costing nothing, it leaves the whole GPU to concurrent training.
#
# Env:
#   WORKERS=N       rollout workers per run (default 16)
#   METRICS=...     --test-model-metrics (default training,combined,validation)
#   NMODELS=N       --num-models-to-test (default 2)
#   DEV=cuda:0      evaluate on GPU instead (only if nothing is training)
#   WANDB=1         --wandb True, so coverage lands online as well as in JSON
#   EXTRA="..."     extra main.py flags, e.g. EXTRA="--seed 12"
#   TAG=<suffix>    appended to the log name; use it when the same config is
#                   queued more than once (per-seed evaluation)
#
# Survives ssh disconnect: launch it under nohup itself.
#   nohup ./train_test_scripts/eval_queue.sh configs/*.yaml > logs/queue.log 2>&1 &
set -u

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 configs/<name>.yaml [configs/<name>.yaml ...]"
    exit 1
fi

WORKERS="${WORKERS:-16}"
METRICS="${METRICS:-training,combined,validation}"
NMODELS="${NMODELS:-2}"
DEV="${DEV:-cpu}"
EXTRA="${EXTRA:-}"
TAG="${TAG:-}"
WANDB_FLAG=""
[ "${WANDB:-0}" = "1" ] && WANDB_FLAG="--wandb True"

mkdir -p logs
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export PYTHONUNBUFFERED=1

CORES=$(python -c "import os; print(len(os.sched_getaffinity(0)))" 2>/dev/null || echo "?")
echo "eval_queue: $# configs, workers=$WORKERS, device=$DEV, cores visible=$CORES"
echo "            metrics=$METRICS, models per metric=$NMODELS"
echo

for CFG in "$@"; do
    if [ ! -f "$CFG" ]; then
        echo "SKIP (not found): $CFG"
        continue
    fi
    NAME=$(basename "$CFG" .yaml)
    [ -n "$TAG" ] && NAME="${NAME}_${TAG}"
    LOG="logs/eval_${NAME}.log"
    [ -f "$LOG" ] && mv "$LOG" "logs/eval_${NAME}.$(date +%Y%m%d_%H%M%S).log"

    printf "%-34s " "$NAME"
    T0=$SECONDS
    GABAR_BATCH_EVAL=1 GABAR_FEATURIZE_WORKERS="$WORKERS" \
       python main.py --config "$CFG" --mode test --device "$DEV" \
            --test-model-metrics "$METRICS" \
            --num-models-to-test "$NMODELS" \
            $WANDB_FLAG $EXTRA > "$LOG" 2>&1
    RC=$?

    # Checked BEFORE the exit code: main.py logs "No models found to test" as
    # a warning and exits 0, so a run that tested nothing at all otherwise
    # reports "ok" and the empty results file looks like a real result.
    if grep -aq "No models found" "$LOG"; then
        printf "NO MODELS %4ds  (nothing trained for this checkpoint key)\n" \
               "$((SECONDS - T0))"
    elif [ "$RC" -eq 0 ]; then
        printf "ok   %5ds\n" "$((SECONDS - T0))"
    else
        printf "FAIL %5ds  (%s)\n" "$((SECONDS - T0))" "$LOG"
    fi
done

echo
echo "results: cache/results/<expid>/results_*.json"
echo "table:   python tools/analyze_results.py"
