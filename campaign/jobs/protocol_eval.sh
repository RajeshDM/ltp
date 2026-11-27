#!/bin/bash
# Evaluate one trained p2_* config under the 2026-11 protocol (C1):
#   1. sweep  - every periodic snapshot on each TRAINING domain's @val_ood
#               -> cache/results/<expid>_sweep/
#   2. select - tools/select_checkpoint.py: best mean Val-OOD coverage, ties
#               -> earlier; written to cache/results/<expid>/selected_checkpoint.json
#   3. test   - that one snapshot on the config's test_domains (the bands)
#               -> cache/results/<expid>_final/
# Constrained, batched decoding throughout. Each step is skipped when its
# result already exists, so a preempted job resumes.
#   campaign/jobs/protocol_eval.sh configs/p2_sd_jc_gripper.yaml
set -u
cd "$(dirname "$0")/../.." || exit 1
CFG=${1:?usage: protocol_eval.sh configs/<name>.yaml}
BASE=${EXPID:-$(basename "$CFG" .yaml)}   # EXPID / EXTRA / MIN_SNAPSHOTS: smoke runs
EXTRA=${EXTRA:-}
SEL=(python tools/select_checkpoint.py --config "$CFG" --expid "$BASE" --min-epochs "${MIN_SNAPSHOTS:-10}")
DEV="${CAMPAIGN_DEV:-cuda:0}"
export GABAR_CONSTRAINED_DECODE=1 GABAR_BATCH_EVAL=1 GABAR_FEATURIZE_WORKERS="${CAMPAIGN_WORKERS:-8}"
mkdir -p logs/protocol

run() {   # run <log> <main.py args...>: fail loudly on an empty result
    local log=$1; shift
    [ -f "$log" ] && mv "$log" "${log%.log}.$(date +%Y%m%d_%H%M%S).log"
    python main.py --config "$CFG" --mode test --device "$DEV" $EXTRA "$@" > "$log" 2>&1
    local rc=$?
    if grep -aq "No models found" "$log"; then echo "NO MODELS for $BASE ($log)"; exit 3; fi
    [ $rc -eq 0 ] || { echo "FAIL rc=$rc ($log)"; tail -5 "$log"; exit $rc; }
}

if ! "${SEL[@]}" --print-epoch > /dev/null 2>&1; then
    echo "[$BASE] sweep: periodic snapshots on Val-OOD"
    run "logs/protocol/${BASE}_sweep.log" --test-model-metrics periodic --num-models-to-test 10 \
        --test-domains "$(python tools/select_checkpoint.py --config "$CFG" --val-domains)" \
        --expid "${BASE}_sweep"
fi
"${SEL[@]}" --write || exit 1
EPOCH=$("${SEL[@]}" --print-epoch) || exit 1
echo "[$BASE] test: epoch $EPOCH on the test bands"
run "logs/protocol/${BASE}_final.log" --test-model-metrics periodic --test-epochs "$EPOCH" \
    --num-models-to-test 1 --expid "${BASE}_final"
echo "[$BASE] done: cache/results/${BASE}_final/"
