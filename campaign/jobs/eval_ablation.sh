#!/bin/bash
# Evaluate one trained LOO config two ways:
#   1. combined-loss checkpoint, every test domain (in-domain + zero-shot)
#   2. zero-shot at every periodic snapshot, both splits (EXPID_SUFFIX=_epochs)
#   campaign/jobs/eval_ablation.sh configs/loo8_jc_no_bind_no_grid.yaml
set -u
cd "$(dirname "$0")/../.." || exit 1
CFG=${1:?usage: eval_ablation.sh configs/<name>.yaml}
export WORKERS="${CAMPAIGN_WORKERS:-8}" DEV="${CAMPAIGN_DEV:-cpu}"
METRICS=combined NMODELS=1 ./train_test_scripts/eval_queue.sh "$CFG"
ZERO_SHOT_ONLY=1 METRICS=periodic NMODELS=12 EXPID_SUFFIX=_epochs \
    ./train_test_scripts/eval_queue.sh "$CFG"
