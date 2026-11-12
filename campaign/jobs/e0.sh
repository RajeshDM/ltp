#!/bin/bash
# E0a: a non-learned baseline on all 8 domains, test and training-scale splits,
# one process per domain (the rollouts are single-threaded Python).
#   campaign/jobs/e0.sh features|achieve
# Output: cache/results/e0/<policy>_<domain>.json (+ .txt), 2 keys per file.
set -u
cd "$(dirname "$0")/../.." || exit 1
POLICY=${1:?usage: e0.sh features|achieve}
mkdir -p cache/results/e0
ITEMS="manyblocks_ipcc_big:200,manyblocks_ipcc_big@train:200
gripper_ipcc:173,gripper_ipcc@train:147
miconic_ipcc:119,miconic_ipcc@train:228
visitall_ipcc:50,visitall_ipcc@train:125
grid_ipcc:48,grid_ipcc@train:192
logistics_ipcc:96,logistics_ipcc@train:156
spanner_ipcc:96,spanner_ipcc@train:234
rovers_ipcc:54,rovers_ipcc@train:312"
# Resumable: an item whose output already holds both splits is skipped. To
# recompute one (the scorer or the featurizer changed), delete its JSON.
printf '%s\n' "$ITEMS" | xargs -P "${CAMPAIGN_WORKERS:-4}" -I@ bash -c '
    D=@; N=${D%%_ipcc*}
    python -c "import json,sys; sys.exit(0 if len(json.load(open(sys.argv[1])))>=2 else 1)" \
        "cache/results/e0/'"$POLICY"'_${N}.json" 2>/dev/null && { echo "skip $N (done)"; exit 0; }
    python tools/random_policy_baseline.py --policy '"$POLICY"' --rollouts 1 \
        --domains "$D" --output "cache/results/e0/'"$POLICY"'_${N}.json" \
        --summary "cache/results/e0/'"$POLICY"'_${N}.txt"'
