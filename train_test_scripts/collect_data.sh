#!/bin/bash
# collect_data.sh — build every cache the experiment suite needs, then exit.
#
# Run ONCE per storage system (sequentially - concurrent first-time
# collection on one filesystem is wasteful). After it finishes, any number
# of GPU nodes sharing that filesystem can launch training runs that only
# READ caches - no concurrency concerns.
#
# Idempotent: complete caches load in seconds and are skipped naturally;
# re-running after an interruption resumes where it left off. A run that
# errors AFTER printing "Size of dataset" still built its caches (the
# epochs-0 trick stops before training; some versions exit uncleanly).
#
# Usage:
#   ./train_test_scripts/collect_data.sh                    # every method
#   ./train_test_scripts/collect_data.sh joint_chain        # one method
#   ./train_test_scripts/collect_data.sh joint union        # several methods
#   ./train_test_scripts/collect_data.sh all8_joint_chain \
#       loo8_joint_chain_no_visitall                        # EXACT configs
#
# The last form is what you want when two nodes split the work by method:
# collecting a method pulls in all 8 of its LOO variants (~2 min each once
# the plans exist), and you may only intend to train three of them. Naming
# configs collects exactly those and nothing else. Arguments are matched
# against configs/<arg>.yaml first, then treated as a method name.
#
# Order: all8 configs first (does the one-time FD plan collection for all
# 8 domains - the expensive part, hours), then LOO / ho2 / ho4 tags
# (featurization only, minutes each).
set -uo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

FEATS=("$@")
if [ ${#FEATS[@]} -eq 0 ]; then
    # joint_chain is the headline method (full GADAR); leaving it out of the
    # default meant a bare `collect_data.sh` built caches for every rung
    # EXCEPT the one the paper leads with.
    FEATS=(union structural joint_lite joint joint_chain)
fi

CONFIGS=()
# Any argument naming a real config is taken literally; the rest are methods.
REMAINING=()
for a in "${FEATS[@]}"; do
    if [ -f "configs/${a}.yaml" ]; then
        CONFIGS+=("configs/${a}.yaml")
    elif [ -f "$a" ]; then
        CONFIGS+=("$a")
    else
        REMAINING+=("$a")
    fi
done
FEATS=(${REMAINING[@]+"${REMAINING[@]}"})

for feat in ${FEATS[@]+"${FEATS[@]}"}; do
    for cfg in configs/all8_${feat}.yaml; do
        [ -f "$cfg" ] && CONFIGS+=("$cfg")
    done
done
for feat in ${FEATS[@]+"${FEATS[@]}"}; do
    for cfg in configs/loo8_${feat}_no_*.yaml configs/ho2_${feat}_*.yaml configs/ho4_${feat}.yaml; do
        [ -f "$cfg" ] || continue
        # 'joint' globs also match 'joint_lite'/'joint_chain' files - skip those
        if [ "$feat" = "joint" ] && [[ "$(basename "$cfg")" == *"joint_lite"* || "$(basename "$cfg")" == *"joint_chain"* ]]; then
            continue
        fi
        CONFIGS+=("$cfg")
    done
done

total=${#CONFIGS[@]}
echo "Data collection for ${total} config(s)${FEATS[0]+, methods: ${FEATS[*]}}"
echo "Started: $(date)"
echo ""

ok=0; failed=0; i=0
for cfg in "${CONFIGS[@]}"; do
    i=$((i + 1))
    name=$(basename "$cfg" .yaml)
    log="logs/collect_${name}.log"
    printf "[%2d/%2d] %-40s " "$i" "$total" "$name"
    start=$(date +%s)
    python main.py --config "$cfg" --mode train --epochs 0 \
        --device cpu --use-gpu False > "$log" 2>&1
    status=$?
    took=$(( $(date +%s) - start ))
    # "Size of dataset" prints only after every domain's cache is built and
    # saved - that is the success signal, regardless of exit status.
    if grep -q "Size of dataset" "$log"; then
        ok=$((ok + 1))
        printf "OK   (%4ds)\n" "$took"
    else
        failed=$((failed + 1))
        printf "FAIL (%4ds) - see %s\n" "$took" "$log"
    fi
done

echo ""
echo "Finished: $(date)  -  ${ok}/${total} OK, ${failed} failed"
if [ "$failed" -gt 0 ]; then
    echo "Re-run this script to resume failed entries (completed caches are skipped)."
    exit 1
fi
echo "All caches ready. GPU nodes on this filesystem can now launch:"
echo "  ./train_test_scripts/run_config.sh configs/<name>.yaml cuda:0"
