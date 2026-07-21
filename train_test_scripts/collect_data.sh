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
#   ./train_test_scripts/collect_data.sh              # everything (all 4 methods)
#   ./train_test_scripts/collect_data.sh joint        # GADAR only (fastest start)
#   ./train_test_scripts/collect_data.sh joint union  # multiple methods
#
# Order: all8 configs first (does the one-time FD plan collection for all
# 8 domains - the expensive part, hours), then LOO / ho2 / ho4 tags
# (featurization only, minutes each).
set -uo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

FEATS=("$@")
if [ ${#FEATS[@]} -eq 0 ]; then
    FEATS=(union structural joint_lite joint)
fi

CONFIGS=()
for feat in "${FEATS[@]}"; do
    for cfg in configs/all8_${feat}.yaml; do
        [ -f "$cfg" ] && CONFIGS+=("$cfg")
    done
done
for feat in "${FEATS[@]}"; do
    for cfg in configs/loo8_${feat}_no_*.yaml configs/ho2_${feat}_*.yaml configs/ho4_${feat}.yaml; do
        [ -f "$cfg" ] || continue
        # 'joint' globs also match 'joint_lite' files - skip those
        if [ "$feat" = "joint" ] && [[ "$(basename "$cfg")" == *"joint_lite"* ]]; then
            continue
        fi
        CONFIGS+=("$cfg")
    done
done

total=${#CONFIGS[@]}
echo "Data collection for ${total} config(s), methods: ${FEATS[*]}"
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
