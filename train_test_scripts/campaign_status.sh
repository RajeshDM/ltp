#!/bin/bash
# campaign_status.sh — everything in flight, training and evaluation, at once.
#
#   ./train_test_scripts/campaign_status.sh
#
# Reads files on the shared disk (plus squeue if present), so it works from
# the login node or any compute node, and says the same thing from each.
# It only calls the tools that already own each question:
#   allocations       squeue
#   training runs     train_test_scripts/queue_status.sh
#   training grid     tools/grid_status.py
#   evaluation        tools/eval_status.py
#   eval claims       logs/eval_claims/*/owner (eval_worker.sh)
#   problems          FAIL / NO MODELS / tracebacks in recent logs
set -u
cd "$(dirname "$0")/.." || exit 1

hr() { printf '\n=== %s ===\n' "$1"; }

if command -v squeue >/dev/null 2>&1; then
    hr "allocations"
    squeue -u "$USER" -o "%.10i %.9P %.10T %.12M %.8C %R" 2>/dev/null
fi

hr "training (live runs)"
# 60, not the default 20: the slowest runs print every ~34 min, and 20 would
# call them dead.
STALE_MIN=60 ./train_test_scripts/queue_status.sh 2>/dev/null | grep -v " done$" \
    | grep -vE "^q_.* ([0-9]+) of \1 configs finished" || true

hr "training grid"
python tools/grid_status.py 2>/dev/null | grep -E "COMPLETE|/3$|cells" \
    | sed 's/^/  /'

hr "evaluation"
python tools/eval_status.py 2>/dev/null | grep -vE "^(rerun just these|  \./train)" \
    | grep -E "evaluating|training|untrained|partial|empty|evaluated for"

if ls logs/eval_claims/*/owner >/dev/null 2>&1; then
    hr "eval claims (config <- machine)"
    for o in logs/eval_claims/*/owner; do
        printf '  %-34s <- %s\n' "$(basename "$(dirname "$o")")" "$(cat "$o")"
    done
fi

hr "problems in the last 24h"
found=0
while IFS= read -r f; do
    hits=$(grep -aE " FAIL |NO MODELS|Traceback|RuntimeError|CUDA out of memory" "$f" | tail -2)
    if [ -n "$hits" ]; then
        found=1
        echo "  $f"
        printf '%s\n' "$hits" | sed 's/^/      /'
    fi
done < <(find logs -maxdepth 1 -name "*.log" -mmin -1440 2>/dev/null)
[ "$found" -eq 0 ] && echo "  none"
echo
