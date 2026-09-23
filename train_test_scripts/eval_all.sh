#!/bin/bash
# eval_all.sh — evaluate many configs, sized to the machine it runs on.
#
#   ./train_test_scripts/eval_all.sh                      # the 3 paper rungs x 8 folds
#   ./train_test_scripts/eval_all.sh configs/a.yaml ...   # a specific list
#
# Splits the configs across LANES concurrent eval_queue.sh lanes and gives
# each lane its own worker count, both derived from the cores this process can
# actually see (sched_getaffinity, not nproc: under a cgroup or an allocation
# they differ, and `configured_workers()` clamps against the affinity mask
# anyway - so a hardcoded 16 on an 8-core allocation just oversubscribes).
#
# Sizing:
#   LANES   = clamp(cores / 16, 1, 4)
#   WORKERS = min(16, cores/LANES - 1)      # -1 leaves a core for the parent
# 16 is the plateau of the measured worker curve (8 workers 118s, 16 workers
# 105s on 50 problems, PERFORMANCE.md); past it a worker owns problems and
# then idles, so more workers buy nothing.
#
# Both are overridable: LANES=1 WORKERS=24 ./eval_all.sh ...
#
# On LANES > 1 with DEV=cuda:0 the lanes share one GPU and time-slice its
# forward passes (36% of a visitall run, 70% of a logistics one). They still
# overlap CPU-bound graph building against each other's GPU time, so two lanes
# is usually the better use of a 32-core box - but it is a guess until you
# time it. LANES=1 is the safe comparison point.
#
# Env:
#   METRICS   default training,combined,validation (one checkpoint per
#             selection rule). METRICS=combined is the paper's stated
#             test-blind rule and ~3x cheaper.
#   NMODELS   checkpoints per metric (default 1)
#   DEV       default cuda:0. The old cpu default predates the featurizer
#             speedups, when graph build dominated and the GPU was worth
#             nothing; it no longer is.
#   WANDB=1   also log coverage online
#   REDO=1    re-evaluate configs already recorded as done
set -u

cd "$(dirname "$0")/.." || exit 1

CORES=$(python -c "import os; print(len(os.sched_getaffinity(0)))" 2>/dev/null || echo 8)
LANES="${LANES:-$(( CORES / 16 ))}"
[ "$LANES" -lt 1 ] && LANES=1
[ "$LANES" -gt 4 ] && LANES=4
if [ -z "${WORKERS:-}" ]; then
    WORKERS=$(( CORES / LANES - 1 ))
    [ "$WORKERS" -gt 16 ] && WORKERS=16
    [ "$WORKERS" -lt 1 ] && WORKERS=1
fi

METRICS="${METRICS:-training,combined,validation}"
NMODELS="${NMODELS:-1}"
DEV="${DEV:-cuda:0}"
WANDB="${WANDB:-0}"

if [ "$#" -ge 1 ]; then
    CONFIGS=("$@")
else
    # The three paper rungs only. `joint` and `structural` are internal
    # ablation rungs with no paper column (RUNBOOK P3, CUT) - a bare
    # loo8_*.yaml glob adds 16 configs nobody will report.
    CONFIGS=(configs/loo8_union_no_*.yaml configs/loo8_joint_lite_no_*.yaml
             configs/loo8_joint_chain_no_*.yaml)
fi

# Skip what is already evaluated (tools/eval_status.py decides from the
# results dump, not from the file existing). These allocations die mid-queue
# routinely, and without this a relaunch re-runs finished configs at ~1h each.
# REDO=1 evaluates everything regardless.
if [ "${REDO:-0}" != "1" ]; then
    # Capture the exit status of eval_status.py itself. Testing `$?` after a
    # `[ ... ]` reads the TEST's status, not python's, so an empty result
    # (nothing left to do) became indistinguishable from a crash and the
    # filter silently did nothing.
    if TODO_RAW=$(python tools/eval_status.py --metrics "$METRICS" \
                  --list-missing "${CONFIGS[@]}" 2>/dev/null); then
        mapfile -t TODO <<< "$TODO_RAW"
        # mapfile on an empty string yields one empty element, not none.
        [ "${#TODO[@]}" -eq 1 ] && [ -z "${TODO[0]}" ] && TODO=()
        skipped=$(( ${#CONFIGS[@]} - ${#TODO[@]} ))
        [ "$skipped" -gt 0 ] && echo "skipping $skipped already evaluated (REDO=1 to force)"
        CONFIGS=("${TODO[@]}")
    else
        echo "eval_status.py failed; evaluating every config given"
    fi
fi

if [ "${#CONFIGS[@]}" -eq 0 ]; then
    echo "nothing to evaluate for metrics=$METRICS - all configs have a complete dump."
    echo "  python tools/eval_status.py     # what is recorded"
    exit 0
fi

mkdir -p logs
echo "eval_all: ${#CONFIGS[@]} configs, $CORES cores -> $LANES lane(s) x $WORKERS workers"
echo "          metrics=$METRICS  models per metric=$NMODELS  device=$DEV"
echo

# Round-robin rather than contiguous blocks: the configs are grouped by rung,
# and a contiguous split would put every slow union run (nf163) in one lane.
for ((l = 0; l < LANES; l++)); do
    lane=()
    for ((i = l; i < ${#CONFIGS[@]}; i += LANES)); do
        lane+=("${CONFIGS[$i]}")
    done
    [ "${#lane[@]}" -eq 0 ] && continue
    log="logs/eval_lane$((l + 1)).log"
    echo "lane $((l + 1)) -> $log  (${#lane[@]} configs)"
    printf '    %s\n' "${lane[@]##*/}"
    WORKERS="$WORKERS" METRICS="$METRICS" NMODELS="$NMODELS" DEV="$DEV" \
        WANDB="$WANDB" TAG="lane$((l + 1))" \
        nohup ./train_test_scripts/eval_queue.sh "${lane[@]}" > "$log" 2>&1 &
done

echo
echo "watch:   tail -f logs/eval_lane*.log"
echo "table:   python tools/analyze_results.py"
echo "NOTE: 'NO MODELS' means the checkpoint key did not resolve - main.py"
echo "      logs that as a warning and exits 0, so it is not a failure line."
