#!/bin/bash
# eval_worker.sh — evaluation across ANY number of machines, no split to plan.
#
#   ./train_test_scripts/eval_worker.sh          # run the same line everywhere
#
# Every machine runs this identical command. Each lane repeatedly asks
# tools/eval_status.py what still needs evaluating, CLAIMS one config with an
# atomic `mkdir logs/eval_claims/<config>` on the shared disk, evaluates it,
# releases the claim, and asks again. Lanes exit when nothing is left.
#
# Why not a fixed split: the machines are independent, several are on the
# preempt partition, and more join whenever the scheduler grants them. A
# fixed split strands a preempted box's half until someone remembers which
# slice it had, and a late joiner cannot take any of it. With claims:
#   - a new machine just runs this line and starts taking work;
#   - a preempted machine's claim goes stale and is taken over (below);
#   - rerunning it anywhere, any number of times, is always safe.
#
# Stale claims: a claim is honoured while it is younger than CLAIM_TTL_MIN
# (default 15) OR its config's eval log was written in the last 30 min
# (main.py prints progress continuously, so a live eval keeps it fresh).
# Past both, the holder is dead - preempted, killed - and the claim is taken
# over. On NFS two lanes reclaiming the SAME dead claim in the same instant
# can both win; the cost is one duplicated evaluation, not a wrong number
# (evaluation is seeded per problem, and the newest dump wins).
#
# Sizing per machine is eval_all.sh's: LANES = clamp(cores/16, 1, 4),
# WORKERS = min(16, cores/LANES - 1), from sched_getaffinity.
#
# Env: METRICS (default training,combined,validation), NMODELS (1),
#      DEV (cuda:0), LANES, WORKERS, CLAIM_TTL_MIN, WANDB=1,
#      ZERO_SHOT_ONLY=1 (held-out domain only),
#      EXPID_SUFFIX=_x (a separate results namespace for a control pass).
#      All are inherited by eval_queue.sh.
set -u

cd "$(dirname "$0")/.." || exit 1

# Refuse to run where evaluation must not run. Launched on the login node it
# sized itself to 64 shared cores and 4 lanes, and with DEV=cuda:0 on a
# GPU-less host every lane would claim a config, crash, release it and claim
# the next - churning the whole queue while loading a machine every cluster
# user depends on. Both checks are cheap and happen before anything is
# claimed. ALLOW_OUTSIDE_SLURM=1 skips the first, for a workstation.
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_OUTSIDE_SLURM:-0}" != "1" ]; then
    echo "REFUSING: not inside a SLURM allocation (no SLURM_JOB_ID) - this"
    echo "  looks like a login node. Run it inside an srun/salloc session on a"
    echo "  compute node. ALLOW_OUTSIDE_SLURM=1 overrides (not on a login node)."
    exit 1
fi
if [[ "${DEV:-cuda:0}" == cuda* ]] \
   && ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "REFUSING: DEV=${DEV:-cuda:0} but no usable CUDA device on $(hostname -s)."
    echo "  Every lane would claim a config, crash, and claim the next."
    echo "  On a CPU-only node use DEV=cpu."
    exit 1
fi

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
CLAIM_TTL_MIN="${CLAIM_TTL_MIN:-15}"
HOST="${HOST_OVERRIDE:-$(hostname -s)}"
CLAIMS="logs/eval_claims"
mkdir -p "$CLAIMS"

# claim <name> -> 0 if this lane now owns it
claim() {
    local d="$CLAIMS/$1"
    if mkdir "$d" 2>/dev/null; then
        echo "$HOST $$ $(date '+%F %T')" > "$d/owner"
        return 0
    fi
    # Held. Honour it while the claim is young or its eval log is live.
    [ -n "$(find "$d" -maxdepth 0 -mmin -"$CLAIM_TTL_MIN" 2>/dev/null)" ] && return 1
    [ -n "$(find logs -maxdepth 1 -name "eval_$1_*.log" -mmin -30 2>/dev/null)" ] && return 1
    # Dead holder: take it over.
    rm -rf "$d"
    if mkdir "$d" 2>/dev/null; then
        echo "$HOST $$ $(date '+%F %T') (took over a stale claim)" > "$d/owner"
        return 0
    fi
    return 1
}

lane() {
    local k=$1 got cfg name
    while :; do
        got=""
        for cfg in $(python tools/eval_status.py --metrics "$METRICS" \
                       --expid-suffix "${EXPID_SUFFIX:-}" --list-missing); do
            # The claim namespace includes the suffix, so a control pass and
            # the real evaluation of the same config never block each other.
            name="$(basename "$cfg" .yaml)${EXPID_SUFFIX:-}"
            if claim "$name"; then got="$cfg"; break; fi
        done
        if [ -z "$got" ]; then
            echo "[$HOST lane$k] $(date '+%T') nothing left to claim - exiting"
            return
        fi
        name="$(basename "$got" .yaml)${EXPID_SUFFIX:-}"
        echo "[$HOST lane$k] $(date '+%T') claimed $name"
        # TAG is unique per machine+lane, so per-config logs never collide
        # and eval_status can see this eval as in flight.
        WORKERS="$WORKERS" METRICS="$METRICS" NMODELS="$NMODELS" DEV="$DEV" \
            WANDB="$WANDB" TAG="${HOST}_lane$k" \
            ./train_test_scripts/eval_queue.sh "$got" | grep -aE " (ok|FAIL|NO MODELS)"
        rm -rf "$CLAIMS/$name"
    done
}

echo "eval_worker on $HOST: $CORES cores -> $LANES lane(s) x $WORKERS workers"
echo "  metrics=$METRICS  models per metric=$NMODELS  device=$DEV" \
     "${ZERO_SHOT_ONLY:+ zero-shot-only}${EXPID_SUFFIX:+  results-suffix=$EXPID_SUFFIX}"
for ((k = 1; k <= LANES; k++)); do
    log="logs/eval_worker_${HOST}_lane${k}.log"
    ( lane "$k" ) >> "$log" 2>&1 &
    disown
    echo "  lane $k -> $log"
done
echo
echo "progress: python tools/eval_status.py"
echo "claims:   ls $CLAIMS    (cat $CLAIMS/<config>/owner for who holds it)"
