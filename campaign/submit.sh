#!/bin/bash
# Submit campaign workers as SLURM batch jobs (they survive a lost ssh session).
#   campaign/submit.sh <preset> [count] [--dry-run]
#   campaign/submit.sh --list
#
# Every preset is sized so it fits the cluster's per-user resource-time caps
# (CLUSTER.md section 2): the caps are on resource x TIME LIMIT summed over
# running jobs, so the lever for more concurrent GPUs is a SHORTER time limit,
# not fewer jobs. MAX is the number that fits at that time limit.
# A worker is a pull loop (campaign/worker.py): it takes whatever job is ready,
# so submitting the same preset twice is safe and never duplicates work.
set -eu
cd "$(dirname "$0")/.."
REPO=$PWD
ACCOUNT=${CAMPAIGN_ACCOUNT:-eecs}

#        preset        partition  gres   cpus mem   time        kind  max  extra
PRESETS="
gpu-dgxh      dgxh     gpu:1   8  64G   1-00:00:00  gpu   2   -
gpu-dgx2      dgx2     gpu:1   4  32G   2-00:00:00  gpu   4   -
gpu-gpu       gpu      gpu:1   4  32G   2-00:00:00  gpu   4   -
gpu-share     share    gpu:1   8  64G   1-00:00:00  gpu   2   -
big-share     share    gpu:1   32 128G  2-00:00:00  cpu   1   --constraint=a40
cpu-share     share    -       32 64G   2-00:00:00  cpu   8   -
preempt-a40   preempt  gpu:1   32 64G   2-00:00:00  cpu   2   --constraint=a40,--requeue
"

if [ "${1:-}" = "--list" ] || [ -z "${1:-}" ]; then
    echo "preset        partition gres   cpus mem  time        prefers max-concurrent"
    printf '%s\n' "$PRESETS" | grep -v '^$'
    echo; echo "kind: gpu = a training worker (takes GPU jobs first); cpu = an evaluation worker (takes CPU jobs first)"
    exit 0
fi

PRESET=$1; COUNT=1; DRY=0; shift
for a in "$@"; do
    case "$a" in --dry-run) DRY=1 ;; ''|*[!0-9]*) echo "bad argument: $a"; exit 1 ;; *) COUNT=$a ;; esac
done
LINE=$(printf '%s\n' "$PRESETS" | awk -v p="$PRESET" '$1==p')
[ -n "$LINE" ] || { echo "unknown preset '$PRESET'; campaign/submit.sh --list"; exit 1; }
read -r _ PART GRES CPUS MEM TIME KIND MAX EXTRA <<< "$LINE"

if [ -e campaign/STOP ]; then echo "REFUSING: campaign/STOP exists"; exit 1; fi
OPTS="-A $ACCOUNT -p $PART -c $CPUS --mem=$MEM -t $TIME -J camp-$PRESET -o $REPO/logs/campaign_slurm_%j.out"
[ "$GRES" != "-" ] && OPTS="$OPTS --gres=$GRES"
[ "$EXTRA" != "-" ] && OPTS="$OPTS $(printf '%s' "$EXTRA" | tr ',' ' ')"

if [ "$DRY" = 0 ]; then
    command -v sbatch >/dev/null || { echo "sbatch not found (module load slurm)"; exit 1; }
    mine=$(squeue -u "$USER" -h -n "camp-$PRESET" 2>/dev/null | wc -l)
    all=$(squeue -u "$USER" -h -n "camp-gpu-dgxh,camp-gpu-dgx2,camp-gpu-gpu,camp-gpu-share,camp-big-share,camp-cpu-share,camp-preempt-a40" 2>/dev/null | wc -l)
    if [ $((mine + COUNT)) -gt "$MAX" ]; then
        echo "REFUSING: $mine already queued/running + $COUNT > $MAX, what $PRESET's quota allows"; exit 1
    fi
    if [ $((all + COUNT)) -gt "${CAMPAIGN_MAX_WORKERS:-12}" ]; then
        echo "REFUSING: $((all + COUNT)) workers > CAMPAIGN_MAX_WORKERS=${CAMPAIGN_MAX_WORKERS:-12}"; exit 1
    fi
fi
mkdir -p logs
for _ in $(seq "$COUNT"); do
    CMD="sbatch --parsable $OPTS --wrap \"bash $REPO/campaign/worker_entry.sh $KIND\""
    if [ "$DRY" = 1 ]; then echo "$CMD"; else
        id=$(eval "$CMD"); echo "submitted $PRESET -> job $id"
    fi
done
