#!/bin/bash
# Submit campaign workers as SLURM batch jobs (they survive a lost ssh session).
#   campaign/submit.sh <preset> [count] [--dry-run]
#   CAMPAIGN_EXCLUDE=<node,node> campaign/submit.sh ...   # avoid broken nodes
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
# The time limit in hours, handed to the worker so its "can this job finish
# before the allocation ends" guard works even where squeue is not on PATH.
LIMIT_H=$(printf '%s' "$TIME" | awk -F'[-:]' '{ if (NF==4) print $1*24+$2+$3/60; else if (NF==3) print $1+$2/60; else print 1 }')
# One bash array, no eval, and a --wrap string with NO spaces: this cluster's
# sbatch refused `--wrap "bash script.sh cpu"` with "Script arguments not
# permitted with --wrap option", i.e. it re-splits the wrapped command. The
# worker kind therefore travels in the environment, not as an argument.
ARGS=(--parsable -A "$ACCOUNT" -p "$PART" -c "$CPUS" --mem="$MEM" -t "$TIME"
      -J "camp-$PRESET" -o "$REPO/logs/campaign_slurm_%j.out"
      --export="ALL,CAMPAIGN_TIME_LIMIT_H=$LIMIT_H,CAMPAIGN_KIND=$KIND")
[ "$GRES" != "-" ] && ARGS+=(--gres="$GRES")
if [ "$EXTRA" != "-" ]; then
    IFS=, read -r -a EXTRA_ARGS <<< "$EXTRA"
    ARGS+=("${EXTRA_ARGS[@]}")
fi
# CAMPAIGN_EXCLUDE=cn-e41,cn-e42: nodes to keep off. For a node that kills
# jobs at launch ("CANCELLED by 0" one second after start, no output file):
# that is the node's own prolog failing, and the scheduler will keep handing
# the same idle node to the next job.
[ -n "${CAMPAIGN_EXCLUDE:-}" ] && ARGS+=(--exclude="$CAMPAIGN_EXCLUDE")
ARGS+=(--wrap "$REPO/campaign/worker_entry.sh")

if [ "$DRY" = 0 ]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        # a script is not a login shell: try the module system before giving up
        [ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh
        command -v module >/dev/null 2>&1 && module load slurm 2>/dev/null
    fi
    command -v sbatch >/dev/null 2>&1 || { echo "sbatch not found: run 'module load slurm' in this shell, then retry"; exit 1; }
    ME=${USER:-$(id -un)}
    mine=$(squeue -u "$ME" -h -n "camp-$PRESET" 2>/dev/null | wc -l)
    all=$(squeue -u "$ME" -h -n "camp-gpu-dgxh,camp-gpu-dgx2,camp-gpu-gpu,camp-gpu-share,camp-big-share,camp-cpu-share,camp-preempt-a40" 2>/dev/null | wc -l)
    if [ $((mine + COUNT)) -gt "$MAX" ]; then
        echo "REFUSING: $mine already queued/running + $COUNT > $MAX, what $PRESET's quota allows"; exit 1
    fi
    if [ $((all + COUNT)) -gt "${CAMPAIGN_MAX_WORKERS:-12}" ]; then
        echo "REFUSING: $((all + COUNT)) workers > CAMPAIGN_MAX_WORKERS=${CAMPAIGN_MAX_WORKERS:-12}"; exit 1
    fi
fi
mkdir -p logs
for _ in $(seq "$COUNT"); do
    if [ "$DRY" = 1 ]; then echo "sbatch ${ARGS[*]}"; else
        # --export=ALL would hand the agent's login secret to every batch job
        id=$(env -u CLAUDE_CODE_OAUTH_TOKEN -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN sbatch "${ARGS[@]}")
        echo "submitted $PRESET -> job $id"
    fi
done
