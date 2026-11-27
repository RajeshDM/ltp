#!/bin/bash
# queue_status.sh — one line per training run: progress, rate, time left, alive.
#
#   ./train_test_scripts/queue_status.sh
#
# Finds runs by CONTENT, not by filename: any log with an `Epoch X/Y` line is
# a training run, any log with a `train_queue:` header is a queue. So it works
# whatever the configs, flags or queue names are - log names are derived from
# the flags (run_config.sh), so a name-matching version would silently miss
# every run launched with a different flag set.
#
# Reads only files, so it works from the login node as well as a compute node
# - `pgrep` sees only local processes and reports nothing from submit-a, which
# looks identical to "everything died".
#
# ALIVE is decided by the log's mtime, not by the .running marker. A marker
# whose log stopped growing is a run that was killed without getting to clean
# up (SLURM revoking an interactive allocation does exactly this), and the
# marker alone would report it as healthy.
#
# Env:
#   LOGS=<glob>      which logs to consider (default logs/*.log)
#   MAX_AGE_DAYS=N   ignore logs untouched for longer (default 3, 0 = no limit)
#   STALE_MIN=N      minutes without a write before a live marker reads STALE
#                    (default 20; raise it if an epoch takes longer than that)
set -u

cd "$(dirname "$0")/.." || exit 1
STALE_MIN="${STALE_MIN:-20}"
MAX_AGE_DAYS="${MAX_AGE_DAYS:-3}"
LOGS="${LOGS:-logs/*.log}"
now=$(date +%s)

runs=""; queues=""
for f in $LOGS; do
    [ -e "$f" ] || continue
    if [ "$MAX_AGE_DAYS" -gt 0 ]; then
        age_d=$(( (now - $(stat -c %Y "$f" 2>/dev/null || echo 0)) / 86400 ))
        [ "$age_d" -ge "$MAX_AGE_DAYS" ] && continue
    fi
    if grep -aqm1 '^train_queue: ' "$f"; then queues="$queues $f"
    elif grep -aqm1 '^Epoch [0-9]' "$f"; then runs="$runs $f"
    fi
done

if [ -z "$runs$queues" ]; then
    echo "no training logs in $LOGS modified in the last ${MAX_AGE_DAYS}d"
    echo "  (LOGS=<glob> to widen, MAX_AGE_DAYS=0 for no limit)"
    exit 0
fi

[ -n "$runs" ] && printf "%-34s %9s %10s %9s  %s\n" "run" "epoch" "per-iv" "left" "state"
for f in $runs; do
    n=$(basename "$f" .log); n=${n%%_mode_*}

    rest=$(grep -a '^Epoch ' "$f" | tail -1); rest=${rest#Epoch }
    ep=${rest%%/*}; tot=${rest#*/}; tot=${tot%% *}

    # The trainer prints `time(N): <secs>` every N epochs. N is read from the
    # line rather than assumed to be 10, and the FIRST sample is dropped: at
    # epoch 0 it covers roughly one epoch plus startup, which under-reports
    # the rate by about N times.
    last=$(grep -aoE 'time\([0-9]+\): [0-9.]+' "$f" | tail -n +2 | tail -1)
    iv=$(printf '%s' "$last" | sed -n 's/time(\([0-9]*\)).*/\1/p')
    t=$(printf '%s' "$last" | sed -n 's/.*: //p')

    left="?"
    if [ -n "$t" ] && [ -n "$iv" ] && [ "$iv" -gt 0 ] 2>/dev/null; then
        left=$(awk -v a="$tot" -v b="$ep" -v c="$t" -v i="$iv" \
               'BEGIN{printf "%.1fh", (a-b)/i*c/3600}')
    fi

    age=$(( (now - $(stat -c %Y "$f" 2>/dev/null || echo "$now")) / 60 ))
    if [ -f "${f%.log}.running" ]; then
        [ "$age" -le "$STALE_MIN" ] && state="RUN" || state="STALE (${age}m, killed?)"
    else
        state="done"
    fi

    printf "%-34s %4s/%-4s %9ss %9s  %s\n" "$n" "$ep" "$tot" "${t:-?}" "$left" "$state"
done

[ -n "$queues" ] && echo
for q in $queues; do
    d=$(grep -acE '^[A-Za-z0-9_.-]+ +[0-9:]+ +(ok|FAIL|NO START|LAUNCH)' "$q")
    tot=$(grep -aoE '^train_queue: [0-9]+' "$q" | head -1 | grep -oE '[0-9]+')
    printf "%-26s %s of %s configs finished\n" "$(basename "$q" .log)" "$d" "${tot:-?}"
done
