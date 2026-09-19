#!/bin/bash
# queue_status.sh — one line per training run: progress, rate, time left, alive.
#
#   ./train_test_scripts/queue_status.sh
#
# Reads only files, so it works from the login node as well as a compute
# node - `pgrep` sees only local processes and reports nothing from submit-a,
# which looks identical to "everything died".
#
# ALIVE is decided by the log's mtime, not by the .running marker. A marker
# whose log stopped growing is a run that was killed without getting to clean
# up (SLURM revoking an interactive allocation does exactly this), and a
# marker alone would report it as healthy.
#
# Columns:
#   epoch/total   from the trainer's `Epoch X/Y` line
#   per10         seconds for the last 10 epochs; the epoch-0 sample is
#                 SKIPPED because it covers ~1 epoch plus startup and
#                 under-reports the rate by roughly 10x
#   left          (total-epoch)/10 * per10
#   state         RUN (log written within STALE_MIN), STALE, or done
set -u

cd "$(dirname "$0")/.." || exit 1
STALE_MIN="${STALE_MIN:-20}"
now=$(date +%s)

printf "%-34s %9s %10s %9s  %s\n" "run" "epoch" "per10" "left" "state"
for f in logs/*_mode_train_*.log; do
    [ -e "$f" ] || continue
    n=$(basename "$f" .log); n=${n%%_mode_train_*}

    line=$(grep -a '^Epoch ' "$f" | tail -1)
    if [ -z "$line" ]; then
        ep=0; tot="?"
    else
        rest=${line#Epoch }; ep=${rest%%/*}
        tot=${rest#*/}; tot=${tot%% *}
    fi

    # tail -n +2 drops the epoch-0 sample (see header).
    t=$(grep -a 'time(10)' "$f" | tail -n +2 | tail -1 | sed 's/.*time(10): //; s/s$//')

    left="?"
    if [ -n "$t" ] && [ "$tot" != "?" ]; then
        left=$(awk -v a="$tot" -v b="$ep" -v c="$t" \
               'BEGIN{printf "%.1fh", (a-b)/10*c/3600}')
    fi

    marker="logs/${n}_mode_train_${f#*_mode_train_}"; marker="${marker%.log}.running"
    age=$(( (now - $(stat -c %Y "$f" 2>/dev/null || echo "$now")) / 60 ))
    if [ -f "$marker" ]; then
        [ "$age" -le "$STALE_MIN" ] && state="RUN" || state="STALE (${age}m, killed?)"
    else
        state="done"
    fi

    printf "%-34s %4s/%-4s %9ss %9s  %s\n" "$n" "$ep" "$tot" "${t:-?}" "$left" "$state"
done

echo
for q in logs/q_*.log; do
    [ -e "$q" ] || continue
    d=$(grep -acE '^[a-z0-9_]+ +[0-9:]+ +(ok|FAIL)' "$q")
    tot=$(grep -aoE '^train_queue: [0-9]+' "$q" | head -1 | grep -oE '[0-9]+')
    printf "%-22s %s of %s configs finished\n" "$(basename "$q" .log)" "$d" "${tot:-?}"
done
