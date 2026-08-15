#!/bin/bash
# status.sh — what is actually running on THIS node, and is any of it dead?
#
#   ./tmp_scripts/status.sh
#
# Answers three questions the campaign log cannot:
#   1. which trainings are alive right now (roots only, not DataLoader forks)
#   2. what each one is doing (last log line) and how long it has been at it
#   3. which launched runs are NOT alive, and why (traceback from their log)
#
# THROWAWAY - see tmp_scripts/README.md.
set -u
cd "$(dirname "$0")/.." || exit 1
ME="${USER:-$(id -un)}"
PAT="^[^ ]*python[^ ]* main\.py"

echo "=== $(hostname)  $(date '+%F %T') ==="
echo "repo:  $PWD"
echo "cores: $(python -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null || echo '?')"

# Memory cap. cgroup v2 limits are hierarchical and SLURM puts the job's cap
# on the job cgroup while the steps below read "max", so walk up and take the
# smallest. This is the number that decides how many runs fit - `free` reports
# the whole machine and has nothing to do with what the OOM killer enforces.
_rel=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup 2>/dev/null | head -1)
_cap=""; _capdir=""
if [ -n "${_rel:-}" ]; then
    _d="/sys/fs/cgroup${_rel}"; _d="${_d%/}"
    while : ; do
        if [ -r "$_d/memory.max" ]; then
            _v=$(cat "$_d/memory.max" 2>/dev/null)
            if [ "$_v" != "max" ] && [ -n "${_v:-}" ]; then
                if [ -z "$_cap" ] || [ "$_v" -lt "$_cap" ] 2>/dev/null; then
                    _cap="$_v"; _capdir="$_d"
                fi
            fi
        fi
        [ "$_d" = "/sys/fs/cgroup" ] && break
        _p=$(dirname "$_d")
        [ "$_p" = "$_d" ] && break                  # reached /, cannot ascend
        _d="$_p"
    done
fi
if [ -n "$_cap" ]; then
    _used=$(cat "$_capdir/memory.current" 2>/dev/null || echo 0)
    echo "mem:   cap $(( _cap / 1073741824 ))G, using $(( _used / 1073741824 ))G  [${_capdir#/sys/fs/cgroup}]"
    echo "       -> $(( _cap / 1073741824 / 20 )) runs at 20G each"
else
    echo "mem:   no cgroup cap found; host free: $(free -g 2>/dev/null | awk '/^Mem:/{print $7}')G"
fi
echo

# ---- alive: root processes only -------------------------------------------
# A DataLoader worker is a fork carrying its parent's cmdline, so counting
# every match overstates by num_workers per run. A run is a process whose
# parent is not itself a main.py process.
ALL=$(pgrep -u "$ME" -f "$PAT" 2>/dev/null)
ROOTS=""
for p in $ALL; do
    ppid=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    printf '%s\n' "$ALL" | grep -qx "${ppid:-0}" || ROOTS="$ROOTS $p"
done

echo "--- RUNNING ---"
if [ -z "${ROOTS# }" ]; then
    echo "  (nothing)"
else
    printf "  %-8s %-10s %-28s %-9s %s\n" PID ELAPSED CONFIG SEED MODE
    for p in $ROOTS; do
        args=$(ps -o args= -p "$p" 2>/dev/null)
        el=$(ps -o etime= -p "$p" 2>/dev/null | tr -d ' ')
        cfg=$(printf '%s' "$args" | grep -o 'configs/[^ ]*' | head -1)
        cfg=$(basename "${cfg:-?}" .yaml)
        seed=$(printf '%s' "$args" | sed -n 's/.*--seed \([0-9]*\).*/\1/p')
        mode=$(printf '%s' "$args" | grep -qe '--mode test' && echo test || echo train)
        dev=$(printf '%s' "$args" | grep -qe '--device cuda' && echo "" || echo " (cpu/eval)")
        printf "  %-8s %-10s %-28s %-9s %s%s\n" \
               "$p" "$el" "$cfg" "${seed:--}" "$mode" "$dev"
    done
fi
echo "  workers/forks not shown: $(printf '%s\n' $ALL | grep -c . 2>/dev/null) total matching processes"

# ---- what each live run is doing ------------------------------------------
echo
echo "--- LAST LOG LINE PER RUN ---"
shopt -s nullglob
LOGS=(logs/sweep_jc_*_run.log logs/all8_joint_chain_s*.log logs/sweep_jc_base_d*.log)
if [ ${#LOGS[@]} -eq 0 ]; then
    echo "  (no campaign run logs here)"
else
    for f in "${LOGS[@]}"; do
        line=$(grep -av '^\s*$' "$f" 2>/dev/null | tail -1 | cut -c1-110)
        age=$(( ( $(date +%s) - $(stat -c %Y "$f" 2>/dev/null || echo 0) ) / 60 ))
        printf "  %-38s [%3dm ago] %s\n" "$(basename "$f")" "$age" "${line:-<empty>}"
    done
fi

# ---- dead: launched but not running ---------------------------------------
echo
echo "--- DIED (log exists, process gone) ---"
DEAD=0
for f in "${LOGS[@]}"; do
    base=$(basename "$f" .log)
    # Is a live root using this log's config?
    cfg=$(echo "$base" | sed 's/_run$//; s/_s[0-9]*$//; s/_d[0-9]*$//')
    alive=0
    for p in $ROOTS; do
        ps -o args= -p "$p" 2>/dev/null | grep -q "configs/${cfg}.yaml" && alive=1
    done
    [ "$alive" -eq 1 ] && continue
    err=$(grep -a -m1 -A3 "Traceback\|Error:\|error:" "$f" 2>/dev/null | tail -3 | tr '\n' ' ' | cut -c1-150)
    if [ -n "$err" ]; then
        echo "  $base"
        echo "      $err"
        DEAD=1
    elif grep -aq "Results written to\|Training complete" "$f" 2>/dev/null; then
        echo "  $base  (finished normally)"
        DEAD=1
    else
        echo "  $base  (no traceback, no completion - check the tail yourself)"
        DEAD=1
    fi
done
[ "$DEAD" -eq 0 ] && echo "  (none)"

# ---- GPU -------------------------------------------------------------------
echo
echo "--- GPU ---"
GPU=$(nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used,memory.total \
                 --format=csv,noheader 2>/dev/null)
echo "  ${GPU:-(nvidia-smi unavailable)}"
echo
echo "--- CAMPAIGN DRIVER ---"
for n in m d; do
    [ -f "logs/campaign_${n}.log" ] || continue
    echo "  node $n: $(tail -1 "logs/campaign_${n}.log" | cut -c1-110)"
done
