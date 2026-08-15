#!/bin/bash
# queue_runner.sh — keep a node as full as it can actually go, and no fuller.
#
#   ./tmp_scripts/queue_runner.sh jobs.txt
#   nohup ./tmp_scripts/queue_runner.sh jobs.txt > logs/queue_runner.log 2>&1 &
#
# Job file: one job per line, `#` comments and blank lines ignored.
#
#     <config-basename> <run-tag> [extra main.py flags...]
#
#     sweep_jc_drop01   run
#     sweep_jc_l2       run
#     all8_joint_chain  s14   --seed 14 --mode train
#
# Why this exists: run_campaign.sh launches a FIXED number of runs per phase,
# sized from the CORE count, and cores are almost never what binds. Three of
# four sweep arms died at "Epoch 0/1000" with no traceback on a 32-core node.
# Host RAM was not the cause (2 TB total, 1.8 TB available, ~16 GB per run) -
# the leading candidate is GPU memory on a card shared with other users, where
# 37 GB was already taken and four runs at ~12 GB each do not fit in 81 GB.
#
# So every launch is gated on THREE resources: a concurrency cap from cores,
# host memory (read from the cgroup limit where one exists, since that is what
# actually kills you), and free memory on the GPU this run will use. It also
# refills a slot the moment one frees rather than at phase boundaries, and
# retries a run that dies in its first five minutes - contention deaths
# usually succeed on a second attempt once the node is quieter.
#
# Env:
#   RAM_PER_RUN_GB=N   host-memory headroom required per launch (default 20)
#   GPU_PER_RUN_GB=N   GPU-memory headroom required per launch (default 15)
#   MAX_SLOTS=N        hard cap on concurrent runs (default: from cores)
#   POLL=N             seconds between checks (default 60)
#   RETRIES=N          re-attempts for a run that dies early (default 1)
#   DEVICE=cuda:0      passed through to run_config.sh
#   STAGGER=N          seconds between launches (default 60)
#   DRY_RUN=1          show the schedule decisions, launch nothing
#
# THROWAWAY - see tmp_scripts/README.md.
set -u
cd "$(dirname "$0")/.." || exit 1

[ "$#" -eq 1 ] || { echo "Usage: $0 <jobs-file>"; exit 1; }
JOBS_FILE="$1"
[ -f "$JOBS_FILE" ] || { echo "No such job file: $JOBS_FILE"; exit 1; }

RAM_PER_RUN_GB="${RAM_PER_RUN_GB:-20}"
GPU_PER_RUN_GB="${GPU_PER_RUN_GB:-15}"
POLL="${POLL:-60}"
RETRIES="${RETRIES:-1}"
DEVICE="${DEVICE:-cuda:0}"
DRY_RUN="${DRY_RUN:-0}"
# Gap between launches, so one run loads its dataset before the next competes
# for the same memory. This is what makes the RAM guard meaningful: without
# it, N launches all pass the check before any of them has allocated.
STAGGER="${STAGGER:-60}"
CORES=$(python -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null || echo 8)
# One main process plus its dataloaders; 5 cores per run matches _common.yaml's
# num_workers: 4. The memory guard below is what usually binds first.
MAX_SLOTS="${MAX_SLOTS:-$(( CORES / 5 ))}"
[ "$MAX_SLOTS" -lt 1 ] && MAX_SLOTS=1

say() { echo "[$(date '+%F %T')] $*"; }

avail_gb() {
    # The CGROUP limit is what kills you, not the machine's memory. On an HPC
    # node `free` reports the whole box (2 TB on dgxh-1) while the job may be
    # capped far below that, and a cgroup OOM kill leaves NO traceback - the
    # process just disappears. Prefer the cgroup accounting when present.
    local max cur v1max v1cur
    if [ -r /sys/fs/cgroup/memory.max ] && [ -r /sys/fs/cgroup/memory.current ]; then
        max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null)
        cur=$(cat /sys/fs/cgroup/memory.current 2>/dev/null)
        if [ "$max" != "max" ] && [ -n "${max:-}" ] && [ -n "${cur:-}" ]; then
            echo $(( (max - cur) / 1073741824 )); return
        fi
    elif [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
        v1max=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null)
        v1cur=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null)
        # An "unlimited" v1 limit is a huge sentinel, not a real cap.
        if [ -n "${v1max:-}" ] && [ "$v1max" -lt 1000000000000000 ] 2>/dev/null; then
            echo $(( (v1max - ${v1cur:-0}) / 1073741824 )); return
        fi
    fi
    free -g 2>/dev/null | awk '/^Mem:/ {print ($7 != "" ? $7 : $4)}' || echo 999
}

mem_source() {   # for the startup line, so the number is interpretable
    if [ -r /sys/fs/cgroup/memory.max ] && \
       [ "$(cat /sys/fs/cgroup/memory.max 2>/dev/null)" != "max" ]; then
        echo "cgroup v2 limit"
    elif [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ] && \
         [ "$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)" -lt 1000000000000000 ] 2>/dev/null; then
        echo "cgroup v1 limit"
    else
        echo "host free -g (no cgroup cap seen)"
    fi
}

gpu_free_gb() {
    # Free memory on the device this run will ACTUALLY use.
    #
    # DEVICE is a CUDA ordinal, and CUDA ordinals index into
    # CUDA_VISIBLE_DEVICES when it is set - while nvidia-smi indexes PHYSICAL
    # devices and ignores the mask entirely. Query without mapping and on a
    # shared 8-GPU node you cheerfully read somebody else's card.
    local dev="${DEVICE#cuda:}" phys used total
    case "$dev" in ''|*[!0-9]*) dev=0 ;; esac
    phys="$dev"
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        phys=$(printf '%s' "$CUDA_VISIBLE_DEVICES" | cut -d, -f$(( dev + 1 )))
    fi
    read -r used total <<< "$(nvidia-smi -i "${phys:-0}" \
        --query-gpu=memory.used,memory.total --format=csv,noheader,nounits \
        2>/dev/null | tr -d ',' )"
    [ -z "${total:-}" ] && { echo 999; return; }   # no nvidia-smi: do not block
    echo $(( (total - used) / 1024 ))
}

gpu_label() {
    local dev="${DEVICE#cuda:}" phys
    case "$dev" in ''|*[!0-9]*) dev=0 ;; esac
    phys="$dev"
    [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && \
        phys=$(printf '%s' "$CUDA_VISIBLE_DEVICES" | cut -d, -f$(( dev + 1 )))
    echo "$DEVICE -> physical GPU ${phys:-0}"
}

# ---- read the queue --------------------------------------------------------
declare -a Q_CFG Q_TAG Q_EXTRA
while read -r line; do
    line="${line%%#*}"
    [ -z "${line// /}" ] && continue
    set -- $line
    cfg="$1"; tag="${2:-run}"; shift 2 2>/dev/null || shift $#
    if [ ! -f "configs/${cfg}.yaml" ]; then
        say "SKIP: configs/${cfg}.yaml does not exist"
        continue
    fi
    Q_CFG+=("$cfg"); Q_TAG+=("$tag"); Q_EXTRA+=("$*")
done < "$JOBS_FILE"

TOTAL=${#Q_CFG[@]}
[ "$TOTAL" -eq 0 ] && { say "job file has no runnable jobs"; exit 1; }

say "queue: $TOTAL job(s) from $JOBS_FILE"
say "$CORES cores -> max $MAX_SLOTS concurrent; require ${RAM_PER_RUN_GB}G available per launch"
say "host memory available: $(avail_gb)G  [source: $(mem_source)]"
say "GPU: $(gpu_label), $(gpu_free_gb)G free; require ${GPU_PER_RUN_GB}G per launch"
_nvis=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
[ "${_nvis:-0}" -gt 1 ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && \
    say "NOTE: $_nvis GPUs on this host and no CUDA_VISIBLE_DEVICES set - every"
[ "${_nvis:-0}" -gt 1 ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && \
    say "      run will pile onto $DEVICE. Spread them with DEVICE=cuda:N per queue."

# ---- schedule --------------------------------------------------------------
# One array of "pid|name|queue-index|start-epoch" records. Parallel arrays and
# `set -u` do not mix well once any of them can go empty.
RUNS=()
PENDING=()          # queue indices waiting for a retry or for memory
ATTEMPTS=()
for ((i = 0; i < TOTAL; i++)); do ATTEMPTS[i]=0; done
next=0; done_ok=0; done_fail=0

reap() {            # prune finished runs; requeue anything that did not finish
    local keep=() rec pid name ix t0 elapsed last
    for rec in ${RUNS[@]+"${RUNS[@]}"}; do
        IFS='|' read -r pid name ix t0 <<< "$rec"
        if kill -0 "$pid" 2>/dev/null; then
            keep+=("$rec"); continue
        fi
        elapsed=$(( $(date +%s) - t0 ))

        # Classify on EVIDENCE, not on elapsed time. An earlier version called
        # anything past five minutes "finished", so three runs that died at
        # ~4-5 minutes were reported as successes and the queue moved on
        # cheerfully. A real run prints "Training complete in ..." when the
        # trainer exits and "Results written to ..." when a train_test run
        # writes its JSON; absent both, it did not finish, however long it ran.
        if grep -aq "Results written to\|Training complete in" "logs/${name}.log" 2>/dev/null; then
            say "finished after $(( elapsed / 60 ))m: $name"
            done_ok=$(( done_ok + 1 ))
            continue
        fi

        last=$(grep -av '^[[:space:]]*$' "logs/${name}.log" 2>/dev/null | tail -1 | cut -c1-120)
        # run_config.sh records the exit status; for a signal death it is the
        # only evidence there is, since nothing gets written to the log.
        code=$(cat "logs/${name}.exit" 2>/dev/null)
        case "${code:-}" in
            137) why="SIGKILL - OOM killer or the scheduler" ;;
            139) why="SIGSEGV" ;;
            134) why="abort - often a CUDA or driver failure" ;;
            "")  why="no exit status recorded (killed before it could be written)" ;;
            *)   why="exit $code" ;;
        esac
        say "DIED after $(( elapsed / 60 ))m without completing: $name"
        say "      cause: $why"
        say "      last log line: ${last:-<empty>}"
        if [ "${ATTEMPTS[ix]}" -le "$RETRIES" ]; then
            say "      requeueing (attempt $(( ATTEMPTS[ix] + 1 )) of $(( RETRIES + 1 )))"
            PENDING+=("$ix")
        else
            say "      retries exhausted - see logs/${name}.log"
            done_fail=$(( done_fail + 1 ))
        fi
    done
    RUNS=(${keep[@]+"${keep[@]}"})
}

launch() {          # queue-index -> 0 on success
    local ix="$1" cfg="${Q_CFG[ix]}" tag="${Q_TAG[ix]}" extra="${Q_EXTRA[ix]}"
    ATTEMPTS[ix]=$(( ATTEMPTS[ix] + 1 ))
    if [ "$DRY_RUN" = "1" ]; then
        say "would launch $cfg (tag=$tag) ${extra:-<no extra flags>}"
        RUNS+=("0|${cfg}_${tag}|$ix|$(date +%s)")
        return 0
    fi
    local out pid
    out=$(RUN_TAG="$tag" ./train_test_scripts/run_config.sh \
              "configs/${cfg}.yaml" "$DEVICE" --wandb True $extra 2>&1)
    echo "$out" | sed 's/^/    /'
    # run_config.sh prints "Started <name> on <dev> (pid <N>)". Taking the pid
    # from there beats pattern-matching ps, which cannot tell two runs of the
    # same config apart and counts DataLoader forks as runs.
    pid=$(printf '%s' "$out" | sed -n 's/.*(pid \([0-9]*\)).*/\1/p' | head -1)
    if [ -z "$pid" ]; then
        say "launch of $cfg produced no pid - treating as failed"
        return 1
    fi
    RUNS+=("$pid|${cfg}_${tag}|$ix|$(date +%s)")
    say "launched $cfg (tag=$tag, pid=$pid); host ${av:-?}G, GPU ${gv:-?}G free at launch"
    return 0
}

while :; do
    reap
    n_live=${#RUNS[@]}
    n_pending=${#PENDING[@]}
    if [ "$n_live" -eq 0 ] && [ "$n_pending" -eq 0 ] && [ "$next" -ge "$TOTAL" ]; then
        say "queue drained: $done_ok finished, $done_fail failed"
        break
    fi

    while [ "$n_live" -lt "$MAX_SLOTS" ]; do
        ix=""
        if [ "${#PENDING[@]}" -gt 0 ]; then          # retries first
            ix="${PENDING[0]}"; PENDING=(${PENDING[@]:1})
        elif [ "$next" -lt "$TOTAL" ]; then
            ix="$next"; next=$(( next + 1 ))
        else
            break
        fi

        # Both resources, because either one alone lets you overcommit the
        # other. On a shared node the GPU is usually the tighter of the two,
        # and other users' jobs move it under you between launches.
        av=$(avail_gb); gv=$(gpu_free_gb)
        if [ "$DRY_RUN" != "1" ] && \
           { [ "$av" -lt "$RAM_PER_RUN_GB" ] || [ "$gv" -lt "$GPU_PER_RUN_GB" ]; }; then
            if [ "$av" -lt "$RAM_PER_RUN_GB" ]; then
                say "holding ${Q_CFG[ix]}: host ${av}G < ${RAM_PER_RUN_GB}G needed"
            else
                say "holding ${Q_CFG[ix]}: GPU ${gv}G free < ${GPU_PER_RUN_GB}G needed (shared card?)"
            fi
            PENDING=("$ix" ${PENDING[@]+"${PENDING[@]}"})
            break
        fi
        launch "$ix" || done_fail=$(( done_fail + 1 ))
        n_live=$(( n_live + 1 ))
        [ "$DRY_RUN" = "1" ] || sleep "$STAGGER"
    done

    [ "$DRY_RUN" = "1" ] && { say "dry run: would run ${#RUNS[@]} concurrently, $(( TOTAL - next )) queued behind"; break; }
    sleep "$POLL"
done
