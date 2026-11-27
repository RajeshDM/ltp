# mps.sh — start/stop a per-user CUDA MPS daemon so concurrent runs actually
# share the GPU instead of taking turns.
#
#   source ./tmp_scripts/mps.sh start     # MUST be sourced: it exports vars
#   ./tmp_scripts/mps.sh status
#   ./tmp_scripts/mps.sh stop
#
# Why: each process gets its own CUDA context, and the GPU can only have one
# context ACTIVE at a time, so N processes time-slice - none of them overlap.
# Measured here: 3 concurrent trainings ran at ~1/3 speed each for no net
# gain, with the card at 99% "utilization" but only 21% of rated power. That
# combination (busy all the time, barely drawing power) is the signature of
# many small kernels that leave most of the SMs idle.
#
# MPS is a daemon that funnels every client's work through ONE shared context,
# so kernels from different processes can run side by side on different SMs.
# It changes scheduling only - not arithmetic - so results stay bit-comparable
# with non-MPS runs. That is what makes it usable mid-project, unlike changing
# batch size or enabling AMP.
#
# It does NOT raise how many runs fit: that is still bounded by host memory
# (the cgroup cap) and GPU memory. It makes the runs you already fit go faster.
#
# THROWAWAY - see tmp_scripts/README.md.

_mps_dirs() {
    export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps-$(id -un)}"
    export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log-$(id -un)}"
}

_mps_running() {
    _mps_dirs
    [ -S "$CUDA_MPS_PIPE_DIRECTORY/control" ] && \
        pgrep -u "$(id -un)" -x nvidia-cuda-mps-control >/dev/null 2>&1
}

mps_status() {
    _mps_dirs
    local mode
    mode=$(nvidia-smi -q 2>/dev/null | awk -F: '/Compute Mode/{gsub(/ /,"",$2); print $2; exit}')
    echo "compute mode : ${mode:-unknown}"
    echo "pipe dir     : $CUDA_MPS_PIPE_DIRECTORY"
    if _mps_running; then
        echo "daemon       : RUNNING (pid $(pgrep -u "$(id -un)" -x nvidia-cuda-mps-control | head -1))"
        echo "clients      : $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l) process(es) on the GPU"
    else
        echo "daemon       : not running"
    fi
    # Only processes that had these variables set at launch use the daemon;
    # anything started earlier keeps its own context and still time-slices.
    if [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ] && _mps_running; then
        echo "note         : only processes LAUNCHED after this point route"
        echo "               through MPS - already-running jobs do not."
    fi
}

mps_start() {
    _mps_dirs
    local mode
    mode=$(nvidia-smi -q 2>/dev/null | awk -F: '/Compute Mode/{gsub(/ /,"",$2); print $2; exit}')
    if [ "$mode" = "Exclusive_Process" ]; then
        echo "Compute mode is Exclusive_Process: only one context is permitted,"
        echo "so a user-started MPS daemon will not help. Ask the admins."
        return 1
    fi
    if _mps_running; then
        echo "MPS already running for $(id -un) at $CUDA_MPS_PIPE_DIRECTORY"
        return 0
    fi
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" || return 1
    if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
        echo "nvidia-cuda-mps-control not on PATH - MPS is not installed here."
        return 1
    fi
    nvidia-cuda-mps-control -d || { echo "failed to start MPS daemon"; return 1; }
    sleep 1
    if _mps_running; then
        echo "MPS started. Launch runs FROM THIS SHELL so they inherit:"
        echo "  CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
        return 0
    fi
    echo "daemon did not come up; see $CUDA_MPS_LOG_DIRECTORY/control.log"
    return 1
}

mps_stop() {
    _mps_dirs
    _mps_running || { echo "MPS not running"; return 0; }
    echo quit | nvidia-cuda-mps-control
    sleep 1
    _mps_running && echo "still running - kill it by pid" || echo "MPS stopped"
}

# Sourced or executed? `start` has to be sourced, because its whole job is to
# put the pipe directory into the environment that later launches inherit.
_MPS_SOURCED=0
[ "${BASH_SOURCE[0]}" != "${0}" ] && _MPS_SOURCED=1

case "${1:-status}" in
    start)
        if [ "$_MPS_SOURCED" -eq 0 ]; then
            echo "Run this as:  source $0 start"
            echo "(executing it would set the variables in a subshell that"
            echo " exits immediately, and your runs would not use MPS)"
            exit 1
        fi
        mps_start
        ;;
    stop)   mps_stop ;;
    status) mps_status ;;
    *)      echo "Usage: source $0 start | $0 status | $0 stop" ;;
esac
