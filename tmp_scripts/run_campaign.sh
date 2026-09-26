#!/bin/bash
# run_campaign.sh — fire-and-forget driver for the two-day, two-H100 campaign.
#
#   ./tmp_scripts/run_campaign.sh m      # on the mangannr node
#   ./tmp_scripts/run_campaign.sh d      # on the dugarp node
#
# Detaches itself (setsid + nohup), so it survives the tmux session, the ssh
# connection and the laptop closing. It prints a PID and returns immediately.
# Everything after that is visible on wandb; nothing needs to be run by hand.
#
# THROWAWAY. Lives in tmp_scripts/ on purpose - see tmp_scripts/README.md.
# What it is doing and why: RUNBOOK.md, "The two-H100 campaign".
#
# Phases run in order of value, so if two days is not enough the tail is what
# gets cut, not the middle. Node M ends on the data-need arms; node D ends on
# per-seed evaluation. Neither node ever waits on the other.
#
# Env:
#   DRY_RUN=1     print the plan and exit, run nothing (check this first)
#   WORKERS=N     override the eval workers used while trainings are running
#                 (default: derived from the core count, see budget.sh)
#   CORES=N       pretend the allocation has N cores (for DRY_RUN inspection)

set -u

usage() { echo "Usage: $0 <m|d>   (m = mangannr node, d = dugarp node)"; exit 1; }
[ "$#" -eq 1 ] || usage
NODE="$1"
case "$NODE" in m|d) ;; *) usage ;; esac

cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"
DRY_RUN="${DRY_RUN:-0}"

# Concurrency is DERIVED from the cores this allocation actually has, not
# assumed to be 32 - the same node has shown up with 2, 16 and 32.
. tmp_scripts/budget.sh
# CORES is overridable so a DRY_RUN can show the plan for the node you are
# about to launch on, from wherever you happen to be sitting.
CORES="${CORES:-$(python -c "import os; print(len(os.sched_getaffinity(0)))" 2>/dev/null || echo 16)}"
compute_budget "$CORES"
WORKERS="${WORKERS:-$EVAL_BUSY}"
if [ "$CORES" -lt 8 ]; then
    echo "Only $CORES cores in this allocation - not enough to run the"
    echo "campaign (a single training run wants ~3). Get a bigger allocation."
    exit 1
fi
# $USER is not guaranteed to be exported - notably not under setsid, which is
# how this script re-execs itself - and `set -u` turns that into an instant
# death right after detaching, where nobody would see it.
ME="${USER:-$(id -un)}"

# ---------------------------------------------------------------------------
# Preflight. Everything that can be checked cheaply is checked BEFORE
# detaching, because the whole point is that nobody is watching afterwards -
# a run that dies in minute one because of the wrong conda env would waste
# both days silently.
# ---------------------------------------------------------------------------
if [ -z "${CAMPAIGN_CHILD:-}" ]; then
    # Preflight lives in ONE place (tmp_scripts/preflight.sh) and is the real
    # end-to-end check: a 4-epoch run whose coverage numbers are read back off
    # the wandb server. Re-implementing a weaker version of it here is how the
    # two drift apart and the campaign launches on a node that cannot report.
    # 1. Did the full verification ever pass here?
    if [ ! -f ".preflight_ok" ]; then
        echo "No .preflight_ok in $REPO."
        echo
        echo "Run the verification first (~10 min, and it is the thing that"
        echo "catches a broken wandb login before it costs you two days):"
        echo
        echo "    ./tmp_scripts/preflight.sh"
        echo
        echo "SKIP_PREFLIGHT=1 overrides, at your own risk."
        [ "${SKIP_PREFLIGHT:-0}" = "1" ] || exit 1
    else
        echo "preflight passed $(date -r .preflight_ok '+%F %T')"
    fi

    # 2. Can THIS shell, right now, run the code? .preflight_ok says the
    # environment worked in SOME shell at SOME time; it cannot say whether
    # this one has the conda env active. That gap cost a full day: five runs
    # launched into a shell with no torch, each dying instantly, while the
    # driver walked through every phase finding nothing. Two seconds here.
    if [ "$DRY_RUN" != "1" ] && ! python -c "import torch, pddlgym, wandb" 2>/dev/null; then
        echo
        echo "This shell cannot import torch/pddlgym/wandb."
        echo "  python: $(command -v python || echo 'not found')"
        _want=$(sed -n 's/^python=//p' .preflight_ok 2>/dev/null)
        [ -n "$_want" ] && echo "  preflight used: $_want"
        echo
        echo "  conda activate <di_ltp_1 | ltp_3>   then relaunch."
        exit 1
    fi
    # Same env, different interpreter, is worth a warning but not a refusal.
    _want=$(sed -n 's/^python=//p' .preflight_ok 2>/dev/null)
    _have=$(command -v python)
    if [ -n "$_want" ] && [ "$_want" != "$_have" ]; then
        echo "  WARN  python differs from the preflight run"
        echo "        preflight: $_want"
        echo "        this shell: $_have"
    fi

    RUNNING=$(pgrep -u "$ME" -f "main.py" | wc -l)
    [ "$RUNNING" -gt 0 ] && echo "  WARN  $RUNNING main.py processes already running here"

    if [ "$DRY_RUN" = "1" ]; then
        echo; echo "=== DRY_RUN: plan for node $NODE ==="
        CAMPAIGN_CHILD=1 DRY_RUN=1 bash "$0" "$NODE"
        exit 0
    fi

    mkdir -p logs
    echo
    echo "=== detaching ==="
    CAMPAIGN_CHILD=1 setsid nohup bash "$0" "$NODE" \
        > "logs/campaign_${NODE}.log" 2>&1 &
    echo "campaign node $NODE started (pid $!)"
    echo "  local log:  tail -f logs/campaign_${NODE}.log"
    echo "  online:     https://wandb.ai/<entity>/ltp_gnn_gru_pyg  (group node_${NODE})"
    echo
    echo "Safe to close this session now."
    exit 0
fi

# ---------------------------------------------------------------------------
# Detached from here down.
# ---------------------------------------------------------------------------
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42
export GABAR_WANDB_GROUP="node_${NODE}"

say() { echo "[$(date '+%F %T')] $*"; }

# All training goes through run_config.sh (nohup + log rotation + RUN_TAG);
# we only need the pid to wait on, and run_config.sh backgrounds internally,
# so track pids by polling for the config name instead of $!.
train() {   # config-basename  run-tag  [extra flags...]
    local cfg="$1" tag="$2"; shift 2
    if [ "$DRY_RUN" = "1" ]; then
        echo "  train  $cfg  tag=$tag  $*"
        return
    fi
    # Launcher output goes to the campaign log, NOT /dev/null: run_config.sh
    # reports a missing config or a failed start on stdout, and discarding it
    # turned a hard failure into a silent one.
    RUN_TAG="$tag" ./train_test_scripts/run_config.sh "configs/${cfg}.yaml" \
        cuda:0 --wandb True --num-workers "$DL_WORKERS" "$@" \
        2>&1 | sed 's/^/    /'
}

verify_started() {   # expected-count - abort loudly if the launches did not take
    [ "$DRY_RUN" = "1" ] && { echo "  check  that $1 trainings are alive"; return; }
    sleep 45                       # past import, argument parsing and config load
    local n; n=$(n_trainings)
    if [ "$n" -lt 1 ]; then
        say "FATAL: launched $1 training(s); none are running 45s later."
        say "       Something is killing them at startup - read the newest"
        say "       logs/*_run.log or logs/*_s*.log for the traceback."
        say "       Stopping instead of walking through every phase finding"
        say "       nothing, which is what wasted the previous attempt."
        exit 1
    fi
    [ "$n" -lt "$1" ] && say "WARN: only $n of $1 trainings are running"
    say "$n training(s) confirmed running"
}

stagger() {   # let one run get past its cache load before the next starts
    [ "$DRY_RUN" = "1" ] || sleep 60
}

wait_for_trainings() {
    [ "$DRY_RUN" = "1" ] && { echo "  wait   until all trainings finish"; return; }
    say "waiting for trainings to finish"
    while [ "$(n_trainings)" -gt 0 ]; do
        sleep 300
    done
    say "all trainings finished"
}

# Number of training runs currently alive.
#
# Neither obvious counting method works. Distinct CONFIG PATHS undercounts:
# node D runs one config twice with different --seed, so two healthy runs
# report as 1. Raw PIDS overcounts: DataLoader workers are forks carrying the
# identical cmdline, so one run with 4 workers reports as 5.
#
# What identifies a run is being a ROOT: a main.py process whose parent is not
# itself a main.py process. run_config.sh backgrounds under nohup and exits,
# so a real run is reparented to init (ppid 1); a DataLoader worker's parent
# is its own trainer, which is in the set.
n_trainings() {
    local pids n=0 p ppid
    # Anchored at the interpreter so a shell whose command line merely
    # CONTAINS this pattern (a grep, this script quoted in a wrapper) is not
    # counted as a training run.
    pids=$(pgrep -u "$ME" -f "^[^ ]*python[^ ]* main\.py .*--device cuda" 2>/dev/null)
    [ -z "$pids" ] && { echo 0; return; }
    for p in $pids; do
        ppid=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
        printf '%s\n' "$pids" | grep -qx "${ppid:-0}" || n=$((n + 1))
    done
    echo "$n"
}

wait_until_below() {   # target
    [ "$DRY_RUN" = "1" ] && { echo "  wait   until fewer than $1 trainings"; return; }
    while [ "$(n_trainings)" -ge "$1" ]; do sleep 300; done
}

evalq() {   # tag  extra-flags  config...
    local tag="$1" extra="$2"; shift 2
    if [ "$DRY_RUN" = "1" ]; then
        echo "  eval   [$tag] $* ${extra:+($extra)}"
        return
    fi
    WANDB=1 WORKERS="$WORKERS" TAG="$tag" EXTRA="$extra" \
        ./train_test_scripts/eval_queue.sh "$@"
}

say "campaign node $NODE starting in $REPO"
say "wandb group $GABAR_WANDB_GROUP"
say "$CORES cores -> up to $SLOTS trainings x $CORES_PER_TRAINING cores; \
eval lane sizes itself to whatever is left, $EVAL_IDLE once training is done"

if [ "$NODE" = "m" ]; then
    # ---- Node M: the sweep, then the data-need arms --------------------
    say "PHASE 1  sweep arms, $SLOTS at a time (the rest queue behind them)"
    ARMS=(sweep_jc_base sweep_jc_drop01 sweep_jc_drop02 sweep_jc_l2 sweep_jc_heads4)
    FIRST=("${ARMS[@]:0:$SLOTS}")
    for cfg in "${FIRST[@]}"; do
        train "$cfg" "run"
        stagger           # simultaneous cache loads thrash the filesystem
    done
    verify_started "${#FIRST[@]}"

    say "PHASE 2  remaining arms as slots free"
    for cfg in "${ARMS[@]:$SLOTS}"; do
        wait_until_below "$SLOTS"
        train "$cfg" "run"
        stagger
    done

    wait_for_trainings

    say "PHASE 3  evaluate all five arms ($EVAL_IDLE workers, GPU now idle)"
    WORKERS=$EVAL_IDLE evalq "sweep" "" \
        configs/sweep_jc_base.yaml configs/sweep_jc_drop01.yaml \
        configs/sweep_jc_drop02.yaml configs/sweep_jc_l2.yaml \
        configs/sweep_jc_heads4.yaml

    # Deliberately on the BASE arm, not on whichever arm won: "how much data
    # does this need" is a question about data, and pinning it to the
    # unregularized reference keeps it comparable to every published number.
    # Picking the winner instead would need a human in the loop, which is the
    # one thing this script cannot have.
    say "PHASE 4  data-need arms on the base config"
    for n in 25 50 100; do
        train sweep_jc_base "d$n" --num-train-problems "$n"
        stagger
    done
    verify_started 3
    wait_for_trainings

    say "PHASE 5  evaluate the data-need arms"
    for n in 25 50 100; do
        WORKERS=$EVAL_IDLE evalq "d$n" "--num-train-problems $n" configs/sweep_jc_base.yaml
    done

else
    # ---- Node D: evaluate what exists, replicate seeds ------------------
    # Lane B first so the trainings are already occupying the GPU while the
    # evaluation lane (CPU) runs alongside them; the reverse order would
    # leave the GPU idle for the whole first evaluation.
    # Two, not three: seed 10 already exists on this filesystem, so 10/12/13
    # is the three seeds the paper and appendix claim. The third replicate
    # would cost a third of a 16-core node for a fourth point nobody needs.
    N_SEEDS=2
    say "PHASE 1  $N_SEEDS seed replicates ($((N_SEEDS * CORES_PER_TRAINING)) \
cores; seed 10 already exists, so this makes three)"
    for s in 12 13; do
        train all8_joint_chain "s$s" --seed "$s" --mode train
        stagger
    done
    verify_started "$N_SEEDS"

    say "PHASE 2  evaluate the existing all8 checkpoints alongside them"
    # Configs this filesystem never trained report NO MODELS and are skipped,
    # so this doubles as an inventory of what node D actually holds.
    _lane=$(eval_workers_for "$N_SEEDS")
    say "         eval lane: $_lane workers alongside $N_SEEDS trainings"
    WORKERS=$_lane \
    # The paper's three rungs only: UNION, GADAR-BIND (joint_lite) and GADAR
    # (joint_chain) are Table 2's columns. Plain `joint` and `structural` are
    # internal ablation rungs that appear nowhere in the paper, and if their
    # checkpoints DO exist here they would spend hours of the eval lane on
    # columns nobody reads. Add them back deliberately if the ladder ever
    # needs filling out.
    evalq "all8" "" \
        configs/all8_union.yaml configs/all8_joint_lite.yaml \
        configs/all8_joint_chain.yaml

    wait_for_trainings

    say "PHASE 3  evaluate each replicate ($EVAL_IDLE workers, GPU now idle)"
    for s in 12 13; do
        WORKERS=$EVAL_IDLE evalq "s$s" "--seed $s" configs/all8_joint_chain.yaml
    done
fi

say "PHASE FINAL  aggregate"
if [ "$DRY_RUN" != "1" ]; then
    python tools/analyze_results.py --csv "campaign_node_${NODE}.csv" \
        2>&1 | tail -40
    say "wrote campaign_node_${NODE}.csv"
fi
say "campaign node $NODE done"
