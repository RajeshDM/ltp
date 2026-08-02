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
    if [ ! -f ".preflight_ok" ]; then
        echo "No .preflight_ok in $REPO."
        echo
        echo "Run the verification first (~10 min, and it is the thing that"
        echo "catches a broken wandb login before it costs you two days):"
        echo
        echo "    ./tmp_scripts/preflight.sh && touch .preflight_ok"
        echo
        echo "SKIP_PREFLIGHT=1 overrides, at your own risk."
        [ "${SKIP_PREFLIGHT:-0}" = "1" ] || exit 1
    else
        echo "preflight passed $(date -r .preflight_ok '+%F %T')"
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
    RUN_TAG="$tag" ./train_test_scripts/run_config.sh "configs/${cfg}.yaml" \
        cuda:0 --wandb True --num-workers "$DL_WORKERS" "$@" >/dev/null
}

stagger() {   # let one run get past its cache load before the next starts
    [ "$DRY_RUN" = "1" ] || sleep 60
}

wait_for_trainings() {
    [ "$DRY_RUN" = "1" ] && { echo "  wait   until all trainings finish"; return; }
    say "waiting for trainings to finish"
    while pgrep -u "$ME" -f "main.py .*--device cuda" >/dev/null; do
        sleep 300
    done
    say "all trainings finished"
}

# Number of main.py training processes currently running (dataloader workers
# share the cmdline, so count distinct configs, not pids).
n_trainings() {
    pgrep -u "$ME" -af "main.py .*--device cuda" 2>/dev/null \
        | grep -o 'configs/[^ ]*' | sort -u | wc -l
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
    for cfg in "${ARMS[@]:0:$SLOTS}"; do
        train "$cfg" "run"
        stagger           # simultaneous cache loads thrash the filesystem
    done

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

    say "PHASE 2  evaluate the existing all8 checkpoints alongside them"
    # Configs this filesystem never trained report NO MODELS and are skipped,
    # so this doubles as an inventory of what node D actually holds.
    _lane=$(eval_workers_for "$N_SEEDS")
    say "         eval lane: $_lane workers alongside $N_SEEDS trainings"
    WORKERS=$_lane \
    evalq "all8" "" \
        configs/all8_union.yaml configs/all8_joint_chain.yaml \
        configs/all8_joint.yaml configs/all8_joint_lite.yaml \
        configs/all8_structural.yaml

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
