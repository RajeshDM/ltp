#!/bin/bash
#
# gabar_run.sh — Unified runner for GABAR training and testing
#
# Covers: single-domain (published GABAR), multi-domain (domain-independent),
#         all 4 run modes (default/toy/sweep/spot), and HP sweeps.
#
# Run with --help for full usage.

set -euo pipefail

# Deterministic execution: fix Python hash seed so grounding enumeration
# order is stable across runs. Combined with torch.use_deterministic_algorithms
# in main.py, this makes inference fully reproducible.
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"

# ─── Domain registry ────────────────────────────────────────────────────────
# name=<pddlgym name>;train=<count>;test=<count>
declare -A DOMAIN_CONFIGS=(
    ["blocks"]="name=manyblocks_ipcc_big;train=200;test=200"
    ["gripper"]="name=gripper_ipcc;train=147;test=173"
    ["miconic"]="name=miconic_ipcc;train=228;test=119"
    ["visitall"]="name=visitall_ipcc;train=125;test=50"
    ["grid"]="name=grid_ipcc;train=192;test=48"
    ["logistics"]="name=logistics_ipcc;train=156;test=96"
    ["spanner"]="name=spanner_ipcc;train=234;test=96"
    ["rovers"]="name=rovers_ipcc;train=312;test=54"
    ["klondike"]="name=klondike_solitaire;train=264;test=99"
)

show_usage() {
    cat <<'USAGE'
Usage: gabar_run.sh <command> [options]

Commands:
  train           Train a single-domain model
  test            Test a pre-trained model
  train_test      Train then immediately test
  sweep           HP sweep (loops over combos, uses sweep run-mode)
  multi           Multi-domain training (union or structural featurization)
  toy             Quick dry-run on CPU (alias for train --run-mode toy)
  spot            Full production run on spot GPU (alias for train_test --run-mode spot)
  list-domains    Show available predefined domains

Options (all commands):
  --domain <name>       Predefined domain shorthand (e.g., blocks, gripper)
                        or raw pddlgym name (e.g., manyblocks_ipcc_big)
  --train-problems <n>  Override training problem count
  --test-problems <n>   Override test problem count
  --lr <rate>           Learning rate (default: 0.0005)
  --heads <n>           Attention heads (default: 1)
  --attn-drop <rate>    Attention dropout (default: 0)
  --drop <rate>         Regular dropout (default: 0)
  --decay <rate>        Weight decay (default: 0)
  --gnn-rounds <n>      GNN message-passing rounds (default: 9)
  --epochs <n>          Number of epochs (default: 300)
  --batch-size <n>      Batch size (default: 16)
  --plan-len <n>        Max plan length (default: 500)
  --ablation <type>     Ablation: main, main_val, no_ag_no_cd (default: main)
  --search <strat>      Search strategy: greedy, dfs (default: greedy)
  --run-mode <mode>     default, toy, sweep, spot (default: default)
  --wandb               Enable wandb logging
  --no-global-node      Disable global node
  --auto-shutdown       Shut down instance after completion (spot mode)
  --patience <n>        Early stopping patience (0=disabled)
  --continue-training   Resume from best checkpoint
  --starting-epoch <n>  Override starting epoch (default: 0)
  --device <dev>        cpu or cuda:0 (default: cuda:0)
  --extra "<flags>"     Extra flags passed verbatim to main.py
  --dry-run             Print the command without running it

Sweep-only options:
  --sweep-lr "<rates>"          Space-separated LR values
  --sweep-heads "<counts>"      Space-separated head counts
  --sweep-attn-drop "<rates>"   Space-separated attn dropout rates
  --sweep-drop "<rates>"        Space-separated dropout rates
  --sweep-decay "<rates>"       Space-separated weight decay rates
  --sweep-gnn "<rounds>"        Space-separated GNN round counts

Multi-domain / test options:
  --domains "<d1,d2,...>"        Comma-separated training domains
  --heldout "<d1,d2,...>"        Comma-separated held-out domains
  --test-domains "<d1:n,...>"    Override test domains with per-domain counts
                                  (e.g. "manyblocks_ipcc_big:200,gripper_ipcc:173")
                                  Domains not in training set are auto-detected as zero-shot
  --featurization <mode>         per_domain, union, structural (default: union)

Examples:
  # Quick sanity check (TOY mode, CPU, 2 epochs, tiny data)
  ./gabar_run.sh toy --domain blocks

  # Single-domain train+test (replicates published GABAR)
  ./gabar_run.sh train_test --domain blocks --epochs 300 --lr 0.0005

  # HP sweep on free Colab GPU
  ./gabar_run.sh sweep --domain blocks --sweep-lr "0.0005 0.001" --sweep-heads "1 4"

  # Full production on spot instance
  ./gabar_run.sh spot --domain blocks --epochs 300 --auto-shutdown

  # Multi-domain with structural featurization
  ./gabar_run.sh multi --domains "manyblocks_ipcc_big,gripper_ipcc" \
      --heldout "spanner_learning" --featurization structural --epochs 300

  # Test a pre-trained model
  ./gabar_run.sh test --domain blocks --test-problems 10

  # Test multi-domain model on specific domains with per-domain counts
  ./gabar_run.sh test --domains "manyblocks_ipcc_big,gripper_ipcc,miconic_ipcc" \
      --test-domains "manyblocks_ipcc_big:200,gripper_ipcc:173"

  # Zero-shot test on unseen domain (auto-detected)
  ./gabar_run.sh test --domains "manyblocks_ipcc_big,gripper_ipcc" \
      --test-domains "spanner_ipcc:96" --featurization structural
USAGE
    exit 0
}

# ─── Parse command ──────────────────────────────────────────────────────────

if [ "$#" -lt 1 ]; then show_usage; fi

COMMAND=$1
shift

if [ "$COMMAND" = "--help" ] || [ "$COMMAND" = "-h" ]; then show_usage; fi

if [ "$COMMAND" = "list-domains" ]; then
    echo "Available domains:"
    for key in $(echo "${!DOMAIN_CONFIGS[@]}" | tr ' ' '\n' | sort); do
        config=${DOMAIN_CONFIGS[$key]}
        name=$(echo "$config" | grep -o 'name=[^;]*' | cut -d'=' -f2)
        train=$(echo "$config" | grep -o 'train=[^;]*' | cut -d'=' -f2)
        test_n=$(echo "$config" | grep -o 'test=[^;]*' | cut -d'=' -f2)
        printf "  %-15s -> %-25s (train=%s, test=%s)\n" "$key" "$name" "$train" "$test_n"
    done
    exit 0
fi

# ─── Defaults ───────────────────────────────────────────────────────────────

DOMAIN_KEY=""
DOMAIN_NAME=""
TRAIN_PROBLEMS=""
TEST_PROBLEMS=""
LR="0.0005"
HEADS="1"
ATTN_DROP="0"
DROP="0"
DECAY="0"
GNN_ROUNDS="9"
EPOCHS="300"
BATCH_SIZE="16"
PLAN_LEN="500"
ABLATION="main"
SEARCH="greedy"
RUN_MODE="default"
WANDB="False"
GLOBAL_NODE="True"
AUTO_SHUTDOWN=""
PATIENCE=""
CONTINUE_TRAINING=""
STARTING_EPOCH=""
DEVICE="cuda:0"
EXTRA=""
DRY_RUN=false

# Sweep arrays (empty = no sweep, use single value)
SWEEP_LR=""
SWEEP_HEADS=""
SWEEP_ATTN_DROP=""
SWEEP_DROP=""
SWEEP_DECAY=""
SWEEP_GNN=""

# Multi-domain
DOMAINS=""
HELDOUT=""
TEST_DOMAINS=""
FEATURIZATION="union"

# ─── Handle command aliases ─────────────────────────────────────────────────

case $COMMAND in
    toy)
        RUN_MODE="toy"
        MODE="train"
        DEVICE="cpu"
        ;;
    spot)
        RUN_MODE="spot"
        MODE="train_test"
        ;;
    sweep)
        RUN_MODE="sweep"
        MODE="train_test"
        ;;
    multi)
        MODE="train_test"
        ;;
    train|test|train_test)
        MODE="$COMMAND"
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Run with --help for usage."
        exit 1
        ;;
esac

# ─── Parse options ──────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --domain)          DOMAIN_KEY="$2"; shift 2 ;;
        --train-problems)  TRAIN_PROBLEMS="$2"; shift 2 ;;
        --test-problems)   TEST_PROBLEMS="$2"; shift 2 ;;
        --lr)              LR="$2"; shift 2 ;;
        --heads)           HEADS="$2"; shift 2 ;;
        --attn-drop)       ATTN_DROP="$2"; shift 2 ;;
        --drop)            DROP="$2"; shift 2 ;;
        --decay)           DECAY="$2"; shift 2 ;;
        --gnn-rounds)      GNN_ROUNDS="$2"; shift 2 ;;
        --epochs)          EPOCHS="$2"; shift 2 ;;
        --batch-size)      BATCH_SIZE="$2"; shift 2 ;;
        --plan-len)        PLAN_LEN="$2"; shift 2 ;;
        --ablation)        ABLATION="$2"; shift 2 ;;
        --search)          SEARCH="$2"; shift 2 ;;
        --run-mode)        RUN_MODE="$2"; shift 2 ;;
        --wandb)           WANDB="True"; shift ;;
        --no-global-node)  GLOBAL_NODE="False"; shift ;;
        --auto-shutdown)   AUTO_SHUTDOWN="--auto-shutdown"; shift ;;
        --patience)        PATIENCE="$2"; shift 2 ;;
        --continue-training) CONTINUE_TRAINING="True"; shift ;;
        --starting-epoch)  STARTING_EPOCH="$2"; shift 2 ;;
        --device)          DEVICE="$2"; shift 2 ;;
        --extra)           EXTRA="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=true; shift ;;
        --sweep-lr)        SWEEP_LR="$2"; shift 2 ;;
        --sweep-heads)     SWEEP_HEADS="$2"; shift 2 ;;
        --sweep-attn-drop) SWEEP_ATTN_DROP="$2"; shift 2 ;;
        --sweep-drop)      SWEEP_DROP="$2"; shift 2 ;;
        --sweep-decay)     SWEEP_DECAY="$2"; shift 2 ;;
        --sweep-gnn)       SWEEP_GNN="$2"; shift 2 ;;
        --domains)         DOMAINS="$2"; shift 2 ;;
        --heldout)         HELDOUT="$2"; shift 2 ;;
        --test-domains)    TEST_DOMAINS="$2"; shift 2 ;;
        --featurization)   FEATURIZATION="$2"; shift 2 ;;
        --mode)            MODE="$2"; shift 2 ;;
        --help|-h)         show_usage ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ─── Resolve domain ─────────────────────────────────────────────────────────

if [ "$COMMAND" != "multi" ]; then
    if [ -z "$DOMAIN_KEY" ]; then
        echo "Error: --domain is required"
        exit 1
    fi

    CONFIG=${DOMAIN_CONFIGS[$DOMAIN_KEY]:-""}
    if [ -n "$CONFIG" ]; then
        DOMAIN_NAME=$(echo "$CONFIG" | grep -o 'name=[^;]*' | cut -d'=' -f2)
        DEFAULT_TRAIN=$(echo "$CONFIG" | grep -o 'train=[^;]*' | cut -d'=' -f2)
        DEFAULT_TEST=$(echo "$CONFIG" | grep -o 'test=[^;]*' | cut -d'=' -f2)
        TRAIN_PROBLEMS=${TRAIN_PROBLEMS:-$DEFAULT_TRAIN}
        TEST_PROBLEMS=${TEST_PROBLEMS:-$DEFAULT_TEST}
    else
        DOMAIN_NAME="$DOMAIN_KEY"
        if [ -z "$TRAIN_PROBLEMS" ] || [ -z "$TEST_PROBLEMS" ]; then
            echo "Error: custom domain '$DOMAIN_KEY' requires --train-problems and --test-problems"
            exit 1
        fi
    fi
fi

# ─── Build command ──────────────────────────────────────────────────────────

build_cmd() {
    local lr=$1 heads=$2 attn_drop=$3 drop=$4 decay=$5 gnn=$6

    local cmd="python main.py --method ltp"

    if [ "$COMMAND" = "multi" ]; then
        cmd+=" --domains ${DOMAINS}"
        [ -n "$HELDOUT" ] && cmd+=" --heldout-domains ${HELDOUT}"
        cmd+=" --featurization ${FEATURIZATION}"
        [ -n "$TRAIN_PROBLEMS" ] && cmd+=" --num-train-problems ${TRAIN_PROBLEMS}"
        [ -n "$TEST_PROBLEMS" ] && cmd+=" --num-test-problems ${TEST_PROBLEMS}"
    else
        cmd+=" --domain ${DOMAIN_NAME}"
        cmd+=" --num-train-problems ${TRAIN_PROBLEMS}"
        cmd+=" --num-test-problems ${TEST_PROBLEMS}"
    fi

    [ -n "$TEST_DOMAINS" ] && cmd+=" --test-domains ${TEST_DOMAINS}"
    cmd+=" --all-problems"
    cmd+=" --lr ${lr}"
    cmd+=" --n-heads ${heads}"
    cmd+=" --attention-dropout ${attn_drop}"
    cmd+=" --dropout ${drop}"
    cmd+=" --weight-decay ${decay}"
    cmd+=" --gnn-rounds ${gnn}"
    cmd+=" --epochs ${EPOCHS}"
    cmd+=" --batch-size ${BATCH_SIZE}"
    cmd+=" --max-plan-length ${PLAN_LEN}"
    cmd+=" --ablation ${ABLATION}"
    cmd+=" --search-strat ${SEARCH}"
    cmd+=" --mode ${MODE}"
    cmd+=" --run-mode ${RUN_MODE}"
    cmd+=" --wandb ${WANDB}"
    cmd+=" --use-global-node ${GLOBAL_NODE}"
    cmd+=" --run-learned-model True"
    cmd+=" --run-non-optimal True"
    cmd+=" --device ${DEVICE}"
    [ -n "$AUTO_SHUTDOWN" ] && cmd+=" ${AUTO_SHUTDOWN}"
    [ -n "$PATIENCE" ] && cmd+=" --early-stopping-patience ${PATIENCE}"
    [ -n "$CONTINUE_TRAINING" ] && cmd+=" --continue-training ${CONTINUE_TRAINING}"
    [ -n "$STARTING_EPOCH" ] && cmd+=" --starting-epoch ${STARTING_EPOCH}"
    [ -n "$EXTRA" ] && cmd+=" ${EXTRA}"

    echo "$cmd"
}

run_cmd() {
    local cmd="$1"
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "$cmd"
    echo "════════════════════════════════════════════════════════════════"
    if [ "$DRY_RUN" = true ]; then
        echo "(dry run — not executed)"
    else
        eval "$cmd"
    fi
}

# ─── Execute ────────────────────────────────────────────────────────────────

if [ "$COMMAND" = "multi" ]; then
    if [ -z "$DOMAINS" ]; then
        echo "Error: --domains is required for multi-domain training"
        exit 1
    fi
    cmd=$(build_cmd "$LR" "$HEADS" "$ATTN_DROP" "$DROP" "$DECAY" "$GNN_ROUNDS")
    run_cmd "$cmd"

elif [ "$COMMAND" = "sweep" ]; then
    # Use sweep arrays if provided, else single values
    lr_list=${SWEEP_LR:-$LR}
    head_list=${SWEEP_HEADS:-$HEADS}
    attn_list=${SWEEP_ATTN_DROP:-$ATTN_DROP}
    drop_list=${SWEEP_DROP:-$DROP}
    decay_list=${SWEEP_DECAY:-$DECAY}
    gnn_list=${SWEEP_GNN:-$GNN_ROUNDS}

    count=0
    for lr in $lr_list; do
        for heads in $head_list; do
            for attn in $attn_list; do
                for drop in $drop_list; do
                    for decay in $decay_list; do
                        for gnn in $gnn_list; do
                            count=$((count + 1))
                        done
                    done
                done
            done
        done
    done
    echo "Sweep: $count configuration(s)"

    run_num=0
    for lr in $lr_list; do
        for heads in $head_list; do
            for attn in $attn_list; do
                for drop in $drop_list; do
                    for decay in $decay_list; do
                        for gnn in $gnn_list; do
                            run_num=$((run_num + 1))
                            echo ""
                            echo "─── Run $run_num / $count ───"
                            cmd=$(build_cmd "$lr" "$heads" "$attn" "$drop" "$decay" "$gnn")
                            run_cmd "$cmd"
                        done
                    done
                done
            done
        done
    done

else
    # Single run: train, test, train_test, toy, spot
    cmd=$(build_cmd "$LR" "$HEADS" "$ATTN_DROP" "$DROP" "$DECAY" "$GNN_ROUNDS")
    run_cmd "$cmd"
fi

echo ""
echo "Done."
