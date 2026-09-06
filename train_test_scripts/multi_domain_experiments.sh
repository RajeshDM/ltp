#!/bin/bash
# multi_domain_experiments.sh — Multi-domain experiment suite
#
# Covers the experiments from CLAUDE.md §3/§6:
#   - Single-domain baselines (skylines)
#   - Multi-domain union-vocab (C1 control / Baseline 0)
#   - Multi-domain structural featurization (C1 primary)
#   - Leave-one-out zero-shot (C1/C5)
#   - Test-only on arbitrary domains (using --test-domains)
#
# Domain problem counts (from pddlgym):
#   blocks:    name=manyblocks_ipcc_big  train=200, test=200
#   gripper:   name=gripper_ipcc         train=147, test=173
#   miconic:   name=miconic_ipcc         train=228, test=119
#   visitall:  name=visitall_ipcc        train=125, test=50
#   grid:      name=grid_ipcc            train=192, test=48
#   logistics: name=logistics_ipcc       train=156, test=96
#   spanner:   name=spanner_ipcc         train=234, test=96
#   rovers:    name=rovers_ipcc          train=312, test=54
#
# Usage: ./train_test_scripts/multi_domain_experiments.sh <experiment> [options]
#
# Experiments:
#   baselines       Single-domain training on each domain (skylines)
#   union           Multi-domain union-vocab training + test (C1 control)
#   structural      Multi-domain structural featurization + test (C1 primary)
#   leave-one-out   Leave-one-out zero-shot (C1/C5) with structural feat.
#   test-only       Test a trained model on specified domains (--test-domains)
#   all             Run baselines + union + structural + leave-one-out
#
# Options:
#   --device <dev>        cpu or cuda:0 (default: cuda:0)
#   --epochs <n>          Training epochs (default: 300)
#   --lr <rate>           Learning rate (default: 0.0005)
#   --train-domains <d>   Override training domains (comma-separated pddlgym names)
#   --test-domains <d>    Override test domains with counts (name:count,...)
#   --heldout <d>         Override held-out domains (comma-separated pddlgym names)
#   --dry-run             Print commands without running

set -euo pipefail

# ─── Defaults ──────────────────────────────────────────────────────────────

DEVICE="cuda:0"
EPOCHS=300
LR="0.0005"
DRY_RUN=""

# Default 3-domain setup (small enough to iterate quickly)
TRAIN_DOMAINS_3="manyblocks_ipcc_big,gripper_ipcc,miconic_ipcc"
TRAIN_DOMAINS_OVERRIDE=""
TEST_DOMAINS_OVERRIDE=""
HELDOUT_OVERRIDE=""

# Per-domain actual test counts (to avoid capping warnings)
BLOCKS_TEST=200
GRIPPER_TEST=173
MICONIC_TEST=119
VISITALL_TEST=50
GRID_TEST=48
LOGISTICS_TEST=96
SPANNER_TEST=96
ROVERS_TEST=54

# ─── Parse args ────────────────────────────────────────────────────────────

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <experiment> [options]"
    echo "Experiments: baselines, union, structural, leave-one-out, test-only, all"
    echo "Run with --help for full usage."
    exit 1
fi

EXPERIMENT=$1
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        --device)         DEVICE="$2"; shift 2 ;;
        --epochs)         EPOCHS="$2"; shift 2 ;;
        --lr)             LR="$2"; shift 2 ;;
        --train-domains)  TRAIN_DOMAINS_OVERRIDE="$2"; shift 2 ;;
        --test-domains)   TEST_DOMAINS_OVERRIDE="$2"; shift 2 ;;
        --heldout)        HELDOUT_OVERRIDE="$2"; shift 2 ;;
        --dry-run)        DRY_RUN="--dry-run"; shift ;;
        --help|-h)
            head -42 "$0" | tail -40
            exit 0
            ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

COMMON="--epochs $EPOCHS --lr $LR --device $DEVICE"

# ─── Helper ────────────────────────────────────────────────────────────────

step_count=0
total_steps=0

run_step() {
    local label="$1"
    shift
    step_count=$((step_count + 1))
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "=== Step $step_count/$total_steps: $label ==="
    echo "════════════════════════════════════════════════════════════════"
    echo "Started: $(date)"
    echo "Command: $*"
    "$@"
    echo "Finished: $(date)"
}

# ─── Experiment: Single-domain baselines ───────────────────────────────────

run_baselines() {
    total_steps=3
    step_count=0

    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Single-domain baselines (skylines for comparison)          ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    run_step "Single-domain BLOCKS" \
        ./train_test_scripts/gabar_run.sh train_test --domain blocks \
        --test-problems $BLOCKS_TEST $COMMON $DRY_RUN

    run_step "Single-domain GRIPPER" \
        ./train_test_scripts/gabar_run.sh train_test --domain gripper \
        --test-problems $GRIPPER_TEST $COMMON $DRY_RUN

    run_step "Single-domain MICONIC" \
        ./train_test_scripts/gabar_run.sh train_test --domain miconic \
        --test-problems $MICONIC_TEST $COMMON $DRY_RUN
}

# ─── Experiment: Union-vocab multi-domain (C1 control) ────────────────────

run_union() {
    local domains="${TRAIN_DOMAINS_OVERRIDE:-$TRAIN_DOMAINS_3}"
    total_steps=1
    step_count=0

    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Multi-domain UNION vocab (Baseline 0 / C1 control)        ║"
    echo "║  Domains: $domains"
    echo "╚══════════════════════════════════════════════════════════════╝"

    local test_doms="${TEST_DOMAINS_OVERRIDE:-manyblocks_ipcc_big:${BLOCKS_TEST},gripper_ipcc:${GRIPPER_TEST},miconic_ipcc:${MICONIC_TEST}}"
    local heldout_arg=""
    [ -n "$HELDOUT_OVERRIDE" ] && heldout_arg="--heldout $HELDOUT_OVERRIDE"

    run_step "Multi-domain union: train+test" \
        ./train_test_scripts/gabar_run.sh multi \
        --domains "$domains" \
        --test-domains "$test_doms" \
        --featurization union \
        $heldout_arg \
        $COMMON $DRY_RUN
}

# ─── Experiment: Structural featurization (C1 primary) ────────────────────

run_structural() {
    local domains="${TRAIN_DOMAINS_OVERRIDE:-$TRAIN_DOMAINS_3}"
    total_steps=1
    step_count=0

    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Multi-domain STRUCTURAL features (C1 primary)             ║"
    echo "║  Domains: $domains"
    echo "╚══════════════════════════════════════════════════════════════╝"

    local test_doms="${TEST_DOMAINS_OVERRIDE:-manyblocks_ipcc_big:${BLOCKS_TEST},gripper_ipcc:${GRIPPER_TEST},miconic_ipcc:${MICONIC_TEST}}"
    local heldout_arg=""
    [ -n "$HELDOUT_OVERRIDE" ] && heldout_arg="--heldout $HELDOUT_OVERRIDE"

    run_step "Multi-domain structural: train+test" \
        ./train_test_scripts/gabar_run.sh multi \
        --domains "$domains" \
        --test-domains "$test_doms" \
        --featurization structural \
        $heldout_arg \
        $COMMON $DRY_RUN
}

# ─── Experiment: Leave-one-out zero-shot (C1/C5) ──────────────────────────

run_leave_one_out() {
    total_steps=3
    step_count=0

    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Leave-one-out zero-shot (C1/C5)                           ║"
    echo "║  Structural featurization, each domain held out in turn     ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    # Train on blocks+gripper, zero-shot test on miconic
    run_step "Train blocks+gripper → zero-shot MICONIC" \
        ./train_test_scripts/gabar_run.sh multi \
        --domains "manyblocks_ipcc_big,gripper_ipcc" \
        --test-domains "manyblocks_ipcc_big:${BLOCKS_TEST},gripper_ipcc:${GRIPPER_TEST},miconic_ipcc:${MICONIC_TEST}" \
        --featurization structural \
        $COMMON $DRY_RUN

    # Train on blocks+miconic, zero-shot test on gripper
    run_step "Train blocks+miconic → zero-shot GRIPPER" \
        ./train_test_scripts/gabar_run.sh multi \
        --domains "manyblocks_ipcc_big,miconic_ipcc" \
        --test-domains "manyblocks_ipcc_big:${BLOCKS_TEST},miconic_ipcc:${MICONIC_TEST},gripper_ipcc:${GRIPPER_TEST}" \
        --featurization structural \
        $COMMON $DRY_RUN

    # Train on gripper+miconic, zero-shot test on blocks
    run_step "Train gripper+miconic → zero-shot BLOCKS" \
        ./train_test_scripts/gabar_run.sh multi \
        --domains "gripper_ipcc,miconic_ipcc" \
        --test-domains "gripper_ipcc:${GRIPPER_TEST},miconic_ipcc:${MICONIC_TEST},manyblocks_ipcc_big:${BLOCKS_TEST}" \
        --featurization structural \
        $COMMON $DRY_RUN
}

# ─── Experiment: Test-only ─────────────────────────────────────────────────

run_test_only() {
    total_steps=1
    step_count=0

    if [ -z "$TEST_DOMAINS_OVERRIDE" ]; then
        echo "Error: --test-domains is required for test-only experiment"
        echo "Example: $0 test-only --train-domains 'manyblocks_ipcc_big,gripper_ipcc' --test-domains 'spanner_ipcc:96'"
        exit 1
    fi

    local domains="${TRAIN_DOMAINS_OVERRIDE:-$TRAIN_DOMAINS_3}"

    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Test-only (using pre-trained model)                        ║"
    echo "║  Training domains (for model identity): $domains"
    echo "║  Test domains: $TEST_DOMAINS_OVERRIDE"
    echo "╚══════════════════════════════════════════════════════════════╝"

    run_step "Test-only" \
        ./train_test_scripts/gabar_run.sh multi \
        --domains "$domains" \
        --test-domains "$TEST_DOMAINS_OVERRIDE" \
        --featurization structural \
        --mode test \
        $COMMON $DRY_RUN
}

# ─── Dispatch ──────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Multi-Domain Experiment Suite                              ║"
echo "║  Device: $DEVICE | Epochs: $EPOCHS | LR: $LR"
echo "║  Experiment: $EXPERIMENT"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "Started at $(date)"

case $EXPERIMENT in
    baselines)
        run_baselines
        ;;
    union)
        run_union
        ;;
    structural)
        run_structural
        ;;
    leave-one-out)
        run_leave_one_out
        ;;
    test-only)
        run_test_only
        ;;
    all)
        run_baselines
        run_union
        run_structural
        run_leave_one_out
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Available: baselines, union, structural, leave-one-out, test-only, all"
        exit 1
        ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  EXPERIMENT COMPLETE: $EXPERIMENT"
echo "║  Finished at $(date)"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Results are in cache/results/"
