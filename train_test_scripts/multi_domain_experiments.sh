#!/bin/bash
# multi_domain_experiments.sh — Overnight run: single-domain baselines + multi-domain experiments
#
# Phase 1 experiments (CLAUDE.md §6):
#   1. Single-domain baselines on 3 domains (comparison targets)
#   2. Multi-domain union-vocab training on same 3 domains (does it still work?)
#   3. Multi-domain with held-out domain (zero-shot test — expected near-zero)
#
# Usage: ./train_test_scripts/multi_domain_experiments.sh [device]
#   Default device: cuda:0
set -euo pipefail

DEVICE=${1:-cuda:0}
EPOCHS=300
LR=0.0005

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Multi-Domain Experiment Suite                              ║"
echo "║  Device: $DEVICE | Epochs: $EPOCHS | LR: $LR               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Plan:"
echo "  1. Single-domain: blocks (train+test)"
echo "  2. Single-domain: gripper (train+test)"
echo "  3. Single-domain: miconic (train+test)"
echo "  4. Multi-domain: blocks+gripper+miconic (train+test on all 3)"
echo "  5. Multi-domain: blocks+gripper (train), miconic held-out (zero-shot)"
echo "  6. Multi-domain: blocks+miconic (train), gripper held-out (zero-shot)"
echo "  7. Multi-domain: gripper+miconic (train), blocks held-out (zero-shot)"
echo ""
echo "Starting at $(date)"
echo ""

COMMON="--epochs $EPOCHS --lr $LR --device $DEVICE"

# ─── 1. Single-domain baselines ────────────────────────────────────────────

echo "════════════════════════════════════════════════════════════════"
echo "=== 1/7: Single-domain BLOCKS (train+test) ==="
echo "════════════════════════════════════════════════════════════════"
echo "Started: $(date)"
./train_test_scripts/gabar_run.sh train_test --domain blocks $COMMON
echo "Finished: $(date)"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "=== 2/7: Single-domain GRIPPER (train+test) ==="
echo "════════════════════════════════════════════════════════════════"
echo "Started: $(date)"
./train_test_scripts/gabar_run.sh train_test --domain gripper $COMMON
echo "Finished: $(date)"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "=== 3/7: Single-domain MICONIC (train+test) ==="
echo "════════════════════════════════════════════════════════════════"
echo "Started: $(date)"
./train_test_scripts/gabar_run.sh train_test --domain miconic $COMMON
echo "Finished: $(date)"
echo ""

# ─── 2. Multi-domain: train on all 3, test on all 3 ───────────────────────

echo "════════════════════════════════════════════════════════════════"
echo "=== 4/7: Multi-domain ALL THREE (train+test) ==="
echo "════════════════════════════════════════════════════════════════"
echo "Started: $(date)"
./train_test_scripts/gabar_run.sh multi --domains "manyblocks_ipcc_big,gripper_ipcc,miconic_ipcc" $COMMON
echo "Finished: $(date)"
echo ""

# ─── 3. Leave-one-out zero-shot experiments ────────────────────────────────

echo "════════════════════════════════════════════════════════════════"
echo "=== 5/7: Multi-domain blocks+gripper, HELD-OUT: miconic ==="
echo "════════════════════════════════════════════════════════════════"
echo "Started: $(date)"
./train_test_scripts/gabar_run.sh multi --domains "manyblocks_ipcc_big,gripper_ipcc" \
    --heldout "miconic_ipcc" $COMMON
echo "Finished: $(date)"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "=== 6/7: Multi-domain blocks+miconic, HELD-OUT: gripper ==="
echo "════════════════════════════════════════════════════════════════"
echo "Started: $(date)"
./train_test_scripts/gabar_run.sh multi --domains "manyblocks_ipcc_big,miconic_ipcc" \
    --heldout "gripper_ipcc" $COMMON
echo "Finished: $(date)"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "=== 7/7: Multi-domain gripper+miconic, HELD-OUT: blocks ==="
echo "════════════════════════════════════════════════════════════════"
echo "Started: $(date)"
./train_test_scripts/gabar_run.sh multi --domains "gripper_ipcc,miconic_ipcc" \
    --heldout "manyblocks_ipcc_big" $COMMON
echo "Finished: $(date)"
echo ""

# ─── Summary ───────────────────────────────────────────────────────────────

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ALL EXPERIMENTS COMPLETE                                   ║"
echo "║  Finished at $(date)                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "What to check in the morning:"
echo "  1. Single-domain success rates (the skylines)"
echo "  2. Multi-domain (4/7) success rates on each domain vs single-domain"
echo "     → How much does union-vocab multi-task training hurt?"
echo "  3. Zero-shot (5-7/7) success rates on held-out domains"
echo "     → Expected: near 0% (new symbols are untrained one-hot slots)"
echo "     → If >0%: union-vocab has some transfer (surprising!)"
echo "     → This is the baseline that structural features must beat"
