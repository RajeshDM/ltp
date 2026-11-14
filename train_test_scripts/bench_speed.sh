#!/bin/bash
# bench_speed.sh — Sequential speed benchmark (batch/seq × GPU/CPU × det/no-det)
#
# Runs 8 configurations one at a time for a fair comparison.
# Usage: ./train_test_scripts/bench_speed.sh [num_test_problems]
#   Default: 50 test problems
set -euo pipefail

N_TEST=${1:-50}
echo "Running speed benchmark with $N_TEST test problems"
echo ""

COMMON="--method ltp --domain manyblocks_ipcc_big --all-problems \
  --num-train-problems 200 --num-test-problems $N_TEST \
  --lr 0.0005 --n-heads 1 --attention-dropout 0 --dropout 0 --weight-decay 0 \
  --gnn-rounds 9 --epochs 300 --batch-size 16 --max-plan-length 200 \
  --ablation main --search-strat greedy --mode test --wandb False \
  --use-global-node True --run-learned-model True --run-non-optimal True"

mkdir -p traces

echo "════════════════════════════════════════════════════════════════"
echo "=== 1/8: GPU batch, no determinism ==="
echo "════════════════════════════════════════════════════════════════"
time GABAR_BATCH_EVAL=1 GABAR_TRACE_SCORES=traces/bench_gpu_batch.jsonl \
  python main.py $COMMON --device cuda:0

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "=== 2/8: GPU sequential, no determinism ==="
echo "════════════════════════════════════════════════════════════════"
time GABAR_TRACE_SCORES=traces/bench_gpu_seq.jsonl \
  python main.py $COMMON --device cuda:0

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "=== 3/8: CPU batch, no determinism ==="
echo "════════════════════════════════════════════════════════════════"
time GABAR_BATCH_EVAL=1 GABAR_TRACE_SCORES=traces/bench_cpu_batch.jsonl \
  python main.py $COMMON --device cpu

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "=== 4/8: CPU sequential, no determinism ==="
echo "════════════════════════════════════════════════════════════════"
time GABAR_TRACE_SCORES=traces/bench_cpu_seq.jsonl \
  python main.py $COMMON --device cpu

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "=== 5/8: GPU batch, deterministic ==="
echo "════════════════════════════════════════════════════════════════"
time GABAR_DETERMINISTIC=1 PYTHONHASHSEED=42 GABAR_BATCH_EVAL=1 \
  GABAR_TRACE_SCORES=traces/bench_gpu_batch_det.jsonl \
  python main.py $COMMON --device cuda:0

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "=== 6/8: GPU sequential, deterministic ==="
echo "════════════════════════════════════════════════════════════════"
time GABAR_DETERMINISTIC=1 PYTHONHASHSEED=42 \
  GABAR_TRACE_SCORES=traces/bench_gpu_seq_det.jsonl \
  python main.py $COMMON --device cuda:0

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "=== 7/8: CPU batch, deterministic ==="
echo "════════════════════════════════════════════════════════════════"
time GABAR_DETERMINISTIC=1 PYTHONHASHSEED=42 GABAR_BATCH_EVAL=1 \
  GABAR_TRACE_SCORES=traces/bench_cpu_batch_det.jsonl \
  python main.py $COMMON --device cpu

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "=== 8/8: CPU sequential, deterministic ==="
echo "════════════════════════════════════════════════════════════════"
time GABAR_DETERMINISTIC=1 PYTHONHASHSEED=42 \
  GABAR_TRACE_SCORES=traces/bench_cpu_seq_det.jsonl \
  python main.py $COMMON --device cpu

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "=== Correctness checks (action-only) ==="
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "--- Batch vs Sequential (same actions?) ---"
python compare_score_traces.py traces/bench_gpu_batch.jsonl traces/bench_gpu_seq.jsonl --action-only
python compare_score_traces.py traces/bench_cpu_batch.jsonl traces/bench_cpu_seq.jsonl --action-only

echo ""
echo "--- GPU vs CPU (same actions?) ---"
python compare_score_traces.py traces/bench_gpu_batch.jsonl traces/bench_cpu_batch.jsonl --action-only
python compare_score_traces.py traces/bench_gpu_seq.jsonl traces/bench_cpu_seq.jsonl --action-only

echo ""
echo "--- Det vs no-det (same actions?) ---"
python compare_score_traces.py traces/bench_gpu_batch.jsonl traces/bench_gpu_batch_det.jsonl --action-only
python compare_score_traces.py traces/bench_gpu_seq.jsonl traces/bench_gpu_seq_det.jsonl --action-only

echo ""
echo "--- Deterministic: GPU vs CPU (bit-exact?) ---"
python compare_score_traces.py traces/bench_gpu_batch_det.jsonl traces/bench_cpu_batch_det.jsonl
python compare_score_traces.py traces/bench_gpu_seq_det.jsonl traces/bench_cpu_seq_det.jsonl

echo ""
echo "Done. Compare wall-clock times above to see speedups."
