# Verification protocol: pristine baseline vs speed-change branch

Temporary apparatus. Goal: prove that ALL speed work (Tier 2 eval glue,
vectorized loss indices, parallel decoder, batched lockstep eval) produces
the exact same per-call candidate scores as the untouched code, and measure
the speedup in the same session.

The pristine baseline is commit `b48b06f` - the last commit before any
executable code was touched (before run-modes/AMP, before Tier 2).
`verification/baseline_trace.patch` adds ONLY score tracing to that commit
(a trace file write per model call - no behavior or speed changes). It is
generated against `b48b06f` exactly and will refuse to apply anywhere else.

Use the SAME device (cpu or cuda:0) for every run - tiny cross-device float
differences would show up as noise. All runs load the same checkpoints from
`models/` and the same caches, so scores are directly comparable.

## Step 1 - pristine baseline run (local branch, never pushed)

```sh
mkdir -p traces
git checkout -b verify-baseline b48b06f
# pull the patch + comparison script from the speed branch into this checkout
git checkout claude/wizardly-rubin-hqMug -- verification/baseline_trace.patch compare_score_traces.py
git apply verification/baseline_trace.patch
git add -A && git commit -m "verification: trace instrumentation on pristine baseline (local only)"

# classic command - NO --run-mode flag (does not exist on this branch);
# time it for the speed comparison
time GABAR_TRACE_SCORES=traces/pristine.jsonl \
  python main.py --method ltp --domain manyblocks_ipcc_big --all-problems \
    --num-train-problems 200 --num-test-problems 10 \
    --lr 0.0005 --n-heads 1 --attention-dropout 0 --dropout 0 --weight-decay 0 \
    --gnn-rounds 9 --epochs 300 --batch-size 16 --max-plan-length 500 \
    --ablation main --search-strat greedy --mode test --wandb False \
    --use-global-node True --run-learned-model True --run-non-optimal True \
    --device cpu
```

## Step 2 - speed-branch runs

```sh
git checkout claude/wizardly-rubin-hqMug

# A: Tier-2 sequential path, v2 decoder
time GABAR_TRACE_SCORES=traces/seq_v2.jsonl \
  ./train_test_scripts/gabar_run.sh test --domain blocks --test-problems 10 --device cpu

# B: sequential path, parallel decoder (batch of 1)
time GABAR_USE_PARALLEL_DECODER=1 GABAR_TRACE_SCORES=traces/seq_par.jsonl \
  ./train_test_scripts/gabar_run.sh test --domain blocks --test-problems 10 --device cpu

# C: full batched lockstep system
time GABAR_BATCH_EVAL=1 GABAR_TRACE_SCORES=traces/batch.jsonl \
  ./train_test_scripts/gabar_run.sh test --domain blocks --test-problems 10 --device cpu
```

## Step 3 - compare everything against pristine

```sh
python compare_score_traces.py traces/pristine.jsonl traces/seq_v2.jsonl
python compare_score_traces.py traces/pristine.jsonl traces/seq_par.jsonl
python compare_score_traces.py traces/pristine.jsonl traces/batch.jsonl
```

Interpretation:
- pristine vs seq_v2 isolates the Tier 2 glue changes (should be exact -
  those changes touch no arithmetic).
- pristine vs seq_par additionally isolates the parallel decoder at batch
  size 1.
- pristine vs batch is the full system: mixed-size mixed-arity batches.
- "scores equal - tie reorder?" -> rerun with --ignore-order; if it then
  passes, only the ordering among equal-scored candidates differs.
- A divergence report shows only the FIRST divergent step per problem;
  later steps fork by construction.

Speedup = time(Step 1) vs time(run C). Runs A/B locate where the speedup
comes from.

## Optional - training-loss verification

Training changes on the speed branch (vectorized loss indices; AMP/patience
machinery, inert in default mode) should be loss-identical. To check, run a
short training on both branches with the same seed and diff the printed
epoch losses:

```sh
# on verify-baseline and on the speed branch (add --run-mode default on the
# speed branch only):
python main.py --method ltp --domain manyblocks_ipcc_big --all-problems \
  --num-train-problems 200 --epochs 20 --mode train --wandb False [same flags]
```

## Cleanup when done

```sh
git checkout claude/wizardly-rubin-hqMug
git branch -D verify-baseline
```
