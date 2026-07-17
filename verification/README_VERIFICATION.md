# Verification protocol: baselines vs speed-change branch

Temporary apparatus. Goal: prove that ALL changes on this branch - the
multi-domain harness era (cache fixes, test-loop restructure) AND the speed
work (Tier 2 eval glue, vectorized loss indices, parallel decoder, batched
lockstep eval) - produce the exact same per-call candidate scores as the
untouched code, and measure the speedup in the same session.

Two baseline commits:

- `536c352` "fixed small bug with traineval function" - ORIGINAL GABAR,
  the last commit before ANY work on this branch (before CLAUDE.md, before
  every multi-domain commit).
- `b48b06f` "CLAUDE.md: add live module map..." - post-multi-domain,
  pre-speed-work. Isolates the multi-domain-era changes from the speed
  changes: original vs this = multi-domain era; this vs speed branch =
  speed work.

`verification/baseline_trace.patch` adds ONLY score tracing (a trace-file
write per model call - no behavior or speed changes). The instrumented file
(`ploi/run_planner_with_ltp_v2.py`) is byte-identical at both baseline
commits, so the SAME patch applies cleanly to both (verified); it refuses
to apply anywhere the file differs. The model/checkpoint files
(`modelutils_ltp.py`, `model_checkpointing.py`) and the hyperparameter
dicts in main.py are also identical across both baselines and the current
branch, so all runs load the same checkpoints from `models/` and read the
same caches - scores are directly comparable. (The unified cache on disk
must be complete, which it is after any successful recent run.)

Use the SAME device (cpu or cuda:0) for every run - tiny cross-device float
differences would show up as noise.

## Step 0 - original-GABAR baseline run (local branch, never pushed)

```sh
mkdir -p traces
git checkout -b verify-original 536c352
git checkout claude/wizardly-rubin-hqMug -- verification/baseline_trace.patch compare_score_traces.py
git apply verification/baseline_trace.patch
git add -A && git commit -m "verification: trace instrumentation on original GABAR (local only)"

# classic command - NO --run-mode flag (does not exist on this branch);
# time it for the speed comparison
time GABAR_TRACE_SCORES=traces/original.jsonl \
  python main.py --method ltp --domain manyblocks_ipcc_big --all-problems \
    --num-train-problems 200 --num-test-problems 10 \
    --lr 0.0005 --n-heads 1 --attention-dropout 0 --dropout 0 --weight-decay 0 \
    --gnn-rounds 9 --epochs 300 --batch-size 16 --max-plan-length 500 \
    --ablation main --search-strat greedy --mode test --wandb False \
    --use-global-node True --run-learned-model True --run-non-optimal True \
    --device cpu
```

## Step 1 - pristine pre-speed-work baseline run (local branch, never pushed)

Same procedure at the post-multi-domain commit:

```sh
git checkout claude/wizardly-rubin-hqMug
git checkout -b verify-baseline b48b06f
git checkout claude/wizardly-rubin-hqMug -- verification/baseline_trace.patch compare_score_traces.py
git apply verification/baseline_trace.patch
git add -A && git commit -m "verification: trace instrumentation on pristine baseline (local only)"

time GABAR_TRACE_SCORES=traces/pristine.jsonl \
  python main.py [same classic command as Step 0]
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

## Step 3 - compare everything at once

Passing 3+ trace files prints a pairwise match matrix plus equivalence
groups ("what matches what"):

```sh
python compare_score_traces.py traces/original.jsonl traces/pristine.jsonl \
    traces/seq_v2.jsonl traces/seq_par.jsonl traces/batch.jsonl
```

Expected if everything is correct: one equivalence group containing all
five traces. If groups split, the boundary tells you which change layer
diverged:

- original vs pristine isolates the multi-domain-era changes (should be
  exact for single-domain runs - that code path was only looped, not
  changed).
- pristine vs seq_v2 isolates the Tier 2 glue changes (should be exact -
  no arithmetic touched).
- seq_v2 vs seq_par isolates the parallel decoder at batch size 1.
- seq_par vs batch isolates mixed-size mixed-arity batching.

For the per-problem detail on any diverging pair, rerun with exactly those
two files:

```sh
python compare_score_traces.py traces/original.jsonl traces/batch.jsonl
```

- "scores equal - tie reorder?" -> rerun with --ignore-order; if it then
  passes, only the ordering among equal-scored candidates differs.
- A divergence report shows only the FIRST divergent step per problem;
  later steps fork by construction.

Speedup = time(Step 0) vs time(run C) - original untouched code vs the
full batched system. Runs A/B locate where the speedup comes from.

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
git branch -D verify-original verify-baseline
```
