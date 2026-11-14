# Coming from GABAR

This repo is GABAR extended into GADAR (domain-agnostic policies). If you
knew the GABAR codebase, most of your knowledge transfers directly. This is
the short list of what is the same, what is new and optional, and the four
things that will actually bite you.

## What has not changed

- **The entry point and command shape.** `python main.py --domain X --mode
  train_test` still works and still means what it meant.
- **The default featurization is published GABAR.** `--featurization`
  defaults to `per_domain`, which is the original one-hot representation.
  You get GABAR unless you ask for something else.
- **The model.** `HeteroGNN_global` encoder and `GNN_GRU` decoder are
  unchanged, including the output-lifted decoding (schema selection is a dot
  product against schema-node embeddings, parameters against object-node
  embeddings).
- **The training loop, the loss, and the plan-collection pipeline.**
- **The graph, for `per_domain`.** The featurizer was heavily optimised but
  the tensors it produces are byte-identical; `tools/graph_fingerprint.py`
  pins that, and the dense-array rewrite is covered by
  `tests/test_sparse_edges.py`. Only the dtype changed (`uint8` instead of
  float64, lossless, cast to float32 on load).

## What is new, and optional

| flag | what it does |
|---|---|
| `--config FILE.yaml` | every setting from a file. `configs/` has one per experiment; precedence is constants < `base:` < file < CLI |
| `--domains a,b,c` | train one model on several domains at once |
| `--featurization` | `per_domain` (GABAR), `union`, `structural`, `joint_lite`, `joint`, `joint_chain` (GADAR) |
| `--test-domains "d:N"` | evaluate on specific domains and problem counts |
| `--heldout-domains` | leave-one-domain-out zero-shot evaluation |

Environment variables, all off by default:

| variable | effect |
|---|---|
| `GABAR_BATCH_EVAL=1` | evaluate all test problems in lockstep, one batched model call per round. ~1.5x |
| `GABAR_FEATURIZE_WORKERS=N` | featurize and step in N forked worker processes. Up to ~2x more, needs cores (see PERFORMANCE.md) |
| `GABAR_PROFILE_BATCH=1` | print where evaluation time goes |
| `GABAR_DETERMINISTIC=1` | bit-reproducible runs. Costs ~1.7x; use for verification only |

Recommended for normal use: `GABAR_BATCH_EVAL=1
GABAR_FEATURIZE_WORKERS=<cores>`.

## The four things that will bite you

**1. Your old checkpoints will not be found.** The `ModelManager` key now
includes the featurization and the input feature widths, because runs
differing only in featurization used to share one directory and evict each
other. To read checkpoints saved before that change:

    GABAR_LEGACY_CKPT_KEY=1 python main.py ...

New runs should not use it, or the collision it fixes comes back.

**2. Coverage numbers will not reproduce your old ones exactly.** Two
deliberate changes to the execution harness:

- When the revisit monitor traps a rollout (every proposal leads to an
  already-visited state), the executor now samples among the model's valid
  proposals instead of repeating the first one, which used to produce
  permanent 2-cycles.
- That sampler is seeded **per problem**. It used to be one generator shared
  across the run, which made the sequential and batched harnesses take
  different fallback actions and report different coverage for the same
  model (measured: 70% vs 80% on visitall).

Both are fixes, but they change trajectories, so a rerun of an old
experiment will not match digit for digit. The model and the graph are not
what changed.

**3. A bare `python main.py` is refused.** With no `--config` and no
`--domain`, it used to fall back to `constants.py` (ManyBlocks,
`train_test`) and silently start training. Usually that meant a shell
variable had expanded to nothing. Override with `GABAR_ALLOW_DEFAULTS=1`.

**4. The cache layout changed.** Featurized graphs live in per-(domain, tag)
sidecars, `cache/results/<Domain>_graphs_0_<N><tag>.pkl`, instead of inside
the shared unified cache, which used to grow past 10 GB and OOM on rewrite.
Old caches still load; legacy in-pickle graph blobs are stripped on read.
You do not need to delete anything, but deleting
`cache/results/*_graphs_*.pkl` forces a rebuild at the smaller `uint8` size.

## Where to look

- `CLAUDE.md` - the navigation contract: which files are live, which are
  legacy, the call paths that matter, and the implicit contracts that will
  bite you if broken. **Read section 1 before changing anything.**
- `PERFORMANCE.md` - measured evaluation timings, what was optimised and how
  it was verified, worker-count scaling, and approaches that were tried and
  rejected with the measurement that killed each.
- `configs/README.md` - which config reproduces which claim.
- `tools/parity_matrix.sh` - proves the batched and parallel evaluators agree
  with the sequential reference. Run it after touching
  `_state_to_graph_ltp`, `_greedy_step_single` or the decoder.
- `tools/graph_fingerprint.py` - proves a featurizer change is byte-identical.

## A first run

    # train a per-domain GABAR model, exactly as before
    python main.py --domain Gripper_ipcc --mode train_test --epochs 500

    # evaluate an existing model, fast
    GABAR_BATCH_EVAL=1 GABAR_FEATURIZE_WORKERS=8 \
      python main.py --domain Gripper_ipcc --mode test

    # a GADAR experiment
    python main.py --config configs/all8_joint_chain.yaml
