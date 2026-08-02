# Evaluation performance: measurements and settings

Everything here was measured, not estimated; where a number is inferred it
says so. Timings are **not comparable across machines** (see Hardware), so
each table names its host.

## Hardware

| tag | GPU | host CPU | cores in the allocation |
|---|---|---|---|
| **A40 node** (`cn-gpu7`) | NVIDIA A40 | older generation | 16 |
| **H100 node** (`dgxh-2`) | NVIDIA H100 | current generation | 2 |

The host CPU matters far more than the GPU for this workload. The identical
serial graph build takes **72.5s on the H100 host and 114.1s on the A40
host** because it is single-threaded Python; the GPU is idle throughout.
Never compare a timing from one node against the other.

## Current baseline (A40 node)

Config `configs/ab_visitall.yaml`, `--mode test`, `GABAR_BATCH_EVAL=1`,
`GABAR_FEATURIZE_WORKERS=16`, no determinism flag.

| test set | wall clock | eval only | peak RSS |
|---|---|---|---|
| visitall, 50 problems (full test set) | **228s** | 215.6s | 1.56 GB |
| visitall, 20 problems | **105s** | | |

Wall clock exceeds "eval only" by ~12.6s of interpreter startup, torch/PyG
imports, CUDA init and the results write. That cost is **fixed**, so it is
6% of a 50-problem run and 12% of a 20-problem one: prefer more domains per
process over more processes.

Profile of the 50-problem run:

| bucket | time | share |
|---|---|---|
| forward pass (GNN + decoder) | 91.4s | 41.9% |
| graph build (parallel) | 72.4s | 33.2% |
| decode + greedy step (parallel) | ~22s | ~10% |
| pyg convert | 26.2s | 12.0% |
| env setup (parallel) | 6.0s | 2.7% |

The network is now the largest single cost. It was 6.4% before this work.

## Harness comparison (A40 node, visitall 20 problems)

| harness | no determinism | with `GABAR_DETERMINISTIC=1` |
|---|---|---|
| sequential | 303s | 413s |
| batched | 223s | 290s |
| batched + 16 workers | **105s** | 179s |

Determinism costs ~1.7x. Batching and pooling are outcome-identical to
sequential without it (12/12 parity cells), so **do not set it for
production runs** - it is for parity checks only.

## What changed, in order of impact

Reference: visitall, 20 problems, batched, H100 node.
**28:51 -> 2:15 (12.8x)**, before the worker pool existed.

| change | effect | how verified |
|---|---|---|
| sparse edge accumulation (dropped a dense `(arity, N^2, k)` array) | 5.1x | byte-identical (`tests/test_sparse_edges.py`) |
| goal membership via set, not list scan (was O(literals x goal)) | 2.05x on featurization (12.5 -> 6.1 ms/state) | byte-identical |
| O(1) object positions + grounding-call dedup | folded into the above | byte-identical |
| sorts keyed by cached `repr` | ~8% | byte-identical |
| one fancy-index scatter instead of per-feature numpy scalar writes | 16% on featurization (6.8 -> 5.7 ms/state) | `graph_fingerprint.py`, 400 states / 5 domains |
| test-mode metadata-only startup | 8-domain startup 95s -> 18s, RSS 16 GB -> 0.9 GB | coverage unchanged |
| parallel rollout workers | 2.1x over batched (A40, 16 cores) | 12/12 parity cells |
| env construction inside the workers | 12.3% (see below) | 12/12 parity cells |

### Env construction in workers, isolated A/B (A40, 50 problems, 16 workers)

| | env setup | total |
|---|---|---|
| before | 45.01s, 44.59s | 245.0s, 246.5s |
| after | 5.59s, 5.66s | 217.4s, 213.5s |

**-39.2s of setup, -30.3s of total (12.3%).** About 9s returns as extra
rollout cost, most likely memory locality: 16 workers each allocate their
own envs instead of sharing already-faulted copy-on-write pages.

This scales with problem count, because env construction is O(N) (~0.9s per
problem serially) while the rollout has a fixed 500-round component and the
batched forward pass amortises as N grows. Setup was ~15.6% of the total at
20 problems and 18.2% at 50. On Blocks (200 test problems) serial setup
would be ~180s.

## Worker count

**H100 node, 32 cores, visitall 50 problems** (the definitive curve):

| workers | 0 | 2 | 4 | 8 | 16 | 25 | 32 |
|---|---|---|---|---|---|---|---|
| total | 389.2s | 234.6s | 159.8s | 118.3s | **104.6s** | 110.6s | 103.5s |
| graph build | 242.2s | 136.6s | 84.9s | 56.1s | 47.4s | 52.6s | 45.3s |

Graph build scales 5.3x; the total plateaus at ~104s from 16 workers on, so
**16 is the operating point** - beyond it you buy ~1% and would do better
spending those cores on a concurrent run. The 25-worker dip is load
imbalance (50 problems into 25 workers leaves an awkward remainder as
problems finish; 16 and 32 divide 50 more gracefully).

`forward pass` is flat at ~28s across every row, as it must be - worker
count does not touch the GPU. Useful as a sanity check on the profiler.

Even at 32 workers graph build is still the largest bucket (45%, against
forward pass at 27%): 500 rounds with a shrinking active set cannot
saturate that many workers.

**A40 node, 16 cores, visitall 20 problems** (too few problems to fill the
workers, which is the point):

| workers | 0 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| graph build | 114.1s | 64.7s | 41.2s | **29.0s** | 38.3s |

Here 16 is worse than 8: a worker idles once its own problems finish. Rule
of thumb:

    workers ~ min(available cores, problems / 2)

`configured_workers()` caps at the affinity mask, so an over-large setting
is clamped rather than harmful. On a 2-core allocation the whole pool is
worth ~5%.

**Intra-run workers and concurrent runs compete for the same cores.** When
running many seeds at once, divide the worker count down or drop it.

## Tried and rejected (with the measurement that killed it)

- **PyG conversion inside the workers.** Looked like a free 9%; graph build
  went 65s -> 631s. Only numpy may cross the pipe: torch installs a
  `ForkingPickler` that relocates every tensor into its own shared-memory
  segment, so each graph costs ~14 file descriptors and mmaps.
- **A stateless featurization pool** (ship `(state, groundings)` to
  workers). A pddlgym state pickles in 4.59ms against ~5.7ms to featurize
  one, so serialisation would be 65% of the work. Hence workers own
  problems, not tasks.
- **Building torch tensors directly instead of numpy.** Targets `pyg
  convert` (2.8% at the time) and torch has higher per-op Python overhead
  than numpy on small arrays.
- **Threading.** Featurization is dict/list/object manipulation and holds
  the GIL. Python 3.13's free-threaded build would work but needs
  free-threaded wheels for torch, PyG, torch_scatter and pddlgym.
- **Caching `repr` on `Literal`.** pddlgym already caches `_str` and `_hash`
  at construction. The cost was call *count*, not call *cost*.

## Recommended settings

    GABAR_BATCH_EVAL=1 GABAR_FEATURIZE_WORKERS=<cores>

No determinism flag. Add `GABAR_PROFILE_BATCH=1` for the bucket breakdown.

## Verification

Re-run the full parity matrix whenever `_state_to_graph_ltp`,
`_greedy_step_single` or the decoder changes - batch and pool would both
inherit such a bug and agree with each other:

    POOL_WORKERS=16 bash tools/parity_matrix.sh configs/ab_visitall.yaml "visitall_ipcc:20" check

For anything else the 4-run check suffices:

    POOL_WORKERS=16 HARNESSES="batch pool" DEVICES=gpu bash tools/parity_matrix.sh ...

For featurizer changes specifically, `tools/graph_fingerprint.py` proves
byte-identity directly. Always run it against itself first (same code, two
invocations): a tool whose own rollouts drift reports avalanche mismatches
that look like a broken refactor.
