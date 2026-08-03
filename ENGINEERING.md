# ENGINEERING.md — improvement history, July–August 2026

Every substantial engineering change since the domain-agnostic project
started, consolidated: what the problem was, what changed, the measured
effect, and how it was verified. Detailed timing tables live in
PERFORMANCE.md; migration notes for people arriving from GABAR live in
ONBOARDING.md; this file is the one-place history and the distilled
principles (§9), written so the lessons can be lifted into sibling
projects whose code shares GABAR's logic but not its language.

All timings carry a hardware label because they are NOT comparable across
hosts: the identical serial graph build is 72.5s on the H100 host and
114.1s on the A40 host (single-threaded Python; the GPU is irrelevant).

## 0. The headline

Evaluating trained models used to take longer than training them.
Reference evaluation (visitall, 20 test problems, batched, H100 node):

    28:51  →  2:15   (12.8x)  featurizer + startup work, serial
    2:15   →  0:55   (2.5x)   + 16 rollout workers (32-core allocation)

Full 8-domain evaluation startup: 95s / 16 GB → 18s / 0.9 GB.
`cache/results` on disk: >100 GB → ~1/8 of that for new artifacts.
Every step verified byte-identical or outcome-identical before it landed.

## 1. Featurization inner loop (`_state_to_graph_ltp`)

The single hottest code in the repo: it runs per executed step at test
time, per state at collection time, for every featurization mode.

| change | problem | effect | verification |
|---|---|---|---|
| Sparse edge accumulation (`_EdgeSlice`) | dense `(arity, N, N, k)` float64 array allocated and scanned per state — multi-GB on large instances; cost was KERNEL time (zero-filling pages), not compute: system time 448s→5.5s, minor page faults 1.12e9→2.35e6 | sequential 29:00→7:02 (4.1x), batched 28:51→5:38 (5.1x), A40 | `tests/test_sparse_edges.py` pins byte-equality incl. edge ordering (emit pass sorts keys to match `np.argwhere`) |
| Goal membership via set | `literal in goal_literals` list scan = O(literals × goal) per state | featurization 12.5→6.1 ms/state (2.05x) | fingerprint tool |
| O(1) object positions + grounding-call dedup | `list.index()` per lookup; duplicate grounding enumeration | folded into the above | fingerprint tool |
| Sorts keyed by cached `repr` | sort comparisons called Python `__lt__` → `__str__` per pair; pddlgym caches `_str` at construction, so key once | ~8% | fingerprint tool |
| One fancy-index scatter | per-feature numpy scalar writes (`arr[i,j,k]=1` has per-call overhead) replaced by building index vectors and one assignment | featurization 6.8→5.7 ms/state (16%) | fingerprint tool, 400 states / 5 domains |

The pattern behind all five: the cost was allocation, membership scans,
and per-element interpreter overhead — never the arithmetic.

## 2. Evaluation harness

| change | problem | effect | verification |
|---|---|---|---|
| Batched lockstep evaluation (`GABAR_BATCH_EVAL=1`) | one model call per problem per step; GPU idle between | worth 0.5% before the sparse-edge fix, 1.25x after — batching cannot help while a kernel-bound serial cost dominates | trace + outcome parity vs sequential |
| Parallel rollout workers (`GABAR_FEATURIZE_WORKERS=N`, `ploi/parallel_rollout.py`) | graph build is independent per problem but serial in the parent | 50 problems, H100/32 cores: 389→104.6s at 16 workers; A40/16 cores: 623→215.9s | 12/12 parity cells (outcome-level) |
| Workers OWN problems (`index % N == w`) | shipping state to a stateless pool is a trap: a pddlgym state pickles in ~4.6ms vs ~5.7ms to featurize one — serialization would be 65% of the work | design constraint, not a speedup | measured before building |
| Envs built inside the workers | env construction (~0.9s each) ran serially in the parent | setup 45.0→5.6s; total −12.3% on 50 problems | 12/12 parity cells |
| Worker cap = available cores (affinity mask, not `os.cpu_count()`) | oversubscription is actively harmful: 2-core host, graph build 72.5s serial, 66.1s @2, 80.7s @16 | clamp, loud warning | measured curve |
| Metadata-only test startup (all three config paths) | `--mode test` loaded every training graph (95s / 16 GB per 8-domain eval) to use none of them | 95s→18s, 16 GB→0.9 GB | coverage unchanged |
| Per-problem fallback RNG (`_fallback_rng_for`) | one shared `random.Random` consumed problem-by-problem (sequential) vs interleaved (batched) → same model, different coverage: 70% vs 80% on visitall | correctness, not speed | parity matrix compares OUTCOMES because traces cannot see this |
| Beam decode cleanups | dead per-slot zero allocation in both beam decoders; mixed-arity handling in `beam_search_parallel` | small | trace parity |

Rejected, with the measurement that killed each (full list in
PERFORMANCE.md): PyG conversion inside workers (65s→631s — torch's
ForkingPickler puts every tensor in its own shared-memory segment);
threading (GIL-bound dict/list work); building torch tensors directly
instead of numpy (higher per-op overhead on small arrays); caching `repr`
on Literal (pddlgym already caches it — the cost was call count).

## 3. Storage and caches

| change | problem | effect |
|---|---|---|
| uint8 graph tensors | every node/edge feature is an indicator, stored as numpy-default float64 — 8x oversized; `cache/results` passed 100 GB | 8x smaller, lossless; consumers cast to float32 on load |
| Sidecar cache layout | featurized graphs for every config accumulated inside one shared pickle per domain (10+ GB, OOM on rewrite) | shared cache holds only raw 5-tuples + metadata; graphs in per-(domain, featurization-tag) sidecars; legacy blobs stripped on read |
| Cache keys from ACTUAL collected count (`_effective_problem_count`) | requesting 200 of a 147-problem domain created a key that could never complete | honest filenames; configs request 0 (= all) and share caches |
| Raw data collected once, featurization-independent | re-collection per representation mode | two-pass multi-domain pipeline: collect per domain, re-featurize per mode via `metadata_override` + `cache_tag` |

## 4. Config and launch system

- **YAML config layer** (`--config`, `apply_config_defaults`): one file per
  experiment, `base:` inheritance (constants < base < experiment < CLI),
  unknown keys are a hard error so a typo cannot silently fall back to a
  default. `configs/README.md` maps configs to paper claims.
- **Run modes** (toy/sweep/spot): pre-canned overrides for cheap iteration.
- **`run_config.sh`**: nohup launcher, log rotation (the stdout log is the
  only place the loss curve lives), `PYTHONUNBUFFERED` (block-buffered
  prints made healthy runs look stalled), `RUN_TAG` so concurrent launches
  of the SAME config (seed replicates, data fractions) do not interleave
  into one log.
- **`eval_queue.sh`**: N configs through `--mode test` one at a time on
  `--device cpu` — evaluation is host-CPU bound (137s GPU vs 135s CPU,
  same 20 problems), so it leaves the GPU to training; two concurrent
  evals oversubscribe the worker pool since both see the full affinity
  mask. Classifies "No models found" BEFORE the exit code (main.py logs it
  as a warning and exits 0 — an untrained key otherwise reports ok with an
  empty results file).
- **Bare-invocation refusal**: `python main.py` with no config/domain used
  to fall back to constants.py and silently start training ManyBlocks —
  the classic empty-shell-variable accident. Now refused
  (`GABAR_ALLOW_DEFAULTS=1` overrides).
- **Campaign scripts** (`tmp_scripts/`, throwaway by design): end-to-end
  preflight (trains 4 real epochs, evaluates, and reads the coverage back
  OFF the wandb server — a valid API key with no network path passes every
  local check and still loses two days), core-count-derived concurrency
  budget (`budget.sh` — the same node has shown up with 2, 16 and 32
  cores), self-detaching phase driver.

## 5. Checkpoint identity and selection

- **Featurization + input widths in the checkpoint key** (`feat`, `nf`,
  `ef`, later `l2`): runs differing only in representation used to share
  one ModelManager directory and evict each other's loss-ranked
  checkpoints. `GABAR_LEGACY_CKPT_KEY=1` reads pre-change checkpoints.
  Any hyperparameter that reaches the optimizer or changes tensor shapes
  must be in the key; anything defaulted-away must keep old directories
  byte-identical (verified against the folder-name/hash functions
  directly before landing).
- **Periodic checkpoints** (`--checkpoint-every`): zero-shot transfer
  peaks well before training loss bottoms out; loss-ranked slots keep late
  epochs and silently discard the best-transferring models.
- **Test-blind selection** (combined train+val loss) decided once,
  recorded in the paper, held fixed — also cut the multi-hour test phase
  ~3x vs testing three metrics.

## 6. Correctness apparatus

- **`tools/graph_fingerprint.py`**: hashes dtype/shape/bytes of every
  array of every featurized graph over seeded rollouts; before/after
  compare is the byte-identity gate for every featurizer refactor.
  Mandatory first step: run it against ITSELF (same code, two
  invocations) — `all_ground_literals` returns a set whose order follows
  PYTHONHASHSEED, so an unsorted rollout diverges between launches and a
  correct tool reports avalanche mismatches that look like a broken
  refactor. Fixed by sorting groundings by cached repr.
- **`tools/parity_matrix.sh`**: {cpu,gpu} × {det,nodet} ×
  {sequential,batched,pooled}, compared on per-step action traces AND
  final outcomes. Outcomes matter independently: the shared-RNG coverage
  bug produced identical traces and different coverage.
- **`compare_score_traces.py`** `--score-only` / `--action-only`: FP-
  tolerant vs execution-level trace comparison (scores may differ in float
  accumulation order while actions agree).
- **Dependency-free unit tests**: sparse-vs-dense equality
  (`test_sparse_edges.py`), lifted spec/metadata/renaming invariance
  (`test_lifted_layer.py`), union merge + structural classes
  (`test_multidomain_metadata.py`).
- **`GABAR_DETERMINISTIC=1`**: bit-reproducible runs at ~1.7x cost — for
  parity checks only, never production.
- **N-way verification against pristine baselines**: the original GABAR
  commit was patched (in a worktree) to emit traces, so the refactored
  harness could be compared against the untouched original, not just
  against itself.

## 7. Observability

- **wandb step bug**: test metrics were logged with `step=epoch`, but the
  logging loop walks domain by domain and the epoch RESTARTS per domain;
  wandb silently drops out-of-order steps, so only the FIRST test domain
  of an 8-domain evaluation ever reached the server. Epoch is now an
  ordinary field; per-key best-so-far is mirrored into `wandb.summary`,
  giving one sortable column per domain in the runs table.
- **wandb run identity**: `wandb.init` ran before the multi-domain name
  was resolved, so every `--domains` run recorded the constants.py default
  domain under an auto-generated name. Runs are now named
  `<group>/<expid>_s<seed>_<mode>`, grouped per machine
  (`GABAR_WANDB_GROUP`), tagged with featurization + seed. Labelling is
  wrapped in try/except: which Run attributes are writable changes across
  wandb releases (`Run.group` is read-only and killed a run when
  assigned) — cosmetic metadata must never take down a training job.
- **Profiling hooks**: `GABAR_PROFILE_BATCH=1` (bucket breakdown of a
  run: forward pass / graph build / decode / pyg convert / env setup),
  `tools/bench_featurize.py` (per-state featurizer cost on real rollout
  states, no model needed), `tools/profile_featurize.py`
  (caller-attribution), a cProfile hook on the eval inner loop.
- **Print hygiene**: startup clutter collapsed to one line; progress lines
  unbuffered under nohup.

## 8. Trainer (small, listed for completeness)

Vectorized index construction in the training loop; removed TF32/
cudnn.benchmark toggles (no measured benefit here, one less source of
nondeterminism). The trainer was never the bottleneck — evaluation was.

## 9. Principles distilled (the transferable list)

1. **Profile before touching.** The dominant cost was page-fault/allocation
   time and interpreter overhead, not arithmetic. A profiler with caller
   attribution found in hours what intuition had misattributed for weeks.
2. **Never allocate dense O(N²) per state when writes are sparse.** Bill
   the allocation, not just the arithmetic.
3. **Membership tests via sets/dicts in per-state loops.** One list scan
   inside a hot loop was a 2x on the whole featurizer.
4. **Sort by precomputed keys.** Comparison callbacks that stringify per
   comparison are hot-loop poison — and sorted order doubles as
   determinism insurance wherever iteration order is hash-dependent.
5. **Batch independent rollouts into one model call per round.** But note
   the ordering: batching was worth 0.5% until the serial kernel-bound
   cost was removed, then 1.25x. Fix the dominant term first.
6. **Parallelism: workers own state; only cheap plain buffers cross the
   pipe.** Measure serialization cost against work cost BEFORE designing
   the pool — if shipping the input costs ~the work, workers must own
   their inputs. Framework tensors never cross process boundaries.
7. **Build expensive per-worker state inside the worker.**
8. **Cap workers at the actual allocation (affinity mask) and at the
   measured plateau.** Oversubscription is harmful, not neutral, and
   `cpu_count()` lies on shared nodes.
9. **Inference must not load training data.** Metadata-only startup.
10. **Storage dtype honesty.** Indicator features at float64 are 8x their
    information content; it compounds into a 100 GB cache.
11. **Split raw from derived caches.** Raw data collected once,
    representation-independent; derived artifacts in sidecars keyed by
    mode tag; keys built from resolved (actual) counts, capped loudly.
12. **Checkpoint identity includes everything that changes shapes or
    meaning** (representation mode, feature widths, every optimizer-
    reaching hyperparameter), with defaults arranged so existing
    directories never move.
13. **Byte-identity gates for every hot-path refactor**, and the gate tool
    is validated against itself first — a nondeterministic gate produces
    false alarms indistinguishable from real breakage.
14. **Parity on outcomes as well as traces; per-unit RNG.** Any shared
    random stream consumed in different orders by two harnesses is a
    silent coverage bug that traces cannot see.
15. **Configs are data**: file-per-experiment, inheritance, unknown keys
    fatal, and bare invocation refused — silent defaults plus an empty
    shell variable equals an accidental training run.
16. **Unattended means verified end-to-end first**: run a tiny real
    pipeline and read the results back from the reporting server before
    walking away; local checks cannot see a broken network path.
17. **Label every measurement with its host** and never compare across
    hosts; host CPU generation moved timings 1.6x with the GPU idle.
18. **Record rejected ideas with the measurement that killed them** — the
    next person will have the same good-looking bad idea.
