# CLAUDE.md — Project North Star

**Read this before writing or changing any code in this repo.**

Every line of code in this project exists to test one of the five hypotheses in
§3. Before adding code, identify which claim it serves. If it serves none, it
does not belong here (see §8, rule 1).

---

## 1. What this repo is

This repo contains **GABAR** (GrAph neural network Based Action Ranking,
NeurIPS 2025): a learned, search-free policy for classical planning. GABAR
converts a PDDL state into an action-centric graph, encodes it with a GNN
(edge/node/global updates + attention), and decodes a grounded action
`(schema, o1..on)` with a GRU that picks the schema, then each parameter
autoregressively. It trains per domain on optimal plans from small instances
and generalizes to much larger instances of the *same* domain.

GABAR's published takeaways: (a) putting ground actions in the graph is
essential (GABAR-ACT ablation), (b) conditional decoding of parameters is
essential (GABAR-CD), (c) ranking actions beats learning value functions
(GABAR-RANK).

**The new project extends GABAR from domain-specific to domain-agnostic.**

### 1.1 The live module map (verified 2026-07 by tracing imports and calls)

The repo contains more files than this; everything not reachable from this
map is legacy (§1.2). Cite files and function names only — line numbers rot
(§8, rule 9).

```
main.py                            ← THE entry point (__main__): arg parsing,
 │                                   multi-domain harness, data section, training,
 │                                   run_tests loop, held-out zero-shot eval
 ├─ ploi/argparsers.py             get_ploi_argument_parser (--domains,
 │                                   --heldout-domains, --test-domains,
 │                                   --featurization, --num-models-to-test,
 │                                   --test-model-metrics; --all-problems is
 │                                   added inside main.py) + apply_config_defaults
 │                                   (--config YAML layer with base: inheritance:
 │                                   constants < base < experiment YAML < CLI)
 ├─ ploi/constants.py              defaults for every flag
 ├─ ploi/datautils_ltp.py          THE data file: plan collection + per-problem
 │                                   cache (collect_training_data), unified-cache
 │                                   orchestration (process_pddl_to_graphs), graph
 │                                   construction (_create_graph_structure_ltp,
 │                                   _state_to_graph_ltp), PyG conversion.
 │                                   Edge features accumulate in `_EdgeSlice`
 │                                   (one sparse {(sender,receiver): {feat: v}}
 │                                   map per parameter position), NOT a dense
 │                                   (arity, N, N, k) array - see §1.4 contract 6
 ├─ ploi/multidomain.py            multi-domain helpers: parse_domain_arg,
 │                                   merge_feature_metadata, merge_action_spaces (C1)
 ├─ ploi/structural.py             structural featurization: StructuralMap,
 │                                   build_structural_metadata (C1/C2)
 ├─ ploi/lifted_layer.py           lifted domain layer + binding layer (5.4/5.5):
 │                                   build_lifted_spec / build_lifted_metadata,
 │                                   lifted_node_keys, add_lifted_node_features,
 │                                   add_lifted_layer_edges. Featurizations
 │                                   'joint_lite' (GADAR-BIND), 'joint'
 │                                   (full GADAR), and 'joint_chain' (joint +
 │                                   build_chain_spec / schema_goal_distances
 │                                   chaining edges, add_chain_node_features
 │                                   goal features, add_grounded_effect_edges);
 │                                   binding edges are added in
 │                                   _get_precondition_satisfaction_position.
 │                                   TYPE COMPILATION (all three modes):
 │                                   build_lifted_spec turns each declared PDDL
 │                                   type into a unary static predicate with one
 │                                   'pre' occurrence per typed slot
 │                                   (spec['type_preds'], spec['type_occ']);
 │                                   add_type_edges links objects to their type
 │                                   symbol ['has_type']
 ├─ ploi/modelutils_ltp.py         HeteroGNN_global encoder; GNN_GRU decoder
 │                                   (compute_action_scores / compute_object_scores)
 │    └─ ploi/attention_layer.py   attention via torch-scatter scatter_softmax
 ├─ ploi/ablations.py              GNN_non_CD, GNN_non_AG_CD, GNN_Val
 ├─ ploi/traineval.py              training loops — live:
 │                                   train_model_graphnetwork_ltp_batch_allows_both
 │                                   (main ablations) / ..._ltp_batch_val (val)
 ├─ ploi/model_checkpointing.py    ModelManager — checkpoints keyed by
 │                                   train_env_name + seed + hyperparameters
 ├─ ploi/run_planner_with_ltp_v2.py  live test path: PlannerTester.test_planners →
 │                                   _run_learned_model → _run_greedy_search →
 │                                   convert_state_and_run_model (self-contained)
 ├─ ploi/run_planner_with_ltp_v1.py  live ONLY as the source of main.py's
 │                                   _create_planner import (→ ploi/planning FD)
 ├─ ploi/test_utils.py             PlannerConfig, PlannerType, metrics,
 │                                   log_model_metrics (defined 2×; last wins)
 ├─ ploi/planning/                 FD wrapper (fd.py, pddl_planner.py, validate.py)
 └─ ploi/baselines/exp_{1,2,3}/    external baselines, only via --exp-baseline* flags
configs/                           one YAML per paper experiment (--config);
                                   suite files are deltas over _common.yaml
                                   (base: key); README.md maps configs to claims
tools/analyze_results.py           aggregates the results_*.json dumps main.py
                                   writes after testing into one table/CSV
tools/random_policy_baseline.py    zero-shot floor: uniform-random applicable
                                   action rollouts, config test-domains syntax
                                   (incl. @train), no model/training needed
train_test_scripts/run_config.sh   nohup-launch one config on one GPU
train_test_scripts/RUNBOOK.md      results-to-claims matrix, prioritized
                                   launch commands, cut order
train_test_scripts/RUNS_STATUS.md  live ledger: which runs are in flight,
                                   what each feeds, readout commands, measured
                                   floors/ceilings to compare against
tests/test_multidomain_metadata.py dependency-free unit tests: union merge,
                                   structural classes, renaming invariance
tests/test_sparse_edges.py         dependency-free unit tests: sparse edge
                                   accumulation is byte-identical to the dense
                                   array it replaced (edges, order, values)
tests/test_lifted_layer.py         dependency-free unit tests: lifted spec
                                   (roles, bindings, occurrence order),
                                   metadata widths, joint vs joint_lite,
                                   joint_chain chaining/goal features,
                                   type compilation (typed vs predicate-typed
                                   domains agree structurally; type renaming
                                   invariance), renaming invariance of lifted
                                   tensors
```

### 1.2 What to ignore (legacy — do not read, never edit)

- **Non-LTP method branches** in main.py (`scenegraph`, `hierarchical`,
  `ploi`) and their modules: `ploi/datautils.py`, `ploi/modelutils.py`,
  `ploi/guiders.py`, `ploi/guidance/`, `ploi/planning/incremental_*.py`,
  `scenegraph_planner.py`, `seek_planners.py`. Reachable only with
  `--method` ≠ ltp; this project never uses them.
- **Auxiliary scripts:** `main_seek.py`, `visualize_graphs.py`,
  `analyse_domain_results.py`, `compute_stats.py`,
  `generate_scrubed_problems.py`, `ploi/scriptable_models.py`,
  `ploi/profile_manager.py`, `ploi/server_information.py`.
- **Dead-but-imported:** `_collect_training_data` and
  `_collect_training_data_ltp` (imported in main.py, never called — live
  collection is `collect_training_data`); `run_planner_with_gnn_ltp`
  (imported from v1, never called); `*_old` functions
  (`graph_dataset_to_pyg_dataset_old`, `compute_action_scores_old`).
- `VAL-master/` (validator source), `train_test_scripts/`, `cache/`,
  `models/` (artifacts, not code).

### 1.3 The call paths that matter (file + function granularity)

**A. Data.** main.py data section → `process_pddl_to_graphs`
(datautils_ltp.py) — per batch: `collect_training_data` (plans + per-problem
unified cache) → `_create_graph_dataset_ltp` → `_create_graph_structure_ltp`
(per-domain feature metadata; skipped if `metadata_override` given) +
`_create_graph_from_state_ltp` → `state_to_graph_wrapper` →
`_state_to_graph_ltp` → `graph_dataset_to_pyg_dataset`. Multi-domain
(`--domains`): pass 1 per-domain collection; pass 2 re-featurizes with
merged/structural metadata via `metadata_override` + `cache_tag`.

**B. Training.** main.py `'ltp'` branch: model_class by `--ablation`
(main → `GNN_GRU`) → `initialize_model` → `train_func` = 
`train_model_graphnetwork_ltp_batch_allows_both` (NOT the similarly named
`..._ltp_batch` — that one is dead on the live path) → checkpoints via
`ModelManager`.

**C. Testing.** main.py test section: per test domain, `PlannerConfig` +
`PlannerTester` (run_planner_with_ltp_v2.py) → `run_tests` (main.py) loads
best models per metric from `ModelManager` → `tester.test_planners` →
`_run_learned_model` → greedy/DFS search calling
`convert_state_and_run_model`, which re-featurizes each state with the
passed `graph_metadata`.

**D. Multi-domain eval (new, C1).** Test loop iterates `train_domain_names`
individually (composite `MULTI-...` key never reaches `pddlgym.make`);
held-out loop builds per-domain metadata (structural: own aliases at
canonical max arities; union: `allow_unknown_symbols=True`) and runs the
same `run_tests` machinery.

### 1.4 Implicit contracts (preserve them)

1. **Feature widths must agree across domains** for mixed batching:
   `graph_metadata['num_node_features'/'num_edge_features']` set model input
   sizes (`args.num_node_features_object` etc. in main.py).
2. **The decoder is output-lifted:** schema selection =
   dot(GRU hidden, schema-node embedding) (`compute_action_scores`);
   parameter selection = dot(hidden, object-node embedding)
   (`compute_object_scores`). No head is sized by |A| or |O| — keep it that
   way (§5.7).
3. **Goal literals are prefixed `WANT`** (`wrap_goal_literal`,
   datautils_ltp.py); structural.py's goal classes depend on this string.
4. **graph_metadata is a plain dict** with keys `node_feature_to_index`,
   `edge_feature_to_index`, `num_node_features`, `num_edge_features`,
   `all_predicates`, `unary/binary_predicates`, `unary_types` (+ structural
   mode adds `max_pred_arity`, `max_action_arity`, `featurization`;
   zero-shot union eval adds `allow_unknown_symbols`). New featurizations
   must emit the same keys.
5. **Cache layout — two files.** The shared
   `cache/results/<Domain>_unified_cache_0_<N>.pkl` holds ONLY raw
   per-problem 5-tuples `(states, objects, plan, groundings, goal_dists)`
   plus small metadata (`graph_metadata<tag>`) — it stays small and is
   loaded/rewritten cheaply. **Featurized graphs live in a per-(domain,tag)
   sidecar** `cache/results/<Domain>_graphs_0_<N><tag>.pkl` holding
   `(graph_list, metadata)`. This replaced the old design that accumulated
   every config's graphs inside the shared pickle (grew to 10+ GB and
   OOM'd on rewrite). `process_pddl_to_graphs` returns the sidecar directly
   if present; on load it strips any legacy in-pickle `all_graphs*`/
   `batch_graphs` blobs (migration). Loaders reject per-problem tuples
   shorter than 5.
6. **Graph tensors are indicator matrices, stored as `uint8`.** Every node
   and edge feature is a flag or a one-hot (§5.3), so `nodes` and `edges`
   are built as `uint8`: lossless, and 8x smaller on disk than numpy's
   default float64, which is what made `cache/results` grow past 100 GB.
   Every consumer casts to float32 on load (`graph_to_pyg_data`,
   `traineval.py`). Do not introduce a non-binary node/edge feature without
   changing the dtype and saying so here.
7. **Edge features are accumulated sparsely** (`_EdgeSlice`). The dense
   `(max_action_arity, num_nodes, num_nodes, num_edge_features)` array it
   replaced cost O(arity*N^2*k) to allocate and scan for every state, which
   dominated test-time re-featurization (multi-GB per state on large
   instances). Writers use `all_edge_features[i, j, k] = 1` unchanged; the
   emit pass sorts keys so edge ordering matches `np.argwhere` exactly.
   `tests/test_sparse_edges.py` pins that equivalence.
8. **Checkpoint identity includes the featurization and the input widths**
   (`feat`, `nf`, `ef` in `training_hyperparameters`, main.py). Without them
   runs differing only in featurization shared one `ModelManager` directory
   and competed for the same loss-ranked slots.

   `<N>` is the ACTUAL collected count: `_effective_problem_count` resolves
   the requested `num_train_problems` (`<=0` = all the domain has) against
   the domain size before the filename is built, so the name is honest and
   any config requesting `>=` the domain size shares the same cache.

### 1.5 Repo hazards (conventions that will bite you)

1. **Similar-name traps.** `collect_training_data` (live) vs
   `_collect_training_data` / `_collect_training_data_ltp` (dead);
   `train_model_graphnetwork_ltp_batch_allows_both` (live) vs
   `..._ltp_batch` (dead on the live path); `_create_planner` exists in both
   v1 (imported by main.py) and v2 (used internally). Before changing "old"
   code, trace the live path (§8, rule 8).
2. **Same-name redefinition:** `log_model_metrics` is defined twice in
   test_utils.py — Python keeps the last. Grep for later definitions before
   editing any function.
3. **`args.datafile` is reassigned mid-flow** in main.py (generic
   `ploi_<domain>.pkl`, then `training_data_<domain>.pkl` inside the ltp
   branch) — the first existence check gates the whole data section.
4. **Cache filename keyed by ACTUAL collected count** (`_effective_problem_count`,
   §1.4 contract 5): requesting 200 of a 147-problem domain yields
   `..._0_147.pkl`; `num_train_problems <= 0` means "all". Configs request
   `0` (use all) so the requested count no longer appears in the key. (Was
   a hazard when the key held the requested count and `all_complete` never
   turned true for small domains — now retired.)
5. **`--num-train-problems` applies per domain** in `--domains` mode unless
   a per-domain `:count` override is given (`parse_domain_arg`).
6. **Two typing conventions in the suite.** Blocksworld, gripper, miconic,
   spanner and rovers declare PDDL types; grid and logistics spell types as
   ordinary unary predicates. `structural.py` collapses every declared type
   to one `typed_object` class (renaming invariance forbids type-name
   features), so before type compilation only the second family's type
   constraints were representable — the cause of gripper's 4-argument
   type-invalid groundings. `build_lifted_spec` now compiles declared types
   into unary static predicates so both families agree; `structural`
   (no lifted layer to host symbol nodes) stays type-blind by design.
7. **Expert-planner failures on hard instances are normal**
   (`Planning failed for problem N`): satisficing `fd-lama-first` vs optimal
   `fd-opt-lmcut` differ in reach; failed problems are skipped — watch the
   per-domain skip count for dataset imbalance.

## 2. The research goal (the contribution, stated once)

> **Capability claim.** We build the first *cross-domain generalizing policy*
> for classical planning: a single model π(s, G, D) that takes the **domain
> description D as input** — alongside state s and goal G — and constructs
> grounded actions in PDDL domains **never seen in training**, with no search
> at test time.

Category (Nowozin's taxonomy): **Capability** — doing something that could not
be done before. Prior to this work, no learned policy could even be *executed*
on a new domain's predicates and action schemas (per-predicate parameters or
per-domain one-hot vocabularies made it architecturally impossible).
Consequence for evaluation: we do not need to beat per-domain GABAR on its own
domains; we need to prove the capability is real and explain what enables it.

One-line version for intros: *GOOSE showed how to make the input
domain-agnostic — for heuristics that still need search. GABAR showed how to
construct grounded actions by ranking — within one domain. Neither can do what
the other does. We show the two are complementary.*

## 3. The five claims and their certifying experiments

Every experiment exists to certify exactly one claim; every claim has exactly
one certifying experiment. This table is the contract.

| # | Claim (takeaway for the reader) | Kind | Certifying experiment |
|---|---|---|---|
| C1 | **The domain description is data, not architecture.** π(s,G,D) executes zero-shot on unseen domains. | Capability | Zero-shot coverage on held-out domains ≫ union-vocab baseline (D-in-weights control, §6 Phase 1) |
| C2 | **Lifted graphs become actionable only when grounded applicability is bound to them.** The grounded–lifted *binding layer* (§5.5) is the enabling component. | Insight (enables C1) | Binding-removal ablation → zero-shot collapse; Prop. 1 (§7) for the *why* |
| C3 | **For transfer, learn order, not magnitude.** Local action ranking is scale-free; cross-domain value regression needs a shared scale that doesn't exist. | Insight | Value-objective ablation trained cross-domain on our exact graph vs. ranking; GOOSE-DI + Müller et al. margin result as external corroboration |
| C4 | **Domain semantics: static or contextual?** Compiled (encode D once) vs. joint (contextualize D per state) conditioning. Informative either way: joint wins → symbol meaning must be computed in context; tie → conditioning is amortizable at per-domain inference cost. | Insight | Method A vs. Method B (§5.6), overall + on structurally-similar (aliasing-prone) domain pairs |
| C5 | **Cross-domain policies transfer strategic structure, measurably, with a provable ceiling.** | Insight | Domain-family analysis (transport-family → unseen transport vs. unrelated), few-shot data-efficiency curves vs. per-domain GABAR, PDDLFuse generated-domain stress test |

Risk profile: C1/C2 are existential — they need zero-shot to beat the controls
meaningfully, not to beat per-domain skylines. C3 needs the value ablation to
actually fail (predicted by theory + GOOSE-DI numbers, but let data decide).
C4/C5 are win-either-way analyses.

## 4. Why this hasn't been done (verified against primary sources)

Rubric: (output type, search at test?, held-out domains?, symbol encoding,
per-domain params?). All read in full, not from memory:

| Work | Output | Search | Held-out domains | Symbols | Per-domain params |
|---|---|---|---|---|---|
| STRIPS-HGN (ICAPS'20) | heuristic | yes | **yes** | structural, grounded | no |
| GOOSE (AAAI'24) | heuristic | yes | **yes** (DI setting) | structural, lifted (LLG) | no |
| WL-GOOSE (ICAPS'24) | heuristic | yes | no | WL colors, per-domain dicts | effectively yes |
| Ståhlberg et al. R-GNN(/[t]) (ICAPS'22–AAAI'25) | value→policy | no | no | identity | **yes** (per-predicate MLPs) |
| ASNets (JAIR'20) | policy | no | no | identity | **yes** (per-schema modules) |
| Müller et al. Q-values (ICAPS'26) | policy | no | no | identity | **yes** |
| Khandelwal et al. (2024) | heuristic | yes | yes (15-puzzle action-set variants only) | propositional | no |
| GABAR (this repo, as published) | **policy** | **no** | no | identity (one-hots) | yes (vocab-sized features) |
| **This project** | **policy** | **no** | **yes** | **structural** | **no** |

Key verified facts to cite correctly:
- GOOSE's LLG **does** encode the full lifted domain: schema subgraph with
  pre/add/del structure, argument bindings via index-feature (IF) nodes.
  "Domain in the graph" is theirs *for heuristics*. Our input-side novelty is
  NOT "put the domain in the graph"; it is the binding layer (§5.5).
- GOOSE's DI setting **does** hold out test domains (trains on IPC'98–'18
  minus the 8 test domains). Never claim "first held-out-domain evaluation";
  claim "first held-out-domain evaluation *of a policy*".
- GOOSE Thm 4.3: MPNNs on LLG cannot compute even h^add/h^max — the lifted
  graph is provably information-lossy. DI-LLG is empirically their worst
  configuration. This is our "why it was not possible" evidence.
- GOOSE Sec. 4 explicitly names lifted policy learning as un-done future work.
- Müller et al. found vanilla Q-regression fails *because it does not
  discriminate actions* and fixed it with a ranking-flavored margin — cite as
  independent convergent evidence for C3.
- LLMs: o1 drops 97.8%→37.3% under symbol renaming (Mystery Blocksworld);
  our model is renaming-invariant *by construction*. LLM code-synthesis
  approaches (Silver et al. etc.) are per-domain test-time compilation
  requiring solved instances + validator loops — a different question.

## 5. Representations (high-level first, then details)

### 5.0 The one-sentence idea

Stop telling the network *which symbol* something is (one-hot identity);
tell it *what role the symbol plays in the lifted domain* (structure), and
bind every ground action's applicability to that lifted structure so the
decoder can rank actions in a domain it has never seen.

### 5.1 Current GABAR graph (the starting point; per-domain)

Built in `ploi/datautils_ltp.py`:
- `_create_graph_structure_ltp`: builds per-domain feature dictionaries.
  **This is where all domain-specificity lives:**
  - node features: one-hot over this domain's action schemas
    (`_node_feature_to_index[action]`), predicates (+ goal copies via
    `G(predicate)`), object types → feature width = 3 + |A| + 2|P| + |T|.
  - edge features: `action_object` flag, positional one-hots `pos_i`,
    `pred_pos_i`, and **precondition-satisfaction flags indexed by
    domain-specific strings** (`precond_str + position`).
- `_state_to_graph_ltp`: per-state graph. Nodes = objects ∪
  ground literals (state + goal) ∪ **action-schema nodes** ∪ global. Edges =
  literal–object (with arg positions) and schema–object edges carrying, per
  parameter position, which preconditions that object satisfies in an
  applicable grounding.
- Model in `ploi/modelutils_ltp.py`: `HeteroGNN_global` encoder (edge → node →
  global updates, attention, global node), `GNN_GRU` decoder. The decoder is
  already output-lifted: schema selection = dot(GRU hidden, schema-node
  embedding) (`compute_action_scores`); parameter selection = dot(hidden,
  object-node embedding) (`compute_object_scores`). No output head is sized
  by |A|.

Two facts to exploit: GABAR's action nodes are already *schema* nodes (one per
schema, groundings live on edges), and the decoder needs no mechanistic change.

### 5.2 Baseline 0 — union vocabulary (control for C1)

Pad the existing one-hot dictionaries to the union over all training domains
(|A_total|, |P_total|, |T_total|, max arity K). D stays in the weights. This is
multi-task learning, not domain conditioning. Expected: fine on training
domains, ~dead zero-shot (new domain's symbols land in untrained slots).
Cheapest to build; every later result is measured against it.

### 5.3 Shared change — structural featurization (no symbol identity)

Replace identity one-hots with features computable for ANY domain:
- node type bits (object / literal / schema / lifted-symbol / global)
- arity, is-goal-atom bit, is-static-predicate bit, type-structural features
- argument/parameter positions: adopt GOOSE-style IF index features (random
  fixed unit vectors per index, arity-unbounded) OR learned positional table;
  compare. Never per-domain one-hot positions.
Renaming invariance must hold by construction: assert that permuting symbol
names yields identical tensors (unit test, §6 Phase 2 gate).

### 5.4 The lifted domain layer (schema graph)

Nodes for: predicate symbols, action schemas, schema parameter slots, types.
Edges for: "predicate p appears at argument-binding (i→slot j) in pre/add/del
of schema a", "slot j has type t", "predicate arg i has type t". This is
LLG-adjacent — cite GOOSE honestly — but ours differs in role (§5.5) and in
engineering (positions as edge features, not chains of index nodes; undirected
message flow; global node retained).

### 5.5 THE novel component — the grounded–lifted binding layer

For each applicable ground action (already enumerated by GABAR's pipeline):
its schema–object edges are annotated not with per-domain precondition strings
but with **links to the lifted precondition nodes they instantiate**: "object o
at parameter slot j satisfies precondition-occurrence node p_{a,pre,k}".
Plus instantiation edges: every ground literal ↔ its predicate-symbol node.

Why this is the contribution: LLG has lifted preconditions but no ground
actions; GABAR has ground applicability but indexed by untransferable strings.
The binding layer expresses ground applicability *in the vocabulary of the
domain description*. Prop. 1 (§7) argues it strictly increases MPNN
expressiveness over LLG. Ablation: keep the graph, strip binding annotations
→ C2 predicts zero-shot collapse.

### 5.6 Two methods, one principle (both are contributions; A is not an ablation of B)

- **Method A — compiled ("domain-as-prompt").** A small schema-encoder GNN
  runs ONCE per domain over §5.4; its output embeddings initialize
  predicate/schema/type features in the instance graph (§5.1 topology + §5.5
  annotations). Per-state graph size ≈ GABAR's → per-domain inference cost.
  Embeds "symbol meaning is static per domain."
- **Method B — joint ("domain-in-the-loop").** One unified graph per state:
  lifted layer (§5.4) ∪ instance graph, connected by instantiation + binding
  edges. Message passing contextualizes symbol meaning per state/goal; lifted
  symbol nodes double as typed information hubs (generalizing the global node,
  which GABAR-G showed is critical). Embeds "symbol meaning is contextual."
Don't oversell A-vs-B as speed-vs-accuracy until measured: B's overhead is
domain-sized constants + one instantiation edge per ground literal.

### 5.7 Decoder changes (deliberately minimal)

Mechanism unchanged (that's part of the story). Only: (a) parameter loop runs
to the *selected schema's* arity (`action_parameter_number_dict` exists),
capped by a global max-arity hyperparameter, not a per-domain constant;
(b) nothing sized by |A| or |O|; (c) schema candidates = schema-node
embeddings (contextual in B, compiled in A).

## 6. Experiment / coding plan (phases with go/no-go gates)

Work on branch `claude/wizardly-rubin-hqMug`. Adjust the plan at each gate —
later phases are intentions, not commitments.

- **Phase 0 — multi-domain harness (infrastructure, serves everything).**
  Generalize `main.py` / `process_pddl_to_graphs` / dataset caching to N
  domains: per-domain metadata, per-domain grounding, mixed-domain batches
  (PyG batching already handles heterogeneous graph sizes; feature widths must
  agree per §5.2 or §5.3). Config for train-domain set + held-out set.
  **Gate:** single-domain GABAR reproduced through the new harness (parity on
  1–2 domains, e.g. Blocks + Gripper).
- **Phase 1 — union-vocab baseline (C1 control).** §5.2. Train on N−1
  domains, test: (i) training domains, (ii) held-out domain zero-shot.
  **Gate:** pipeline solid; control numbers recorded (expect near-zero
  zero-shot — that's the point).
- **Phase 2 — structural features + Method A (first real zero-shot signal).**
  §5.3 + §5.4 + §5.5 + compiled encoder. Renaming-invariance unit test.
  **Gate (the big one):** zero-shot coverage > union-vocab by a clear margin
  on ≥2 held-out domains → C1 alive, proceed. If not: debug with the
  aliasing/expressiveness lenses before adding machinery; check whether
  training-domain performance survived the identity-feature removal (the
  "expressiveness tax" — measure and record it regardless).
- **Phase 3 — Method B (joint).** §5.6. Compare A vs. B overall + on
  aliasing-prone pairs (C4). Measure actual per-state cost of both.
- **Phase 4 — ablations + analysis.** Binding-removal (C2), value-objective
  (C3), leave-one-domain-out sweep over all 8 GABAR domains, domain-family
  analysis + few-shot curves + PDDLFuse stress test (C5). LLM baseline reuse
  from GABAR appendix methodology if needed.
- **Phase 5 — theory.** Prop. 1 construction (adapt GOOSE Thm 4.3
  counterexample: two tasks, different optimal actions, LLG-indistinguishable,
  distinguishable with binding layer). Renaming-invariance proposition.

Standing decisions already made: start **grounded** (parity with GABAR; lifted
decoding is future work), both A and B are first-class methods, evaluation =
zero-shot headline + few-shot secondary. Open decisions (decide in Phase 0):
exact train/held-out domain split; pddlgym vs. pymimir for multi-domain
parsing; where PDDLFuse domains enter (Phase 4 only, or also as training
augmentation — keep out of training for the headline runs to keep the claim
clean).

## 7. Prop. 1 (sketch to be formalized)

Statement: there exist pairs (s, D), (s', D') whose sets of optimal actions
differ, such that any MPNN on LLG(s,D), LLG(s',D') produces identical
outputs, but MPNNs on our binding-layer graph distinguish them.
Route: GOOSE Thm 4.3's pairs are "symmetric to MPNNs in LLG" while differing
in h^add/h^max; adapt so the pairs differ in *which action is optimal* /
*which actions are applicable*, and show the binding annotations break the
symmetry. Secondary proposition: π is invariant under domain isomorphism
(renaming) — also defines the transfer ceiling: structurally isomorphic but
semantically different domains provably force identical behavior (motivates
few-shot track; aliasing is a theorem, not a bug).

## 8. Rules for code written in this repo

1. **Every change serves a claim.** Name the claim (C1–C5) in the commit
   message or PR description. Code serving no claim is scope creep.
2. **The baseline is sacred.** Never break single-domain GABAR parity
   (Phase 0 gate is a permanent regression test).
3. **Invariance is tested, not assumed.** The renaming unit test runs in CI
   for every representation change.
4. **Record negative results.** C3/C4 are claims either way; a "failed"
   ablation is data, not failure. Keep numbers in the results log, not chat.
5. **Don't rebuild, extend.** The decoder, trainer, and search wrapper stay;
   changes concentrate in graph construction (`datautils_ltp.py`) and feature
   metadata. If a change wants to touch the decoder mechanism, re-read §5.7
   and justify against C1–C5 first.
6. **Match the repo's style; keep it simple.** Plain functions and
   tuples/dicts over classes — add a class only when state genuinely demands
   it. Extendable for the plans in this file, no speculative structure
   beyond them. Comments concise and only where the code can't say it.
7. **Caches are versioned, failures are loud.** Raw per-problem data is
   featurization-independent — collect once, namespace featurized variants
   by `cache_tag`, key by the training-domain set. When a cached artifact's
   schema changes (e.g. the per-problem tuple), version the check and reject
   stale entries loudly — never tolerate them silently (§1.4, contract 5).
   Cap requested problem counts at what exists, loudly.
8. **Trace before fixing.** Similar-looking functions coexist (§1.5): before
   changing "old" code, dry-run the exact target command, confirm the
   function is on the live path (§1.1/§1.3) and the bug is reachable, and
   grep the file for a later same-name definition. Treat pre-extension code
   as an API — extend around it rather than editing behavior in place.
9. **Keep the map current — file + function granularity, never line
   numbers.** §1.1–§1.5 are the navigation contract for every future agent.
   Any commit that adds, deletes, or rewires a live file — or moves a
   load-bearing function between files — updates the map (and §1.2's ignore
   list, if a file became legacy) in the same commit. Line numbers rot and
   must not appear in this document.

## 9. Glossary

- **π(s,G,D)** — domain-conditioned policy; D moves from weights to input.
- **Binding layer** — ground-action applicability annotated with links to the
  lifted precondition occurrences it instantiates (§5.5). The novel component.
- **Lifted layer / schema graph** — graph of the domain description itself
  (§5.4).
- **Method A / compiled** — schema encoder runs once per domain; static symbol
  embeddings. **Method B / joint** — lifted layer inside every state graph;
  contextual symbol embeddings.
- **Union-vocab** — multi-task control with D in the weights (§5.2).
- **Aliasing** — structurally identical symbols with different semantics
  across domains; unresolvable zero-shot by construction (see Prop. 1
  corollary), addressed by few-shot.
- **LLG** — GOOSE's lifted learning graph (closest prior representation;
  heuristics only).
- **IF features** — GOOSE's random-unit-vector index encodings for argument
  positions; arity-unbounded.

## 10. Verification chores before submission (do not lose these)

- Re-check two May-2026 arXiv papers (2605.25720, 2605.18674) camera-readies
  for any cross-domain policy experiments.
- Fresh literature pass for cross-domain policy claims at writing time.
- LLM-baseline numbers refresh (models move fast).
