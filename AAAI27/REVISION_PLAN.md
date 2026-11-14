# GADAR revision plan (post AAAI-27 review)

Review: 4/10, below threshold, with an explicit statement that a revision
addressing W1–W4 "would be a substantially different and considerably more
persuasive submission" and that supplying the isolating ablation plus
training-domain-selected checkpoints would warrant 5–6.

Full review: `review/aaai2027_review.md`. Paper as submitted: `gadar.tex`.

The reviewer accepts C1 (the capability claim) cleanly and calls it the
strongest part of the paper. Everything below is about C2 (the two novel
relations) and C3 (competence), which the review finds unsupported and
narrow respectively. Nothing here requires changing the idea.

---

## 0. Two statements in the paper are not true as written

These are not "weaknesses to address" — they are factual corrections, and
both were verified against the code before writing this plan. Fix them
first, because a reviewer who finds either one independently will discount
everything else.

**0.1 "Each configuration is trained for three seeds and the results
averaged" (Experimental Setup) is false.** No `loo8_*` or `all8_*` config
sets a seed, so every run inherits `constants.SEED = 10`. Every number in
both tables is single-seed. Either run the seeds (§3.3) or change the
sentence to state what was actually done. Leaving it is not an option.

**0.2 "adds no learned parameters" (abstract, results, conclusion) is
false.** `modelutils_ltp.py` builds the edge MLP as
`nn.Linear(2*n_features + n_edge_features + n_global, n_hidden)`, and
`n_edge_features` comes straight from `graph_metadata['num_edge_features']`.
Adding relation types widens the edge-feature indicator, which widens that
first layer. The chain layer therefore *does* add learned parameters — just
no *domain-specific* ones, and no new supervision. The reviewer flagged this
(W10) and is correct. Reword to "no new supervision and no domain-sized
parameters", and report the actual parameter counts for BIND vs GADAR.

---

## 1. What the review is actually asking for

Three reasons for the score, in the reviewer's own order of weight:

1. **C2 has no supporting experiment.** Both ladder steps bundle several
   changes, so neither binding nor chain edges is isolated. The 4%→68%
   result has a specific competing explanation: `E_eff` plus the
   goal-relevance bit encode "this object closes an open goal" one hop from
   the ranked quantity, and on Visitall — one movement schema, goal = a set
   of visited atoms — that is very nearly a policy by itself.
2. **Checkpoint selection uses the held-out domain**, so the headline
   numbers are oracle-selected best-of-N, not zero-shot.
3. **The empirical base is too narrow**: three of eight folds, one success,
   two near-failures, no dispersion, no instance counts, no error analysis,
   no comparison to the domain-independent heuristic line.

Everything else in the review is secondary to these.

---

## 2. Tier 1 — cheap, and each removes a specific objection

No training. Days, not weeks.

| # | Item | Work | Removes |
|---|---|---|---|
| 1.1 | Correct the 3-seed sentence | text | §0.1 |
| 1.2 | Reword the parameter claim + report BIND/GADAR parameter counts | text + one print | W10 |
| 1.3 | **Re-report zero-shot under training-domain checkpoint selection** | eval runs only | **W2 (essential)** |
| 1.4 | State the supported PDDL fragment in Problem Setup | text | W11 |
| 1.5 | Specify what the achievement oracle defers to | text | W8 (half) |
| 1.6 | Instance counts per split; consistent decimals; dispersion where it exists | text | W9 (half) |
| 1.7 | Untrained control as a full column in Table 1; pure-greedy numbers promoted from prose | table | W12/#14 |
| 1.8 | Per-decision wall clock + mean applicable groundings per state, vs the planner | already in the results JSONs and PERFORMANCE.md | W6 |
| 1.9 | **Random-policy floor beside every zero-shot number** (§9) | already computed | the first control a reviewer runs |

**1.3 is the highest value-per-hour item in the whole plan.** The
checkpoints exist; it is an evaluation pass, not a training run. Run the
combined-loss-selected checkpoint on each held-out split:

```bash
GABAR_BATCH_EVAL=1 GABAR_FEATURIZE_WORKERS=8 \
  python main.py --config configs/loo8_joint_chain_no_visitall.yaml \
    --mode test --device cuda:0 --test-model-metrics combined --num-models-to-test 1
```

Report both numbers and the gap. The gap is itself a finding: if it is
small, the objection evaporates; if it is large, that is worth knowing
before a reviewer discovers it.

For 1.8, the per-state inference times are already in every
`cache/results/<expid>/results_*.json`, and PERFORMANCE.md has the measured
harness timings. What is missing is *mean applicable groundings per state* —
one counter in the featurizer — and the planner-side comparison.

---

## 3. Tier 2 — the de-confounding ablation (the decisive one)

### 3.1 New featurization modes

The pieces are already separable in `ploi/lifted_layer.py`:
`build_chain_spec` / `schema_goal_distances` (chain edges),
`add_chain_node_features` (goal distance + goal relevance),
`add_grounded_effect_edges` (`E_eff`), and binding edges inside
`_get_precondition_satisfaction_position`. What is missing is the gating.

Add modes, not a fork — the repo's one-featurizer discipline is why the
ladder is runnable as configs at all:

| mode | contents | answers |
|---|---|---|
| `joint_chain_only` | BIND + `E_chain`, no `E_eff`, no gdist, no goal-relevance | does the chain layer alone produce the jump? |
| `joint_goal_only` | BIND + `E_eff` + gdist + goal-relevance, no `E_chain` | do the goal-directed features alone produce it? |
| `joint_nobind` | structural + lifted layer, no `E_bind` | is binding a distinct contribution? |

Implementation is four booleans derived from the featurization name,
threaded where those four functions are called. Estimate: one day, plus a
renaming-invariance test per mode (the existing unit test extends).

**Pre-register the reading before running** (research skill §3):

- 68% survives `joint_chain_only` and drops under `joint_goal_only`
  → C2 established as claimed; the paper's story stands.
- 68% survives `joint_goal_only`
  → the chain layer is *not* what carries Visitall. Restate the
  contribution in terms of goal-directed grounded features. This is a
  finding, not a failure, and it is better to find it than to have a
  reviewer suggest it twice.
- Both partial → the components interact; report the 2×2 and say so.

### 3.2 Runs

Minimum for a rebuttal, Visitall fold only: **3 runs**
(`joint_chain_only`, `joint_goal_only`, `joint_nobind`), one round at three
concurrent, roughly 30 h on current hardware.

Full de-confounding across all three reported folds: **9 runs**, three
rounds, ~90 h.

### 3.3 Seeds (W9, and §0.1)

Three seeds everywhere is 27+ runs and does not fit. Proposal: **three seeds
for the three rungs on the Visitall fold** (9 runs), report dispersion
there, and state plainly that other cells are single-seed. That is honest,
affordable, and covers the cell the paper leads with — which is exactly
where a reviewer will ask whether 68 vs 4 is bigger than seed variance.

### 3.4 Error analysis (W7)

Instrument the executor to record a termination reason per rollout:
top-ranked action inapplicable / dead end / step bound / revisit thrash,
plus top-1 applicability rate per rung per split (currently reported only
for UNION). For Miconic specifically, plot success rate against expert plan
length — that tests the compounding-error hypothesis the paper asserts but
never measures.

This overlaps with the decoder diagnostic already discussed separately
(per-argument-position accuracy, teacher-forced vs free-running). Build the
termination counters first; they are cheaper and answer the reviewer
directly.

---

## 4. Tier 3 — breadth and framing

**4.1 Complete the leave-one-out matrix (W4).** Five unreported folds.
Full breadth is 15 runs (5 folds × 3 rungs); prioritize **GADAR only on all
five** (5 runs, ~60 h) and add UNION/BIND for any fold where GADAR shows
signal. A reader currently cannot tell whether Visitall is representative or
exceptional, and that single question carries a lot of the reviewer's
scepticism.

**4.2 Structural similarity between held-out and training domains (W3).**
The reviewer's sharpest analytical point: Grid and Logistics remain in
training and both contain movement schemas close to Visitall's, so the
"unseen domain" may be a near-duplicate of seen structure. This is cheap to
measure — the lifted spec is already built per domain, so compare schema
signatures, arity profiles, and chain patterns across folds — and it turns
a weakness into a contribution: a predictor of *when* transfer occurs.
Recommended even under a tight budget.

**Now motivated by a result, not just by the reviewer.** Controlled
comparison on Logistics, 2026-09-05, same fold (no_miconic), same 20
problems, in-domain for both:

| featurization | epoch | V1 | coverage |
|---|---|---|---|
| union | 490 | 30.6% | **40.0%** |
| joint_chain | 470 | **40.4%** | **0.0%** |

joint_chain constructs valid actions *more* often than union and solves
none. So Logistics' persistent 0% is **not** a construction failure — the
decoder can build the 4-parameter groundings. It is a ranking failure: the
actions are applicable and not goal-directed.

The hypothesis that fits, and that the theory already predicts, is
**aliasing**. Logistics is the suite's worst case: `load-truck(?obj ?veh
?loc)` / `load-airplane(?obj ?veh ?loc)` are structurally isomorphic, as
are the two unloads — two transport modalities that differ only in symbol
identity, which is exactly what union keeps and structural features erase
by construction (CLAUDE.md §7: aliasing is a theorem, not a bug).

The whole `all8_joint_chain` column ranks like "how many isomorphic schema
pairs does this domain have":

    gripper 98.8   grid 97.9   visitall 92.0    <- few or none
    miconic 63.0                                <- up/down
    manyblocks 21.5   rovers 9.3   spanner 1.0  <- pairs
    logistics 0.0                               <- two full modalities

**The schema pairings above are from memory and MUST be verified against
the PDDL files before any of this is written up.** The measurement is the
4.2 analysis itself: count structurally isomorphic schema pairs per domain
from the lifted spec, correlate against that coverage column. Zero GPU. If
it holds, 4.2 stops being a defensive answer to W3 and becomes a predictor
of when the method works — which is a contribution, and the honest framing
of the ladder's one clear in-domain loss to UNION.

Two corollaries to carry into the error analysis (3.4):

- **V1 is not a quality proxy.** Here the two metrics anticorrelate. Report
  both, never V1 alone.
- **Aliasing is not a data problem.** If two schemas are structurally
  indistinguishable, more examples of both supply contradictory targets for
  identical inputs. This is independent of the generalization gap measured
  below, and no amount of data addresses it.

**4.3 The GOOSE baseline question (W5).** Running GOOSE's DI heuristic with
greedy hill-climbing is a real integration project. The argument is cheap
and, I think, correct: greedy hill-climbing evaluates *every* applicable
successor per step (|A| forward passes plus |A| successor generations),
while GADAR constructs one action from one graph. That is a genuine
computational distinction and it is measurable — so **pair the argument with
the Tier-1.8 cost numbers** rather than asserting it. If a number can be
obtained, the reviewer rates it the most valuable addition after the
de-confounding ablation.

**4.4 Presentation (W12).** Table 1 split column out of the domain column;
untrained control as a real column (1.7); a reduced Figure 1 showing only
binding and chain edges; thin the figurative prose where it slows a
technical sentence.

---

## 5. Tier 4 — if time allows

**5.1 A formal statement for the chain layer (review #12).** Two tasks
indistinguishable to L rounds of message passing without `E_chain` and
distinguishable with it. The reviewer calls it "a small amount of work"
that would put the central intuition on the footing of GOOSE's Theorem 4.3,
which the paper already invokes for the analogous claim. Worth it only
after §3 shows the chain layer is in fact what matters — proving a property
of a component that turns out not to carry the result would be wasted.

---

## 6. Budget

Assuming ~30 h per 500-epoch 8-domain run, three concurrent per node, two
nodes.

| Package | Runs | Wall clock (2 nodes) | Priority |
|---|---|---|---|
| Tier 1 (all) | 0 training, ~9 evals | ~1 day | do first |
| 3.1–3.2 de-confounding, Visitall only | 3 | ~1 day | essential |
| 3.3 seeds, Visitall, 3 rungs | 9 | ~2 days | essential |
| 3.2 de-confounding, all 3 folds | +6 | ~1 day | strong |
| 4.1 remaining folds, GADAR only | 5 | ~1 day | strong |
| 3.4 error analysis | 0 training | ~2 days code | strong |
| 4.2 structural similarity | 0 training | ~1 day | cheap, high value |

**Roughly a week of compute and a week of code** covers Tier 1, Tier 2 and
the cheap half of Tier 3 — which is precisely the set the reviewer says
would move the paper to 5–6.

---

## 7. Paper changes independent of any experiment

1. **Stop leading with 68%.** Abstract, introduction and conclusion all
   generalize from one cell. State the range across measured folds and let
   Visitall be the best case rather than the characterization.
2. **Say what the ladder does and does not isolate.** The paper asserts
   "adjacent systems differ in exactly one component" three times, and it is
   not true. Either the new modes make it true, or the text says which steps
   are bundles. Do not leave the assertion standing.
3. **Fix §0.1 and §0.2.**
4. **PDDL fragment in Problem Setup**, not in the last sentence: negative
   preconditions, conditional effects, equality, constants, derived
   predicates, and whether any of the eight domains needed preprocessing.
5. **Instance counts and consistent precision** in both tables.
6. **Frame the "first domain-agnostic policy" claim** as being about the
   output type (constructing grounded actions without search) and defend it
   against the greedy-heuristic reading explicitly (4.3).

---

## 8. What not to do

- **Do not chase the per-domain GABAR gap.** The reviewer never asks for
  it, and the paper is already right that a single multi-domain model and
  eight per-domain models are different artifacts.
- **Do not add domains.** Breadth within the existing eight (4.1) is what is
  asked for; new domains would restart the cache and answer nothing the
  reviewer raised.
- **Do not soften the capability claim.** C1 is accepted and called the
  strongest part of the paper. The revision is about C2 and C3.
- **Expanding the training set is licensed by a measured generalization
  gap — size it with the data-fraction curve first.** Status revised
  2026-09-05 from *deferred* to *licensed, unsized*. Normalizing the
  summed train/val losses by the epoch-0 ratio (at initialization the
  model is identical on both sets, so that ratio is purely how much is
  being summed), per-example validation loss is **4–12x** training loss on
  every headline run:

      no_manyblocks 10.9x   no_miconic 6.9–12.0x   sweep base 8.7x
      no_visitall 6.6x      no_grid 5.3x           no_spanner 4.3x
      no_gripper 3.8x       sweep drop01 2.8x      sweep l2 3.0x

  The two regularized sweep arms have the smallest gaps and `sweep_jc_l2`
  has the **lowest absolute validation loss in the suite** (12.56 vs base
  16.41, −23%) — independent corroboration that the model is data-limited,
  and a trained checkpoint nobody has evaluated yet. Evaluate it before
  spending anything on data (`eval_queue.sh configs/sweep_jc_l2.yaml
  configs/sweep_jc_base.yaml`); lower validation loss does not imply better
  coverage, and best-loss selection has already discarded the
  best-transferring checkpoints once in this project.

  **Adding capacity now would make this worse, not better.** With a 4–12x
  gap the model is memorizing; a wider or deeper net memorizes harder. Data
  first, capacity after the gap closes.

  Two things the gap does NOT cover: validation here is a stratified 10% of
  *states* from the same problems, so it measures generalization to new
  states, not new problems and not new domains; and **aliasing (§4.2) is
  orthogonal to it** — where two schemas are structurally indistinguishable
  more data supplies contradictory targets for identical inputs.

  Size the expansion with the curve (RUNBOOK, node M day 2) — two short
  trainings, not a dataset build. The slope from 25→50→100 says whether
  the target is 2x or 10x:

      for n in 25 50 100; do
        RUN_TAG="d$n" ./train_test_scripts/run_config.sh \
          configs/<arm>.yaml cuda:0 --num-train-problems $n --mode train
      done

  `d` is already in the checkpoint key, so each fraction gets its own
  directory. Read the slope at the full count: still climbing steeply →
  a large expansion is worth it; already bending → a modest one, or better
  regularization, captures most of what is available. (A flat curve would
  contradict the measured gap and means something else is wrong — read it
  as a signal to check the split, not as licence to skip the data work.)
  Budget a data-collection pass per fraction (new unified caches and
  sidecars; the full-size ones are not reused).

  Two prices, both of which the reviewer will check:

  1. **Comparability.** Training GADAR on more data than the per-domain
     GABAR baselines makes the comparison unfair in GADAR's favour. The
     fix is retraining GABAR on the same expanded set — eight trainings,
     which is the real cost of the expansion, not an afterthought.
  2. **Bigger instances contaminate the claim** (status
     *excluded-by-design* for the headline runs). "Trains on small
     instances, generalizes to much larger" is the setup; training on
     larger instances weakens it, and the training plans come from
     satisficing `fd-lama-first`, whose plan quality degrades with size —
     so the supervision itself gets worse exactly where the instances get
     bigger. If run at all, run it as a separate labelled arm, never folded
     into the headline numbers — and not on Logistics, whose 0% is an
     aliasing failure (§4.2) that instance size does not touch.

---

## 9. FIXED: the batched evaluator crashed on mixed arity

Diagnosed and fixed 2026-08 from `logs/loo8_union_no_miconic_run.log`. Kept
here because it explains missing results, and because the workaround below is
still the fallback if the fix is ever suspected.

```
ploi/modelutils_ltp.py, beam_search_parallel -> get_best_object_embeddings_ltp
RuntimeError: The size of tensor a (200) must match the size of tensor b (67)
```

`beam_search_parallel` selects one parameter row per graph with

```python
parameter_locations = torch.arange(parameter_number,
                                   all_objects_batches_all_params.shape[0],
                                   self.max_number_action_parameters)
```

which is correct only if every graph contributes exactly
`max_number_action_parameters` rows.
`get_best_action_object_scores_locations` documents the same assumption
(`ao_scores: [batch_size * max_params, max_nodes]`). When a batch mixes
arities the row count is not `number_graphs * max_params`, the stride returns
the wrong number of graphs (67 against 200 here, consistent with stride 3
over 201 rows), and the per-graph offsets built from `n_node` no longer
align.

**Fix applied.** Both places now derive the row->graph map from
`n_parameters`, the same arithmetic `compute_object_scores` has always used:
`get_best_action_object_scores_locations` takes an optional `n_parameters`
(omitting it keeps the legacy map, so the training path and `ablations.py`
are untouched), and `parameter_locations` uses cumulative offsets with a
per-graph clamp instead of a fixed stride. `beam_search_parallel` also checks
`sum(n_parameters) == ao_scores.shape[0]` once at entry, so a future
disagreement names itself rather than surfacing as a broadcast error deep in
the loop. `beam_search_v2` is deliberately untouched - it is the parity
reference.

Pinned by `tests/test_beam_parallel_indexing.py`: the mixed-arity map is
exact, a uniform batch is byte-identical to the legacy path (so no
previously-working run moves), the row-count disagreement raises with both
numbers, and the failing batch recovers 200 graphs where the old stride
recovered 67.

### How wide the blast radius was

`n_parameters` is the graph's own `max_action_arity`. During TRAINING
`pad_pyg_action_scores` pads every graph to the global max, so all graphs
agree and the old stride was right. At TEST time only one domain is
featurized and nothing pads, so `n_parameters` is that domain's arity while
`max_number_action_parameters` came from the merged training action space.
The stride was therefore correct only when
`test_domain_max_arity == model.max_number_action_parameters` - i.e. the
crash hit every evaluation of a multi-domain model on a narrower domain,
including most zero-shot cells (the held-out domain is usually narrower than
the union of seven training domains). This is not one broken config; it is
why a large part of the LOO grid has checkpoints and empty results.

### Verified 2026-09-05 (cn-gpu5, H100, union no_miconic, epoch 490)

- **Crash closed.** `--test-domains visitall_ipcc:20` (arity 1, cap 3):
  18/20 solved, no error. On `07c19b1` the same command raises
  `size of tensor a (20) must match the size of tensor b (7)` - 20 graphs,
  stride 3 -> 7, the same signature as the original 200/67. The regression
  test is real, not vacuous.
- **Correctness, not just non-crashing.** `tools/parity_matrix.sh` on the
  same config, all 8 cells ({cpu,gpu} x {det,nodet} x {seq,batch}): outcome
  matches AND *SAME actions*. `beam_search_v2` decodes one graph at a time
  with no stride, so agreement with it certifies the new row->graph map.
- **No previously-working case moved.** `logistics_ipcc:20` (arity == cap):
  40.0%, V1 30.6%.
- **No inference cost.** `configs/ab_visitall.yaml`, `visitall_ipcc:50`,
  batched, 8 workers, before vs after: `forward pass` 93.39s -> 93.02s
  (0.4%, noise), total 255.2s -> 249.3s, coverage 35/50 both, plan quality
  identical to 16 digits (0.9116086450110004) - i.e. byte-identical
  decisions, as `test_uniform_case_matches_legacy` predicts. The 2% total is
  `graph build` + `pyg convert`, which the change does not touch.

**Fallback if it is ever suspected again:** drop `GABAR_BATCH_EVAL=1`, which
routes through `beam_search_v2` and never calls the affected code. ~2.5x
slower, same outcomes.

Cost: the eval phase of every completed training whose test domains are
narrower than its training set - at least `loo8_union_no_miconic` and
`loo8_joint_lite_no_miconic`, and every zero-shot cell of the LOO grid.
Checkpoints survive, so these are re-runnable with `--mode test`; no
retraining is needed. Every `--mode test` invocation in
`tmp_scripts/queue_runner.sh` sets `GABAR_BATCH_EVAL=1`, so the whole queue
was exposed.

### Resolved: the first zero-shot number the fix unblocked

`union no_miconic` zero-shot on miconic scores **0%, V1 0.0%, every problem
dead at round 1** — the executor's `if not valid_actions: return True`, i.e.
the whole beam was inapplicable.

Two follow-ups settled what that means:

1. **It is a policy result, not a pipeline fault.** With
   `GABAR_CONSTRAINED_DECODE=1` (each parameter slot restricted to objects
   some applicable grounding puts there, sequential path only): **V1 100%,
   coverage still 0%**. Groundings are enumerated and legal actions are
   decodable on the unseen domain; handed only legal actions, the union
   model still cannot order them. That is the C1 control, stated precisely:
   it fails at *constructing* legal actions, and also at ranking them once
   construction is free.
2. **But the cell has no discriminative power.** The random-policy floor on
   `Miconic_ipcc@test` is also **0.00%**. 0 against 0 separates nothing; the
   union control's zero here is not evidence that anything else is better.

### Every zero-shot number must be reported against its split's random floor

`cache/results/random_floor_summary.txt` (uniform choice among applicable
ground actions, same executor/monitor/step bound):

| domain | test | train-split |
|---|---|---|
| Visitall | 6.00% | **99.73%** |
| Miconic | 1.12% | **88.30%** |
| Rovers | **30.86%** | 75.21% |
| Spanner | **43.75%** | 63.39% |
| Grid | 0.00% | 2.95% |
| Manyblocks | 0.17% | 1.67% |
| Gripper | 0.00% | 0.00% |
| Logistics | 0.00% | 0.00% |

Consequences, both directions:

- **The `@train` splits are the dangerous ones.** The suite's zero-shot
  cells include `miconic_ipcc@train:228`, whose floor is 88.3%, and
  visitall's is 99.7%. A zero-shot number below the floor on those splits is
  *worse than random* however large it looks in isolation, and a reviewer
  who runs the obvious control will find it. Nothing on an `@train` split
  should be reported without the floor beside it.
- **The discriminative cells are the hard test splits** with a floor at or
  near zero: Gripper, Logistics, Miconic, Grid, Manyblocks. That is where a
  nonzero GADAR/GADAR-BIND number carries the C1/C2 claim.
- Rovers and Spanner test sit in between (30.9% / 43.8%) — beatable, but the
  margin over the floor is the result, not the raw coverage.

Add a random-floor row (or column) to Table 1. It costs nothing, it is
already computed, and it converts "our zero-shot coverage is X%" into a
claim that survives the first control a reviewer thinks of.

The miconic test floor is **1.12%** at the full 119 problems (the 5-problem
probe said 0.00%; a 5-problem floor is not a reportable number). It stays a
near-zero, discriminative cell — and 1.12% rather than 0% is the honest bar
the union control's 0% has to be stated against.
