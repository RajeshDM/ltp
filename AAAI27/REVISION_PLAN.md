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

---

## 9. Known issue: the batched evaluator crashes on mixed arity

Diagnosed 2026-08 from `logs/loo8_union_no_miconic_run.log`. **This is a code
bug, so re-running from scratch does not avoid it** - every fresh run hits it
when it reaches its eval phase.

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

**Workaround, available now:** drop `GABAR_BATCH_EVAL=1`. That routes
evaluation through `forward_beam_decode` -> `beam_search_v2`, which never
calls the broken function. ~2.5x slower; the parity matrix already
established the two harnesses agree on outcomes, so the numbers are the same
numbers.

**Fix:** index the parameter rows by per-graph cumulative offsets from
`n_parameters` rather than a uniform stride - the pattern
`compute_object_scores` already uses for `current_parameter_indexes`. It is
inside the batched decoder, so `tools/parity_matrix.sh` against the
sequential reference is the gate afterwards.

Cost so far: the eval phase of at least two completed trainings
(`loo8_union_no_miconic`, `loo8_joint_lite_no_miconic`), which have
checkpoints but no results JSON. Every `--mode test` invocation in
`tmp_scripts/queue_runner.sh` sets `GABAR_BATCH_EVAL=1`, so the whole queue
was exposed.
