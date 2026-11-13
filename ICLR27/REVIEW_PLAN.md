# ICLR 2027 reviews: what they found, and the experiments that answer them

Reviews: 1 / 1 / 2 (HhM5, QEpz, 9YTx). This file is the plan; numbers go in
`train_test_scripts/RUNS_STATUS.md` as they land. Written 2026-11-11.

## 1. What the three reviews agree on

| # | Point | Who | Answered by |
|---|---|---|---|
| A | The components are never separated: which of enables/threatens, achieves/deletes, goal distance/relevance, binding edges matters? | QEpz, 9YTx | E2 |
| B | A non-learned policy ranking by the hand-coded goal features alone might match GADAR | QEpz | E0 |
| C | In-domain numbers are far below published GABAR, even single-domain (Logistics 3%, Rovers 7%); is the pipeline sound? "Re-run GABAR on your split with your pipeline" | QEpz | E1 |
| D | Does training on LAMA-first (suboptimal) plans matter? | HhM5 | E1 |
| E | No comparison with heuristic search (GOOSE, a greedy heuristic policy, LAMA) | all three | E0 (LAMA, timing), E3 (stretch) |
| F | UNION "trained for 100 epochs vs 500" | QEpz | text error, §2 |
| G | Method not self-contained (loss, decoder, arity), non-standard terminology | HhM5 | rewrite, §5 |

## 2. Errors in the submitted paper that we found while planning

These are ours to fix regardless of any experiment.

1. **The "GADAR-BIND" rung that was run has no binding edges.** The paper
   says GADAR-BIND adds occurrence nodes and `E_bind`. The leave-one-out
   campaign ran it as featurization `joint_lite`, which has predicate-symbol
   nodes, instantiation edges and role edges only: no occurrence nodes, no
   `E_bind`. (The repo used "GADAR-BIND" for "GADAR minus binding";
   `configs/README.md` said so.) So the in-domain BIND row and the
   current-model zero-shot BIND cells are a *no-binding* system, and the
   BIND -> GADAR step bundled occurrence nodes, binding, chain edges, effect
   edges and goal features. The rung the paper describes is `joint`. The
   three Visitall/Miconic/Grid BIND cells taken from the AAAI table were
   `joint` runs, so that row mixed two systems.
2. **UNION was not trained for 100 epochs.** Every `loo8_union_*` config
   inherits `epochs: 500` from `_common.yaml`. The appendix table line
   "Epochs, UNION: 100" was stale.
2b. **CONFIRMED 2026-11-12: single-effect schemas were dropped by the
   featurizer.** pddlgym leaves a schema with exactly one effect literal as a
   bare `Literal` (no `.literals`), and `lifted_layer` and `structural.py` read
   effects with `getattr(op.effects, "literals", [])`, which turned that
   effect into "no effects": no add/delete occurrence, no `enables` edge into
   or out of the schema, no `achieves_goal`/`deletes_true` edges. A diagnostic
   on the cluster found exactly two such schemas in the eight domains:
   **Miconic `board`** (the paper's own board -> depart example) and **Rovers
   `calibrate`** (it produces what `take_image` needs). Those are two of the
   weakest in-domain results (GADAR Miconic 61.9, Rovers 9.0; single-domain
   41.2 and 7.4, against GABAR 99.2 and 82). Logistics (3%) is NOT explained
   by this. UNION and GABAR do not use the lifted layer and are unaffected.
   Preconditions are read directly and have never crashed: they are
   conjunctions.

   The fix (`structural.op_literals`) changes the graphs of Miconic and Rovers
   only; the other six domains are byte-identical (`gate_fingerprint` still
   passes on Grid/Gripper/Logistics/Visitall). Because the same config name now
   yields different graphs, two things are keyed so old and new can never mix:
   the checkpoint key gains `fx` (1 when the training set contains such a
   domain and the featurization is lifted/structural; defaulted away at 0, so
   existing directories keep their names), and the Miconic/Rovers graph
   sidecars get a `_fx1` tag. **Every September GADAR/BIND model whose
   training set contains Miconic or Rovers (all 16 leave-one-out models, plus
   the single-domain Miconic and Rovers ones) was trained on the old graphs
   and must be retrained for any number that goes in a paper.** The old
   models remain evaluable only with the old code (commit `4ee1b6d`).
   E1 now includes the two single-domain runs that test the fix directly
   (`e1_sd_miconic_fixed`, `e1_sd_rovers_fixed`); E2 includes the full
   model and `joint_lite` retrained on the fixed featurizer as the reference
   for its ablations. The full 8-fold, 2-rung retrain of the paper ladder is
   not in the manifest: it is a decision (about 16 trainings x 14 h).
2c. **FOUND 2026-11-13: a checkpoint best under two metrics left an empty
   cell, and the in-domain table filled it from an older dump.** `run_tests`
   skipped a checkpoint "already tested" under an earlier metric of the same
   run, so with `METRICS=training,combined,validation` the `combined` cell of a
   domain was empty whenever the best-combined epoch was also the
   best-training one. `eval_status.py` still called the config done, and
   `indomain_table.py` ("newest dump that HAS the cell") then read `combined`
   from any older dump - during the September crunch, possibly an evaluation
   of a half-trained model. Whether a reported Table 2 cell was affected is
   NOT known: it depends on which dumps exist on the cluster. Fixed in two
   places: `run_tests` records the result under every metric that selected the
   checkpoint, and `indomain_table.py` recovers the cell from the same dump
   (and prints `NOTE STALE ...` where it cannot). Re-run
   `python tools/indomain_table.py` on the cluster and compare with the
   submitted Table 2 before reusing any of its numbers. Zero-shot cells are
   not affected (one metric per domain).

2d. **Audit of the post-July code, 2026-11-13 (what was checked, and how).**
   Verified against real pddlgym operators with `tools/audit_featurizer.py`
   (independent PDDL reader; IPC-style Miconic, typed Gripper, predicate-typed
   Logistics, Blocks with a zero-arity predicate, a Spanner type hierarchy):
   after the 2b fix the lifted graph matches the text in every checked edge
   and node class; the pre-fix code fails with 101 mismatches. Read and found
   correct: delete effects (`is_anti` is a real Literal attribute), success
   accounting (goal reached in the real env; denominator = all problems),
   best-checkpoint ordering, seed separation of model directories, the
   object-count bookkeeping with lifted nodes present. Limits of the
   representation, reported by the audit as WARN and NOT in the eight
   domains as far as the text of the fork is known: a parameter that
   appears in no precondition gets no candidate edge (typed Gripper
   `move ?to`). A negative precondition was encoded as a positive one, which
   inverted the chain edges through it: fixed 2026-11-14 (2e). Not audited to the same standard: the model and training
   loops on mixed-domain batches (read, not executed - no torch in the audit
   environment); the September mixed-arity fix and its unit test are the
   evidence there.
2e. **FIXED 2026-11-14: negative preconditions lost their sign in the lifted
   layer.** `(not (p ?x))` was recorded as a precondition on p, so "A adds p"
   was drawn as ENABLING a schema it disables, "A deletes p" as threatening
   one it enables, and the goal-distance buckets followed the wrong relation.
   GABAR and UNION were never affected (their feature is named `Notp<pos>`);
   `structural` only records applicability. Now role `npre`, encoded with the
   existing columns, so no feature width changes. Verified with real pddlgym:
   50 graph fingerprints (5 domains x 10 lifted modes) byte-identical before
   and after, only the negative-precondition domain changed; the audit passes
   on it and reports 281 mismatches for the previous code. Consequence for
   the campaign: NONE unless `tools/audit_featurizer.py` prints a featurizer
   version with bit 2 for one of the eight domains - then that domain's
   sidecar and every model trained on it are renamed automatically (`fx`) and
   retrain.
3. **The zero-shot table mixed two model generations** (July models for
   Visitall/Miconic/Grid, September models elsewhere) selected by different
   rules. The redo reports one protocol for every cell (§4).

## 3. Experiments, in priority order

Each has its readings written down before it runs (research skill §3).

### E0. Free or CPU-only (hours)

**E0a. Rule baselines under the same executor** (answers B).
`tools/random_policy_baseline.py --policy features` ranks applicable actions
by: achieves an open goal atom > does not delete a satisfied goal atom >
schema goal distance > goal-relevant arguments, ties random. `--policy
achieve` is the achievement oracle as a policy.

```bash
D=manyblocks_ipcc_big:200,gripper_ipcc:173,miconic_ipcc:119,visitall_ipcc:50,grid_ipcc:48,logistics_ipcc:96,spanner_ipcc:96,rovers_ipcc:54
T=manyblocks_ipcc_big@train,gripper_ipcc@train,miconic_ipcc@train,visitall_ipcc@train,grid_ipcc@train,logistics_ipcc@train,spanner_ipcc@train,rovers_ipcc@train
for P in features achieve; do
  python tools/random_policy_baseline.py --policy $P --domains $D,$T
done
cat cache/results/features_floor_summary.txt cache/results/achieve_floor_summary.txt
```

Readings:
- `features` >= GADAR zero-shot on a domain -> on that domain the network
  adds nothing to the hand-coded features. Report it, and do not claim
  learned transfer there.
- `features` clearly below GADAR -> the gap is what was learned from the
  other seven domains; this is the control QEpz asked for.
- `features` high on a domain where GADAR is ~0 (plausible on Gripper,
  Logistics) -> the information is in the graph and the network fails to use
  it zero-shot: a learning problem, not a representation problem.

**E0b. LAMA-first on the test sets, with time** (answers part of E). Already
in every results dump (`nonopt` column of `tools/analyze_results.py`):
coverage 100%. Add the planner's and the policy's wall-clock per problem
from the same dumps; that comparison is the honest answer to "why a policy".

**E0c. Gates before any ablation training.**

```bash
python tools/check_ablation_modes.py --domain Manyblocks_ipcc_big
python tools/check_ablation_modes.py --domain Logistics_ipcc
# byte-identity of the unchanged modes across the refactor (CLAUDE.md §1.1):
python tools/graph_fingerprint.py --help    # record at 82b901f, record at HEAD, --compare
for t in tests/test_*.py; do python $t >/dev/null 2>&1 && echo "ok $t" || echo "FAIL $t"; done
```

### E1. Is the in-domain gap the pipeline, the training data, or the representation? (~1 GPU-day; answers C, D)

Published GABAR trained on optimal plans; this suite trains everything on
`fd-lama-first`. Until 2026-11-11 the plan cache and the checkpoint key did
not record the planner, so the flag could not even be varied safely; both do
now (`_planner_tag`, key `tp`).

```bash
# September left three interrupted GABAR runs behind (blocks ~430/500, grid
# ~270/500, gripper ~10/500 epochs). Look at them, then remove those three
# directories so the reruns do not share loss-ranked slots with them:
ls -d models/*bs16* | grep -E "/(Manyblocks_ipcc_big|Grid_ipcc|Gripper_ipcc)_seed"

# GABAR through our pipeline, our test instances, LAMA plans (Miconic is done):
nohup ./train_test_scripts/train_queue.sh cuda:0 configs/baseline_{logistics,rovers,blocks,gripper}.yaml > logs/q_gabar_a.log 2>&1 &
nohup ./train_test_scripts/train_queue.sh cuda:0 configs/baseline_{visitall,grid,spanner}.yaml > logs/q_gabar_b.log 2>&1 &
# the two failing domains on OPTIMAL plans, GABAR and single-domain GADAR:
nohup ./train_test_scripts/train_queue.sh cuda:0 configs/baseline_{logistics,rovers}_opt.yaml configs/sd_joint_chain_{logistics,rovers}_opt.yaml > logs/q_opt.log 2>&1 &
# then, on a CPU-rich node:
REDO=1 METRICS=combined NMODELS=1 ./train_test_scripts/eval_all.sh configs/baseline_*.yaml configs/sd_joint_chain_*_opt.yaml
python tools/analyze_results.py 2>/dev/null | grep -E "^(baseline_|sd_joint_chain_.*_opt)"
```

Watch the `Planning failed for problem N` count in the `_opt` logs: lmcut
solves fewer training instances than LAMA, and a domain that loses most of
its training set answers a different question.

Readings (Logistics; Rovers likewise):

| GABAR, LAMA plans | GABAR, optimal plans | Conclusion |
|---|---|---|
| ~79 | - | pipeline and data are fine; GADAR's 3% is the representation. Go to the aliasing analysis. |
| low | ~79 | suboptimal training plans are the cause. Retrain the suite on optimal plans wherever lmcut covers the training set; every in-domain number is expected to move. |
| low | low | our pipeline does not reproduce published GABAR on this domain. Stop and debug (test set, hyperparameters, plan-length bound) before any other experiment is trusted. |

Miconic already reproduced (99.2% through our pipeline, 2026-09-26).

### E2. Component ablations (5 GPUs x ~2 days; answers A)

Modes (`ploi/lifted_layer.py::_MODE_COMPONENTS`), all at the `joint_chain`
feature widths so the network is identical and only graph content differs:

| featurization | bind | chain | eff | goal | role |
|---|---|---|---|---|---|
| `joint_chain` | x | x | x | x | full (trained) |
| `jc_no_bind` | | x | x | x | "GADAR without binding edges" (QEpz) |
| `jc_no_chain` | x | | x | x | minus enables/threatens |
| `jc_no_eff` | x | x | | x | minus achieves/deletes |
| `jc_no_goal` | x | x | x | | minus goal distance/relevance |
| `joint` | x | | | | the rung the paper calls GADAR-BIND |
| `jc_only_{chain,eff,goal}` | x | one | of | three | add-one (second wave) |
| `joint_lite` | | | | | lifted layer only (trained; mislabelled BIND) |

First wave: the four leave-one-out modes + `joint`, on three folds
(held out: Blocks, Visitall, Grid, the folds with any zero-shot signal) =
15 runs. Configs: `configs/loo8_<mode>_no_{manyblocks,visitall,grid}.yaml`.

```bash
# one GPU per line, e.g.:
nohup ./train_test_scripts/train_queue.sh cuda:0 configs/loo8_jc_no_bind_no_{manyblocks,visitall,grid}.yaml > logs/q_abl_bind.log 2>&1 &
nohup ./train_test_scripts/train_queue.sh cuda:0 configs/loo8_jc_no_chain_no_{manyblocks,visitall,grid}.yaml > logs/q_abl_chain.log 2>&1 &
nohup ./train_test_scripts/train_queue.sh cuda:0 configs/loo8_jc_no_eff_no_{manyblocks,visitall,grid}.yaml > logs/q_abl_eff.log 2>&1 &
nohup ./train_test_scripts/train_queue.sh cuda:0 configs/loo8_jc_no_goal_no_{manyblocks,visitall,grid}.yaml > logs/q_abl_goal.log 2>&1 &
nohup ./train_test_scripts/train_queue.sh cuda:0 configs/loo8_joint_no_{manyblocks,visitall,grid}.yaml > logs/q_abl_joint.log 2>&1 &
```

Evaluation, per finished config (REDO=1: eval_status only knows the paper rungs):

```bash
# in-domain + zero-shot at the lowest-loss checkpoint
REDO=1 METRICS=combined NMODELS=1 ./train_test_scripts/eval_all.sh configs/loo8_jc_*_no_*.yaml configs/loo8_joint_no_{manyblocks,visitall,grid}.yaml
# zero-shot at every periodic snapshot, both splits
REDO=1 ZERO_SHOT_ONLY=1 METRICS=periodic NMODELS=12 EXPID_SUFFIX=_epochs ./train_test_scripts/eval_all.sh configs/loo8_jc_*_no_*.yaml configs/loo8_joint_no_{manyblocks,visitall,grid}.yaml
python tools/indomain_table.py --rungs joint_chain,jc_no_bind,jc_no_chain,jc_no_eff,jc_no_goal,joint,joint_lite --folds manyblocks,visitall,grid
python tools/crossfold_select.py                 # test instances
python tools/crossfold_select.py --split train   # training-scale instances
```

Two statistics per mode, both selection-free:
- **in-domain** mean coverage over the seven training domains (stable;
  ladder gaps there are 9 points);
- **zero-shot** *mean over all periodic snapshots* (`mean` column of
  `crossfold_select.py`). Single-checkpoint zero-shot coverage moved between
  0 and 74 within 50 epochs in September, so no single-checkpoint number
  can carry an ablation.

Readings:
- removing a component costs nothing on either statistic -> it is not a
  contribution; cut it from the method and the claims.
- only `jc_no_goal`/`jc_no_eff` hurt -> the gain is the hand-coded goal
  lookahead, as QEpz suspects. Combine with E0a: if the rule baseline also
  matches, the paper's claim reduces to those features.
- `jc_no_bind` hurts -> binding is a contribution, with the first evidence
  for C2 this project has.
- nothing separates beyond seed noise -> say so; add a second seed for the
  full model and the most-different ablation before reading anything.

A difference under ~5 points with one seed is not a finding.

### E3. Stretch: learned-heuristic baseline

GOOSE's domain-independent LLG model + GBFS on the eight held-out test sets.
External code and weights; days, not hours. Only after E0-E2. Without it,
the related-work text must not suggest the comparison was made.

## 4. Reporting protocol for the next version

- One model generation, one selection rule, every cell.
- Zero-shot: mean over periodic snapshots, test and training-scale splits,
  beside random and the two rule baselines.
- In-domain: lowest combined loss, beside GABAR through our pipeline on the
  same instances (E1), not the published numbers.
- Rungs named for what they contain: `joint_lite` is "lifted layer",
  `joint` is "+ binding", `joint_chain` is the full model.

## 5. Writing (HhM5)

The method section must stand without the GABAR paper: what the decoder
outputs (a distribution over schema nodes, then one over object nodes per
argument), that the loss is cross-entropy over those nodes, that at training
time the argument loop runs to the arity of the expert's schema and at test
time to the arity of the predicted schema. Use the planning community's
terms: goal `G` in the task definition; "add/delete effects", not
"produces/destroys"; "plan" for a sequence, "ground action" for one step;
`a` defined before use. Training data (planner, optimality) belongs in the
method, not only the setup.

## 6. Cut order if GPUs are short

First to go: `jc_only_*`, then E3, then the Grid fold of E2, then GABAR
baselines beyond Logistics/Rovers/Blocks. Never cut: E0a, E1 on
Logistics and Rovers, `jc_no_bind` and `jc_no_goal` on two folds.
