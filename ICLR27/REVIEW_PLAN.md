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
