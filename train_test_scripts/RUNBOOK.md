# RUNBOOK — every result the paper needs, prioritized

One place to look up: which runs feed which claim/table, in what order to
launch them, and the exact commands. Updated 2026-07-23 after trimming
(ablation rungs on 4 representative LOO splits; diversity from 2 runs;
all8 endpoints only). Precedent for trimming: GOOSE-DI evaluates on a
SINGLE fixed held-out split — our LOO x8 for the headline systems is
already stricter than the field standard, so subsets for ablations are
normal practice (and the paper says so).

Budget reality: at batch 64 a run is ~8.5 h training + testing (now ~3x
shorter with combined-only selection) → ~1-1.3 runs/GPU/day.

## Results-to-claims matrix (trimmed)

| Paper artifact | Claim | Runs required | Count |
|---|---|---|---|
| Table 2: GADAR + UNION columns, all 8 LOO splits | C1 | `loo8_joint_no_*` x8, `loo8_union_no_*` x8 | 16 |
| Table 2: DOM + BIND columns, 4 splits | C2 + ladder | `loo8_{structural,joint_lite}_no_{manyblocks,miconic,spanner,rovers}` | 8 |
| Table 2: Random floor | C1 integrity | `tools/random_policy_baseline.py` (CPU) | 1 cmd |
| Table 2: GABAR† ceiling | calibration | published numbers | 0 |
| Table 3: price of generality (endpoints) | C1(c) | `all8_union`, `all8_joint` | 2 |
| Diversity figure (4→6→7; Miconic + Spanner curves) | C5 | `ho4_joint`, `ho2_joint_miconic_spanner` | 2 |
| Planner ≈100%, plan-quality refs, inference times | framing | inside every run's results JSON | 0 |
| Renaming invariance | Prop. 2 | `python tests/test_lifted_layer.py` | 0 |

**28 GPU runs for the complete paper. 8 (`loo8_joint`) in flight → 20 to
launch.** The ablation subset {manyblocks, miconic, spanner, rovers} =
canonical / family-transfer / structurally-distinct / largest-vocabulary;
it is named in the paper's Ablations paragraph — keep them in sync if you
swap a domain.

Optional strengtheners, only if GPUs idle (in order of value):
`ho2_joint_blocks_rovers` (adds a Rovers diversity curve),
`all8_structural` + `all8_joint_lite` (fills Table 3's middle columns),
`ho2_joint_gripper_grid` (adds a Grid curve), remaining ho2/ho4.

## Priority order (launch top-down as GPUs free up)

- **P0 — Random floor (CPU, run today, any node, no GPU):**
  ```bash
  nohup python tools/random_policy_baseline.py --domains \
  manyblocks_ipcc_big:200,gripper_ipcc:173,miconic_ipcc:119,visitall_ipcc:50,grid_ipcc:48,logistics_ipcc:96,spanner_ipcc:96,rovers_ipcc:54,manyblocks_ipcc_big@train:200,gripper_ipcc@train:147,miconic_ipcc@train:228,visitall_ipcc@train:125,grid_ipcc@train:192,logistics_ipcc@train:156,spanner_ipcc@train:234,rovers_ipcc@train:312 \
  > logs/random_baseline.log 2>&1 &
  ```
- **P1 — UNION control column, all 8 (existential: C1 is measured against it):**
  ```bash
  ./train_test_scripts/run_config.sh configs/loo8_union_no_manyblocks.yaml cuda:0
  ./train_test_scripts/run_config.sh configs/loo8_union_no_gripper.yaml    cuda:0
  ./train_test_scripts/run_config.sh configs/loo8_union_no_miconic.yaml    cuda:0
  ./train_test_scripts/run_config.sh configs/loo8_union_no_visitall.yaml   cuda:0
  ./train_test_scripts/run_config.sh configs/loo8_union_no_grid.yaml       cuda:0
  ./train_test_scripts/run_config.sh configs/loo8_union_no_logistics.yaml  cuda:0
  ./train_test_scripts/run_config.sh configs/loo8_union_no_spanner.yaml    cuda:0
  ./train_test_scripts/run_config.sh configs/loo8_union_no_rovers.yaml     cuda:0
  ```
- **P2 — BIND ablation, 4 splits (certifies the binding layer, C2):**
  ```bash
  ./train_test_scripts/run_config.sh configs/loo8_joint_lite_no_manyblocks.yaml cuda:0
  ./train_test_scripts/run_config.sh configs/loo8_joint_lite_no_miconic.yaml    cuda:0
  ./train_test_scripts/run_config.sh configs/loo8_joint_lite_no_spanner.yaml    cuda:0
  ./train_test_scripts/run_config.sh configs/loo8_joint_lite_no_rovers.yaml     cuda:0
  ```
- **P3 — DOM ablation, same 4 splits:** as P2 with `loo8_structural_no_*`.
- **P4 — Table 3 endpoints:**
  ```bash
  ./train_test_scripts/run_config.sh configs/all8_union.yaml cuda:0
  ./train_test_scripts/run_config.sh configs/all8_joint.yaml cuda:0
  ```
- **P5 — diversity (C5):**
  ```bash
  ./train_test_scripts/run_config.sh configs/ho4_joint.yaml                cuda:0
  ./train_test_scripts/run_config.sh configs/ho2_joint_miconic_spanner.yaml cuda:0
  ```
- **P6 — optional strengtheners** (list above), only if GPUs idle.

## Rovers arity note (2026-07-23)

Rovers has schemas of arity up to 6; every other domain tops out at 4, so
rovers-held-out models used to train at ka=4 and CRASHED (CUDA assert) or
silently could not decode rovers' high-arity schemas. Fixed two ways: the
decoder now clamps gracefully (warning instead of crash), and every config
that holds out rovers (`*_no_rovers`, `ho2_*_blocks_rovers`, `ho4_*`) sets
`max_action_arity: 6` so the model trains at full width — reusing the
existing `kp3_ka6` sidecars, zero refeaturization. **The already-finished
`loo8_joint_no_rovers` run was trained at ka=4 and must be retrained after
pulling** (its rovers zero-shot numbers would be structurally handicapped).

## Per-filesystem prerequisites (once, before launching there)

```bash
git pull origin claude/wizardly-rubin-hqMug   # configs need base: support
conda activate <di_ltp_1 | ltp_3>
./train_test_scripts/collect_data.sh          # idempotent; complete caches skip in seconds
```
Cache state at time of writing: mangannr FS has joint 14/14; dugarp FS has
union/structural/joint_lite 42/42 and the joint LOO (kp3_ka4) sidecars.
`collect_data.sh` (optionally with method args) fills whatever is missing
and is safe to re-run.

## Monitoring / hygiene

```bash
tail -f logs/<name>.log                                            # live loss
pgrep -u $USER -af main.py | grep -o 'configs/[^ ]*' | sort | uniq -c   # distinct runs on this node
pkill -u $USER -f "main.py.*--device cuda"                         # kill training, spare CPU collection
```
One run shows as several identical pgrep lines (DataLoader workers) — count
configs, not processes. `pkill` is per-machine.

## Checkpoint selection — decided

The paper states selection by **combined** train+val loss (test-blind).
`_common.yaml` sets `test_model_metrics: combined`, which also cuts the
multi-hour test phase ~3x for every run launched from here on. In-flight
runs testing all three metrics are fine — report their combined-selected
numbers. Do not cherry-pick a different metric per run.

## Aggregation

Results land in `cache/results/<expid>/results_*.json` on each filesystem,
plus `cache/results/results_random_policy.json` from P0. Collect and
tabulate:
```bash
rsync -av <other-fs>:<repo>/cache/results/'*'/results_*.json cache/results/  # mirror across filesystems
python tools/analyze_results.py --csv paper_numbers.csv
```
Per-state inference times (GADAR vs BIND vs DOM cost note) come from the
Time column in the same JSONs — no extra runs.

## If time still runs out — cut in this order

1. Drop P5's `ho2_joint_miconic_spanner` (curve becomes 2 points, 4→7;
   still supports C5, weaker).
2. Drop `all8_union` (Table 3 becomes GADAR vs GABAR† only; UNION's
   in-domain competence can cite the loo8_union training-domain rows).
3. Shrink the ablation subset from 4 LOO splits to 2 (keep miconic +
   spanner; update the paper's Ablations paragraph to match).
4. Never cut: P0, P1, and at least 2 splits of P2 — without the floor,
   the control, and the binding ablation there is no paper.

---
# The two-H100 campaign (2 days, 32 cores each)

Two nodes, two different filesystems: **M** = the `mangannr` checkout, **D**
= the `dugarp` checkout.

## The rule that makes this easy: each node evaluates its own checkpoints

No model directories are ever copied between the nodes. Checkpoints stay
where they were trained and are evaluated there; only the small
`results_*.json` files get mirrored, at the end, exactly as the Aggregation
section above already describes. That is why the split below is by *role*
rather than by "half the runs each" - a role split needs no synchronisation
at any point in the two days.

Seeds are partitioned so a model directory never means two different things:
**seed 10 = the paper's existing runs, seed 11 = the sweep family (node M),
seeds 12-14 = the replicates (node D).**

## Core budget, per node

Evaluation is host-CPU bound, not GPU bound: 137s on GPU against 135s on CPU
for the same 20 visitall problems (H100 node, batched). So evaluation runs
`--device cpu` and never contends with training for the GPU. Training in turn
is not GPU-saturated: one run draws ~170W of 700W at ~1980 MHz and ~10 GB of
80 GB.

| job | cores | why |
|---|---|---|
| one training run | ~5 | 1 main + `num_workers: 4` dataloader workers |
| one evaluation run | 16 | `GABAR_FEATURIZE_WORKERS=16`, the plateau (PERFORMANCE.md) |

**Check the core count first - do not assume 32.** `dgxh-2` has been seen
with 2, 16 and 32 depending on the allocation, and the difference changes the
plan:

| cores | training slots | dataloaders each | eval lane while training | eval lane when idle |
|---|---|---|---|---|
| 32 | 4 | 4 | 12 | 16 |
| 16 | 4 | 2 | 4 | 16 |
| 8 | 3 | 1 | 2 | 8 |

At 16 cores the dataloaders halve rather than a training slot being dropped:
four concurrent trainings is the *GPU* power ceiling (4 x 170W ~ 700W)
regardless of cores, and dataloader workers idle during the forward/backward
pass, so they are the cheaper thing to cut. `tmp_scripts/budget.sh` holds this
arithmetic and both the preflight and the campaign driver derive from it -
the preflight prints the budget it computed, so you see the allocation you
actually got before you walk away.

Never run two evaluations at once: each wants 16 workers and
`configured_workers()` clamps against the affinity mask, which both runs see
as the full 32, so they oversubscribe instead of sharing.
`eval_queue.sh` exists to serialise them for you.

## Unattended: the whole thing in two commands per node

The per-lane commands below are the manual form, kept because they are what
the campaign script actually runs and what you want when something needs
re-running by hand. To just launch it and leave:

```bash
git pull origin claude/wizardly-rubin-hqMug
conda activate <di_ltp_1 | ltp_3>

./tmp_scripts/preflight.sh && touch .preflight_ok   # ~10 min, verifies wandb
./tmp_scripts/run_campaign.sh m                     # or d on the other node
```

`run_campaign.sh` detaches itself with `setsid`, so it outlives the tmux
session and the ssh connection. It refuses to start without `.preflight_ok`.
`DRY_RUN=1` prints the phase plan without running anything.

`preflight.sh` is deliberately end-to-end rather than a checklist: it runs
four real epochs through the multi-domain harness and joint_chain
featurization, evaluates the checkpoint, and then **reads the resulting
coverage numbers back off the wandb server**. A valid API key with no network
path passes every local check and still loses two days of results.

Everything reports to wandb project `ltp_gnn_gru_pyg`, grouped `node_m` /
`node_d`, with runs named `<group>/<expid>_s<seed>_<mode>`. Coverage lands in
the run **summary** (one sortable column per test domain in the runs table),
not only in history, so the two-day result is one table rather than a dozen
charts.

`tmp_scripts/` is throwaway - `rm -rf tmp_scripts .preflight_ok` when done.

## Before launching, on BOTH nodes

```bash
git pull origin claude/wizardly-rubin-hqMug
conda activate <di_ltp_1 | ltp_3>          # the wrong env is the usual cause of a weird failure
python -c "import os; print(len(os.sched_getaffinity(0)), 'cores')"   # expect 32
ls models/ | head                          # what this node can evaluate today
./train_test_scripts/collect_data.sh       # idempotent; complete caches skip in seconds
```

`ls models/` matters: node D's day-1 evaluation lane assumes the paper's
checkpoints are on that filesystem. If they are not, swap D's lanes (train
first, evaluate on day 2).

---

## Node M — the SWEEP node

Question it answers: does any regularization fix the overfitting-to-training-
domains failure mode? `_common.yaml` has `dropout`, `attention_dropout` and
`weight_decay` all at 0.0, so these are the untried levers.

**Day 1, hours 0+: four arms, 20 cores, GPU at its ceiling.**

```bash
for cfg in sweep_jc_base sweep_jc_drop01 sweep_jc_drop02 sweep_jc_l2; do
  ./train_test_scripts/run_config.sh configs/$cfg.yaml cuda:0
done
```

`sweep_jc_heads4` goes in as soon as one of the four finishes. Expect little
from it: `ploi/attention_layer.py` has `is_concat=False`, full width per
head, one shared score projection, and combines heads with `.mean(dim=1)` -
an averaging ensemble, not extra capacity. A large effect there is a reason
to re-read that file, not to celebrate.

All five arms are `mode: train`, seed 11, `epochs: 1000`,
`checkpoint_every: 100`. Testing is deliberately separate so the arms can be
compared on loss before anything is spent evaluating them.

**Day 1, once the first periodic checkpoints land: the 12-core eval lane.**
Twelve, not sixteen, because four trainings are holding 20 cores. The curve
is flat enough there (8 workers 118s, 16 workers 105s on 50 problems) that
it costs a few percent.

```bash
nohup env WORKERS=12 ./train_test_scripts/eval_queue.sh \
  configs/sweep_jc_base.yaml configs/sweep_jc_drop01.yaml \
  configs/sweep_jc_drop02.yaml configs/sweep_jc_l2.yaml \
  > logs/queue_sweep.log 2>&1 &
```

**Day 2: the data-need question.** Pick the winning arm, then vary
`--num-train-problems` (already in the checkpoint key as `d`, so each
fraction gets its own directory). `RUN_TAG` is REQUIRED here - three
launches of one config would otherwise all write the same log file.

```bash
for n in 25 50 100; do
  RUN_TAG="d$n" ./train_test_scripts/run_config.sh \
    configs/<winning-arm>.yaml cuda:0 --num-train-problems $n --mode train
done
```

Budget a first pass for data collection: a new problem count means new
unified caches and new sidecars (`cache/results/<Domain>_graphs_0_<n>*.pkl`),
it does not reuse the full-size ones.

---

## Node D — the SEEDS + EVAL node

Question it answers: are the paper's numbers stable, and what are the two
cells still marked provisional?

**Day 1, lane A (16 cores): evaluate what already exists.** This is the
lane that closes out the paper's open numbers - the two
`% UNION PROVISIONAL` cells in Table 2 come from `all8_union`.

```bash
nohup ./train_test_scripts/eval_queue.sh \
  configs/all8_union.yaml configs/all8_joint_chain.yaml \
  configs/all8_joint.yaml configs/all8_joint_lite.yaml \
  configs/all8_structural.yaml \
  > logs/queue_all8.log 2>&1 &
```

It reports `NO MODELS` per config rather than failing, so pointing it at
configs this filesystem never trained is harmless - that is also the fastest
inventory of what node D actually has.

**Day 1, lane B (15 cores): three seed replicates.** The paper and appendix
both say three seeds averaged; the tables are currently single-seed. Seed 10
already exists, so these three plus it give four.

```bash
for s in 12 13 14; do
  RUN_TAG="s$s" ./train_test_scripts/run_config.sh \
    configs/all8_joint_chain.yaml cuda:0 --seed $s --mode train
done
```

`--mode train` is forced on purpose: `all8_joint_chain.yaml` is
`train_test`, and three 8-domain test phases starting at unpredictable times
would blow the core budget. They get evaluated on day 2, in a queue.

**Day 2: evaluate the replicates, and take a spread.**

```bash
nohup ./train_test_scripts/eval_queue.sh configs/all8_joint_chain.yaml \
  > logs/queue_seeds.log 2>&1 &
python tools/analyze_results.py --csv node_d.csv
```

## Monitoring both nodes

```bash
pgrep -u $USER -af main.py | grep -o 'configs/[^ ]*' | sort | uniq -c  # runs here
nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv -l 5
tail -f logs/queue_*.log                                              # eval progress
```

Power draw is the honest saturation signal, not `utilization.gpu` (which is
time-occupancy: a GPU running one small kernel per millisecond reports ~80%).
Under 3-4 concurrent trainings expect 500-700W; well under that means the
node has room for another training run.

## What NOT to do

- Two evaluations at once on one node (see the core budget above).
- Copying `models/` between the filesystems. Each node evaluates its own.
- Launching the same config twice without `RUN_TAG` - the second launch
  rotates the first one's log and then both append to the same path, and the
  loss curve is the only place training history lives.
- Reusing seed 10 or 11 on node D, or seeds 12-14 on node M.
