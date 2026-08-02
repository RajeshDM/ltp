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

# The 32-core H100 campaign (2 days)

## Core budget — training and evaluation CAN share the node

Evaluation is host-CPU bound, not GPU bound: 137s on GPU against 135s on CPU
for the same 20 visitall problems (H100 node, batched). So evaluation runs
with `--device cpu` and never contends with training for the GPU. Training
in turn is not GPU-saturated: one run draws ~170W of 700W at ~1980 MHz and
~10 GB of 80 GB.

Budget on 32 cores:

| job | cores | why |
|---|---|---|
| one training run | ~5 | 1 main + `num_workers: 4` dataloader workers |
| one evaluation run | 16 | `GABAR_FEATURIZE_WORKERS=16`, the plateau (PERFORMANCE.md) |

**3 trainings + 1 evaluation = 31 cores.** That is the shape to hold. Four
concurrent trainings is the GPU ceiling (4 x 170W ~ 700W), not the CPU one,
so go to 4 trainings only while no evaluation is running.

What NOT to do: two concurrent evaluations. Each wants 16 workers, and the
pool degrades under oversubscription rather than sharing gracefully
(`configured_workers()` clamps to the affinity mask, which both runs see as
32). Evaluations go one at a time.

## Order of work

Nothing is evaluable until the first checkpoints land, so:

**Hours 0-N: training only, 4 arms at once.** All five `sweep_jc_*` arms are
`mode: train`, seed 11, `epochs: 1000`, `checkpoint_every: 100`.

```bash
for cfg in sweep_jc_base sweep_jc_drop01 sweep_jc_drop02 sweep_jc_l2; do
  ./train_test_scripts/run_config.sh configs/$cfg.yaml cuda:0
done
# heads4 goes in as soon as one of the four finishes
```

**From the first periodic checkpoint on: add the evaluation lane.** Drop to
3 concurrent trainings and run one evaluation at a time:

```bash
GABAR_BATCH_EVAL=1 GABAR_FEATURIZE_WORKERS=16 \
  python main.py --config configs/sweep_jc_base.yaml --mode test --device cpu \
    --test-model-metrics training,combined,validation --num-models-to-test 2
```

`--device cpu` is deliberate and costs nothing measurable; it keeps the GPU
entirely for training.

## Why these five arms

The failure mode in the paper's numbers is overfitting to the training
domains, and `_common.yaml` currently runs with `dropout`,
`attention_dropout` and `weight_decay` all at 0.0. Those are the untried
levers, so they go first. `n_heads` is arm 4 and is expected to be small:
`ploi/attention_layer.py` has `is_concat=False`, full width per head, a
single shared score projection, and combines heads with `.mean(dim=1)` — an
averaging ensemble, not extra capacity.

More epochs alone is not one of the arms. Loss stops moving well before 500,
so 1000 is here to make the arms comparable at a fixed budget, not as a
treatment.

## Checkpoint keys

`weight_decay` reaches the optimizer but was absent from the ModelManager
key, so every L2 value used to write into one directory. It is now `'l2'` in
`training_hyperparameters` (main.py), defaulted away at 0.0 — verified that
runs at 0.0 keep the exact folder name and hash they had before, so no
existing checkpoint moved. `--expid` is NOT part of the key: two arms must
differ in a keyed hyperparameter (or the seed) or they will share a
directory whatever their expid says.
