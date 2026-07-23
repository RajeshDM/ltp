# RUNBOOK — every result the paper needs, prioritized

One place to look up: which runs feed which claim/table, in what order to
launch them, and the exact commands. Written 2026-07-23, ~1 week to
submission. Budget reality: at batch 64 a run is ~8.5 h training +
several hours testing → plan ~1 run/GPU/day (a bit more with
combined-only testing, see below).

## Results-to-claims matrix

| Paper artifact | Claim | Runs required | Count |
|---|---|---|---|
| Table 2 (LOO zero-shot ladder) | C1 + C2 | `loo8_{union,structural,joint_lite,joint}_no_*` | 32 |
| Table 2 Random column (floor) | C1 integrity | `tools/random_policy_baseline.py` (CPU) | 1 cmd |
| Table 2 GABAR† column (ceiling) | calibration | published GABAR numbers | 0 |
| Table 3 (in-domain, price of generality) | C1(c) | `all8_{union,structural,joint_lite,joint}` | 4 |
| Diversity figure (4→6→7 train domains) | C5 | `ho4_joint`, `ho2_joint_*`×4, `ho4_union` (anchor) | 6 |
| Diversity, remaining rungs (optional) | C5 (nice) | `ho2_union_*`×4, ho2/ho4 structural + joint_lite | 14 |
| Planner ≈100% + plan-quality refs | framing | already inside every run's results JSON (`run_non_optimal`) | 0 |
| Per-state inference time DOM/BIND/GADAR | cost note | extract from test logs of runs above | 0 |
| Renaming invariance | Prop. 2 | `python tests/test_lifted_layer.py` (CI) | 0 |

Total GPU runs for a complete paper: **42** (32 + 4 + 6). Already in
flight: `loo8_joint` ×8 → **34 to launch.** The optional 14 complete the
diversity curve for every rung; cut them first.

## Priority order (launch top-down as GPUs free up)

- **P0 — Random floor (CPU, run today, any node, no GPU):**
  ```bash
  nohup python tools/random_policy_baseline.py --domains \
  manyblocks_ipcc_big:200,gripper_ipcc:173,miconic_ipcc:119,visitall_ipcc:50,grid_ipcc:48,logistics_ipcc:96,spanner_ipcc:96,rovers_ipcc:54,manyblocks_ipcc_big@train:200,gripper_ipcc@train:147,miconic_ipcc@train:228,visitall_ipcc@train:125,grid_ipcc@train:192,logistics_ipcc@train:156,spanner_ipcc@train:234,rovers_ipcc@train:312 \
  > logs/random_baseline.log 2>&1 &
  ```
- **P1 — UNION control column (existential: C1 is measured against it):**
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
- **P2 — BIND column (certifies the binding layer, C2):** same 8 commands
  with `loo8_joint_lite_no_*`.
- **P3 — DOM column (completes the ladder):** same with
  `loo8_structural_no_*`.
- **P4 — in-domain all8 (Table 3):** endpoints first —
  `all8_union`, `all8_joint`, then `all8_structural`, `all8_joint_lite`.
- **P5 — diversity minimum (C5):** `ho4_joint`, `ho2_joint_blocks_rovers`,
  `ho2_joint_gripper_grid`, `ho2_joint_miconic_spanner`,
  `ho2_joint_visitall_logistics`, `ho4_union`.
- **P6 — only if GPUs idle:** `ho2_union_*`×4, then ho2/ho4 for
  structural and joint_lite.

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
`_common.yaml` now sets `test_model_metrics: combined`, which also cuts the
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

## If time runs out — cut in this order

1. Drop P6 (diversity curve becomes GADAR + UNION-anchor only — still a figure).
2. Drop `ho2_joint_*` (curve becomes 2 points, 4→7; still supports C5, weaker).
3. Shrink Table 3 to UNION vs GADAR columns (endpoints price generality;
   middle rungs' in-domain numbers can cite the LOO runs' training-domain
   rows in the appendix).
4. Never cut: P0, P1, P2 — without the floor, the control, and the
   binding ablation there is no paper.
