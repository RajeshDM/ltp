# Experiment configs

One YAML per paper experiment. Keys are argparse dest names (underscores).

**Layering:** shared settings live once in `_common.yaml`; each suite config
declares `base: _common.yaml` and contains only its delta (expid, domains,
featurization, test_domains, + any overrides). Precedence:
`constants.py` < `_common.yaml` < experiment file < explicit CLI flags.
To change a shared default (e.g. batch size, epochs), edit `_common.yaml`
once. To override it for one experiment, add the key to that file.
Suite-wide default batch_size is 64 (probed 2026-07: ~3.3x faster than 16
at equal quality per wall-clock; ~9 GB GPU memory).

## THE PAPER SUITE (8 domains, 4 methods, 36 training runs)

Method-to-featurization mapping:

| Paper name | featurization | What it is |
|---|---|---|
| UNION (control) | `union` | symbol one-hots, D in the weights |
| GADAR-DOM (Method 0) | `structural` | symbol-free structural classes only |
| GADAR-BIND (B-lite) | `joint_lite` | + lifted domain layer in every state graph |
| GADAR (full, Method B) | `joint` | + occurrence nodes + binding layer |

Configs: `all8_<feat>.yaml` (train all 8, in-domain eval on all 8) and
`loo8_<feat>_no_<domain>.yaml` (train 7, test all 8 - the held-out domain
is auto-detected zero-shot). 4 x (1 + 8) = 36 runs.

Launch one per GPU:
```bash
./train_test_scripts/run_config.sh configs/loo8_joint_no_spanner.yaml cuda:0
```

Toy end-to-end check (GPU - the test path is GPU-only by design):
```bash
python main.py --domains manyblocks_ipcc_big,gripper_ipcc --featurization joint \
    --num-train-problems 5 --num-test-problems 2 --all-problems \
    --test-domains "manyblocks_ipcc_big:2,gripper_ipcc:2,miconic_ipcc:2,miconic_ipcc@train:2" \
    --epochs 3 --mode train_test --device cuda:0 --use-gpu True --expid toy_joint \
    --run-non-optimal False --wandb False
```
Cache warmup first (once, sequential - shared NFS pickle safety):
```bash
for f in union structural joint_lite joint; do
  python main.py --config configs/all8_$f.yaml --mode train --epochs 0 --device cpu --use-gpu False
  for d in manyblocks gripper miconic visitall grid logistics spanner rovers; do
    cfg=$(ls configs/loo8_${f}_no_${d}* 2>/dev/null | head -1)
    [ -n "$cfg" ] && python main.py --config $cfg --mode train --epochs 0 --device cpu --use-gpu False
  done
done
```

Run any config with:

```bash
python main.py --config configs/<name>.yaml --device cuda:0
# or, backgrounded with logging:
./train_test_scripts/run_config.sh configs/<name>.yaml cuda:0
```

Machine-specific things (`--device`, `--wandb`, `--continue-training`,
`--dry-run`) deliberately stay OUT of configs so the same file runs anywhere.

## The headline experiment (C1 Phase-2 gate): 6 train / 2 held-out

| Config | Trains on | Zero-shot tests | Role |
|---|---|---|---|
| `heldout2_union.yaml` | blocks, gripper, visitall, grid, logistics, rovers | miconic, spanner | C1 CONTROL (expect ~0) |
| `heldout2_structural.yaml` | same 6 | miconic, spanner | C1 PRIMARY (must beat control) |

Held-out choice: miconic is transport-family (logistics/gripper are in
training - tests family transfer, C5); spanner is structurally distinct.
To change the split, edit `domains:` and the zero-shot set follows
automatically (any test domain not in `domains:` is zero-shot).

NOTE: first run collects expert plans for visitall/grid/logistics/rovers
at 200 requested problems - hours of FD time. Do the cache warmup
(README step in main instructions) BEFORE launching on GPUs.

## The 3-domain suite (11 training runs, proven caches)

| Config | Trains on | Tests | Claim |
|---|---|---|---|
| `baseline_blocks.yaml` | blocks (per-domain feat.) | blocks | skyline |
| `baseline_gripper.yaml` | gripper | gripper | skyline |
| `baseline_miconic.yaml` | miconic | miconic | skyline |
| `union_3dom.yaml` | all 3, union vocab | all 3 in-domain | C1 control (Baseline 0) |
| `structural_3dom.yaml` | all 3, structural | all 3 in-domain | C1 (expressiveness tax) |
| `loo_union_no_blocks.yaml` | gripper+miconic, union | all 3 (blocks zero-shot) | C1 control |
| `loo_union_no_gripper.yaml` | blocks+miconic, union | all 3 (gripper zero-shot) | C1 control |
| `loo_union_no_miconic.yaml` | blocks+gripper, union | all 3 (miconic zero-shot) | C1 control |
| `loo_structural_no_blocks.yaml` | gripper+miconic, structural | all 3 (blocks zero-shot) | C1 primary |
| `loo_structural_no_gripper.yaml` | blocks+miconic, structural | all 3 (gripper zero-shot) | C1 primary |
| `loo_structural_no_miconic.yaml` | blocks+gripper, structural | all 3 (miconic zero-shot) | C1 primary |

The C1 headline number = zero-shot success of `loo_structural_no_X` vs
`loo_union_no_X` on domain X, for each X.

`toy_check.yaml` is a 2-epoch CPU sanity run (not part of the suite).

Methods A and B (compiled vs joint domain conditioning, CLAUDE.md 5.6) are
NOT implemented yet; when they land as featurization/model modes, they get
configs following the same pattern.

## Hyperparameter sweep (`sweep_jc_*`) — serves no claim directly

Five arms over `all8_joint_chain`, identical except for one knob, all seed 11
and `mode: train`. They exist to find settings for the C1/C4 headline runs,
not to produce a table cell. Launch and core budget: RUNBOOK.md, "The 32-core
H100 campaign".

| config | knob |
|---|---|
| `sweep_jc_base` | none (reference) |
| `sweep_jc_drop01` | `dropout: 0.1` |
| `sweep_jc_drop02` | `dropout: 0.2`, `attention_dropout: 0.1` |
| `sweep_jc_l2` | `weight_decay: 1e-4` |
| `sweep_jc_heads4` | `n_heads: 4` |

## Domain facts (pddlgym)

| shorthand | pddlgym name | train problems | test problems |
|---|---|---|---|
| blocks | manyblocks_ipcc_big | 200 | 200 |
| gripper | gripper_ipcc | 147 | 173 |
| miconic | miconic_ipcc | 228 | 119 |

Requested counts are capped at what exists (loudly), so 200/200 everywhere is
safe and keeps the unified caches shared across runs (cache filename is keyed
by the REQUESTED count).

## Analysis

After runs finish (results land in `cache/results/<expid>/results_*.json`):

```bash
python tools/analyze_results.py                 # table of all experiments
python tools/analyze_results.py --csv out.csv   # also write CSV
```
