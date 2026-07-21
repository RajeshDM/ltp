# Experiment configs

One YAML per paper experiment. Keys are argparse dest names (underscores).
Precedence: `constants.py` < YAML < explicit CLI flags.

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
