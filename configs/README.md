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

## The paper suite (11 training runs)

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
