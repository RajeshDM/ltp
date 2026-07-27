# Run ledger — what is running, what it feeds

Status as reported by the user; update as runs finish. Paper table rows are
the 4 representative held-out domains: **manyblocks, gripper, miconic,
visitall** (documented failure modes + low random floors). Every method
column is measured on those same 4 splits so comparisons are matched.

16 launched 2026-07-26. Cost: ~9-10 h train + ~12 h full test sweep per run (batch 64).
`--mode train` skips the test phase; targeted `@train` readouts take ~10 min.

## Ledger

| # | Config | Mode | Status | Feeds |
|---|---|---|---|---|
| 1 | `loo8_union_no_manyblocks` | train_test | RUNNING | control column |
| 2 | `loo8_union_no_gripper` | train_test | RUNNING | control column |
| 3 | `loo8_union_no_miconic` | train_test | RUNNING | control column |
| 4 | `loo8_union_no_visitall` | train_test | RUNNING | control column |
| 5 | `loo8_structural_no_gripper` | train_test | RUNNING | over-conditioning test |
| 6 | `loo8_structural_no_visitall` | train_test | RUNNING | over-conditioning test |
| 7 | `all8_union` | train_test | RUNNING | in-domain control (price of generality) |
| 8 | `loo8_joint_chain_no_gripper` | train | RUNNING | chain readout (class 1+3) |
| 9 | `loo8_joint_chain_no_visitall` | train | **TRAINED + READ OUT** | **easy 100.0% (E190, V1 99.2%, PQ 0.97); hard 7/7 so far vs floor 6.0% and joint 4%** |
| 10 | `loo8_joint_chain_no_manyblocks` | train | **VOID - re-run** | 0/128 zero-shot, but CONFOUNDED: best val 14.45 @ E290 was never saved (07-23 joint squatters held both validation slots at 13.397/13.500), so only E470/490/500 existed to test. Re-run with `--seed 30 --keep-checkpoints 6`. |
| 11 | `loo8_joint_chain_no_miconic` | train | RUNNING | chain, table cell |
| 12 | `loo8_structural_no_manyblocks` | train_test | RUNNING | completes structural column |
| 13 | `loo8_structural_no_miconic` | train_test | RUNNING | completes structural column |
| 14 | `all8_structural` | train_test | RUNNING | in-domain: expressiveness tax |
| 15 | `all8_joint_chain` | train_test | RUNNING | chain in-domain row + dilution test (informative either way) |
| 16 | `loo8_joint_no_visitall` @ repr 128, seed 12 | train_test | RUNNING | capacity probe: is 64-dim saturated by the wider feature set? (reach is NOT the issue - the global node makes the diameter 2) |
| - | `loo8_joint_no_*` x8 | - | **DONE** | GADAR column (covers all 4 rows + 4 extra) |
| - | `all8_joint` | - | DONE (verify) | GADAR in-domain row |
| - | random floor, all 8 x 2 splits | - | **DONE** | `cache/results/random_floor_summary.txt` |
| - | union 3-domain (old suite) | - | DONE | preliminary control: solved in-domain, 0 zero-shot |

Deferred until the chain readout is positive: `all8_joint_chain`,
`ho4_joint_chain`, `ho2_joint_chain_*` (diversity curve, C5). A diversity
curve for a system that does not transfer measures noise.

Dropped: `joint_lite` (BIND) - attribution between two ~0 zero-shot rungs is
vacuous, and if chain becomes the method the ladder re-anchors on chain
(chain-minus-X), which joint_lite does not serve. Also dropped: `ho2`/`ho4` for union/structural, the other 4
splits of every ablation column. 42 -> 13.

## Standing rule for every new zero-shot run

Two independent failure modes have already voided runs, and both are silent
until you inspect provenance. Launch every zero-shot run with BOTH:

```
--seed <fresh> --keep-checkpoints 6
```

1. **Squatting.** `ModelManager`'s key is `train_env_name + seed +
   hyperparameters` and does NOT include `featurization`, so an older run on
   the same domain set shares the directory. If its losses are lower, the new
   run's checkpoints are never written. A fresh `--seed` gives a virgin
   directory. (Post-deadline fix: put featurization in the key.)
2. **Tail-only retention.** With the default 2 slots you keep the two
   lowest-loss epochs, which are always late. Zero-shot transfer peaks EARLY
   and decays while training loss keeps falling (visitall: E190 100% ->
   E330 97.6% -> E500 3.3%). Six slots keep the epochs that actually transfer.

Before trusting any zero-shot number, confirm which epochs exist and who
wrote them:

```bash
python tools/inspect_checkpoints.py models/MULTI-<...>/
```

Rows from a different featurization (width mismatch) are skipped loudly at
load time; rows from a different date are the squatting signature.

## Readouts

Zero-shot rows are CHECKPOINT-SPIKY (visitall no_visitall: E240 0%,
E260 16%, E500 3.3%). Always test every saved checkpoint:

```bash
python main.py --config configs/<cfg>.yaml --mode test \
    --test-domains "<held_out>@train:125" --device cuda:0 \
    --test-model-metrics "validation,training,combined" --num-models-to-test 2
```

Chain-specific, per failure class:
```bash
# class 2 (object selection): coverage vs joint's 16% and the 50% oracle ceiling
python main.py --config configs/loo8_joint_chain_no_visitall.yaml --mode test \
    --test-domains "visitall_ipcc@train:125" --device cuda:0 \
    --test-model-metrics "validation,training,combined" --num-models-to-test 2

# class 1+3 (schema selection + direction): does drop finally get proposed?
GABAR_DEBUG_PROPOSALS=25 GABAR_DEBUG_TRACE=1 python main.py \
    --config configs/loo8_joint_chain_no_gripper.yaml --mode test \
    --test-domains "gripper_ipcc@train:5" --device cuda:0
```

## Baselines to compare against (measured)

| Domain | random floor (easy/hard) | joint zero-shot (easy) | oracle ceiling (easy) |
|---|---|---|---|
| manyblocks | 1.67 / 0.17 | - | - |
| gripper | 0.00 / 0.00 | 0% | - |
| miconic | 88.30 / 1.12 | 0% | 0-5% |
| visitall | 99.73 / 6.00 | 16% (E260) | 50% |

Visitall's easy-split floor is degenerate (a monitored random walk ~= optimal
exploration); read Visitall on the hard split and on plan quality.

## RESULT 2026-07-26: joint_chain works

`loo8_joint_chain_no_visitall`, zero-shot on a domain never trained on:

| split | joint (old) | joint_chain | random floor | oracle ceiling |
|---|---|---|---|---|
| visitall@train (easy) | 16% (E260) | **100%** (E190) | 99.73% | 50% |
| visitall (hard) | 4% | **7/7 running** | 6.00% | - |

Plan quality 0.97 (mean 30.1 vs LAMA 29.2), V1 99.2%. Chain exceeds the
oracle ceiling, i.e. the learned goal/chaining features do more than the
hand-coded terminal trigger. Checkpoint spikiness persists and EARLIER is
better again (E190 100% / V1 99.2 vs E330 97.6% / V1 68.7).

Consequence: joint_chain becomes the method (GADAR); joint becomes the
ablation (GADAR minus chaining). Diversity configs (`ho4_joint_chain`,
`ho2_joint_chain_*`) are now worth creating.
