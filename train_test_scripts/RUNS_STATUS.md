# Run ledger — what is running, what it feeds

## MORNING CHECKLIST 2026-07-28 (everything left, in order)

1. **Read out the overnight self-tests** (A, B, E1 finish training ~03:00-05:00,
   then test themselves): `python tools/digest.py --print` on both machines.
   - A `loo8_joint_chain_no_visitall` seed 30: the final-form headline.
   - B `all8_joint_chain` seed 30: in-domain GADAR column (8 cells).
   - E1 `loo8_joint_chain_no_spanner` seed 30: bar to clear = untrained
     control 0.0/0.0 (measured), with sane PQ; random floor 63/44 for context.
2. **UNION cells via the 07-23 worktree** (old union checkpoints do not load
   under current code: width mismatch, SKIP lines; test with the code that
   trained them):
   ```
   git worktree add ../ltp_0723 $(git log --format=%h --until="2026-07-23T06:00" -1)
   cd ../ltp_0723 && ln -s ../ltp/cache cache && ln -s ../ltp/models models
   ```
   - submit-a fs: loo8_union_no_miconic test (miconic@train:228 + :119,
     metrics training,combined, 2 models, NO --seed) -> 2 main-table cells,
     ~1.5h. THE priority.
   - dgx2-5 fs: all8_union test (8 hard splits) -> in-domain UNION column,
     ~4-5h. Secondary table; can ship as [--] if time runs out.
   Test-only: reads checkpoints/sidecars, writes timestamped JSONs. Cannot
   corrupt anything; only GPU contention matters.
3. Grid full-set retest verdict: claimable only if best epoch beats the
   untrained 17.5% with PQ >~0.3. E150 was pacing ~12.8%, PQ 0.11-0.21 ->
   expect to keep grid out.
4. Human-only: GABAR-dagger cells from the published paper, bib keys
   (gammazero, hoo_pomdp, halo_orchestrator, learning_to_search_2026,
   lookahead_policies_2026, llm_evolved_heuristics_2026), todonums
   ([125]/[312], ~100% planner), Overleaf recompile + page count.

## FINAL BATCH 2026-07-27 (launched ~22:00, done ~14-16h later)

GADAR := joint_chain WITH type compilation (current code, _ty2). These four
runs complete every remaining cell of the paper:

| run | kind | fills | readout |
|---|---|---|---|
| A `loo8_joint_chain_no_visitall` seed 30, ckpt-every 50 | retrain+test | **headline**: Visitall GADAR E+H on the final form (replaces pre-compilation 100/68) | all 10 periodic epochs tested, hard split first |
| B `all8_joint_chain` seed 30, ckpt-every 50 | retrain+test | in-domain GADAR column (8 hard splits) | combined, 1 model |
| C `all8_union` (no seed = default) | test-only | in-domain UNION column | training+combined, 2 models |
| D `loo8_union_no_miconic` (no seed) | test-only, submit-a | zero-shot UNION Miconic E+H | training+combined, 2 models |

Morning readout: `python tools/digest.py --print` on both machines.
Decision rule for A: if final-form Visitall hard is comparable to 68%, the
paper is one featurization end to end; if clearly worse, either report the
lower number or define GADAR as the pre-compilation variant (drop type
compilation from the method section) — method definition is ours to make,
but the table and §method must match whichever we pick.

Already-final numbers (do not re-run): Miconic GADAR 35.96 E / 0.00 H
(_ty2); UNION Visitall 0/0; BIND column (old feat, uniform); untrained
controls (Visitall 26.4/0.0, Miconic 2.63). Grid + Logistics: measured,
failed their controls, cut from the paper.

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

## 2026-07-27: type compilation invalidates the joint_chain column

Declared PDDL types are now compiled into the lifted layer (unary static
predicate per type, 'pre' occurrence per typed slot, `has_type` object edge),
so typed and predicate-typed domains agree. Lifted feature widths changed:
every `joint*` checkpoint predates it and every `joint*` cache sidecar takes
a new `_ty` tag. The four joint_chain splits must be retrained together or
the GADAR column mixes two featurizations - including `no_visitall`, whose
100%/68% was trained on 7 domains that were type-blind at the time.

`structural` and `union` are unaffected (no lifted layer / symbol one-hots
already carry types), so those columns stand.

Verify compilation fired before trusting a run - the count is printed per
domain at metadata build:

```bash
grep "lifted layer" <log> | sort -u
#   gripper/miconic/spanner/rovers/blocksworld -> 1..7 declared type(s)
#   grid/logistics                             -> 0 (predicate-typed)
```

## 2026-07-27 launch: 5 type-compiled joint_chain splits, zero-shot only

Held out: manyblocks, gripper, miconic (typed - the family type compilation
targets) + grid, logistics (predicate-typed - the control that says whether
compilation disturbed the family that already worked).

Test scope is overridden on the CLI to the held-out domain's two splits and
NOTHING else: the config's 9-entry sweep is ~12 h and we are deciding whether
to change the design, not filling the in-domain rows. Held-out counts:

| held out | easy (@train) | hard |
|---|---|---|
| manyblocks_ipcc_big | 200 | 200 |
| gripper_ipcc | 147 | 173 |
| miconic_ipcc | 228 | 119 |
| grid_ipcc | 192 | 48 |
| logistics_ipcc | 156 | 96 |

## Cluster convention: the device is ALWAYS cuda:0

Every job runs on its own machine or its own GPU-isolated allocation, so each
process sees exactly one device and it is numbered 0. `cuda:1` and above
select a GPU that is not visible to that process. Parallelism comes from
launching on different hosts/allocations, never from different device
indices in the same command.

## Standing rule for every new zero-shot run

Two independent failure modes have already voided runs, and both are silent
until you inspect provenance. Launch every zero-shot run with:

```
--seed <fresh> --keep-checkpoints 6 --checkpoint-every 50
```
and read it out with `--test-model-metrics periodic`.

1. **Squatting.** `ModelManager`'s key is `train_env_name + seed +
   hyperparameters` and does NOT include `featurization`, so an older run on
   the same domain set shares the directory. If its losses are lower, the new
   run's checkpoints are never written. A fresh `--seed` gives a virgin
   directory. (Post-deadline fix: put featurization in the key.)
2. **Tail-only retention.** Every loss-ranked slot ends up holding a late
   epoch, because the loss keeps falling: with the default 2 slots on a
   500-epoch run you keep E490 and E500, and widening to 6 only widens the
   tail. Zero-shot transfer peaks EARLY and decays while the loss keeps
   improving (visitall: E190 100% -> E330 97.6% -> E500 3.3%), so the epochs
   worth testing are precisely the ones loss ranking discards. That is what
   `--checkpoint-every 50` fixes: an unconditional snapshot every 50 epochs
   into a fourth `periodic` metric, ordered EARLIEST first, capped by
   dropping the latest (which the loss-ranked metrics already keep).
   `--keep-checkpoints 6` still helps against squatting; it is not a
   substitute for periodic.

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

## RESULT 2026-09-25: 8-fold zero-shot epoch sweep (type-compiled suite)

`EXPID_SUFFIX=_epochs`, periodic checkpoints every 50 epochs (epoch 0 is
never saved, so the sweep has no untrained control), hard (test) split only,
single seed. Coverage %, every nonzero cell; all other (rung, fold, epoch)
cells are 0.0. `loo8_union_no_gripper` untrained (1 checkpoint).

| rung | fold | per-epoch coverage | cross-fold e* -> cov |
|---|---|---|---|
| GADAR (joint_chain) | visitall | 0.0 at all of 50..400 (every problem hits step 501) | 0.0 |
| GADAR | grid | 100: 10.4, 150: 2.1, 300: 20.8 | E250 -> 0.0 |
| GADAR | manyblocks | 0.5 .. 4.5 | E300 -> 3.0 |
| BIND (joint_lite) | visitall | 50: 52, 100: 28, 150: 0, 200: 26, 250+: 0 | E200 -> 26.0 |
| BIND | manyblocks | 0.0 .. 1.5 | E50 -> 1.0 |
| UNION | visitall | 150: 74, all others 0 | E50 -> 0.0 |

Random floors (hard): visitall 6.0, manyblocks 0.2, grid 0.0.

Reading:
- The July `no_visitall` 68%/100% was the type-BLIND model (voided above,
  2026-07-27). Its type-compiled retrain transfers to Visitall at no swept
  epoch. Both results can be right: type compilation changed what the model
  learns from the five typed training domains. Not yet separated from seed
  variance - one seed each.
- Visitall transfer is checkpoint-unstable for every rung, the control
  included (0 <-> 74 within 50 epochs). A single Visitall number measures
  the checkpoint, not the representation.
- Several runs stopped at 300-400 of 500 epochs (allocations died); the
  per-fold epoch lists above are what exists.

## PLAN 2026-09-26: final results for the revision, and the cut order

Zero-shot transfer did not hold (result above), so the paper's in-domain
question becomes central: does GADAR trained ALONE match GABAR trained alone
on the same test problems, and what does sharing one model across seven
domains cost? Priorities, written once:

| P | result | status | run |
|---|---|---|---|
| 1 | single-domain GADAR, 8 domains | trained | eval `sd_joint_chain_*` |
| 1 | GABAR through our harness, same test problems, 8 domains | 3 configs existed, 5 added | `train_then_eval.sh configs/baseline_*.yaml` |
| - | 7-domain GADAR in-domain (LOO, n=7) | DONE | `tools/indomain_table.py` |
| - | zero-shot, 3 rungs x 8 folds, cross-fold rule | DONE (union_no_gripper untrained) | `tools/crossfold_select.py` |
| 3 | UNION in-domain (7 folds) | eval never ran in-domain | eval `loo8_union_no_*` |
| 4 | BIND in-domain (8 folds) | eval never ran in-domain | eval `loo8_joint_lite_no_*` |

Cut first -> last: BIND in-domain, UNION in-domain, baselines beyond the
three original domains. Never cut: single-domain GADAR eval, the three
original GABAR baselines. Not attempted this revision (no time):
`loo8_union_no_gripper` training, seeds, an untrained-network control
(epoch 0 is never checkpointed), `all8_*` (pre-type-compilation).

Caveat for the comparison: `baseline_*` use GABAR's published settings
(batch 16, <=200 training problems); `sd_joint_chain_*` use the suite's
(batch 64, all problems). Same test problems, same epochs and selection.

## RESULT 2026-09-26: in-domain, single-domain GADAR vs seven-domain GADAR vs GABAR

Hard (test) split, combined-loss checkpoint, one seed. LOO = mean over the 7
leave-one-out models that trained on the domain. SD = `sd_joint_chain_*`.
GABAR = `baseline_*` through our harness, same test problems.

| domain | GABAR† (published) | GABAR (ours) | LOO GADAR | SD GADAR |
|---|---|---|---|---|
| blocks | 100.0 | pending (410/500) | 74.6 | 95.5 |
| gripper | 100.0 | pending (training) | 94.5 | 100.0 |
| miconic | 100.0 | 99.2 (E330) | 61.9 | 41.2 |
| spanner | 92.0 | - | 47.9 | 85.4 |
| visitall | 90.7 | - | 81.1 | 54.0 |
| grid | 96.3 | - | 94.6 | 87.5 |
| logistics | 79.0 | - | 2.7 | 3.1 |
| rovers | 82.0 | - | 9.0 | 7.4 |
| mean | 92.5 | | 58.3 | 59.3 |

Reading: sharing one model across seven domains costs ~1 point on average
(58.3 vs 59.3), with positive transfer to visitall (+27), miconic (+21),
grid (+7) and negative to spanner (-38), blocks (-21). Logistics and rovers
fail single-domain too, so their gap to GABAR is the representation, not
sharing; UNION reaches 59.4 in-domain on logistics (n=1), consistent with
truck/airplane aliasing. Miconic: representation cost is large (SD 41.2 vs
matched GABAR 99.2).

## FINAL 2026-09-26: in-domain, all folds (as submitted to ICLR 2027)

Mean over the LOO models trained on each domain (UNION 6/7 folds: no_gripper
untrained; BIND and GADAR 7/7), combined-loss checkpoint, hard split.

| domain | UNION | BIND | GADAR | GADAR_1 (single-domain) |
|---|---|---|---|---|
| blocks | 57.2 | 43.4 | 74.6 | 95.5 |
| gripper | 84.1 | 95.4 | 94.5 | 100.0 |
| miconic | 86.1 | 67.0 | 61.9 | 41.2 |
| visitall | 77.3 | 78.6 | 81.1 | 54.0 |
| grid | 89.6 | 64.9 | 94.6 | 87.5 |
| logistics | 18.6 | 0.3 | 2.7 | 3.1 |
| spanner | 48.6 | 40.0 | 47.9 | 85.4 |
| rovers | 9.6 | 8.2 | 9.0 | 7.4 |
| mean | 58.9 | 49.7 | 58.3 | 59.3 |

Zero-shot table as submitted: Blocks/others from the cross-fold sweep above;
the Visitall, Miconic and Grid rows use the AAAI-submission (July, pre-type-
compilation) numbers, by the author's decision.

## 2026-11-27 02:55: Table-1 view - GABAR vs GADAR alone, both protocols (interim)

Test split, combined checkpoint, one seed. "cd" = constrained decoding
(cache/results/<cfg>_cdec). GADAR = the EXISTING batch-64, lr 5e-4
single-domain models (sd_joint_chain_*), to be retrained at the E5 setting.

| domain | GABAR published | GABAR ours | GABAR cd | GADAR alone | GADAR alone cd |
|---|---|---|---|---|---|
| Blocks | 100 | 52.5 | 47.0 | 95.5 | **98.5** |
| Gripper | 100 | 100.0 | 100.0 | 100.0 | 100.0 |
| Miconic | 100 | 99.2 | 100.0 | 44.5 | 34.5 |
| Visitall | 90.7 | 70.0 | 74.0 | 54.0 | 54.0 |
| Grid | 96.3 | 89.6 | 97.9 | 87.5 | 85.4 |
| Logistics | 79.0 | 0.0 | 2.1 | 3.1 | **36.5** |
| Spanner | 92.0 | 91.7 | 91.7 | 85.4 | 90.6 |
| Rovers | 82.0 | 7.4 | 46.3 | 20.4 | **57.4** |

Under the same (constrained) protocol GADAR alone already beats GABAR on
Blocks, Logistics and Rovers, ties Gripper, and trails on Miconic, Visitall,
Grid, Spanner - the domains where batch 64 undertrained it (Miconic at
batch 16: 100.0 / 99.2 cd). Constraining changes GABAR little except on Rovers
and Grid. Not concluded: one seed; the GADAR column is the old training setup.

E5 so far (batch 64): Miconic lr 1e-3 97.5 (cd 98.3; batch 16: 100.0, cd
99.2); Visitall lr 1e-3 42.0 (cd 34.0), lr 2e-3 76.0 at E70 (best combined
loss that early - check its curve before trusting it); Miconic lr 2e-3 and
the Visitall batch-16 control still running.

## 2026-11-27 01:50: batched constrained decoding reproduces the sequential runs

Full test sets, same checkpoints, batched evaluator + 7 rollout workers
(share GPU nodes, 8 CPUs) vs the sequential runs (results <cfg>_cdecb vs
<cfg>_cdec):

| model | sequential cov | batched cov | failed problems | wall time seq -> batched |
|---|---|---|---|---|
| GABAR Rovers | 46.3 | 46.3 | same 29 | ~21 min -> 14.6 min |
| GADAR alone Rovers | 57.4 | 57.4 | same 23 | ~25 min -> 22.5 min |
| GABAR Logistics | 2.1 | 2.1 | same 94 | ~80 min -> 14.1 min |
| GADAR alone Logistics | 36.5 | 36.5 | same 61 | ~80 min -> 19.3 min |

Identical coverage and identical failed-problem sets; plan lengths and
revisit counts identical except GADAR-alone Logistics (average solved length
349.3 vs 349.8, revisits 25842 vs 25861: one solved problem took a slightly
different path - an equal-score tie or fallback order). The batched
constrained path is trusted from here on. Speed: 4-6x on Logistics, little
on Rovers, where the unpruned beam (up to 2 x 3^6 candidates per state, with
a host-side legality loop per beam) dominates rather than featurization.

## 2026-11-27: constrained decoding on the batched evaluator; E5 queued

`beam_search_parallel_constrained` (new; the unconstrained decoders are
untouched): same candidates and scores as beam_search_v2's constrained
decode - GABAR_CHECK_PARALLEL_BEAM on Rovers, 0 mismatches over 500+ state
pairs (GABAR) and 100+ (GADAR alone). Outcome/speed check on the full Rovers
and Logistics test sets running as cdecb-* jobs (logs/cdec/*_batched_*.out,
results cache/results/<config>_cdecb/), against the sequential 46.3 / 57.4 /
2.1 / 36.5. eval_queue.sh needs no change: GABAR_CONSTRAINED_DECODE=1 is now
honoured on the batched path it uses.
E5 (REVIEW_PLAN E5): batch 64 with lr 1e-3 / 2e-3 on Miconic and Visitall,
and Visitall at batch 16, queued in the campaign on 4 dgx2 workers.
Every single-domain model (GABAR x 8, GADAR alone x 8, the E4 ladder, E5) now
gets a constrained evaluation as its own campaign job (`cdec_<config>`,
results cache/results/<config>_cdec/), waiting for its training job where it
has one; E2's evaluations run constrained too. Unconstrained evaluations by
the training jobs are kept, so earlier numbers stay comparable.

## 2026-11-27: decision - constrained decoding is the evaluation protocol

Author's decision: every system reported from now on is evaluated with
GABAR_CONSTRAINED_DECODE=1 (each parameter slot limited to objects some
applicable grounding puts there), GABAR re-run under it too; published
GABAR did not use it, so published numbers are not compared against
constrained ones without saying so. Basis: Rovers GABAR 7.4 -> 46.3, GADAR
alone 20.4 -> 57.4; Logistics GADAR alone 3.1 -> 36.5. A wider unconstrained
beam is not a practical substitute: 9 schemas x 5 objects ran out of GPU
memory, and 2 schemas x 5 objects ran at ~19 h per Rovers evaluation (vs
~25 min constrained) - those two runs were cancelled at 1.6 h (partial:
GABAR 9/54 solved, GADAR alone 11/54). The 9-schema x 3-object runs are
left to finish. Status of "wider beam": cut-ranked on cost, not refuted.
GABAR Rovers, all 9 schemas x 3 objects, unconstrained: 13.0 (7/54; 2 h
batched) - vs 7.4 at 2 schemas and 46.3 constrained. Widening the schema
choice barely helps; top1-valid 4.7.
Note: constrained decoding runs on the SEQUENTIAL evaluator only; the
batched one (eval_queue.sh's default) silently ignores the flag.

## 2026-11-26: E4 Miconic ladder at batch 16 (campaign/reports/e4.txt)

Miconic alone, test split (119), combined checkpoint, one seed, unconstrained
decoding. GABAR through our pipeline: 99.2.

| run | features | cov | no monitor | epoch | combined loss |
|---|---|---|---|---|---|
| sd_union_miconic_bs16 | GABAR's, multi-domain harness | 100.0 | 100.0 | E470 | 10.8 |
| sd_structural_miconic_bs16 | identity removed | 43.7 | 0.0 | E480 | 11.3 |
| sd_joint_lite_miconic_bs16 | + lifted layer | 99.2 | 99.2 | E490 | 11.5 |
| sd_joint_chain_miconic_bs16 | full GADAR | 100.0 | 21.0 | E490 | 5.7 |
| sd_joint_chain_miconic (batch 64, before) | full GADAR | 44.5 | 0.0 | E490 | - |
| sd_joint_chain_miconic_bs16_rs128 | full GADAR, width 128 | 43.7 | 0.0 | E200 | 107.8 |

Pre-registered readings (REVIEW_PLAN E4):
- union 100.0, not "far below 99": the multi-domain harness reproduces
  GABAR on Miconic. Parity gate passes.
- full GADAR 44.5 -> 100.0 at batch 16: MOST OF THE GAP WAS TRAINING LENGTH
  (batch 64 = a quarter of the gradient steps). Per the plan: rerun every
  sd_* at batch 16; the LOO models (also batch 64) likely gain too. E2 must
  run at batch 16.
- structural 43.7 vs union 100.0: removing symbol identity costs ~56
  points (origin/destin/above indistinguishable by class) - and the lifted
  layer alone (joint_lite) recovers it fully (99.2). That is the
  expressiveness tax, measured, and its repair.
- width 128: NOT a capacity reading. Its combined loss is ~108 against ~6
  at width 64 and its best checkpoint is E200: training diverged (lr 5e-4
  unchanged at twice the width is the likely cause). Unexamined, not refuted.
Also found: full GADAR needs the revisit monitor on Miconic (21.0 without)
where joint_lite does not (99.2) - the chain/goal components make the policy
loop-prone here. Worth one line in the E2 analysis.

GABAR through our pipeline, Blocks (baseline_blocks, E500): 52.5 (published
100; GADAR alone at batch 64: 95.5). Another domain where the pipeline does
not reproduce GABAR; recorded, not pursued (decision 2026-11-27).

Beam-width test (logs/beam/): the A=9,K=5 runs ran out of GPU memory (4 GiB
request on a 6 GiB share GPU) - the unpruned beam is 9 x 5^6 candidates on
a 6-parameter schema. The A=9,K=3 and A=2,K=5 runs continue.

## 2026-11-26: constrained decoding on Rovers - the bottleneck was building legal actions

Same checkpoints, re-evaluated with GABAR_CONSTRAINED_DECODE=1 (each
parameter slot limited to objects some applicable grounding puts there;
sequential evaluator, since the batched one has no constraint and
eval_queue.sh forces it - run via logs/cdec/run_cdec.sh on share GPU nodes;
results in cache/results/<config>_cdec/). Rovers test split, 54 problems,
combined checkpoint, one seed. Random policy 30.9.

| model | unconstrained cov | constrained cov | no monitor | top1-valid | max len |
|---|---|---|---|---|---|
| GABAR (baseline_rovers, E280) | 7.4 | **46.3** | 5.6 | 24.6 -> 99.9 | 23 -> 439 |
| GADAR alone (sd_joint_chain_rovers, E380) | 20.4 | **57.4** | 3.7 | 44.0 -> 100.0 | 80 -> 336 |

Reading (the outcome written in the Rovers entry below): coverage jumped
well above random, so building legal actions WAS the main bottleneck, for
both systems. Under the same protocol GADAR beats GABAR on Rovers (57.4 vs
46.3). Ranking is still imperfect: coverage without the monitor stays at
4-6%, ~11k blocked revisits, plans up to 439 of the 500-step bound.
Not concluded: one seed; published GABAR (82) used its own setup; this is a
TEST-TIME protocol change, so it must apply to every system reported, and
it uses nothing beyond the applicable groundings the graph already carries.
Logistics, final (96 problems): GADAR alone 3.1 -> **36.5**
(sd_joint_chain_logistics, E490), GABAR 0.0 -> 2.1 (baseline_logistics,
E380); top1-valid 50.1/27.5 -> 100.0, both still 0.0 without the monitor and
plans up to 489-499 of 500 steps. On Logistics the constraint rescues GADAR
but not GABAR: GABAR's ranking among legal actions is the failure there.

## 2026-11-26: why learned policies lose to random on Rovers (from existing dumps)

Test split, combined checkpoint. `top1-valid` = fraction of steps whose
first-ranked proposal is an applicable action; `repeated` = total revisits
the monitor blocked; successes' plan lengths.

| run | cov | cov no monitor | top1-valid | repeated | avg/max len (solved) |
|---|---|---|---|---|---|
| GABAR, Rovers (baseline_rovers) | 7.4 | 3.7 | 24.6 | 11638 | 14 / 23 |
| GADAR alone, Rovers | 20.4 | 3.7 | 44.0 | 17379 | 34 / 80 |
| GADAR LOO, Rovers (7 folds) | 3.7-11.1 | 0-7.4 | 4.3-46.0 | 3509-5511 | 13-30 / 18-41 |
| GABAR, Miconic (contrast) | 99.2 | 99.2 | 97.8 | 0 | 181 / 295 |
| GADAR alone, Miconic (contrast) | 44.5 | 0.0 | 92.4 | 42768 | 322 / 497 |

Reading (classified per the research skill; not pre-registered):
- Rovers is an ACTION-CONSTRUCTION failure, for GABAR and GADAR alike: the
  top-ranked proposal is applicable on only 4-46% of steps (Miconic: 92-98).
  The executor takes the first applicable proposal in the beam and gives up
  when there is none, while the random policy always picks an applicable
  action - which is how random (30.9) beats them. Only short plans ever
  succeed (max 23-80 steps).
- Miconic GADAR-alone is the opposite: it constructs legal actions (92.4)
  but ranks them badly - zero coverage without the monitor, 42768 blocked
  revisits, plans near the 500-step bound. A RANKING failure, which is what
  E4 probes.
- Not concluded: whether Rovers' failures end on "no applicable proposal"
  or at the step bound (the dumps keep no per-problem step counts).
Cheapest decisive test: re-evaluate the existing Rovers models with
GABAR_CONSTRAINED_DECODE=1 (each parameter slot restricted to objects some
applicable grounding puts there; every proposal applicable by construction;
uses only the groundings the graph already carries). Coverage jumping well
above 30.9 = construction was the bottleneck; staying near 7-20 = ranking too.
Evaluation only, no training.

## 2026-11-26: E0a rule baselines (campaign/reports/e0.txt)

Test split, coverage %, same executor/monitor/500-step bound as the policy,
one rollout (random: 3). Against GADAR zero-shot / LOO in-domain / SD:

| domain | random | achieve | features | GADAR ZS | GADAR LOO | GADAR SD | GABAR ours |
|---|---|---|---|---|---|---|---|
| Blocks | 0.2 | 2.5 | 8.5 | 3.0 | 74.6 | 95.5 | - |
| Gripper | 0.0 | 0.0 | 5.2 | 0 | 94.5 | 100.0 | - |
| Miconic | 1.1 | 2.5 | 60.5 | 0 | 61.9 | 44.5 | 99.2 |
| Visitall | 6.0 | 52.0 | 52.0 | 0 | 81.1 | 54.0 | 70.0 |
| Grid | 0.0 | 0.0 | 0.0 | 0 | 94.6 | 87.5 | - |
| Logistics | 0.0 | 0.0 | 0.0 | 0 | 2.7 | 3.1 | 0.0 |
| Spanner | 43.8 | 39.6 | 39.6 | 0 | 47.9 | 85.4 | 91.7 |
| Rovers | 30.9 | 35.2 | 3.7 | 0 | 9.0 | 20.4 | 7.4 |

Pre-registered readings (REVIEW_PLAN E0a) applied:
- Blocks: `features` 8.5 >= GADAR zero-shot 3.0 -> the network adds nothing
  over the hand-coded goal features zero-shot on the paper's one nonzero
  zero-shot domain. Do not claim learned transfer there.
- Miconic, Visitall, Spanner: `features` high (40-60) where GADAR zero-shot
  is 0 -> the information is in the graph and the network fails to use it
  zero-shot: a learning problem, not a representation problem.
Not pre-registered, recorded as found: on Rovers a UNIFORM RANDOM policy
(30.9) beats every learned policy, GABAR through our pipeline included
(GADAR LOO 9.0, SD 20.4, GABAR 7.4); on Spanner random (43.8) is within 4
points of GADAR LOO (47.9). A learned policy below random points at the
executor or the learned ranking trapping itself (revisits, a 500-step bound
spent looping), not at missing information - worth checking before any
Rovers number is interpreted.

## 2026-11-27: E4 - why single-domain GADAR trails GABAR (Miconic 44.5 vs 99.2)

Found: every sd_* and LOO run trains at batch 64, the GABAR baselines at 16
(4x fewer gradient steps over the same 500 epochs). E4 = a five-rung Miconic
ladder at batch 16 (union -> structural -> joint_lite -> joint_chain, plus
joint_chain at width 128), priority 1 in the campaign. Readings in
REVIEW_PLAN E4. Hidden width is now in the checkpoint key (`rs`). E2 held
until E4 reads out (ablate the model E4 settles on).

## 2026-11-27: decision - published GABAR numbers used as given; E2 released

No further GABAR runs (reproduction jobs held). On Logistics/Rovers the
GADAR-vs-GABAR gap is across pipelines and is not attributed to the
representation (REVIEW_PLAN E1).

## 2026-11-26: E1 says our pipeline does not reproduce GABAR (Logistics, Rovers)

GABAR through our pipeline: Logistics 0.0% (LAMA) / 0.0% (optimal), Rovers
7.4% / 5.6%, against published 79 / 82. Miconic 99.2, Spanner 91.7, Visitall
70.0. Pre-registered reading "low | low": stop and debug. E2 trainings HELD;
`e1_gabar_{logistics,rovers}_prejuly` (GABAR at 536c352) decide regression
vs. setup. Single-domain GADAR after the single-effect fix: Miconic 41.2 ->
44.5 (noise), Rovers 7.4 -> 20.4. The submitted Table 2 multi-domain columns
are unchanged by the reporting fix (BIND Gripper 95.4 -> 95.5 only).
Campaign: workers now queue their own successors and re-read the manifest.

## 2026-11-14: negative preconditions keep their sign (lifted layer)

REVIEW_PLAN 2e. No width change; graphs of domains without a negative
precondition are byte-identical (50/50 fingerprints locally), so nothing
trained on the single-effect-fixed featurizer is invalidated - provided the
audit shows no version bit 2 on the eight real domains (not yet run there).

## 2026-11-13: audit of the post-July code

Asked for after the single-effect bug. Result: one more silent
wrong-number path, in reporting, not in the model - a checkpoint that is the
best under two metrics left an empty `combined` cell and
`tools/indomain_table.py` filled it from an older dump (REVIEW_PLAN 2c;
fixed in `run_tests` and in the tool, which now prints NOTE lines). The
featurizer is now checked against the PDDL text by
`tools/audit_featurizer.py` (campaign gate `gate_audit_featurizer`). What was
and was not verified: REVIEW_PLAN 2d. TODO on the cluster: run the audit on
the eight real domains, and re-run `tools/indomain_table.py` to see whether
any submitted Table 2 cell moves.

## 2026-11-12: featurizer bug found by the first campaign run

The E0 rule baseline crashed in Miconic on `op.effects.literals`; a cluster
diagnostic confirmed pddlgym leaves one-effect schemas as bare Literals and
that exactly two schemas in the eight domains are such: Miconic `board`,
Rovers `calibrate`. The lifted layer dropped them. Fixed (`op_literals`),
keyed (`fx`, `_fx1`), and every lifted-layer model trained before is on
different graphs for those two domains. Details and consequences:
ICLR27/REVIEW_PLAN.md 2b. Also: gate_fingerprint passes (0 mismatches on
320 states x 2 featurizations), so the ablation-mode refactor itself is
byte-identical.

## 2026-11-11: ICLR-27 reviews (1/1/2) -> experiment plan

Plan, commands, pre-registered readings and cut order: `ICLR27/REVIEW_PLAN.md`.
Nothing below has been run yet.

| id | experiment | answers | cost |
|---|---|---|---|
| E0a | rule baselines (`random_policy_baseline.py --policy features|achieve`) | is anything learned beyond the hand-coded goal features | CPU, hours |
| E0b | LAMA coverage + wall-clock from existing dumps | heuristic-search comparison | free |
| E1 | GABAR through our pipeline on all 8; Logistics/Rovers on optimal plans | is the in-domain gap pipeline, training data or representation | ~1 GPU-day |
| E2 | component ablations `jc_no_{bind,chain,eff,goal}` + `joint`, 3 folds | which component matters | ~8 GPU-days |
| E3 | GOOSE DI-LLG + GBFS | learned-heuristic baseline | stretch |

Found while planning (see the plan, section 2): the rung reported as
"GADAR-BIND" in September was `joint_lite`, which has no occurrence nodes
and no binding edges; binding has never been isolated in the type-compiled
models. And the training planner was in neither the plan-cache name nor the
checkpoint key until now, so optimal-vs-satisficing training data could not
be varied without silently reusing LAMA plans.
