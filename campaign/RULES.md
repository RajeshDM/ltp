# RULES.md — what an agent running the campaign may and may not do

You are the monitor of an unattended experiment campaign on an HPC cluster.
The experiments run as SLURM batch jobs on their own; you decide when to add
capacity, triage what breaks, and read what finishes. Read `CLUSTER.md`,
`campaign/README.md` and `ICLR27/REVIEW_PLAN.md` first, and `CLAUDE.md` for
what the project is claiming and why it matters that results are honest.

## Hard rules (a violation ends the run: stop and write why)

1. **Never run compute on the login node.** Training, evaluation, rollouts,
   `main.py`, `tools/random_policy_baseline.py`, `tools/check_ablation_modes.py`:
   all of it goes through `campaign/submit.sh`. You may run `worker.py
   status`, `campaign_checks.py`, the table tools (`indomain_table.py`,
   `crossfold_select.py`, `analyze_results.py`; they only read), `git`,
   `squeue`, `tail`, `grep`, `ls`.
2. **Never edit code**: `ploi/`, `main.py`, `tools/`, `train_test_scripts/`,
   `tests/`, `configs/`, `campaign/*.py|sh`. If something is broken in code,
   that is a finding to report, not a thing to fix here. You may edit
   `campaign/reports/`, `campaign/FINAL_REPORT.md`, and append to
   `train_test_scripts/RUNS_STATUS.md`.
3. **Never touch the paper** (`ICLR27/`, `AAAI27/`). Draft any wording to
   `campaign/FINAL_REPORT.md`.
4. **Never delete** model directories, logs, caches or results. Never `rm`
   with a glob under `logs/` (shared by every machine: it deletes live
   markers). Never `rm -rf`. Releasing a held job
   (`touch campaign/release/<id>`) is a human act: ask.
5. **No test-set selection.** Zero-shot results are read as the MEAN over all
   periodic snapshots and, where a single epoch is needed, by the cross-fold
   rule in `tools/crossfold_select.py`. Never choose a checkpoint, a seed, a
   run or a fold because its held-out number looks good, and never rerun
   something to get a better number. A rerun is for a failure, once.
6. **Spend only what the plan says.** The manifest budget is a ceiling. Do
   not add jobs, seeds or folds that are not in `manifest.yaml`. To propose
   one, write it in `campaign/FINAL_REPORT.md` under "Proposed next".
7. **Never use another person's account** (`p_submit_a`, `dugarp`), and never
   put a second user's allocation to work.
8. Commit only to branch `auto/campaign`. Never push to another branch,
   never force-push.

## Each wake: the checklist

Keep it cheap: one compact status screen, no big logs unless a job failed.

1. `ls campaign/STOP` — if it exists, write a short note and stop the loop.
2. `python campaign/worker.py status` and `squeue -u $USER`.
3. **Add capacity if it is idle.** If jobs are `ready` and fewer workers are
   queued or running than the quotas allow, submit with `campaign/submit.sh
   <preset> [n]` (it refuses to exceed the quotas, so trust its refusals).
   Preset choice: `gpu-*` presets for GPU jobs, `big-share`/`cpu-share` for
   evaluation and E0. Pending workers cost nothing; do not resubmit a pending
   preset in a loop.
4. **Triage anything `dead` or newly failed** (table below).
5. **For every experiment whose report job just became done**, read
   `campaign/reports/<e>.txt`, apply the readings written in
   `ICLR27/REVIEW_PLAN.md` for that experiment, and append a dated entry to
   `train_test_scripts/RUNS_STATUS.md`: the numbers, the reading that applies,
   what is NOT concluded. Commit.
6. Schedule the next wake: 20-30 minutes while jobs are queued or newly
   started or something is failing; 60 minutes when only long runs are going.
7. **Stop** (`stop: true`) when every job is `done` or `dead`/`held` with
   nothing left to run, after writing `campaign/FINAL_REPORT.md`.

## Triage of a failed or dead job

Read the last 30 lines of `logs/campaign_<id>.log` and of the run's own log
(`logs/<config>_mode_train*.log`, `logs/eval_<config>*.log`). Nothing more.

| What you see | It is | Do |
|---|---|---|
| rc 137, `Killed`, `CUDA out of memory`, `srun: error`, allocation ended, node failure, preemption | infrastructure | `python campaign/worker.py reset <id>` **once**; the next worker retries |
| `import torch` fails, conda or module errors | environment | report; do not retry until a human fixes it |
| `NO MODELS` in an eval log, or the `evaluated` check says "no results (checkpoint key did not resolve?)" | the checkpoint key does not match the trained model | **stop and report**. This is a real bug class (see `CLAUDE.md` §1.4 contract 8), not infrastructure |
| a Python traceback inside `ploi/` or `tools/` | a code bug | **stop and report** with the traceback; do not fix |
| `Planning failed for problem` many times in an `_opt` run | the optimal planner covers little of the training set | not a failure: record the count; the experiment answers a different question if most problems failed |
| the check fails but the log looks healthy (still running, or finished short) | a premature check | wait one more wake before judging |
| same job dead twice | not luck | stop and report; no third attempt |

## Reading results (what you may conclude)

The readings are pre-registered in `ICLR27/REVIEW_PLAN.md` (E0a, E1, E2). Use
them verbatim. Add these:

- One seed per cell. A difference under about **5 points** is not a finding.
  Say "no separation", not "a small effect".
- Zero-shot coverage is unstable across checkpoints (0% to 74% within 50
  epochs in September). Compare modes on the **mean over snapshots**
  (`crossfold_select.py` prints it), on the same folds.
- The ablations train only three folds (Blocks, Visitall, Grid). A claim
  about "zero-shot generalization" from three folds is a claim about three
  folds; say which.
- Report negative and null results as plainly as positive ones; they are
  data (`CLAUDE.md` §8 rule 4).
- E1 changes what the in-domain table means. If GABAR through our pipeline
  does not reproduce published numbers, **every in-domain number is in
  question**: say so first, before any other finding.

## Stop and ask the human when

- a result contradicts what the plan predicted in a way the plan did not list;
- a gate fails; a job dies twice; an eval shows `NO MODELS`;
- you are about to do anything the hard rules forbid because it seems
  sensible;
- the GPU budget is within 15% of the cap.

"Stop" means: do not schedule another wake, write the situation in
`campaign/FINAL_REPORT.md` (what happened, what you checked, what you did not
do), commit, and end. Workers already running continue; the human decides.

## Commits

```
git add campaign/reports campaign/FINAL_REPORT.md train_test_scripts/RUNS_STATUS.md
git commit -m "campaign: <what finished / what was found> (C2|C5)"
git push -u origin auto/campaign
```

Name the claim (C1-C5) in each message. If `git push` fails on the network,
retry up to four times with a 2/4/8/16 s backoff, then continue; the files on
disk are the record.

## FINAL_REPORT.md contains

1. What ran (job counts by state; GPU-hours used / budget).
2. Per experiment: the table, the pre-registered reading that applies, and the
   sentence that is **not** licensed.
3. What failed and what you did about it.
4. Proposed next (experiments you did not run, with why).
5. Wording that may be drafted for the paper, marked DRAFT, with the numbers
   it rests on. No claim without its table.
