# LOOP_PROMPT.md — the prompt that drives the monitoring loop

Paste this (or just `/loop Read campaign/RULES.md, then follow campaign/LOOP_PROMPT.md.`)
into Claude Code on the cluster, in the repo, on branch `auto/campaign`.

---

You are the monitor of the experiment campaign in this repository. Your rules
are in `campaign/RULES.md`; read it, `CLUSTER.md` and `ICLR27/REVIEW_PLAN.md`
now if you have not in this session. Do exactly one pass of the wake
checklist in RULES.md:

1. `ls campaign/STOP` (stop if present).
2. `python campaign/worker.py status`; `squeue -u $USER`.
3. Add worker capacity with `campaign/submit.sh` if jobs are ready and the
   quotas allow.
4. Triage dead or newly failed jobs by the table in RULES.md.
5. For each experiment whose report just became available, apply the
   pre-registered reading, append to `train_test_scripts/RUNS_STATUS.md`,
   commit to `auto/campaign`.
6. Schedule the next wake with the same prompt: 20-30 minutes while anything
   is queued, starting or failing; 60 minutes when only long runs are going.
7. When nothing is left to run, write `campaign/FINAL_REPORT.md`, commit, and
   stop the loop.

Be brief. Do not read large logs unless a job failed. If a rule in RULES.md
says stop and ask, stop: write the situation to `campaign/FINAL_REPORT.md`,
commit, and do not schedule another wake.
