# tmp_scripts/ — throwaway

Scripts here exist for ONE campaign and are meant to be deleted afterwards.
Nothing in the repo imports from here, and nothing here is a dependency of
`main.py`, `configs/` or `tools/`.

To clean up when the campaign is over:

    rm -rf tmp_scripts/

If something in here turns out to be worth keeping, move it to
`train_test_scripts/` (launchers) or `tools/` (analysis) and document it in
CLAUDE.md §1.1 — do not let it quietly become load-bearing while living in a
directory named "tmp".

## Current contents

- `queue_runner.sh` — dynamic scheduler: takes a job file, keeps as many
  trainings running as MEMORY (not cores) allows, refills a slot the moment
  one frees, and retries a run that dies in its first five minutes. Built
  after four concurrent 8-domain trainings were OOM-killed at "Epoch 0" on a
  32-core node — the fixed per-phase concurrency in `run_campaign.sh` was
  sized from the wrong resource.
- `status.sh` — what is actually running here, what each run is doing, and
  which launched runs are gone plus the traceback that killed them.

- `run_campaign.sh` — unattended two-day campaign driver for the two H100
  nodes. Detaches itself, runs every phase in order, reports to wandb.
  See RUNBOOK.md, "The two-H100 campaign", for what it is doing and why.
