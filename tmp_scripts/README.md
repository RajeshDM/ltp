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

- `run_campaign.sh` — unattended two-day campaign driver for the two H100
  nodes. Detaches itself, runs every phase in order, reports to wandb.
  See RUNBOOK.md, "The two-H100 campaign", for what it is doing and why.
