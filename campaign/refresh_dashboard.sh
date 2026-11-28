#!/bin/bash
# One dashboard refresh, in the order that matters: pull the second account's
# share, re-queue evaluations the selection rule disagrees with, verify every
# finished run, then regenerate the page. Publishing it is the agent's step.
cd "$(dirname "$0")/.." || exit 1
timeout 300 campaign/sync_second.sh > /dev/null 2>&1
timeout 900 python tools/requalify_selection.py | grep -v "still running"
timeout 900 python tools/verify_runs.py
timeout 600 python tools/campaign_dashboard.py
