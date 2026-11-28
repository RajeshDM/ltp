#!/bin/bash
# Pull the second account's share of the protocol campaign into this repo, so
# the dashboard and the done-checks see it: its job status (JSON), the ladder
# runs' training logs, and their result dumps (text only - no models, no
# caches). Read-only on the other side. Run before tools/campaign_dashboard.py.
#   SECOND_HOST=dugarp@submit-a.hpc.engr.oregonstate.edu
#   SECOND_REPO=/nfs/hpc/share/dugarp/rajesh/gadar/2_full_repo/ltp
cd "$(dirname "$0")/.." || exit 1
H=${SECOND_HOST:-dugarp@submit-a.hpc.engr.oregonstate.edu}
R=${SECOND_REPO:-/nfs/hpc/share/dugarp/rajesh/gadar/2_full_repo/ltp}
PY=${SECOND_PY:-/nfs/hpc/share/dugarp/miniconda3/envs/ltp_3/bin/python}
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 "$H")
clean() { grep -v -E 'Oregon State|Unauthorized|Consent to|Expectation of|^=+$|bind: warning|^$'; }
"${SSH[@]}" "cd $R && CAMPAIGN_MANIFEST=campaign/manifest_second.yaml $PY campaign/worker.py status --all --json" \
    2>/dev/null > campaign/reports/second_status.json.tmp \
  && python -c "import json; json.load(open('campaign/reports/second_status.json.tmp'))" 2>/dev/null \
  && mv campaign/reports/second_status.json.tmp campaign/reports/second_status.json \
  || { echo "sync_second: status unavailable"; rm -f campaign/reports/second_status.json.tmp; }
"${SSH[@]}" "cd $R && tar -cf - --ignore-failed-read \$(ls -d logs/p2_lad_* logs/protocol/p2_lad_* cache/results/p2_lad_* 2>/dev/null) 2>/dev/null" \
    2> >(clean >&2) | tar -xf - 2>/dev/null
echo "sync_second: $(ls -d cache/results/p2_lad_*_final 2>/dev/null | wc -l) final dumps, $(ls logs/p2_lad_*_mode_train*.log 2>/dev/null | wc -l) train logs"
