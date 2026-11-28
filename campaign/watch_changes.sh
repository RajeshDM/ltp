#!/bin/bash
# Block until the 2026-11 protocol campaign changes state, then print what
# changed and exit. A Claude session runs this in the background; its exit is
# the signal to regenerate the dashboard (tools/campaign_dashboard.py) and
# republish the plan page. Exits anyway after MAX_MIN (default 90) so ETAs
# stay fresh. Reads only files and squeue - safe from the login node.
cd "$(dirname "$0")/.." || exit 1
POLL=${POLL_SEC:-300}; MAX=${MAX_MIN:-90}
sig() {
    python campaign/worker.py status --all --json 2>/dev/null \
      | python -c "import json,sys; print('\n'.join(sorted(f\"{r['id']} {r['state']}\" for r in json.load(sys.stdin) if r['id'].startswith(('p2','gate_p2')))))"
    ls campaign/reports/p2_smoke.txt campaign/reports/p2_collect.txt 2>/dev/null
}
start=$(date +%s); before=$(sig)
while :; do
    sleep "$POLL"
    now=$(sig)
    if [ "$now" != "$before" ]; then
        echo "CHANGED at $(date '+%F %T')"; diff <(echo "$before") <(echo "$now") | grep '^[<>]'; exit 0
    fi
    [ $(( $(date +%s) - start )) -ge $(( MAX * 60 )) ] && { echo "HEARTBEAT at $(date '+%F %T')"; exit 0; }
done
