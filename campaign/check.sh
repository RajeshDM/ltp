#!/bin/bash
# One screen of everything: allocations, jobs by state, why anything died, results so far.
#   campaign/check.sh          (alias:  camp)
cd "$(dirname "$0")/.." || exit 1
command -v squeue >/dev/null 2>&1 || { [ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh; command -v module >/dev/null 2>&1 && module load slurm 2>/dev/null; }
echo "=== $(date '+%F %T')  branch $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
echo; echo "=== your SLURM jobs (R = running, PD = waiting for a node)"
squeue -u "${USER:-$(id -un)}" -o "%.10i %.9P %.17j %.3t %.10M %.5c %.30R" 2>/dev/null || echo "(squeue not available)"
echo; echo "=== campaign jobs"
python campaign/worker.py status 2>&1 | head -30
DEAD=$(python campaign/worker.py status --json 2>/dev/null | python -c "
import json,sys
try:
    print(' '.join(r['id'] for r in json.load(sys.stdin) if r['state']=='dead'))
except Exception:
    pass")
if [ -n "$DEAD" ]; then
    echo; echo "=== DEAD jobs: last lines of each log (paste these)"
    for id in $DEAD; do echo "--- $id"; tail -8 "logs/campaign_$id.log" 2>/dev/null | cut -c1-200; done
fi
echo; echo "=== results written so far (campaign/reports)"
ls -l campaign/reports 2>/dev/null | grep -v "^total\|gitkeep" || echo "(none yet)"
echo; echo "=== most recently written campaign logs"
ls -t logs/campaign_*.log 2>/dev/null | head -4 | while read -r f; do printf '%s  %s\n' "$(date -r "$f" '+%m-%d %H:%M')" "$f"; done
