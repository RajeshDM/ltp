#!/bin/bash
# Once the plan collection is done (campaign/reports/p2_collect.txt), copy the
# plan caches and the report to the second account and start its workers at
# ~75% of its caps (dgx2 3, gpu 3, dgxh 1, share 1). Parity of its stack was
# verified 2026-11-27 (our Blocks GADAR model: 92/18/0 on both accounts).
# Run with nohup on a node that can ssh to dugarp@submit-a (the claude host).
cd /nfs/hpc/share/mangannr/di_ltp/1_di_ltp/ltp || exit 1
H=dugarp@submit-a.hpc.engr.oregonstate.edu
R2=/nfs/hpc/share/dugarp/rajesh/gadar/2_full_repo/ltp
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 "$H")
until [ -s campaign/reports/p2_collect.txt ]; do sleep 120; done
echo "$(date) collection done; copying caches"
tar -cf - cache/results/*_unified_cache_*.pkl campaign/reports/p2_collect.txt \
  | "${SSH[@]}" "cd $R2 && tar -xf - && ls cache/results/*_unified_cache_*.pkl | wc -l" || exit 1
"${SSH[@]}" "cd $R2 && git pull -q origin auto/campaign && bash -lc 'module load slurm 2>/dev/null
  export CAMPAIGN_MANIFEST=campaign/manifest_second.yaml CAMPAIGN_ENV=ltp_3
  export CONDA_SH=/nfs/hpc/share/dugarp/miniconda3/etc/profile.d/conda.sh
  export PYTHONPATH=/nfs/hpc/share/dugarp/rajesh/gadar/p2/pddlgym
  bash campaign/submit.sh gpu-dgx2 3; bash campaign/submit.sh gpu-gpu 3
  bash campaign/submit.sh gpu-dgxh 1; bash campaign/submit.sh gpu-share 1'"
echo "$(date) second-account workers submitted"
