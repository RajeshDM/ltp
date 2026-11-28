#!/bin/bash
# Fill the plan/graph caches of the 2026-11 protocol once (validation splits of
# all 10 domains; Spanner's extended and the typed twins' training sets), so
# training jobs do not each re-plan them. Spanner Val-OOD alone is ~2 h of LAMA.
cd /nfs/hpc/share/mangannr/di_ltp/1_di_ltp/ltp || exit 1
source /nfs/hpc/share/mangannr/anaconda3/etc/profile.d/conda.sh && conda activate di_ltp_1
[ -n "${SLURM_JOB_ID:-}" ] || { echo "refusing outside SLURM"; exit 1; }
# Fast Downward fails every problem in <1 s on el8 nodes (cn-a/b/c): plans
# would be missing, not wrong, but the job would be wasted. Submit with
# --constraint=el9.
grep -q "release 9" /etc/redhat-release || { echo "refusing on $(hostname): not el9 (FD does not run)"; exit 1; }
pids=()
for c in p2_sd_gabar_spanner p2_sd_gabar_manyblocks p2_sd_gabar_gripper p2_sd_gabar_miconic p2_sd_gabar_visitall \
         p2_sd_gabar_grid p2_sd_gabar_logistics p2_sd_gabar_rovers p2_typed_gabar_grid p2_typed_gabar_logistics; do
    python main.py --config configs/$c.yaml --mode collect --device cpu > logs/protocol/collect_$c.log 2>&1 &
    pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
grep -h "\[collect\]" logs/protocol/collect_*.log
grep -l "Traceback" logs/protocol/collect_*.log && rc=1
[ $rc -eq 0 ] && grep -h "\[collect\]" logs/protocol/collect_*.log > campaign/reports/p2_collect.txt
exit $rc
