#!/bin/bash
# slurm_train_queue.sbatch — run a training queue as a BATCH job.
#
# Why this exists: launching from `srun --pty bash` ties every run to that
# interactive shell. When the shell exits - auto-logout, a dropped ssh, the
# allocation's walltime - SLURM revokes the allocation and kills everything
# in its cgroup. `nohup` protects against SIGHUP and is irrelevant to that,
# so the queue dies mid-config having written no error anywhere, leaving a
# stale logs/<name>.running behind. A batch job is owned by SLURM instead of
# by a terminal, so it survives logout and gets its own walltime.
#
# Usage (sbatch options go on the COMMAND LINE, so one file serves both
# nodes/partitions):
#
#   sbatch -A eecs -p dgxh --gres=gpu:1 -c 32 --mem=128G -t 2-00:00:00 \
#          --constraint=vram80g -J gadarA -o logs/slurm_%j.out \
#          train_test_scripts/slurm_train_queue.sbatch \
#          configs/loo8_joint_chain_no_visitall.yaml \
#          configs/loo8_joint_chain_no_gripper.yaml
#
# Then: squeue -u $USER      # is it running
#       scancel <jobid>      # stop it
#       tail -f logs/slurm_<jobid>.out
#
# EXTRA is passed through to every run, e.g.
#   sbatch --export=ALL,EXTRA="--checkpoint-every 50" ...
set -u

echo "=== job ${SLURM_JOB_ID:-none} on $(hostname) at $(date '+%F %T') ==="
echo "    node=${SLURM_JOB_NODELIST:-?}  cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "    configs: $*"
echo

# sbatch inherits the submitting shell's environment (--export=ALL is the
# default), so the conda env comes along - but only if it was active at
# submit time. Check it here rather than discovering it in a traceback 20
# minutes in: this exact failure ("No module named 'torch'") has already
# eaten one campaign.
if ! python -c "import torch" 2>/dev/null; then
    echo "FATAL: 'import torch' fails in this job."
    echo "  python: $(command -v python 2>/dev/null || echo none)"
    echo "  Submit from a shell with the conda env active, so sbatch's"
    echo "  default --export=ALL carries it in."
    exit 1
fi
echo "    python: $(command -v python)  torch: $(python -c 'import torch; print(torch.__version__)')"
echo "    cuda available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo

exec "$(dirname "$0")/train_queue.sh" cuda:0 "$@"
