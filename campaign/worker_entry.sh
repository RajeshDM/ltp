#!/bin/bash
# Runs INSIDE a SLURM allocation (campaign/submit.sh submits it, or paste it in
# an interactive srun session): activate the environment, then become a worker.
#   bash campaign/worker_entry.sh [gpu|cpu]     # which kind of job to prefer
# (or CAMPAIGN_KIND=gpu|cpu in the environment, which is how submit.sh passes
# it: this cluster's sbatch will not take arguments after --wrap)
# A batch shell has not read ~/.bashrc, so conda is sourced explicitly.
set -u
cd "$(dirname "$0")/.." || exit 1
KIND=${1:-${CAMPAIGN_KIND:-}}
CONDA_SH=${CONDA_SH:-/nfs/hpc/share/mangannr/anaconda3/etc/profile.d/conda.sh}
if [ ! -f "$CONDA_SH" ] && command -v conda >/dev/null 2>&1; then
    CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
fi
# shellcheck disable=SC1090
[ -f "$CONDA_SH" ] && source "$CONDA_SH" && conda activate "${CAMPAIGN_ENV:-di_ltp_1}"
if ! python -c "import torch" 2>/dev/null; then
    echo "REFUSING: 'import torch' fails with $(command -v python || echo 'no python')."
    echo "  conda.sh tried: $CONDA_SH   env: ${CAMPAIGN_ENV:-di_ltp_1}"
    echo "  Set CONDA_SH=/path/to/etc/profile.d/conda.sh if conda lives elsewhere."
    exit 1
fi
echo "campaign worker: $(hostname) job=${SLURM_JOB_ID:-none} $(date '+%F %T')"
nvidia-smi -L 2>/dev/null | head -2 || echo "(no GPU)"
exec python campaign/worker.py run ${KIND:+--kind "$KIND"}
