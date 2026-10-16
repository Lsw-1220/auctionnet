#!/bin/bash
#SBATCH --job-name=dgab_hparam_online
#SBATCH --comment="DGAB expectile/alpha reference-condition online evaluation"
#SBATCH --partition=amd
#SBATCH --output=log/%x_%A_%a.out
#SBATCH --error=log/%x_%A_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-199%24

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate autobidding

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SELECTION_ROOT="${DGAB_HPARAM_ROOT:-/share/home/u10092/project/lsw/dgabv2/dgab/hyperparameter_results}"
OUTPUT_ROOT="${DGAB_HPARAM_ONLINE_OUTPUT:-exp_data/dgab_hparam_online}"

echo "start=$(date --iso-8601=seconds) task=${SLURM_ARRAY_TASK_ID} selection_root=${SELECTION_ROOT}"
srun python -u run_dgab_hparam_online.py \
  --selection-root "${SELECTION_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --task-id "${SLURM_ARRAY_TASK_ID}"
echo "finish=$(date --iso-8601=seconds) task=${SLURM_ARRAY_TASK_ID}"
