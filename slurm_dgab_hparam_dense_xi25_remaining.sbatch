#!/bin/bash
#SBATCH --job-name=dgab_hp_dense25
#SBATCH --comment="DGAB RQ4 dense xi=25, excluding alpha=0 and shared default"
#SBATCH --partition=amd
#SBATCH --output=log/%x_%A_%a.out
#SBATCH --error=log/%x_%A_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-79%24

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate autobidding

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Dense slots 3 (shared default) and 5 (alpha=0) are intentionally omitted.
# Slot mapping follows main.hparam_train.configuration in the DGAB project.
SLOTS=(0 1 2 4 6 7 8 9)
SLOT_INDEX=$((SLURM_ARRAY_TASK_ID / 10))
EPISODE=$((SLURM_ARRAY_TASK_ID % 10))
SLOT=${SLOTS[$SLOT_INDEX]}
TASK_ID=$((SLOT * 10 + EPISODE))

SELECTION_ROOT="${DGAB_HPARAM_ROOT:-/share/home/u10092/project/lsw/dgabv2/dgab/hyperparameter_results}"
# A distinct root is essential: old xi=1.0 success markers must not be reused.
OUTPUT_ROOT="${DGAB_HPARAM_DENSE_XI25_OUTPUT:-exp_data/dgab_hparam_dense_xi25_ep0_9}"

echo "start=$(date --iso-8601=seconds) slot=${SLOT} episode=${EPISODE} xi=25"
srun python -u run_dgab_hparam_online.py \
  --selection-root "${SELECTION_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --task-id "${TASK_ID}"
echo "finish=$(date --iso-8601=seconds) slot=${SLOT} episode=${EPISODE}"
