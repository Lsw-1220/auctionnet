#!/bin/bash
#SBATCH --job-name=dgab_sparse
#SBATCH --comment="DGABShare pre/post sparse PV and v-goal retest with default opponents"
#SBATCH --partition=amd
#SBATCH --output=log/%x_%A_%a.out
#SBATCH --error=log/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-419%16

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p log

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate autobidding

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MODEL_NAMES=(dgab_pre dgab_post)
MODEL_DIRS=(
  saved_model/dgab_sparse_retest/dgab_pre
  saved_model/dgab_sparse_retest/dgab_post
)
MODEL_CKPTS=(step_15000.pt best_actor.pt)

PV_VALUES=(0.0003 0.00035 0.0004 0.00045 0.0005 0.00055 0.0006)
PV_TAGS=(00030 00035 00040 00045 00050 00055 00060)
VGOAL_VALUES=(1 1.1 1.2)
VGOAL_TAGS=(100 110 120)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
MODEL_INDEX=$((TASK_ID / 210))
WITHIN_MODEL=$((TASK_ID % 210))
PV_INDEX=$((WITHIN_MODEL / 30))
WITHIN_PV=$((WITHIN_MODEL % 30))
EPISODE=$((WITHIN_PV / 3))
VGOAL_INDEX=$((WITHIN_PV % 3))

MODEL_NAME="${MODEL_NAMES[$MODEL_INDEX]}"
MODEL_DIR="${MODEL_DIRS[$MODEL_INDEX]}"
MODEL_CKPT="${MODEL_CKPTS[$MODEL_INDEX]}"
PV="${PV_VALUES[$PV_INDEX]}"
PV_TAG="${PV_TAGS[$PV_INDEX]}"
VGOAL="${VGOAL_VALUES[$VGOAL_INDEX]}"
VGOAL_TAG="${VGOAL_TAGS[$VGOAL_INDEX]}"
printf -v EP_TAG "%02d" "${EPISODE}"

OUTPUT_DIR="exp_data/dgab_sparse_retest/${MODEL_NAME}/vgoal_${VGOAL_TAG}/pv_${PV_TAG}/shards"
OUTPUT_NAME="${MODEL_NAME}_vgoal_${VGOAL_TAG}_pv_${PV_TAG}_ep${EP_TAG}"
mkdir -p "${OUTPUT_DIR}"

echo "start=$(date --iso-8601=seconds)"
echo "job=${SLURM_ARRAY_JOB_ID} task=${TASK_ID} host=$(hostname)"
echo "model=${MODEL_NAME} pv=${PV} episode=${EPISODE} v_goal_multiplier=${VGOAL}"

srun python benchmark_multistrat.py \
  --budget_rate 1.0 \
  --episodes "${EPISODE}" \
  --strategies DGABShare \
  --pv "${PV}" \
  --pv_num 500000 \
  --seed 42 \
  --device cpu \
  --dgabshare_dir "${MODEL_DIR}" \
  --dgabshare_ckpt "${MODEL_CKPT}" \
  --v_goal_multiplier "${VGOAL}" \
  --exploration_scale 0 \
  --output_dir "${OUTPUT_DIR}" \
  --output "${OUTPUT_NAME}" \
  --fail_fast

echo "finish=$(date --iso-8601=seconds)"
