#!/bin/bash
#SBATCH --job-name=qga_dgab_sp_new
#SBATCH --comment="QGA and DGAB sparse PV 0.00055/0.0006, episodes 10-29"
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

DGAB_MODEL_ROOT=saved_model/dgab_sparse_20260927172551

PVS=(0.00055 0.0006)
STRATEGIES=(QGA DGABShare)
EPISODE_COUNT=20
TASKS_PER_PV=$((${#STRATEGIES[@]} * EPISODE_COUNT))
PV_INDEX=$((SLURM_ARRAY_TASK_ID / TASKS_PER_PV))
LOCAL_ID=$((SLURM_ARRAY_TASK_ID % TASKS_PER_PV))
STRATEGY_INDEX=$((LOCAL_ID / EPISODE_COUNT))
EPISODE=$((LOCAL_ID % EPISODE_COUNT + 10))
PV=${PVS[$PV_INDEX]}
STRATEGY=${STRATEGIES[$STRATEGY_INDEX]}

if [[ "${STRATEGY}" == "QGA" ]]; then
  MODEL_ROOT=saved_model/sparse_benchmark
else
  MODEL_ROOT=${DGAB_MODEL_ROOT}
fi
V_GOAL_MULTIPLIER=1.0  # DGAB fixed; unused by QGA

OUTPUT_ROOT="${OUTPUT_ROOT:-exp_data/qga_dgab_sparse_pv_ep10_29}"
OUTPUT_DIR="${OUTPUT_ROOT}/dgab_latest_xi_1.0/pv_${PV}/${STRATEGY}"
OUTPUT="${STRATEGY}_pv_${PV}_ep${EPISODE}"
mkdir -p "${OUTPUT_DIR}"

echo "start=$(date --iso-8601=seconds) strategy=${STRATEGY} pv=${PV} episode=${EPISODE} model_root=${MODEL_ROOT} xi=${V_GOAL_MULTIPLIER}"
srun python run_sparse_model_benchmark.py \
  --budget_rate 1.0 \
  --episodes "${EPISODE}" \
  --strategies "${STRATEGY}" \
  --pv "${PV}" \
  --pv_num 500000 \
  --seed 42 \
  --device cpu \
  --model_root "${MODEL_ROOT}" \
  --v_goal_multiplier "${V_GOAL_MULTIPLIER}" \
  --output_dir "${OUTPUT_DIR}" \
  --output "${OUTPUT}"
echo "finish=$(date --iso-8601=seconds) strategy=${STRATEGY} pv=${PV} episode=${EPISODE}"
