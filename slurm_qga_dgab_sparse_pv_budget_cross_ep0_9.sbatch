#!/bin/bash
#SBATCH --job-name=qga_dgab_sp_grid
#SBATCH --comment="QGA/DGAB sparse PV x budget grid, episodes 0-9"
#SBATCH --partition=amd
#SBATCH --output=log/%x_%A_%a.out
#SBATCH --error=log/%x_%A_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-319%24

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate autobidding

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PVS=(0.0004 0.00045 0.00055 0.0006)
BUDGET_RATIOS=(0.5 0.75 1.25 1.5)
STRATEGIES=(QGA DGABShare)
EPISODE_COUNT=10

TASKS_PER_PV=$((${#BUDGET_RATIOS[@]} * ${#STRATEGIES[@]} * EPISODE_COUNT))
TASKS_PER_BUDGET=$((${#STRATEGIES[@]} * EPISODE_COUNT))
PV_INDEX=$((SLURM_ARRAY_TASK_ID / TASKS_PER_PV))
PV_LOCAL=$((SLURM_ARRAY_TASK_ID % TASKS_PER_PV))
BUDGET_INDEX=$((PV_LOCAL / TASKS_PER_BUDGET))
BUDGET_LOCAL=$((PV_LOCAL % TASKS_PER_BUDGET))
STRATEGY_INDEX=$((BUDGET_LOCAL / EPISODE_COUNT))
EPISODE=$((BUDGET_LOCAL % EPISODE_COUNT))

PV=${PVS[$PV_INDEX]}
BUDGET_RATE=${BUDGET_RATIOS[$BUDGET_INDEX]}
STRATEGY=${STRATEGIES[$STRATEGY_INDEX]}
V_GOAL_MULTIPLIER=1.0

if [[ "${STRATEGY}" == "QGA" ]]; then
  MODEL_ROOT=saved_model/sparse_benchmark
  MODEL_DIR="${MODEL_ROOT}/QGA"
  test -f "${MODEL_DIR}/normalize_dict.pkl"
  test -f "${MODEL_DIR}/checkpoints/actor_step_11000.pt"
  test -f "${MODEL_DIR}/checkpoints/critic_step_11000.pt"
else
  MODEL_ROOT=saved_model/dgab_sparse_20260927172551
  MODEL_DIR="${MODEL_ROOT}/DGAB_Pretrain"
  test -f "${MODEL_DIR}/normalize_dict.pkl"
  test -f "${MODEL_DIR}/step_15000.pt"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-exp_data/qga_dgab_sparse_pv_budget_cross_ep0_9}"
OUTPUT_DIR="${OUTPUT_ROOT}/pv_${PV}/budget_${BUDGET_RATE}/${STRATEGY}"
OUTPUT="${STRATEGY}_pv_${PV}_budget_${BUDGET_RATE}_ep${EPISODE}"
mkdir -p "${OUTPUT_DIR}"

echo "start=$(date --iso-8601=seconds) strategy=${STRATEGY} pv=${PV} budget_rate=${BUDGET_RATE} episode=${EPISODE} xi=${V_GOAL_MULTIPLIER}"
srun python run_sparse_model_benchmark.py \
  --budget_rate "${BUDGET_RATE}" \
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
echo "finish=$(date --iso-8601=seconds) strategy=${STRATEGY} pv=${PV} budget_rate=${BUDGET_RATE} episode=${EPISODE}"
