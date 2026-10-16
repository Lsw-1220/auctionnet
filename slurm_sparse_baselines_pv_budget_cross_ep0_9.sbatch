#!/bin/bash
#SBATCH --job-name=sparse_base_grid
#SBATCH --comment="Nine sparse baselines, PV x budget grid, episodes 0-9"
#SBATCH --partition=amd
#SBATCH --output=log/%x_%A_%a.out
#SBATCH --error=log/%x_%A_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-719%12

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate autobidding

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Submit twice, with PV_GROUP=0 and PV_GROUP=1.  Each submission contains
# two PV values and at most 12 concurrent one-core tasks (24 across both).
: "${PV_GROUP:?Submit with --export=ALL,PV_GROUP=0 or PV_GROUP=1}"
if [[ "${PV_GROUP}" != "0" && "${PV_GROUP}" != "1" ]]; then
  echo "PV_GROUP must be 0 or 1, got ${PV_GROUP}" >&2
  exit 2
fi

PVS=(0.0004 0.00045 0.00055 0.0006)
BUDGET_RATIOS=(0.5 0.75 1.25 1.5)
STRATEGIES=(BC BCQ GUIDE IQL PID TD3_BC DT GAVE GAS)
EPISODE_COUNT=10

TASKS_PER_PV=$((${#BUDGET_RATIOS[@]} * ${#STRATEGIES[@]} * EPISODE_COUNT))
TASKS_PER_BUDGET=$((${#STRATEGIES[@]} * EPISODE_COUNT))
PV_INDEX=$((PV_GROUP * 2 + SLURM_ARRAY_TASK_ID / TASKS_PER_PV))
PV_LOCAL=$((SLURM_ARRAY_TASK_ID % TASKS_PER_PV))
BUDGET_INDEX=$((PV_LOCAL / TASKS_PER_BUDGET))
BUDGET_LOCAL=$((PV_LOCAL % TASKS_PER_BUDGET))
STRATEGY_INDEX=$((BUDGET_LOCAL / EPISODE_COUNT))
EPISODE=$((BUDGET_LOCAL % EPISODE_COUNT))

PV=${PVS[$PV_INDEX]}
BUDGET_RATE=${BUDGET_RATIOS[$BUDGET_INDEX]}
STRATEGY=${STRATEGIES[$STRATEGY_INDEX]}
MODEL_ROOT=saved_model/sparse_benchmark

if [[ "${STRATEGY}" == "DT" ]]; then
  test -f "${MODEL_ROOT}/DT/checkpoint_00002000/dt.pt"
  test -f "${MODEL_ROOT}/DT/checkpoint_00002000/normalize_dict.pkl"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-exp_data/sparse_baselines_pv_budget_cross_ep0_9}"
OUTPUT_DIR="${OUTPUT_ROOT}/pv_${PV}/budget_${BUDGET_RATE}/${STRATEGY}"
OUTPUT="${STRATEGY}_pv_${PV}_budget_${BUDGET_RATE}_ep${EPISODE}"
mkdir -p "${OUTPUT_DIR}"

echo "start=$(date --iso-8601=seconds) pv_group=${PV_GROUP} strategy=${STRATEGY} pv=${PV} budget_rate=${BUDGET_RATE} episode=${EPISODE}"
srun python run_sparse_model_benchmark.py \
  --budget_rate "${BUDGET_RATE}" \
  --episodes "${EPISODE}" \
  --strategies "${STRATEGY}" \
  --pv "${PV}" \
  --pv_num 500000 \
  --seed 42 \
  --device cpu \
  --model_root "${MODEL_ROOT}" \
  --v_goal_multiplier 1.0 \
  --output_dir "${OUTPUT_DIR}" \
  --output "${OUTPUT}"
echo "finish=$(date --iso-8601=seconds) strategy=${STRATEGY} pv=${PV} budget_rate=${BUDGET_RATE} episode=${EPISODE}"
