#!/bin/bash
#SBATCH --job-name=dense_all_grid
#SBATCH --comment="Dense models, PV x budget grid, episodes 0-9"
#SBATCH --partition=amd
#SBATCH --output=log/%x_%A_%a.out
#SBATCH --error=log/%x_%A_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-879%12

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate autobidding

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Submit twice: PV_GROUP=0 covers 0.004/0.0045; PV_GROUP=1 covers 0.0055/0.006.
: "${PV_GROUP:?Submit with --export=ALL,PV_GROUP=0 or PV_GROUP=1}"
if [[ "${PV_GROUP}" != "0" && "${PV_GROUP}" != "1" ]]; then
  echo "PV_GROUP must be 0 or 1, got ${PV_GROUP}" >&2
  exit 2
fi

PVS=(0.004 0.0045 0.0055 0.006)
BUDGET_RATIOS=(0.5 0.75 1.25 1.5)
STRATEGIES=(BC BCQ DGABShare GUIDE IQL PID QGA TD3_BC DT GAS GAVE)
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

DGAB_DIR=saved_model/dgab_shared_20260927172519
GAVE_DIR=saved_model/GAVE_dense/gave_400k_2729766
GAS_DIR=saved_model/GAS_dense/gas_paper_400k_2729782
QGA_DIR=saved_model/QGA_dense/dgab_mdp_bs48_steps16000_20260921

if [[ "${STRATEGY}" == "DGABShare" ]]; then
  test -f "${DGAB_DIR}/step_4000.pt"
  test -f "${DGAB_DIR}/normalize_dict.pkl"
elif [[ "${STRATEGY}" == "QGA" ]]; then
  test -f "${QGA_DIR}/checkpoints/actor_step_8000.pt"
  test -f "${QGA_DIR}/checkpoints/critic_step_8000.pt"
  test -f "${QGA_DIR}/normalize_dict.pkl"
elif [[ "${STRATEGY}" == "GAVE" ]]; then
  test -f "${GAVE_DIR}/step_5000.pt"
elif [[ "${STRATEGY}" == "GAS" ]]; then
  test -d "${GAS_DIR}"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-exp_data/dense_all_pv_budget_cross_ep0_9}"
OUTPUT_DIR="${OUTPUT_ROOT}/pv_${PV}/budget_${BUDGET_RATE}/${STRATEGY}"
OUTPUT="${STRATEGY}_pv_${PV}_budget_${BUDGET_RATE}_ep${EPISODE}"
mkdir -p "${OUTPUT_DIR}"

echo "start=$(date --iso-8601=seconds) pv_group=${PV_GROUP} strategy=${STRATEGY} pv=${PV} budget_rate=${BUDGET_RATE} episode=${EPISODE} xi=25"
srun python run_dense_opponent_benchmark.py \
  --budget_rate "${BUDGET_RATE}" \
  --episodes "${EPISODE}" \
  --strategies "${STRATEGY}" \
  --pv "${PV}" \
  --pv_num 500000 \
  --seed 42 \
  --device cpu \
  --dgabshare_dir "${DGAB_DIR}" \
  --dgabshare_ckpt step_4000.pt \
  --v_goal_multiplier 25 \
  --gave_dir "${GAVE_DIR}" \
  --gave_ckpt step_5000.pt \
  --gas_dir "${GAS_DIR}" \
  --qga_dir "${QGA_DIR}" \
  --qga_ckpt_step 8000 \
  --output_dir "${OUTPUT_DIR}" \
  --output "${OUTPUT}"
echo "finish=$(date --iso-8601=seconds) strategy=${STRATEGY} pv=${PV} budget_rate=${BUDGET_RATE} episode=${EPISODE}"
