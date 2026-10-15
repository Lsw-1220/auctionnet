#!/bin/bash
#SBATCH --job-name=dgabsearch_full
#SBATCH --comment="DGABSearch dense/sparse budget and traffic generalization, episodes 0-9"
#SBATCH --partition=amd
#SBATCH --output=log/%x_%A_%a.out
#SBATCH --error=log/%x_%A_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-179%24

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate autobidding

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

BUDGET_RATIOS=(0.5 0.75 1.0 1.25 1.5)
if (( SLURM_ARRAY_TASK_ID < 90 )); then
  ENVIRONMENT=dense
  LOCAL_ID=${SLURM_ARRAY_TASK_ID}
  REFERENCE_PV=0.005
  NON_REFERENCE_PVS=(0.004 0.0045 0.0055 0.006)
  V_GOAL_MULTIPLIER=25
  MODEL_DIR=saved_model/dgab_shared_20260927172519
  CKPT=step_4000.pt
else
  ENVIRONMENT=sparse
  LOCAL_ID=$((SLURM_ARRAY_TASK_ID - 90))
  REFERENCE_PV=0.0005
  NON_REFERENCE_PVS=(0.0004 0.00045 0.00055 0.0006)
  V_GOAL_MULTIPLIER=1.0
  MODEL_ROOT=saved_model/dgab_sparse_20260927172551
  MODEL_DIR="${MODEL_ROOT}/DGAB_Pretrain"
  CKPT=step_15000.pt
fi

if (( LOCAL_ID < 50 )); then
  BUDGET_INDEX=$((LOCAL_ID / 10))
  EPISODE=$((LOCAL_ID % 10))
  BUDGET_RATE=${BUDGET_RATIOS[$BUDGET_INDEX]}
  PV=${REFERENCE_PV}
  EXPERIMENT=budget
  VALUE=${BUDGET_RATE}
else
  PV_LOCAL_ID=$((LOCAL_ID - 50))
  PV_INDEX=$((PV_LOCAL_ID / 10))
  EPISODE=$((PV_LOCAL_ID % 10))
  BUDGET_RATE=1.0
  PV=${NON_REFERENCE_PVS[$PV_INDEX]}
  EXPERIMENT=pv
  VALUE=${PV}
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-exp_data/dgabsearch_full_scenarios}"
OUTPUT_DIR="${OUTPUT_ROOT}/${ENVIRONMENT}/${EXPERIMENT}_${VALUE}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT="dgabsearch_${ENVIRONMENT}_${EXPERIMENT}_${VALUE}_ep${EPISODE}"
ADVERTISER_ARGS=()
if [[ -n "${SMOKE_ADVERTISER:-}" ]]; then
  ADVERTISER_ARGS=(--advertisers "${SMOKE_ADVERTISER}")
fi
echo "start=$(date --iso-8601=seconds) env=${ENVIRONMENT} experiment=${EXPERIMENT} value=${VALUE} episode=${EPISODE} xi=${V_GOAL_MULTIPLIER}"

if [[ "${ENVIRONMENT}" == dense ]]; then
  srun python run_dense_opponent_benchmark.py \
    --budget_rate "${BUDGET_RATE}" --episodes "${EPISODE}" \
    --strategies dgabsearch --pv "${PV}" --pv_num 500000 \
    --seed 42 --device cpu --dgabshare_dir "${MODEL_DIR}" \
    --dgabshare_ckpt "${CKPT}" --dgabsearch_dir "${MODEL_DIR}" \
    --dgabsearch_ckpt "${CKPT}" --v_goal_multiplier "${V_GOAL_MULTIPLIER}" \
    "${ADVERTISER_ARGS[@]}" --output_dir "${OUTPUT_DIR}" --output "${OUTPUT}"
else
  srun python run_sparse_model_benchmark.py \
    --budget_rate "${BUDGET_RATE}" --episodes "${EPISODE}" \
    --strategies dgabsearch --pv "${PV}" --pv_num 500000 \
    --seed 42 --device cpu --model_root "${MODEL_ROOT}" \
    --dgabsearch_dir "${MODEL_DIR}" --dgabsearch_ckpt "${CKPT}" \
    --v_goal_multiplier "${V_GOAL_MULTIPLIER}" \
    "${ADVERTISER_ARGS[@]}" --output_dir "${OUTPUT_DIR}" --output "${OUTPUT}"
fi

echo "finish=$(date --iso-8601=seconds)"
