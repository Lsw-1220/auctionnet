#!/bin/bash
#SBATCH --job-name=dgabqsearch_sparse
#SBATCH --comment="DGABQSearch sparse budget and traffic generalization, episodes 0-9"
#SBATCH --partition=amd
#SBATCH --output=log/%x_%A_%a.out
#SBATCH --error=log/%x_%A_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-89%24

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
NON_REFERENCE_PVS=(0.0004 0.00045 0.00055 0.0006)
if (( SLURM_ARRAY_TASK_ID < 50 )); then
  BUDGET_INDEX=$((SLURM_ARRAY_TASK_ID / 10))
  EPISODE=$((SLURM_ARRAY_TASK_ID % 10))
  BUDGET_RATE=${BUDGET_RATIOS[$BUDGET_INDEX]}
  PV=0.0005
  EXPERIMENT=budget
  VALUE=${BUDGET_RATE}
else
  LOCAL_ID=$((SLURM_ARRAY_TASK_ID - 50))
  PV_INDEX=$((LOCAL_ID / 10))
  EPISODE=$((LOCAL_ID % 10))
  BUDGET_RATE=1.0
  PV=${NON_REFERENCE_PVS[$PV_INDEX]}
  EXPERIMENT=pv
  VALUE=${PV}
fi

MODEL_ROOT=saved_model/dgab_sparse_20260927172551
MODEL_DIR="${MODEL_ROOT}/DGAB_Pretrain"
CKPT=step_15000.pt
V_GOAL_MULTIPLIER=1.0
NOISE_STD=0.1
OUTPUT_ROOT="${OUTPUT_ROOT:-exp_data/dgabqsearch_sparse_full}"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT}_${VALUE}"
OUTPUT="dgabQsearch_sparse_${EXPERIMENT}_${VALUE}_ep${EPISODE}"
mkdir -p "${OUTPUT_DIR}"

ADVERTISER_ARGS=()
if [[ -n "${SMOKE_ADVERTISER:-}" ]]; then
  ADVERTISER_ARGS=(--advertisers "${SMOKE_ADVERTISER}")
fi

echo "start=$(date --iso-8601=seconds) experiment=${EXPERIMENT} value=${VALUE} episode=${EPISODE} xi=${V_GOAL_MULTIPLIER} noise=${NOISE_STD}"
srun python run_sparse_model_benchmark.py \
  --budget_rate "${BUDGET_RATE}" --episodes "${EPISODE}" \
  --strategies dgabQsearch --pv "${PV}" --pv_num 500000 \
  --seed 42 --device cpu --model_root "${MODEL_ROOT}" \
  --dgabqsearch_dir "${MODEL_DIR}" --dgabqsearch_ckpt "${CKPT}" \
  --dgabqsearch_noise_std "${NOISE_STD}" \
  --v_goal_multiplier "${V_GOAL_MULTIPLIER}" \
  "${ADVERTISER_ARGS[@]}" --output_dir "${OUTPUT_DIR}" --output "${OUTPUT}"
echo "finish=$(date --iso-8601=seconds)"
