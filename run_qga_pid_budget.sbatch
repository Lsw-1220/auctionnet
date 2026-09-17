#!/bin/bash
#SBATCH --job-name=qga_pid_budget
#SBATCH --comment="QGA step6000 and PID budget sweep"
#SBATCH --partition=amd
#SBATCH --output=log/%x_%A_%a.out
#SBATCH --error=log/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --array=0-99%16

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

BUDGET_RATES=(0.5 0.75 1.0 1.25 1.5)
BUDGET_TAGS=(050 075 100 125 150)
STRATEGIES=(QGA PID)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
BUDGET_INDEX=$((TASK_ID / 20))
WITHIN_BUDGET=$((TASK_ID % 20))
EPISODE=$((WITHIN_BUDGET / 2))
STRATEGY_INDEX=$((WITHIN_BUDGET % 2))

BUDGET_RATE="${BUDGET_RATES[$BUDGET_INDEX]}"
BUDGET_TAG="${BUDGET_TAGS[$BUDGET_INDEX]}"
STRATEGY="${STRATEGIES[$STRATEGY_INDEX]}"
printf -v EP_TAG "%02d" "${EPISODE}"

OUTPUT_DIR="exp_data/qga_pid_budget/budget_${BUDGET_TAG}/shards"
OUTPUT_NAME="budget_${BUDGET_TAG}_ep${EP_TAG}_${STRATEGY}"
mkdir -p "${OUTPUT_DIR}"

echo "start=$(date --iso-8601=seconds)"
echo "job=${SLURM_ARRAY_JOB_ID} task=${TASK_ID} host=$(hostname)"
echo "budget_rate=${BUDGET_RATE} episode=${EPISODE} strategy=${STRATEGY}"

srun python run_dense_opponent_benchmark.py \
  --budget_rate "${BUDGET_RATE}" \
  --episodes "${EPISODE}" \
  --strategies "${STRATEGY}" \
  --pv 0.005 \
  --pv_num 500000 \
  --seed 42 \
  --device cpu \
  --qga_dir saved_model/QGA_dense/QGA \
  --qga_ckpt_step 6000 \
  --output_dir "${OUTPUT_DIR}" \
  --output "${OUTPUT_NAME}"

echo "finish=$(date --iso-8601=seconds)"
