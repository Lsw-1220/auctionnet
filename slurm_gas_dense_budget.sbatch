#!/bin/bash
#SBATCH --job-name=gas_dense_budget
#SBATCH --comment="GAS dense budget sweep"
#SBATCH --partition=amd
#SBATCH --output=log/%x_%A_%a.out
#SBATCH --error=log/%x_%A_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-49%32

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate autobidding

BUDGET_RATIOS=(0.5 0.75 1.0 1.25 1.5)
EPISODE_COUNT=10
BUDGET_INDEX=$((SLURM_ARRAY_TASK_ID / EPISODE_COUNT))
EPISODE=$((SLURM_ARRAY_TASK_ID % EPISODE_COUNT))
BUDGET_RATIO=${BUDGET_RATIOS[$BUDGET_INDEX]}
OUTPUT_DIR="exp_data/gas_dense_budget/budget_${BUDGET_RATIO}"
mkdir -p "$OUTPUT_DIR"

python run_dense_opponent_benchmark.py \
  --budget_rate "$BUDGET_RATIO" \
  --episodes "$EPISODE" \
  --strategies GAS \
  --pv 0.005 \
  --pv_num 500000 \
  --seed 42 \
  --device cpu \
  --gas_dir saved_model/GAS_dense/gas_paper_400k_2729782 \
  --output_dir "$OUTPUT_DIR" \
  --output "GAS_budget_${BUDGET_RATIO}_ep${EPISODE}"
