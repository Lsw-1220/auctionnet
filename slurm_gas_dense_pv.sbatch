#!/bin/bash
#SBATCH --job-name=gas_dense_pv
#SBATCH --comment="GAS dense PV generalization"
#SBATCH --partition=amd
#SBATCH --output=log/%x_%A_%a.out
#SBATCH --error=log/%x_%A_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-4%5

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate autobidding

PVS=(0.004 0.0045 0.005 0.0055 0.006)
PV=${PVS[$SLURM_ARRAY_TASK_ID]}
OUTPUT_DIR="exp_data/gas_dense_pv/pv_${PV}"
mkdir -p "$OUTPUT_DIR"

python run_dense_opponent_benchmark.py \
  --budget_rate 1.0 \
  --episodes 0 \
  --strategies GAS \
  --pv "$PV" \
  --pv_num 500000 \
  --seed 42 \
  --device cpu \
  --gas_dir saved_model/GAS_dense/gas_paper_400k_2729782 \
  --output_dir "$OUTPUT_DIR" \
  --output "GAS_pv_${PV}_ep0"
