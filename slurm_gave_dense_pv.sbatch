#!/bin/bash
#SBATCH --job-name=gave_dense_pv
#SBATCH --comment="GAVE dense PV generalization"
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
OUTPUT_DIR="exp_data/gave_dense_pv/pv_${PV}"

mkdir -p "$OUTPUT_DIR"

echo "Start: GAVE pv=${PV}, budget_ratio=1.0, episode=0"

python run_dense_opponent_benchmark.py \
  --budget_rate 1.0 \
  --episodes 0 \
  --strategies GAVE \
  --pv "$PV" \
  --pv_num 500000 \
  --seed 42 \
  --device cpu \
  --gave_dir saved_model/GAVE_dense/gave_400k_2729766 \
  --gave_ckpt step_5000.pt \
  --output_dir "$OUTPUT_DIR" \
  --output "GAVE_pv_${PV}_ep0"

echo "Finish: GAVE pv=${PV}, budget_ratio=1.0, episode=0"
