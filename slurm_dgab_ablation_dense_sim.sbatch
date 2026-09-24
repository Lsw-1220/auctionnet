#!/bin/bash
#SBATCH --job-name=dgab_abl_dnsim
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
STRATEGIES=(v1_dense v2_dense v3_dense v4_dense v5_dense); EPISODE=$((SLURM_ARRAY_TASK_ID%10)); STRATEGY=${STRATEGIES[$((SLURM_ARRAY_TASK_ID/10))]}
OUT=exp_data/dgab_ablation/dense_sim/${STRATEGY}/episode_${EPISODE}; mkdir -p "$OUT" log
python run_dense_opponent_benchmark.py --budget_rate 1.0 --episodes "$EPISODE" --strategies "$STRATEGY" --pv 0.005 --pv_num 500000 --seed 42 --device cpu --ablation_model_root saved_model/dgab_ablation --output_dir "$OUT" --output "${STRATEGY}_pv_0.005_ep${EPISODE}"

