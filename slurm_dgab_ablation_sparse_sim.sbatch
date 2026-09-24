#!/bin/bash
#SBATCH --job-name=dgab_abl_spsim
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
STRATEGIES=(v1_sparse v2_sparse v3_sparse v4_sparse v5_sparse); EPISODE=$((SLURM_ARRAY_TASK_ID%10)); STRATEGY=${STRATEGIES[$((SLURM_ARRAY_TASK_ID/10))]}
OUT=exp_data/dgab_ablation/sparse_sim/${STRATEGY}/episode_${EPISODE}; mkdir -p "$OUT" log
python run_sparse_model_benchmark.py --budget_rate 1.0 --episodes "$EPISODE" --strategies "$STRATEGY" --pv 0.0005 --pv_num 500000 --seed 42 --device cpu --model_root saved_model/sparse_benchmark --ablation_model_root saved_model/dgab_ablation --v_goal_multiplier 1.0 --output_dir "$OUT" --output "${STRATEGY}_pv_0.0005_ep${EPISODE}"

