#!/bin/bash
#SBATCH --job-name=dgab_abl_off1
#SBATCH --partition=amd
#SBATCH --output=log/%x_%A_%a.out
#SBATCH --error=log/%x_%A_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-119%32
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate autobidding
ENVS=(dense sparse); VARIANTS=(v0 v1 v2 v3 v4 v5); SEEDS=(42 43 44 45 46 47 48 49 50 51)
PER_ENV=$((${#VARIANTS[@]}*${#SEEDS[@]})); E=$((SLURM_ARRAY_TASK_ID/PER_ENV)); R=$((SLURM_ARRAY_TASK_ID%PER_ENV)); V=$((R/${#SEEDS[@]})); S=$((R%${#SEEDS[@]}))
ENV=${ENVS[$E]}; VARIANT=${VARIANTS[$V]}; SEED=${SEEDS[$S]}
if [[ "$ENV" == dense ]]; then DATA=/share/home/u10092/project/lsw/dgab/data/off/period-7.csv; else DATA=/share/home/u10092/project/lsw/dgab/data/MDP/traffic/period-7.csv; fi
OUT=exp_data/dgab_ablation/offline_single/${VARIANT}_${ENV}/seed_${SEED}; mkdir -p "$OUT" log
python run_dgab_ablation_offline.py --environment "$ENV" --variant "$VARIANT" --seed "$SEED" --data "$DATA" --output "$OUT/result.csv"
