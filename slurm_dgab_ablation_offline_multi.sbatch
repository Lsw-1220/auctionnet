#!/bin/bash
#SBATCH --job-name=dgab_abl_offm
#SBATCH --partition=amd
#SBATCH --output=log/%x_%A_%a.out
#SBATCH --error=log/%x_%A_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-71%32
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate autobidding
ENVS=(dense sparse); VARIANTS=(v0 v1 v2 v3 v4 v5); PERIODS=(8 9 10 11 12 13); SEED=42
PER_ENV=$((${#VARIANTS[@]}*${#PERIODS[@]})); E=$((SLURM_ARRAY_TASK_ID/PER_ENV)); R=$((SLURM_ARRAY_TASK_ID%PER_ENV)); V=$((R/${#PERIODS[@]})); P=$((R%${#PERIODS[@]}))
ENV=${ENVS[$E]}; VARIANT=${VARIANTS[$V]}; PERIOD=${PERIODS[$P]}
if [[ "$ENV" == dense ]]; then DATA=/share/home/u10092/project/lsw/dgab/data/off/period-${PERIOD}.csv; else DATA=/share/home/u10092/project/lsw/dgab/data/MDP/traffic/period-${PERIOD}.csv; fi
OUT=exp_data/dgab_ablation/offline_multi/${VARIANT}_${ENV}/period_${PERIOD}; mkdir -p "$OUT" log
python run_dgab_ablation_offline.py --environment "$ENV" --variant "$VARIANT" --seed "$SEED" --data "$DATA" --output "$OUT/result.csv"
