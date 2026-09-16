#!/bin/bash
#SBATCH --job-name=dense_budget
#SBATCH --comment="dense-best opponents, strategy-episode shards"
#SBATCH --partition=amd
#SBATCH --output=log/%x_%A_%a.out
#SBATCH --error=log/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --array=0-69%16

set -euo pipefail

: "${BUDGET_RATE:?submit with BUDGET_RATE}"
: "${BUDGET_TAG:?submit with BUDGET_TAG}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p log "exp_data/budget_sweep/${BUDGET_TAG}/shards"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate autobidding

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

STRATEGIES=(BC BCQ DT GUIDE TD3_BC IQL DGABShare_26000)
EPISODE=$((SLURM_ARRAY_TASK_ID / 7))
STRATEGY_INDEX=$((SLURM_ARRAY_TASK_ID % 7))
STRATEGY="${STRATEGIES[$STRATEGY_INDEX]}"
printf -v EP_TAG "%02d" "${EPISODE}"

echo "start=$(date --iso-8601=seconds)"
echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
echo "budget_rate=${BUDGET_RATE} episode=${EPISODE} strategy=${STRATEGY}"

srun python run_dense_opponent_benchmark.py \
  --budget_rate "${BUDGET_RATE}" \
  --episodes "${EPISODE}" \
  --strategies "${STRATEGY}" \
  --pv 0.005 \
  --pv_num 500000 \
  --seed 42 \
  --device cpu \
  --dgabshare_dir saved_model/dgab_shared_20260912135525 \
  --dgabshare_ckpt step_26000.pt \
  --v_goal_multiplier 15 \
  --output_dir "exp_data/budget_sweep/${BUDGET_TAG}/shards" \
  --output "budget_${BUDGET_TAG}_ep${EP_TAG}_${STRATEGY}"

echo "finish=$(date --iso-8601=seconds)"
