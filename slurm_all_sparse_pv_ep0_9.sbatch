#!/bin/bash
#SBATCH --job-name=all_sparse_pv10
#SBATCH --comment="All sparse policies, four PV levels, episodes 0-9"
#SBATCH --partition=amd
#SBATCH --output=log/%x_%A_%a.out
#SBATCH --error=log/%x_%A_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-439%32

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate autobidding

PVS=(0.0004 0.00045 0.00055 0.0006)
STRATEGIES=(BC BCQ DGABShare GUIDE IQL PID TD3_BC DT GAVE GAS QGA)
EPISODE_COUNT=10
TASKS_PER_PV=$((${#STRATEGIES[@]} * EPISODE_COUNT))

PV_INDEX=$((SLURM_ARRAY_TASK_ID / TASKS_PER_PV))
WITHIN_PV=$((SLURM_ARRAY_TASK_ID % TASKS_PER_PV))
STRATEGY_INDEX=$((WITHIN_PV / EPISODE_COUNT))
EPISODE=$((WITHIN_PV % EPISODE_COUNT))
PV=${PVS[$PV_INDEX]}
STRATEGY=${STRATEGIES[$STRATEGY_INDEX]}

if [[ "$PV" == "0.00055" || "$PV" == "0.0006" ]]; then
  V_GOAL_MULTIPLIER=1.2
else
  V_GOAL_MULTIPLIER=1.0
fi

OUTPUT_DIR="exp_data/all_sparse_pv_ep0_9/pv_${PV}/${STRATEGY}"
mkdir -p "$OUTPUT_DIR"

echo "Start: strategy=${STRATEGY}, pv=${PV}, episode=${EPISODE}, v_goal=${V_GOAL_MULTIPLIER}"

python run_sparse_model_benchmark.py \
  --budget_rate 1.0 \
  --episodes "$EPISODE" \
  --strategies "$STRATEGY" \
  --pv "$PV" \
  --pv_num 500000 \
  --seed 42 \
  --device cpu \
  --model_root saved_model/sparse_benchmark \
  --v_goal_multiplier "$V_GOAL_MULTIPLIER" \
  --output_dir "$OUTPUT_DIR" \
  --output "${STRATEGY}_pv_${PV}_ep${EPISODE}"

echo "Finish: strategy=${STRATEGY}, pv=${PV}, episode=${EPISODE}"
