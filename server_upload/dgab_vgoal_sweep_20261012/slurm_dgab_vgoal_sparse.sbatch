#!/bin/bash
#SBATCH --job-name=dgab_vgoal_sparse
#SBATCH --comment="DGAB sparse v-goal multiplier budget/value sweep"
#SBATCH --partition=amd
#SBATCH --output=log/%x_%A_%a.out
#SBATCH --error=log/%x_%A_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-629%24

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

VGOALS=(0.8 1.0 1.2 1.4 1.6 1.8 2.0)
BUDGET_RATIOS=(0.5 0.75 1.0 1.25 1.5)
# Baseline PV=0.0005 is covered by the budget_rate=1.0 task.
NON_BASELINE_PVS=(0.0004 0.00045 0.00055 0.0006)
TASKS_PER_VGOAL=90

VGOAL_INDEX=$((SLURM_ARRAY_TASK_ID / TASKS_PER_VGOAL))
LOCAL_ID=$((SLURM_ARRAY_TASK_ID % TASKS_PER_VGOAL))
VGOAL=${VGOALS[$VGOAL_INDEX]}

if (( LOCAL_ID < 50 )); then
  BUDGET_INDEX=$((LOCAL_ID / 10))
  EPISODE=$((LOCAL_ID % 10))
  BUDGET_RATE=${BUDGET_RATIOS[$BUDGET_INDEX]}
  PV=0.0005
  EXPERIMENT=budget
  VALUE=${BUDGET_RATE}
  OUTPUT_DIR="exp_data/dgab_vgoal_sweep_20261012/sparse/vgoal_${VGOAL}/budget_${BUDGET_RATE}"
else
  PV_LOCAL_ID=$((LOCAL_ID - 50))
  PV_INDEX=$((PV_LOCAL_ID / 10))
  EPISODE=$((PV_LOCAL_ID % 10))
  BUDGET_RATE=1.0
  PV=${NON_BASELINE_PVS[$PV_INDEX]}
  EXPERIMENT=pv
  VALUE=${PV}
  OUTPUT_DIR="exp_data/dgab_vgoal_sweep_20261012/sparse/vgoal_${VGOAL}/pv_${PV}"
fi

mkdir -p "${OUTPUT_DIR}"
echo "start=$(date --iso-8601=seconds) env=sparse vgoal=${VGOAL} experiment=${EXPERIMENT} value=${VALUE} episode=${EPISODE}"

srun python run_sparse_model_benchmark.py \
  --budget_rate "${BUDGET_RATE}" \
  --episodes "${EPISODE}" \
  --strategies DGABShare \
  --pv "${PV}" \
  --pv_num 500000 \
  --seed 42 \
  --device cpu \
  --model_root saved_model/dgab_sparse_20260927172551 \
  --v_goal_multiplier "${VGOAL}" \
  --output_dir "${OUTPUT_DIR}" \
  --output "DGABShare_sparse_vgoal_${VGOAL}_${EXPERIMENT}_${VALUE}_ep${EPISODE}"

echo "finish=$(date --iso-8601=seconds)"
