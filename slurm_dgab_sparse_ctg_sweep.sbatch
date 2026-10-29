#!/bin/bash
#SBATCH --job-name=dgab_sparse_ctg
#SBATCH --partition=amd
#SBATCH --output=log/%x_%A_%a.out
#SBATCH --error=log/%x_%A_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-39%24

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate autobidding

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Four multipliers x ten episodes. Existing completed pilot episodes are reused.
CTG_MULTIPLIERS=(0.6 0.8 1.2 1.4)
INDEX=$((SLURM_ARRAY_TASK_ID / 10))
EPISODE=$((SLURM_ARRAY_TASK_ID % 10))
CTG_MULTIPLIER=${CTG_MULTIPLIERS[$INDEX]}

MODEL_ROOT="saved_model/dgab_sparse_20260927172551"
test -f "${MODEL_ROOT}/DGAB_Pretrain/step_15000.pt"
test -f "${MODEL_ROOT}/DGAB_Pretrain/normalize_dict.pkl"

OUTPUT_DIR="exp_data/dgab_sparse_ctg_pilot/ctg_${CTG_MULTIPLIER}/episode_${EPISODE}"
OUTPUT_PREFIX="DGABShare_ctg_${CTG_MULTIPLIER}_ep${EPISODE}"
if [ -f "${OUTPUT_DIR}/${OUTPUT_PREFIX}_results.csv" ] && \
   [ -f "${OUTPUT_DIR}/${OUTPUT_PREFIX}_manifest.json" ]; then
  echo "Completed output already exists: ${OUTPUT_DIR}; skipping"
  exit 0
fi
if [ -e "${OUTPUT_DIR}" ]; then
  echo "Incomplete output directory already exists: ${OUTPUT_DIR}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}"

srun python run_sparse_model_benchmark.py \
  --budget_rate 1.0 \
  --episodes "${EPISODE}" \
  --strategies DGABShare \
  --pv 0.0005 \
  --pv_num 500000 \
  --seed 42 \
  --device cpu \
  --model_root "${MODEL_ROOT}" \
  --v_goal_multiplier 1.0 \
  --ctg_multiplier "${CTG_MULTIPLIER}" \
  --output_dir "${OUTPUT_DIR}" \
  --output "${OUTPUT_PREFIX}"
