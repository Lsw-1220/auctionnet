#!/bin/bash
#SBATCH --job-name=dgab_dense_ctg
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

CTG_MULTIPLIERS=(0.6 0.8 1.2 1.4)
INDEX=$((SLURM_ARRAY_TASK_ID / 10))
EPISODE=$((SLURM_ARRAY_TASK_ID % 10))
CTG_MULTIPLIER=${CTG_MULTIPLIERS[$INDEX]}

MODEL_DIR="saved_model/dgab_shared_20260927172519"
CHECKPOINT="step_4000.pt"
test -f "${MODEL_DIR}/${CHECKPOINT}"
test -f "${MODEL_DIR}/normalize_dict.pkl"

OUTPUT_DIR="exp_data/dgab_dense_ctg_sweep/ctg_${CTG_MULTIPLIER}/episode_${EPISODE}"
OUTPUT_PREFIX="DGABShare_dense_ctg_${CTG_MULTIPLIER}_ep${EPISODE}"
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

srun python run_dense_opponent_benchmark.py \
  --strategies DGABShare \
  --pv 0.005 \
  --budget_rate 1.0 \
  --pv_num 500000 \
  --seed 42 \
  --device cpu \
  --episodes "${EPISODE}" \
  --dgabshare_dir "${MODEL_DIR}" \
  --dgabshare_ckpt "${CHECKPOINT}" \
  --v_goal_multiplier 25 \
  --action_multiplier 1.0 \
  --ctg_multiplier "${CTG_MULTIPLIER}" \
  --output_dir "${OUTPUT_DIR}" \
  --output "${OUTPUT_PREFIX}"
