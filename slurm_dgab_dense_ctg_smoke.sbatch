#!/bin/bash
#SBATCH --job-name=dgab_dense_ctg_smoke
#SBATCH --partition=amd
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate autobidding

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MODEL_DIR="saved_model/dgab_shared_20260927172519"
CHECKPOINT="step_4000.pt"
test -f "${MODEL_DIR}/${CHECKPOINT}"
test -f "${MODEL_DIR}/normalize_dict.pkl"

ROOT="exp_data/dgab_dense_ctg_smoke"
DEFAULT_DIR="${ROOT}/default_ep0"
EXPLICIT_DIR="${ROOT}/explicit_1.0_ep0"
for output_dir in "${DEFAULT_DIR}" "${EXPLICIT_DIR}"; do
  if [ -e "${output_dir}" ]; then
    echo "Output already exists: ${output_dir}" >&2
    exit 1
  fi
done
mkdir -p "${DEFAULT_DIR}" "${EXPLICIT_DIR}"

COMMON_ARGS=(
  --strategies DGABShare
  --pv 0.005
  --budget_rate 1.0
  --pv_num 500000
  --seed 42
  --device cpu
  --episodes 0
  --dgabshare_dir "${MODEL_DIR}"
  --dgabshare_ckpt "${CHECKPOINT}"
  --v_goal_multiplier 25
  --action_multiplier 1.0
)

srun python run_dense_opponent_benchmark.py \
  "${COMMON_ARGS[@]}" \
  --output_dir "${DEFAULT_DIR}" \
  --output "DGABShare_dense_ctg_default_ep0"

srun python run_dense_opponent_benchmark.py \
  "${COMMON_ARGS[@]}" \
  --ctg_multiplier 1.0 \
  --output_dir "${EXPLICIT_DIR}" \
  --output "DGABShare_dense_ctg_1.0_ep0"

python -c "import pandas as pd; a=pd.read_csv('${DEFAULT_DIR}/DGABShare_dense_ctg_default_ep0_results.csv'); b=pd.read_csv('${EXPLICIT_DIR}/DGABShare_dense_ctg_1.0_ep0_results.csv'); pd.testing.assert_frame_equal(a, b, check_exact=True); print('PASS: default CTG equals explicit ctg_multiplier=1.0')"
