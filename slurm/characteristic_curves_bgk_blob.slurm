#!/bin/bash
#SBATCH --job-name=characteristic_curves_bgk_blob_drainage
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

PROJECT_ROOT=/home/cywang/sponge/JAX-LaB
GEOMETRY=/home/cywang/sponge/spongecore/data/structure/blob/blob_0000.npy
SIMULATION=drainage
OUTPUT_DIR="${PROJECT_ROOT}/results/characteristic_curves_bgk_blob_${SIMULATION}"
LOG="${OUTPUT_DIR}/run.log"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/images"
exec > >(tee -a "${LOG}") 2>&1

source "${PROJECT_ROOT}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}"
cd "${OUTPUT_DIR}"

echo "Job started at:  $(date)"
echo "Host:            $(hostname)"
echo "GPU(s):          ${CUDA_VISIBLE_DEVICES}"
echo "Python:          $(python3 --version)"
echo "Geometry:        ${GEOMETRY}"
echo "Simulation:      ${SIMULATION}"
echo "Working dir:     $(pwd)"
echo "================================================================================"

python3 -u "${PROJECT_ROOT}/examples/multiphase/BGK/characteristic_curves.py" \
    --geometry "${GEOMETRY}" \
    --simulation "${SIMULATION}"

echo "================================================================================"
echo "Job finished at: $(date)"
