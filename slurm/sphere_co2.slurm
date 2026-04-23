#!/bin/bash
#SBATCH --job-name=sphere_co2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

PROJECT_ROOT=/home/cywang/sponge/JAX-LaB
GEOMETRY="${PROJECT_ROOT}/assets/374_05_03_256.mat"
OUTPUT_DIR="${PROJECT_ROOT}/results/sphere"
LOG="${OUTPUT_DIR}/run.log"

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${LOG}") 2>&1

source "${PROJECT_ROOT}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}"
cd "${OUTPUT_DIR}"

echo "Job started at:  $(date)"
echo "Host:            $(hostname)"
echo "GPU(s):          ${CUDA_VISIBLE_DEVICES}"
echo "Python:          $(python3 --version)"
echo "Geometry:        ${GEOMETRY}"
echo "Working dir:     $(pwd)"
echo "================================================================================"

python3 -u "${PROJECT_ROOT}/examples/multiphase/MRT/displacement.py" \
    --geometry "${GEOMETRY}"

echo "================================================================================"
echo "Job finished at: $(date)"
