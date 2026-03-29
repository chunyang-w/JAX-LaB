#!/bin/bash
#SBATCH --job-name=cc_bgk_drho0.007_w4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

PROJECT_ROOT=/home/cywang/sponge/JAX-LaB
GEOMETRY="${PROJECT_ROOT}/examples/multiphase/BGK/assets/374_05_03_256.mat"
SIMULATION=drainage
DRHO=0.007
WIDTH=4
OUTPUT_DIR="${PROJECT_ROOT}/results/characteristic_curves_bgk_${SIMULATION}_drho${DRHO}_w${WIDTH}"
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
echo "drho:            ${DRHO}"
echo "width:           ${WIDTH}"
echo "Working dir:     $(pwd)"
echo "================================================================================"

python3 -u "${PROJECT_ROOT}/examples/multiphase/BGK/characteristic_curves.py" \
    --geometry "${GEOMETRY}" \
    --simulation "${SIMULATION}" \
    --drho "${DRHO}" \
    --width "${WIDTH}"

echo "================================================================================"
echo "Job finished at: $(date)"
