#!/bin/bash
#SBATCH --job-name=poly_sphere_co2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

PROJECT_ROOT=/home/cywang/sponge/JAX-LaB
GEOMETRY=/home/cywang/sponge/spongecore/data/structure/poly_sphere/poly_sphere_0000.npy
OUTPUT_DIR="${PROJECT_ROOT}/results/poly_sphere"
LOG="${OUTPUT_DIR}/run.log"

mkdir -p "${OUTPUT_DIR}"

# Redirect all stdout + stderr to run.log in real time
exec > >(tee -a "${LOG}") 2>&1

# Activate the uv-managed virtual environment (jax-lab, Python 3.10)
source "${PROJECT_ROOT}/.venv/bin/activate"

export PYTHONPATH="${PROJECT_ROOT}"

# Change into the output directory so all relative output paths
# (VTK files, checkpoints) land in results/poly_sphere/
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
