#!/bin/bash
#SBATCH --job-name=displacement_co2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

PROJECT_ROOT=/home/cywang/sponge/JAX-LaB
OUTPUT_DIR="${PROJECT_ROOT}/results/co2"
LOG="${OUTPUT_DIR}/run.log"

# ── Output directory ──────────────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"

# Redirect all stdout and stderr through tee so output is written to run.log
# in real time while still flowing to the SLURM stdout stream.
exec > >(tee -a "${LOG}") 2>&1

# Symlink the geometry assets folder into the output directory so that the
# relative path ./assets/374_09_03_256.mat used in displacement_studies.py
# resolves correctly when the working directory is results/co2/.
ln -sfn "${PROJECT_ROOT}/assets" "${OUTPUT_DIR}/assets"

# ── Environment ───────────────────────────────────────────────────────────────
# Activate the uv-managed virtual environment (jax-lab, Python 3.10)
source "${PROJECT_ROOT}/.venv/bin/activate"

# Expose the src/ package to the Python import system
export PYTHONPATH="${PROJECT_ROOT}"

# ── Run ───────────────────────────────────────────────────────────────────────
# Change into the output directory so that all relative paths produced by the
# script land here:
#   - VTK field files  → results/co2/output_<r>/  and  results/co2/output/
#   - checkpoint dirs  → results/co2/checkpoints_/
#   - input .mat file  → results/co2/assets/374_09_03_256.mat  (via symlink)
cd "${OUTPUT_DIR}"

echo "Job started at:  $(date)"
echo "Host:            $(hostname)"
echo "GPU(s):          ${CUDA_VISIBLE_DEVICES}"
echo "Python:          $(python3 --version)"
echo "Working dir:     $(pwd)"
echo "================================================================================"

python3 -u "${PROJECT_ROOT}/examples/multiphase/MRT/displacement_studies.py"

echo "================================================================================"
echo "Job finished at: $(date)"
