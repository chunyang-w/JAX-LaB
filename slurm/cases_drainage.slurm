#!/bin/bash
# ==============================================================================
# SLURM job script for cases/drainage/characteristic_curves.py
#
# Runs drainage or imbibition Pc-S simulation on a cropped geometry produced by
# cases/drainage/crop_geometry.ipynb (default: 64³ sub-cube of the sphere pack).
#
# ------------------------------------------------------------------------------
# COMMAND-LINE EQUIVALENT (no SLURM — run directly in a terminal):
#
#   cd /home/cywang/sponge/JAX-LaB
#   source .venv/bin/activate
#   export PYTHONPATH="${PWD}"
#   mkdir -p results/cases_drainage_{drainage,imbibition}
#
#   # drainage
#   uv run python3 -u ../characteristic_curves.py \
#       --geometry ../374_05_03_64.mat \
#       --simulation drainage \
#       --buffer 8 \
#       --drho 0.0092 \
#       --timesteps 100000 \
#       --io-rate 1000

#   # imbibition
#   python3 -u cases/drainage/characteristic_curves.py \
#       --geometry cases/drainage/374_05_03_64.mat \
#       --simulation imbibition \
#       --buffer 8 \
#       --drho 0.0092 \
#       --timesteps 250000 \
#       --io-rate 1000
#
#   # with uv (no manual venv activation needed):
#   uv run python3 -u cases/drainage/characteristic_curves.py \
#       --geometry cases/drainage/374_05_03_64.mat \
#       --simulation drainage
# ------------------------------------------------------------------------------
#
#SBATCH --job-name=cases_drainage
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ==============================================================================
# User-configurable parameters — edit these before submitting
# ==============================================================================
CUBE_SIZE=64         # must match the .mat file produced by crop_geometry.ipynb
SIMULATION=drainage  # drainage | imbibition
DRHO=0.0092          # density perturbation at inlet/outlet (driving force)
BUFFER=8             # inlet/outlet buffer planes along x
TIMESTEPS=250000     # total simulation timesteps
IO_RATE=1000         # VTK output + Pc-S record every N steps
# ==============================================================================

PROJECT_ROOT=/home/cywang/sponge/JAX-LaB
GEOMETRY="${PROJECT_ROOT}/cases/drainage/374_05_03_${CUBE_SIZE}.mat"
OUTPUT_DIR="${PROJECT_ROOT}/results/cases_drainage_${SIMULATION}_${CUBE_SIZE}"
LOG="${OUTPUT_DIR}/run.log"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/images"
exec > >(tee -a "${LOG}") 2>&1

source "${PROJECT_ROOT}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}"
cd "${OUTPUT_DIR}"

echo "Job started at:  $(date)"
echo "Host:            $(hostname)"
echo "GPU(s):          ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "Python:          $(python3 --version)"
echo "Geometry:        ${GEOMETRY}"
echo "Simulation:      ${SIMULATION}  (drho=${DRHO}  buffer=${BUFFER})"
echo "Timesteps:       ${TIMESTEPS}   io_rate=${IO_RATE}"
echo "Working dir:     $(pwd)"
echo "================================================================================"

python3 -u "${PROJECT_ROOT}/cases/drainage/characteristic_curves.py" \
    --geometry   "${GEOMETRY}"   \
    --simulation "${SIMULATION}" \
    --drho       "${DRHO}"       \
    --buffer     "${BUFFER}"     \
    --timesteps  "${TIMESTEPS}"  \
    --io-rate    "${IO_RATE}"

echo "================================================================================"
echo "Job finished at: $(date)"
