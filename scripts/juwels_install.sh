#!/bin/bash
set -e

echo "========================================"
echo " PyExner HPC Mamba Installation Script"
echo "========================================"

# --------------------------------------------------
# Modules
# --------------------------------------------------

module purge

module load Stages/2025
module load GCC
module load ParaStationMPI
module load CMake
module load PnetCDF
module load CUDA

# --------------------------------------------------
# Environment
# --------------------------------------------------

export CC=$(which mpicc)
export MPICC=$(which mpicc)

export PNETCDF_DIR=$EBROOTPNETCDF

# mpi4jax CUDA-aware MPI
export MPI4JAX_USE_CUDA_MPI=1

# Disable problematic GPU-aware MPI IO
export MPICH_GPU_SUPPORT_ENABLED=0

# Keep conda caches local to repo
export CONDA_PKGS_DIRS=$(pwd)/.conda_pkgs
export MAMBA_ROOT_PREFIX=$(pwd)/.mamba

mkdir -p $CONDA_PKGS_DIRS
mkdir -p $MAMBA_ROOT_PREFIX

# --------------------------------------------------
# Diagnostics
# --------------------------------------------------

echo ""
echo "========================================"
echo " Diagnostics"
echo "========================================"

module list

echo ""
echo "MPI:"
which mpicc
mpicc --version || true

echo ""
echo "CUDA:"
which nvcc || true
nvcc --version || true

echo ""
echo "PnetCDF:"
echo $PNETCDF_DIR

echo ""
echo "Python:"
which python3
python3 --version

# --------------------------------------------------
# Install mamba if missing
# --------------------------------------------------

if ! command -v mamba &> /dev/null
then
    echo ""
    echo "[INFO] Installing mamba..."

    mkdir -p $HOME/.local/bin

    curl -Ls https://micro.mamba.pm/api/mamba/linux-64/latest \
        | tar -xvj bin/mamba

    mv bin/mamba $HOME/.local/bin/

    export PATH=$HOME/.local/bin:$PATH

else
    echo ""
    echo "[INFO] mamba already available"
fi

echo ""
echo "mamba executable:"
which mamba

# --------------------------------------------------
# Initialize shell hook
# --------------------------------------------------

eval "$(mamba shell hook --shell bash)"

# --------------------------------------------------
# Create environment
# --------------------------------------------------

ENV_NAME=pyexner

echo ""
echo "[INFO] Creating environment..."

mamba create -y -n $ENV_NAME \
    python=3.12 \
    pip \
    numpy \
    matplotlib \
    xarray \
    pyyaml \
    cython \
    setuptools \
    wheel

# --------------------------------------------------
# Activate environment
# --------------------------------------------------

mamba activate $ENV_NAME

# --------------------------------------------------
# Install mpi4py against system MPI
# --------------------------------------------------

echo ""
echo "[INFO] Installing mpi4py..."

MPICC=$(which mpicc) \
pip install --no-binary=mpi4py mpi4py

# --------------------------------------------------
# Install PnetCDF Python wrapper
# --------------------------------------------------

echo ""
echo "[INFO] Installing pnetcdf..."

PNETCDF_DIR=$PNETCDF_DIR \
CC=$(which mpicc) \
MPICC=$(which mpicc) \
pip install pnetcdf

# --------------------------------------------------
# Install CUDA JAX
# --------------------------------------------------

echo ""
echo "[INFO] Installing CUDA-enabled JAX..."

pip install \
    --upgrade \
    "jax[cuda12]"

# --------------------------------------------------
# Install mpi4ja
# --------------------------------------------------

echo ""
echo "[INFO] Installing mpi4jax..."

CUDA_ROOT=$CUDA_HOME \
pip install --no-cache-dir --force-reinstall mpi4jax

# --------------------------------------------------
# Install PyExner
# --------------------------------------------------

echo ""
echo "[INFO] Installing PyExner..."

pip install -e .[cuda12]

# --------------------------------------------------
# Validation
# --------------------------------------------------

echo ""
echo "========================================"
echo " Validation"
echo "========================================"

python -c "
import jax
print('JAX:', jax.__version__)
print('Devices:', jax.devices())
"

python -c "
from mpi4py import MPI
print(MPI.Get_library_version())
"

python -c "
import pnetcdf
print('PnetCDF OK')
"

python -c "
import mpi4jax
print('mpi4jax OK')
"

python -c "
import PyExner
print('PyExner OK')
"

echo ""
echo "========================================"
echo " Installation Complete"
echo "========================================"

echo ""
echo "Activate environment with:"
echo "mamba activate $ENV_NAME"

echo ""
echo "Recommended runtime variables:"
echo "export MPI4JAX_USE_CUDA_MPI=1"
echo "export MPICH_GPU_SUPPORT_ENABLED=0"