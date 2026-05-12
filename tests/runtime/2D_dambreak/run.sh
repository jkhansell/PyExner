#!/bin/bash

source ~/.bashrc

module purge
module load PrgEnv-gnu
module load nvidia-mixed
module use /soft/modulefiles
module load cray-mpich
module load cudatoolkit-standalone/12.9.0
module load cuda/12.9
module load craype-accel-nvidia80
module load craype-x86-milan
module load cray-parallel-netcdf/1.12.3.9
module load spack-pe-base cmake

export MPICH_GPU_SUPPORT_ENABLED=1

MPI4JAX_USE_CUDA_MPI=1 mpiexec -n 4 ./set_affinity.sh mamba run -n pyexner python run2D_dambreak.py input.yaml