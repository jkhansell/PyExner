#!/bin/bash
#SBATCH -A ehrtas
#SBATCH --nodes=1
#SBATCH --job-name=test_multi_node
#SBATCH --output=test_multi_node-%j.out
#SBATCH --error=test_multi_node-%j.err
#SBATCH --time=00:10:00
#SBATCH --gres gpu:4
#SBATCH --partition booster


source /p/project/ehrtas/villalobos1/bedload/PyExner/machines/juwels.gpu

MPI4JAX_USE_CUDA_MPI=1 srun -N 1 -n 4 --gres gpu:4 mamba run --name PyExner python dambreak.py

