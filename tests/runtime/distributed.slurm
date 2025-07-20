#!/bin/bash
#SBATCH -A ehrtas
#SBATCH --nodes=1
#SBATCH --job-name=test_multi_node
#SBATCH --output=test_multi_node-%j.out
#SBATCH --error=test_multi_node-%j.err
#SBATCH --time=00:10:00
#SBATCH --gres gpu:4
#SBATCH --partition develbooster


srun -N 1 --gres gpu:4 mamba run --name PyExner python dambreak.py 2 2

