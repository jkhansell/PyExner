#!/bin/bash

cd strong_scaling/gpus_1
sbatch distributed.slurm
cd - > /dev/null

cd strong_scaling/gpus_2
sbatch distributed.slurm
cd - > /dev/null

cd strong_scaling/gpus_4
sbatch distributed.slurm
cd - > /dev/null

cd strong_scaling/gpus_8
sbatch distributed.slurm
cd - > /dev/null

cd strong_scaling/gpus_16
sbatch distributed.slurm
cd - > /dev/null

cd strong_scaling/gpus_32
sbatch distributed.slurm
cd - > /dev/null

cd strong_scaling/gpus_64
sbatch distributed.slurm
cd - > /dev/null

cd strong_scaling/gpus_128
sbatch distributed.slurm
cd - > /dev/null

cd strong_scaling/gpus_256
sbatch distributed.slurm
cd - > /dev/null

cd weak_scaling/gpus_1
sbatch distributed.slurm
cd - > /dev/null

cd weak_scaling/gpus_2
sbatch distributed.slurm
cd - > /dev/null

cd weak_scaling/gpus_4
sbatch distributed.slurm
cd - > /dev/null

cd weak_scaling/gpus_8
sbatch distributed.slurm
cd - > /dev/null

cd weak_scaling/gpus_16
sbatch distributed.slurm
cd - > /dev/null

cd weak_scaling/gpus_32
sbatch distributed.slurm
cd - > /dev/null

cd weak_scaling/gpus_64
sbatch distributed.slurm
cd - > /dev/null

cd weak_scaling/gpus_128
sbatch distributed.slurm
cd - > /dev/null

cd weak_scaling/gpus_256
sbatch distributed.slurm
cd - > /dev/null

