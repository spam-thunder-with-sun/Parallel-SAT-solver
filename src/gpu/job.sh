#!/bin/bash

#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:00:05
#SBATCH --job-name=GPU_SAT
#SBATCH --output=out/out_sgemm.txt
#SBATCH --error=out/err_sgemm.txt

make preclean
make build

srun sat_to_matrix_mult

make clean