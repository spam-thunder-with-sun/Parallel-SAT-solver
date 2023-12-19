#!/bin/bash

#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=GPU_SAT
#SBATCH --output=out/out.txt
#SBATCH --error=out/err.txt

make preclean
make build

srun sat_to_matrix_mult

make clean