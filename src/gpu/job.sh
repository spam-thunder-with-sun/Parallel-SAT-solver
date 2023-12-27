#!/bin/bash

#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --time=00:00:59
#SBATCH --job-name=GPU_SAT
#SBATCH --output=out/out_test.txt
#SBATCH --error=out/err_test.txt

make preclean
make build

srun sat_to_matrix_mult

make clean