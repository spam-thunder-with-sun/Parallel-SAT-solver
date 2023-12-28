#!/bin/bash

#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gpus-per-node=a30.24:0
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=GPU_SAT_jnh1
#SBATCH --output=out/out_jnh1.txt
#SBATCH --error=out/err_jnh1.txt

make preclean
make build

srun sat_to_matrix_mult ../input/Bejing/3blocks.cnf

make clean