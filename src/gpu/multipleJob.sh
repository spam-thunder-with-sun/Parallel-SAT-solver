#!/bin/bash

#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gpus-per-node=a30.24:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=GPU_SAT_MULTIPLE
#SBATCH --output=out/out_multiple.txt
#SBATCH --error=out/err_multiple.txt

filenames1=("2bitadd_10.cnf" "4blocksb.cnf" "2bitadd_11.cnf" "4blocks.cnf" "2bitadd_12.cnf" "e0ddr2-10-by-5-1.cnf" "2bitcomp_5.cnf" "e0ddr2-10-by-5-4.cnf" "2bitmax_6.cnf" "enddr2-10-by-5-1.cnf" "3bitadd_31.cnf" "enddr2-10-by-5-8.cnf" "3bitadd_32.cnf" "ewddr2-10-by-5-1.cnf" "3blocks.cnf" "ewddr2-10-by-5-8.cnf")
filenames2=("hole6.cnf" "jnh17.cnf" "jnh205.cnf" "jnh212.cnf" "jnh220.cnf" "jnh307.cnf" "jnh7.cnf" "jnh10.cnf" "jnh18.cnf" "jnh206.cnf" "jnh213.cnf" "jnh2.cnf" "jnh308.cnf" "jnh8.cnf" "jnh11.cnf" "jnh19.cnf" "jnh207.cnf" "jnh214.cnf" "jnh301.cnf" "jnh309.cnf" "jnh9.cnf" "jnh12.cnf" "jnh1.cnf" "jnh208.cnf" "jnh215.cnf" "jnh302.cnf" "jnh310.cnf")
filenames3=("jnh13.cnf" "jnh201.cnf" "jnh209.cnf" "jnh216.cnf" "jnh303.cnf" "jnh3.cnf" "trivial2.cnf" "jnh14.cnf" "jnh202.cnf" "jnh20.cnf" "jnh217.cnf" "jnh304.cnf" "jnh4.cnf" "jnh15.cnf" "jnh203.cnf" "jnh210.cnf" "jnh218.cnf" "jnh305.cnf" "jnh5.cnf" "zebra.cnf" "jnh16.cnf" "jnh204.cnf" "jnh211.cnf" "jnh219.cnf" "jnh306.cnf" "jnh6.cnf" "small.cnf")
filenames4=("g125.17.cnf" "g250.15.cnf" "g125.18.cnf" "g250.29.cnf")
path1="../input/Bejing/"
path2="../input/dimacs/"
path4="../input/gcp-large/"

make preclean
make build

for filename in ${filenames2[@]};
do
    srun sat_to_matrix_mult $path2$filename
done

make clean

: '
 --gres=gpu:a100:1

for filename in ${filenames1[@]};
do
    srun customDPLL $path1$filename
done

for filename in ${filenames3[@]};
do
    srun customDPLL $path2$filename
done

for filename in ${filenames2[@]};
do
    srun customDPLL $path2$filename
done

for filename in ${filenames3[@]};
do
    srun customDPLL $path3$filename
done

'