#!/bin/bash
#SBATCH --job-name=onlinemlp
#SBATCH -n 10
#SBATCH -o py_out%j.out
#SBATCH -e py_err%j.err

mkdir -p out

for (( u=1; u<=10000; u*=10 ))
do
    srun -n 1 -c 1 --exclusive run.sh -u $u -t 0.5 &
done

