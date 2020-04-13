#!/bin/bash
#SBATCH --job-name=onlinemlp
#SBATCH -n 4
#SBATCH -o py_out%j.out
#SBATCH -e py_err%j.err

echo 'Sinusoids,Standard Deviation,History Length,Forecast Length,Prediction Period,Units,Epochs,Loss'
for (( u=1; u<=10000; u*=10 )); do
    srun -n 1 -c 1 --exclusive run.sh -t 0.1 -u $u
done
