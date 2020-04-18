#!/bin/bash
#SBATCH --job-name=onlinemlp
#SBATCH -n 16
#SBATCH -o py_out%j.out
#SBATCH -e py_err%j.err

echo 'History Length,Forecast Length,Units,Epochs,Loss'

# forecast length
for f in {10..100..10}; do
  # history length
  for l in {50..500..50}; do
    # units
    for u in {20..200..20}; do
      # epochs
      for e in {1..10}; do
        srun -n 1 -c 16 --exclusive run.sh -f $f -l $l -e $e -u $u >> results.csv &
      done
    done
  done
done
