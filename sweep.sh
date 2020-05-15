#!/bin/bash
#SBATCH --job-name=onlinemlp
#SBATCH -n 500
#SBATCH -o py_out%j.out
#SBATCH -e py_err%j.err

echo 'History Length,Forecast Length,Units,Epochs,Loss@1/5,Loss@2/5,Loss@3/5,Loss@4/5,Loss@End' >> results.csv

# forecast length
for f in {10..100..10}; do
  # history length
  for l in {50..500..50}; do
    # units
    for u in {20..200..20}; do
      # epochs
      for e in {1..10}; do
        (( i=i%500 )); (( i++==0 )) && wait
        srun -n 1 -c 1 --exclusive run.sh -f $f -l $l -e $e -u $u >> results.csv &
      done
    done
  done
done
