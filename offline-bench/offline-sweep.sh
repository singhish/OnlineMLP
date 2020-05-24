#!/bin/bash
#SBATCH --job-name=offlinemlp
#SBATCH -n 500
#SBATCH -o py_out%j.out
#SBATCH -e py_err%j.err
#SBATCH --mail-user=ishrat@email.sc.edu
#SBATCH --mail-type=ALL

echo 'History Length,Forecast Length,Units,Epochs,Train Length,Loss' >> offline-results.csv

# forecast length
for f in {10..100..10}; do
  # history length
  for l in {50..500..50}; do
    # units
    for u in {20..200..20}; do
      # epochs
      for e in {1..10}; do
        # train length
        for t in 0.2 0.4 0.6 0.8; do
          (( i=i%500 )); (( i++==0 )) && wait
          srun -n 1 -c 1 --exclusive offline-run.sh -f $f -l $l -e $e -u $u -t $t >> offline-results.csv &
        done
      done
    done
  done
done
