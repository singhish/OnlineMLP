#!/bin/bash
#SBATCH --job-name=offmlpf1
#SBATCH -n 100
#SBATCH -o py_out%j.out
#SBATCH -e py_err%j.err
#SBATCH --mail-user=ishrat@email.sc.edu
#SBATCH --mail-type=ALL

# input dimension
for i in {10..100..10}; do
  # units in hidden layer
  for u in {5..50..5}; do
    # epochs
    for e in {1..10}; do
      ((k = k % 100))
      ((k++ == 0)) && wait
      srun -n 1 -c 1 --exclusive offline-run.sh $@ -i $i -e $e -u $u >> offline-results.csv &
    done
  done
done
