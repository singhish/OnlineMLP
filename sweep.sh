#!/bin/bash
#SBATCH --job-name=onlinemlp
#SBATCH -n 28
#SBATCH -o py_out%j.out
#SBATCH -e py_err%j.err

echo 'Sinusoids,Standard Deviation,History Length,Forecast Length,Prediction Period,Units,Epochs,Loss'

# forecast length
for f in {5..150..5}; do
  # prediction period
  for p in {1..20}; do
    # skip invalid combinations
    if (( p >= f )); then
      continue
    fi

    # history length
    for l in {10..100..10}; do
      # units
      for u in {10..1000..10}; do
        # epochs
        for e in {25..400..25}; do
          srun -n 1 -c 1 --exclusive run.sh -f $f -p $p -l $l -u $u -e $e
        done
      done
    done
  done
done
