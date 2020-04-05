#!/bin/bash
#SBATCH --job-name=onlinemlp
#SBATCH -n 28
#SBATCH -o py_out%j.out
#SBATCH -e py_err%j.err

mkdir -p out

# history length
for l in {10..100..10};
do
  # forecast length
  for f in {5..150..5};
  do
    # prediction period
    for p in {1..20};
    do
      # epochs
      for e in {25..500..25};
      do
        # units
        for u1 in 0 1 10 100 1000 10000;
        do
          for u2 in 0 1 10 100 1000 10000;
          do
            for u3 in 0 1 10 100 1000 10000;
            do
              # skip over invalid combinations
              if [ $p -gt $f ];
              then
                continue
              fi

              touch out/l${l}-f${f}-p${p}-u_${u1}_${u2}_${u3}-e${e}.csv
              srun -n 1 -c 1 --exclusive job.sh -l ${l} -f ${f} -p ${p} -u ${u1} ${u2} ${u3} -e ${e} >> out/l${l}-f${f}-p${p}-u_${u1}_${u2}_${u3}-e${e}.csv &
            done
          done
        done
      done
    done
  done
done
