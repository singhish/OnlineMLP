#!/bin/bash
#SBATCH --job-name=onlinemlp
#SBATCH -n 28
#SBATCH -o py_out%j.out
#SBATCH -e py_err%j.err

mkdir -p out

# forecast length
for f in {5..150..5};
do
    # prediction period
    for p in {1..20};
    do
        # skip invalid combinations
        if (( p >= f ));
        then
            continue
        fi

        # history length
        for l in {10..100..10};
        do
            # units
            for u in {10..1000..100};
            do
                # epochs
                for e in {25..400..25};
                do
                    # run python script, log output for each combination
                    touch out/loss-l${l}-f${f}-p${p}-e${e}.csv
                    srun -n 1 -c 1 --exclusive run.sh -l $l -f $f -p $p -u $u -e $e >> \
                            out/loss-l${l}-f${f}-p${p}-u${u}-e${e}.csv &
                done
            done
        done
    done
done
