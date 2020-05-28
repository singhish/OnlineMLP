echo 'History Length,Forecast Length,Units,Epochs,Interval,Loss' >> results.csv

for s in 1 2 3; do
  for std in 1 5 10; do
    sbatch sweep.sh --s $s --std ${std}
  done
done