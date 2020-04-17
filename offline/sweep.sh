#!/bin/bash

OUTFILE=offline-sweep-"$(date '+%Y%m%d%H%M%S')".csv

echo 'Sinusoids,Standard Deviation,History Length,Forecast Length,Units,Epochs,Loss' > "$OUTFILE"

# forecast length
for f in {10..100..10}; do
  # history length
  for l in {50..500..50}; do
    # units
    for u in {20..200..20}; do
      # epochs
      for e in {1..10}; do
        python offline-bench.py -l $l -f $f -e $e -u $u >> "$OUTFILE"
      done
    done
  done
done
