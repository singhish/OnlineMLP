#!/bin/bash

echo -e '\nSubmitted Python job $@'

module load python3/anaconda/2019.10

source activate /work/ishrat/conda_env

python3 ../benchmark.py $@

source deactivate