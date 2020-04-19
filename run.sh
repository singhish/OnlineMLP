#!/bin/bash

module load python3/anaconda/2019.10
conda activate /work/ishrat/conda_env
python3 -m bench $@
conda deactivate
