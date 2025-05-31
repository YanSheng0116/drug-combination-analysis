#!/bin/bash
# update Conda environment and sync to pip
conda env update -f environment.yml
conda activate drug-comb-env
pip freeze | grep -vE "^(torch|cuda)" > requirements.txt