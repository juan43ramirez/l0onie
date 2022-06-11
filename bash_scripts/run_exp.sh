#!/bin/bash

module load python/3.8
source ~/venvs/coin/bin/activate

cd ${SRC_CODE_PATH} # this is an environment variable set by trigger_***.sh

python ./main.py "$@" 
