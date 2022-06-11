#!/bin/bash

# -----------------------------------------------------------------------------
#                               TO BE CUSTOMIZED
# -----------------------------------------------------------------------------
# Directory containing the source code
SRC_CODE_PATH="$HOME/github/l0onie"

# Bash script with call to Python
main_bash_script="${SRC_CODE_PATH}/bash_scripts/run_exp.sh"

# SLURM options
slurm_log_dir="$HOME/slurm_logs/l0onie"
notify_email="" # Leave empty ("") for no email
partition="long"

# WandB -- You will likely want to keep these untouched!
use_wandb=True
run_group=baseline

# -----------------------------------------------------------------------------

base_config="--config=${SRC_CODE_PATH}/configs/baseline.py"

# Config lists
declare -a seeds=(1)

# ----------------Base model architectures from COIN paper----------------------

# ---- 1.2 bpp --> We have not run this baseline yet
# num_hidden=13
# hidden_width=49

# ---- 0.81 bpp (Baseline for 0.6 MP + Loonie experiments)
# num_hidden=13
# hidden_width=40

# ---- 0.6 bpp
# num_hidden=10
# hidden_width=40

# ---- 0.3 bpp
# num_hidden=10
# hidden_width=28

# ---- 0.15 bpp
# num_hidden=5
# hidden_width=30

# ---- 0.07 bpp
# num_hidden=5
# hidden_width=20

# -----------------------------------------------------------------------------

# The parameter of this function is the python arguments
submit_sbatch () {
    sbatch --job-name=l0onie-%j.out \
        --time=01:00:00 \
        --cpus-per-task 4 \
        --mem=16G \
        --gres=gpu:rtx8000:1 \
        --partition=$partition \
        --output=$slurm_log_dir/baseline-slurm-%j.out \
        --mail-type=ALL --mail-user=$notify_email \
        $main_bash_script $1
}


export SRC_CODE_PATH

# Indicate the considered config file
config_arg="${base_config}"

# Set up WandB flags
wandb_arg="--config.wandb.use_wandb=${use_wandb} --config.wandb.run_group=${run_group}"

# Model flags
_hidden_dims=""
for i in $(seq $num_hidden);
do
    _hidden_dims="$_hidden_dims$hidden_width"
    if [ $i -ne $num_hidden ];
    then
        _hidden_dims="$_hidden_dims,"
    fi
done
model_arg="--config.model.hidden_dims=(${_hidden_dims})"


for seed in ${seeds[@]}; do
    seed_arg="--config.train.seed=$seed"

    for image_id in $(seq 1 24); do
        img_arg="--config.train.image_id=${image_id}"

        all_args="${config_arg} ${wandb_arg} ${model_arg} ${seed_arg} ${img_arg}"

        submit_sbatch "${all_args}"

    done
done