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
run_group=magnitude_pruning

# -----------------------------------------------------------------------------

base_config="--config=${SRC_CODE_PATH}/configs/magnitude_pruning.py"

# Config lists
declare -a seeds=(1)

# # 13x40 -> initial bpp = 0.81 -> target bpp = 0.6
# num_hidden=13
# declare -a hidden_widths=(40)
# declare -a target_bpps=(0.6)

# # 10x40 -> initial bpp = 0.6 -> target bpp = 0.3
# num_hidden=10
# declare -a hidden_widths=(40)
# declare -a target_bpps=(0.3)

# # 10x28 -> initial bpp = 0.3 -> target bpp = 0.15
# num_hidden=10
# declare -a hidden_widths=(28)
# declare -a target_bpps=(0.15)

# # 5x30 -> initial bpp = 0.15 -> target bpp = 0.07
# num_hidden=5
# declare -a hidden_widths=(30)
# declare -a target_bpps=(0.07)


# Images presented in the appendix of COIN paper
declare -a image_ids=$(seq 1 24)
# declare -a image_ids=(3 7 5) # For selecting individual images


# -----------------------------------------------------------------------------

# The parameter of this function is the python arguments
submit_sbatch () {
    sbatch --job-name=l0onie-%j.out \
        --time=01:00:00 \
        --cpus-per-task 4 \
        --mem=16G \
        --gres=gpu:rtx8000:1 \
        --partition=$partition \
        --output=$slurm_log_dir/magnitude_pruning-slurm-%j.out \
        --mail-type=ALL --mail-user=$notify_email \
        $main_bash_script $1
}

export SRC_CODE_PATH

# Indicate the considered config file
config_arg="${base_config}"

# Set up WandB flags
wandb_arg="--config.wandb.use_wandb=${use_wandb} --config.wandb.run_group=${run_group}"

for seed in ${seeds[@]}; do
    seed_arg="--config.train.seed=$seed"

    for image_id in ${image_ids[@]}; do
        img_arg="--config.train.image_id=${image_id}"

        for hidden_width in ${hidden_widths[@]}; do
            _hidden_dims=""
            for i in $(seq $num_hidden);
            do
                _hidden_dims="$_hidden_dims$hidden_width,"
            done
            model_arg="--config.model.hidden_dims=(${_hidden_dims})"

            for tbpp in ${target_bpps[@]}; do
                bpp_arg="--config.train.target_bpp=${tbpp}"

                all_args="${config_arg} ${wandb_arg} ${model_arg} ${seed_arg}\
                ${img_arg} ${bpp_arg}"

                submit_sbatch "${all_args}"
            done
        done
    done
done
