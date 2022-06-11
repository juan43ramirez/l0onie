#!/bin/bash

# -----------------------------------------------------------------------------
#                               TO BE CUSTOMIZED
# -----------------------------------------------------------------------------
# Directory containing the source code
SRC_CODE_PATH="$HOME/github/coin"

# Bash script with call to Python
main_bash_script="${SRC_CODE_PATH}/bash_scripts/run_exp.sh"

# SLURM options
slurm_log_dir="$HOME/slurm_logs/coin"
notify_email="" # Leave empty ("") for no email
partition="long"

# WandB -- You will likely want to keep these untouched!
use_wandb=True
run_group=dual_lr_sweep

# -----------------------------------------------------------------------------

base_config="--config=${SRC_CODE_PATH}/configs/unstructured_sweep.py"

# Config lists
declare -a seeds=(1)
declare -a image_ids=(8) # "hardest" image for feasibility and psnr

# # 13x40 -> initial bpp = 0.81 -> target bpp = 0.6
# num_hidden=13
# declare -a hidden_widths=(40)
# declare -a target_bpps=(0.6)
# declare -a dual_lr_array=(8e-4)

# # 10x40 -> initial bpp = 0.6 -> target bpp = 0.3
# num_hidden=10
# declare -a hidden_widths=(40)
# declare -a target_bpps=(0.3)
# declare -a dual_lr_array=(1e-3)

# # 10x28 -> initial bpp = 0.3 -> target bpp = 0.15
# num_hidden=10
# declare -a hidden_widths=(28)
# declare -a target_bpps=(0.15)
# declare -a dual_lr_array=(3e-3)

# # 5x30 -> initial bpp = 0.15 -> target bpp = 0.07
# num_hidden=5
# declare -a hidden_widths=(30)
# declare -a target_bpps=(0.07)
# declare -a dual_lr_array=(7e-3)


# Hyper-parameters for tuning
declare -a weights_lr_array=(1e-3)
declare -a gates_lr_array=(7e-4)

# -----------------------------------------------------------------------------

# The parameter of this function is the python arguments
submit_sbatch () {
    sbatch --job-name=coin-slurm-%j.out \
        --time=01:00:00 \
        --cpus-per-task 4 \
        --mem=16G \
        --gres=gpu:rtx8000:1 \
        --partition=$partition \
        --output=$slurm_log_dir/unstruc-sweep-slurm-%j.out \
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
    _hidden_dims="$_hidden_dims$hidden_width,"
done
model_arg="--config.model.hidden_dims=(${_hidden_dims})"

# TODO: SGDM for dual variable

for seed in ${seeds[@]}; do
    seed_arg="--config.train.seed=$seed"

    for image_id in ${image_ids[@]}; do
        img_arg="--config.train.image_id=${image_id}"

        for tbpp in ${target_bpps[@]}; do
            bpp_arg="--config.train.target_bpp=${tbpp}"

            for weights_lr in ${weights_lr_array[@]}; do
                wlr_arg="--config.optim.weights_lr=${weights_lr}"

                for gates_lr in ${gates_lr_array[@]}; do
                    glr_arg="--config.optim.gates_lr=${gates_lr}"

                    for dual_lr in ${dual_lr_array[@]}; do
                        dlr_arg="--config.optim.dual_lr=${dual_lr}"

                        for hidden_width in ${hidden_widths[@]}; do
                            _hidden_dims=""
                            for i in $(seq $num_hidden);
                            do
                                _hidden_dims="$_hidden_dims$hidden_width,"
                            done
                            model_arg="--config.model.hidden_dims=(${_hidden_dims})"

                            all_args="${config_arg} ${wandb_arg} ${model_arg} ${seed_arg}\
                            ${img_arg} ${bpp_arg} ${wlr_arg} ${glr_arg} ${dlr_arg}"

                            submit_sbatch "${all_args}"
                        done
                    done
                done
            done
        done
    done
done