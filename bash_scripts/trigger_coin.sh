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
run_group=workshop

# -----------------------------------------------------------------------------

base_config="--config=${SRC_CODE_PATH}/configs/debug_task.py" 

# Config lists
declare -a seeds=(1)
declare -a image_ids=(15)
declare -a target_bpps=(0.6)
declare -a task_types=("structured" "unstructured" "coin_baseline")

# -----------------------------------------------------------------------------

# The parameter of this function is the python arguments
submit_sbatch () {
    sbatch --job-name=l0onie-%j.out \
        --time=0:35:00 \
        --cpus-per-task 4 \
        --mem=16G \
        --gres=gpu:rtx8000:1 \
        --partition=$partition \
        --output=$slurm_log_dir/l0onie-%j.out \
        --mail-type=ALL --mail-user=$notify_email \
        $main_bash_script $1
}


export SRC_CODE_PATH

# Set up WandB flags
wandb_arg="--config.wandb.use_wandb=${use_wandb} --config.wandb.run_group=${run_group}"

for seed in ${seeds[@]}; do

    seed_arg="--config.train.seed=$seed"
    
    for image_id in ${image_ids[@]}; do
        img_arg="--config.train.image_id=${image_id}"
        
        for task_type in ${task_types[@]}; do
            config_arg="${base_config}:${task_type}"

            all_args="${config_arg} ${img_arg} ${seed_arg} ${wandb_arg}"
            if [ "${task_type}" = "coin_baseline" ];then
                submit_sbatch "${all_args}"
            else
                for target_bpp in ${target_bpps[@]}; do
                    # tdst tuple must be passed as string for ml_collections parser  
                    bpp_arg="--config.train.target_bpp=${target_bpp}"
                    submit_sbatch "${all_args} ${bpp_arg}"
                done
            fi
            
        done
    done
done
