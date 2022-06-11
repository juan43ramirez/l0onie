import ml_collections as mlc
import torch

from configs import default_config

MLC_PH = mlc.config_dict.config_dict.placeholder


def get_config():
    # ** This is set in bash_scripts/coin_baseline.sh

    config = default_config.get_config()

    config.task_type = "coin_baseline"

    config.train.seed = MLC_PH(int)  # **
    config.train.max_steps = 50_000
    config.train.image_id = MLC_PH(int)  # **
    config.train.dtype = torch.float32

    config.compression.freq = 10
    config.compression.dtype = torch.float16

    config.model.hidden_dims = MLC_PH(tuple)  # **
    config.model.use_bias = True
    config.model.w0 = 30.0
    config.model.w0_initial = 30.0

    config.optim.primal_optim = "Adam"
    config.optim.weights_lr = 2e-4

    config.verbose.verbose = True

    config.wandb.log_img_reconst = True
    config.wandb.log_img_freq = 5000

    return config
