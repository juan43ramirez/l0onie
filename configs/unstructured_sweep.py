import ml_collections as mlc
import torch

from configs import default_config

MLC_PH = mlc.config_dict.config_dict.placeholder


def get_config():
    # ** This is set in bash_scripts/unstructured_sweep.sh

    config = default_config.get_config()

    config.task_type = "gated"
    config.sparsity_type = "unstructured"

    config.gates = mlc.ConfigDict()
    config.gates.droprate_init = 0.5

    config.train.seed = MLC_PH(int)  # **
    config.train.image_id = MLC_PH(int)  # **
    config.train.dtype = torch.float32
    config.train.target_bpp = MLC_PH(float)  # **

    config.compression.freq = 10
    config.compression.dtype = torch.float16

    config.model.hidden_dims = MLC_PH(tuple)  # **
    config.model.use_bias = True
    config.model.w0 = 30.0
    config.model.w0_initial = 30.0

    config.optim.primal_optim = "Adam"
    config.optim.weights_lr = MLC_PH(float)  # **
    config.optim.gates_lr = MLC_PH(float)  # **

    config.optim.dual_optim = "SGD"
    config.optim.dual_restarts = True
    config.optim.dual_lr = MLC_PH(float)  # **

    # Log, but avoid printing metrics which are going to WandB anyway.
    config.verbose.verbose = True
    config.verbose.log_freq = -1

    config.wandb.log_img_reconst = True
    config.wandb.log_img_freq = 5000

    return config
