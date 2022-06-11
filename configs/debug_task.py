import ml_collections as mlc
import torch

from configs import default_config

MLC_PH = mlc.config_dict.config_dict.placeholder

# python main.py --config=configs/debug_task.py:magnitude_pruning --config.train.target_bpp=0.15 --config.wandb.use_wandb=False


def get_config(config_string):

    config = basic_config()

    if config_string == "gated":
        config = gated_config(config)
    elif config_string == "coin_baseline":
        config = coin_baseline_config(config)
    elif config_string == "magnitude_pruning":
        config = mp_config(config)
    else:
        raise ValueError("Unknown config string: f{config_string}")

    return config


def mp_config(config):
    config.task_type = "magnitude_pruning"
    config.sparsity_type = "unstructured"
    config.mp = mlc.ConfigDict()
    config.mp.keep_ratio = MLC_PH(float)
    return config


def coin_baseline_config(config):
    config.task_type = "coin_baseline"
    return config


def gated_config(config):

    config.task_type = "gated"
    config.sparsity_type = "unstructured"

    config.optim.gates_lr = 1e-3

    config.optim.dual_optim = "SGD"
    config.optim.dual_restarts = True
    config.optim.dual_lr = 5.0e-4

    config.gates = mlc.ConfigDict()
    config.gates.droprate_init = 0.5  # Initialize gates to be around 0.5

    return config


def basic_config():

    config = default_config.get_config()

    config.train.seed = 1
    config.train.image_id = 15
    config.train.dtype = torch.float32
    config.train.target_bpp = MLC_PH(float)

    config.compression.freq = 10
    config.compression.dtype = torch.float16

    config.model.hidden_dims = tuple(10 * [28])
    config.model.use_bias = True

    config.model.w0 = 30.0
    config.model.w0_initial = 30.0

    config.optim.primal_optim = "Adam"
    config.optim.weights_lr = 1.0e-3

    config.verbose.verbose = True

    config.wandb.use_wandb = True
    config.wandb.run_group = "workshop"
    config.wandb.log_img_reconst = True
    config.wandb.log_img_freq = 5000

    return config
