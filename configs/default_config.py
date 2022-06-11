import ml_collections as mlc
import torch

MLC_PH = mlc.config_dict.config_dict.placeholder


def get_config():
    config = mlc.ConfigDict()

    config.task_type = MLC_PH(str)
    config.sparsity_type = MLC_PH(str)

    config.train = train_config()
    config.compression = compression_config()

    config.model = model_config()
    config.optim = optimization_config()

    config.verbose = verbose_config()
    config.wandb = wandb_config()

    return config


def optimization_config():
    config = mlc.ConfigDict()
    config.primal_optim = "Adam"
    config.weights_lr = 2e-4
    return config


def train_config():
    config = mlc.ConfigDict()

    config.image_id = MLC_PH(int)

    # Seed is optional. If not specified, a random seed will be considered
    config.seed = MLC_PH(int)
    config.max_steps = 50_000

    config.dtype = torch.float32

    return config


def compression_config():
    config = mlc.ConfigDict()
    config.freq = MLC_PH(int)
    config.dtype = torch.float16
    return config


def model_config():
    config = mlc.ConfigDict()

    # Model architecture
    config.hidden_dims = MLC_PH(tuple)
    config.use_bias = True

    # SIREN parameters
    config.w0 = 30.0
    config.w0_initial = 30.0

    return config


def verbose_config():
    config = mlc.ConfigDict()
    config.verbose = False
    config.log_freq = 100
    return config


def wandb_config():
    config = mlc.ConfigDict()

    config.use_wandb = True
    config.use_wandb_offline = False
    config.run_group = MLC_PH(str)

    # Whether to log images reconstructed by the model
    config.log_img_reconst = True
    # Images are expensive to log, so we may log them sporadically
    config.log_img_freq = MLC_PH(int)
    # Whether to log histograms of the model's gates
    config.log_gates_hist = True

    return config
