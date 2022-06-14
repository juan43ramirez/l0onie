import os
from typing import List, Optional, Tuple
import random
import string
import pdb

import numpy as np
import torch
from cooper import LagrangianFormulation

import gated
import utils
import wandb


def create_wandb_subdir(subdir_name: str):
    """
    Create a subdirectory in wandb.
    """
    try:
        os.mkdir(os.path.join(wandb.run.dir, subdir_name))
    except:
        pass


def prepare_wandb(config):
    """
    Disable WandB logging or make logging offline.
    """

    if not config.wandb.use_wandb:
        wandb.setup(
            wandb.Settings(
                mode="disabled",
                program=__name__,
                program_relpath=__name__,
                disable_code=True,
            )
        )
    else:
        if config.wandb.use_wandb_offline:
            os.environ["WANDB_MODE"] = "offline"

    wandb.init(project="kodak", entity="l0-coin")
    wandb.config.update(config.to_dict())


def train_dict(
    log_dict: dict,
    log_gates_hist: Optional[bool] = False,
    formulation: Optional[LagrangianFormulation] = None,
    gated_layers: Optional[List[gated.BaseGatedLayer]] = None,
):
    """
    Prepare a dictionary for logging to WandB. Format metrics in log_dict,
    append multiplier metrics and add gate histograms.

    Args:

        log_dict: Dictionary of existing metrics to log.
        log_gates_hist: Whether to log the gate histograms.
        formulation: Formulation for (Constrained) Minimization Problem at hand.
        gated_layers: Sparsifiable layers to log gate histograms for.
    """

    # log_dict = format_gated_metrics(log_dict)

    # Get multipliers
    multiplier_dict = collect_multipliers(formulation)
    log_dict.update(multiplier_dict)

    # Re-name keys for wandb folder structure
    log_dict = {"train/" + key: val for key, val in log_dict.items()}

    if log_gates_hist:
        hist_dict = collect_gates_hist(gated_layers)
        log_dict.update(hist_dict)

    return log_dict


def prepare_image(reconst_image: torch.Tensor, image: utils.basic_utils.Image):
    """
    Prepare an image to be logged to WandB.

    Args:
        reconst_image: Reconstructed image.
        img_shape: Shape of reconstructed image for re-shaping.
        image_id: ID of image to log.
    """

    wandb_image = log_image(reconst_image, image.shape)

    img_shape = (image.shape[1], image.shape[2], 3)
    reconst_image = reconst_image.reshape(*img_shape).permute(2, 0, 1)

    # Image must be clamped and sent to cpu and float for wandb logging
    reconst_image = torch.clamp(reconst_image, 0, 1).to("cpu", torch.float)

    image_name = "img_" + str(image.id).zfill(2)
    wandb_image = wandb.Image(reconst_image, caption="")

    return wandb_image, image_name


def collect_multipliers(formulation):
    # Check if formulation has inequality constraints (idx 0 of state)
    if formulation.state()[0] is not None:
        mult_vals = formulation.state()[0]
        if len(mult_vals.shape) == 0:
            # if scalar, convert to iterable for comprehension
            mult_vals = mult_vals.unsqueeze(0)

        lmbda_dict = {}
        for mult_ix, lmbda in enumerate(mult_vals):
            lmbda_dict["lambda_" + str(mult_ix + 1).zfill(2)] = lmbda

    return lmbda_dict


def collect_gates_hist(gated_layers):
    wandb_dict = {}
    for layer_ix, layer in enumerate(gated_layers):
        # This logs "soft" or "hard thresholded" gates depending on is_training
        wandb_dict.update(log_gates_histogram(layer, layer_ix))

    return wandb_dict


def log_image(reconst_image, target_shape):
    """Save image reconstruction"""
    img_shape = (target_shape[1], target_shape[2], 3)
    reconst_image = reconst_image.reshape(*img_shape).permute(2, 0, 1)
    # Image must be clamped and sent to cpu and float for wandb logging
    reconst_image = torch.clamp(reconst_image, 0, 1).to("cpu", torch.float)

    return wandb.Image(reconst_image, caption="")


def log_gates_histogram(layer, idx):
    """
    Log a histogram of the gates of layer to wandb summary.
    is_training=True for training; sampling once for each of the gate's distributions.
    is_training=False for validation; select the median of each gate's distribution.
    """

    log_name = "gates_medians/layer_" + str(idx + 1).zfill(2)

    weight_gates, bias_gates = layer.evaluate_gates()
    weight_gates = weight_gates.detach().cpu().numpy()
    histogram = np.histogram(weight_gates, bins=50)

    return {log_name: wandb.Histogram(np_histogram=histogram)}


def get_baseline_model(entity, project, filters, image_id, device):
    """
    Get the best compressed models of runs according to filters.
    """

    api = wandb.Api(overrides={"entity": entity, "project": project})
    runs = api.runs(path=entity + "/" + project, filters=filters, order="-created_at")
    assert len(runs) == 1, "More than one 'baseline' run found, revise TAGS"

    for one_run in runs:

        rand_folder = ''.join(random.choice(string.ascii_lowercase) for i in range(5))

        # Get best compressed model in terms of psnr
        wandb_file_name = "models/best_model.pt"
        local_file_name= f"models/{rand_folder}/models/best_model.pt"
        one_run.file(wandb_file_name).download(root=f"./models/{rand_folder}", replace=True)

        # We rename in order to keep a baseline for each image in models/
        new_name = f"models/{rand_folder}/models/best_model_img" + str(image_id) + ".pt"
        os.rename(local_file_name, new_name)

        model = torch.load(new_name, map_location=device)

        return model, one_run.config, one_run.summary
