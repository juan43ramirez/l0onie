import logging
import math
import os

import cooper
import torch
import wandb
from absl import app
from absl.flags import FLAGS
from ml_collections.config_flags import config_flags as MLC_FLAGS

import gated
from utils import basic_utils, cmp_utils, core_utils, wandb_utils

MLC_FLAGS.DEFINE_config_file("config", default="configs/debug_task.py")
logger = logging.getLogger(__name__)


def main(_):

    config = FLAGS.config
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if config.verbose.verbose:
        logging.basicConfig(level=logging.INFO)

    if config.train.seed is not None:
        basic_utils.change_random_seed(config.train.seed)

    wandb_utils.prepare_wandb(config=config)
    wandb_utils.create_wandb_subdir("models")
    model_save_dir = os.path.join(wandb.run.dir, "models")

    # Make sure that the frequency of image logging and of compression loops are
    # compatible. Otherwise, steps for which images could be logged will not
    # correspond to compression steps.
    if config.wandb.use_wandb and config.wandb.log_img_reconst:
        assert config.wandb.log_img_freq > 0 and config.compression.freq > 0
        assert config.wandb.log_img_freq % config.compression.freq == 0

    # Load image, move to device and cast as train dtype
    image = core_utils.prepare_data(config.train.image_id, DEVICE, config.train.dtype)

    if config.task_type == "magnitude_pruning":
        keep_ratio = core_utils.prepare_keep_ratio(config, image)
        wandb.config.update({"*keep_ratio": keep_ratio})

    model = core_utils.prepare_model(
        task=config.task_type,
        sparsity_type=config.sparsity_type,
        device=DEVICE,
        train_dtype=config.train.dtype,
        model_config=config.model,
        gates_config=config.gates if config.task_type == "gated" else None,
        keep_ratio=keep_ratio if config.task_type == "magnitude_pruning" else None,
        image_id=config.train.image_id,
    )

    if config.task_type == "gated":
        cmp = cmp_utils.CoinCMP(
            target_bpp=config.train.target_bpp,
            loss_module=torch.nn.MSELoss(),
            image_shape=image.shape,
        )
    elif config.task_type in ["magnitude_pruning", "coin_baseline"]:
        cmp = cmp_utils.CoinBaselineProblem(loss_module=torch.nn.MSELoss())

    formulation = cooper.LagrangianFormulation(cmp)

    # Primal and dual optimizers can be accessed as attributes of constrained_optimizer
    constrained_optimizer = core_utils.prepare_optimizer(
        task_type=config.task_type,
        optim_cfg=config.optim,
        model=model,
        formulation=formulation,
    )

    meters = {
        "train_psnr": basic_utils.BestMeter(direction="max"),
        "comp_psnr": basic_utils.BestMeter(direction="max"),
        "loss": basic_utils.BestMeter(direction="min"),
    }

    # --------------------------------------------------------------------------
    # Compression step before starting training
    # --------------------------------------------------------------------------
    validation_loop(-1, model, DEVICE, config, cmp, image, meters, None, False)

    for step_id in range(config.train.max_steps):

        # ----------------------------------------------------------------------
        # Train step
        # ----------------------------------------------------------------------
        train_log_dict, train_image_reconst = core_utils.train_step(
            model=model,
            cmp=cmp,
            formulation=formulation,
            constrained_optimizer=constrained_optimizer,
            image=image,
            cast_dtype=config.compression.dtype,
        )

        if (
            config.task_type == "gated"
            and config.optim.dual_restarts
            and formulation.state()[0].item() == 0.0
        ):
            # If Lagrange multipliers were restarted, restart the best psnr meters
            # Therefore, from now on we keep the best psnr among *feasible* models
            if "has_restarted" in locals() and has_restarted:
                pass
            else:
                meters["train_psnr"].reset()
                meters["comp_psnr"].reset()

            # Enforce that this restart only happens once per experiment.
            has_restarted = True

        _ = meters["train_psnr"].update(train_log_dict["psnr"].item())
        train_log_dict["best_psnr"] = meters["train_psnr"].val

        if config.verbose.verbose and step_id % config.verbose.log_freq == 0:
            logging.info(f"Step {step_id}/{config.train.max_steps}")
            logging.info(str(train_log_dict))

        if config.wandb.use_wandb:

            if config.task_type == "gated":
                do_log_gates = (
                    config.wandb.log_gates_hist
                    and (config.wandb.log_img_freq is not None)
                    and (step_id % config.wandb.log_img_freq == 0)
                )

                train_log_dict = wandb_utils.train_dict(
                    log_dict=train_log_dict,
                    log_gates_hist=do_log_gates,
                    formulation=formulation,
                    gated_layers=model.layers_dict["gated"],
                )
            else:
                # Re-name keys for wandb folder structure
                train_log_dict = basic_utils.prefix_keys(train_log_dict, "train/")

            wandb.log(train_log_dict, step=step_id)

        # ----------------------------------------------------------------------
        # Compression step
        # ----------------------------------------------------------------------
        do_purge = (config.compression.freq is not None) and (
            step_id % config.compression.freq == 0
        )
        is_last_epoch = step_id + 1 == config.train.max_steps
        if do_purge or is_last_epoch:

            compress_dict, compressed_model = validation_loop(
                step_id,
                model,
                DEVICE,
                config,
                cmp,
                image,
                meters,
                model_save_dir,
                is_last_epoch,
            )

    # Save last model and compressed model
    last_model_path = os.path.join(model_save_dir, "last_model.pt")
    torch.save(model, last_model_path)
    last_comp_model_path = os.path.join(model_save_dir, "last_compressed_model.pt")
    torch.save(compressed_model, last_comp_model_path)


def validation_loop(
    step_id, model, device, config, cmp, image, meters, model_save_dir, is_last_epoch
):
    """Compresses and quantizes model and evaluates performance."""

    compressed_model, quantize_dtype = core_utils.compress_model(
        model, device, config.compression.dtype
    )
    assert quantize_dtype == config.compression.dtype

    compress_dict, predicted_image = core_utils.purge_stats(
        loss_function=cmp.loss_func,
        compressed_model=compressed_model,
        device=device,
        dtype=quantize_dtype,
        image=image,
    )

    if config.verbose.verbose and (
        step_id % config.verbose.log_freq == 0 or step_id == -1
    ):
        logging.info(f"\tPurged performance - Step {step_id}/{config.train.max_steps}")
        logging.info("\t" + str(compress_dict))

    # Update meters
    psnr_improved = meters["comp_psnr"].update(compress_dict["psnr"].item())
    compress_dict["best_psnr"] = meters["comp_psnr"].val
    meters["loss"].update(compress_dict["loss"])

    if psnr_improved and step_id > 0:
        # Save the model with best psnr
        model_path = os.path.join(model_save_dir, "best_model.pt")
        torch.save(compressed_model, model_path)

    if config.wandb.use_wandb:
        # Re-name keys for wandb folder structure
        compress_dict = basic_utils.prefix_keys(compress_dict, "compression/")

        do_log_image = (config.wandb.log_img_freq is not None) and (
            step_id % config.wandb.log_img_freq == 0
        )
        if do_log_image or is_last_epoch:
            # Preprocess predicted image for wandb
            wandb_image, image_name = wandb_utils.prepare_image(
                reconst_image=predicted_image, image=image
            )
            compress_dict[image_name] = wandb_image

        wandb.log(compress_dict, step=step_id)

    return compress_dict, compressed_model


if __name__ == "__main__":
    app.run(main)
