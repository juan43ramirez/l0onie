import copy
import functools
import logging
import pdb
from typing import Optional, Tuple

import cooper
import imageio
import torch
from torchvision import transforms

import gated
import magnitude_pruning.frozen_gated_modules as frozen_gated
import magnitude_pruning.magnitude_pruning as mp
import utils
import utils.coin_utils as coin_utils

logger = logging.getLogger(__name__)

OPTIM_DICT = {
    "None": None,
    "SGD": torch.optim.SGD,
    "SGDM": functools.partial(torch.optim.SGD, momentum=0.9),
    "Adam": torch.optim.Adam,
    "Adagrad": torch.optim.Adagrad,
    "RMSprop": torch.optim.RMSprop,
    # Optimizers with extrapolation, from cooper.optim
    "ExtraSGD": cooper.optim.ExtraSGD,
    "ExtraSGDM": functools.partial(cooper.optim.ExtraSGD, momentum=0.9),
    "ExtraAdam": cooper.optim.ExtraAdam,
}


def prepare_keep_ratio(config, image):

    assert config.task_type == "magnitude_pruning"

    if hasattr(config.mp, "keep_ratio") and config.mp.keep_ratio is not None:
        return config.mp.keep_ratio
    else:

        assert (
            hasattr(config.train, "target_bpp") and config.train.target_bpp is not None
        )
        # Transform BPP into keep_ratio for MP

        model_kwargs = {
            "in_dim": 2,
            "out_dim": 3,
            "hidden_dims": config.model.hidden_dims,
            "use_bias": config.model.use_bias,
        }

        dummy_model = gated.GatedSirenModel(
            sparsity_type="unstructured", **model_kwargs
        )

        found_tdst, achieved_bpp, return_flag = coin_utils.tdst_from_bpp(
            config.train.target_bpp,
            dummy_model,
            config.compression.dtype,
            image.shape,
            is_magnitude_pruning=True,
        )
        assert return_flag == "converged"

        # Must keep found_tdst percentage of parameters during magnitude pruning
        return found_tdst


def prepare_model(
    task: str,
    sparsity_type: str,
    device: str,
    train_dtype: torch.dtype,
    model_config: Optional[dict],
    gates_config: Optional[dict],
    keep_ratio: Optional[float],
    image_id: Optional[int],
):

    if task == "magnitude_pruning":
        assert sparsity_type == "unstructured"

        filters = {
            "$and": [
                {"config.model.hidden_dims": list(model_config.hidden_dims)},
                {"config.train.image_id": image_id},
                {"config.wandb.run_group": "baseline"},
                {"state": "finished"},
            ]
        }
        # Download a baseline, fully dense model with WandB api
        foo = utils.wandb_utils.get_baseline_model(
            "l0-coin", "kodak", filters, image_id, device
        )
        baseline_model, baseline_config, baseline_wandb_summary = foo

        tr_psnr = baseline_wandb_summary["train/best_psnr"]
        comp_psnr = baseline_wandb_summary["compression/best_psnr"]
        logger.info(f"Imported model with PSNR (train/comp): {tr_psnr} / {comp_psnr}")

        # Add non-trainable gates to its parameters
        gated_model = frozen_gated.FrozenGatedModel(baseline_model)
        # move to train_dtype as gated_model's dtype may be problematic. But
        # first, store the original dtype of baseline_model
        gated_model = gated_model.to(device, train_dtype)

        # Magnitude prune the model layer by layer with unstructured sparsity
        mp.unstructured_layerwise_prune_model(
            gated_model, keep_ratio, dense_outer_layers=True
        )

        return gated_model

    else:
        # Input dimensions represent pixel locations and output
        # dimensions represent rgb pixel values
        model_kwargs = {
            "in_dim": 2,
            "out_dim": 3,
            "hidden_dims": model_config.hidden_dims,
            "use_bias": model_config.use_bias,
            "w0_initial": model_config.w0_initial,
            "w0": model_config.w0,
            "final_activation": torch.nn.Identity(),
        }

        if task == "gated":
            gated_siren = gated.GatedSirenModel(
                droprate_init=gates_config.droprate_init,
                sparsity_type=sparsity_type,
                **model_kwargs,
            )
            return gated_siren.to(device, train_dtype)

        elif task == "coin_baseline":
            # This Siren only has nn.Linear layers inside
            siren = gated.GatedSirenModel(sparsity_type=None, **model_kwargs)
            return siren.to(device, train_dtype)

        else:
            raise ValueError(
                f"Unknown task: {task}. Supported tasks: magnitude_pruning, gated, coin_baseline"
            )


def prepare_optimizer(
    task_type, optim_cfg, model, formulation
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.Optimizer]]:

    primal_optim_class = OPTIM_DICT[optim_cfg.primal_optim]
    if task_type == "gated":
        primal_optim = primal_optim_class(
            [
                {"params": model.params_dict["net"], "lr": optim_cfg.weights_lr},
                {"params": model.params_dict["gates"], "lr": optim_cfg.gates_lr},
            ]
        )
        dual_optim_class = OPTIM_DICT[optim_cfg.dual_optim]
        dual_optim = cooper.optim.partial_optimizer(
            dual_optim_class, lr=optim_cfg.dual_lr
        )
        dual_restarts = optim_cfg.dual_restarts
    else:
        primal_optim = primal_optim_class(model.parameters(), lr=optim_cfg.weights_lr)
        dual_optim = None
        dual_restarts = None

    constrained_optimizer = cooper.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optim,
        dual_optimizer=dual_optim,
        dual_restarts=dual_restarts,
    )
    return constrained_optimizer


def prepare_data(image_id, device, dtype):
    # Load image
    file_name = f"kodak_dataset/kodim{str(image_id).zfill(2)}.png"
    img_data = imageio.v3.imread(file_name)
    # img contains values in {0..255}/255. (uint8s casted as floats)
    img_data = transforms.ToTensor()(img_data).float().to(device, dtype)
    img = utils.basic_utils.Image(id=image_id, data=img_data)
    img.populate_coordinates_and_targets(device, dtype)
    return img


def train_step(model, cmp, formulation, constrained_optimizer, image, cast_dtype):
    """
    Executes a single training step.
    """
    model.train()

    constrained_optimizer.zero_grad()
    # This is just the loss for baseline tasks
    lagrangian = formulation.composite_objective(
        cmp.closure, image.coordinates, image.targets, model, cast_dtype
    )
    formulation.custom_backward(lagrangian)
    constrained_optimizer.step()

    if isinstance(model, gated.GatedSirenModel):
        # Clamp gate parameters for training stability
        [layer.clamp_parameters() for layer in model.layers_dict["gated"]]

    log_dict = {
        "loss": cmp.state.loss.item(),
        "psnr": coin_utils.get_clamped_psnr(image.targets, cmp.state.misc["outputs"]),
    }

    if cmp.is_constrained:
        log_dict["lagrangian"] = lagrangian.item()

    if "reg_stats" in cmp.state.misc:
        train_bpp = (
            cmp.state.misc["reg_stats"].model_expected_bits.item() / cmp.num_pixels
        )
        comp_bpp = cmp.state.misc["reg_stats"].model_eval_bits.item() / cmp.num_pixels
        log_dict["proxy_bpp"] = train_bpp
        log_dict["hard_bpp"] = comp_bpp

    return log_dict, cmp.state.misc["outputs"]


def compress_model(model, device, quantize_dtype):
    """
    Compress a model via quantization of its parameters. If the model is gated,
    it is purged before quantization.
    """
    # We deepcopy the training model to avoid quantizing it unintentionally
    purged_model = copy.deepcopy(model)

    # Check sparsity_type is not None to ensure it is not a baseline task
    if isinstance(model, gated.GatedSirenModel) and model.sparsity_type is not None:
        purged_model = gated.purge_gated_model(purged_model)
    elif isinstance(model, frozen_gated.FrozenGatedModel):
        purged_model = frozen_gated.FrozenGatedModel(purged_model, do_absorb=True)
    else:
        # Make sure this is a baseline model
        assert isinstance(model, gated.GatedSirenModel) and model.sparsity_type is None

    # Quantize (floating point) parameters
    try:
        # The half precision version of torch.sin is only implemented in CUDA
        test_input = torch.tensor([[0.0, 0.0]], device=device, dtype=quantize_dtype)
        purged_model.to(device, quantize_dtype).forward(test_input)
    except:
        logger.warning(
            f"Function torch.sine is not supported for combination of {quantize_dtype}\
            and {device}. Defaulting to torch.float32."
        )
        # Overwrite dtype for downstream use
        quantize_dtype = torch.float32

    compressed_model = purged_model.to(device, quantize_dtype)

    return compressed_model, quantize_dtype


def purge_stats(loss_function, compressed_model, device, dtype, image):
    # At this stage compress is a bare nn.Module
    compressed_model.eval()

    # Forward
    _coordinates = image.coordinates.to(device, dtype)
    _targets = image.targets.to(device, dtype)

    with torch.no_grad():
        pred = compressed_model.forward(_coordinates)
        _loss = loss_function(pred, _targets, compressed_model)
        _psnr = coin_utils.get_clamped_psnr(pred, _targets)

    log_dict = {
        "loss": _loss.item(),
        "psnr": _psnr,
        "bpp": coin_utils.bpp(image.shape, compressed_model).item(),
        "kb_size": coin_utils.model_size_in_bits(compressed_model).item() / 8_000.0,
    }

    return log_dict, pred
