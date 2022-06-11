# The implementation of these functions is adapted from the official COIN repo:
# https://github.com/EmilienDupont/coin/blob/main/util.py

import logging
import pdb
from typing import Dict

import numpy as np
import torch
from torch._C import dtype

logger = logging.getLogger(__name__)

import gated
from utils import core_utils

DTYPE_BIT_SIZE: Dict[dtype, int] = {
    torch.float32: 32,
    torch.float: 32,
    torch.float64: 64,
    torch.double: 64,
    torch.float16: 16,
    torch.half: 16,
    torch.bfloat16: 16,
    torch.cdouble: 128,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.short: 16,
    torch.int32: 32,
    torch.int: 32,
    torch.int64: 64,
    torch.long: 64,
    torch.bool: 1,
}


def to_coordinates_and_targets(
    image_data: torch.Tensor, device: torch.device, dtype: torch.dtype
):
    """Converts an image to a set of coordinates and targets.
    Args:
        image_data: Shape (channels, height, width).
    """
    # Coordinates are indices of all non zero locations of a tensor of ones of
    # same shape as spatial dimensions of image

    image_shape = image_data.shape
    coordinates = torch.ones(image_shape[1:]).nonzero(as_tuple=False).float()
    # Normalize coordinates to lie in [-.5, .5]
    coordinates /= torch.tensor([[image_shape[1] - 1, image_shape[2] - 1]])

    # Convert to range [-1, 1]
    # Fixed "bug" (?) from COIN repo which gave coordinates in wrong range
    coordinates = 2.0 * (coordinates - 0.5)

    # Convert image to a tensor of targets of shape (num_points, channels)
    targets = image_data.reshape(image_shape[0], -1).T

    return coordinates.to(device, dtype), targets.to(device, dtype)


def model_size_in_bits(model: torch.nn.Module):
    """Calculate total number of bits to store `model` parameters and buffers."""
    # We count nonzero elements to account for unstructured sparsity case
    total_bits = 0
    for _tensors in (model.parameters(), model.buffers()):
        for t in _tensors:
            total_bits += torch.count_nonzero(t) * DTYPE_BIT_SIZE[t.dtype]

    return total_bits


def bpp(image_shape, model):
    """Computes size in bits per pixel of model."""
    num_pixels = np.prod(image_shape) / 3  # Dividing by 3 because of RGB channels
    return model_size_in_bits(model=model) / num_pixels


def binarize_unstructured_gates(model, target_density):

    assert model.sparsity_type == "unstructured"

    for module in model.layers_dict["gated"]:
        assert isinstance(module, gated.BaseGatedLayer)
        # Iterate over all gated modules and turn off (1-target_density) of parameters
        if isinstance(module, gated.GatedLinear):

            param_tensors = [module.weight_log_alpha]
            if hasattr(module, "bias_log_alpha"):
                param_tensors.append(module.bias_log_alpha)

            for param in param_tensors:
                t_shape = param.shape
                device, dtype = param.device, param.dtype
                num_entries = np.prod(t_shape)

                mask = -10.0 * np.ones(num_entries)
                mask[: int(target_density * num_entries)] = 10.0

                reshaped_mask = np.random.permutation(mask).reshape(t_shape)
                tensor_mask = torch.from_numpy(reshaped_mask).to(device, dtype)
                param.data = tensor_mask


def tdst_from_bpp(target_bpp, model, compress_dtype, image_shape):
    if model.sparsity_type is None:
        target_density = dense_tdst_from_bpp(
            target_bpp, model, compress_dtype, image_shape
        )
        return target_density, target_bpp, "converged"
    elif model.sparsity_type == "unstructured":
        return unstructured_tdst_from_bpp(
            target_bpp, model, compress_dtype, image_shape
        )
    elif model.sparsity_type == "structured":
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown sparsity type: {model.sparsity_type}")


def unstructured_tdst_from_bpp(
    target_bpp, model, compress_dtype, image_shape, atol=1e-8
):

    import scipy.optimize as spo

    assert model.sparsity_type == "unstructured"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def achieved_minus_target(target_density):
        binarize_unstructured_gates(model, target_density)
        compressed_model, return_dtype = core_utils.compress_model(
            model, device, compress_dtype
        )
        assert return_dtype == compress_dtype

        return bpp(image_shape, compressed_model) - target_bpp

    bpp_at_1 = achieved_minus_target(target_density=1.0) + target_bpp
    if target_bpp > bpp_at_1:
        logger.warning(
            f"Provided model is too small for requested BPP. At full density, can achieve {bpp_at_1:.2f} bits per pixel."
        )
        return 1.0, bpp_at_1.cpu().item(), "unreachable_bpp"

    # In unstructured case model is not actually "purged" (units are not removed)
    # Setting 0 as tdst will just turn off all sparsifiable parameters.
    # Commenting this line since it gets computed in bisection_search
    # bpp_at_0 = achieved_minus_target(target_density=0.0) + target_bpp

    root_tdst, root_results = spo.bisect(
        achieved_minus_target,
        1.0,
        0.0,
        xtol=atol,
        maxiter=100,
        full_output=True,
    )
    achieved_bpp = achieved_minus_target(root_tdst) + target_bpp

    return root_tdst, achieved_bpp.cpu().item(), "converged"


def dense_tdst_from_bpp(target_bpp, model, compress_dtype, image_shape):

    assert model.sparsity_type is None

    # No need to purge model for bpp calculation as it does not have gated modules
    model_bits = model_size_in_bits(model)
    target_bits = target_bpp * np.prod(image_shape) / 3

    model_dtype = model.sequential[0].weight.data.dtype
    model_num_params = int(model_bits / DTYPE_BIT_SIZE[model_dtype])
    target_num_params = int(target_bits / DTYPE_BIT_SIZE[compress_dtype])

    if target_num_params <= model_num_params:
        raise ValueError(
            "Target bpp is too high. Try a lower value, or try a different model."
        )
    target_density = target_num_params / model_num_params

    return target_density


@torch.no_grad()
def psnr(image1: torch.Tensor, image2: torch.Tensor):
    """Calculates PSNR between two images."""
    sq_diff = (image1 - image2).detach().pow(2)
    return 20.0 * np.log10(1.0) - 10.0 * sq_diff.mean().log10().to("cpu").item()


def clamp_image(image):
    """Clamp image values to be in [0, 1] and round to range of unsigned int."""
    # Values may lie outside [0, 1], so clamp input
    _img = torch.clamp(image, 0.0, 1.0)
    # Pixel values lie in {0, ..., 255}, so round float tensor
    return torch.round(_img * 255) / 255.0


def get_clamped_psnr(image: torch.Tensor, image_reconst: torch.Tensor):
    """Get PSNR between true image and reconstructed image. As reconstructed
    image comes from output of neural net, ensure that values are in [0, 1] and
    are unsigned ints.
    Args:
        image: Ground truth image.
        image_reconst: Image reconstructed by model.
    """
    return psnr(image, clamp_image(image_reconst))
