import dataclasses
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

import utils.coin_utils as coin_utils

from .layers import GatedLinear


@dataclasses.dataclass
class ModelRegStats:
    """
    A class to store the regularization statistics for a given model.
    """

    layer_expected_bits: torch.Tensor
    layer_eval_bits: torch.Tensor
    model_expected_bits: torch.Tensor
    model_eval_bits: torch.Tensor


def layer_regularization(layer: nn.Module, cast_dtype: torch.dtype) -> Tuple[float]:
    """
    Compute training and compression bit size for a layer. This function could be
    extended to support vanilla Pytorch layers.
    """

    if hasattr(layer, "expected_size_in_bits"):
        return layer.expected_size_in_bits(), layer.evaluation_size_in_bits(cast_dtype)
    else:
        raise NotImplementedError


class BaseGatedModel(nn.Module):
    """Base class for sigmoid-gated models."""

    def __init__(self):
        super().__init__()

    def gather_layers_and_params(
        self,
    ) -> Tuple[Dict[str, nn.Module], Dict[str, nn.Parameter]]:
        """
        Gather all gated and dense layers and parameters of the model.

        Args:
            model: The model to gather layers and parameters from.

        Returns:
            layers_dict: All model layers grouped by "gated" and "dense".
            params_dict: All parameters of the model grouped by "net" (weights
                and biases), "gates" and "bn".
        """

        layers_dict: Dict[str, List] = {"gated": [], "dense": []}
        params_dict: Dict[str, List] = {"net": [], "gates": []}

        for m in self.modules():

            # Gather weights and biases for this module
            # Make sure this is not an activation/pooling layer
            if hasattr(m, "weight"):
                params_dict["net"].append(m.weight)
            if hasattr(m, "bias") and (m.bias is not None):
                params_dict["net"].append(m.bias)

            if isinstance(m, GatedLinear):
                # Gather sparsifiable module and parameters for the gates
                layers_dict["gated"].append(m)
                params_dict["gates"].append(m.weight_log_alpha)

                if hasattr(m, "bias_log_alpha") and (m.bias_log_alpha is not None):
                    params_dict["gates"].append(m.bias_log_alpha)

        return layers_dict, params_dict

    def regularization(self, cast_dtype: torch.dtype) -> ModelRegStats:
        """
        Compute expected bit size for the model.
        """

        # Extract all groups of layers for readability
        gated, dense = (self.layers_dict["gated"], self.layers_dict["dense"])

        # Compute bit sizes for all gated layers
        expected_bits_list = []
        evaluation_bits_list = []
        for layer in dense + gated:
            expected_bits, evaluation_bits = layer_regularization(layer, cast_dtype)

            expected_bits_list.append(expected_bits)
            evaluation_bits_list.append(evaluation_bits)

        # Using stack to preserve computational graph (and gradients)
        expected_bits_tensor = torch.stack(expected_bits_list)
        evaluation_bits_tensor = torch.stack(evaluation_bits_list)

        return ModelRegStats(
            layer_expected_bits=expected_bits_tensor,
            layer_eval_bits=evaluation_bits_tensor,
            model_expected_bits=torch.sum(expected_bits_tensor),
            model_eval_bits=torch.sum(evaluation_bits_tensor),
        )


def purge_gated_model(model: nn.Module) -> nn.Module:
    """Create a purged model from a model."""

    layers = []
    for layer in model.sequential:
        if isinstance(layer, GatedLinear):
            layers.append(layer.create_purged_copy())
        else:
            layers.append(layer)

    return nn.Sequential(*layers)
