import torch

from .frozen_gated_modules import FrozenGatedLinear, FrozenGatedModel


def unstructured_layerwise_prune_model(
    model: FrozenGatedModel, keep_ratio: float, dense_outer_layers: bool
):
    """
    Magnitude prune each layer of a model given a layerwise target density.
    Returns a model with added dummy gates to freeze parameters that were
    removed in the Magnitude Pruning step.
    """
    assert isinstance(model, FrozenGatedModel)

    for layer in model.sequential:

        if isinstance(layer, FrozenGatedLinear):
            # Magnitude prune this layer's parameters by setting their gates to 0

            out_features, in_features = layer.weight.data.shape
            if dense_outer_layers and (in_features == 2 or out_features == 3):
                continue

            # We consider weight and bias separately, each is magnitude pruned
            params_list = [layer.weight.data]
            gates_list = [layer.weight_gates]
            if hasattr(layer, "bias"):
                params_list.append(layer.bias.data)
                gates_list.append(layer.bias_gates)

            for param_data, gates in zip(params_list, gates_list):
                # In the unstructured case, any L_p norm p>1 gives the same result
                norms = param_data.abs()

                # Get the norm for which target_sparsity of the values fall below
                norm_cutoff = torch.quantile(norms, q=(1.0 - keep_ratio))

                # We will turn off all gates associated to kernels with norms BELOW this number
                off_gates_mask = norms < norm_cutoff

                # Turn off gates whose with too-low norms
                gates[off_gates_mask] = 0.0
