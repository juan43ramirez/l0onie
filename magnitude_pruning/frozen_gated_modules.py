from copy import deepcopy

import torch
from torch import nn


class FrozenGatedModel(nn.Module):
    """Implements a model composed of FrozenGatedLinear layers."""

    def __init__(self, model, do_absorb=False):
        super().__init__()

        # Make sure we are not creating a FrozenGatedModel from a raw gated model
        # I.e. ensure that model was "purged" beforehand
        assert isinstance(model, FrozenGatedModel) or (model.sparsity_type is None)

        layers = []
        for layer in model.sequential:
            _layer = deepcopy(layer)
            if isinstance(_layer, nn.Linear):
                assert not (do_absorb)

                # Create gates for weights and biases, initialized as "all on"
                # TODO: So far only handles the unstructured case
                _layer = FrozenGatedLinear(_layer)
            elif isinstance(_layer, FrozenGatedLinear):
                # Ensure that we explicitly wanted to absorb gates in
                # FrozenGatedLinear layer
                assert do_absorb

                # Purge layer by absorbing gates into weights and biases
                _layer = _layer.absorb_gates_copy()
            layers.append(_layer)

        self.sequential = nn.Sequential(*layers)

    def forward(self, input):
        return self.sequential(input)


class FrozenGatedLinear(nn.Module):
    """Implements a linear layer with frozen gates on its parameters."""

    def __init__(self, layer, gate_type="unstructured"):
        super().__init__()

        assert isinstance(layer, nn.Linear)

        assert gate_type == "unstructured"

        device = layer.weight.device
        self.weight = deepcopy(layer.weight)
        self.weight_gates = torch.ones(
            self.weight.shape, device=device, requires_grad=False
        )

        if hasattr(layer, "bias"):
            self.bias = deepcopy(layer.bias)
            self.bias_gates = torch.ones(
                self.bias.shape, device=device, requires_grad=False
            )

    def forward(self, input):
        if hasattr(self, "weight_gates"):
            _weight = self.weight * self.weight_gates
        else:
            _weight = self.weight
        output = input.mm(_weight.T)

        if hasattr(self, "bias"):
            if hasattr(self, "bias_gates"):
                _bias = self.bias * self.bias_gates
            else:
                _bias = self.bias
            # Use .add_ for broadcasting
            output.add_(_bias)

        return output

    def absorb_gates_copy(self):

        layer = deepcopy(self)
        with torch.no_grad():
            layer.weight.data = layer.weight.data * layer.weight_gates
            del layer.weight_gates
            if hasattr(layer, "bias"):
                layer.bias.data = layer.bias.data * layer.bias_gates
                del layer.bias_gates

        return layer
