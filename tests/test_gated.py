from copy import deepcopy

import pytest
import torch
from torch import nn

from gated.siren import GatedSirenModel
from magnitude_pruning.frozen_gated_modules import FrozenGatedLinear, FrozenGatedModel


@pytest.fixture()
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture()
def layer_in_features():
    return 10


@pytest.fixture()
def layer_out_features():
    return 15


@pytest.fixture()
def layer_input(layer_in_features, device):
    input = torch.randn(100, layer_in_features)
    return input.to(device)


@pytest.fixture()
def linear_layer(device, layer_in_features, layer_out_features):
    kwargs = {
        "in_features": layer_in_features,
        "out_features": layer_out_features,
        "bias": True,
    }

    layer = nn.Linear(**kwargs)
    layer.to(device)

    return layer


@pytest.fixture()
def gated_linear(linear_layer):
    layer = FrozenGatedLinear(linear_layer)
    return layer


@pytest.fixture()
def off_gated_linear(gated_linear):
    new_layer = deepcopy(gated_linear)
    new_layer.weight_gates[0, :] = 0.0
    if hasattr(new_layer, "bias_gates"):
        new_layer.bias_gates[-1] = 0.0
    return new_layer


@pytest.fixture()
def baseline_model(device):
    kwargs = {
        "in_dim": 2,
        "out_dim": 3,
        "hidden_dims": (7, 19, 23),
        "use_bias": True,
        "w0_initial": 30.0,
        "w0": 30.0,
        "sparsity_type": None,
    }

    model = GatedSirenModel(**kwargs)
    model.to(device)

    return model


@pytest.fixture()
def gated_model(baseline_model):
    model = FrozenGatedModel(baseline_model)
    return model


class TestFrozenGatedLayer:
    def test_forward_at_init(self, linear_layer, gated_linear, layer_input):
        # At initialization, all gates are 1 and outputs should be the same

        linear_out = linear_layer(layer_input)
        gated_out = gated_linear(layer_input)

        assert torch.allclose(linear_out, gated_out)

    def test_forward_after_off(self, gated_linear, off_gated_linear, layer_input):
        # Setting elements in gates to zero should result on a change in output

        on_output = gated_linear(layer_input)
        off_output = off_gated_linear(layer_input)

        assert not torch.allclose(on_output, off_output)

    def test_gradients(self, gated_linear, layer_input, device):
        # Make sure gradients for params are updated but not for gates
        out_features, in_features = gated_linear.weight.shape
        target = torch.randn(100, out_features).to(device)

        orig_weight = deepcopy(gated_linear.weight.data)
        orig_weight_gates = deepcopy(gated_linear.weight_gates.data)

        if hasattr(gated_linear, "bias"):
            orig_bias = deepcopy(gated_linear.bias.data)
            orig_bias_gates = deepcopy(gated_linear.bias_gates.data)

        optimizer = torch.optim.SGD(gated_linear.parameters(), lr=0.1)
        output = gated_linear(layer_input)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()

        # Weight should have changed
        assert not torch.allclose(orig_weight, gated_linear.weight.data)
        # Weight gates should not have changed
        assert torch.allclose(orig_weight_gates, gated_linear.weight_gates.data)

        if hasattr(gated_linear, "bias"):
            # Bias should have changed
            assert not torch.allclose(orig_bias, gated_linear.bias.data)
            # Bias gates should not have changed
            assert torch.allclose(orig_bias_gates, gated_linear.bias_gates.data)


class TestFrozenGatedModel:
    def test_dense_gated_model_init(self, baseline_model, gated_model, device):
        # At initialization, all gates are 1 and outputs should be the same

        input = torch.randn(100, 2)
        input = input.to(device)

        baseline_out = baseline_model(input)
        gated_out = gated_model(input)

        assert torch.allclose(baseline_out, gated_out)
