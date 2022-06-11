import numpy as np
import pytest
import torch

import gated
from gated.siren import siren_layer_init


@pytest.fixture
def siren_kwargs():
    return {"c": 6.0, "w0": 30.0, "is_first": False}


@pytest.fixture
def layer_dim_kwargs():
    return {"in_features": 2, "out_features": 100, "use_bias": True}


@pytest.fixture(params=["unstructured"])
def siren_layer(request, siren_kwargs, layer_dim_kwargs):
    layer = gated.GatedLinear(
        **layer_dim_kwargs, sparsity_type=request.param, droprate_init=0.1
    )

    siren_layer_init(
        layer=layer, dim_in=layer_dim_kwargs["in_features"], **siren_kwargs
    )
    if torch.cuda.is_available():
        layer = layer.cuda()

    return layer


class TestSirenLayer:
    def test_siren_layer_init(self, siren_layer, siren_kwargs, layer_dim_kwargs):

        in_features = layer_dim_kwargs["in_features"]

        if siren_kwargs["is_first"]:
            w_std = 1 / in_features
        else:
            w_std = np.sqrt(siren_kwargs["c"] / in_features) / siren_kwargs["w0"]

        # Ensure that initialization is indeed uniform [-w_std, w_std]
        assert siren_layer.weight.abs().max() <= w_std

        if hasattr(siren_layer, "bias") and siren_layer.bias is not None:
            assert siren_layer.bias.abs().max() <= w_std

    def test_siren_layer_forward(self, siren_layer, layer_dim_kwargs):

        input = torch.randn(100, layer_dim_kwargs["in_features"])
        input = input.to(siren_layer.weight.device)

        output = siren_layer(input)

        # Output must be in [-1, 1] as a sine activation is used
        assert torch.all(output.abs().max() <= 1.0)
