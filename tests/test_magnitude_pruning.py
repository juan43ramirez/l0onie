import pytest
import torch

from gated import GatedSirenModel
from magnitude_pruning.frozen_gated_modules import FrozenGatedLinear, FrozenGatedModel
from magnitude_pruning.magnitude_pruning import unstructured_layerwise_prune_model


@pytest.fixture()
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture()
def baseline_model(device):
    kwargs = {
        "in_dim": 2,
        "out_dim": 3,
        "hidden_dims": (100, 100, 100),
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


class TestMagnitudePruning:
    def test_debug(self, gated_model, keep_ratio=0.7):
        unstructured_layerwise_prune_model(gated_model, keep_ratio, dense_outer_layers=False)
        for layer in gated_model.sequential:
            if isinstance(layer, FrozenGatedLinear):
                num_active = layer.weight_gates.count_nonzero()
                num_total = layer.weight_gates.numel()
                fraction = num_active / num_total

                tensor_tdst = torch.tensor(keep_ratio)
                assert torch.allclose(tensor_tdst, fraction)
