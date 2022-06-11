import pytest
import torch

import gated


@pytest.fixture(params=["unstructured"])
def siren_model(request):
    kwargs = {
        "in_dim": 2,
        "out_dim": 3,
        "hidden_dims": (7, 19, 23),
        "use_bias": True,
        "w0": 30.0,
        "sparsity_type": request.param,
        "w0_initial": 30.0,
        "droprate_init": 0.1,
        "final_activation": torch.nn.Identity(),
    }

    model = gated.GatedSirenModel(**kwargs)
    if torch.cuda.is_available():
        model = model.cuda()

    return model


class TestSirenModel:
    def test_siren_model_forward(self, siren_model):

        batch_size = 100
        input = torch.randn(batch_size, siren_model.in_dim)
        input = input.to(siren_model.sequential[0].weight.device)
        output = siren_model(input)
        assert output.shape == (batch_size, siren_model.out_dim)

    def test_siren_model_init(self, siren_model):
        # 4 blocks of layer plus activation. Last layer has an nn.Identity() as
        # activation
        expected_len = 2 + 2 + 2 + (1 + 1)
        assert len(siren_model.sequential) == expected_len
