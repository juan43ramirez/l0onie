import pytest
import torch

import gated


@pytest.fixture()
def model_kwargs():
    return {
        "in_dim": 2,
        "out_dim": 3,
        "hidden_dims": (7, 19, 23),
        "use_bias": True,
        "w0": 30.0,
        "w0_initial": 30.0,
        "droprate_init": 0.1,
        "final_activation": torch.nn.Identity(),
    }


@pytest.fixture()
def unstructured_siren(model_kwargs):
    siren_model = gated.GatedSirenModel(sparsity_type="unstructured", **model_kwargs)
    if torch.cuda.is_available():
        siren_model = siren_model.cuda()

    # Re initialize weight_log_alpha parameters to allow for sparsity without training
    # Medians are 0 if the weight_log_alpha of their distribution is below -2.4
    for module in siren_model.layers_dict["gated"]:
        if isinstance(module, gated.GatedLinear):
            # In unstructured sparsity, we have one log_alpha per parameter
            # Here make 50% of log_alphas negative and 50% positive (rather than groups)
            module.weight_log_alpha.data = torch.randn_like(
                module.weight_log_alpha
            ).sign()
            module.weight_log_alpha.data *= 10.0

            if hasattr(module, "bias_log_alpha"):
                module.bias_log_alpha.data = torch.randn_like(
                    module.bias_log_alpha
                ).sign()
                module.bias_log_alpha.data *= 10.0

    return siren_model


class TestPurging:
    @pytest.mark.parametrize("model", [pytest.lazy_fixture("unstructured_siren")])
    def test_purge(self, model):

        purged_model = gated.purge_gated_model(model)

        with torch.no_grad():

            model.eval()
            purged_model.eval()

            device = model.sequential[0].weight.device
            input = torch.randn(100, 2, device=device)

            model_output = model(input)
            purged_output = purged_model(input)

            # This test fails with a lower tolerance
            assert torch.allclose(model_output, purged_output, atol=1e-6)
