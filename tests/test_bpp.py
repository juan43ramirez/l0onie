import pdb

import pytest
import torch

import gated
from utils.basic_utils import Image
from utils.coin_utils import bpp, model_size_in_bits


@pytest.fixture(params=[tuple(10 * [100]), (28, 28)])
def model_kwargs(request):
    return {
        "in_dim": 2,
        "out_dim": 3,
        "hidden_dims": request.param,
        "use_bias": True,
        "w0": 30.0,
        "w0_initial": 30.0,
        "droprate_init": 0.5,
        "final_activation": torch.nn.Identity(),
    }


@pytest.fixture(params=["unstructured"])
def model(request, model_kwargs):
    return gated.GatedSirenModel(sparsity_type=request.param, **model_kwargs)


@pytest.fixture()
def dummy_image():
    return Image(id=None, data=torch.randn(1, 3, 32, 32))


class TestBPP:
    def test_purge_quantize(self, model, dummy_image):
        """
        Assess that purging and/or quantizing a model reduces its associated
        bit-per-pixels given a dummy_image. Performed both for structured and
        unstructured GatedSirenModels.
        """

        image_shape = dummy_image.shape

        initial_bpp = bpp(image_shape, model)

        # Purge model
        purged_model = gated.purge_gated_model(model)
        purged_bpp = bpp(image_shape, purged_model)

        # Quantize model
        quantized_model = model.to(torch.half)
        quantized_bpp = bpp(image_shape, quantized_model)

        # Quantize purged model
        quantized_purged_model = purged_model.to(torch.half)
        quantized_purged_bpp = bpp(image_shape, quantized_purged_model)

        assert quantized_purged_bpp < purged_bpp < initial_bpp
        assert quantized_purged_bpp < quantized_bpp < initial_bpp

    def test_sparsity_effect(self, model, dummy_image):
        """
        Test that as sparsity of an GatedSirenModel increases, the bit-per-pixel of
        its resulting PurgedSirenModel decreases.
        """

        image_shape = dummy_image.shape

        # Purge model once
        purged_model = gated.purge_gated_model(model)
        purged_bpp = bpp(image_shape, purged_model)

        # ------------------------------------------- Make the model more sparse
        for module in model.layers_dict["gated"]:
            if isinstance(module, gated.GatedLinear):

                if model.sparsity_type == "structured":
                    n_gates = module.weight_log_alpha.shape[0]
                    # Set 3/4ths of log_alpha to -10, the remaining 1/4 to 10
                    module.weight_log_alpha.data[0 : n_gates * 3 // 4] = -10.0
                    module.weight_log_alpha.data[n_gates * 3 // 4 :] = 10.0

                elif model.sparsity_type == "unstructured":
                    # Here we make more than 50% of log_alphas negative, as opposed
                    # to unstructured_siren

                    nrow_gates, ncol_gates = module.weight_log_alpha.shape
                    for row in range(nrow_gates):
                        # Randomly set 70% of the log_alphas[row, :] to -10

                        # We substract 0.7 to make ~70% of log_alphas negative
                        new_vals = (torch.rand(ncol_gates) - 0.7).sign()

                        module.weight_log_alpha.data[row, :] = new_vals
                        module.weight_log_alpha.data *= 10.0

        # Purge model anew
        new_purged_model = gated.purge_gated_model(model)
        new_purged_bpp = bpp(image_shape, new_purged_model)

        assert new_purged_bpp < purged_bpp

    def test_model_size_in_bits(self, model):

        model_reg_stats = model.regularization(cast_dtype=torch.float16)

        purged_model = gated.purge_gated_model(model)
        casted_model = purged_model.to(torch.float16)

        assert model_size_in_bits(casted_model) == model_reg_stats.model_eval_bits
