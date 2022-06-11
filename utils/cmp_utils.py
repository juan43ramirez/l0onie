import logging

import cooper
import numpy as np
import torch

import gated

logger = logging.getLogger(__name__)


def construct_loss(loss_module: torch.nn.Module):
    if torch.cuda.is_available():
        loss_module = loss_module.cuda()

    def loss_func(pred, target_var, model, cast_dtype=None):
        loss = loss_module(pred, target_var)

        # Add sparsity_type check for BaselineModels
        if isinstance(model, gated.BaseGatedModel) and model.sparsity_type is not None:
            reg = model.regularization(cast_dtype=cast_dtype)
            return loss, reg
        else:
            return loss

    return loss_func


class CoinBaselineProblem(cooper.ConstrainedMinimizationProblem):
    """Baseline minimization problem for training a COIN model"""

    def __init__(self, loss_module: torch.nn.Module):
        self.loss_func = construct_loss(loss_module)
        super().__init__(is_constrained=False)

    def closure(self, inputs, targets, model, cast_dtype=None):
        outputs = model.forward(inputs)
        loss = self.loss_func(outputs, targets, model, cast_dtype)
        return cooper.CMPState(loss=loss, misc={"outputs": outputs})


class CoinCMP(cooper.ConstrainedMinimizationProblem):
    """Constrained minimization problem for training a gated COIN model"""

    def __init__(self, target_bpp, loss_module, image_shape):

        assert isinstance(target_bpp, float), "Only modelwise target_bpp is supported"
        self.target_bpp = target_bpp

        self.loss_func = construct_loss(loss_module)
        self.num_pixels = np.prod(image_shape) / 3
        super().__init__(is_constrained=True)

    def closure(self, inputs, targets, model, cast_dtype=None):

        outputs = model.forward(inputs)
        loss, reg_stats = self.loss_func(outputs, targets, model, cast_dtype)

        # Differentiable surrogate
        # Subtraction of bpp is not required here since only the gradient matters
        proxy_ineq_defect = reg_stats.model_expected_bits / self.num_pixels
        # "Hard" inequality constraint -- accounts for gate-thresholding and comp dtype
        ineq_defect = (reg_stats.model_eval_bits / self.num_pixels) - self.target_bpp

        # Store model output and other 'miscellaneous' objects in misc dict
        misc = {"outputs": outputs, "reg_stats": reg_stats}
        closure_state = cooper.CMPState(
            loss=loss,
            ineq_defect=ineq_defect,
            proxy_ineq_defect=proxy_ineq_defect,
            misc=misc,
        )

        return closure_state
