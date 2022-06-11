import math
import pdb
from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPE_BIT_SIZE: Dict[torch.dtype, int] = {
    torch.float32: 32,
    torch.float: 32,
    torch.float64: 64,
    torch.double: 64,
    torch.float16: 16,
    torch.half: 16,
    torch.bfloat16: 16,
    torch.cdouble: 128,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.short: 16,
    torch.int32: 32,
    torch.int: 32,
    torch.int64: 64,
    torch.long: 64,
    torch.bool: 1,
}

# LIMIT_A = gamma; LIMIT_B = zeta -- 'stretching' parameters (Sect 4, p7)
LIMIT_A, LIMIT_B, EPS = -0.1, 1.1, 1e-6


class BaseGatedLayer(nn.Module):
    """Base class for gated layers. This class is not intended to be used directly."""

    def __init__(
        self,
        sparsity_type: str,
        use_bias: bool = False,
        droprate_init: float = 0.5,
    ):

        if droprate_init <= 0.0 or droprate_init >= 1.0:
            ValueError(f"Expected droprate_init in (0,1). Got {droprate_init}")

        super(BaseGatedLayer, self).__init__()

        self.sparsity_type = sparsity_type
        self.use_bias = use_bias
        self.droprate_init = droprate_init
        self.temperature = 2.0 / 3.0

    def init_parameters(self, wmode: str):
        """Initialize layer parameters, including the {weight, bias}_log_alpha parameters.
        Use wmode="fan_in" for fully-connected layers, and "fan_out" for convolutional
        layers."""

        # Initialize weight and bias parameters
        nn.init.kaiming_normal_(self.weight, mode=wmode)
        if self.use_bias:
            self.bias.data.normal_(0, 1e-2)

        # Initialize gate parameters
        gate_mean_init = math.log((1 - self.droprate_init) / self.droprate_init)
        self.weight_log_alpha.data.normal_(gate_mean_init, 1e-1)
        if hasattr(self, "bias_log_alpha"):
            self.bias_log_alpha.data.normal_(gate_mean_init, 1e-1)

    def construct_gates(self):

        if self.sparsity_type == "unstructured":
            self.group_size = 1
            self.dim_z = self.weight.shape
            if self.use_bias:
                self.bias_log_alpha = nn.Parameter(torch.Tensor(self.bias.shape))
        elif self.sparsity_type == "structured":
            if isinstance(self, GatedLinear):
                self.group_size = self.out_features
                self.dim_z = (self.in_features,)
        else:
            raise ValueError(f"Did not understant {self.sparsity_type} sparsity")

        self.weight_log_alpha = nn.Parameter(torch.Tensor(*self.dim_z))

    def clamp_parameters(self):
        """Clamp weight_log_alpha parameters for numerical stability."""
        self.weight_log_alpha.data.clamp_(min=math.log10(1e-4), max=math.log10(1e4))
        self.weight.data.clamp_(min=-3.0, max=3.0)

        if hasattr(self, "bias_log_alpha"):
            self.bias_log_alpha.data.clamp_(min=math.log10(1e-4), max=math.log10(1e4))
            self.bias.data.clamp_(min=-3.0, max=3.0)

    def get_gate_mean(self, log_alpha) -> torch.Tensor:
        """Returns the mean of the gate parameters."""
        return torch.sigmoid(
            log_alpha - self.temperature * math.log(-LIMIT_A / LIMIT_B)
        )

    def evaluate_gates(self):
        """
        Obtain medians for the stochastic gates, used for forwards. Active gates
        may have fractional values (not necessarily binary 0/1).
        """

        # TODO: only unstructured sparsity supported for now
        weight_z = torch.sigmoid(self.weight_log_alpha / self.temperature)
        weight_z = weight_z * (LIMIT_B - LIMIT_A) + LIMIT_A
        weight_z = torch.clamp(weight_z, min=0, max=1)

        bias_z = None
        if self.use_bias and self.sparsity_type == "unstructured":
            bias_z = torch.sigmoid(self.bias_log_alpha / self.temperature)
            bias_z = bias_z * (LIMIT_B - LIMIT_A) + LIMIT_A
            bias_z = torch.clamp(bias_z, min=0, max=1)

        return weight_z, bias_z

    def evaluate_params(self) -> Dict[str, Union[torch.Tensor, None]]:
        """
        Evaluate the parameters of the layer based on an inner evaluation of the
        gates. This function gets called when performing a forward pass.
        """
        weight_z, bias_z = self.evaluate_gates()

        if self.sparsity_type == "unstructured":
            bias = None if not self.use_bias else bias_z * self.bias
        else:
            if isinstance(self, GatedLinear):
                weight_z = weight_z.view(-1, 1)
                # No need to alter bias if using input sparsity
                bias = self.bias

        weight = weight_z * self.weight

        return {"weight": weight, "bias": bias}

    def expected_size_in_bits(self):
        """
        Compute the expected size of the layer in bits, given means of gates
        """

        assert self.sparsity_type == "unstructured"

        weight_q0 = self.get_gate_mean(self.weight_log_alpha)
        # weight_q0.clamp_(EPS, 1 - EPS)
        active_gates = torch.sum(weight_q0)

        if hasattr(self, "bias_log_alpha"):
            bias_q0 = self.get_gate_mean(self.bias_log_alpha)
            active_gates += torch.sum(bias_q0)

        return active_gates * DTYPE_BIT_SIZE[self.weight.dtype]

    def evaluation_size_in_bits(self, cast_dtype: torch.dtype):
        """
        Compute the size of the layer in bits at eval time, given a cast_dtype
        """

        params_dict = self.evaluate_params()

        _weight = params_dict["weight"].to(cast_dtype)
        active_gates = torch.count_nonzero(_weight)
        if params_dict["bias"] is not None:
            _bias = params_dict["bias"].to(cast_dtype)
            active_gates += torch.count_nonzero(_bias)

        return active_gates * DTYPE_BIT_SIZE[cast_dtype]


class GatedLinear(BaseGatedLayer):
    """Implementation of a fully connected layer with Sigmoid gates."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool,
        sparsity_type: str,
        droprate_init: float,
    ):

        # Call BaseGatedLayer constructor.
        super(GatedLinear, self).__init__(
            sparsity_type=sparsity_type, use_bias=use_bias, droprate_init=droprate_init
        )

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        # Counts total number of trainable parameters. This is not a count of active params.
        self.num_sparsifiable_params = self.weight.data.nelement()

        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            self.num_sparsifiable_params += self.bias.data.nelement()
        else:
            self.bias = None

        self.construct_gates()
        self.init_parameters(wmode="fan_out")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer according to a sampled weight matrix.
        Can be used for inference, but consider using a PurgedModel from
        sparse.purged_models to avoid the overhead of using gates.
        """
        params_dict = self.evaluate_params()
        return F.linear(input, params_dict["weight"], params_dict["bias"])

    def create_purged_copy(self) -> torch.nn.Linear:
        param_dict = self.evaluate_params()

        layer = nn.Linear(self.in_features, self.out_features, bias=self.use_bias)

        assert layer.weight.data.shape == param_dict["weight"].shape
        layer.weight.data = param_dict["weight"]

        if self.use_bias:
            assert layer.bias.data.shape == param_dict["bias"].shape
            layer.bias.data = param_dict["bias"]

        return layer

    def __repr__(self):
        s = "{name}({in_features} -> {out_features}, droprate_init={droprate_init}"
        if not self.use_bias:
            s += ", bias=False"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)
