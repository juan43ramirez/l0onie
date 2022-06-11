import pdb
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from .layers import GatedLinear
from .models import BaseGatedModel


class Sine(nn.Module):
    """Sine activation with scaling.
    Args:
        w0 (float): Omega_0 parameter from SIREN paper.
    """

    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor):
        return torch.sin(self.w0 * x)

    def __repr__(self):
        s = "{name}(w0={w0})"
        return s.format(name=self.__class__.__name__, w0=self.w0)


def siren_layer_init(
    layer: Union[nn.Linear, GatedLinear],
    dim_in: int,
    is_first: bool = False,
    c: float = 6.0,
    w0: float = 30.0,
    droprate_init: float = None,
):
    """Initialize layers following SIREN paper
    Args:
        layer: Layer for parameter initialization.
        dim_in: input dimension.
        is_first: Whether this is first layer of model.
        c: c value from SIREN paper used for weight initialization.
        w0: omega_0 parameter from SIREN paper.
    """

    initial_sigmoid = 1.0 - droprate_init if droprate_init is not None else 1.0

    # First layer has a different initialization
    w_std = (1 / dim_in) if is_first else (np.sqrt(c / dim_in) / w0)
    w_std = w_std / initial_sigmoid

    nn.init.uniform_(layer.weight, -w_std, w_std)

    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.uniform_(layer.bias, -w_std, w_std)


class GatedSirenModel(BaseGatedModel):
    """MLP Siren model with gates."""

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (50, 50),
        use_bias: bool = True,
        sparsity_type: Union[None, str] = "unstructured",
        droprate_init: Union[None, float] = 0.1,
        w0_initial: float = 30.0,
        w0: float = 30.0,
        final_activation: Optional[nn.Module] = nn.Identity(),
    ):
        super().__init__()

        self.input_size = (1, in_dim)
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.sparsity_type = sparsity_type
        self.use_bias = use_bias
        self.w0_initial = w0_initial
        self.w0 = w0
        self.final_activation = final_activation

        assert len(self.hidden_dims) > 0

        # --------------------- Construct Linear Layers ---------------------

        layers = []  # all model layers, including activations

        for layer_ix, dimh in enumerate(self.hidden_dims):
            if layer_ix == 0:
                # First layer is only gated if unstructured sparsity
                is_gated = self.sparsity_type == "unstructured"
                layer, act_layer = self.create_layer(
                    is_gated=is_gated,
                    in_features=self.in_dim,
                    out_featues=dimh,
                    use_bias=self.use_bias,
                    is_first=layer_ix == 0,
                    droprate_init=droprate_init,
                )
            else:
                # Other layers are gated as long as sparsity_type is defined
                is_gated = self.sparsity_type is not None
                layer, act_layer = self.create_layer(
                    is_gated=is_gated,
                    in_features=self.hidden_dims[layer_ix - 1],
                    out_featues=dimh,
                    use_bias=self.use_bias,
                    is_first=layer_ix == 0,
                    droprate_init=droprate_init,
                )

            layers += [layer, act_layer]

        # Final layer
        last_layer, foo = self.create_layer(
            is_gated=self.sparsity_type is not None,
            in_features=hidden_dims[-1],
            out_featues=out_dim,
            use_bias=self.use_bias,
            is_first=False,
            droprate_init=droprate_init,
        )
        layers.append(last_layer)

        if final_activation is not None:
            layers.append(final_activation)

        self.sequential = nn.Sequential(*layers)  # sequential object for forward.

        self.layers_dict, self.params_dict = self.gather_layers_and_params()
        # We need to set all_linear for purging functions
        self.all_linear = self.layers_dict["gated"]

    def create_layer(
        self,
        is_gated,
        in_features,
        out_featues,
        use_bias,
        is_first,
        droprate_init: float = None,
    ):

        layer_size = {"in_features": in_features, "out_features": out_featues}
        if is_gated:
            layer = GatedLinear(
                **layer_size,
                use_bias=use_bias,
                sparsity_type=self.sparsity_type,
                droprate_init=droprate_init,
            )

        else:
            layer = nn.Linear(**layer_size, bias=use_bias)

        w0 = self.w0_initial if is_first else self.w0
        siren_layer_init(
            layer=layer,
            c=6.0,
            w0=w0,
            dim_in=in_features,
            is_first=is_first,
            droprate_init=droprate_init if is_gated else None,
        )

        # Initialize weights and biases as uniform
        act_layer = Sine(w0=w0)

        return layer, act_layer

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.sequential.forward(input)
