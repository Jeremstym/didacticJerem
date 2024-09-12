import math
from typing import Literal, Tuple, Union, Dict, Callable

import torch
from scipy.special import binom, factorial
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn import init

from didactic.models.layers import _QKVLinearProjection, _QKVMatrixMultiplication

ModuleType = Union[str, Callable[..., nn.Module]]

class BidirectionalMultimodalAttention(nn.Module):
    """Bidirectional Multimodal Attention."""

    def __init__(
        self,
        d_token: int,
        n_heads: int,
        dropout: float,
        bias: bool = True,
        initialization: Literal["kaiming", "xavier"] = "kaiming",
    ):
        """Initializes class instance.

        Args:
            d_token: Token size. Must be a multiple of `n_heads`.
            n_heads: Number of attention heads. If greater than 1, then the module will have an additional output layer
                (so called "mixing" layer).
            dropout: Dropout rate for the attention map. The dropout is applied to *probabilities* and does not affect
                logits.
            bias: If `True`, then input (and output, if presented) layers also have bias.
            initialization: Initialization for input projection layers. Must be one of ['kaiming', 'xavier'].
        """
        super().__init__()
        self.mod0_linear_proj = _QKVLinearProjection(d_token, n_heads, bias=bias, initialization=initialization)
        self.mod1_linear_proj = _QKVLinearProjection(d_token, n_heads, bias=bias, initialization=initialization)
        self.mod0_self_mat_mul = _QKVMatrixMultiplication(d_token, n_heads, dropout, bias=bias)
        self.mod1_self_mat_mul = _QKVMatrixMultiplication(d_token, n_heads, dropout, bias=bias)
        self.mod0_cross_mat_mul = _QKVMatrixMultiplication(d_token, n_heads, dropout, bias=bias)
        self.mod1_cross_mat_mul = _QKVMatrixMultiplication(d_token, n_heads, dropout, bias=bias)

    def forward(self, x0: Tensor, x1: Tensor) -> Tuple[Tensor, Tensor]:
        """Performs a forward pass through the attention operations.

        Args:
            x0: (N, S_0, E), Tokens from 1st modality.
            x1: (N, S_1, E), Tokens from 2nd modality.

        Returns:
            (N, S_0, E) + (N, S_1, E), attention output tokens for both modalities.
        """
        q0, k0, v0 = self.mod0_linear_proj(x0, x0)
        q1, k1, v1 = self.mod1_linear_proj(x1, x1)

        self_0, _ = self.mod0_self_mat_mul(q0, k0, v0)
        cross_0, _ = self.mod0_cross_mat_mul(q0, k1, v1)
        self_1, _ = self.mod1_self_mat_mul(q1, k1, v1)
        cross_1, _ = self.mod1_cross_mat_mul(q1, k0, v0)

        x0 = self_0 + cross_0
        x1 = self_1 + cross_1

        return x0, x1
