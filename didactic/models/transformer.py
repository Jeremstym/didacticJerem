import enum
import math
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union, cast
import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .layers import get_nn_module, MultiheadAttention, MultiheadCrossAttention
from .baselines import BidirectionalMultimodalAttention

ModuleType = Union[str, Callable[..., nn.Module]]
_INTERNAL_ERROR_MESSAGE = "Internal error. Please, open an issue."


class FT_Transformer(nn.Module):
    """Transformer with extra features.

    This module is the backbone of `FTTransformer`."""

    WARNINGS = {"first_prenormalization": True, "prenormalization": True}

    class FFN(nn.Module):
        """The Feed-Forward Network module used in every `Transformer` block."""

        def __init__(
            self,
            *,
            d_token: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout: float,
            activation: ModuleType,
        ):
            super().__init__()
            self.activation = get_nn_module(activation)
            _is_glu_activation = self.activation.__class__.__name__.lower().endswith("glu")
            self.linear_first = nn.Linear(
                d_token,
                d_hidden * (2 if _is_glu_activation else 1),
                bias_first,
            )
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

        def forward(self, x: Tensor) -> Tensor:
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x


    def __init__(
        self,
        *,
        d_token: int,
        n_self_blocks: int,
        n_cross_blocks: int,
        n_bidirectional_blocks: int,
        attention_n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: str,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: str,
        ffn_normalization: str,
        residual_dropout: float,
        prenormalization: bool,
        first_prenormalization: bool,
    ) -> None:
        """
        Parameters
        ----------
        d_token
            The size of one token for `_CategoricalFeatureTokenizer`.
        n_self_blocks
            Number of the `FT_Transformer` self-attention blocks, which should be non-negative.
        n_cross_blocks
            Number of the `FT_Transformer` cross-attention blocks, which should be non-negative if `cross_attention` is True.
        attention_n_heads
            Number of attention heads in each `FT_Transformer` block, which should be positive.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        attention_initialization
            Weights initialization scheme for Multi Headed Attention module.
        attention_normalization
            Normalization policy for attention layers. "layer_norm" is a good default.
        ffn_d_hidden
            Number of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        prenormalization, first_prenormalization
            Prenormalization to stabilize the training.
        cross_attention
            If 'true', the transformer will use cross attention instead of self-attention only.
        """
        super().__init__()
        if not prenormalization:
            assert (
                not first_prenormalization
            ), "If `prenormalization` is False, then `first_prenormalization` must be False"
            
            if self.WARNINGS["prenormalization"]:
                warnings.warn(
                    "prenormalization is set to False. Are you sure about this? "
                    "The training can become less stable. "
                    "You can turn off this warning by tweaking the "
                    "rtdl.Transformer.WARNINGS dictionary.",
                    UserWarning,
                )

        if prenormalization and first_prenormalization and self.WARNINGS["first_prenormalization"]:
            warnings.warn(
                "first_prenormalization is set to True. Are you sure about this? "
                "For example, the vanilla FTTransformer with "
                "first_prenormalization=True performs SIGNIFICANTLY worse. "
                "You can turn off this warning by tweaking the "
                "rtdl.Transformer.WARNINGS dictionary.",
                UserWarning,
            )
        
        assert not(n_cross_blocks and n_bidirectional_blocks), "Cannot use both cross-attention and bidirectional attention blocks"

        self.d_token = d_token
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.attention_initialization = attention_initialization
        self.attention_normalization = attention_normalization
        self.ffn_d_hidden = ffn_d_hidden
        self.ffn_dropout = ffn_dropout
        self.ffn_activation = ffn_activation
        self.ffn_normalization = ffn_normalization
        self.residual_dropout = residual_dropout
        self.prenormalization = prenormalization
        self.first_prenormalization = first_prenormalization       

        self.n_self_blocks = n_self_blocks
        self.n_cross_blocks = n_cross_blocks
        self.n_bidirectional_blocks = n_bidirectional_blocks
        
        layers = []

        
        if self.n_self_blocks:
            layers += [
                self._init_attention_block(layer_idx)
                for layer_idx in range(self.n_self_blocks)
            ]

        if self.n_cross_blocks:
            layers += [
                self._init_cross_attention_block(layer_idx)
                for layer_idx in range(self.n_self_blocks, self.n_self_blocks + self.n_cross_blocks)
            ]

        self.blocks = nn.ModuleList(layers)

    def _init_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        if self.n_cross_blocks:
            layer = nn.ModuleDict()
            for modality_side in ["l", "r"]:
                layer.update(
                    {
                        f"{modality_side}_attention": MultiheadAttention(
                            d_token=self.d_token,
                            n_heads=self.attention_n_heads,
                            dropout=self.attention_dropout,
                            bias=True,
                            initialization=self.attention_initialization,
                        ),
                        f"{modality_side}_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                        f"{modality_side}_ffn": self.FFN(
                            d_token=self.d_token,
                            d_hidden=self.ffn_d_hidden,
                            bias_first=True,
                            bias_second=True,
                            dropout=self.ffn_dropout,
                            activation=self.ffn_activation,
                        ),
                        f"{modality_side}_ffn_residual_dropout": nn.Dropout(self.residual_dropout),
                    }
                )
                if layer_idx or not self.prenormalization or self.first_prenormalization:
                    if modality_side == "l":
                        layer[f"{modality_side}_attention_normalization"] = get_nn_module(self.attention_normalization)
                    # layer[f"{modality_side}_attention_normalization"] = get_nn_module(self.attention_normalization)
                if modality_side == "l":
                    layer[f"{modality_side}_ffn_normalization"] = get_nn_module(self.ffn_normalization)
                # layer[f"{modality_side}_ffn_normalization"] = get_nn_module(self.ffn_normalization)
        else:
            layer = nn.ModuleDict(
                {
                    f"attention": MultiheadAttention(
                        d_token=self.d_token,
                        n_heads=self.attention_n_heads,
                        dropout=self.attention_dropout,
                        bias=True,
                        initialization=self.attention_initialization,
                    ),
                    f"attention_residual_dropout": nn.Dropout(self.residual_dropout),
                    f"ffn": self.FFN(
                        d_token=self.d_token,
                        d_hidden=self.ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=self.ffn_dropout,
                        activation=self.ffn_activation,
                    ),
                    f"ffn_residual_dropout": nn.Dropout(self.residual_dropout),
                }
            )
            if layer_idx or not self.prenormalization or self.first_prenormalization:
                layer[ f"attention_normalization"] = get_nn_module(self.attention_normalization)
            layer[ f"ffn_normalization"] = get_nn_module(self.ffn_normalization)
        
        return layer

    def _init_cross_attention_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                "cross_attention": MultiheadCrossAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
            }
        )

        for modality_side in ["l", "r"]:
            layer.update(
                {
                    f"{modality_side}_cross_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                    f"{modality_side}_ffn_0": self.FFN(
                        d_token=self.d_token,
                        d_hidden=self.ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=self.ffn_dropout,
                        activation=self.ffn_activation,
                    ),
                    f"{modality_side}_ffn_0_residual_dropout": nn.Dropout(self.residual_dropout),
                }
            )
            if layer_idx or not self.prenormalization or self.first_prenormalization:
                layer[f"{modality_side}_cross_attention_normalization"] = get_nn_module(self.attention_normalization)
            layer[f"{modality_side}_ffn_0_normalization"] = get_nn_module(self.ffn_normalization)

            layer.update(
                {
                    f"{modality_side}_self_attention": MultiheadAttention(
                        d_token=self.d_token,
                        n_heads=self.attention_n_heads,
                        dropout=self.attention_dropout,
                        bias=True,
                        initialization=self.attention_initialization,
                    ),
                    f"{modality_side}_self_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                    f"{modality_side}_ffn_1": self.FFN(
                        d_token=self.d_token,
                        d_hidden=self.ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=self.ffn_dropout,
                        activation=self.ffn_activation,
                    ),
                    f"{modality_side}_ffn_1_residual_dropout": nn.Dropout(self.residual_dropout),
                }
            )
            if layer_idx or not self.prenormalization or self.first_prenormalization:
                layer[f"{modality_side}_self_attention_normalization"] = get_nn_module(self.attention_normalization)
            layer[f"{modality_side}_ffn_1_normalization"] = get_nn_module(self.ffn_normalization)

        return layer

    def _init_bidirectional_block(self, layer_idx: int) -> nn.ModuleDict:
        layer = nn.ModuleDict(
            {
                "bidirectional_attention": BidirectionalMultimodalAttention(
                    d_token=self.d_token,
                    n_heads=self.attention_n_heads,
                    dropout=self.attention_dropout,
                    bias=True,
                    initialization=self.attention_initialization,
                ),
            }
        )

        for modality_side in ["l", "r"]:
            layer.update(
                {
                    f"{modality_side}_bidirectional_attention_residual_dropout": nn.Dropout(self.residual_dropout),
                    f"{modality_side}_ffn_bidirectional": self.FFN(
                        d_token=self.d_token,
                        d_hidden=self.ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=self.ffn_dropout,
                        activation=self.ffn_activation,
                    ),
                    f"{modality_side}_ffn_bidirectional_residual_dropout": nn.Dropout(self.residual_dropout),
                }
            )
            if layer_idx or not self.prenormalization or self.first_prenormalization:
                layer[f"{modality_side}_bidirectional_attention_normalization"] = get_nn_module(self.attention_normalization)
            layer[f"{modality_side}_ffn_bidirectional_normalization"] = get_nn_module(self.ffn_normalization)

        return layer


    def _start_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "l_cross_attention",
                    "r_cross_attention",
                    "l_ffn_0",
                    "r_ffn_0",
                    "l_self_attention",
                    "r_self_attention",
                    "l_ffn_1",
                    "r_ffn_1",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                    "l_attention",
                    "r_attention",
                    "l_ffn",
                    "r_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "bidirectional_attention":
                assert layer_name in [
                    "l_bidirectional_attention",
                    "r_bidirectional_attention",
                    "l_ffn_bidirectional",
                    "r_ffn_bidirectional",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = x
        if self.prenormalization:
            if (norm_key := f"{layer_name}_normalization") in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer: nn.ModuleDict, layer_name: str, x: Tensor, x_residual: Tensor, stage: str = "self_attention") -> Tensor:
        match stage:
            case "cross_attention":
                assert layer_name in [
                    "l_cross_attention",
                    "r_cross_attention",
                    "l_ffn_0",
                    "r_ffn_0",
                    "l_self_attention",
                    "r_self_attention",
                    "l_ffn_1",
                    "r_ffn_1",
                ], _INTERNAL_ERROR_MESSAGE
            case "self_attention":
                assert layer_name in [
                    "attention",
                    "ffn",
                    "l_attention",
                    "r_attention",
                    "l_ffn",
                    "r_ffn",
                ], _INTERNAL_ERROR_MESSAGE
            case "bidirectional_attention":
                assert layer_name in [
                    "l_bidirectional_attention",
                    "r_bidirectional_attention",
                    "l_ffn_bidirectional",
                    "r_ffn_bidirectional",
                ], _INTERNAL_ERROR_MESSAGE
            case _:
                raise ValueError(f"Invalid stage: {stage}, should be 'cross_attention' or 'self_attention'")

        x_residual = layer[f"{layer_name}_residual_dropout"](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"{layer_name}_normalization"](x)
        return x

    def forward(self, x: Tensor, x_context: Optional[Tensor] = None) -> Tensor:
        """
        Performs a forward pass through the successive transformer blocks.

        Args:
            x: (N, S, E), Sequence of tokens, where S is the sequence length, N is the batch size, and E is the
                embedding dimension.
            x_context: (N, S', E), Sequence of tokens from the second modality, if `n_bidirectional_blocks` is not 0.
                `S'` can be different from `S`, but `N` and `E` must be the same between both sequences.

        Returns:
            (N, S, E) / (N, S+S', E), The output sequence of the transformer.
        """

        if self.n_cross_blocks and x_context is None:
            raise ValueError(
                "x_context from which K and V are extracted, must be "
                "provided since the model includes cross-attention blocks."
            )
        
        if x.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x_context.ndim != 3:
            raise ValueError("The input tensor must have 3 dimensions: (n_objects, n_tokens, d_token)")
        if x_context is not None and x.shape[-1] != x_context.shape[-1]:
            raise ValueError("The input tensors must have the same embedding dimension")

        self.batch_size = x.shape[0] # Save the batch size for later use in explainability

        self_attention_blocks = self.blocks[: self.n_self_blocks]
        cross_attention_blocks = self.blocks[self.n_self_blocks :]
        bidirectional_attention_blocks = [] #FIXME: build bidirectional attention blocks later

        for block in self_attention_blocks:
            block = cast(nn.ModuleDict, block)
            if x_context is not None:

                # Normalize the tokens from both modalities if prenormalization is enabled
                x_residual = self._start_residual(block, "l_attention", x, stage="self_attention")

                # Forward pass through the self-attention block
                x_residual, _ = block["l_attention"](x_residual, x_residual)

                # Residual connections after the attention layer for both modalities
                x = self._end_residual(block, "l_attention", x, x_residual, stage="self_attention")

                # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                x_residual = self._start_residual(block, "l_ffn", x, stage="self_attention")
                x_residual = block["l_ffn"](x_residual)
                x = self._end_residual(block, "l_ffn", x, x_residual, stage="self_attention")

                # Do the same for the context modality
                x_context_residual = self._start_residual(block, "r_attention", x_context, stage="self_attention")
                x_context_residual, _ = block["r_attention"](x_context_residual, x_context_residual)
                x_context = self._end_residual(block, "r_attention", x_context, x_context_residual, stage="self_attention")
                x_context_residual = self._start_residual(block, "r_ffn", x_context, stage="self_attention")
                x_context_residual = block["r_ffn"](x_context_residual)
                x_context = self._end_residual(block, "r_ffn", x_context, x_context_residual, stage="self_attention")

            else:
                # Normalize the tokens from both modalities if prenormalization is enabled
                x_residual = self._start_residual(block, "attention", x, stage="self_attention")

                # Forward pass through the self-attention block
                x_residual, _ = block["attention"](x_residual, x_residual)

                # Residual connections after the attention layer for both modalities
                x = self._end_residual(block, "attention", x, x_residual, stage="self_attention")

                # Forward pass through the normalization, FFN layer, and residual connection for both modalities
                x_residual = self._start_residual(block, "ffn", x, stage="self_attention")
                x_residual = block["ffn"](x_residual)
                x = self._end_residual(block, "ffn", x, x_residual, stage="self_attention")

        
        for block in cross_attention_blocks:
            block = cast(nn.ModuleDict, block)

            # Normalize the tokens from both modalities if prenormalization is enabled
            x_residual = self._start_residual(block, "l_cross_attention", x, stage="cross_attention")
            x_context_residual = self._start_residual(block, "r_cross_attention", x_context, stage="cross_attention")

            # Forward pass through the cross-attention block
            x_residual, x_context_residual = block["cross_attention"](x_residual, x_context_residual)

            # Residual connections after the attention layer for both modalities
            x = self._end_residual(block, "l_cross_attention", x, x_residual, stage="cross_attention")
            x_context = self._end_residual(block, "r_cross_attention", x_context, x_context_residual, stage="cross_attention")

            # Forward pass through the normalization, FFN layer, and residual connection for both modalities
            x_residual = self._start_residual(block, "l_ffn_0", x, stage="cross_attention")
            x_residual = block["l_ffn_0"](x_residual)
            x = self._end_residual(block, "l_ffn_0", x, x_residual, stage="cross_attention")

            x_context_residual = self._start_residual(block, "r_ffn_0", x_context, stage="cross_attention")
            x_context_residual = block["r_ffn_0"](x_context_residual)
            x_context = self._end_residual(block, "r_ffn_0", x_context, x_context_residual, stage="cross_attention")

            # Forward pass through the self-attention block inside the cross-attention module for both modalities
            x_residual = self._start_residual(block, "l_self_attention", x, stage="cross_attention")
            x_residual, _ = block["l_self_attention"](x_residual, x_residual)
            x = self._end_residual(block, "l_self_attention", x, x_residual, stage="cross_attention")

            x_context_residual = self._start_residual(block, "r_self_attention", x_context, stage="cross_attention")
            x_context_residual, _ = block["r_self_attention"](x_context_residual, x_context_residual)
            x_context = self._end_residual(block, "r_self_attention", x_context, x_context_residual, stage="cross_attention")

            # Forward pass through the normalization, FFN layer, and residual connection for both modalities
            x_residual = self._start_residual(block, "l_ffn_1", x, stage="cross_attention")
            x_residual = block["l_ffn_1"](x_residual)
            x = self._end_residual(block, "l_ffn_1", x, x_residual, stage="cross_attention")

            x_context_residual = self._start_residual(block, "r_ffn_1", x_context, stage="cross_attention")
            x_context_residual = block["r_ffn_1"](x_context_residual)
            x_context = self._end_residual(block, "r_ffn_1", x_context, x_context_residual, stage="cross_attention")

        for block in bidirectional_attention_blocks:
            block = cast(nn.ModuleDict, block)

            # Normalize the tokens from both modalities if prenormalization is enabled
            x_residual = self._start_residual(block, "l_bidirectional_attention", x, stage="bidirectional_attention")
            x_context_residual = self._start_residual(block, "r_bidirectional_attention", x_context, stage="bidirectional_attention")

            # Forward pass through the bidirectional attention block
            x_residual, x_context_residual = block["bidirectional_attention"](x_residual, x_context_residual)

            # Residual connections after the attention layer for both modalities
            x = self._end_residual(block, "l_bidirectional_attention", x, x_residual, stage="bidirectional_attention")
            x_context = self._end_residual(block, "r_bidirectional_attention", x_context, x_context_residual, stage="bidirectional_attention")

            # Forward pass through the normalization, FFN layer, and residual connection for both modalities
            x_residual = self._start_residual(block, "l_ffn_bidirectional", x, stage="bidirectional_attention")
            x_residual = block["l_ffn_bidirectional"](x_residual)
            x = self._end_residual(block, "l_ffn_bidirectional", x, x_residual, stage="bidirectional_attention")

            x_context_residual = self._start_residual(block, "r_ffn_bidirectional", x_context, stage="bidirectional_attention")
            x_context_residual = block["r_ffn_bidirectional"](x_context_residual)
            x_context = self._end_residual(block, "r_ffn_bidirectional", x_context, x_context_residual, stage="bidirectional_attention") 

        return x, x_context