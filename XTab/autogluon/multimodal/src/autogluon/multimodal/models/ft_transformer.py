import enum
import math
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union, cast
import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .utils import init_weights

ModuleType = Union[str, Callable[..., nn.Module]]
_INTERNAL_ERROR_MESSAGE = "Internal error. Please, open an issue."


def _is_glu_activation(activation: ModuleType):
    return isinstance(activation, str) and activation.endswith("glu") or activation in [ReGLU, GEGLU]


def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    if isinstance(module_type, str):
        if module_type == "reglu":
            return ReGLU()
        elif module_type == "geglu":
            return GEGLU()
        elif module_type == "gelu":
            return nn.GELU()
        elif module_type == "relu":
            return nn.ReLU()
        elif module_type == "leaky_relu":
            return nn.LeakyReLU()
        elif module_type == "layer_norm":
            return nn.LayerNorm(*args)
        else:
            try:
                cls = getattr(nn, module_type)
            except AttributeError as err:
                raise ValueError(f"Failed to construct the module {module_type} with the arguments {args}") from err
            return cls(*args)
    else:
        return module_type(*args)


def _all_or_none(values):
    return all(x is None for x in values) or all(x is not None for x in values)


def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].

    References:
    ----------
    [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].

    References:
    ----------
    [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


class ReGLU(nn.Module):
    """
    The ReGLU activation function from [1].

    References:
    ----------
    [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    """
    The GEGLU activation function from [1].

    References:
    ----------
    [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)


class CLSToken(nn.Module):
    """[CLS]-token for BERT-like inference.

    To learn about the [CLS]-based inference, see [1].

    When used as a module, the [CLS]-token is appended **to the end** of each item in
    the batch.

    References:
    ----------
    [1] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" 2018
    """

    def __init__(self, d_token: int, initialization: str) -> None:
        """
        Args:
            d_token: the size of token
            initialization: initialization policy for parameters. Must be one of
                :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
                corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`. In
                the paper [gorishniy2021revisiting], the 'uniform' initialization was
                used.

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        super().__init__()
        initialization_ = _TokenInitialization.from_str(initialization)
        self.weight = nn.Parameter(Tensor(d_token))
        initialization_.apply(self.weight, d_token)

    def expand(self, *leading_dimensions: int) -> Tensor:
        """Expand (repeat) the underlying [CLS]-token to a tensor with the given leading dimensions.

        A possible use case is building a batch of [CLS]-tokens. See `_CLSToken` for
        examples of usage.

        Note:
            Under the hood, the `torch.Tensor.expand` method is applied to the
            underlying :code:`weight` parameter, so gradients will be propagated as
            expected.

        Args:
            leading_dimensions: the additional new dimensions

        Returns:
            tensor of the shape :code:`(*leading_dimensions, len(self.weight))`
        """
        if not leading_dimensions:
            return self.weight
        new_dims = (1,) * (len(leading_dimensions) - 1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, x: Tensor) -> Tensor:
        """Append self **to the end** of each item in the batch (see `_CLSToken`)."""
        return torch.cat([x, self.expand(len(x), 1)], dim=1)


class _TokenInitialization(enum.Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"

    @classmethod
    def from_str(cls, initialization: str) -> "_TokenInitialization":
        try:
            return cls(initialization)
        except ValueError:
            valid_values = [x.value for x in _TokenInitialization]
            raise ValueError(f"initialization must be one of {valid_values}")

    def apply(self, x: Tensor, d: int) -> None:
        d_sqrt_inv = 1 / math.sqrt(d)
        if self == _TokenInitialization.UNIFORM:
            # used in the paper "Revisiting Deep Learning Models for Tabular Data";
            # is equivalent to `nn.init.kaiming_uniform_(x, a=math.sqrt(5   ))` (which is
            # used by torch to initialize nn.Linear.weight, for example)
            nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
        elif self == _TokenInitialization.NORMAL:
            nn.init.normal_(x, std=d_sqrt_inv)


class MultiheadAttention(nn.Module):
    """Multihead Attention (self-/cross-) with optional 'linear' attention.

    To learn more about Multihead Attention, see [1]. See the implementation
    of `Transformer` and the examples below to learn how to use the compression technique
    from [2] to speed up the module when the number of tokens is large.

    References:
    ----------
    [1] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" 2018
    [2] Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma "Linformer: Self-Attention with Linear Complexity", 2020
    """

    def __init__(
        self,
        *,
        d_token: int,
        n_heads: int,
        dropout: float,
        bias: bool,
        initialization: str,
    ) -> None:
        """
        Parameters
        ----------
        d_token:
            the token size. Must be a multiple of :code:`n_heads`.
        n_heads:
            the number of heads. If greater than 1, then the module will have
            an addition output layer (so called "mixing" layer).
        dropout:
            dropout rate for the attention map. The dropout is applied to
            *probabilities* and do not affect logits.
        bias:
            if `True`, then input (and output, if presented) layers also have bias.
            `True` is a reasonable default choice.
        initialization:
            initialization for input projection layers. Must be one of
            :code:`['kaiming', 'xavier']`. `kaiming` is a reasonable default choice.

        Raises
        ----------
            AssertionError: if requirements for the inputs are not met.
        """
        super().__init__()
        if n_heads > 1:
            assert d_token % n_heads == 0, "d_token must be a multiple of n_heads"
        assert initialization in ["kaiming", "xavier"]

        self.W_q = nn.Linear(d_token, d_token, bias)
        self.W_k = nn.Linear(d_token, d_token, bias)
        self.W_v = nn.Linear(d_token, d_token, bias)
        self.W_out = nn.Linear(d_token, d_token, bias) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.attn = None
        self.attn_gradients = None

        for m in [self.W_q, self.W_k, self.W_v]:
            # the "xavier" branch tries to follow torch.nn.MultiheadAttention;
            # the second condition checks if W_v plays the role of W_out; the latter one
            # is initialized with Kaiming in torch
            if initialization == "xavier" and (m is not self.W_v or self.W_out is not None):
                # gain is needed since W_qkv is represented with 3 separate layers (it
                # implies different fan_out)
                nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def save_attn(self, attn):
        self.attn = attn
    
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn(self):
        return self.attn

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: Optional[nn.Linear],
        value_compression: Optional[nn.Linear],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Perform the forward pass.

        Parameters
        ----------
        x_q:
            query tokens
        x_kv:
            key-value tokens
        key_compression:
            Linformer-style compression for keys
        value_compression:
            Linformer-style compression for values

        Returns:
        ----------
            (tokens, attention_stats)
        """
        assert _all_or_none(
            [key_compression, value_compression]
        ), "If key_compression is (not) None, then value_compression must (not) be None"
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0, _INTERNAL_ERROR_MESSAGE
        if key_compression is not None:
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)  # type: ignore

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention_logits = q @ k.transpose(1, 2) / math.sqrt(d_head_key)
        attention_probs = F.softmax(attention_logits, dim=-1)

        self.save_attn(attention_probs)
        if attention_probs.requires_grad:
            attention_probs.register_hook(self.save_attn_gradients)
        
        if self.dropout is not None:
            attention_probs = self.dropout(attention_probs)
        x = attention_probs @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x


class AdditiveAttention(nn.Module):
    """Additive Attention with linear complexity to input sequence length.

    Additive attention was proposed and used in FastFormer.
    See Ref. [1] for details.
    This implementation is motivated by: https://github.com/jrzaurin/pytorch-widedeep.git

    References:
    ----------
    [1] Wu, Chuhan, et al. "Fastformer: Additive attention can be all you need." arXiv preprint arXiv:2108.09084 (2021).
    """

    def __init__(
        self,
        *,
        d_token: int,
        n_heads: int,
        dropout: float,
        bias: bool,
        share_qv_weights: bool,
        initialization: str,
    ) -> None:
        """
        Parameters
        ----------
        d_token:
            the token size. Must be a multiple of :code:`n_heads`.
        n_heads:
            the number of heads. If greater than 1, then the module will have
            an addition output layer (so called "mixing" layer).
        dropout:
            dropout rate for the attention map. The dropout is applied to
            *probabilities* and do not affect logits.
        bias:
            if `True`, then input (and output, if presented) layers also have bias.
            `True` is a reasonable default choice.
        share_qv_weights:
            if 'True', then value and query transformation parameters are shared.
        initialization:
            initialization for input projection layers. Must be one of
            :code:`['kaiming', 'xavier']`. `kaiming` is a reasonable default choice.
        """
        super().__init__()

        assert d_token % n_heads == 0, "d_token must be a multiple of n_heads"
        assert initialization in ["kaiming", "xavier"]

        self.head_dim = d_token // n_heads
        self.n_heads = n_heads
        self.share_qv_weights = share_qv_weights
        self.dropout = nn.Dropout(dropout)
        trainable = []
        if share_qv_weights:
            self.qv_proj = nn.Linear(d_token, d_token, bias=bias)
            trainable.extend([self.qv_proj])
        else:
            self.q_proj = nn.Linear(d_token, d_token, bias=bias)
            self.v_proj = nn.Linear(d_token, d_token, bias=bias)
            trainable.extend([self.q_proj, self.v_proj])

        self.k_proj = nn.Linear(d_token, d_token, bias=bias)
        self.W_q = nn.Linear(d_token, n_heads)
        self.W_k = nn.Linear(d_token, n_heads)
        self.r_out = nn.Linear(d_token, d_token)
        trainable.extend([self.k_proj, self.W_q, self.W_k, self.r_out])

        if initialization == "xavier":
            self.apply(init_weights)
        else:
            for m in trainable:
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        *args,  # Not used. just to make the input consistent with MultiheadAttention.
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        batch_size, n_q_tokens, d_token = x_q.shape
        batch_size, n_k_tokens, d_token = x_kv.shape

        q = self.qv_proj(x_q) if self.share_qv_weights else self.q_proj(x_q)
        v = self.qv_proj(x_kv) if self.share_qv_weights else self.v_proj(x_kv)
        k = self.k_proj(x_kv)

        alphas = (self.W_q(q) / math.sqrt(self.head_dim)).softmax(dim=1)
        q_r = q.reshape(batch_size, n_q_tokens, self.n_heads, self.head_dim)
        global_query = torch.einsum(" b s h, b s h d -> b h d", alphas, q_r)
        global_query = global_query.reshape(batch_size, self.n_heads * self.head_dim).unsqueeze(1)

        p = k * global_query

        betas = (self.W_k(p) / math.sqrt(self.head_dim)).softmax(dim=1)
        p_r = p.reshape(batch_size, n_k_tokens, self.n_heads, self.head_dim)
        global_key = torch.einsum(" b s h, b s h d -> b h d", betas, p_r)
        global_key = global_key.reshape(batch_size, self.n_heads * self.head_dim).unsqueeze(1)

        u = v * global_key
        output = q + self.dropout(self.r_out(u))

        return output, {
            "query_weight": alphas,
            "key_weight": betas,
        }


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
            self.linear_first = nn.Linear(
                d_token,
                d_hidden * (2 if _is_glu_activation(activation) else 1),
                bias_first,
            )
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

        def forward(self, x: Tensor) -> Tensor:
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x

    class Head(nn.Module):
        """The final module of the `Transformer` that performs BERT-like inference."""

        def __init__(
            self,
            *,
            d_in: int,
            bias: bool,
            activation: ModuleType,
            normalization: ModuleType,
            d_out: int,
        ):
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
            self.linear_first = nn.Linear(d_in, d_in, bias)
            self.linear_second = nn.Linear(d_in, d_out, bias)

        def forward(self, x: Tensor) -> Tensor:
            x = x[:, -1]
            x = self.linear_first(x)
            x = self.normalization(x)
            x = self.activation(x)
            x = self.linear_second(x)
            return x

    class ContrastiveHead(nn.Module):
        """The final module of the `Transformer` that performs BERT-like inference."""

        def __init__(
            self,
            *,
            d_in: int,
            bias: bool,
            activation: ModuleType,
            normalization: ModuleType,
            d_out: int,
        ):
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
            self.linear1 = nn.Linear(d_in, d_in, bias)
            self.linear2 = nn.Linear(d_in, d_out, bias)

        def forward(self, x: Tensor) -> Tensor:
            x = x[:, :-1]
            x = self.linear1(x)
            x = self.normalization(x)
            x = self.activation(x)
            x = self.linear2(x)
            return x

    class ReconstructionHead(nn.Module):
        """The final module of the `Transformer` that performs BERT-like inference."""

        def __init__(
            self,
            *,
            d_in: int,
            bias: bool,
            activation: ModuleType,
            normalization: ModuleType,
            n_num_features: Optional[int] = 0,
            category_sizes: Optional[List[int]] = None,
        ):
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_in, bias)

            self.num_out = nn.ModuleList([nn.Linear(d_in, 1) for _ in range(n_num_features)])

            if category_sizes:
                self.cat_out = nn.ModuleList([nn.Linear(d_in, o) for o in category_sizes])
            else:
                self.cat_out = None
            self.category_sizes = category_sizes

        def forward(self, x: Tensor):
            x = x[:, :-1]
            x = self.linear(x)
            x = self.normalization(x)
            x = self.activation(x)

            if self.cat_out:
                x_cat = x[:, : len(self.category_sizes), :]
                cat_out = [f(x_cat[:, i]) for i, f in enumerate(self.cat_out)]
            else:
                cat_out = None

            x_num = x
            if self.category_sizes:
                x_num = x[:, len(self.category_sizes) :, :]
            num_out = [f(x_num[:, i]) for i, f in enumerate(self.num_out)]
            if len(num_out) > 0:
                num_out = torch.concat(num_out, dim=1)
            else:
                num_out = None
            return {"num_out": num_out, "cat_out": cat_out}

    def __init__(
        self,
        *,
        d_token: int,
        n_self_blocks: int,
        n_cross_blocks: int,
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
        last_layer_query_idx: Union[None, List[int], slice],
        n_tokens: Optional[int],
        kv_compression_ratio: Optional[float],
        kv_compression_sharing: Optional[str],
        head_activation: ModuleType,
        head_normalization: ModuleType,
        d_out: int,
        projection: Optional[bool] = False,
        additive_attention: Optional[bool] = False,
        share_qv_weights: Optional[bool] = False,
        row_attention: Optional[bool] = False,
        row_attention_layer: Optional[str] = None,
        global_token: Optional[bool] = False,
        cross_attention: Optional[bool] = False,
        final_module: Optional[bool] = False,
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
        n_tokens
            Number of tokens of the input sequence.
        kv_compression_ratio
            The compression ration to reduce the input sequence length.
        kv_compression_sharing
            If `true` the projections will share weights.
        head_activation
            Activation function type of the MLP layer.
        head_normalization
            Normalization scheme of the MLP layer.
        d_out
            Output dimension.
        projection
            Whether to use a project head.
        additive_attention
            If 'true' the transformer will use additive attention with linear complexity to sequence length.
        share_qv_weights
            if 'true', then value and query transformation parameters are shared in additive attention.
        row_attention
            If 'true', the transformer will use row attention.
        row_attention_layer
            The layer to apply row attention.
        global_token
            If 'true', the transformer will use global token.
        cross_attention
            If 'true', the transformer will use cross attention instead of self-attention only.
        final_module
            If 'true', the transformer will use a XTab-like final module after the cross-attention blocks.
        """
        super().__init__()
        assert n_self_blocks > 0, "n_self_blocks must be positive"
        self.n_self_blocks = n_self_blocks
        self.batch_size = None
        if cross_attention:
            assert n_cross_blocks > 0
            self.n_cross_blocks = n_cross_blocks
        if row_attention:
            row_attention_layer = row_attention_layer if row_attention_layer else "last"
        else:
            row_attention_layer = None
        if isinstance(last_layer_query_idx, int):
            raise ValueError(
                "last_layer_query_idx must be None, list[int] or slice. "
                f"Do you mean last_layer_query_idx=[{last_layer_query_idx}] ?"
            )
        if not prenormalization:
            assert (
                not first_prenormalization
            ), "If `prenormalization` is False, then `first_prenormalization` must be False"
        assert _all_or_none([n_tokens, kv_compression_ratio, kv_compression_sharing]), (
            "If any of the following arguments is (not) None, then all of them must (not) be None: "
            "n_tokens, kv_compression_ratio, kv_compression_sharing"
        )
        assert (
            additive_attention or not share_qv_weights
        ), "If `share_qv_weights` is True, then `additive_attention` must be True"
        assert kv_compression_sharing in [None, "headwise", "key-value", "layerwise"]
        if not prenormalization:
            if self.WARNINGS["prenormalization"]:
                warnings.warn(
                    "prenormalization is set to False. Are you sure about this? "
                    "The training can become less stable. "
                    "You can turn off this warning by tweaking the "
                    "rtdl.Transformer.WARNINGS dictionary.",
                    UserWarning,
                )
            assert (
                not first_prenormalization
            ), "If prenormalization is False, then first_prenormalization is ignored and must be set to False"
        if prenormalization and first_prenormalization and self.WARNINGS["first_prenormalization"]:
            warnings.warn(
                "first_prenormalization is set to True. Are you sure about this? "
                "For example, the vanilla FTTransformer with "
                "first_prenormalization=True performs SIGNIFICANTLY worse. "
                "You can turn off this warning by tweaking the "
                "rtdl.Transformer.WARNINGS dictionary.",
                UserWarning,
            )

        def make_kv_compression():
            assert n_tokens and kv_compression_ratio, _INTERNAL_ERROR_MESSAGE  # for mypy
            # https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/examples/linformer/linformer_src/modules/multihead_linear_attention.py#L83
            return nn.Linear(n_tokens, int(n_tokens * kv_compression_ratio), bias=False)

        self.shared_kv_compression = (
            make_kv_compression() if kv_compression_ratio and kv_compression_sharing == "layerwise" else None
        )

        self.prenormalization = prenormalization
        self.last_layer_query_idx = last_layer_query_idx
        self.row_attention = row_attention
        self.row_attention_layer = row_attention_layer
        self.global_token = global_token
        self.cross_attention = cross_attention
        self.final_module = final_module

        self.blocks = nn.ModuleList([])
        if self.row_attention:
            self.row_attention_layers = nn.ModuleDict(
                {
                    "row_attention": MultiheadAttention(
                        d_token=d_token,
                        n_heads=attention_n_heads,
                        dropout=attention_dropout,
                        bias=True,
                        initialization=attention_initialization,
                    ),
                    "row_ffn": FT_Transformer.FFN(
                        d_token=d_token,
                        d_hidden=ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=ffn_dropout,
                        activation=ffn_activation,
                    ),
                    "row_attention_residual_dropout": nn.Dropout(residual_dropout),
                    "row_ffn_residual_dropout": nn.Dropout(residual_dropout),
                    "row_output": nn.Identity(),  # for hooks-based introspection
                }
            )
            # for p in self.row_attention_layers.parameters():
            #     nn.init.zeros_(p)

        if self.cross_attention:
            for layer_idx in range(self.n_self_blocks):
                layer = nn.ModuleDict(
                    {
                        "attention": MultiheadAttention(
                            d_token=d_token,
                            n_heads=attention_n_heads,
                            dropout=attention_dropout,
                            bias=True,
                            initialization=attention_initialization,
                        ),
                        "attention_residual_dropout": nn.Dropout(residual_dropout),
                        "ffn": FT_Transformer.FFN(
                            d_token=d_token,
                            d_hidden=ffn_d_hidden,
                            bias_first=True,
                            bias_second=True,
                            dropout=ffn_dropout,
                            activation=ffn_activation,
                        ),
                        "ffn_residual_dropout": nn.Dropout(residual_dropout),
                        "ts_attention": MultiheadAttention(
                            d_token=d_token,
                            n_heads=attention_n_heads,
                            dropout=attention_dropout,
                            bias=True,
                            initialization=attention_initialization,
                        ),
                        "ts_attention_residual_dropout": nn.Dropout(residual_dropout),
                        "ts_ffn": FT_Transformer.FFN(
                            d_token=d_token,
                            d_hidden=ffn_d_hidden,
                            bias_first=True,
                            bias_second=True,
                            dropout=ffn_dropout,
                            activation=ffn_activation,
                        ),
                        "ts_ffn_residual_dropout": nn.Dropout(residual_dropout),
                        "output": nn.Identity(),  # for hooks-based introspection
                    }
                )

                if layer_idx or not prenormalization or first_prenormalization:
                    layer["attention_normalization"] = _make_nn_module(attention_normalization, d_token)
                layer["ffn_normalization"] = _make_nn_module(ffn_normalization, d_token)
                if kv_compression_ratio and self.shared_kv_compression is None:
                    layer["key_compression"] = make_kv_compression()
                    if kv_compression_sharing == "headwise":
                        layer["value_compression"] = make_kv_compression()
                    else:
                        assert kv_compression_sharing == "key-value", _INTERNAL_ERROR_MESSAGE
                
                self.blocks.append(layer)

            for layer_idx in range(n_cross_blocks):
                layer = nn.ModuleDict(
                    {
                        "cross_attention_tabimg": MultiheadAttention(
                            d_token=d_token,
                            n_heads=attention_n_heads,
                            dropout=attention_dropout,
                            bias=True,
                            initialization=attention_initialization,
                        ),
                        "cross_attention_tabimg_residual_dropout": nn.Dropout(residual_dropout),
                        "cross_attention_ffn_tabimg": FT_Transformer.FFN(
                            d_token=d_token,
                            d_hidden=ffn_d_hidden,
                            bias_first=True,
                            bias_second=True,
                            dropout=ffn_dropout,
                            activation=ffn_activation,
                        ),
                        "cross_attention_ffn_tabimg_residual_dropout": nn.Dropout(residual_dropout),
                        "cross_attention_imgtab": MultiheadAttention(
                            d_token=d_token,
                            n_heads=attention_n_heads,
                            dropout=attention_dropout,
                            bias=True,
                            initialization=attention_initialization,
                        ),
                        "cross_attention_imgtab_residual_dropout": nn.Dropout(residual_dropout),
                        "cross_attention_ffn_imgtab": FT_Transformer.FFN(
                            d_token=d_token,
                            d_hidden=ffn_d_hidden,
                            bias_first=True,
                            bias_second=True,
                            dropout=ffn_dropout,
                            activation=ffn_activation,
                        ),
                        "cross_attention_ffn_imgtab_residual_dropout": nn.Dropout(residual_dropout),
                        "self_attention_tabtab": MultiheadAttention(
                            d_token=d_token,
                            n_heads=attention_n_heads,
                            dropout=attention_dropout,
                            bias=True,
                            initialization=attention_initialization,
                        ),
                        "self_attention_tabtab_residual_dropout": nn.Dropout(residual_dropout),
                        "self_attention_ffn_tabtab": FT_Transformer.FFN(
                            d_token=d_token,
                            d_hidden=ffn_d_hidden,
                            bias_first=True,
                            bias_second=True,
                            dropout=ffn_dropout,
                            activation=ffn_activation,
                        ),
                        "self_attention_ffn_tabtab_residual_dropout": nn.Dropout(residual_dropout),
                        "self_attention_imgimg": MultiheadAttention(
                            d_token=d_token,
                            n_heads=attention_n_heads,
                            dropout=attention_dropout,
                            bias=True,
                            initialization=attention_initialization,
                        ),
                        "self_attention_imgimg_residual_dropout": nn.Dropout(residual_dropout),
                        "self_attention_ffn_imgimg": FT_Transformer.FFN(
                            d_token=d_token,
                            d_hidden=ffn_d_hidden,
                            bias_first=True,
                            bias_second=True,
                            dropout=ffn_dropout,
                            activation=ffn_activation,
                        ),
                        "self_attention_ffn_imgimg_residual_dropout": nn.Dropout(residual_dropout),
                        "cross_output": nn.Identity(),  # for hooks-based introspection
                    }
                )
                if layer_idx or not prenormalization or first_prenormalization:
                    layer["cross_attention_tabimg_normalization"] = _make_nn_module(attention_normalization, d_token)
                    layer["cross_attention_imgtab_normalization"] = _make_nn_module(attention_normalization, d_token)
                    layer["self_attention_tabtab_normalization"] = _make_nn_module(attention_normalization, d_token)
                    layer["self_attention_imgimg_normalization"] = _make_nn_module(attention_normalization, d_token)
                layer["cross_attention_ffn_tabimg_normalization"] = _make_nn_module(ffn_normalization, d_token)
                layer["cross_attention_ffn_imgtab_normalization"] = _make_nn_module(ffn_normalization, d_token)
                layer["self_attention_ffn_tabtab_normalization"] = _make_nn_module(ffn_normalization, d_token)
                layer["self_attention_ffn_imgimg_normalization"] = _make_nn_module(ffn_normalization, d_token)
                                
                self.blocks.append(layer)

            if self.final_module:
                layer = nn.ModuleDict(
                    {
                        "final_attention": MultiheadAttention(
                            d_token=d_token,
                            n_heads=attention_n_heads,
                            dropout=attention_dropout,
                            bias=True,
                            initialization=attention_initialization,
                        ),
                        "final_attention_residual_dropout": nn.Dropout(residual_dropout),
                        "final_ffn": FT_Transformer.FFN(
                            d_token=d_token,
                            d_hidden=ffn_d_hidden,
                            bias_first=True,
                            bias_second=True,
                            dropout=ffn_dropout,
                            activation=ffn_activation,
                        ),
                        "final_ffn_residual_dropout": nn.Dropout(residual_dropout),
                        "output": nn.Identity(),  # for hooks-based introspection
                    }
                )

                self.blocks.append(layer)


        else:
            for layer_idx in range(n_self_blocks):
                layer = nn.ModuleDict(
                    {
                        "attention": AdditiveAttention(
                            d_token=d_token,
                            n_heads=attention_n_heads,
                            dropout=attention_dropout,
                            bias=True,
                            share_qv_weights=share_qv_weights,
                            initialization=attention_initialization,
                        )
                        if additive_attention
                        else MultiheadAttention(
                            d_token=d_token,
                            n_heads=attention_n_heads,
                            dropout=attention_dropout,
                            bias=True,
                            initialization=attention_initialization,
                        ),
                        "ffn": FT_Transformer.FFN(
                            d_token=d_token,
                            d_hidden=ffn_d_hidden,
                            bias_first=True,
                            bias_second=True,
                            dropout=ffn_dropout,
                            activation=ffn_activation,
                        ),
                        "attention_residual_dropout": nn.Dropout(residual_dropout),
                        "ffn_residual_dropout": nn.Dropout(residual_dropout),
                        "output": nn.Identity(),  # for hooks-based introspection
                    }
                )

                if layer_idx or not prenormalization or first_prenormalization:
                    layer["attention_normalization"] = _make_nn_module(attention_normalization, d_token)
                layer["ffn_normalization"] = _make_nn_module(ffn_normalization, d_token)
                if kv_compression_ratio and self.shared_kv_compression is None:
                    layer["key_compression"] = make_kv_compression()
                    if kv_compression_sharing == "headwise":
                        layer["value_compression"] = make_kv_compression()
                    else:
                        assert kv_compression_sharing == "key-value", _INTERNAL_ERROR_MESSAGE
                if row_attention:
                    self.row_attention_layers = nn.ModuleDict(
                        {
                            "row_attention": MultiheadAttention(
                                d_token=d_token,
                                n_heads=attention_n_heads,
                                dropout=attention_dropout,
                                bias=True,
                                initialization=attention_initialization,
                            ),
                            "row_ffn": FT_Transformer.FFN(
                                d_token=d_token,
                                d_hidden=ffn_d_hidden,
                                bias_first=True,
                                bias_second=True,
                                dropout=ffn_dropout,
                                activation=ffn_activation,
                            ),
                            "row_attention_residual_dropout": nn.Dropout(residual_dropout),
                            "row_ffn_residual_dropout": nn.Dropout(residual_dropout),
                            "row_output": nn.Identity(),  # for hooks-based introspection
                        }
                    )
                    # for p in self.row_attention_layers.parameters():
                    #     nn.init.zeros_(p)
                    layer.update(self.row_attention_layers)
                
                self.blocks.append(layer)

        self.head = (
            FT_Transformer.Head(
                d_in=d_token,
                d_out=d_out,
                bias=True,
                activation=head_activation,  # type: ignore
                normalization=head_normalization if prenormalization else "Identity",
            )
            if projection
            else nn.Identity()
        )

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer["key_compression"], layer["value_compression"])
            if "key_compression" in layer and "value_compression" in layer
            else (layer["key_compression"], layer["key_compression"])
            if "key_compression" in layer
            else (None, None)
        )

    def _start_residual(self, layer, stage, x, cross_attention=False):
        if cross_attention:
            assert stage in ["cross_attention_tabimg",
            "cross_attention_imgtab",
            "cross_attention_ffn_tabimg",
            "cross_attention_ffn_imgtab",
            "self_attention_tabtab",
            "self_attention_imgimg",
            "self_attention_ffn_imgimg",
            "self_attention_ffn_tabtab"], _INTERNAL_ERROR_MESSAGE
        elif self.cross_attention:
            assert stage in ["attention", "ffn", "ts_attention", "ts_ffn"], _INTERNAL_ERROR_MESSAGE
        elif not self.row_attention:
            assert stage in ["attention", "ffn"], _INTERNAL_ERROR_MESSAGE
        else:
            assert stage in ["attention", "ffn", "row_attention", "row_ffn"], _INTERNAL_ERROR_MESSAGE
        x_residual = x
        if self.prenormalization:
            norm_key = f"{stage}_normalization"
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer, stage, x, x_residual, cross_attention=False):
        if cross_attention:
            assert stage in ["cross_attention_tabimg",
            "cross_attention_imgtab",
            "cross_attention_ffn_tabimg",
            "cross_attention_ffn_imgtab",
            "self_attention_tabtab",
            "self_attention_imgimg",
            "self_attention_ffn_imgimg",
            "self_attention_ffn_tabtab"], _INTERNAL_ERROR_MESSAGE
        elif self.cross_attention:
            assert stage in ["attention", "ffn", "ts_attention", "ts_ffn"], _INTERNAL_ERROR_MESSAGE
        elif not self.row_attention:
            assert stage in ["attention", "ffn"], _INTERNAL_ERROR_MESSAGE
        else:
            assert stage in ["attention", "ffn", "row_attention", "row_ffn"], _INTERNAL_ERROR_MESSAGE
        x_residual = layer[f"{stage}_residual_dropout"](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"{stage}_normalization"](x)
        return x

    def _start_global_token(self, x):
        if self.global_token:
            x = torch.concat(
                [torch.mean(x, dim=1).unsqueeze(1), x],
                dim=1,
            )
        return x

    def _end_global_token(self, x):
        if self.global_token:
            x = x[:, 1:]
        return x

    def forward(self, x: Tensor, x_context: Optional[Tensor] = None) -> Tensor:
        assert x.ndim == 3, "The input must have 3 dimensions: (n_objects, n_tokens, d_token)"

        self.batch_size = x.shape[0]
        
        for layer_idx, layer in enumerate(self.blocks):
            layer = cast(nn.ModuleDict, layer)

            if self.cross_attention:
                if x_context is None:
                    raise ValueError("x_context must be provided for cross-attention")
                if layer_idx in list(range(self.n_self_blocks)):
                    x = self._start_global_token(x)
                    x_residual = self._start_residual(layer, "attention", x)
                    x_residual = layer["attention"](
                        x_residual,
                        x_residual,
                        None,
                        None,
                    )
                    x = self._end_residual(layer, "attention", x, x_residual)
                    x_residual = self._start_residual(layer, "ffn", x)
                    x_residual = layer["ffn"](x_residual)
                    x = self._end_residual(layer, "ffn", x, x_residual)
                    x = layer["output"](x)
                    x = self._end_global_token(x)

                    x_context = self._start_global_token(x_context)
                    x_residual = self._start_residual(layer, "ts_attention", x_context)
                    x_residual = layer["ts_attention"](
                        x_residual,
                        x_residual,
                        None,
                        None,
                    )
                    x_context = self._end_residual(layer, "ts_attention", x_context, x_residual)
                    x_residual = self._start_residual(layer, "ts_ffn", x_context)
                    x_residual = layer["ts_ffn"](x_residual)
                    x_context = self._end_residual(layer, "ts_ffn", x_context, x_residual)
                    x_context = layer["output"](x_context)
                    x_context = self._end_global_token(x_context)
                                 
                if layer_idx in list(range(self.n_self_blocks, self.n_self_blocks + self.n_cross_blocks)): 
                    x = x_ctx_img = self._start_global_token(x)
                    x_context = self._start_global_token(x_context)     
                    x_residual = self._start_residual(layer, "cross_attention_tabimg", x, cross_attention=True)
                    x_residual = layer["cross_attention_tabimg"](
                        x_residual,
                        x_context,
                        None,
                        None,
                    )
                    x = self._end_residual(layer, "cross_attention_tabimg", x, x_residual, cross_attention=True)
                    x_residual = self._start_residual(layer, "cross_attention_ffn_tabimg", x, cross_attention=True)
                    x_residual = layer["cross_attention_ffn_tabimg"](x_residual)
                    x = self._end_residual(layer, "cross_attention_ffn_tabimg", x, x_residual, cross_attention=True)
                    # x = layer["cross_attention_tabimg_normalization"](x)
                    x_residual = self._start_residual(layer, "self_attention_tabtab", x, cross_attention=True)
                    x_residual = layer["self_attention_tabtab"](
                        x_residual,
                        x_residual,
                        None,
                        None,
                    )
                    x = self._end_residual(layer, "self_attention_tabtab", x, x_residual, cross_attention=True)
                    x_residual = self._start_residual(layer, "self_attention_ffn_tabtab", x, cross_attention=True)
                    x_residual = layer["self_attention_ffn_tabtab"](x_residual)
                    x = self._end_residual(layer, "self_attention_ffn_tabtab", x, x_residual, cross_attention=True)

                    x_residual = self._start_residual(layer, "cross_attention_imgtab", x_context, cross_attention=True)
                    x_residual = layer["cross_attention_imgtab"](
                        x_residual,
                        x_ctx_img,
                        None,
                        None,
                    )
                    x_context = self._end_residual(layer, "cross_attention_imgtab", x_context, x_residual, cross_attention=True)
                    x_residual = self._start_residual(layer, "cross_attention_ffn_imgtab", x_context, cross_attention=True)
                    x_residual = layer["cross_attention_ffn_imgtab"](x_residual)
                    x_context = self._end_residual(layer, "cross_attention_ffn_imgtab", x_context, x_residual, cross_attention=True)
                    # x_context = layer["cross_attention_imgtab_normalization"](x_context)
                    x_residual = self._start_residual(layer, "self_attention_imgimg", x_context, cross_attention=True)
                    x_residual = layer["self_attention_imgimg"](
                        x_residual,
                        x_residual,
                        None,
                        None,
                    )
                    x_context = self._end_residual(layer, "self_attention_imgimg", x_context, x_residual, cross_attention=True)
                    x_residual = self._start_residual(layer, "self_attention_ffn_imgimg", x_context, cross_attention=True)
                    x_residual = layer["self_attention_ffn_imgimg"](x_residual)
                    x_context = self._end_residual(layer, "self_attention_ffn_imgimg", x_context, x_residual, cross_attention=True)

                    # if layer_idx == self.n_cross_blocks - 1:
                    #     x = torch.cat([x, x_context], dim=1)
                    #     x = layer["cross_output"](x)
                    # else:
                    #     x = x_tab
                    #     x_context = x_ts

                    x = self._end_global_token(x)
                    x_context = self._end_global_token(x_context)

                if layer_idx == self.n_self_blocks + self.n_cross_blocks - 1:
                    x = torch.cat([x, x_context], dim=1)
                    x = layer["cross_output"](x)
                    if self.final_module:
                        x = self._start_global_token(x)
                        x_residual = self._start_residual(layer, "final_attention", x)
                        x_residual = layer["final_attention"](
                            x_residual,
                            x_residual,
                            None,
                            None,
                        )
                        x = self._end_residual(layer, "final_attention", x, x_residual)
                        x_residual = self._start_residual(layer, "final_ffn", x)
                        x_residual = layer["final_ffn"](x_residual)
                        x = self._end_residual(layer, "final_ffn", x, x_residual)
                        x = layer["output"](x)
                        x = self._end_global_token(x)

            else:
                if self.row_attention_layer == "first" and layer_idx == 0:
                    x = torch.transpose(x, 0, 1)
                    x = self._start_global_token(x)
                    x_residual = self._start_residual(layer, "row_attention", x)
                    x_residual, _ = layer["row_attention"](
                        x_residual,
                        x_residual,
                        None,
                        None,
                    )
                    x = self._end_residual(layer, "row_attention", x, x_residual)
                    x_residual = self._start_residual(layer, "row_ffn", x)
                    x_residual = layer["row_ffn"](x_residual)
                    x = self._end_residual(layer, "row_ffn", x, x_residual)
                    x = layer["row_output"](x)
                    x = self._end_global_token(x)
                    x = torch.transpose(x, 0, 1)

                query_idx = self.last_layer_query_idx if layer_idx + 1 == len(self.blocks) else None

                x = self._start_global_token(x)
                x_residual = self._start_residual(layer, "attention", x)
                x_residual = layer["attention"](
                    x_residual if query_idx is None else x_residual[:, query_idx],
                    x_residual,
                    *self._get_kv_compressions(layer),
                )
                if query_idx is not None:
                    x = x[:, query_idx]
                x = self._end_residual(layer, "attention", x, x_residual)

                x_residual = self._start_residual(layer, "ffn", x)
                x_residual = layer["ffn"](x_residual)
                x = self._end_residual(layer, "ffn", x, x_residual)
                x = layer["output"](x)
                x = self._end_global_token(x)

                if self.row_attention_layer == "shared" or (
                    self.row_attention_layer == "last" and layer_idx + 1 == len(self.blocks)
                ):
                    x = torch.transpose(x, 0, 1)
                    x = self._start_global_token(x)
                    x_residual = self._start_residual(layer, "row_attention", x)
                    x_residual = layer["row_attention"](
                        x_residual,
                        x_residual,
                        None,
                        None,
                    )
                    x = self._end_residual(layer, "row_attention", x, x_residual)
                    x_residual = self._start_residual(layer, "row_ffn", x)
                    x_residual = layer["row_ffn"](x_residual)
                    x = self._end_residual(layer, "row_ffn", x, x_residual)
                    x = layer["row_output"](x)
                    x = self._end_global_token(x)
                    x = torch.transpose(x, 0, 1)
       

        x = self.head(x)

        # mean attention map over attention heads
        # batch_size = x.shape[0]
        # attention_tuple = attention_map.split(batch_size, dim=0)
        # attention_map = torch.stack(attention_tuple, dim=0).mean(dim=0)

        return x
