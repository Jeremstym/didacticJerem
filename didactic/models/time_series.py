from typing import Any, Dict, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TimeSeriesEmbedding(nn.Module):
    """Embedding for time series which resamples the time dim and/or passes through an arbitrary learnable model."""

    def __init__(self, resample_dim: int, model: nn.Module = None, corruption_rate: float = 0.0, pooling: bool = False) -> None:
        """Initializes class instance.

        Args:
            resample_dim: Target size for an interpolation resampling of the time series.
            model: Model that learns to embed the time series. If not provided, no projection is learned and the
                embedding is simply the resampled time series. The model should take as input a tensor of shape
                (N, `resample_dim`) and output a tensor of shape (N, E), where E is the embedding size.
        """
        super().__init__()
        self.model = model
        self.resample_dim = resample_dim
        self.corruption_rate = corruption_rate
        self.pooling = pooling

    def corrupt(self, x: Tensor, corruption_rate: float = 0.1) -> Tensor:
        """Corrupts the input tensor by setting a fraction of its values to zero.

        Args:
            x: (N, S, E), Input tensor to corrupt.
            corruption_rate: Fraction of values to set to zero.

        Returns:
            (N, S, E), Corrupted input tensor.
        """
        mask = torch.rand_like(x) < corruption_rate
        return x * mask

    def sequence_pooling_ts(self, x: Tensor, pooling: str = "mean") -> Tensor:
        """Applies a pooling operation along the time dimension of the input tensor.

        Args:
            x: (N, S, E), Input tensor to pool.
            pooling: Pooling operation to apply. Can be one of ["mean", "max"].

        Returns:
            (N, E), Pooled input tensor.
        """
        if pooling == "mean":
            return x.mean(dim=1)
        elif pooling == "max":
            return x.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling operation")
    
    def forward(self, time_series: Dict[Any, Tensor] | Sequence[Tensor]) -> Tensor:
        """Stacks the time series, optionally 1) resampling them and/or 2) projecting them to a target embedding.

        Args:
            time_series: (K: S, V: (N, ?)) or S * (N, ?): Time series batches to embed, where the dimensionality of each
                time series can vary.

        Returns:
            (N, S, E), Embedding of the time series.
        """
        if not isinstance(time_series, dict):
            time_series = {idx: t for idx, t in enumerate(time_series)}

        # Resample time series to make sure all of them are of `resample_dim`
        for t_id, t in time_series.items():
            if t.shape[-1] != self.resample_dim:
                # Temporarily reshape time series batch tensor to be 3D to be able to use torch's interpolation
                # (N, ?) -> (N, `resample_dim`)
                time_series[t_id] = F.interpolate(t.unsqueeze(1), size=self.resample_dim, mode="linear").squeeze(dim=1)

        # Extract the time series from the dictionary and stack them along the batch dimension
        x = list(time_series.values())  # (S, N, `resample_dim`)

        if self.model:
            # If provided with a learnable model, use it to predict the embedding of each time series separately
            x = [self.model(attr) for attr in x]  # (S, N, `resample_dim`) -> (S, N, E)

        # Stack the embeddings of all the time series (along the batch dimension) to make only one tensor
        x = torch.stack(x, dim=1)  # (S, N, E) -> (N, S, E)

        if self.corruption_rate:
            x = self.corrupt(x, self.corruption_rate)

        if self.pooling:
            x = self.sequence_pooling_ts(x) # (N, S, E) -> (N, E)

        return x
