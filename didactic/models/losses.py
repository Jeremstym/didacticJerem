import math
from typing import Literal, Tuple, Union, Dict, Callable, List

import torch
from scipy.special import binom, factorial
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn import init

import didactic.models.transformer
from didactic.models.layers import _QKVLinearProjection, _QKVMatrixMultiplication

import hydra
from omegaconf import DictConfig

from .layers import get_nn_module

ModuleType = Union[str, Callable[..., nn.Module]]


class OrthogonalLoss(nn.Module):
    """Orthogonal Loss."""

    def __init__(self):
        """Initializes class instance.

        Args:
            d_token: Token size. Must be a multiple of `n_heads`.
            n_heads: Number of attention heads. If greater than 1, then the module will have an additional output layer
                (so called "mixing" layer).
        """
        super().__init__()

    def forward(self, x_unique: Tensor, x_shared: Tensor) -> Tensor:
        """Performs a forward pass through the loss function.

        Args:
            x_unique: (N, E), Tokens.
            x_shared: (N, E), Tokens.

        Returns:
            Scalar loss value.
        """
        assert x_unique.shape[1] == x_shared.shape[1], "Input tensors must have the same embedding dimension."
        assert x_shared.ndim == 2, "Input tensor must have 2 dimensions, (N, E)."
        x = torch.mm(x_shared, x_unique.t())
        return torch.norm(x, p="fro") ** 2


class DecouplingLoss(nn.Module):
    """Normalized Temperature-scaled Cross-Entropy Loss with Decoupling."""

    def __init__(self, temperature: float = 0.1):
        """Initializes class instance.

        Args:
            temperature: Temperature scaling factor.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, x_unique: Tensor, x_shared: Tensor, ts: Tensor) -> Tensor:
        """Performs a forward pass through the loss function.

        Args:
            x_unique: (N, E), Embeddings.
            x_shared: (N, E), Embeddings.

        Returns:
            Scalar loss value.
        """
        x_unique = F.normalize(x_unique, p=2, dim=1)
        x_shared = F.normalize(x_shared, p=2, dim=1)
        ts = F.normalize(ts, p=2, dim=1)
        sim_shared = torch.mm(x_shared, ts.t())
        sim_unique = torch.mm(x_unique, ts.t())
        sim_shared /= self.temperature
        sim_unique /= self.temperature
        sim_shared = torch.exp(sim_shared)
        sim_unique = torch.exp(sim_unique)

        loss_ts_unique = -torch.log(sim_shared.diag() / torch.sum(sim_unique, dim=1)).mean()
        loss_ts_shared = -torch.log(sim_shared.diag() / torch.sum(sim_unique, dim=0)).mean()

        loss = (loss_ts_unique + loss_ts_shared) / 2

        return loss

class NTXentLossDecoupling2(nn.Module):
    """Normalized Temperature-scaled Cross-Entropy Loss with Decoupling."""

    def __init__(self, temperature: float = 0.1):
        """Initializes class instance.

        Args:
            temperature: Temperature scaling factor.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, x_unique: Tensor, x_shared: Tensor, ts: Tensor) -> Tensor:
        """Performs a forward pass through the loss function.

        Args:
            x_unique: (N, E), Embeddings.
            x_shared: (N, E), Embeddings.

        Returns:
            Scalar loss value.
        """
        x_unique = F.normalize(x_unique, p=2, dim=1)
        x_shared = F.normalize(x_shared, p=2, dim=1)
        ts = F.normalize(ts, p=2, dim=1)
        sim_shared = torch.mm(x_shared, ts.t())
        sim_unique = torch.mm(x_unique, x_shared.t())
        sim_shared /= self.temperature
        sim_unique /= self.temperature
        sim_shared = torch.exp(sim_shared)
        sim_unique = torch.exp(sim_unique)

        loss_ts_unique = -torch.log(sim_shared.diag() / torch.sum(sim_unique, dim=1)).mean()
        loss_ts_shared = -torch.log(sim_shared.diag() / torch.sum(sim_unique, dim=0)).mean()

        loss = (loss_ts_unique + loss_ts_shared) / 2

        return loss

class SupInfoNCELossDecoupling(nn.Module):
    """Normalized Temperature-scaled Cross-Entropy Loss with Decoupling."""

    def __init__(self, temperature: float = 0.1, margin: float = 0.0):
        """Initializes class instance.

        Args:
            temperature: Temperature scaling factor.
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, x_unique: Tensor, x_shared: Tensor, ts: Tensor) -> Tensor:
        """Performs a forward pass through the loss function.

        Args:
            x_unique: (N, E), Embeddings.
            x_shared: (N, E), Embeddings.

        Returns:
            Scalar loss value.
        """
        x_unique = F.normalize(x_unique, p=2, dim=1)
        x_shared = F.normalize(x_shared, p=2, dim=1)
        ts = F.normalize(ts, p=2, dim=1)
        sim_shared = torch.mm(x_shared, ts.t())
        sim_unique = torch.mm(x_unique, ts.t())
        sim_shared -= self.margin
        sim_shared /= self.temperature
        sim_unique /= self.temperature
        sim_shared = torch.exp(sim_shared)
        sim_unique = torch.exp(sim_unique)

        loss_ts_unique = -torch.log(sim_shared.diag() / (torch.sum(sim_unique, dim=1) + sim_shared.diag())).mean()
        loss_ts_shared = -torch.log(sim_shared.diag() / (torch.sum(sim_unique, dim=0) + sim_shared.diag())).mean()

        loss = (loss_ts_unique + loss_ts_shared) / 2

        return loss

class SupInfoNCELossDecoupling2(nn.Module):
    """Normalized Temperature-scaled Cross-Entropy Loss with Decoupling."""

    def __init__(self, temperature: float = 0.1, margin: float = 0.0):
        """Initializes class instance.

        Args:
            temperature: Temperature scaling factor.
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, x_unique: Tensor, x_shared: Tensor, ts: Tensor) -> Tensor:
        """Performs a forward pass through the loss function.

        Args:
            x_unique: (N, E), Embeddings.
            x_shared: (N, E), Embeddings.

        Returns:
            Scalar loss value.
        """
        x_unique = F.normalize(x_unique, p=2, dim=1)
        x_shared = F.normalize(x_shared, p=2, dim=1)
        ts = F.normalize(ts, p=2, dim=1)
        sim_shared = torch.mm(x_shared, ts.t())
        sim_unique = torch.mm(x_unique, x_shared.t())
        sim_shared -= self.margin
        sim_shared /= self.temperature
        sim_unique /= self.temperature
        sim_shared = torch.exp(sim_shared)
        sim_unique = torch.exp(sim_unique)

        loss_ts_unique = -torch.log(sim_shared.diag() / (torch.sum(sim_unique, dim=1)) + sim_shared.diag()).mean()
        loss_ts_shared = -torch.log(sim_shared.diag() / (torch.sum(sim_unique, dim=0)) + sim_shared.diag()).mean()

        loss = (loss_ts_unique + loss_ts_shared) / 2

        return loss

class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy Loss."""

    def __init__(self, kernel: Literal["gaussian", "laplacian", "linear"], sigma: float = 1.0):
        """Initializes class instance.

        Args:
            kernel: Kernel function. Must be one of ['gaussian', 'laplacian', 'linear'].
            sigma: Kernel parameter.
        """
        super().__init__()
        self.kernel = kernel
        self.sigma = sigma

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute the Maximum Mean Discrepancy (MMD) between two sequences of tokens.

        Args:
        x: First sample (shape: [N, S, E])
        y: Second sample (shape: [N, S', E])
        kernel: Kernel type ('rbf' or 'multiscale')

        Returns:
        MMD loss
        """
        N, S, E = x.shape
        N, S_prime, E = y.shape

        # Reshape x and y to [N*S, E] and [N*S', E]
        x_flat = x.reshape(-1, E)
        y_flat = y.reshape(-1, E)

        xx = torch.mm(x_flat, x_flat.t())
        yy = torch.mm(y_flat, y_flat.t())
        zz = torch.mm(x_flat, y_flat.t())

        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)

        dxx = rx.t() + rx - 2.0 * xx
        dyy = ry.t() + ry - 2.0 * yy
        dxy = rx.t() + ry - 2.0 * zz

        XX, YY, XY = (
            torch.zeros(xx.shape).to(x.device),
            torch.zeros(yy.shape).to(x.device),
            torch.zeros(zz.shape).to(x.device),
        )

        if kernel == "multiscale":
            bandwidth_range = [0.2, 0.5, 0.9, 1.3]
            for a in bandwidth_range:
                XX += a**2 * (a**2 + dxx) ** -1
                YY += a**2 * (a**2 + dyy) ** -1
                XY += a**2 * (a**2 + dxy) ** -1

        elif kernel == "rbf":
            bandwidth_range = [10, 15, 20, 50]
            for a in bandwidth_range:
                XX += torch.exp(-0.5 * dxx / a)
                YY += torch.exp(-0.5 * dyy / a)
                XY += torch.exp(-0.5 * dxy / a)

        return torch.mean(XX / (N * S) ** 2 + YY / (N * S_prime) ** 2 - 2.0 * XY / (N**2 * S * S_prime))


class CLSAlignment(nn.Module):
    """Classification Alignment Loss."""

    def __init__(self):
        super().__init__()

    def forward(self, x_cls: Tensor, y_cls: Tensor) -> Tensor:
        """Performs a forward pass through the loss function.

        Args:
            x: (N, E), Token tensor.
            y: (N, E), Token tensor.

        Returns:
            Scalar loss value.
        """
        assert x_cls.shape[1] == y_cls.shape[1], "Input tensors must have the same embedding dimension."
        return F.mse_loss(x_cls, y_cls)


class TabularPredictor(nn.Module):
    '''Masked Tabular Reconstruction'''
    def __init__(self, tabular_embedding_dim: int, cat_lengths_tabular: List[int], num_con: int, num_unique_cat: int=None) -> None:
        super(TabularPredictor, self).__init__()
        self.num_con = num_con
        self.num_cat = len(cat_lengths_tabular)
        if num_unique_cat is None:
            self.num_unique_cat = sum(cat_lengths_tabular)
        else:
            self.num_unique_cat = num_unique_cat
        # continuous regessor
        self.con_regressor = nn.Linear(tabular_embedding_dim, 1, bias=True)
        # categorical classifier
        self.cat_classifier = nn.Linear(tabular_embedding_dim, self.num_unique_cat, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            m.weight.data.normal_(mean=0.0, std=.02)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)
        if isinstance(m, nn.Linear) and m.bias is not None:
            m.bias.data.zero_()
        
    def forward(self, x_unique: Tensor) -> Tensor:
        # remove clstokens
        # x_shared = x_shared[:, :-1,]
        # x_unique = x_unique[:, :-1,]

        # x = torch.cat((x_shared, x_unique), dim=1)
        # continuous regessor
        con_x = self.con_regressor(x_unique[:, :self.num_con])
        # categorical classifier
        cat_x = self.cat_classifier(x_unique[:, self.num_con:])
        return (con_x, cat_x)

class ReconstructionLoss(nn.Module):
    """
    Loss function for tabular data reconstruction.
    Loss function for multimodal contrastive learning based off of the CLIP paper.

    Embeddings are taken, L2 normalized and dot product between modalities is calculated to generate a cosine
    similarity between all combinations of subjects in a cross-modal fashion. Tempered by temperature.
    Loss is calculated attempting to match each subject's embeddings between the modalities i.e. the diagonal.
    """

    def __init__(self, num_con: int, cat_lengths_tabular: Tensor, d_token: int) -> None:
        super(ReconstructionLoss, self).__init__()

        self.num_con = num_con
        self.num_cat = len(cat_lengths_tabular)

        cat_offsets = torch.cat([torch.tensor([0]), cat_lengths_tabular.cumsum(0)[:-1]]).to(cat_lengths_tabular.device)
        self.register_buffer("cat_offsets", cat_offsets, persistent=False)
        self.softmax = nn.Softmax(dim=1)

        self.tabular_predictor = TabularPredictor(d_token, cat_lengths_tabular, num_con)

    def forward(self, x: Tensor, y: Tensor, mask: Tensor = None) -> Tensor:
        """
        Args:
            x: (N, S, E), Token tensor.
            y: (N, S, 1), Token tensor.
            mask: (N, S), Mask tensor.

        Returns:
            Scalar loss value.
        """
        out = self.tabular_predictor(x)
        N, _, num_unique_cat = out[1].shape
        # (N*N1, num_unique_cat)
        out_cat = out[1].reshape(N * self.num_cat, num_unique_cat)
        # (N, N2)
        out_con = out[0].squeeze(-1)
        target_con = y[:, : self.num_con]
        target_cat = (y[:, self.num_con :].long() + self.cat_offsets).reshape(N * self.num_cat)
        if mask:
            mask_con = mask[:, : self.num_con]
            mask_cat = mask[:, self.num_con :].reshape(N * self.num_cat)
        else:
            mask_con = torch.ones_like(target_con)
            mask_cat = torch.ones_like(target_cat)

        # cat loss
        prob_cat = self.softmax(out_cat)
        onehot_cat = F.one_hot(target_cat, num_classes=num_unique_cat)
        loss_cat = -onehot_cat * torch.log(prob_cat + 1e-8)
        loss_cat = loss_cat.sum(dim=1)
        loss_cat = (loss_cat * mask_cat).sum() / mask_cat.sum()

        # con loss
        loss_con = (out_con - target_con) ** 2
        loss_con = (loss_con * mask_con).sum() / mask_con.sum()

        loss = (loss_cat + loss_con) / 2

        return loss #, prob_cat, target_cat, mask_cat

class TripletSoftplusLoss(nn.Module):
    """Triplet Softplus Loss."""

    def __init__(self, temperature: float = 1.0):
        """Initializes class instance."""
        super().__init__()
        self.temperature = temperature

    def forward(self, tab_unique: Tensor, tab_shared: Tensor, ts_anchor: Tensor) -> Tensor:
        """Performs a forward pass through the loss function.

        Args:
            ts_anchor: (N, E), Anchor token.
            tab_shared: (N, E), Shared token.
            tab_unique: (N, E), Unique token.

        Returns:
            Scalar loss value.
        """
        # Normalize embeddings
        ts_anchor = F.normalize(ts_anchor, p=2, dim=1)
        tab_shared = F.normalize(tab_shared, p=2, dim=1)
        tab_unique = F.normalize(tab_unique, p=2, dim=1)

        return F.softplus((torch.mm(ts_anchor, tab_unique.t()) - torch.mm(ts_anchor, tab_shared.t()))/self.temperature).mean()

class CLIPLoss(nn.Module):
    """CLIP Loss."""

    def __init__(self, temperature: float = 1.0, margin: float = 0.0):
        """Initializes class instance.

        Args:
            temperature: Temperature scaling factor.
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, tab_unique: Tensor, ts_anchor: Tensor, labels=None) -> Tensor:
        """Performs a forward pass through the loss function.

        Args:
            tab_unique: (N, E), Unique token.
            ts_anchor: (N, E), Anchor

        Returns:
            Scalar loss value.
        """
        tab_unique = F.normalize(tab_unique, p=2, dim=1)
        ts_anchor = F.normalize(ts_anchor, p=2, dim=1)
        tab_unique = torch.mm(tab_unique, ts_anchor.t())
        tab_unique -= torch.eye(tab_unique.shape[0]).to(tab_unique.device) * self.margin
        tab_unique /= self.temperature
        tab_unique = torch.exp(tab_unique)
        x_1 = tab_unique.diag() / tab_unique.sum(dim=1)
        x_2 = tab_unique.diag() / tab_unique.sum(dim=0)
        return (-torch.log(x_1).mean() - torch.log(x_2).mean()) / 2
        # return -torch.log(x).mean()

class SupConCLIPLoss(nn.Module):
    """SupCLIP Loss."""

    def __init__(self, temperature: float = 1.0, margin: float = 0.0):
        """Initializes class instance.

        Args:
            temperature: Temperature scaling factor.
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, tab_unique: Tensor, ts_anchor: Tensor, labels: Tensor) -> Tensor:
        """Performs a forward pass through the loss function.

        Args:
            tab_unique: (N, E), Unique token.
            ts_anchor: (N, E), Anchor

        Returns:
            Scalar loss value.
        """
        batch_size = tab_unique.shape[0]
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(tab_unique.device)

        # logits_mask = torch.scatter(
        #     torch.ones_like(mask),
        #     1,
        #     torch.arange(batch_size).view(-1, 1).to(tab_unique.device),
        #     0
        # )
        # mask = mask * logits_mask
        
        # Remove self contrastive elements in diagonal
        # label_mask -= torch.eye(label_mask.shape[0]).to(label_mask.device)

        tab_unique = F.normalize(tab_unique, p=2, dim=1)
        ts_anchor = F.normalize(ts_anchor, p=2, dim=1)
        similarity = torch.mm(tab_unique, ts_anchor.t())
        similarity -= torch.eye(similarity.shape[0]).to(similarity.device) * self.margin
        similarity /= self.temperature
        exp_similarity = torch.exp(similarity)
        # Write in this form to avoid -inf in log
        x_1 = similarity - torch.log(exp_similarity.sum(dim=1, keepdim=True))
        x_2 = similarity - torch.log(exp_similarity.sum(dim=0, keepdim=True))
        x_1 = (x_1 * mask).sum(dim=1) / mask.sum(dim=1)
        x_2 = (x_2 * mask).sum(dim=0) / mask.sum(dim=0)
        return (-x_1.mean() - x_2.mean()) / 2

class LaaFLoss(nn.Module):
    """SupCLIP Loss."""

    def __init__(self, d_token: int, temperature: float = 1.0, margin: float = 0.0):
        """Initializes class instance.

        Args:
            temperature: Temperature scaling factor.
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.label_dictionary = nn.Embedding(3, d_token)

    def forward(self, tab_unique: Tensor, ts_anchor: Tensor, labels: Tensor = None) -> Tensor:
        """Performs a forward pass through the loss function.

        Args:
            tab_unique: (N, E), Unique token.
            ts_anchor: (N, E), Anchor

        Returns:
            Scalar loss value.
        """

        label_embedding = self.label_dictionary(labels)
        # Add label embedding to tab_unique
        tab_unique += label_embedding

        # Normalize embeddings and calculate similarity
        tab_unique = F.normalize(tab_unique, p=2, dim=1)
        ts_anchor = F.normalize(ts_anchor, p=2, dim=1)
        similarity = torch.mm(tab_unique, ts_anchor.t())
        similarity -= torch.eye(similarity.shape[0]).to(similarity.device) * self.margin
        similarity /= self.temperature
        similarity = torch.exp(similarity)
        x_1 = similarity.diag() / (similarity - torch.diag(similarity.diag())).sum(dim=1)
        x_2 = similarity.diag() / (similarity - torch.diag(similarity.diag())).sum(dim=0)

        return (-torch.log(x_1).mean() - torch.log(x_2).mean()) / 2

class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross-Entropy Loss with Decoupling."""

    def __init__(self, temperature: float = 0.1):
        """Initializes class instance.

        Args:
            temperature: Temperature scaling factor.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, tab_unique: Tensor, ts_anchor: Tensor, labels=None) -> Tensor:
        """Performs a forward pass through the loss function.

        Args:
            tab_unique: (N, E), Embeddings.
            ts_anchor: (N, E), Embeddings.

        Returns:
            Scalar loss value.
        """
        batch_size = tab_unique.size(0)

        # compute similarity between the embeddings of both views of the data
        z = torch.cat([tab_unique, ts_anchor], dim=0)
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # (N * 2, N * 2)

        # Create a mask for the positive samples, i.e. the corresponding samples in each view
        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives_mask = torch.cat([sim_ij, sim_ji], dim=0)

        # For the denominator, we need to include both the positive and negative samples, so we use the inverse of the
        # identity matrix, so that we only exclude the similarities between each embedding and itself
        pairwise_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=tab_unique.device)).float()

        numerator = torch.exp(positives_mask / self.temperature)
        denominator = pairwise_mask * torch.exp(similarity / self.temperature)

        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)

        return loss

class SupConNTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross-Entropy Loss with Decoupling."""

    def __init__(self, temperature: float = 0.1):
        """Initializes class instance.

        Args:
            temperature: Temperature scaling factor.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, tab_unique: Tensor, ts_anchor: Tensor, labels: Tensor) -> Tensor:
        """Performs a forward pass through the loss function.

        Args:
            tab_unique: (N, E), Embeddings.
            ts_anchor: (N, E), Embeddings.

        Returns:
            Scalar loss value.
        """
        batch_size = tab_unique.size(0)
        labels = labels.view(-1, 1)
        double_labels = torch.cat([labels, labels], dim=0) # (2N, 1)
        mask = torch.eq(double_labels, double_labels.t()).float().to(tab_unique.device) # (2N, 2N)

        # compute similarity between the embeddings of both views of the data
        z = torch.cat([tab_unique, ts_anchor], dim=0)
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) # (2N, 2N)

        # Create a mask for the positive samples, i.e. the corresponding samples in each view
        masked_similarity = similarity * mask
        sim_ij = masked_similarity[:batch_size, batch_size:]
        sim_ji = masked_similarity[batch_size:, :batch_size]
        positives_mask = torch.cat([sim_ij, sim_ji], dim=0) # (2N, N)

        # For the denominator, we need to include both the positive and negative samples, so we use the inverse of the
        # identity matrix, so that we only exclude the similarities between each embedding and itself
        pairwise_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=tab_unique.device)).float()

        denominator = pairwise_mask * torch.exp(similarity / self.temperature)
        logits = positives_mask / self.temperature - torch.log(torch.sum(denominator, dim=1, keepdim=True))
        grouped_logits = torch.sum(logits, dim=1) / torch.sum(mask, dim=1)

        loss = - torch.sum(grouped_logits) / (2 * batch_size)

        return loss
