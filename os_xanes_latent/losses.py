"""Reconstruction and latent-space regularization losses."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SpectralL1Loss(nn.Module):
    """L1 reconstruction loss with peak- and gradient-sensitive terms.

    The default weights match those used in the study: peak weight 2 and gradient
    weight 1.
    """

    def __init__(self, peak_weight: float = 2.0, gradient_weight: float = 1.0) -> None:
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.peak_weight = peak_weight
        self.gradient_weight = gradient_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        device = y_true.device

        l1_loss = self.l1_loss(y_pred, y_true)

        pre = F.pad(y_true[:, :, :-1], (1, 0), value=float("-inf"))
        post = F.pad(y_true[:, :, 1:], (0, 1), value=float("-inf"))
        peaks = (y_true > pre) & (y_true > post)
        peak_mask = peaks.float()
        kernel = torch.ones((1, 1, 5), device=device)
        peak_mask = F.conv1d(peak_mask, kernel, padding=2).clamp(max=1)
        peak_loss = self.l1_loss(y_pred * peak_mask, y_true * peak_mask)

        y_true_grad = y_true[:, :, 1:] - y_true[:, :, :-1]
        y_pred_grad = y_pred[:, :, 1:] - y_pred[:, :, :-1]
        gradient_loss = self.l1_loss(y_pred_grad, y_true_grad)

        return l1_loss + self.peak_weight * peak_loss + self.gradient_weight * gradient_loss


def compute_mmd(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Maximum Mean Discrepancy using the Gaussian RBF kernel used in the study."""

    def gaussian_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        squared_distance = torch.cdist(a, b).pow(2)
        return torch.exp(-squared_distance / (2 * sigma**2))

    xx = gaussian_kernel(x, x)
    yy = gaussian_kernel(y, y)
    xy = gaussian_kernel(x, y)
    return xx.mean() + yy.mean() - 2 * xy.mean()


def mmd_to_standard_normal(latent: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Compute MMD between latent codes and an equally sized N(0, I) sample."""
    prior = torch.randn_like(latent)
    return compute_mmd(latent, prior, sigma=sigma)
