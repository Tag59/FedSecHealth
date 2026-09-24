"""Reconstruction quality metrics (evaluation only: the attacker never sees them).

Tabular data is compared in standardised feature space (relative L2 error,
cosine similarity). Images are compared in [0, 1] pixel space with MSE, PSNR
and SSIM (Wang et al., 2004). For batches, reconstructions are first matched
to ground-truth samples with the Hungarian algorithm, since gradient inversion
recovers a set, not an ordered list.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


def _gaussian_window(size: int = 7, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Mean SSIM per image for batches (B, C, H, W) in [0, data_range]. Gaussian 7x7 window."""
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    ch = a.shape[1]
    w = _gaussian_window().to(a).repeat(ch, 1, 1, 1)

    def filt(t):
        return nn.functional.conv2d(t, w, groups=ch)

    mu_a, mu_b = filt(a), filt(b)
    var_a = filt(a * a) - mu_a**2
    var_b = filt(b * b) - mu_b**2
    cov = filt(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    )
    return s.flatten(1).mean(1)


def psnr(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = ((a - b) ** 2).flatten(1).mean(1)
    return 10 * torch.log10(data_range**2 / mse.clamp_min(1e-10))


def match_batch(x_rec: torch.Tensor, x_true: torch.Tensor) -> np.ndarray:
    """Permutation p such that x_rec[p[i]] best matches x_true[i] (min total MSE)."""
    cost = torch.cdist(x_true.flatten(1), x_rec.flatten(1)) ** 2
    _, cols = linear_sum_assignment(cost.cpu().numpy())
    return cols


def tabular_metrics(x_rec: torch.Tensor, x_true: torch.Tensor, threshold: float = 0.1) -> dict:
    """Per-sample relative L2 error and cosine similarity, averaged over the batch."""
    x_rec = x_rec[match_batch(x_rec, x_true)].flatten(1).float()
    x_true = x_true.flatten(1).float()
    rel = torch.linalg.norm(x_rec - x_true, dim=1) / (torch.linalg.norm(x_true, dim=1) + 1e-12)
    cos = nn.functional.cosine_similarity(x_rec, x_true, dim=1)
    return {
        "mse": ((x_rec - x_true) ** 2).mean().item(),
        "rel_error": rel.mean().item(),
        "cosine": cos.mean().item(),
        "success": bool((rel < threshold).float().mean().item() >= 0.5),
    }


def image_metrics(x_rec_px: torch.Tensor, x_true_px: torch.Tensor, threshold: float = 0.6) -> dict:
    """PSNR / SSIM in pixel space; success if mean SSIM >= ``threshold``."""
    x_rec_px = x_rec_px[match_batch(x_rec_px, x_true_px)].clamp(0, 1).float()
    x_true_px = x_true_px.float()
    s = ssim(x_rec_px, x_true_px)
    return {
        "mse": ((x_rec_px - x_true_px) ** 2).mean().item(),
        "psnr": psnr(x_rec_px, x_true_px).mean().item(),
        "ssim": s.mean().item(),
        "success": bool(s.mean().item() >= threshold),
    }
