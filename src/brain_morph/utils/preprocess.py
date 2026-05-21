# Copyright (C) 2026 Ivan Doronin <iadoronin@yandex.ru>
# This file is part of brain-morph, licensed under GNU GPL v3.0.
# See LICENSE file in the project root for full license text.

import torch
import torch.nn.functional as F
from torch import Tensor


def _log_kernel_3d(sigma: float, size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """3D Laplacian-of-Gaussian kernel, shape (size, size, size)."""
    r = size // 2
    coords = torch.arange(-r, r + 1, dtype=dtype, device=device)
    z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
    r2 = x**2 + y**2 + z**2
    s2 = sigma**2
    kernel = (r2 - 3 * s2) / (s2**2) * torch.exp(-r2 / (2 * s2))
    return kernel - kernel.mean()


def log_filter(vol: Tensor, sigma: float = 10.0, size: int = 5) -> Tensor:
    """Laplacian-of-Gaussian filter — analogue of MATLAB mlog.

    Args:
        vol:   (C, D, H, W) float tensor.
        sigma: LoG scale in voxels.
        size:  Kernel size (forced odd).

    Returns:
        (C, D, H, W) float tensor, values normalised to [0, 1].
    """
    if size % 2 == 0:
        size += 1
    vol = vol.float()
    kernel = _log_kernel_3d(sigma, size, vol.device, vol.dtype)
    kernel = kernel.unsqueeze(0).unsqueeze(0)   # (1, 1, s, s, s)
    pad = size // 2

    out = []
    for c in range(vol.shape[0]):
        ch = vol[c].unsqueeze(0).unsqueeze(0)   # (1, 1, D, H, W)
        filtered = F.conv3d(ch, kernel, padding=pad).squeeze().abs()
        vmax = filtered.amax()
        out.append(filtered / vmax if vmax > 0 else filtered)

    return torch.stack(out, dim=0)


def histogram_matching(moving: Tensor, fixed: Tensor, n_bins: int = 256) -> Tensor:
    """Match the histogram of *moving* to that of *fixed* via CDF mapping.

    Operates channel-by-channel. Pure PyTorch — no scipy required.

    Args:
        moving: (C, D, H, W) source tensor.
        fixed:  (C, D, H, W) reference tensor.
        n_bins: Number of histogram bins.

    Returns:
        (C, D, H, W) tensor with moving intensities remapped to match fixed.
    """
    moving = moving.float()
    fixed = fixed.float()
    out = torch.empty_like(moving)

    for c in range(moving.shape[0]):
        m = moving[c].flatten()
        f = fixed[c].flatten()
        m_min, m_max = m.min(), m.max()
        f_min, f_max = f.min(), f.max()

        m_cdf = torch.histc(m, bins=n_bins, min=m_min.item(), max=m_max.item()).cumsum(0)
        f_cdf = torch.histc(f, bins=n_bins, min=f_min.item(), max=f_max.item()).cumsum(0)
        m_cdf = m_cdf / m_cdf[-1]
        f_cdf = f_cdf / f_cdf[-1]

        # For each source bin find closest fixed bin by CDF value
        matched_bin = (m_cdf.unsqueeze(1) - f_cdf.unsqueeze(0)).abs().argmin(dim=1)
        f_bin_centers = f_min + (matched_bin.float() + 0.5) * (f_max - f_min) / n_bins

        m_bin_idx = ((m - m_min) / (m_max - m_min + 1e-12) * (n_bins - 1)).long().clamp(0, n_bins - 1)
        out[c] = f_bin_centers[m_bin_idx].reshape(moving[c].shape)

    return out


def preprocess(
    vol: Tensor,
    mode: str = "log",
    sigma: float = 10.0,
    size: int = 5,
    threshold: float = 0.01,
) -> Tensor:
    """Apply standard preprocessing to a volume before registration.

    Mirrors MATLAB msimanneal preprocessing logic.

    Args:
        vol:       (C, D, H, W) float tensor.
        mode:      ``'raw'`` | ``'log'`` | ``'log_binary'``
                   ``'log'``        — LoG only (analogue of 'young')
                   ``'log_binary'`` — LoG then threshold (analogue of 'adult')
                   ``'raw'``        — no change
        sigma:     LoG sigma in voxels.
        size:      LoG kernel size.
        threshold: Binarization threshold applied after LoG in 'log_binary'.

    Returns:
        Preprocessed (C, D, H, W) float tensor.
    """
    if mode == "raw":
        return vol.float()
    if mode == "log":
        return log_filter(vol, sigma=sigma, size=size)
    if mode == "log_binary":
        return (log_filter(vol, sigma=sigma, size=size) > threshold).float()
    raise ValueError(f"Unknown mode '{mode}'. Choose: 'raw', 'log', 'log_binary'.")
