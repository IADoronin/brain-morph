import sys
import os

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

from mesh_transformer_3d import MeshTransformer3D
from tension_metrics import TensionMetric


def _pearson_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    a_c = a_flat - a_flat.mean()
    b_c = b_flat - b_flat.mean()
    denom = (a_c.norm() * b_c.norm()).clamp_min(1e-12)
    return (a_c @ b_c) / denom


def _ncc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Normalised cross-correlation (mean-subtracted, L2-normalised)."""
    return _pearson_corr(a, b)


def _mse_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Negative MSE so higher = better (consistent sign with corr)."""
    return -(a.float() - b.float()).pow(2).mean()


_SIMILARITY = {
    "corr": _pearson_corr,
    "ncc": _ncc,
    "mse": _mse_sim,
}


def normalise_channel_weights(
    weights: "torch.Tensor | None",
    n_channels: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a normalised (C,) weight vector; uniform when weights=None."""
    if weights is None:
        return torch.full((n_channels,), 1.0 / n_channels, device=device)
    w = weights.to(device=device, dtype=torch.float32).flatten()
    if w.numel() != n_channels:
        raise ValueError(f"channel_weights length {w.numel()} != n_channels {n_channels}")
    return w / w.sum().clamp_min(1e-12)


def _sim_multichannel(
    a: torch.Tensor,
    b: torch.Tensor,
    sim_fn,
    channel_weights: "torch.Tensor | None",
) -> torch.Tensor:
    """Similarity for (D,H,W) or (C,D,H,W); channel-wise weighted average."""
    if a.dim() == 3:
        return sim_fn(a, b)
    C = a.shape[0]
    w = normalise_channel_weights(channel_weights, C, a.device)
    sims = torch.stack([sim_fn(a[c], b[c]) for c in range(C)])
    return (sims * w).sum()


def registration_cost(
    image_moving: torch.Tensor,
    image_fixed: torch.Tensor,
    transformer: MeshTransformer3D,
    grid_target: torch.Tensor,
    lam: float,
    mask: "torch.Tensor | None" = None,
    metric: "TensionMetric | None" = None,
    similarity: str = "corr",
    tension_mode: str = "abs",
    channel_weights: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """Registration objective: similarity − λ·tension.

    Higher is better (follows MATLAB msimanneal convention).

    Args:
        image_moving:    Moving image, shape ``(D, H, W)`` or ``(C, D, H, W)``.
        image_fixed:     Fixed image, same shape as ``image_moving``.
        transformer:     ``MeshTransformer3D`` with cached cell assignment.
        grid_target:     Candidate control-point grid, shape ``(ny, nx, nz, 3)``.
        lam:             Regularisation weight λ.
        mask:            Binary tissue mask ``(D, H, W)`` for cell-weighted tension.
        metric:          Custom tension metric; ``None`` → ``VolumeTension(tension_mode)``.
        similarity:      ``"corr"`` (Pearson), ``"ncc"``, or ``"mse"`` (negative MSE).
        tension_mode:    Passed to default ``VolumeTension`` when ``metric=None``.
        channel_weights: Per-channel importance, shape ``(C,)``.  ``None`` → uniform.
                         Ignored for single-channel ``(D, H, W)`` images.

    Returns:
        Scalar cost tensor (higher = better).
    """
    if similarity not in _SIMILARITY:
        raise ValueError(f"Unknown similarity '{similarity}'. Choose from {list(_SIMILARITY)}")

    multichannel = image_moving.dim() == 4
    warped = transformer.transform(image_moving, grid_target)
    if not multichannel:
        warped = warped.squeeze(0)  # (1, D, H, W) → (D, H, W)

    sim = _sim_multichannel(warped, image_fixed, _SIMILARITY[similarity], channel_weights)
    ten = transformer.tension(
        grid_target,
        mode=tension_mode,
        mask=mask,
        metric=metric,
    )
    return sim - lam * ten
