# Copyright (C) 2026 Ivan Doronin <iadoronin@yandex.ru>
# Based on original MATLAB implementation by Sergey Shuvaev (CSHL, 2014-2021).
# This file is part of brain-morph, licensed under GNU GPL v3.0.
# See LICENSE file in the project root for full license text.

import torch
from .tension_metrics import TensionMetric, VolumeTension


def compute_tension_3d(
    grid_part: torch.Tensor,
    grid_full: torch.Tensor | None = None,
    mode: str = "abs",
    voxel_size: tuple[float, float, float] | None = None,
    cell_weights: torch.Tensor | None = None,
    metric: TensionMetric | None = None,
) -> torch.Tensor:
    """Compute deformation energy (tension) of a 3D mesh transformation.

    Mirrors MATLAB ``mregularize``.  Normalises both grids per-axis, then
    delegates to ``metric`` (default: ``VolumeTension(mode)``).

    Args:
        grid_part: Deformed (or sub-region) grid, shape ``(ny, nx, nz, 3)``.
        grid_full: Reference grid for per-axis normalisation; ``None`` uses
            ``grid_part`` (equivalent to MATLAB ``nargin == 1``).  Must have
            the same shape as ``grid_part``.
        mode: Passed to the default ``VolumeTension`` when ``metric=None``.
            Ignored when a custom ``metric`` is provided (use the metric's own
            ``mode`` constructor argument).
        voxel_size: Physical voxel size ``(sz_d, sz_h, sz_w)`` in µm for
            isotropic weighting.  ``None`` preserves per-axis normalisation.
        cell_weights: Per-cell weights ``(ny-1, nx-1, nz-1)`` ∈ ``[0, 1]``.
            Cells over missing tissue receive weight ``0`` (no penalty).
        metric: Any callable matching ``TensionMetric``.  ``None`` defaults to
            ``VolumeTension(mode=mode)``.

    Returns:
        Scalar tension tensor.
    """
    if grid_full is None:
        grid_full = grid_part

    grid_part = grid_part.to(dtype=torch.float32)
    grid_full = grid_full.to(dtype=torch.float32)

    # Per-axis normalisation to [0, 1] using the range of grid_full
    mins = grid_full.amin(dim=(0, 1, 2), keepdim=True)
    maxs = grid_full.amax(dim=(0, 1, 2), keepdim=True)
    spans = (maxs - mins).clamp_min(1e-12)

    part = (grid_part - mins) / spans
    base = (grid_full - mins) / spans

    if voxel_size is not None:
        scale = torch.tensor(voxel_size, dtype=part.dtype, device=part.device)
        part = part * scale
        base = base * scale

    if metric is None:
        metric = VolumeTension(mode=mode)

    return metric(part, base, cell_weights)
