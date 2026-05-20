# Copyright (C) 2026 Ivan Doronin <iadoronin@yandex.ru>
# This file is part of brain-morph, licensed under GNU GPL v3.0.
# See LICENSE file in the project root for full license text.

import torch
from typing import Protocol, runtime_checkable


@runtime_checkable
class TensionMetric(Protocol):
    """Callable protocol for deformation tension metrics.

    Receives already-normalised grids (normalisation is handled by
    ``compute_tension_3d``).  Any callable with this signature qualifies —
    no explicit subclassing needed.
    """

    def __call__(
        self,
        grid_target: torch.Tensor,                # (ny, nx, nz, 3) normalised
        grid_ref: torch.Tensor,                   # (ny, nx, nz, 3) normalised
        cell_weights: torch.Tensor | None = None, # (ny-1, nx-1, nz-1) | None
    ) -> torch.Tensor: ...                        # scalar


class VolumeTension:
    """Det-based tension via 5-tetrahedra hexahedral decomposition.

    Matches the MATLAB ``mregularize`` algorithm exactly.  Each grid cell is
    split into 5 tetrahedra; the metric sums |V_deformed − V_ref| (mode="abs")
    or (V_deformed − V_ref)² (mode="squared") over all cells and tetrahedra.

    Args:
        mode: ``"abs"`` — MATLAB-compatible; ``"squared"`` — smooth gradient
            for autograd-based optimisers.
    """

    def __init__(self, mode: str = "abs"):
        self.mode = mode

    def __call__(
        self,
        grid_target: torch.Tensor,
        grid_ref: torch.Tensor,
        cell_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 8 corner slices — each (ny-1, nx-1, nz-1, 3)
        N000 = grid_target[:-1, :-1, :-1];  O000 = grid_ref[:-1, :-1, :-1]
        N001 = grid_target[:-1, :-1,  1:];  O001 = grid_ref[:-1, :-1,  1:]
        N010 = grid_target[:-1,  1:, :-1];  O010 = grid_ref[:-1,  1:, :-1]
        N011 = grid_target[:-1,  1:,  1:];  O011 = grid_ref[:-1,  1:,  1:]
        N100 = grid_target[ 1:, :-1, :-1];  O100 = grid_ref[ 1:, :-1, :-1]
        N101 = grid_target[ 1:, :-1,  1:];  O101 = grid_ref[ 1:, :-1,  1:]
        N110 = grid_target[ 1:,  1:, :-1];  O110 = grid_ref[ 1:,  1:, :-1]
        N111 = grid_target[ 1:,  1:,  1:];  O111 = grid_ref[ 1:,  1:,  1:]

        def _vol(p0, p1, p2, p3):
            mat = torch.stack([p1 - p0, p2 - p0, p3 - p0], dim=-1)
            return torch.linalg.det(mat).abs()

        tet_pairs = [
            ((N001, N000, N010, N100), (O001, O000, O010, O100)),
            ((N001, N011, N010, N111), (O001, O011, O010, O111)),
            ((N010, N110, N111, N100), (O010, O110, O111, O100)),
            ((N001, N101, N111, N100), (O001, O101, O111, O100)),
            ((N001, N100, N111, N010), (O001, O100, O111, O010)),
        ]

        if self.mode == "abs":
            cell_t = sum((_vol(*n) - _vol(*o)).abs() for n, o in tet_pairs)
        else:  # "squared"
            cell_t = sum((_vol(*n) - _vol(*o)).pow(2) for n, o in tet_pairs)

        if cell_weights is not None:
            cell_t = cell_t * cell_weights.to(dtype=cell_t.dtype, device=cell_t.device)

        return cell_t.sum() / 6.0


class BendingTension:
    """Bending energy via second finite differences of the displacement field.

    Penalises curvature of u = grid_target − grid_ref along each spatial axis.
    Constant and linear displacements contribute exactly zero — affine
    transformations (translation, rotation, uniform scaling) are unconstrained.

    Computationally cheaper than ``VolumeTension`` (~27 ops/node vs ~135).

    Args:
        mode: ``"abs"`` — sum of |d²u|; ``"squared"`` — sum of (d²u)², smooth
            gradient for autograd.

    Note:
        ``cell_weights`` is accepted for API compatibility but is not applied —
        the metric operates on grid nodes, not cells.
    """

    def __init__(self, mode: str = "abs"):
        self.mode = mode

    def __call__(
        self,
        grid_target: torch.Tensor,
        grid_ref: torch.Tensor,
        cell_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        u = grid_target - grid_ref  # displacement field (ny, nx, nz, 3)

        # Central second differences along each spatial axis
        d2_a = u[2:, :,  :,  :] - 2.0 * u[1:-1, :,   :,  :] + u[:-2, :,   :,  :]
        d2_b = u[:,  2:, :,  :] - 2.0 * u[:,    1:-1, :,  :] + u[:,   :-2, :,  :]
        d2_c = u[:,  :,  2:, :] - 2.0 * u[:,    :,  1:-1, :] + u[:,   :,  :-2, :]

        if self.mode == "abs":
            return d2_a.abs().sum() + d2_b.abs().sum() + d2_c.abs().sum()
        else:  # "squared"
            return d2_a.pow(2).sum() + d2_b.pow(2).sum() + d2_c.pow(2).sum()
