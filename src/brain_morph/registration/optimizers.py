from __future__ import annotations

import math
import sys
import os
from abc import ABC, abstractmethod

import torch

_utils_dir = os.path.join(os.path.dirname(__file__), "..", "utils")
_reg_dir   = os.path.dirname(__file__)
for _p in (_utils_dir, _reg_dir):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cost import registration_cost, normalise_channel_weights
from mesh_transformer_3d import MeshTransformer3D
from tension_metrics import TensionMetric


class MeshOptimizer(ABC):
    """Abstract base for all mesh registration optimizers."""

    @abstractmethod
    def optimize(
        self,
        image_moving: torch.Tensor,
        image_fixed: torch.Tensor,
        transformer: MeshTransformer3D,
        grid_start: torch.Tensor,
        n_steps: int,
        lam: float = 1e-3,
        mask: torch.Tensor | None = None,
        metric: TensionMetric | None = None,
        channel_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run optimisation and return the best grid found.

        Args:
            image_moving: Moving image ``(D, H, W)``.
            image_fixed: Fixed image ``(D, H, W)``.
            transformer: Prebuilt ``MeshTransformer3D``.
            grid_start: Initial control-point grid ``(ny, nx, nz, 3)``.
            n_steps: Total number of optimisation steps.
            lam: Regularisation weight.
            mask: Binary tissue mask ``(D, H, W)``; ``None`` → no masking.
            metric: Custom tension metric; ``None`` → ``VolumeTension``.

        Returns:
            Optimised grid ``(ny, nx, nz, 3)``.
        """


# ---------------------------------------------------------------------------
# Simulated Annealing
# ---------------------------------------------------------------------------

class SAOptimizer(MeshOptimizer):
    """Simulated annealing over the control-point grid.

    Mirrors the core loop of MATLAB ``msimanneal``:
    - attention-weighted node selection (recalculated every ``attention_freq`` steps)
    - Metropolis acceptance criterion
    - adaptive step size (coeff grows on accept, shrinks on reject)

    Works with *any* metric including non-differentiable ones.

    Args:
        temp_start: Initial temperature.
        temp_end: Final temperature (geometric cooling).
        coeff_start: Initial step amplitude as a fraction of mean cell size.
        coeff_drop: Multiplier on reject (``coeff *= coeff_drop``); inverse on accept.
        attention_freq: Steps between attention-map refreshes.  ``0`` → uniform.
        similarity: Similarity measure passed to ``registration_cost``.
    """

    def __init__(
        self,
        temp_start: float = 1e-3,
        temp_end: float = 1e-3 / 30,
        coeff_start: float = 0.2,
        coeff_drop: float = 0.9966,
        attention_freq: int = 100,
        similarity: str = "corr",
        callback=None,
        callback_freq: int = 50,
    ):
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.coeff_start = coeff_start
        self.coeff_drop = coeff_drop
        self.attention_freq = attention_freq
        self.similarity = similarity
        self.callback = callback
        self.callback_freq = callback_freq

    def _cell_size(self, grid: torch.Tensor) -> torch.Tensor:
        """Mean spacing between adjacent nodes (single scalar)."""
        ny, nx, nz = grid.shape[:3]
        sp = []
        if ny > 1:
            sp.append((grid[1:] - grid[:-1]).norm(dim=-1).mean())
        if nx > 1:
            sp.append((grid[:, 1:] - grid[:, :-1]).norm(dim=-1).mean())
        if nz > 1:
            sp.append((grid[:, :, 1:] - grid[:, :, :-1]).norm(dim=-1).mean())
        return torch.stack(sp).mean() if sp else grid.new_ones(1)

    @torch.no_grad()
    def _build_attention_probs(
        self,
        warped: torch.Tensor,
        image_fixed: torch.Tensor,
        transformer: MeshTransformer3D,
        channel_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Flat probability tensor over grid nodes, shape (ny*nx*nz,)."""
        diff = (warped.float() - image_fixed.float()).abs()  # (D,H,W) or (C,D,H,W)
        if diff.dim() == 4:  # multi-channel: weighted mean over channels
            w = normalise_channel_weights(channel_weights, diff.shape[0], diff.device)
            diff = (diff * w.reshape(-1, 1, 1, 1)).sum(0)  # (D, H, W)
        diff_flat = diff.flatten()  # (D*H*W,)
        index_map = transformer.cell_index_map.to(device=diff_flat.device)

        ny, nx, nz = [s - 1 for s in transformer.grid_init.shape[:3]]
        n_cells = ny * nx * nz

        cell_score = torch.zeros(n_cells, device=diff_flat.device)
        cell_score.scatter_add_(0, index_map, diff_flat)

        # Smooth: each of the 8 corners of a cell gets the cell score
        score_grid = cell_score.reshape(ny, nx, nz)
        # Pad with zeros so corner contributions sum up at each node
        node_score = torch.zeros(ny + 1, nx + 1, nz + 1, device=score_grid.device)
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    node_score[di:di+ny, dj:dj+nx, dk:dk+nz] += score_grid

        probs = node_score.flatten().clamp_min(0)
        total = probs.sum()
        if total < 1e-12:
            probs = probs + 1.0
        return probs / probs.sum()

    def optimize(
        self,
        image_moving: torch.Tensor,
        image_fixed: torch.Tensor,
        transformer: MeshTransformer3D,
        grid_start: torch.Tensor,
        n_steps: int,
        lam: float = 1e-3,
        mask: torch.Tensor | None = None,
        metric: TensionMetric | None = None,
        channel_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        multichannel = image_moving.dim() == 4
        grid = grid_start.clone().detach()
        cell_sz = self._cell_size(grid).item()
        coeff = self.coeff_start

        with torch.no_grad():
            warped = transformer.transform(image_moving, grid)
            if not multichannel:
                warped = warped.squeeze(0)
            cost = registration_cost(
                image_moving, image_fixed, transformer, grid, lam, mask, metric,
                similarity=self.similarity, channel_weights=channel_weights,
            ).item()
        cost_best = cost
        grid_best = grid.clone()

        probs: torch.Tensor | None = None

        for step in range(n_steps):
            # Geometric cooling
            frac = step / max(n_steps - 1, 1)
            temp = self.temp_start * (self.temp_end / self.temp_start) ** frac

            # Refresh attention map
            if self.attention_freq > 0 and step % self.attention_freq == 0:
                probs = self._build_attention_probs(
                    warped, image_fixed, transformer, channel_weights
                )

            # Sample a node
            ny, nx, nz = grid.shape[:3]
            n_nodes = ny * nx * nz
            if probs is not None:
                node_flat = torch.multinomial(probs, 1).item()
            else:
                node_flat = torch.randint(n_nodes, (1,)).item()

            ni = node_flat // (nx * nz)
            nj = (node_flat % (nx * nz)) // nz
            nk = node_flat % nz

            # Propose move
            delta = torch.randn(3) * coeff * cell_sz
            grid_new = grid.clone()
            grid_new[ni, nj, nk] = grid_new[ni, nj, nk] + delta

            # Evaluate new cost
            with torch.no_grad():
                warped_new = transformer.transform(image_moving, grid_new)
                if not multichannel:
                    warped_new = warped_new.squeeze(0)
                cost_new = registration_cost(
                    image_moving, image_fixed, transformer, grid_new, lam, mask, metric,
                    similarity=self.similarity, channel_weights=channel_weights,
                ).item()

            # Metropolis
            delta_cost = cost_new - cost
            accept = delta_cost > 0 or (
                math.exp(delta_cost / max(temp, 1e-30)) > torch.rand(1).item()
            )

            if accept:
                grid = grid_new
                warped = warped_new
                cost = cost_new
                coeff /= self.coeff_drop  # step grows slightly
                if cost > cost_best:
                    cost_best = cost
                    grid_best = grid.clone()
            else:
                coeff *= self.coeff_drop  # step shrinks

            if self.callback is not None and step % self.callback_freq == 0:
                self.callback(step, cost, warped)

        return grid_best


# ---------------------------------------------------------------------------
# Gradient descent
# ---------------------------------------------------------------------------

class GradientOptimizer(MeshOptimizer):
    """Gradient-based optimizer wrapping any ``torch.optim`` algorithm.

    Requires ``metric`` to support autograd (use ``mode="squared"``).  The
    similarity term uses ``tension_mode="squared"`` by default to keep
    gradients smooth.

    Args:
        optimizer_cls: ``torch.optim`` class (``Adam``, ``SGD``, ``LBFGS``, …).
        lr: Learning rate.
        optimizer_kwargs: Extra kwargs forwarded to the optimizer constructor.
        similarity: Similarity measure.
        tension_mode: Tension mode; should be ``"squared"`` for autograd.
    """

    def __init__(
        self,
        optimizer_cls: type = torch.optim.Adam,
        lr: float = 1e-3,
        optimizer_kwargs: dict | None = None,
        similarity: str = "corr",
        tension_mode: str = "squared",
    ):
        self.optimizer_cls = optimizer_cls
        self.lr = lr
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.similarity = similarity
        self.tension_mode = tension_mode

    def optimize(
        self,
        image_moving: torch.Tensor,
        image_fixed: torch.Tensor,
        transformer: MeshTransformer3D,
        grid_start: torch.Tensor,
        n_steps: int,
        lam: float = 1e-3,
        mask: torch.Tensor | None = None,
        metric: TensionMetric | None = None,
        channel_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        grid = grid_start.clone().float().requires_grad_(True)
        opt = self.optimizer_cls([grid], lr=self.lr, **self.optimizer_kwargs)

        for _ in range(n_steps):
            opt.zero_grad()
            cost = registration_cost(
                image_moving.float(),
                image_fixed.float(),
                transformer,
                grid,
                lam,
                mask,
                metric,
                similarity=self.similarity,
                tension_mode=self.tension_mode,
                channel_weights=channel_weights,
            )
            (-cost).backward()
            opt.step()

        return grid.detach()


# ---------------------------------------------------------------------------
# Hybrid: SA exploration → gradient refinement
# ---------------------------------------------------------------------------

class HybridOptimizer(MeshOptimizer):
    """Two-phase optimizer: SA for global search, gradient for local refinement.

    Args:
        sa_optimizer: Configured ``SAOptimizer`` instance.
        gd_optimizer: Configured ``GradientOptimizer`` instance.
        sa_fraction: Fraction of total steps given to SA (remainder to GD).
    """

    def __init__(
        self,
        sa_optimizer: SAOptimizer | None = None,
        gd_optimizer: GradientOptimizer | None = None,
        sa_fraction: float = 0.7,
    ):
        self.sa_optimizer = sa_optimizer or SAOptimizer()
        self.gd_optimizer = gd_optimizer or GradientOptimizer()
        self.sa_fraction = sa_fraction

    def optimize(
        self,
        image_moving: torch.Tensor,
        image_fixed: torch.Tensor,
        transformer: MeshTransformer3D,
        grid_start: torch.Tensor,
        n_steps: int,
        lam: float = 1e-3,
        mask: torch.Tensor | None = None,
        metric: TensionMetric | None = None,
        channel_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n_sa = max(1, int(n_steps * self.sa_fraction))
        n_gd = max(1, n_steps - n_sa)

        grid_sa = self.sa_optimizer.optimize(
            image_moving, image_fixed, transformer, grid_start,
            n_sa, lam, mask, metric, channel_weights,
        )
        return self.gd_optimizer.optimize(
            image_moving, image_fixed, transformer, grid_sa,
            n_gd, lam, mask, metric, channel_weights,
        )
