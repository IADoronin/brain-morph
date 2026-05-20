from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

_utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils"))
_reg_dir   = os.path.abspath(os.path.dirname(__file__))
for _p in (_utils_dir, _reg_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mesh_transformer_3d import MeshTransformer3D
from tension_metrics import TensionMetric
from optimizers import MeshOptimizer


def _make_regular_grid(ny: int, nx: int, nz: int) -> torch.Tensor:
    """Regular grid in [-1, 1]^3, shape (ny, nx, nz, 3)."""
    return torch.stack(
        torch.meshgrid(
            *[torch.linspace(-1, 1, s) for s in (ny, nx, nz)],
            indexing="ij",
        ),
        dim=-1,
    )


def _downsample_image(image: torch.Tensor, scale: int) -> torch.Tensor:
    """Downsample (D,H,W) or (C,D,H,W) by integer factor with box-average anti-aliasing."""
    if scale <= 1:
        return image
    if image.dim() == 3:
        h = image.unsqueeze(0).unsqueeze(0).float()   # (1, 1, D, H, W)
        h = F.avg_pool3d(h, kernel_size=scale, stride=scale, padding=0)
        return h.squeeze(0).squeeze(0).to(image.dtype)
    else:  # (C, D, H, W)
        h = image.unsqueeze(0).float()                 # (1, C, D, H, W)
        h = F.avg_pool3d(h, kernel_size=scale, stride=scale, padding=0)
        return h.squeeze(0).to(image.dtype)


def _downsample_mask(mask: torch.Tensor, scale: int) -> torch.Tensor:
    """Downsample (D, H, W) bool mask by integer factor."""
    if scale <= 1:
        return mask
    h = mask.float().unsqueeze(0).unsqueeze(0)
    h = F.avg_pool3d(h, kernel_size=scale, stride=scale, padding=0)
    return (h.squeeze(0).squeeze(0) > 0.5).to(mask.dtype)


def interpolate_grid(
    grid: torch.Tensor,
    target_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Trilinearly interpolate a control-point grid to a new node count.

    Args:
        grid: Source grid ``(ny, nx, nz, 3)``.
        target_shape: Desired ``(ny_new, nx_new, nz_new)``.

    Returns:
        Interpolated grid ``(ny_new, nx_new, nz_new, 3)``.
    """
    ny, nx, nz = target_shape
    # (1, 3, ny_src, nx_src, nz_src)
    g5 = grid.permute(3, 0, 1, 2).unsqueeze(0).float()
    out = F.interpolate(g5, size=(ny, nx, nz), mode="trilinear", align_corners=True)
    return out.squeeze(0).permute(1, 2, 3, 0)  # (ny, nx, nz, 3)


@dataclass
class Stage:
    """One stage of a multi-resolution registration pipeline.

    Args:
        grid_shape: Control-point grid dimensions ``(ny, nx, nz)``.
        optimizer:  Any ``MeshOptimizer`` instance (SA, GD, Hybrid, …).
        n_steps:    Number of optimisation steps for this stage.
        lam:        Regularisation weight λ.
        metric:     Tension metric; ``None`` → ``VolumeTension`` (default).
    """

    grid_shape:   tuple[int, int, int]
    optimizer:    MeshOptimizer
    n_steps:      int
    lam:          float = 1e-3
    metric:       TensionMetric | None = None
    image_scale:  int = 1  # integer downsample factor; 1 = full resolution


class RegistrationPipeline:
    """Coarse-to-fine registration: each Stage refines the previous result.

    Grid from stage *k* is trilinearly interpolated to the node count of
    stage *k+1*, then handed to that stage's optimizer as the starting point.
    Different optimizers can be assigned to different stages — e.g. SA for
    coarse exploration and gradient descent for fine-tuning.

    Example::

        pipeline = RegistrationPipeline([
            Stage((3, 3, 3), SAOptimizer(n_steps=500), n_steps=500, lam=1e-2),
            Stage((5, 5, 5), SAOptimizer(n_steps=1000), n_steps=1000, lam=1e-3),
            Stage((7, 7, 7), GradientOptimizer(lr=1e-3), n_steps=200, lam=1e-4),
        ])
        grid_result = pipeline.run(im_moving, im_fixed, mask=brain_mask)

    Args:
        stages: Ordered list of ``Stage`` objects, coarse → fine.
    """

    def __init__(self, stages: list[Stage]):
        if not stages:
            raise ValueError("stages must be non-empty")
        self.stages = stages

    def run(
        self,
        image_moving:    torch.Tensor,
        image_fixed:     torch.Tensor,
        mask:            torch.Tensor | None = None,
        channel_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run all stages and return the final optimised grid.

        Args:
            image_moving:    Moving image ``(D, H, W)`` or ``(C, D, H, W)``.
            image_fixed:     Fixed image, same shape as ``image_moving``.
            mask:            Binary tissue mask ``(D, H, W)``; forwarded to every stage.
            channel_weights: Per-channel importance weights ``(C,)``.
                             ``None`` → uniform.  Ignored for single-channel images.

        Returns:
            Optimised control-point grid ``(ny, nx, nz, 3)`` of the last stage.
        """
        grid: torch.Tensor | None = None

        for stage in self.stages:
            # Downsample images and mask for this stage if requested
            im_mov = _downsample_image(image_moving, stage.image_scale)
            im_fix = _downsample_image(image_fixed,  stage.image_scale)
            mask_s = _downsample_mask(mask, stage.image_scale) if mask is not None else None
            image_shape = tuple(im_mov.shape[-3:])  # (D, H, W) for both 3-D and 4-D inputs

            grid_init = _make_regular_grid(*stage.grid_shape)

            if grid is None:
                grid_start = grid_init
            else:
                grid_start = interpolate_grid(grid, stage.grid_shape)

            transformer = MeshTransformer3D(grid_init, image_shape)
            grid = stage.optimizer.optimize(
                im_mov,
                im_fix,
                transformer,
                grid_start,
                n_steps=stage.n_steps,
                lam=stage.lam,
                mask=mask_s,
                metric=stage.metric,
                channel_weights=channel_weights,
            )

        return grid  # type: ignore[return-value]  # stages is non-empty
