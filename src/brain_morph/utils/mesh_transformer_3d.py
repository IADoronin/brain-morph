# Copyright (C) 2026 Ivan Doronin <iadoronin@yandex.ru>
# Based on original MATLAB implementation by Sergey Shuvaev (CSHL, 2014-2021).
# This file is part of brain-morph, licensed under GNU GPL v3.0.
# See LICENSE file in the project root for full license text.

import torch
import torch.nn.functional as F

from . import mask_3d
from .mesh_transform_3d import get_basis_3d, get_bilinear_transform_3d, iter_grid_cells_3d
from .compute_tension_3d import compute_tension_3d


class MeshTransformer3D:
    """
    Bilinear mesh transformer for 3D images with cached cell assignment.

    Precomputes which voxel belongs to which grid cell once at construction time,
    so that transform() can skip the expensive per-voxel classification on every call.
    Use this class when you need to apply multiple transformations that share the
    same grid_init and image shape (e.g., iterative registration, batch processing).
    """

    def __init__(
        self,
        grid_init: torch.Tensor,
        image_shape: tuple,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            grid_init: Initial control point grid of shape (ny, nx, nz, 3).
            image_shape: Spatial dimensions (D, H, W) of the images to transform.
            device: Device for the precomputed mesh (default: CPU).
            dtype: Floating-point dtype for mesh coordinates.
        """
        if grid_init.shape[-1] != 3:
            raise ValueError("grid_init must have 3D control points at last dimension")
        if len(image_shape) != 3:
            raise ValueError("image_shape must be a 3-tuple (D, H, W)")

        self.grid_init = grid_init
        self.image_shape = tuple(image_shape)

        _device = device if device is not None else torch.device("cpu")

        mesh = torch.stack(
            torch.meshgrid(
                *[torch.linspace(-1, 1, steps=s, device=_device, dtype=dtype)
                  for s in image_shape],
                indexing="ij",
            ),
            dim=-1,
        )  # (D, H, W, 3)
        self._mesh_f = mesh.flatten(0, -2)  # (D*H*W, 3)
        self._cell_index_map = self._build_cell_index_map()

    def _build_cell_index_map(self) -> torch.Tensor:
        n_voxels = self._mesh_f.shape[0]
        index_map = torch.full(
            (n_voxels,), -1, dtype=torch.long, device=self._mesh_f.device
        )
        cell_centers: list[torch.Tensor] = []
        for cell_idx, cell_init in enumerate(iter_grid_cells_3d(self.grid_init)):
            cell_init_d = cell_init.to(dtype=self._mesh_f.dtype, device=self._mesh_f.device)
            mask = mask_3d.innerpoints(cell_init_d, self._mesh_f)
            index_map[mask] = cell_idx
            cell_centers.append(cell_init_d.flatten(0, -2).mean(0))

        # Fallback for voxels on cell boundaries that innerpoints missed due to
        # floating-point precision (scalars slightly > 0 instead of = 0).
        # Assign each such voxel to the cell whose centroid is nearest.
        unassigned = index_map == -1
        if unassigned.any():
            centers = torch.stack(cell_centers)  # (n_cells, 3)
            dists = torch.cdist(self._mesh_f[unassigned], centers)  # (n_unassigned, n_cells)
            index_map[unassigned] = dists.argmin(dim=1)

        return index_map

    @property
    def cell_index_map(self) -> torch.Tensor:
        """Shape (D*H*W,), dtype long: cell index per voxel (-1 = unassigned)."""
        return self._cell_index_map

    def transform(
        self,
        image: torch.Tensor,
        grid_target: torch.Tensor,
        method: str = "bilinear",
    ) -> torch.Tensor:
        """
        Warp an image using the precomputed cell assignment.

        Args:
            image: Tensor of shape (D, H, W) or (C, D, H, W).
            grid_target: Target control point grid with the same shape as grid_init.
            method: Interpolation mode for grid_sample ('bilinear' or 'nearest').

        Returns:
            Warped tensor of shape (C, D, H, W) — channel dim is added for 3D input.
        """
        if grid_target.shape != self.grid_init.shape:
            raise ValueError("grid_target must have the same shape as grid_init")
        if image.dim() == 3:
            image = image.unsqueeze(0)

        spatial = tuple(image.shape[1:])
        if spatial != self.image_shape:
            raise ValueError(
                f"Image spatial shape {spatial} does not match "
                f"precomputed shape {self.image_shape}"
            )

        mesh_f = self._mesh_f.to(device=image.device, dtype=image.dtype)
        index_map = self._cell_index_map.to(device=image.device)

        mesh_transformed_f = torch.full_like(mesh_f, -1.0)

        for cell_idx, (cell_init, cell_target) in enumerate(
            zip(iter_grid_cells_3d(self.grid_init), iter_grid_cells_3d(grid_target))
        ):
            mask = index_map == cell_idx
            if not mask.any():
                continue
            cell_init_d = cell_init.to(device=image.device, dtype=image.dtype)
            cell_target_d = cell_target.to(device=image.device, dtype=image.dtype)
            transform_mat = get_bilinear_transform_3d(cell_init_d, cell_target_d)
            mesh_transformed_f[mask] = get_basis_3d(mesh_f[mask]) @ transform_mat

        mesh_transformed = mesh_transformed_f.reshape(*self.image_shape, 3).flip([-1])

        return F.grid_sample(
            image.unsqueeze(0),
            mesh_transformed.unsqueeze(0),
            mode=method,
            align_corners=True,
            padding_mode="zeros",
        ).squeeze(0)

    def _compute_cell_weights(self, mask: torch.Tensor) -> torch.Tensor:
        """Fraction of valid voxels per cell derived from a binary mask.

        Args:
            mask: Boolean (or float) tensor of shape ``(D, H, W)``.  A value of
                ``True`` / ``1`` marks tissue; ``False`` / ``0`` marks empty space.

        Returns:
            Float tensor of shape ``(ny-1, nx-1, nz-1)`` with values in
            ``[0, 1]``.  A cell fully covered by tissue gets weight ``1.0``; a
            cell over empty space gets ``0.0``.
        """
        mask_flat = mask.flatten().to(dtype=torch.float32,
                                       device=self._cell_index_map.device)
        ny, nx, nz = [s - 1 for s in self.grid_init.shape[:3]]
        n_cells = ny * nx * nz

        cell_sums = torch.zeros(n_cells, device=mask_flat.device)
        cell_sums.scatter_add_(0, self._cell_index_map, mask_flat)
        cell_counts = torch.bincount(self._cell_index_map, minlength=n_cells).float()
        return (cell_sums / cell_counts.clamp_min(1)).reshape(ny, nx, nz)

    def tension(
        self,
        grid_target: torch.Tensor,
        grid_ref: torch.Tensor | None = None,
        mode: str = "abs",
        voxel_size: tuple[float, ...] | None = None,
        mask: torch.Tensor | None = None,
        metric=None,
    ) -> torch.Tensor:
        """Deformation energy between ``grid_target`` and a reference grid.

        Args:
            grid_target: Deformed control-point grid, shape ``(ny, nx, nz, 3)``.
            grid_ref: Reference grid for energy computation.  ``None`` uses
                ``self.grid_init`` (typical case).  Pass the previous iteration's
                grid for step-wise regularisation in sequential registrations.
            mode: ``"abs"`` (MATLAB-compatible) or ``"squared"`` (smooth
                gradient for autograd-based optimisers).
            voxel_size: Physical voxel size in µm for isotropic weighting.
            mask: Binary mask ``(D, H, W)`` of valid tissue.  Cells over empty
                regions (e.g. detached olfactory bulb) receive weight ``0`` and
                do not contribute to the penalty.

        Returns:
            Scalar tension tensor.
        """
        ref = grid_ref if grid_ref is not None else self.grid_init
        weights = self._compute_cell_weights(mask) if mask is not None else None
        return compute_tension_3d(
            grid_target, ref,
            mode=mode,
            voxel_size=voxel_size,
            cell_weights=weights,
            metric=metric,
        )

    def transform_chunked(
        self,
        image: torch.Tensor,
        grid: torch.Tensor,
        chunk_size: int = 50,
    ) -> torch.Tensor:
        """Apply deformation to a full-resolution image in memory-efficient chunks.

        Useful when the full sampling grid ``(D, H, W, 3)`` does not fit in RAM.
        The coarse registration grid is upsampled to the full image resolution
        once, then ``F.grid_sample`` is applied slice-by-slice along D.

        Args:
            image:      ``(C, D, H, W)`` full-resolution image.
            grid:       ``(ny, nx, nz, 3)`` deformation grid from registration
                        (may be at a coarser resolution than *image*).
            chunk_size: Number of D-slices processed per iteration.

        Returns:
            ``(C, D, H, W)`` warped image, same dtype as *image*.
        """
        import gc
        import torch.nn.functional as F

        C, D, H, W = image.shape

        # Upsample coarse grid to full image resolution: (ny,nx,nz,3) → (D,H,W,3)
        g = grid.permute(3, 0, 1, 2).unsqueeze(0).float()   # (1, 3, ny, nx, nz)
        g_full = F.interpolate(g, size=(D, H, W), mode="trilinear", align_corners=True)
        g_full = g_full.squeeze(0).permute(1, 2, 3, 0)       # (D, H, W, 3)

        result = torch.empty(C, D, H, W, dtype=image.dtype, device=image.device)

        for d_start in range(0, D, chunk_size):
            d_end = min(d_start + chunk_size, D)
            grid_chunk = g_full[d_start:d_end].unsqueeze(0)          # (1, chunk, H, W, 3)
            im_chunk = image[:, d_start:d_end].unsqueeze(0).float()  # (1, C, chunk, H, W)
            with torch.no_grad():
                warped = torch.nn.functional.grid_sample(
                    im_chunk, grid_chunk,
                    mode="bilinear", align_corners=True, padding_mode="border",
                )
            result[:, d_start:d_end] = warped.squeeze(0).to(image.dtype)
            del warped, grid_chunk, im_chunk
            gc.collect()

        return result


def mesh_transform_3d(
    image: torch.Tensor,
    grid_init: torch.Tensor,
    grid_target: torch.Tensor,
    method: str = "bilinear",
) -> torch.Tensor:
    """
    Convenience wrapper: build a MeshTransformer3D and apply it once.

    For repeated transforms over the same grid_init and image shape,
    build a MeshTransformer3D instance directly to reuse the cell index map.
    """
    spatial = tuple(image.shape[-3:])
    transformer = MeshTransformer3D(grid_init, spatial)
    return transformer.transform(image, grid_target, method)

