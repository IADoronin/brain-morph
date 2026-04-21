#%%
import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mask_3d
from mesh_transform_3d import get_basis_3d, get_bilinear_transform_3d, iter_grid_cells_3d


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
        for cell_idx, cell_init in enumerate(iter_grid_cells_3d(self.grid_init)):
            cell_init_d = cell_init.to(dtype=self._mesh_f.dtype, device=self._mesh_f.device)
            mask = mask_3d.innerpoints(cell_init_d, self._mesh_f)
            index_map[mask] = cell_idx
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

# %%
