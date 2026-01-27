#%%
import torch
from torch.nn import functional as F
from typing import Tuple


def _normalize_mesh(mesh: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
    """Normalize mesh coordinates to [-1, 1].

    If mesh values are already in [-1,1] they are returned unchanged. Otherwise
    they are interpreted as voxel coordinates (0..size-1) and converted.
    mesh: Tensor of shape (3, nx, ny, nz) or (nx, ny, nz, 3)
    grid_size: (D, H, W)
    """
    if mesh.dim() == 4 and mesh.shape[0] == 3:
        m = mesh
    elif mesh.dim() == 4 and mesh.shape[-1] == 3:
        # (nx,ny,nz,3) -> (3,nx,ny,nz)
        m = mesh.permute(3, 0, 1, 2).contiguous()
    else:
        raise ValueError("mesh must have shape (3,nx,ny,nz) or (nx,ny,nz,3)")

    max_abs = m.abs().max()
    D, H, W = grid_size
    if max_abs <= 1.0:
        return m

    # treat as voxel coordinates -> normalize
    # channels: 0->x (width), 1->y (height), 2->z (depth)
    scales = torch.tensor([W - 1, H - 1, D - 1], dtype=m.dtype, device=m.device).reshape((3, 1, 1, 1))
    return (m / scales) * 2.0 - 1.0


def _ensure_mesh_channels_first(mesh: torch.Tensor) -> torch.Tensor:
    """Return mesh as (3, nx, ny, nz)."""
    if mesh.dim() == 4 and mesh.shape[0] == 3:
        return mesh
    if mesh.dim() == 4 and mesh.shape[-1] == 3:
        return mesh.permute(3, 0, 1, 2).contiguous()
    raise ValueError("mesh must have shape (3,nx,ny,nz) or (nx,ny,nz,3)")


def mesh_transform(
    volume: torch.Tensor,
    mesh_initial: torch.Tensor,
    mesh_transformed: torch.Tensor,
    precision: str = "exact",
    align_corners: bool = True,
    padding_mode: str = "border",
) -> torch.Tensor:
    """Warp a 3D `volume` according to control-point displacement between meshes.

    - Inputs and outputs are torch tensors and may live on GPU.
    - `mesh_initial` and `mesh_transformed` are control grids describing the
      correspondence. Accepted shapes: (3, nx, ny, nz) or (nx, ny, nz, 3).
    - Mesh coordinates may be in normalized coordinates [-1,1] or in voxel
      coordinates [0..W-1 / H-1 / D-1]; normalization is detected automatically.
    - Uses `F.grid_sample` for resampling.

    Parameters:
    - volume: tensor shaped (D,H,W) or (C,D,H,W) or with batch (N,C,D,H,W). If
      no batch dim provided, treated as batch=1.
    - precision: 'exact' (default) uses trilinear interpolation for displacement
      upsampling and trilinear sampling; 'coarse' uses nearest upsampling of the
      displacement field (faster, less accurate); 'nearest' uses nearest sampling.

    Returns:
    - Transformed volume with same shape as input `volume`.
    """
    # Normalize input dims: make input shape (N, C, D, H, W)
    orig_shape = volume.shape
    is_batched = (volume.dim() == 5)
    if volume.dim() == 3:
        vol = volume.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    elif volume.dim() == 4:
        # (C,D,H,W) -> (1,C,D,H,W)
        vol = volume.unsqueeze(0)
    elif volume.dim() == 5:
        vol = volume
    else:
        raise ValueError("volume must be 3D, 4D (C,D,H,W) or 5D (N,C,D,H,W)")

    N, C, D, H, W = vol.shape

    # Ensure meshes are (3, nx, ny, nz)
    mesh_i = _ensure_mesh_channels_first(mesh_initial)
    mesh_t = _ensure_mesh_channels_first(mesh_transformed)

    # Normalize meshes to [-1,1] if necessary
    mesh_i_n = _normalize_mesh(mesh_i, (D, H, W)).to(vol.device).to(vol.dtype)
    mesh_t_n = _normalize_mesh(mesh_t, (D, H, W)).to(vol.device).to(vol.dtype)

    # Control-point displacement in normalized coords
    disp_ctrl = mesh_t_n - mesh_i_n  # (3, nx, ny, nz)

    # Prepare for interpolation: shape (1,3,nx,ny,nz)
    disp = disp_ctrl.unsqueeze(0)

    # Upsample displacement to full volume resolution
    mode_up = "trilinear" if precision == "exact" else "nearest"
    disp_upsampled = F.interpolate(
        disp,
        size=(D, H, W),
        mode=mode_up,
        align_corners=align_corners if mode_up == "trilinear" else None,
    )
    # disp_upsampled: (1,3,D,H,W)

    # Build base identity grid in normalized coords: shape (D,H,W,3)
    grid_coords = [torch.linspace(-1.0, 1.0, steps=s, device=vol.device, dtype=vol.dtype)
                   for s in (D, H, W)]
    gz, gy, gx = torch.meshgrid(*grid_coords, indexing="ij")
    base_grid = torch.stack([gx, gy, gz], dim=-1)  # (D,H,W,3)

    # displacement is (1,3,D,H,W) -> (D,H,W,3)
    disp_field = disp_upsampled[0].permute(1, 2, 3, 0)

    # Final sampling grid
    sampling_grid = base_grid + disp_field

    # Expand to batch
    sampling_grid = sampling_grid.unsqueeze(0)  # (1,D,H,W,3)

    # Choose sampling mode for grid_sample. For 5D volumes PyTorch expects
    # 'bilinear' (which performs trilinear interpolation for 5D inputs).
    sample_mode = "bilinear" if precision == "exact" else (
        "nearest" if precision == "nearest" else "bilinear"
    )

    out = F.grid_sample(
        vol,
        sampling_grid,
        mode=sample_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    # Squeeze extra batch dims if original didn't have them
    if volume.dim() == 3:
        return out[0, 0]
    if volume.dim() == 4:
        return out[0]
    return out


def create_regular_mesh(num_cells:Tuple[int,int,int], image_shape:Tuple[int,int,int], normalized: bool = True) -> torch.Tensor:
    """Create a regular control mesh.

    Returns tensor shape (3, nx, ny, nz). If `normalized`, coords are in [-1,1].
    Otherwise returns voxel coords in [0..size-1].
    """
    grid_bias = [torch.linspace(0.0, 1.0, steps=n+1) for n in num_cells]
    grid = torch.stack(torch.meshgrid(*grid_bias, indexing="ij"), dim=0)
    scale = 2.0 if normalized else torch.tensor(image_shape).reshape((3, 1, 1, 1))
    return grid * scale - (1.0 if normalized else 0.0)


def run_random_node_displacement_test(
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    control_shape=(4, 4, 4),
    vol_shape=(64, 64, 64),
    displacement_frac: float = 0.1,
) -> None:
    """Quick test: every control node displaced randomly by `displacement_frac` (relative normalized units).

    Builds a simple synthetic volume, warps it and prints a few diagnostics.
    """
    nx, ny, nz = control_shape
    D, H, W = vol_shape
    device = torch.device(device)

    # simple test volume: concentric shells
    grid_coords = [torch.linspace(-1.0, 1.0, steps=s, device=device, dtype=dtype)
                   for s in (D, H, W)]
    gz, gy, gx = torch.meshgrid(*grid_coords, indexing="ij")
    vol = torch.exp(-((gx ** 2 + gy ** 2 + gz ** 2) * 8.0)).to(device=device, dtype=dtype)

    mesh_i = create_regular_mesh((nx, ny, nz), (D, H, W), normalized=True).to(device=device, dtype=dtype)

    # random unit displacements
    rng = torch.randn_like(mesh_i)
    norms = torch.sqrt((rng ** 2).sum(dim=0, keepdim=True))
    norms = norms.clamp_min(1e-6)
    rng_unit = rng / norms
    mesh_t = mesh_i + rng_unit * displacement_frac

    warped = mesh_transform(vol, mesh_i, mesh_t, precision="exact").to(device="cpu")

    # diagnostics
    diff = (warped - vol.cpu()).abs().mean().item()
    print(f"Random-node-displacement test: vol_shape={vol_shape}, control={control_shape}")
    print(f"Mean absolute difference after warp: {diff:.6f}")


if __name__ == "__main__":
    run_random_node_displacement_test()
