#%%
import os
import sys
import importlib.util

import pytest

_utils_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "utils"))


def _load_module(name, filename):
    if _utils_dir not in sys.path:
        sys.path.insert(0, _utils_dir)
    path = os.path.join(_utils_dir, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


torch = None
try:
    import torch
except Exception:
    pass

pytestmark = pytest.mark.skipif(torch is None, reason="torch is required")

_mod = None
if torch is not None:
    _mod = _load_module("mesh_transformer_3d", "mesh_transformer_3d.py")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_gaussian_volume(D=20, H=24, W=28, channels=None):
    coords = [torch.linspace(-1.0, 1.0, steps=s) for s in (D, H, W)]
    gz, gy, gx = torch.meshgrid(*coords, indexing="ij")
    vol = torch.exp(-(gx ** 2 + gy ** 2 + gz ** 2) * 6.0)  # (D, H, W)
    if channels is not None:
        vol = vol.unsqueeze(0).repeat(channels, 1, 1, 1)  # (C, D, H, W)
    return vol


def make_grid(ny=3, nx=3, nz=3):
    return torch.stack(
        torch.meshgrid(
            *[torch.linspace(-1, 1, steps=s) for s in (ny, nx, nz)],
            indexing="ij",
        ),
        dim=-1,
    )  # (ny, nx, nz, 3)


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

def test_instantiation():
    grid = make_grid()
    t = _mod.MeshTransformer3D(grid, (20, 24, 28))
    assert t.cell_index_map.shape == (20 * 24 * 28,)
    assert t.cell_index_map.dtype == torch.long


def test_wrong_grid_dim_raises():
    bad = torch.zeros(3, 3, 3, 2)  # last dim should be 3
    with pytest.raises(ValueError):
        _mod.MeshTransformer3D(bad, (20, 24, 28))


def test_wrong_image_shape_dim_raises():
    grid = make_grid()
    with pytest.raises(ValueError):
        _mod.MeshTransformer3D(grid, (20, 24))  # 2-tuple instead of 3


# ---------------------------------------------------------------------------
# cell index map coverage
# ---------------------------------------------------------------------------

def test_cell_index_map_covers_interior():
    grid = make_grid()
    t = _mod.MeshTransformer3D(grid, (20, 24, 28))
    unassigned = (t.cell_index_map == -1).sum().item()
    total = 20 * 24 * 28
    assert unassigned / total < 0.01, f"Too many unassigned voxels: {unassigned}/{total}"


def test_cell_index_map_range():
    ny, nx, nz = 3, 4, 5
    grid = make_grid(ny, nx, nz)
    n_cells = (ny - 1) * (nx - 1) * (nz - 1)
    t = _mod.MeshTransformer3D(grid, (16, 16, 16))
    valid = t.cell_index_map[t.cell_index_map >= 0]
    assert valid.max().item() <= n_cells - 1


# ---------------------------------------------------------------------------
# identity transform
# ---------------------------------------------------------------------------

def test_identity_3d_input():
    grid = make_grid()
    vol = make_gaussian_volume()  # (D, H, W)
    t = _mod.MeshTransformer3D(grid, vol.shape)
    result = t.transform(vol, grid)
    assert result.shape == (1, *vol.shape), "3D input gets a channel dim added"
    assert torch.allclose(result.squeeze(0), vol, atol=1e-4)


def test_identity_4d_input():
    grid = make_grid()
    vol = make_gaussian_volume(channels=3)  # (3, D, H, W)
    t = _mod.MeshTransformer3D(grid, vol.shape[1:])
    result = t.transform(vol, grid)
    assert result.shape == vol.shape
    assert torch.allclose(result, vol, atol=1e-4)


def test_identity_single_channel():
    grid = make_grid()
    vol = make_gaussian_volume(channels=1)  # (1, D, H, W)
    t = _mod.MeshTransformer3D(grid, vol.shape[1:])
    result = t.transform(vol, grid)
    assert result.shape == vol.shape
    assert torch.allclose(result, vol, atol=1e-4)


# ---------------------------------------------------------------------------
# non-identity transforms
# ---------------------------------------------------------------------------

def test_transform_changes_volume():
    grid = make_grid()
    vol = make_gaussian_volume(channels=1)
    t = _mod.MeshTransformer3D(grid, vol.shape[1:])
    result = t.transform(vol, grid + 0.2)
    assert (result - vol).abs().mean().item() > 1e-4


def test_scale_up_then_down():
    """Scale up then scale down should roughly restore the original."""
    grid = make_grid()
    vol = make_gaussian_volume(channels=1)
    t = _mod.MeshTransformer3D(grid, vol.shape[1:])

    scaled_up = t.transform(vol, grid * 1.5)
    # build a new transformer with the same grid (grid_init = grid, target = grid / 1.5)
    restored = t.transform(scaled_up, grid / 1.5)

    # rough round-trip: should be closer to original than scaled_up is
    diff_restored = (restored - vol).abs().mean().item()
    diff_scaled = (scaled_up - vol).abs().mean().item()
    assert diff_restored < diff_scaled


# ---------------------------------------------------------------------------
# reuse: same transformer, different inputs / targets
# ---------------------------------------------------------------------------

def test_reuse_different_images():
    grid = make_grid()
    vol1 = make_gaussian_volume(channels=1)
    vol2 = torch.rand_like(vol1)
    grid_target = grid + torch.rand_like(grid) * 0.1

    t = _mod.MeshTransformer3D(grid, vol1.shape[1:])
    r1 = t.transform(vol1, grid_target)
    r2 = t.transform(vol2, grid_target)

    assert r1.shape == vol1.shape
    assert r2.shape == vol2.shape
    assert not torch.allclose(r1, r2, atol=1e-3)


def test_reuse_different_targets():
    grid = make_grid()
    vol = make_gaussian_volume(channels=1)
    t = _mod.MeshTransformer3D(grid, vol.shape[1:])

    r1 = t.transform(vol, grid + 0.15)
    r2 = t.transform(vol, grid - 0.15)
    assert not torch.allclose(r1, r2, atol=1e-3)


# ---------------------------------------------------------------------------
# standalone function == class result
# ---------------------------------------------------------------------------

def test_standalone_matches_class():
    grid = make_grid()
    vol = make_gaussian_volume(channels=1)
    grid_target = grid + torch.rand_like(grid) * 0.1

    t = _mod.MeshTransformer3D(grid, vol.shape[1:])
    result_class = t.transform(vol, grid_target)
    result_fn = _mod.mesh_transform_3d(vol, grid, grid_target)

    assert torch.allclose(result_class, result_fn, atol=1e-6)


# ---------------------------------------------------------------------------
# validation errors
# ---------------------------------------------------------------------------

def test_wrong_target_shape_raises():
    grid = make_grid(3, 3, 3)
    vol = make_gaussian_volume(channels=1)
    t = _mod.MeshTransformer3D(grid, vol.shape[1:])
    with pytest.raises(ValueError):
        t.transform(vol, make_grid(4, 4, 4))


def test_wrong_image_shape_raises():
    grid = make_grid()
    t = _mod.MeshTransformer3D(grid, (20, 24, 28))
    wrong_vol = make_gaussian_volume(D=10, H=10, W=10, channels=1)
    with pytest.raises(ValueError):
        t.transform(wrong_vol, grid)


# ---------------------------------------------------------------------------
# dtype and device
# ---------------------------------------------------------------------------

def test_dtype_preserved():
    grid = make_grid()
    vol = make_gaussian_volume(channels=2).to(torch.float64)
    t = _mod.MeshTransformer3D(grid, vol.shape[1:], dtype=torch.float64)
    result = t.transform(vol, grid)
    assert result.dtype == vol.dtype


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])


# =============================================================================
# Visual tests (run as script or Jupyter cells with #%%)
# =============================================================================
# %%
import random
from math import cos, sin, pi

if _utils_dir not in sys.path:
    sys.path.insert(0, _utils_dir)

import volume as vol_mod
Volume = vol_mod.Volume

if torch is not None and _mod is not None:
    _load_module("mesh_transform_3d", "mesh_transform_3d.py")


def gen_rand_color():
    theta = random.random() * pi / 3
    phi = random.random() * pi / 2
    return torch.tensor(
        [255 * sin(theta) * cos(phi),
         255 * sin(theta) * sin(phi),
         255 * cos(theta)],
        dtype=torch.uint8,
    )


def create_colored_volume(D=100, H=120, W=160, block=18, step=20):
    """Grid of colored cubes, similar to test_mt3d.py."""
    im = torch.zeros((3, D, H, W), dtype=torch.uint8)
    for i in range(D // step):
        for j in range(H // step):
            for k in range(W // step):
                color = gen_rand_color().reshape(3, 1, 1, 1)
                s = 1  # 1-px border
                im[:, i*step+s:i*step+s+block,
                      j*step+s:j*step+s+block,
                      k*step+s:k*step+s+block] = color * torch.ones(3, block, block, block, dtype=torch.uint8)
    return (im / im.max()).float()


# %%
# --- setup: build image and grid ---
grid_vis = make_grid(2, 3, 4)
image_vis = create_colored_volume()
print("Image shape:", image_vis.shape)
transformer_vis = _mod.MeshTransformer3D(grid_vis, image_vis.shape[1:])
print("Cell index map shape:", transformer_vis.cell_index_map.shape)
print(f"Unassigned voxels: {(transformer_vis.cell_index_map == -1).sum().item()}")

print("--- Original image ---")
Volume(image_vis).visualize()

# %%
# --- test 1: identity transform ---
image_identity = transformer_vis.transform(image_vis, grid_vis)
print("--- Identity transform ---")
Volume(image_identity).visualize()
print("--- Difference (should be ~zero) ---")
Volume((image_identity - image_vis).abs()).visualize()

# %%
# --- test 2: scale up (zoom in) ---
image_scaled = transformer_vis.transform(image_vis, grid_vis * 1.5)
print("--- Scale up x1.5 ---")
Volume(image_scaled).visualize()

# %%
# --- test 3: scale down (zoom out) ---
image_small = transformer_vis.transform(image_vis, grid_vis * 0.6)
print("--- Scale down x0.6 ---")
Volume(image_small).visualize()

# %%
# --- test 4: translation ---
image_shifted = transformer_vis.transform(image_vis, grid_vis + torch.tensor([0.3, 0.3, 0.3]))
print("--- Translation (+0.3 in all axes) ---")
Volume(image_shifted).visualize()

# %%
# --- test 5: random local deformation ---
torch.manual_seed(42)
grid_random = grid_vis + torch.randn_like(grid_vis) * 0.05
image_deformed = transformer_vis.transform(image_vis, grid_random)
print("--- Random local deformation ---")
Volume(image_deformed).visualize()

# %%
# --- test 5': reversed transformation ---
transformer_vis_reversed = _mod.MeshTransformer3D(grid_random, image_vis.shape[1:])

image_reversed = transformer_vis_reversed.transform(image_deformed, grid_vis)
print("--- Random local deformation ---")
Volume(image_reversed).visualize()
Volume(image_reversed).rotate(15,15,(100,120,160)).visualize()
Volume(image_vis).rotate(15,15,(100,120,160)).visualize()

# %%
# --- test 6: reuse — apply different targets to same image without rebuilding ---
print("--- Reuse demo: same transformer, three different targets ---")
for shift in [0.2, -0.2, 0.0]:
    result = transformer_vis.transform(image_vis, grid_vis + shift)
    Volume(result).visualize()
# %%
