import os
import sys
import math
import importlib.util

import pytest


# load module by path to avoid package/import issues
mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "utils", "mesh_transform.py"))
spec = importlib.util.spec_from_file_location("mesh_transform", mod_path)
mesh_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mesh_mod)


torch = None
try:
    import torch
except Exception:
    torch = None


pytestmark = pytest.mark.skipif(torch is None, reason="torch is required for these tests")


def make_simple_volume(D=16, H=16, W=16, device="cpu", dtype=torch.float32):
    grid_coords = [torch.linspace(-1.0, 1.0, steps=s, device=device, dtype=dtype)
                   for s in (D, H, W)]
    gz, gy, gx = torch.meshgrid(*grid_coords, indexing="ij")
    vol = torch.exp(-((gx ** 2 + gy ** 2 + gz ** 2) * 8.0))
    return vol


def test_identity_transform():
    D, H, W = 16, 16, 16
    vol = make_simple_volume(D, H, W)
    mesh = mesh_mod.create_regular_mesh((3, 3, 3), (D, H, W), normalized=True)
    out = mesh_mod.mesh_transform(vol, mesh, mesh, precision="exact")
    assert out.shape == vol.shape
    assert torch.allclose(out, vol, atol=1e-6), "Identity transform must reproduce input"


def test_random_displacement_changes():
    D, H, W = 16, 16, 16
    vol = make_simple_volume(D, H, W)
    mesh_i = mesh_mod.create_regular_mesh((3, 3, 3), (D, H, W), normalized=True)
    rng = torch.randn_like(mesh_i)
    norms = torch.sqrt((rng ** 2).sum(dim=0, keepdim=True)).clamp_min(1e-6)
    rng_unit = rng / norms
    mesh_t = mesh_i + rng_unit * 0.1
    out = mesh_mod.mesh_transform(vol, mesh_i, mesh_t, precision="exact")
    diff = (out - vol).abs().mean().item()
    assert diff > 1e-6, "Random displacement should change the volume"
    assert diff < 0.5, "Change is unreasonably large"


def test_shape_and_dtype_preserved():
    # test with channel and batch dims
    D, H, W = 16, 16, 16
    C = 2
    vol = make_simple_volume(D, H, W).unsqueeze(0).repeat(C, 1, 1, 1)  # (C,D,H,W)
    mesh = mesh_mod.create_regular_mesh((3, 3, 3), (D, H, W), normalized=True)
    out = mesh_mod.mesh_transform(vol, mesh, mesh, precision="exact")
    assert out.shape == vol.shape
    assert out.dtype == vol.dtype


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
