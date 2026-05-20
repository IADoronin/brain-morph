#%%
import os
import sys
import pytest

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (
    os.path.join(_root, "src", "utils"),
    os.path.join(_root, "src", "registration"),
    os.path.join(_root, "src"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

torch = None
try:
    import torch
except Exception:
    pass

pytestmark = pytest.mark.skipif(torch is None, reason="torch is required")

if torch is not None:
    from pipeline import Stage, RegistrationPipeline, interpolate_grid
    from optimizers import SAOptimizer, GradientOptimizer
    from mesh_transformer_3d import MeshTransformer3D
    from cost import registration_cost


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_image(shape=(12, 14, 16), seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.rand(shape, generator=g)


def sa_fast(**kw):
    return SAOptimizer(temp_start=1e-4, temp_end=1e-5, attention_freq=0, **kw)


def gd_fast(**kw):
    return GradientOptimizer(lr=1e-3, tension_mode="squared", **kw)


# ---------------------------------------------------------------------------
# PL1 — interpolate_grid: identity stays identity
# ---------------------------------------------------------------------------

def test_PL1_interpolate_identity():
    g = torch.stack(
        torch.meshgrid(*[torch.linspace(-1, 1, s) for s in (3, 3, 3)], indexing="ij"),
        dim=-1,
    )
    g_up = interpolate_grid(g, (5, 5, 5))
    # corners must be preserved exactly
    assert torch.allclose(g_up[0, 0, 0], g[0, 0, 0], atol=1e-5)
    assert torch.allclose(g_up[-1, -1, -1], g[-1, -1, -1], atol=1e-5)


# ---------------------------------------------------------------------------
# PL2 — interpolate_grid: output shape
# ---------------------------------------------------------------------------

def test_PL2_interpolate_shape():
    g = torch.randn(3, 4, 5, 3)
    g_up = interpolate_grid(g, (6, 7, 8))
    assert g_up.shape == (6, 7, 8, 3)


# ---------------------------------------------------------------------------
# PL3 — interpolate_grid: upsample then downsample ≈ original
# ---------------------------------------------------------------------------

def test_PL3_interpolate_roundtrip():
    g = torch.stack(
        torch.meshgrid(*[torch.linspace(-1, 1, s) for s in (4, 4, 4)], indexing="ij"),
        dim=-1,
    )
    g_up   = interpolate_grid(g, (8, 8, 8))
    g_down = interpolate_grid(g_up, (4, 4, 4))
    assert torch.allclose(g_down, g, atol=1e-4)


# ---------------------------------------------------------------------------
# PL4 — single-stage pipeline == direct optimizer call
# ---------------------------------------------------------------------------

def test_PL4_single_stage_matches_direct():
    im1 = make_image(seed=1)
    im2 = make_image(seed=2)

    torch.manual_seed(0)
    sa_a = sa_fast()
    pipeline = RegistrationPipeline([Stage((3, 3, 3), sa_a, n_steps=5, lam=1e-3)])
    grid_pipe = pipeline.run(im1, im2)

    # Re-run with same seed directly
    grid_init = torch.stack(
        torch.meshgrid(*[torch.linspace(-1, 1, 3)]*3, indexing="ij"), dim=-1
    )
    t = MeshTransformer3D(grid_init, tuple(im1.shape))
    torch.manual_seed(0)
    sa_b = sa_fast()
    grid_direct = sa_b.optimize(im1, im2, t, grid_init, n_steps=5, lam=1e-3)

    assert torch.allclose(grid_pipe, grid_direct, atol=1e-6)


# ---------------------------------------------------------------------------
# PL5 — two-stage pipeline: output shape matches last stage grid
# ---------------------------------------------------------------------------

def test_PL5_two_stage_output_shape():
    im1 = make_image(seed=3)
    im2 = make_image(seed=4)

    pipeline = RegistrationPipeline([
        Stage((3, 3, 3), sa_fast(), n_steps=5, lam=1e-2),
        Stage((4, 4, 4), sa_fast(), n_steps=5, lam=1e-3),
    ])
    grid = pipeline.run(im1, im2)
    assert grid.shape == (4, 4, 4, 3)


# ---------------------------------------------------------------------------
# PL6 — different optimizers per stage (SA → GD)
# ---------------------------------------------------------------------------

def test_PL6_sa_then_gd():
    im1 = make_image(seed=5)
    im2 = make_image(seed=6)

    pipeline = RegistrationPipeline([
        Stage((3, 3, 3), sa_fast(), n_steps=5, lam=1e-2),
        Stage((4, 4, 4), gd_fast(), n_steps=5, lam=1e-3),
    ])
    grid = pipeline.run(im1, im2)
    assert grid.isfinite().all()
    assert grid.shape == (4, 4, 4, 3)


# ---------------------------------------------------------------------------
# PL7 — pipeline with mask: no errors, finite output
# ---------------------------------------------------------------------------

def test_PL7_with_mask():
    im1 = make_image(seed=7)
    im2 = make_image(seed=8)
    mask = torch.ones(im1.shape, dtype=torch.bool)
    mask[:4] = False  # half the volume empty

    pipeline = RegistrationPipeline([
        Stage((3, 3, 3), sa_fast(), n_steps=5, lam=1e-2),
    ])
    grid = pipeline.run(im1, im2, mask=mask)
    assert grid.isfinite().all()


# ---------------------------------------------------------------------------
# PL8 — empty stages list raises ValueError
# ---------------------------------------------------------------------------

def test_PL8_empty_stages_raises():
    with pytest.raises(ValueError):
        RegistrationPipeline([])
