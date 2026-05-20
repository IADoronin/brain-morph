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

_tension_mod = None
_transformer_mod = None
if torch is not None:
    _tension_mod = _load_module("compute_tension_3d", "compute_tension_3d.py")
    _transformer_mod = _load_module("mesh_transformer_3d", "mesh_transformer_3d.py")


# ---------------------------------------------------------------------------
# Loop reference — точная копия оригинальной реализации, эталон корректности
# ---------------------------------------------------------------------------

def _tension_loop(grid_part, grid_full=None):
    if grid_full is None:
        grid_full = grid_part

    grid_part = grid_part.to(dtype=torch.float32)
    grid_full = grid_full.to(dtype=torch.float32)

    mins = grid_full.amin(dim=(0, 1, 2), keepdim=True)
    maxs = grid_full.amax(dim=(0, 1, 2), keepdim=True)
    spans = (maxs - mins).clamp_min(1e-12)

    part = (grid_part - mins) / spans
    base = (grid_full - mins) / spans

    ny, nx, nz, _ = part.shape

    def _tet_det(a, b, c, d):
        mat = torch.stack([b - a, c - a, d - a], dim=1)
        return torch.det(mat)

    tension = torch.tensor(0.0, dtype=torch.float32)

    for i in range(ny - 1):
        for j in range(nx - 1):
            for k in range(nz - 1):
                N111 = part[i, j, k];     N112 = part[i, j, k + 1]
                N121 = part[i, j+1, k];   N122 = part[i, j+1, k+1]
                N211 = part[i+1, j, k];   N212 = part[i+1, j, k+1]
                N221 = part[i+1, j+1, k]; N222 = part[i+1, j+1, k+1]

                O111 = base[i, j, k];     O112 = base[i, j, k + 1]
                O121 = base[i, j+1, k];   O122 = base[i, j+1, k+1]
                O211 = base[i+1, j, k];   O212 = base[i+1, j, k+1]
                O221 = base[i+1, j+1, k]; O222 = base[i+1, j+1, k+1]

                for (n_a, n_b, n_c, n_d, o_a, o_b, o_c, o_d) in [
                    (N112, N111, N121, N211, O112, O111, O121, O211),
                    (N112, N122, N121, N222, O112, O122, O121, O222),
                    (N121, N221, N222, N211, O121, O221, O222, O211),
                    (N112, N212, N222, N211, O112, O212, O222, O211),
                    (N112, N211, N222, N121, O112, O211, O222, O121),
                ]:
                    vol_n = torch.abs(_tet_det(n_a, n_b, n_c, n_d))
                    vol_o = torch.abs(_tet_det(o_a, o_b, o_c, o_d))
                    tension = tension + torch.abs(vol_n - vol_o)

    return tension / 6.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_grid(ny=3, nx=3, nz=3):
    return torch.stack(
        torch.meshgrid(
            *[torch.linspace(-1, 1, steps=s) for s in (ny, nx, nz)],
            indexing="ij",
        ),
        dim=-1,
    )  # (ny, nx, nz, 3)


def make_deformed(grid, scale=0.1, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    return grid + torch.randn(grid.shape, generator=g) * scale


def _new(grid_part, grid_full=None, **kwargs):
    return _tension_mod.compute_tension_3d(grid_part, grid_full, **kwargs)


# ---------------------------------------------------------------------------
# Блок A: корректность — новая реализация == петлевой эталон
# ---------------------------------------------------------------------------

def test_A1_matches_loop_identity():
    g = make_grid(3, 3, 3)
    assert torch.allclose(_new(g, g), _tension_loop(g, g), atol=1e-5)


def test_A2_matches_loop_small_deform():
    g = make_grid(3, 3, 3)
    d = make_deformed(g, scale=0.05)
    assert torch.allclose(_new(d, g), _tension_loop(d, g), atol=1e-4)


def test_A3_matches_loop_large_deform():
    g = make_grid(3, 3, 3)
    d = make_deformed(g, scale=0.3)
    assert torch.allclose(_new(d, g), _tension_loop(d, g), atol=1e-4)


def test_A4_matches_loop_nonsquare():
    g = make_grid(4, 3, 5)
    d = make_deformed(g, scale=0.1)
    assert torch.allclose(_new(d, g), _tension_loop(d, g), atol=1e-4)


def test_A5_matches_loop_separate_grid_full():
    """grid_full задаёт только масштаб нормировки — та же форма, другой диапазон."""
    g = make_grid(3, 3, 3)
    g_full = make_grid(3, 3, 3) * 2.0  # та же форма, вдвое шире
    d = make_deformed(g, scale=0.1)
    assert torch.allclose(_new(d, g_full), _tension_loop(d, g_full), atol=1e-4)


# ---------------------------------------------------------------------------
# Блок B: математические свойства
# ---------------------------------------------------------------------------

def test_B1_identity_zero():
    g = make_grid(3, 3, 3)
    assert _new(g, g).item() < 1e-6


def test_B2_nonnegative():
    g = make_grid(3, 3, 3)
    for seed in [0, 1, 42]:
        d = make_deformed(g, scale=0.2, seed=seed)
        assert _new(d, g).item() >= 0.0


def test_B3_monotone_with_amplitude():
    g = make_grid(3, 3, 3)
    torch.manual_seed(0)
    u = torch.randn_like(g)
    vals = [_new(g + a * u, g).item() for a in [0.05, 0.15, 0.30]]
    assert vals[0] < vals[1] < vals[2]


def test_B4_grid_full_affects_scale():
    g = make_grid(3, 3, 3)
    d = make_deformed(g, scale=0.1)
    g_big = g * 2.0
    assert not torch.allclose(_new(d, g), _new(d, g_big), atol=1e-5)


# ---------------------------------------------------------------------------
# Блок C: параметр mode
# ---------------------------------------------------------------------------

def test_C1_default_equals_abs():
    g = make_grid(3, 3, 3)
    d = make_deformed(g)
    assert torch.allclose(_new(d, g), _new(d, g, mode="abs"))


def test_C2_squared_smaller_for_small_deform():
    g = make_grid(3, 3, 3)
    d = make_deformed(g, scale=0.05)
    assert _new(d, g, mode="squared").item() < _new(d, g, mode="abs").item()


def test_C3_squared_larger_for_large_deform():
    g = make_grid(3, 3, 3)
    d = make_deformed(g, scale=2.0)
    assert _new(d, g, mode="squared").item() > _new(d, g, mode="abs").item()


def test_C4_squared_autograd():
    g = make_grid(3, 3, 3)
    d = make_deformed(g, scale=0.1).requires_grad_(True)
    loss = _new(d, g, mode="squared")
    loss.backward()
    assert d.grad is not None
    assert d.grad.isfinite().all()


# ---------------------------------------------------------------------------
# Блок D: параметр cell_weights
# ---------------------------------------------------------------------------

def test_D1_none_equals_ones():
    g = make_grid(3, 3, 3)
    d = make_deformed(g)
    w = torch.ones(2, 2, 2)
    assert torch.allclose(_new(d, g), _new(d, g, cell_weights=w), atol=1e-5)


def test_D2_zeros_gives_zero():
    g = make_grid(3, 3, 3)
    d = make_deformed(g, scale=0.3)
    w = torch.zeros(2, 2, 2)
    assert _new(d, g, cell_weights=w).item() == 0.0


def test_D3_partial_between_zero_and_full():
    g = make_grid(3, 3, 3)
    d = make_deformed(g, scale=0.1)
    full = _new(d, g).item()
    w = torch.zeros(2, 2, 2)
    w[0] = 1.0  # первые 4 ячейки из 8
    partial = _new(d, g, cell_weights=w).item()
    assert 0.0 < partial < full


def test_D4_sum_of_cells_equals_full():
    """Сумма вкладов всех ячеек по отдельности == полный tension."""
    g = make_grid(3, 3, 3)
    d = make_deformed(g, scale=0.1)
    full = _new(d, g).item()
    total = 0.0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                w = torch.zeros(2, 2, 2)
                w[i, j, k] = 1.0
                total += _new(d, g, cell_weights=w).item()
    assert abs(total - full) < full * 0.001


# ---------------------------------------------------------------------------
# Блок E: числовая устойчивость и крайние случаи
# ---------------------------------------------------------------------------

def test_E1_minimal_grid_2x2x2():
    g = make_grid(2, 2, 2)
    d = make_deformed(g, scale=0.1)
    assert _new(d, g).isfinite()


def test_E2_degenerate_grid_no_nan():
    g = torch.zeros(3, 3, 3, 3)
    assert _new(g, g).isfinite()


def test_E3_large_deformation_finite():
    g = make_grid(3, 3, 3)
    d = g * 10.0
    assert _new(d, g).isfinite()


# ---------------------------------------------------------------------------
# Блок F: MeshTransformer3D._compute_cell_weights
# ---------------------------------------------------------------------------

def test_F1_output_shape():
    grid = make_grid(3, 4, 5)
    t = _transformer_mod.MeshTransformer3D(grid, (20, 24, 28))
    mask = torch.ones(20, 24, 28, dtype=torch.bool)
    w = t._compute_cell_weights(mask)
    assert w.shape == (2, 3, 4), f"expected (2, 3, 4), got {w.shape}"


def test_F2_all_ones_mask():
    grid = make_grid(3, 3, 3)
    t = _transformer_mod.MeshTransformer3D(grid, (20, 24, 28))
    mask = torch.ones(20, 24, 28, dtype=torch.bool)
    w = t._compute_cell_weights(mask)
    assert torch.allclose(w, torch.ones_like(w), atol=1e-5)


def test_F3_all_zeros_mask():
    grid = make_grid(3, 3, 3)
    t = _transformer_mod.MeshTransformer3D(grid, (20, 24, 28))
    mask = torch.zeros(20, 24, 28, dtype=torch.bool)
    w = t._compute_cell_weights(mask)
    assert (w == 0.0).all()


def test_F4_weights_in_unit_range():
    torch.manual_seed(7)
    grid = make_grid(3, 3, 3)
    t = _transformer_mod.MeshTransformer3D(grid, (20, 24, 28))
    mask = torch.randint(0, 2, (20, 24, 28), dtype=torch.bool)
    w = t._compute_cell_weights(mask)
    assert (w >= 0.0).all() and (w <= 1.0).all()


def test_F5_spatial_correspondence():
    """Ячейки в первой половине H получают вес ≈ 1 при маске там же."""
    grid = make_grid(3, 3, 3)
    t = _transformer_mod.MeshTransformer3D(grid, (20, 24, 28))
    mask = torch.zeros(20, 24, 28, dtype=torch.bool)
    mask[:, :12, :] = True  # первая половина по H (y < 0 в нормированном пространстве)
    w = t._compute_cell_weights(mask)
    assert w[:, 0, :].mean().item() > 0.7, "нижние ячейки должны иметь вес > 0.7"
    assert w[:, 1, :].mean().item() < 0.3, "верхние ячейки должны иметь вес < 0.3"


# ---------------------------------------------------------------------------
# Блок G: MeshTransformer3D.tension()
# ---------------------------------------------------------------------------

def test_G1_method_matches_standalone():
    g = make_grid(3, 3, 3)
    d = make_deformed(g, scale=0.1)
    t = _transformer_mod.MeshTransformer3D(g, (20, 24, 28))
    assert torch.allclose(t.tension(d), _new(d, g), atol=1e-4)


def test_G2_method_identity_zero():
    g = make_grid(3, 3, 3)
    t = _transformer_mod.MeshTransformer3D(g, (20, 24, 28))
    assert t.tension(g).item() < 1e-6


def test_G3_method_mask_zeros():
    g = make_grid(3, 3, 3)
    d = make_deformed(g, scale=0.2)
    t = _transformer_mod.MeshTransformer3D(g, (20, 24, 28))
    mask = torch.zeros(20, 24, 28, dtype=torch.bool)
    assert t.tension(d, mask=mask).item() == 0.0


def test_G4_method_mask_ones_equals_no_mask():
    g = make_grid(3, 3, 3)
    d = make_deformed(g, scale=0.1)
    t = _transformer_mod.MeshTransformer3D(g, (20, 24, 28))
    mask = torch.ones(20, 24, 28, dtype=torch.bool)
    assert torch.allclose(t.tension(d, mask=mask), t.tension(d), atol=1e-5)


def test_G5_method_grid_ref_none_equals_init():
    g = make_grid(3, 3, 3)
    d = make_deformed(g, scale=0.1)
    t = _transformer_mod.MeshTransformer3D(g, (20, 24, 28))
    assert torch.allclose(t.tension(d, grid_ref=None), t.tension(d, grid_ref=g), atol=1e-5)


def test_G6_method_grid_ref_custom_differs():
    g = make_grid(3, 3, 3)
    d = make_deformed(g, scale=0.1, seed=0)
    g_ref = make_deformed(g, scale=0.05, seed=1)
    t = _transformer_mod.MeshTransformer3D(g, (20, 24, 28))
    assert not torch.allclose(t.tension(d, grid_ref=None), t.tension(d, grid_ref=g_ref), atol=1e-5)


def test_G7_method_mode_squared_backward():
    g = make_grid(3, 3, 3)
    d = make_deformed(g, scale=0.1).requires_grad_(True)
    t = _transformer_mod.MeshTransformer3D(g, (20, 24, 28))
    loss = t.tension(d, mode="squared")
    loss.backward()
    assert d.grad is not None
    assert d.grad.isfinite().all()


def test_G8_reuse_transformer():
    g = make_grid(3, 3, 3)
    t = _transformer_mod.MeshTransformer3D(g, (20, 24, 28))
    results = [t.tension(make_deformed(g, scale=0.1, seed=s)).item() for s in [0, 1, 2]]
    assert len(set(results)) == 3


# ---------------------------------------------------------------------------
# Визуальные ячейки (не pytest — только для интерактивного запуска)
# ---------------------------------------------------------------------------

#%%
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if torch is not None and _tension_mod is not None:
        g = make_grid(3, 3, 3)
        torch.manual_seed(0)
        u = torch.randn_like(g)

        #%% tension vs amplitude
        alphas = torch.linspace(0, 0.5, 30)
        t_abs = [_new(g + a * u, g, mode="abs").item() for a in alphas]
        t_sq  = [_new(g + a * u, g, mode="squared").item() for a in alphas]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(alphas.numpy(), t_abs, label='mode="abs"')
        ax.plot(alphas.numpy(), t_sq,  label='mode="squared"', linestyle="--")
        ax.set_xlabel("deformation amplitude")
        ax.set_ylabel("tension")
        ax.legend()
        ax.set_title("Tension vs deformation amplitude")
        plt.tight_layout()
        plt.savefig("/tmp/tension_vs_amplitude.png", dpi=100)
        print("Saved /tmp/tension_vs_amplitude.png")

        #%% cell_weights для маски с выпавшей ОЛ
        if _transformer_mod is not None:
            t_obj = _transformer_mod.MeshTransformer3D(g, (20, 24, 28))
            mask = torch.zeros(20, 24, 28, dtype=torch.bool)
            mask[:5, :, :] = True  # "ОЛ" = первые 5 срезов по глубине
            w = t_obj._compute_cell_weights(mask)
            print("cell_weights (ОЛ-маска), shape", w.shape)
            print(w[:, :, 0])  # срез по z=0
