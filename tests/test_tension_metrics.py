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
_metrics_mod = None
if torch is not None:
    _tension_mod = _load_module("compute_tension_3d", "compute_tension_3d.py")
    _metrics_mod = _load_module("tension_metrics", "tension_metrics.py")


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


def _ten(grid_part, grid_full=None, **kw):
    return _tension_mod.compute_tension_3d(grid_part, grid_full, **kw)


# ---------------------------------------------------------------------------
# Блок V: VolumeTension
# ---------------------------------------------------------------------------

def test_V1_matches_default_abs():
    """compute_tension_3d(metric=VolumeTension()) == compute_tension_3d() в abs."""
    g = make_grid(3, 3, 3)
    d = make_deformed(g)
    vol = _metrics_mod.VolumeTension(mode="abs")
    assert torch.allclose(_ten(d, g), _ten(d, g, metric=vol), atol=1e-5)


def test_V2_matches_default_squared():
    """То же для mode='squared'."""
    g = make_grid(3, 3, 3)
    d = make_deformed(g)
    vol = _metrics_mod.VolumeTension(mode="squared")
    assert torch.allclose(
        _ten(d, g, mode="squared"),
        _ten(d, g, mode="squared", metric=vol),
        atol=1e-5,
    )


def test_V3_cell_weights_zeros():
    g = make_grid(3, 3, 3)
    d = make_deformed(g)
    vol = _metrics_mod.VolumeTension()
    w = torch.zeros(2, 2, 2)
    assert _ten(d, g, cell_weights=w, metric=vol).item() == 0.0


def test_V4_identity_zero():
    g = make_grid(3, 3, 3)
    vol = _metrics_mod.VolumeTension()
    assert _ten(g, g, metric=vol).item() < 1e-6


def test_V5_autograd_squared():
    g = make_grid(3, 3, 3)
    d = make_deformed(g, scale=0.1).requires_grad_(True)
    vol = _metrics_mod.VolumeTension(mode="squared")
    loss = _ten(d, g, metric=vol)
    loss.backward()
    assert d.grad is not None
    assert d.grad.isfinite().all()


def test_V6_nonsquare_grid():
    """VolumeTension работает для несимметричных сеток."""
    g = make_grid(4, 3, 5)
    d = make_deformed(g, scale=0.1)
    vol = _metrics_mod.VolumeTension()
    result = _ten(d, g, metric=vol)
    assert result.isfinite()


# ---------------------------------------------------------------------------
# Блок B: BendingTension
# ---------------------------------------------------------------------------

def test_B1_identity_zero():
    g = make_grid(4, 4, 4)
    bend = _metrics_mod.BendingTension()
    # metric receives normalised grids; here we pass them directly
    assert bend(g, g).item() < 1e-6


def test_B2_constant_displacement_zero():
    """u = const → нет кривизны → bending = 0."""
    g = make_grid(4, 4, 4)
    u_const = torch.ones_like(g) * 0.5
    bend = _metrics_mod.BendingTension()
    assert bend(g + u_const, g).item() < 1e-6


def test_B3_linear_displacement_zero():
    """u = alpha * coords → линейное поле → вторые производные = 0 → bending = 0."""
    g = make_grid(5, 5, 5)
    u_linear = g * 0.3  # u_i = 0.3 * x_j; вторые разности равны 0 на равномерной сетке
    bend = _metrics_mod.BendingTension()
    assert bend(g + u_linear, g).item() < 1e-5


def test_B4_quadratic_nonzero():
    """u = coords² → вторые производные ≠ 0."""
    g = make_grid(5, 5, 5)
    u_quad = g ** 2
    bend = _metrics_mod.BendingTension()
    assert bend(g + u_quad, g).item() > 0


def test_B5_monotone_with_amplitude():
    g = make_grid(5, 5, 5)
    torch.manual_seed(0)
    u0 = torch.randn_like(g)
    bend = _metrics_mod.BendingTension(mode="squared")
    vals = [bend(g + a * u0, g).item() for a in [0.05, 0.15, 0.30]]
    assert vals[0] < vals[1] < vals[2]


def test_B6_squared_autograd():
    g = make_grid(5, 5, 5)
    d = make_deformed(g, scale=0.2).requires_grad_(True)
    bend = _metrics_mod.BendingTension(mode="squared")
    loss = bend(d, g)
    loss.backward()
    assert d.grad is not None
    assert d.grad.isfinite().all()


def test_B7_smooth_less_than_noisy():
    """Гладкое поле смещений имеет меньшую bending energy, чем случайный шум."""
    g = make_grid(6, 6, 6)
    # Гладкое: синус с небольшой частотой
    coords = g[..., 0:1].expand_as(g)
    u_smooth = torch.sin(coords * 1.5) * 0.3
    # Шумное: случайный шум той же амплитуды
    torch.manual_seed(0)
    u_noisy = torch.randn_like(g) * 0.3
    bend = _metrics_mod.BendingTension(mode="squared")
    assert bend(g + u_smooth, g).item() < bend(g + u_noisy, g).item()


def test_B8_cell_weights_accepted_no_error():
    """BendingTension принимает cell_weights без ошибки (игнорирует его)."""
    g = make_grid(4, 4, 4)
    d = make_deformed(g, scale=0.1)
    bend = _metrics_mod.BendingTension()
    w = torch.ones(3, 3, 3)
    result = bend(d, g, cell_weights=w)
    assert result.isfinite()


# ---------------------------------------------------------------------------
# Блок P: протокол — произвольный callable
# ---------------------------------------------------------------------------

def test_P1_class_callable():
    g = make_grid(3, 3, 3)
    d = make_deformed(g)

    class ZeroMetric:
        def __call__(self, t, r, w=None):
            return torch.tensor(0.0)

    assert _ten(d, g, metric=ZeroMetric()).item() == 0.0


def test_P2_lambda_callable():
    g = make_grid(3, 3, 3)
    d = make_deformed(g)
    fixed_metric = lambda t, r, w=None: torch.tensor(7.0)
    assert _ten(d, g, metric=fixed_metric).item() == 7.0


def test_P3_metric_receives_normalised_grids():
    """grid_ref нормирован в [0, 1]; grid_target может чуть выходить за границы."""
    g = make_grid(3, 3, 3)
    d = make_deformed(g, scale=0.05)

    received = {}

    def capturing_metric(t, r, w=None):
        received["t"] = t.detach().clone()
        received["r"] = r.detach().clone()
        return torch.tensor(0.0)

    _ten(d, g, metric=capturing_metric)
    # grid_full нормируется на свой же диапазон → строго [0, 1]
    assert received["r"].amin().item() >= -1e-5
    assert received["r"].amax().item() <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# Визуальные ячейки (не pytest — только для интерактивного запуска)
# ---------------------------------------------------------------------------

#%%
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if torch is not None and _metrics_mod is not None:
        g = make_grid(5, 5, 5)
        torch.manual_seed(0)
        u0 = torch.randn_like(g)

        alphas = torch.linspace(0, 0.5, 25)
        vol   = _metrics_mod.VolumeTension(mode="squared")
        bend  = _metrics_mod.BendingTension(mode="squared")

        t_vol  = [_ten(g + a * u0, g, mode="squared").item() for a in alphas]
        t_bend = [_ten(g + a * u0, g, metric=bend).item() for a in alphas]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(alphas.numpy(), t_vol,  label="VolumeTension (squared)")
        ax.plot(alphas.numpy(), t_bend, label="BendingTension (squared)", linestyle="--")
        ax.set_xlabel("deformation amplitude")
        ax.set_ylabel("tension")
        ax.legend()
        ax.set_title("VolumeTension vs BendingTension")
        plt.tight_layout()
        plt.savefig("/tmp/tension_metrics_comparison.png", dpi=100)
        print("Saved /tmp/tension_metrics_comparison.png")
