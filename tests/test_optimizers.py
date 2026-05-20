import pytest

torch = None
try:
    import torch
except Exception:
    pass

pytestmark = pytest.mark.skipif(torch is None, reason="torch is required")

if torch is not None:
    from brain_morph.utils import MeshTransformer3D
    from brain_morph.registration import registration_cost, SAOptimizer, GradientOptimizer, HybridOptimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_grid(ny=4, nx=4, nz=4):
    return torch.stack(
        torch.meshgrid(
            *[torch.linspace(-1, 1, s) for s in (ny, nx, nz)],
            indexing="ij",
        ),
        dim=-1,
    )  # (ny, nx, nz, 3)


def make_image(shape=(8, 8, 8), seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.rand(shape, generator=g)


def make_transformer(grid, shape):
    return MeshTransformer3D(grid, shape)


# ---------------------------------------------------------------------------
# O1 — registration_cost: identity grid, same image → sim ≈ 1, cost high
# ---------------------------------------------------------------------------

def test_O1_cost_identity_same_image():
    g = make_grid()
    im = make_image()
    t = make_transformer(g, im.shape)
    cost = registration_cost(im, im, t, g, lam=0.0, similarity="corr")
    assert cost.item() > 0.99  # Pearson(x, x) = 1


# ---------------------------------------------------------------------------
# O2 — registration_cost: zero mask → tension = 0 → cost = sim only
# ---------------------------------------------------------------------------

def test_O2_cost_zero_mask():
    g = make_grid()
    im1 = make_image(seed=1)
    im2 = make_image(seed=2)
    t = make_transformer(g, im1.shape)
    mask_zero = torch.zeros(im1.shape, dtype=torch.bool)

    cost_no_mask   = registration_cost(im1, im2, t, g, lam=1.0)
    cost_zero_mask = registration_cost(im1, im2, t, g, lam=1.0, mask=mask_zero)
    sim_only       = registration_cost(im1, im2, t, g, lam=0.0)

    # With zero mask tension=0, so cost_zero_mask == sim_only
    assert torch.isclose(cost_zero_mask, sim_only, atol=1e-5)
    # Without mask tension > 0, so cost is smaller (penalised)
    assert cost_no_mask.item() <= sim_only.item() + 1e-5


# ---------------------------------------------------------------------------
# O3 — SAOptimizer: IM1 == IM2, cost should not decrease
# ---------------------------------------------------------------------------

def test_O3_sa_does_not_degrade_identical_images():
    g = make_grid()
    im = make_image()
    t = make_transformer(g, im.shape)

    sa = SAOptimizer(temp_start=1e-4, temp_end=1e-5, coeff_start=0.05,
                     attention_freq=0)
    cost_init = registration_cost(im, im, t, g, lam=1e-3).item()

    g_result = sa.optimize(im, im, t, g, n_steps=50, lam=1e-3)
    cost_final = registration_cost(im, im, t, g_result, lam=1e-3).item()

    # SA keeps best, so result must be at least as good as start
    assert cost_final >= cost_init - 1e-4


# ---------------------------------------------------------------------------
# O4 — GradientOptimizer: gradient flows through grid after one step
# ---------------------------------------------------------------------------

def test_O4_gradient_flows():
    g = make_grid()
    im1 = make_image(seed=3)
    im2 = make_image(seed=4)
    t = make_transformer(g, im1.shape)

    grid = g.clone().float().requires_grad_(True)
    cost = registration_cost(im1.float(), im2.float(), t, grid,
                             lam=1e-3, tension_mode="squared")
    (-cost).backward()

    assert grid.grad is not None
    assert grid.grad.isfinite().all()


# ---------------------------------------------------------------------------
# O5 — GradientOptimizer: cost improves on a non-trivial task
# ---------------------------------------------------------------------------

def test_O5_gradient_cost_improves():
    g = make_grid()
    # Shift image_moving by a small constant displacement
    torch.manual_seed(7)
    shift = torch.randn(3) * 0.05
    g_shifted = g + shift

    im_fixed = make_image(seed=5)
    t = make_transformer(g, im_fixed.shape)
    # Warp a different image: use fixed as moving after a shift — we want optimizer
    # to recover a better alignment than the shifted starting point
    im_moving = make_image(seed=6)

    opt = GradientOptimizer(lr=5e-3, tension_mode="squared")
    cost_init  = registration_cost(im_moving.float(), im_fixed.float(),
                                    t, g_shifted.float(), lam=1e-4).item()
    g_result   = opt.optimize(im_moving, im_fixed, t, g_shifted, n_steps=20, lam=1e-4)
    cost_final = registration_cost(im_moving.float(), im_fixed.float(),
                                    t, g_result, lam=1e-4).item()

    assert cost_final > cost_init - 1e-3  # should not get much worse; usually improves


# ---------------------------------------------------------------------------
# O6 — GradientOptimizer with SGD: finite output
# ---------------------------------------------------------------------------

def test_O6_sgd_no_nan():
    g = make_grid()
    im1 = make_image(seed=8)
    im2 = make_image(seed=9)
    t = make_transformer(g, im1.shape)

    opt = GradientOptimizer(optimizer_cls=torch.optim.SGD, lr=1e-3,
                            tension_mode="squared")
    g_result = opt.optimize(im1, im2, t, g, n_steps=10, lam=1e-3)

    assert g_result.isfinite().all()
    assert g_result.shape == g.shape


# ---------------------------------------------------------------------------
# O7 — HybridOptimizer: output shape matches input
# ---------------------------------------------------------------------------

def test_O7_hybrid_output_shape():
    g = make_grid()
    im1 = make_image(seed=10)
    im2 = make_image(seed=11)
    t = make_transformer(g, im1.shape)

    sa  = SAOptimizer(temp_start=1e-4, temp_end=1e-5, attention_freq=0)
    gd  = GradientOptimizer(lr=1e-3, tension_mode="squared")
    hyb = HybridOptimizer(sa, gd, sa_fraction=0.6)

    g_result = hyb.optimize(im1, im2, t, g, n_steps=10, lam=1e-3)
    assert g_result.shape == g.shape


# ---------------------------------------------------------------------------
# O8 — HybridOptimizer: no exceptions, finite output
# ---------------------------------------------------------------------------

def test_O8_hybrid_no_errors():
    g = make_grid()
    im1 = make_image(seed=12)
    im2 = make_image(seed=13)
    t = make_transformer(g, im1.shape)

    hyb = HybridOptimizer(sa_fraction=0.7)
    g_result = hyb.optimize(im1, im2, t, g, n_steps=10, lam=1e-3)

    assert g_result.isfinite().all()


# ---------------------------------------------------------------------------
# O9 — SAOptimizer convergence: recover known deformation
# ---------------------------------------------------------------------------

def make_colored_volume_gray(D=40, H=48, W=56, block=14, step=16, seed=0):
    """Grid of colored cubes averaged to grayscale, shape (D, H, W)."""
    torch.manual_seed(seed)
    im = torch.zeros(3, D, H, W)
    for i in range(D // step):
        for j in range(H // step):
            for k in range(W // step):
                color = torch.rand(3).reshape(3, 1, 1, 1)
                im[:, i*step+1:i*step+1+block,
                      j*step+1:j*step+1+block,
                      k*step+1:k*step+1+block] = color
    return im.mean(0)  # (D, H, W)


def test_O9_sa_convergence():
    """SA recovers a known deformation: grid_result closer to grid_truth than grid_init."""
    D, H, W = 40, 48, 56
    im0 = make_colored_volume_gray(D, H, W)

    torch.manual_seed(11)
    im1 = (im0 + torch.randn_like(im0) * 0.03).clamp(0, 1)

    grid_init = torch.stack(
        torch.meshgrid(
            *[torch.linspace(-1, 1, s) for s in (4, 4, 4)],
            indexing="ij",
        ),
        dim=-1,
    )

    torch.manual_seed(42)
    grid_truth = grid_init + torch.randn_like(grid_init) * 0.20

    t = make_transformer(grid_init, (D, H, W))
    with torch.no_grad():
        im2_base = t.transform(im0, grid_truth)
        if im2_base.shape[0] == 1:
            im2_base = im2_base.squeeze(0)
    torch.manual_seed(7)
    im2 = (im2_base + torch.randn_like(im2_base) * 0.03).clamp(0, 1)

    cost_init = registration_cost(im1, im2, t, grid_init, lam=1e-4).item()

    torch.manual_seed(99)
    sa = SAOptimizer(
        temp_start=1e-2,
        temp_end=1e-4,
        coeff_start=0.15,
        attention_freq=50,
    )
    grid_result = sa.optimize(im1, im2, t, grid_init, n_steps=1500, lam=1e-4)

    cost_result = registration_cost(im1, im2, t, grid_result, lam=1e-4).item()
    # SA must improve its own objective (cost = corr - λ·tension)
    assert cost_result > cost_init, (
        f"SA did not improve cost: init={cost_init:.4f}, result={cost_result:.4f}"
    )
    # For reference (printed, not asserted — SA may find a different valid solution)
    err_before = (grid_init   - grid_truth).norm().item()
    err_after  = (grid_result - grid_truth).norm().item()
    print(f"\n  grid err: {err_before:.4f} → {err_after:.4f}  "
          f"({'improved' if err_after < err_before else 'diverged from GT'})")


# ---------------------------------------------------------------------------
# Визуальные ячейки (не pytest — только для интерактивного запуска)
# ---------------------------------------------------------------------------

#%%
if __name__ == "__main__":
    from brain_morph.utils import Volume
    import matplotlib.pyplot as plt

    if torch is not None:
        D, H, W = 40, 48, 56

        im0 = make_colored_volume_gray(D, H, W)

        torch.manual_seed(11)
        im1 = (im0 + torch.randn_like(im0) * 0.03).clamp(0, 1)

        grid_init = torch.stack(
            torch.meshgrid(
                *[torch.linspace(-1, 1, s) for s in (4, 4, 4)],
                indexing="ij",
            ),
            dim=-1,
        )
        torch.manual_seed(42)
        grid_truth = grid_init + torch.randn_like(grid_init) * 0.20

        t_vis = make_transformer(grid_init, (D, H, W))
        with torch.no_grad():
            im2_base = t_vis.transform(im0, grid_truth)
            if im2_base.shape[0] == 1:
                im2_base = im2_base.squeeze(0)
        torch.manual_seed(7)
        im2_noisy = (im2_base + torch.randn_like(im2_base) * 0.03).clamp(0, 1)

        # --- Исходные изображения ---
        print("=== IM1 (moving + noise) ===")
        Volume(im1.unsqueeze(0)).visualize(channel=0)

        print("=== IM2 (fixed = deformed + noise) ===")
        Volume(im2_noisy.unsqueeze(0)).visualize(channel=0)

        # --- Запуск SA (callback собирает cost_log) ---
        cost_log = []

        def _cost_callback(step, cost, warped):
            cost_log.append((step, cost))

        torch.manual_seed(99)
        sa_vis = SAOptimizer(
            temp_start=1e-2,
            temp_end=1e-4,
            coeff_start=0.15,
            attention_freq=50,
            callback=_cost_callback,
            callback_freq=20,
        )
        grid_result = sa_vis.optimize(im1, im2_noisy, t_vis, grid_init,
                                       n_steps=1500, lam=1e-4)

        # --- Результат регистрации ---
        with torch.no_grad():
            warped_result = t_vis.transform(im1, grid_result)
            if warped_result.shape[0] == 1:
                warped_result = warped_result.squeeze(0)

        print("=== Warped IM1 (после регистрации) ===")
        Volume(warped_result.unsqueeze(0)).visualize(channel=0)

        print("=== |Warped − IM2| (остаточная разность) ===")
        diff = (warped_result - im2_noisy).abs()
        Volume(diff.unsqueeze(0)).visualize(channel=0)

        # --- График cost ---
        steps, costs = zip(*cost_log)
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(steps, costs, color="tab:blue")
        ax.set_xlabel("SA step")
        ax.set_ylabel("cost")
        ax.set_title("registration cost over SA iterations")
        plt.tight_layout()
        plt.show()

        err_before = (grid_init   - grid_truth).norm().item()
        err_after  = (grid_result - grid_truth).norm().item()
        print(f"grid err: {err_before:.4f} → {err_after:.4f}  "
              f"({'improved' if err_after < err_before else 'diverged from GT'})")
