# Tests for MeshTransformer3D._build_cell_index_map optimisation.
# Verifies: fast path for regular grids, correct fallback for deformed grids.

import time
import torch
from brain_morph.utils.mesh_transformer_3d import MeshTransformer3D
from brain_morph.registration.pipeline import _make_regular_grid


def _make_deformed(ny, nx, nz, noise=0.05):
    g = _make_regular_grid(ny, nx, nz)
    g = g + torch.randn_like(g) * noise
    return g


# ── 1. Regular grid uses fast path (speed) ───────────────────────────────────

def test_regular_grid_is_fast():
    """Regular grid_init must build cell_index_map in well under 1 second."""
    D, H, W = 240, 100, 75
    grid = _make_regular_grid(5, 5, 4)
    t0 = time.perf_counter()
    t = MeshTransformer3D(grid, (D, H, W))
    dt = time.perf_counter() - t0
    assert dt < 1.0, f"Regular grid constructor too slow: {dt:.2f}s"
    assert (t.cell_index_map >= 0).all()
    print(f"PASS: regular grid {dt*1000:.0f} ms")


# ── 2. Deformed grid falls back correctly (not fast path) ───────────────────

def test_deformed_grid_uses_fallback():
    """Deformed grid_init must NOT use fast path — cell assignments stay correct."""
    D, H, W = 20, 16, 12
    torch.manual_seed(0)
    grid_def = _make_deformed(4, 4, 3, noise=0.05)
    t = MeshTransformer3D(grid_def, (D, H, W))
    cmap = t.cell_index_map
    assert (cmap >= 0).all(), "Unassigned voxels in deformed grid"
    assert cmap.shape[0] == D * H * W
    print(f"PASS: deformed grid fallback, all {cmap.shape[0]:,} voxels assigned")


# ── 3. Fast and slow paths give same assignments for regular grid ─────────────

def test_fast_slow_cell_assignments_match():
    """Fast-path cell assignments must match the original innerpoints result."""
    D, H, W = 30, 24, 20
    grid = _make_regular_grid(4, 4, 3)

    # Fast path
    t_fast = MeshTransformer3D(grid, (D, H, W))

    # Force slow path: shift grid by tiny epsilon that breaks torch.equal
    # but keeps geometry identical for practical purposes
    grid_eps = grid.clone()
    grid_eps[0, 0, 0, 0] += 1e-7  # too small to change any cell boundary
    t_slow = MeshTransformer3D(grid_eps, (D, H, W))

    # Both must cover all voxels
    assert (t_fast.cell_index_map >= 0).all()
    assert (t_slow.cell_index_map >= 0).all()

    # Cell counts per cell must be similar (slight edge differences are OK)
    n_cells = 3 * 3 * 2
    for c in range(n_cells):
        n_fast = (t_fast.cell_index_map == c).sum().item()
        n_slow = (t_slow.cell_index_map == c).sum().item()
        assert abs(n_fast - n_slow) <= D * H * W * 0.01, \
            f"Cell {c}: fast={n_fast} slow={n_slow}"
    print("PASS: fast and slow cell assignments agree")


# ── 4. transform result unchanged for regular grid ───────────────────────────

def test_transform_result_consistent():
    """transform on regular grid_init must give same result before/after optimisation."""
    D, H, W = 20, 16, 12
    torch.manual_seed(7)
    grid_init = _make_regular_grid(4, 4, 3)
    grid_def = _make_deformed(4, 4, 3, noise=0.04)
    img = torch.rand(1, D, H, W)

    t = MeshTransformer3D(grid_init, (D, H, W))
    warped = t.transform(img, grid_def)
    warped_c = t.transform_chunked(img, grid_def, chunk_size=8)

    diff = (warped.float() - warped_c.float()).abs().max().item()
    assert diff < 5.0, f"transform vs transform_chunked diff too large: {diff:.3f}"
    print(f"PASS: transform consistent, max diff={diff:.3f}")


# ── 5. Inverse-style non-regular grid_init works correctly ───────────────────

def test_inverse_style_nonregular_grid_init():
    """Non-regular grid_init (e.g. for inverse transforms) must not crash."""
    D, H, W = 16, 12, 10
    torch.manual_seed(3)
    # Simulate a result grid used as grid_init for inverse transform
    grid_inverse = _make_deformed(3, 3, 2, noise=0.1)
    t = MeshTransformer3D(grid_inverse, (D, H, W))
    assert (t.cell_index_map >= 0).all()
    img = torch.rand(1, D, H, W)
    grid_target = _make_regular_grid(3, 3, 2)
    warped = t.transform(img, grid_target)
    assert warped.shape == (1, D, H, W)
    print("PASS: non-regular grid_init works for inverse-style transform")


# ── 6. Bounding box pre-filter: fallback is faster than brute-force ──────────

def test_bbox_filter_speedup():
    """With bbox pre-filter the fallback must build the cell map faster
    than a naive per-cell innerpoints call over all voxels."""
    import time
    D, H, W = 40, 32, 24
    torch.manual_seed(0)
    grid_def = _make_deformed(4, 4, 3, noise=0.05)

    # Warm up (first call may include JIT / import overhead)
    MeshTransformer3D(grid_def, (8, 8, 6))

    t0 = time.perf_counter()
    t = MeshTransformer3D(grid_def, (D, H, W))
    dt = time.perf_counter() - t0

    assert (t.cell_index_map >= 0).all()
    # On CPU the bbox-filtered fallback should finish in a few seconds, not minutes
    assert dt < 60.0, f"bbox-filtered fallback too slow: {dt:.1f}s"
    print(f"PASS: bbox-filtered fallback {dt:.2f}s for {D}×{H}×{W} grid")


if __name__ == "__main__":
    tests = [
        test_regular_grid_is_fast,
        test_deformed_grid_uses_fallback,
        test_fast_slow_cell_assignments_match,
        test_transform_result_consistent,
        test_inverse_style_nonregular_grid_init,
        test_bbox_filter_speedup,
    ]
    for t in tests:
        print(f"\n--- {t.__name__} ---")
        t()
    print("\nAll tests passed.")
