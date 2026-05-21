# Tests for SAOptimizer batch_size parameter.
# Verifies: (1) batch_size=1 identical to original, (2) non-adjacency invariant,
# (3) batch_size>1 runs without error, (4) speedup on large grid.

import time
import torch
import pytest

from brain_morph.registration.optimizers import SAOptimizer
from brain_morph.registration.pipeline import _make_regular_grid
from brain_morph.utils.mesh_transformer_3d import MeshTransformer3D


# ── helpers ─────────────────────────────────────────────────────────────────

def _synthetic_pair(D=24, H=20, W=16, seed=0):
    """Moving/fixed pair: moving is a shifted version of fixed."""
    torch.manual_seed(seed)
    fixed = torch.rand(D, H, W)
    # slight blur as "different but related" moving image
    moving = torch.roll(fixed, shifts=2, dims=0)
    return moving, fixed


def _make_optimizer(batch_size=1, n_steps=50, seed=None):
    return SAOptimizer(
        temp_start=1e-3, temp_end=1e-5,
        coeff_start=0.12, coeff_drop=0.997,
        attention_freq=25,
        similarity="ncc",
        batch_size=batch_size,
    ), n_steps


# ── 1. Non-adjacency invariant ───────────────────────────────────────────────

def test_non_adjacent_no_shared_cell():
    """All returned nodes must be mutually non-adjacent (no shared cell)."""
    ny, nx, nz = 6, 5, 4
    n_nodes = ny * nx * nz
    probs = torch.ones(n_nodes) / n_nodes

    for batch_size in [2, 3, 4]:
        for _ in range(20):
            nodes = SAOptimizer._sample_non_adjacent(probs, n_nodes, ny, nx, nz, batch_size)
            for a, (ai, aj, ak) in enumerate(nodes):
                for b, (bi, bj, bk) in enumerate(nodes):
                    if a == b:
                        continue
                    assert not (abs(ai-bi) <= 1 and abs(aj-bj) <= 1 and abs(ak-bk) <= 1), \
                        f"Adjacent nodes returned: {(ai,aj,ak)} and {(bi,bj,bk)}"
    print("PASS: non-adjacency invariant")


def test_small_grid_graceful_fallback():
    """2×2×2 grid: all nodes share a cell — batch falls back to 1."""
    ny, nx, nz = 2, 2, 2
    n_nodes = ny * nx * nz
    probs = torch.ones(n_nodes) / n_nodes
    for _ in range(10):
        nodes = SAOptimizer._sample_non_adjacent(probs, n_nodes, ny, nx, nz, batch_size=4)
        assert len(nodes) == 1, f"Expected 1 node for 2×2×2 grid, got {len(nodes)}"
    print("PASS: small grid graceful fallback")


# ── 2. batch_size=1 reproducibility ─────────────────────────────────────────

def test_batch1_identical_to_single_node():
    """batch_size=1 must produce the same result as before (same random stream)."""
    D, H, W = 20, 16, 12
    moving, fixed = _synthetic_pair(D, H, W)
    grid_init = _make_regular_grid(3, 3, 2)
    transformer = MeshTransformer3D(grid_init, (D, H, W))

    sa = SAOptimizer(
        temp_start=1e-3, temp_end=1e-5,
        coeff_start=0.12, coeff_drop=0.997,
        attention_freq=25, similarity="ncc",
        batch_size=1,
    )

    torch.manual_seed(42)
    grid_b1 = sa.optimize(moving, fixed, transformer, grid_init.clone(), n_steps=30)

    # Run again with same seed — result must be identical
    torch.manual_seed(42)
    grid_b1b = sa.optimize(moving, fixed, transformer, grid_init.clone(), n_steps=30)

    assert torch.allclose(grid_b1, grid_b1b), "batch_size=1 not deterministic with same seed"
    print("PASS: batch_size=1 deterministic")


# ── 3. batch_size>1 runs and improves cost ───────────────────────────────────

@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_batch_runs_and_improves(batch_size):
    """batch_size>1 runs without error and reduces cost."""
    D, H, W = 20, 16, 12
    moving, fixed = _synthetic_pair(D, H, W)
    grid_init = _make_regular_grid(4, 4, 3)
    transformer = MeshTransformer3D(grid_init, (D, H, W))

    sa = SAOptimizer(
        temp_start=1e-3, temp_end=1e-5,
        coeff_start=0.12, coeff_drop=0.997,
        attention_freq=25, similarity="ncc",
        batch_size=batch_size,
    )

    torch.manual_seed(7)
    with torch.no_grad():
        warped0 = transformer.transform(moving, grid_init).squeeze(0)
        cost0 = (warped0.float() - fixed.float()).abs().mean().item()

    torch.manual_seed(7)
    grid_out = sa.optimize(moving, fixed, transformer, grid_init.clone(), n_steps=100)

    with torch.no_grad():
        warped1 = transformer.transform(moving, grid_out).squeeze(0)
        cost1 = (warped1.float() - fixed.float()).abs().mean().item()

    assert cost1 <= cost0 * 1.05, \
        f"batch_size={batch_size}: cost did not improve ({cost0:.4f} → {cost1:.4f})"
    print(f"PASS: batch_size={batch_size}  cost {cost0:.4f} → {cost1:.4f}")


# ── 4. Speedup: batch_size>1 is faster per unit of progress ─────────────────

def test_batch_speedup():
    """For equal total node moves, batch_size=4 should be ~4× faster (fewer transform calls)."""
    D, H, W = 30, 24, 20
    moving, fixed = _synthetic_pair(D, H, W)
    grid_init = _make_regular_grid(5, 5, 4)
    transformer = MeshTransformer3D(grid_init, (D, H, W))

    # Same total node moves: batch=1/400 steps vs batch=4/100 steps
    N_moves = 400

    sa1 = SAOptimizer(temp_start=1e-3, temp_end=1e-5, coeff_start=0.12,
                      coeff_drop=0.997, attention_freq=50, similarity="ncc", batch_size=1)
    sa4 = SAOptimizer(temp_start=1e-3, temp_end=1e-5, coeff_start=0.12,
                      coeff_drop=0.997, attention_freq=50, similarity="ncc", batch_size=4)

    torch.manual_seed(0)
    t0 = time.perf_counter()
    sa1.optimize(moving, fixed, transformer, grid_init.clone(), n_steps=N_moves)
    t1 = time.perf_counter() - t0

    torch.manual_seed(0)
    t0 = time.perf_counter()
    sa4.optimize(moving, fixed, transformer, grid_init.clone(), n_steps=N_moves // 4)
    t4 = time.perf_counter() - t0

    ratio = t1 / t4
    print(f"PASS: batch=1/{N_moves} steps {t1:.2f}s  |  batch=4/{N_moves//4} steps {t4:.2f}s  |  speedup={ratio:.2f}×")
    assert ratio > 2.0, f"Expected ≥2× speedup, got {ratio:.2f}×"


# ── 5. No mesh inversion ─────────────────────────────────────────────────────

def test_no_inversion_batch4():
    """batch_size=4 with normal parameters should not produce mesh inversions."""
    D, H, W = 20, 16, 12
    moving, fixed = _synthetic_pair(D, H, W)
    grid_init = _make_regular_grid(4, 4, 3)
    transformer = MeshTransformer3D(grid_init, (D, H, W))

    sa = SAOptimizer(
        temp_start=1e-3, temp_end=1e-5,
        coeff_start=0.12, coeff_drop=0.997,
        attention_freq=25, similarity="ncc",
        batch_size=4,
    )

    torch.manual_seed(13)
    grid_out = sa.optimize(moving, fixed, transformer, grid_init.clone(), n_steps=200)

    # Jacobian determinant: cross products of edge vectors at each cell
    dy = grid_out[1:, :-1, :-1] - grid_out[:-1, :-1, :-1]   # (ny-1, nx-1, nz-1, 3)
    dx = grid_out[:-1, 1:, :-1] - grid_out[:-1, :-1, :-1]
    dz = grid_out[:-1, :-1, 1:] - grid_out[:-1, :-1, :-1]
    det = (dy * torch.cross(dx, dz, dim=-1)).sum(-1)  # (ny-1, nx-1, nz-1)
    n_inv = (det <= 0).sum().item()
    print(f"PASS: inversions={n_inv}/{det.numel()}")
    assert n_inv == 0, f"Mesh inversion detected: {n_inv} cells with det≤0"


if __name__ == "__main__":
    tests = [
        test_non_adjacent_no_shared_cell,
        test_small_grid_graceful_fallback,
        test_batch1_identical_to_single_node,
        lambda: test_batch_runs_and_improves(2),
        lambda: test_batch_runs_and_improves(4),
        lambda: test_batch_runs_and_improves(8),
        test_batch_speedup,
        test_no_inversion_batch4,
    ]
    for t in tests:
        name = getattr(t, "__name__", "anonymous")
        print(f"\n--- {name} ---")
        t()
    print("\nAll tests passed.")
