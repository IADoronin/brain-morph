# Copyright (C) 2026 Ivan Doronin <iadoronin@yandex.ru>
# Tests for streaming.warp_to_file — run BEFORE connecting to __init__.py

import os
import math
import tempfile
import numpy as np
import torch
import cv2 as cv
import pytest


def _make_tiff_series(tmpdir: str, D: int, H: int, W: int) -> str:
    """Create D grayscale TIFF files with a simple gradient pattern."""
    for i in range(D):
        img = np.zeros((H, W), dtype=np.uint16)
        img[:] = (i / max(D - 1, 1)) * 65535
        img[H // 4: 3 * H // 4, W // 4: 3 * W // 4] = 32768
        cv.imwrite(os.path.join(tmpdir, f"slice_{i:04d}.tif"), img)
    return os.path.join(tmpdir, "slice_*.tif")


def _identity_grid(ny: int, nx: int, nz: int) -> torch.Tensor:
    """Identity control-point grid matching _make_regular_grid convention.

    Channel (0,1,2) = (D-src, H-src, W-src) in [-1,1].
    ny maps to D, nx to H, nz to W (same as F.interpolate size=(D,H,W) convention).
    """
    return torch.stack(
        torch.meshgrid(
            torch.linspace(-1, 1, ny),
            torch.linspace(-1, 1, nx),
            torch.linspace(-1, 1, nz),
            indexing="ij",
        ),
        dim=-1,
    )  # (ny, nx, nz, 3)


def test_output_file_created():
    """warp_to_file produces a valid NIfTI file."""
    from brain_morph.utils.streaming import warp_to_file

    with tempfile.TemporaryDirectory() as tmp:
        pattern = _make_tiff_series(tmp, D=10, H=16, W=12)
        grid = _identity_grid(4, 4, 4)
        out = os.path.join(tmp, "out.nii.gz")
        warp_to_file(pattern, grid, out, scale=1, ratio=1.0, chunk_size=4)
        assert os.path.exists(out), "output file not created"
        print("PASS: output file created")


def test_output_shape():
    """Output NIfTI shape matches D_out × H_out × W_out."""
    import nibabel as nib
    from brain_morph.utils.streaming import warp_to_file

    D, H, W = 12, 20, 16
    scale, ratio = 1, 1.0
    z_step = scale / ratio
    D_out = math.ceil(D / z_step)

    with tempfile.TemporaryDirectory() as tmp:
        pattern = _make_tiff_series(tmp, D=D, H=H, W=W)
        grid = _identity_grid(4, 4, 4)
        out = os.path.join(tmp, "out.nii.gz")
        warp_to_file(pattern, grid, out, scale=scale, ratio=ratio, chunk_size=5)
        vol = nib.load(out).get_fdata()
        assert vol.shape == (D_out, H, W), f"shape mismatch: {vol.shape} vs ({D_out},{H},{W})"
        print(f"PASS: shape {vol.shape}")


def test_identity_grid_preserves_content():
    """Identity grid should produce output close to input (within interpolation error)."""
    import nibabel as nib
    from brain_morph.utils.streaming import warp_to_file

    D, H, W = 10, 16, 12

    with tempfile.TemporaryDirectory() as tmp:
        pattern = _make_tiff_series(tmp, D=D, H=H, W=W)
        grid = _identity_grid(5, 5, 5)
        out = os.path.join(tmp, "out.nii.gz")
        warp_to_file(pattern, grid, out, scale=1, ratio=1.0, chunk_size=3)
        vol = nib.load(out).get_fdata().astype(np.float32)

        # Load reference directly
        import glob, re
        paths = sorted(glob.glob(pattern), key=lambda p: int(re.findall(r'\d+', os.path.basename(p))[-1]))
        ref = np.stack([cv.imread(p, cv.IMREAD_UNCHANGED).astype(np.float32) for p in paths], axis=0)

        # Interior (avoid boundary padding artefacts from grid_sample)
        m = 1
        diff = np.abs(vol[m:-m, m:-m, m:-m] - ref[m:-m, m:-m, m:-m])
        max_diff = diff.max()
        rel_err = max_diff / 65535
        print(f"PASS: identity grid max diff = {max_diff:.1f}  rel = {rel_err:.4f}")
        assert rel_err < 0.05, f"identity grid error too large: {rel_err:.4f}"


def test_chunk_size_does_not_affect_result():
    """chunk_size=2 and chunk_size=10 must yield identical output."""
    import nibabel as nib
    from brain_morph.utils.streaming import warp_to_file

    D, H, W = 8, 12, 10

    with tempfile.TemporaryDirectory() as tmp:
        pattern = _make_tiff_series(tmp, D=D, H=H, W=W)
        grid = _identity_grid(4, 4, 4)

        out2 = os.path.join(tmp, "out2.nii.gz")
        out10 = os.path.join(tmp, "out10.nii.gz")
        warp_to_file(pattern, grid, out2,  scale=1, ratio=1.0, chunk_size=2)
        warp_to_file(pattern, grid, out10, scale=1, ratio=1.0, chunk_size=10)

        v2  = nib.load(out2).get_fdata().astype(np.float32)
        v10 = nib.load(out10).get_fdata().astype(np.float32)
        max_diff = np.abs(v2 - v10).max()
        print(f"PASS: chunk invariance max diff = {max_diff:.6f}")
        assert max_diff < 0.02, f"chunk size affects result: max_diff={max_diff}"


def test_scale_downsampling():
    """scale=2 halves H and W dimensions of output."""
    import nibabel as nib
    from brain_morph.utils.streaming import warp_to_file

    D, H, W, scale = 8, 24, 20, 2

    with tempfile.TemporaryDirectory() as tmp:
        pattern = _make_tiff_series(tmp, D=D, H=H, W=W)
        grid = _identity_grid(4, 4, 4)
        out = os.path.join(tmp, "out.nii.gz")
        warp_to_file(pattern, grid, out, scale=scale, ratio=1.0, chunk_size=4)
        vol = nib.load(out).get_fdata()
        assert vol.shape[1] == H // scale, f"H not halved: {vol.shape}"
        assert vol.shape[2] == W // scale, f"W not halved: {vol.shape}"
        print(f"PASS: scale={scale} output shape {vol.shape}")


def test_missing_pattern_raises():
    """FileNotFoundError when pattern matches nothing."""
    from brain_morph.utils.streaming import warp_to_file

    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError):
            warp_to_file("/nonexistent/*.tif", _identity_grid(3, 3, 3),
                         os.path.join(tmp, "x.nii.gz"))
        print("PASS: FileNotFoundError raised")


if __name__ == "__main__":
    tests = [
        test_output_file_created,
        test_output_shape,
        test_identity_grid_preserves_content,
        test_chunk_size_does_not_affect_result,
        test_scale_downsampling,
        test_missing_pattern_raises,
    ]
    for t in tests:
        print(f"\n--- {t.__name__} ---")
        t()
    print("\nAll tests passed.")
