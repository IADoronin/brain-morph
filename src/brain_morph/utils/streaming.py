# Copyright (C) 2026 Ivan Doronin <iadoronin@yandex.ru>
# This file is part of brain-morph, licensed under GNU GPL v3.0.
# See LICENSE file in the project root for full license text.

"""Streaming (chunked) application of a registration grid to large TIFF volumes.

Applies a coarse control-point grid to a full-resolution TIFF series without
loading the entire volume into RAM.  At each step only a small sub-volume is
held in memory; results are streamed to disk via numpy.memmap.

Memory per chunk (chunk_size=50, H=800, W=600, D_sub≈170):
  grid chunk  : ~288 MB
  input chunk : ~326 MB
  output chunk: ~ 96 MB
  total       : ~710 MB   (vs ~9 GB for the full volume)
"""

from __future__ import annotations

import gc
import glob
import math
import os
import re

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sort_key(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    nums = re.findall(r'-?\d+', stem)
    return int(nums[-1]) if nums else 0


def _read_blend(paths: list, idx: float, scale: int, h_out: int, w_out: int) -> np.ndarray:
    """Read one output slice; linearly blend adjacent input slices for fractional index."""
    n = len(paths)
    lo, hi = int(math.floor(idx)), int(math.ceil(idx))

    def _load(i: int) -> np.ndarray:
        img = cv.imread(paths[max(0, min(i, n - 1))], cv.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Cannot read: {paths[max(0, min(i, n-1))]}")
        img = img.astype(np.float32)
        if scale > 1:
            img = cv.resize(img, (w_out, h_out), interpolation=cv.INTER_AREA)
        return img

    if lo == hi or hi >= n:
        return _load(lo)
    t = idx - lo
    return _load(lo) * (1.0 - t) + _load(hi) * t


def _load_d_range(
    paths: list,
    d_out_start: int,
    d_out_end: int,
    scale: int,
    z_step: float,
    h_out: int,
    w_out: int,
) -> torch.Tensor:
    """Load output D-slices [d_out_start:d_out_end) → (1, D, H, W) float32.

    Each output slice d maps to input float index d * z_step (blended).
    """
    slices = []
    for d in range(d_out_start, d_out_end):
        in_float = d * z_step
        if in_float >= len(paths):
            break
        sl = _read_blend(paths, in_float, scale, h_out, w_out)
        slices.append(sl if sl.ndim == 2 else sl[:, :, 0])  # take first channel
    if not slices:
        return torch.zeros(1, 0, h_out, w_out)
    arr = np.stack(slices, axis=0)          # (D, H, W)
    return torch.from_numpy(arr[np.newaxis].copy())   # (1, D, H, W)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def warp_to_file(
    tiff_pattern: str,
    grid: torch.Tensor,
    output_path: str,
    scale: int = 1,
    ratio: float = 1.0,
    chunk_size: int = 50,
) -> None:
    """Apply a registration grid to a full-resolution TIFF series and save to NIfTI.

    The volume is never fully loaded into RAM; only a D-chunk of input slices
    is held at a time.  Output is written incrementally via numpy.memmap.

    The grid must be the *same* coarse grid returned by ``RegistrationPipeline.run``
    (shape ``(ny, nx, nz, 3)``, coordinates in ``[-1, 1]``).

    Args:
        tiff_pattern: Glob pattern for per-slice TIFF files.
        grid:         Coarse control-point grid ``(ny, nx, nz, 3)``.
        output_path:  Destination ``.nii.gz`` file.
        scale:        XY downsampling factor (1 = no downsampling).
        ratio:        XY/Z voxel-size ratio (same as used during registration).
        chunk_size:   Number of output D-slices processed per iteration.
    """
    try:
        import nibabel as nib
    except ImportError as e:
        raise ImportError("nibabel is required: pip install nibabel") from e

    # --- 1. Probe the series ---
    paths = sorted(glob.glob(tiff_pattern), key=_sort_key)
    if not paths:
        raise FileNotFoundError(f"No files matched: {tiff_pattern}")

    probe = cv.imread(paths[0], cv.IMREAD_UNCHANGED)
    if probe is None:
        raise IOError(f"Cannot read: {paths[0]}")
    h, w = probe.shape[:2]
    n_files = len(paths)
    z_step = scale / ratio
    D_out = math.ceil(n_files / z_step)
    H_out = max(1, h // scale)
    W_out = max(1, w // scale)

    print(f"Input:  {n_files} TIFF files")
    print(f"Output: {D_out} × {H_out} × {W_out}  (scale={scale}, ratio={ratio})")

    # --- 2. Pre-allocate output memmap on disk ---
    tmp_path = output_path + ".mmap"
    mmap = np.memmap(tmp_path, dtype="float32", mode="w+",
                     shape=(D_out, H_out, W_out))

    coarse_5d = grid.permute(3, 0, 1, 2).unsqueeze(0).float()  # (1, 3, ny, nx, nz)

    # --- 3. Chunk loop ---
    for d_start in range(0, D_out, chunk_size):
        d_end = min(d_start + chunk_size, D_out)

        # 3a. Compute deformation coords for this output D-chunk
        d_q = torch.linspace(-1, 1, D_out)[d_start:d_end]
        h_q = torch.linspace(-1, 1, H_out)
        w_q = torch.linspace(-1, 1, W_out)
        dd, hh, ww = torch.meshgrid(d_q, h_q, w_q, indexing="ij")
        # Query coarse grid in (x,y,z)=(W,H,D) order expected by F.grid_sample
        query = torch.stack([ww, hh, dd], dim=-1).unsqueeze(0)  # (1,cD,H,W,3)
        with torch.no_grad():
            g = F.grid_sample(
                coarse_5d, query,
                mode="bilinear", align_corners=True, padding_mode="border",
            )  # (1, 3, cD, H, W)  — channels: [D-src, H-src, W-src]
        g = g.squeeze(0).permute(1, 2, 3, 0)  # (cD, H, W, 3)

        # 3b. Which input D-slices are actually sampled?
        d_src_idx = (g[..., 0] + 1.0) / 2.0 * (D_out - 1)  # [0, D_out-1]
        sub_start = max(0, int(d_src_idx.min().item()) - 2)
        sub_end   = min(D_out, int(d_src_idx.max().item()) + 3)
        D_sub = max(sub_end - sub_start, 1)

        # 3c. Load only those input slices
        im_sub = _load_d_range(paths, sub_start, sub_end, scale, z_step, H_out, W_out)
        # im_sub: (1, D_sub, H, W)

        # 3d. Remap D-coordinate from full-volume space to sub-volume space
        d_sub_local = (d_src_idx - sub_start) / max(D_sub - 1, 1) * 2.0 - 1.0

        # Reconstruct grid with remapped D, then flip (D,H,W) → (W,H,D) for grid_sample
        g_sub = torch.stack([d_sub_local, g[..., 1], g[..., 2]], dim=-1)
        g_sub = g_sub.flip(-1)  # → (W-src, H-src, D-sub-local)

        # 3e. Warp sub-volume
        with torch.no_grad():
            warped = F.grid_sample(
                im_sub.unsqueeze(0),   # (1, 1, D_sub, H, W)
                g_sub.unsqueeze(0),    # (1, cD, H, W, 3)
                mode="bilinear", align_corners=True, padding_mode="border",
            ).squeeze()                # (cD, H, W)

        # 3f. Write chunk to disk
        actual = min(d_end - d_start, warped.shape[0] if warped.dim() > 2 else 1)
        if warped.dim() == 2:
            warped = warped.unsqueeze(0)
        mmap[d_start:d_start + actual] = warped[:actual].numpy()

        del im_sub, g, g_sub, warped, d_src_idx, d_sub_local
        gc.collect()
        print(f"  [{d_start:5d}:{d_start+actual:5d}]  sub_range=[{sub_start},{sub_end}]  D_sub={D_sub}")

    # --- 4. Flush memmap and save NIfTI ---
    mmap.flush()
    nib.save(nib.Nifti1Image(mmap, affine=np.eye(4)), output_path)
    del mmap
    os.remove(tmp_path)
    print(f"Saved → {output_path}")
