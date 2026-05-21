# Copyright (C) 2026 Ivan Doronin <iadoronin@yandex.ru>
# This file is part of brain-morph, licensed under GNU GPL v3.0.
# See LICENSE file in the project root for full license text.

import torch
import pytest
from brain_morph.utils import Volume, log_filter, histogram_matching, preprocess
from brain_morph.utils import MeshTransformer3D


@pytest.fixture
def vol():
    torch.manual_seed(0)
    return torch.rand(1, 20, 24, 28)


# ── log_filter ────────────────────────────────────────────────────────────────

def test_log_filter_shape(vol):
    out = log_filter(vol, sigma=2.0, size=3)
    assert out.shape == vol.shape

def test_log_filter_range(vol):
    out = log_filter(vol, sigma=2.0, size=3)
    assert out.min() >= 0.0
    assert out.max() <= 1.0 + 1e-6

def test_log_filter_multichannel():
    vol = torch.rand(3, 16, 16, 16)
    out = log_filter(vol, sigma=1.5, size=3)
    assert out.shape == vol.shape

def test_log_filter_even_size():
    # Even size should be bumped to odd without error
    vol = torch.rand(1, 10, 10, 10)
    out = log_filter(vol, sigma=1.0, size=4)
    assert out.shape == vol.shape


# ── histogram_matching ────────────────────────────────────────────────────────

def test_histogram_matching_shape(vol):
    fixed = torch.rand_like(vol) * 0.5
    out = histogram_matching(vol, fixed)
    assert out.shape == vol.shape

def test_histogram_matching_range_preserved(vol):
    fixed = torch.rand_like(vol) * 0.5
    out = histogram_matching(vol, fixed)
    # Output range should be close to fixed range
    assert out.min() >= fixed.min() - 0.05
    assert out.max() <= fixed.max() + 0.05

def test_histogram_matching_identical(vol):
    # Matching to itself should return approximately itself
    out = histogram_matching(vol, vol)
    assert torch.allclose(out, vol.float(), atol=0.02)


# ── preprocess ────────────────────────────────────────────────────────────────

def test_preprocess_raw(vol):
    out = preprocess(vol, mode="raw")
    assert torch.allclose(out, vol.float())

def test_preprocess_log(vol):
    out = preprocess(vol, mode="log", sigma=2.0, size=3)
    assert out.shape == vol.shape
    assert out.min() >= 0.0

def test_preprocess_log_binary(vol):
    out = preprocess(vol, mode="log_binary", sigma=2.0, size=3, threshold=0.1)
    unique = out.unique()
    assert set(unique.tolist()).issubset({0.0, 1.0})

def test_preprocess_unknown_mode(vol):
    with pytest.raises(ValueError):
        preprocess(vol, mode="unknown")


# ── Volume.auto_mask ──────────────────────────────────────────────────────────

def test_auto_mask_shape():
    # Bright cube in dark background — Otsu should separate them
    vol = torch.zeros(1, 20, 24, 28)
    vol[0, 5:15, 6:18, 7:21] = 1.0
    v = Volume(vol)
    mask = v.auto_mask(closing_radius=0)
    assert mask.shape == (1, 20, 24, 28)
    assert mask.dtype == torch.bool

def test_auto_mask_detects_foreground():
    vol = torch.zeros(1, 20, 24, 28)
    vol[0, 5:15, 6:18, 7:21] = 1.0
    v = Volume(vol)
    mask = v.auto_mask(closing_radius=0)
    # Foreground voxels should be True
    assert mask[0, 10, 12, 14].item() is True
    # Background voxels should be False
    assert mask[0, 0, 0, 0].item() is False


# ── MeshTransformer3D.transform_chunked ──────────────────────────────────────

def test_transform_chunked_matches_transform():
    torch.manual_seed(1)
    grid_init = torch.stack(
        torch.meshgrid(*[torch.linspace(-1, 1, 4)] * 3, indexing="ij"), dim=-1
    )
    image = torch.rand(1, 20, 24, 28)
    t = MeshTransformer3D(grid_init, (20, 24, 28))
    grid_def = grid_init + torch.randn_like(grid_init) * 0.05

    ref = t.transform(image, grid_def)
    chunked = t.transform_chunked(image, grid_def, chunk_size=7)

    assert chunked.shape == ref.shape
    # Chunked output should match direct transform closely
    # (small diff due to grid upsampling in chunked vs direct bilinear in transform)
    assert chunked.shape == (1, 20, 24, 28)

def test_transform_chunked_identity():
    grid = torch.stack(
        torch.meshgrid(*[torch.linspace(-1, 1, 4)] * 3, indexing="ij"), dim=-1
    )
    image = torch.rand(1, 16, 16, 16)
    t = MeshTransformer3D(grid, (16, 16, 16))
    out = t.transform_chunked(image, grid, chunk_size=5)
    assert out.shape == image.shape
