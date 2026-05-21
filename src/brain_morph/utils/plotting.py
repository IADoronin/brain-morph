# Copyright (C) 2026 Ivan Doronin <iadoronin@yandex.ru>
# This file is part of brain-morph, licensed under GNU GPL v3.0.
# See LICENSE file in the project root for full license text.

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

STAGE_COLORS = ['#4C8BF5', '#E8555A', '#2ECC71', '#F5A623']


def plot_cost_stages(
    cost_log: list[tuple[int, float]],
    smooth: bool = True,
    height: float = 3.5,
) -> None:
    """Plot registration cost per stage (V2: separate subplot, independent Y scale).

    Stage boundaries are detected by step counter reset.

    Args:
        cost_log: List of (step, cost) tuples accumulated during optimization.
        smooth:   Draw smoothed line on top of raw trace.
        height:   Figure height in inches.
    """
    if len(cost_log) < 2:
        return

    stages_data: list[list] = []
    current: list = []
    for i, (s, c) in enumerate(cost_log):
        if i > 0 and s < cost_log[i - 1][0]:
            stages_data.append(current)
            current = []
        current.append((s, c))
    stages_data.append(current)

    n = len(stages_data)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, height))
    if n == 1:
        axes = [axes]
    for i, (stage_data, ax) in enumerate(zip(stages_data, axes)):
        steps_, costs_ = zip(*stage_data)
        color = STAGE_COLORS[i % len(STAGE_COLORS)]
        ax.plot(steps_, costs_, color=color, alpha=0.3, linewidth=0.7)
        if smooth and len(costs_) > 20:
            w = max(1, len(costs_) // 20)
            sm = np.convolve(costs_, np.ones(w) / w, mode='same')
            ax.plot(steps_, sm, color=color, linewidth=2)
        ax.set_title(f'Stage {i + 1}', fontsize=10)
        ax.set_xlabel('Step', fontsize=9)
        if i == 0:
            ax.set_ylabel('NCC', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)
    plt.suptitle('Registration cost per stage', fontsize=11, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_overlay(
    warped: torch.Tensor,
    fixed: torch.Tensor,
    step: int | None = None,
) -> None:
    """Mid-slice overlay of two volumes: warped=green, fixed=red (XY / XZ / YZ).

    Accepts (D, H, W) or (C, D, H, W). Automatically resizes fixed to match
    warped when shapes differ (e.g. during coarse registration stages).

    Args:
        warped: Warped moving image.
        fixed:  Fixed (target) image.
        step:   Current optimisation step shown in title (optional).
    """
    w = (warped[0] if warped.dim() == 4 else warped).float()
    f = (fixed[0]  if fixed.dim()  == 4 else fixed).float()

    if f.shape != w.shape:
        f = F.interpolate(
            f.unsqueeze(0).unsqueeze(0),
            size=w.shape, mode='trilinear', align_corners=True,
        ).squeeze()

    suptitle = 'Green=warped moving  Red=fixed'
    if step is not None:
        suptitle = f'step={step}  |  ' + suptitle

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (dim, name) in zip(axes, [(0, 'XY'), (1, 'XZ'), (2, 'YZ')]):
        mid = w.shape[dim] // 2
        sl_w = w.select(dim, mid).cpu()
        sl_f = f.select(dim, mid).cpu()
        rgb = torch.stack([sl_f, sl_w, torch.zeros_like(sl_w)], dim=-1).clamp(0, 1)
        ax.imshow(rgb.numpy(), aspect='auto')
        ax.set_title(name, fontsize=9)
        ax.axis('off')
    plt.suptitle(suptitle, fontsize=9, y=1.01)
    plt.tight_layout()
    plt.show()
