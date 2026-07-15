"""Heatmap rendering for probe images (matplotlib Agg -> wandb.Image)."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402


def latent_grid_figure(maps: torch.Tensor, grid_size, title: str, max_panels: int = 16):
    """maps: [N, M] per-position weights per latent -> one heatmap panel per latent."""
    gh, gw = grid_size
    N, M = maps.shape
    m = min(M, max_panels)
    ncols = min(8, m)
    nrows = math.ceil(m / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.6 * ncols, 1.6 * nrows), squeeze=False)
    data = maps.detach().float().cpu()
    vmin, vmax = float(data.min()), float(data.max())
    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        ax.axis("off")
        if i < m:
            ax.imshow(data[:, i].reshape(gh, gw), vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_title(str(i), fontsize=6)
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    return fig


def position_map_figure(values: torch.Tensor, grid_size, title: str):
    """values: [N] -> single per-position heatmap."""
    gh, gw = grid_size
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    im = ax.imshow(values.detach().float().cpu().reshape(gh, gw), cmap="viridis")
    ax.axis("off")
    ax.set_title(title, fontsize=8)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    return fig


def fig_to_wandb(fig):
    import wandb

    img = wandb.Image(fig)
    plt.close(fig)
    return img
