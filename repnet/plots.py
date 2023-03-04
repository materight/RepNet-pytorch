"""Utility functions for plotting."""
import cv2
import numpy as np
import matplotlib as mpl, matplotlib.pyplot as plt
from typing import List


def plot_heatmap(dist: np.ndarray, log_scale: bool = False):
    """Plot the temporal self-similarity matrix into an OpenCV image."""
    np.fill_diagonal(dist, np.nan)
    if log_scale:
        dist = np.log(1 + dist)
    dist = -dist # Invert the distance
    zmin, zmax = np.nanmin(dist), np.nanmax(dist)
    heatmap = (dist - zmin) / (zmax - zmin) # Normalize into [0, 1]
    heatmap = np.nan_to_num(heatmap, nan=1)
    heatmap = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
    return heatmap



def plot_repetitions(frames: List[np.ndarray], counts: List[int], fps: int, out_path: str):
    """Generate video with repetition counts."""
    colormap = plt.cm.PuBu
    sum_counts = np.cumsum(counts)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5), tight_layout=True)
    h, w, _ = np.shape(frames[0])
    wx, wy, wr = 95 / 112 * w, 17 / 112 * h, 15 / 112 * h
    tx, ty, ts = 95 / 112 * w, 19 / 112 * h, 35
    img0 = ax.imshow(frames[0])
    wedge1 = mpl.patches.Wedge((wx, wy), wr, 0, 0, color=colormap(1.))
    wedge2 = mpl.patches.Wedge((wx, wy), wr, 0, 0, color=colormap(0.5))
    ax.add_patch(wedge1)
    ax.add_patch(wedge2)
    txt = ax.text(tx, ty, '0', size=ts, ha='center', va='center', alpha=0.9, color='white')

    def _update(i):
        """Update plot with next frame."""
        img0.set_data(frames[i])
        current_rep_count = int(sum_counts[i])
        wedge1.set_color(colormap(1.0 if current_rep_count % 2 == 0 else 0.5))
        wedge2.set_color(colormap(0.5 if current_rep_count % 2 == 0 else 1.0))
        wedge1.set_theta1(-90)
        wedge1.set_theta2(-90 - 360 * (1 - sum_counts[i] % 1.0))
        wedge2.set_theta1(-90 - 360 * (1 - sum_counts[i] % 1.0))
        wedge2.set_theta2(-90)
        txt.set_text(current_rep_count)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()

    anim = mpl.animation.FuncAnimation(fig, _update, frames=len(frames), interval=1000/fps, blit=False)
    anim.save(out_path, dpi=200)
    plt.close()
