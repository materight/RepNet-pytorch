"""Utility functions for plotting."""
import cv2
import numpy as np
from typing import List, Optional
from sklearn.decomposition import PCA


def plot_heatmap(dist: np.ndarray, log_scale: bool = False) -> np.ndarray:
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


def plot_pca(embeddings: List[np.ndarray]) -> np.ndarray:
    """Plot the 1D PCA of the embeddings into an OpenCV image."""
    projection = PCA(n_components=1).fit_transform(embeddings).flatten()
    projection = (projection - projection.min()) / (projection.max() - projection.min())
    h, w = 200, len(projection) * 4
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    y = ((1 - projection) * h).astype(np.int32)
    x = (np.arange(len(y)) / len(y) * w).astype(np.int32)
    pts = np.stack([x, y], axis=1).reshape((-1, 1, 2))
    img = cv2.polylines(img, [pts], False, (102, 60, 0), 1, cv2.LINE_AA)
    return img


def plot_repetitions(frames: List[np.ndarray], counts: List[float], periodicity: Optional[List[float]]) -> List[np.ndarray]:
    """Generate video with repetition counts and return frames."""
    blue_dark, blue_light = (102, 60, 0), (215, 175, 121)
    h, w, _ = frames[0].shape
    pbar_r = max(int(min(w, h) * 0.1), 20)
    pbar_c = (pbar_r + 5, pbar_r + 5)
    txt_s = pbar_r / 30
    assert len(frames) == len(counts), 'Number of frames and counts must match.'
    out_frames = []
    for i, (frame, count) in enumerate(zip(frames, counts)):
        frame = frame.copy()
        # Draw progress bar
        frame = cv2.ellipse(frame, pbar_c, (pbar_r, pbar_r), -90, 0, 360, blue_dark, -1, cv2.LINE_AA)
        frame = cv2.ellipse(frame, pbar_c, (pbar_r, pbar_r), -90, 0, 360 * (count % 1.0), blue_light, -1, cv2.LINE_AA)
        txt_box, _ = cv2.getTextSize(str(int(count)), cv2.FONT_HERSHEY_SIMPLEX, txt_s, 2)
        txt_c = (pbar_c[0] - txt_box[0] // 2, pbar_c[1] + txt_box[1] // 2)
        frame = cv2.putText(frame, str(int(count)), txt_c, cv2.FONT_HERSHEY_SIMPLEX, txt_s, (255, 255, 255), 2, cv2.LINE_AA)
        # Draw periodicity plot on the right if available
        if periodicity is not None:
            periodicity = np.asarray(periodicity)
            padx, pady, window_size = 5, 10, 64
            pcanvas_h, pcanvas_w = frame.shape[0], min(frame.shape[0], frame.shape[1])
            pcanvas = np.full((pcanvas_h, pcanvas_w, 3), 255, dtype=np.uint8)
            pcanvas[pady::int((pcanvas_h - pady*2) / 10), :, :] = (235, 235, 235) # Draw horizontal grid
            y = ((1 - periodicity[:i+1][-window_size:]) * (pcanvas_h - pady*2) + pady).astype(np.int32)
            x = ((np.arange(len(y)) / window_size) * (pcanvas_w - padx*2)).astype(np.int32)
            pts = np.stack([x, y], axis=1).reshape((-1, 1, 2))
            pcanvas = cv2.polylines(pcanvas, [pts], False, blue_dark, 1, cv2.LINE_AA)
            pcanvas = cv2.circle(pcanvas, (x[-1], y[-1]), 2, (0, 0, 255), -1, cv2.LINE_AA)
            frame = np.concatenate([frame, pcanvas], axis=1)
        out_frames.append(frame)
    return out_frames
