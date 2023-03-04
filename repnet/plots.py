"""Utility functions for plotting."""
import cv2
import numpy as np
from typing import List



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



def plot_repetitions(frames: List[np.ndarray], counts: List[int]) -> List[np.ndarray]:
    """Generate video with repetition counts and return frames."""
    blue_dark, blue_light = (102, 60, 0), (215, 175, 121)
    h, w, _ = frames[0].shape
    pbar_r = max(int(min(w, h) * 0.1), 20)
    pbar_c = (pbar_r + 5, pbar_r + 5)
    txt_s = pbar_r / 30
    assert len(frames) == len(counts), 'Number of frames and counts must match.'
    # Draw progress bar
    out_frames = []
    for frame, count in zip(frames, counts):
        frame = frame.copy()
        color_bg, color_fg = (blue_dark, blue_light) if int(count) % 2 == 0 else (blue_light, blue_dark)
        frame = cv2.ellipse(frame, pbar_c, (pbar_r, pbar_r), -90, 0, 360, color_bg, -1, cv2.LINE_AA)
        frame = cv2.ellipse(frame, pbar_c, (pbar_r, pbar_r), -90, 0, 360 * (count % 1.0), color_fg, -1, cv2.LINE_AA)
        txt_box, _ = cv2.getTextSize(str(int(count)), cv2.FONT_HERSHEY_SIMPLEX, txt_s, 2)
        txt_c = (pbar_c[0] - txt_box[0] // 2, pbar_c[1] + txt_box[1] // 2)
        frame = cv2.putText(frame, str(int(count)), txt_c, cv2.FONT_HERSHEY_SIMPLEX, txt_s, (255, 255, 255), 2, cv2.LINE_AA)
        out_frames.append(frame)
    return out_frames
