"""
RGBD Preprocessing Module
==========================
Converts raw RGBD frames into VLM-ready representations.

Supports three depth strategies:
  - "hha"           : Encode depth as pseudo-RGB HHA image (Gupta et al. 2014)
  - "depth_channel" : Concatenate normalised depth as 4th channel
  - "depth_as_text" : Extract object distances / surface normals as text string

Also computes per-frame surface normal maps and optical flow for downstream
keyframe extraction.
"""

import base64
import io
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from config import RGBDConfig

log = logging.getLogger(__name__)


# ── HHA Encoding ─────────────────────────────────────────────────────────────

def depth_to_hha(
    depth_raw: np.ndarray,
    cfg: RGBDConfig,
) -> np.ndarray:
    """
    Encode a depth map as an HHA image (H=horizontal disparity,
    H=height above ground, A=angle with gravity).

    Returns uint8 RGB image of same H×W as depth_raw.
    """
    depth_m = depth_raw.astype(np.float32) * cfg.depth_scale
    depth_m = np.clip(depth_m, cfg.depth_min, cfg.depth_max)

    h, w = depth_m.shape

    # ── Channel 1: Horizontal disparity (inverse depth, normalised) ────────
    disparity = 1.0 / (depth_m + 1e-6)
    disparity_norm = _norm_u8(disparity)

    # ── Channel 2: Height above ground ────────────────────────────────────
    # Approximate by projecting each pixel to 3D and taking the Y coordinate.
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    z = depth_m
    y_3d = (y_coords - cfg.cy) * z / cfg.fy       # world Y (up-axis inverted)
    height_norm = _norm_u8(-y_3d)                  # higher pixel → brighter

    # ── Channel 3: Angle with gravity ─────────────────────────────────────
    normals = _compute_surface_normals(depth_m, cfg)
    gravity = np.array([0.0, -1.0, 0.0])
    cos_angle = np.abs(np.einsum("hwc,c->hw", normals, gravity))
    angle_norm = (cos_angle * 255).astype(np.uint8)

    hha = np.stack([disparity_norm, height_norm, angle_norm], axis=-1)
    return hha


def _compute_surface_normals(depth_m: np.ndarray, cfg: RGBDConfig) -> np.ndarray:
    """Estimate per-pixel surface normals from a metric depth map."""
    h, w = depth_m.shape
    normals = np.zeros((h, w, 3), dtype=np.float32)

    # Build 3-D point cloud
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    x3 = (x_coords - cfg.cx) * depth_m / cfg.fx
    y3 = (y_coords - cfg.cy) * depth_m / cfg.fy
    z3 = depth_m
    pts = np.stack([x3, y3, z3], axis=-1)          # (H, W, 3)

    # Cross-product of neighbouring vectors
    dzdx = np.gradient(pts, axis=1)
    dzdy = np.gradient(pts, axis=0)
    n = np.cross(dzdx, dzdy)

    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    normals = n / (norm + 1e-8)
    return normals


def _norm_u8(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)


# ── Depth-as-Text ─────────────────────────────────────────────────────────────

def depth_to_text(
    depth_raw: np.ndarray,
    rgb: np.ndarray,
    cfg: RGBDConfig,
    n_regions: int = 9,
) -> str:
    """
    Summarise depth information as a compact text string.
    Divides the frame into a 3×3 grid and reports the median depth
    and dominant surface orientation per cell.

    Returns a single-line string suitable for inclusion in the VLM prompt.
    """
    depth_m = depth_raw.astype(np.float32) * cfg.depth_scale
    depth_m = np.clip(depth_m, cfg.depth_min, cfg.depth_max)
    h, w = depth_m.shape
    rows, cols = 3, 3
    cell_h, cell_w = h // rows, w // cols

    descriptions = []
    labels = ["top-left", "top-center", "top-right",
              "mid-left", "center",     "mid-right",
              "bot-left", "bot-center", "bot-right"]

    for i in range(rows):
        for j in range(cols):
            cell = depth_m[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            valid = cell[cell > cfg.depth_min]
            if valid.size == 0:
                descriptions.append(f"{labels[i*cols+j]}:no-data")
            else:
                med = np.median(valid)
                descriptions.append(f"{labels[i*cols+j]}:{med:.2f}m")

    return "depth_grid[" + ", ".join(descriptions) + "]"


# ── Frame Fusion Entry Point ───────────────────────────────────────────────────

def fuse_rgbd_frame(
    rgb: np.ndarray,
    depth_raw: np.ndarray,
    cfg: RGBDConfig,
) -> dict:
    """
    Fuse an RGB + raw depth frame into VLM-ready representations.

    Returns a dict with keys:
      - "pil_image"    : PIL Image ready for VLM (RGB or HHA side-by-side)
      - "depth_text"   : depth summary string (always computed)
      - "depth_m"      : metric depth map (float32 H×W)
      - "surface_normals": normal map (float32 H×W×3)
    """
    depth_m = depth_raw.astype(np.float32) * cfg.depth_scale
    depth_m = np.clip(depth_m, cfg.depth_min, cfg.depth_max)

    if cfg.depth_strategy == "hha":
        hha = depth_to_hha(depth_raw, cfg)
        # Side-by-side: RGB | HHA
        combined = np.concatenate([rgb, hha], axis=1)
        pil = Image.fromarray(combined.astype(np.uint8))

    elif cfg.depth_strategy == "depth_channel":
        depth_norm = _norm_u8(depth_m)
        # Returns RGBA where A = depth
        rgba = np.dstack([rgb, depth_norm])
        # VLMs accept 3-channel; encode depth as blue channel overlay
        blended = rgb.copy()
        blended[:, :, 2] = (0.5 * blended[:, :, 2] + 0.5 * depth_norm).astype(np.uint8)
        pil = Image.fromarray(blended.astype(np.uint8))

    else:  # depth_as_text fallback — just pass RGB
        pil = Image.fromarray(rgb.astype(np.uint8))

    depth_text = depth_to_text(depth_raw, rgb, cfg)
    normals = _compute_surface_normals(depth_m, cfg)

    return {
        "pil_image": pil,
        "depth_text": depth_text,
        "depth_m": depth_m,
        "surface_normals": normals,
    }


# ── Image → Base64 for API ────────────────────────────────────────────────────

def pil_to_base64(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    """Encode a PIL image as a base64 string for Ollama vision API."""
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Video / Frame Loader ───────────────────────────────────────────────────────

def load_rgbd_frame(
    rgb_path: Optional[str] = None,
    depth_path: Optional[str] = None,
    rgb_array: Optional[np.ndarray] = None,
    depth_array: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load an RGBD frame from file paths or pre-loaded arrays.
    Depth files are expected to be 16-bit PNG (millimetres, RealSense standard).
    Returns (rgb uint8 H×W×3, depth_raw uint16 H×W).
    """
    if rgb_array is not None:
        rgb = rgb_array
    elif rgb_path:
        rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Provide either rgb_path or rgb_array")

    if depth_array is not None:
        depth = depth_array
    elif depth_path:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH).astype(np.uint16)
    else:
        log.warning("No depth provided — using zero depth map")
        depth = np.zeros(rgb.shape[:2], dtype=np.uint16)

    return rgb, depth


def iter_video_frames(
    video_path: str,
    depth_dir: Optional[str] = None,
    step: int = 1,
):
    """
    Iterate over frames of an RGB video file, optionally pairing with
    depth images stored as 16-bit PNGs in depth_dir (named frame_XXXXXX.png).

    Yields (frame_idx, rgb_array, depth_array) tuples.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0

    depth_dir_path = Path(depth_dir) if depth_dir else None

    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            depth = np.zeros(rgb.shape[:2], dtype=np.uint16)
            if depth_dir_path:
                candidates = [
                    depth_dir_path / f"frame_{frame_idx:06d}.png",
                    depth_dir_path / f"{frame_idx:06d}.png",
                    depth_dir_path / f"depth_{frame_idx:06d}.png",
                ]
                for c in candidates:
                    if c.exists():
                        depth = cv2.imread(str(c), cv2.IMREAD_ANYDEPTH).astype(np.uint16)
                        break

            yield frame_idx, fps, rgb, depth
        frame_idx += 1

    cap.release()
