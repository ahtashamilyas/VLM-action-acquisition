"""
Keyframe Extraction Module
===========================
Detects action boundaries from RGBD video using optical-flow-based
end-effector velocity estimation, then selects representative keyframes
for each detected action segment.

Algorithm (inspired by SeeDo, Wang et al. 2024):
  1. Compute dense optical flow between consecutive frames.
  2. Compute per-frame mean flow magnitude (proxy for motion velocity).
  3. Detect velocity minima → action boundary candidates.
  4. Filter boundaries by minimum inter-boundary gap.
  5. Sample keyframes within each segment (start / mid / end).
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.signal import find_peaks

from config import KeyframeConfig

log = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class ActionSegment:
    """A temporal segment corresponding to one candidate action."""
    start_frame: int
    end_frame: int
    keyframe_indices: list[int]            # indices into the full frame list
    mean_velocity: float                   # average optical flow magnitude
    depth_change: float                    # average depth-map change in segment


@dataclass
class KeyframeBundle:
    """All data associated with one selected keyframe."""
    frame_idx: int
    timestamp_s: float
    rgb: np.ndarray                        # H×W×3 uint8
    depth_raw: np.ndarray                  # H×W uint16
    fused_pil: object                      # PIL Image (from rgbd_preprocessing)
    depth_text: str
    velocity: float
    segment_id: int


# ── Optical Flow ─────────────────────────────────────────────────────────────

def compute_optical_flow_magnitude(
    gray_prev: np.ndarray,
    gray_curr: np.ndarray,
) -> float:
    """
    Compute Farneback dense optical flow and return the mean
    L2 magnitude (normalised by image diagonal).
    """
    flow = cv2.calcOpticalFlowFarneback(
        gray_prev, gray_curr,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    h, w = gray_curr.shape
    diagonal = np.sqrt(h**2 + w**2)
    return float(mag.mean() / diagonal)


def compute_depth_change(
    depth_prev: np.ndarray,
    depth_curr: np.ndarray,
    scale: float = 0.001,
) -> float:
    """
    Mean absolute depth change between two frames (in metres).
    Ignores zero (invalid) depth pixels.
    """
    d_prev = depth_prev.astype(np.float32) * scale
    d_curr = depth_curr.astype(np.float32) * scale
    mask = (d_prev > 0) & (d_curr > 0)
    if mask.sum() == 0:
        return 0.0
    return float(np.abs(d_curr[mask] - d_prev[mask]).mean())


# ── Boundary Detection ────────────────────────────────────────────────────────

def detect_action_boundaries(
    velocity_series: np.ndarray,
    cfg: KeyframeConfig,
) -> list[int]:
    """
    Given a 1-D velocity signal, return frame indices that correspond
    to action boundaries (local minima of velocity, i.e. pauses between actions).

    Strategy:
      - Smooth with a Gaussian kernel.
      - Find minima that are below the threshold.
      - Enforce minimum gap between boundaries.
    """
    # Smooth signal
    kernel_size = max(3, len(velocity_series) // 20 | 1)   # odd number
    smoothed = _gaussian_smooth(velocity_series, kernel_size)

    # Find valleys (minima of velocity = transition between actions)
    inverted = -smoothed
    peaks, props = find_peaks(
        inverted,
        distance=cfg.min_action_gap_frames,
        prominence=cfg.velocity_threshold * 0.5,
    )

    # Also include the very start and end
    boundaries = sorted(set([0] + peaks.tolist() + [len(velocity_series) - 1]))
    return boundaries


def _gaussian_smooth(signal: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv2.getGaussianKernel(kernel_size, kernel_size / 4.0).flatten()
    return np.convolve(signal, kernel, mode="same")


# ── Keyframe Sampler ──────────────────────────────────────────────────────────

def sample_keyframes_in_segment(
    start: int,
    end: int,
    n: int,
) -> list[int]:
    """
    Sample n keyframe indices uniformly within [start, end] (inclusive).
    Always includes the midpoint.
    """
    if end <= start:
        return [start]
    if n == 1:
        return [(start + end) // 2]
    indices = np.linspace(start, end, n, dtype=int).tolist()
    mid = (start + end) // 2
    if mid not in indices:
        indices[len(indices) // 2] = mid
    return sorted(set(indices))


# ── Main Extractor ────────────────────────────────────────────────────────────

class KeyframeExtractor:
    """
    Processes a stream of RGBD frames and returns ActionSegments with
    associated keyframe bundles.

    Usage:
        extractor = KeyframeExtractor(cfg)
        extractor.ingest(frame_idx, fps, rgb, depth, fused)
        segments = extractor.finalize()
    """

    def __init__(self, cfg: KeyframeConfig, depth_scale: float = 0.001):
        self.cfg = cfg
        self.depth_scale = depth_scale

        # Internal buffers
        self._frames: list[dict] = []            # raw frame store
        self._velocities: list[float] = []
        self._depth_changes: list[float] = []
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_depth: Optional[np.ndarray] = None

    def ingest(
        self,
        frame_idx: int,
        fps: float,
        rgb: np.ndarray,
        depth: np.ndarray,
        fused: dict,                             # output of fuse_rgbd_frame
    ):
        """Add one frame to the extractor's buffer."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        if self._prev_gray is not None:
            vel = compute_optical_flow_magnitude(self._prev_gray, gray)
            dc = compute_depth_change(self._prev_depth, depth, self.depth_scale)
        else:
            vel, dc = 0.0, 0.0

        self._velocities.append(vel)
        self._depth_changes.append(dc)
        self._frames.append({
            "frame_idx": frame_idx,
            "timestamp_s": frame_idx / fps,
            "rgb": rgb,
            "depth": depth,
            "fused": fused,
            "velocity": vel,
        })

        self._prev_gray = gray
        self._prev_depth = depth

    def finalize(self) -> list[ActionSegment]:
        """
        Detect boundaries, build segments, select keyframes.
        Returns list of ActionSegment objects.
        """
        if len(self._frames) < 2:
            log.warning("Too few frames to detect actions")
            return []

        vel_arr = np.array(self._velocities)
        dc_arr = np.array(self._depth_changes)

        # Velocity-based boundary detection
        boundaries = detect_action_boundaries(vel_arr, self.cfg)
        log.info(f"Detected {len(boundaries)-1} action segments from {len(self._frames)} frames")

        segments = []
        for seg_id, (b_start, b_end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if b_end - b_start < 2:
                continue

            kf_indices = sample_keyframes_in_segment(
                b_start, b_end,
                self.cfg.max_keyframes_per_action
            )
            mean_vel = float(vel_arr[b_start:b_end].mean())
            mean_dc = float(dc_arr[b_start:b_end].mean())

            seg = ActionSegment(
                start_frame=self._frames[b_start]["frame_idx"],
                end_frame=self._frames[b_end]["frame_idx"],
                keyframe_indices=kf_indices,
                mean_velocity=mean_vel,
                depth_change=mean_dc,
            )
            segments.append(seg)

        return segments

    def get_keyframe_bundle(self, local_idx: int, seg_id: int) -> KeyframeBundle:
        """Retrieve a KeyframeBundle for a local frame index."""
        f = self._frames[local_idx]
        return KeyframeBundle(
            frame_idx=f["frame_idx"],
            timestamp_s=f["timestamp_s"],
            rgb=f["rgb"],
            depth_raw=f["depth"],
            fused_pil=f["fused"]["pil_image"],
            depth_text=f["fused"]["depth_text"],
            velocity=f["velocity"],
            segment_id=seg_id,
        )

    def get_all_bundles(
        self, segments: list[ActionSegment]
    ) -> dict[int, list[KeyframeBundle]]:
        """
        Return {segment_id: [KeyframeBundle, ...]} for all segments.
        """
        result = {}
        for seg_id, seg in enumerate(segments):
            result[seg_id] = [
                self.get_keyframe_bundle(li, seg_id)
                for li in seg.keyframe_indices
            ]
        return result
