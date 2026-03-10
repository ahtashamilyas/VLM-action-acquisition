"""
REFLECT Dataset Loader
========================
Loads the REFLECT benchmark dataset from Columbia University.
Dataset: https://www.cs.columbia.edu/~liuzeyi/reflect_data/

Expected directory structure (after download and extraction):
  reflect_data/
    <task_name>/
      <episode_id>/
        rgb/
          frame_000000.png
          frame_000001.png
          ...
        depth/
          frame_000000.png   (16-bit PNG, millimetres)
          ...
        metadata.json         (optional: fps, task_description, failure_timestep)

If the dataset is not available locally, a synthetic generator is provided
for testing the pipeline end-to-end without real data.
"""

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class REFLECTEpisode:
    task_name: str
    episode_id: str
    rgb_dir: Path
    depth_dir: Optional[Path]
    fps: float
    task_description: str
    failure_timestep: Optional[float]   # seconds (ground truth, if available)
    metadata: dict


# ── Dataset Loader ────────────────────────────────────────────────────────────

class REFLECTDataset:
    """
    Iterates over episodes in the REFLECT benchmark dataset.
    """

    KNOWN_TASKS = [
        "boilWater", "sauteeCarrot", "appleInFridge",
        "secureObjects", "heatPotato", "makeCoffee", "putAppleBowl"
    ]

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"REFLECT dataset not found at {data_root}.\n"
                "Download from: https://www.cs.columbia.edu/~liuzeyi/reflect_data/\n"
                "Or use SyntheticREFLECTDataset for testing."
            )

    def list_episodes(self) -> list[REFLECTEpisode]:
        episodes = []
        for task_dir in sorted(self.data_root.iterdir()):
            if not task_dir.is_dir():
                continue
            for ep_dir in sorted(task_dir.iterdir()):
                if not ep_dir.is_dir():
                    continue
                rgb_dir = ep_dir / "rgb"
                depth_dir = ep_dir / "depth" if (ep_dir / "depth").exists() else None
                meta_path = ep_dir / "metadata.json"
                meta = {}
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)

                episodes.append(REFLECTEpisode(
                    task_name=task_dir.name,
                    episode_id=ep_dir.name,
                    rgb_dir=rgb_dir,
                    depth_dir=depth_dir,
                    fps=float(meta.get("fps", 30.0)),
                    task_description=meta.get(
                        "task_description",
                        f"robot performing task: {task_dir.name}"
                    ),
                    failure_timestep=meta.get("failure_timestep"),
                    metadata=meta,
                ))
        log.info(f"Found {len(episodes)} episodes in {self.data_root}")
        return episodes

    def iter_frames(
        self,
        episode: REFLECTEpisode,
        step: int = 1,
    ) -> Iterator[tuple[int, float, np.ndarray, np.ndarray]]:
        """
        Yields (frame_idx, fps, rgb uint8 H×W×3, depth uint16 H×W)
        for each frame in the episode.
        """
        rgb_files = sorted(episode.rgb_dir.glob("*.png")) + \
                    sorted(episode.rgb_dir.glob("*.jpg"))
        for i, rgb_path in enumerate(rgb_files):
            if i % step != 0:
                continue
            rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
            depth = np.zeros(rgb.shape[:2], dtype=np.uint16)
            if episode.depth_dir:
                depth_path = episode.depth_dir / rgb_path.name
                if depth_path.exists():
                    depth = cv2.imread(
                        str(depth_path), cv2.IMREAD_ANYDEPTH
                    ).astype(np.uint16)
            yield i, episode.fps, rgb, depth


# ── Synthetic Dataset (for testing without real data) ─────────────────────────

class SyntheticREFLECTDataset:
    """
    Generates synthetic RGBD frame sequences that mimic the REFLECT dataset
    structure, allowing the full pipeline to be tested without downloading
    the actual dataset.

    Each episode simulates a sequence of action phases:
      PICK_AND_PLACE → PREPARING → CUTTING → COOK_COOL
    """

    def __init__(
        self,
        n_episodes: int = 3,
        frames_per_episode: int = 120,
        fps: float = 10.0,
        resolution: tuple[int, int] = (480, 640),
        seed: int = 42,
    ):
        self.n_episodes = n_episodes
        self.frames_per_episode = frames_per_episode
        self.fps = fps
        self.h, self.w = resolution
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    SYNTHETIC_TASKS = [
        {
            "task_name": "sauteeCarrot",
            "task_description": "robot sautéing carrot slices in a saucepan",
            "phases": ["PICK_AND_PLACE", "PREPARING", "CUTTING", "PICK_AND_PLACE", "COOK_COOL"],
        },
        {
            "task_name": "boilWater",
            "task_description": "robot boiling water in a pot on the stove",
            "phases": ["PICK_AND_PLACE", "POURING", "PICK_AND_PLACE", "COOK_COOL"],
        },
        {
            "task_name": "makeCoffee",
            "task_description": "robot making coffee and serving it on the table",
            "phases": ["PICK_AND_PLACE", "POURING", "COOK_COOL", "PICK_AND_PLACE"],
        },
    ]

    def list_episodes(self) -> list[REFLECTEpisode]:
        episodes = []
        for i in range(self.n_episodes):
            task = self.SYNTHETIC_TASKS[i % len(self.SYNTHETIC_TASKS)]
            episodes.append(REFLECTEpisode(
                task_name=task["task_name"],
                episode_id=f"synthetic_{i:03d}",
                rgb_dir=Path("synthetic"),
                depth_dir=Path("synthetic"),
                fps=self.fps,
                task_description=task["task_description"],
                failure_timestep=None,
                metadata={"synthetic": True, "phases": task["phases"]},
            ))
        return episodes

    def iter_frames(
        self,
        episode: REFLECTEpisode,
        step: int = 1,
    ) -> Iterator[tuple[int, float, np.ndarray, np.ndarray]]:
        """Generate synthetic RGBD frames with realistic motion patterns."""
        phases = episode.metadata.get("phases", ["PICK_AND_PLACE"])
        n_phases = len(phases)
        frames_per_phase = self.frames_per_episode // n_phases

        frame_idx = 0
        for phase_i, phase in enumerate(phases):
            for local_f in range(frames_per_phase):
                if frame_idx % step != 0:
                    frame_idx += 1
                    continue

                t = local_f / frames_per_phase  # 0.0 → 1.0 within phase

                # RGB: colour varies by phase to simulate different scenes
                rgb = self._make_rgb_frame(phase, t, frame_idx)
                depth = self._make_depth_frame(phase, t)

                yield frame_idx, self.fps, rgb, depth
                frame_idx += 1

    def _make_rgb_frame(self, phase: str, t: float, idx: int) -> np.ndarray:
        """Generate a plausible RGB kitchen frame for the given action phase."""
        phase_colors = {
            "CUTTING":       (220, 200, 180),
            "POURING":       (180, 210, 230),
            "MIXING":        (200, 220, 200),
            "PREPARING":     (230, 215, 195),
            "PICK_AND_PLACE":(210, 200, 210),
            "COOK_COOL":     (240, 180, 160),
        }
        base = np.array(phase_colors.get(phase, (200, 200, 200)), dtype=np.uint8)
        img = np.full((self.h, self.w, 3), base, dtype=np.uint8)

        # Add spatial noise
        noise = (np.random.randn(self.h, self.w, 3) * 15).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Simulate moving robot arm (white rectangle)
        arm_x = int(self.w * (0.3 + 0.4 * t))
        arm_y = int(self.h * (0.2 + 0.3 * np.sin(t * np.pi)))
        cv2.rectangle(img, (arm_x, arm_y), (arm_x + 40, arm_y + 80), (240, 240, 240), -1)

        # Simulate target object (coloured circle)
        obj_x = int(self.w * 0.55)
        obj_y = int(self.h * 0.6)
        cv2.circle(img, (obj_x, obj_y), 25, (100, 160, 100), -1)

        # Counter / table surface
        cv2.rectangle(img, (0, int(self.h * 0.65)), (self.w, self.h), (160, 140, 120), -1)

        return img

    def _make_depth_frame(self, phase: str, t: float) -> np.ndarray:
        """Generate a synthetic 16-bit depth map (millimetres)."""
        depth = np.full((self.h, self.w), 1500, dtype=np.float32)  # 1.5m background

        # Table surface at ~0.8m
        depth[int(self.h * 0.65):, :] = 800.0

        # Robot arm at varying depth
        arm_depth = 600.0 + 200.0 * t  # moves closer during action
        arm_x = int(self.w * (0.3 + 0.4 * t))
        arm_y = int(self.h * (0.2 + 0.3 * np.sin(t * np.pi)))
        depth[arm_y:arm_y+80, arm_x:arm_x+40] = arm_depth

        # Object on table
        cy, cx = int(self.h * 0.6), int(self.w * 0.55)
        yy, xx = np.ogrid[:self.h, :self.w]
        mask = (xx - cx)**2 + (yy - cy)**2 < 25**2
        depth[mask] = 780.0

        # Add Gaussian noise
        depth += np.random.randn(self.h, self.w).astype(np.float32) * 10.0
        depth = np.clip(depth, 100, 5000)
        return depth.astype(np.uint16)
