"""
REFLECT Dataset Loader
=======================
Handles the actual REFLECT dataset structure:

  <data_root>/
    boilWater1/          <- episode folder (task name + number)
      videos/
        color/           <- Zarr array, 4D chunks: frame.0.0.0 (JPEG-XL, uint8)
        depth/           <- Zarr array, 3D chunks: frame.0.0   (raw uint16)
        color.mp4        <- fallback RGB video
      replay_buffer.zarr/
    boilWater2/
    sauteeCarrot1/
    ...

Episode detection: a folder is an episode if its name ends with a digit
AND contains a "videos/" subdirectory.
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

TASK_DESCRIPTIONS = {
    "appleInFridge":  "robot storing an apple in the fridge",
    "boilWater":      "robot boiling water in a pot on the stove",
    "cutCarrot":      "robot cutting a carrot on a cutting board",
    "heatPot":        "robot heating a pot on the stove",
    "heatPotato":     "robot heating a potato in the microwave",
    "makeCoffee":     "robot making coffee and serving it on the table",
    "putAppleBowl":   "robot putting an apple in a bowl",
    "putFruitsBowl":  "robot putting fruits into a bowl",
    "putPearDrawer":  "robot putting a pear in a drawer",
    "sauteeCarrot":   "robot sauteing carrot slices in a saucepan",
    "secureObjects":  "robot securing objects by placing them in safe locations",
}

def _task_name(folder: str) -> str:
    return folder.rstrip("0123456789")

def _task_desc(folder: str) -> str:
    return TASK_DESCRIPTIONS.get(_task_name(folder), f"robot: {_task_name(folder)}")

def _is_episode_dir(d: Path) -> bool:
    """An episode dir ends with a digit AND has a videos/ subdirectory."""
    return d.is_dir() and d.name[-1].isdigit() and (d / "videos").is_dir()

def _read_meta(zarr_dir: Path) -> dict:
    p = zarr_dir / ".zarray"
    return json.loads(p.read_text()) if p.exists() else {}

def _sorted_indices(zarr_dir: Path, ndim: int) -> list:
    """
    Collect frame indices from chunk filenames.
    Color (4D): chunks named  42.0.0.0  -> index 42
    Depth (3D): chunks named  42.0.0    -> index 42
    """
    idxs = []
    for f in zarr_dir.iterdir():
        if f.name.startswith("."):
            continue
        parts = f.name.split(".")
        if len(parts) == ndim:          # 4 parts for 4D, 3 parts for 3D
            try:
                idxs.append(int(parts[0]))
            except ValueError:
                pass
    return sorted(idxs)


@dataclass
class REFLECTEpisode:
    task_name: str
    episode_id: str
    episode_dir: Path
    video_path: Optional[Path]
    color_zarr: Optional[Path]
    depth_zarr: Optional[Path]
    fps: float
    task_description: str
    failure_timestep: Optional[float]
    metadata: dict


class REFLECTDataset:

    def __init__(self, data_root: str):
        self.data_root = Path(data_root).resolve()
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"REFLECT dataset not found at {data_root}.\n"
                "Download from: https://www.cs.columbia.edu/~liuzeyi/reflect_data/\n"
                "Or use SyntheticREFLECTDataset for testing."
            )

    def list_episodes(self, task_filter: Optional[str] = None) -> list:
        episodes = []
        for ep_dir in sorted(self.data_root.iterdir()):
            # Must end with digit AND have videos/ subdir
            if not _is_episode_dir(ep_dir):
                log.debug(f"Skipping {ep_dir.name} — not an episode dir")
                continue
            if task_filter and not ep_dir.name.lower().startswith(task_filter.lower()):
                continue

            vd = ep_dir / "videos"
            color_zarr = vd / "color" if (vd / "color" / ".zarray").exists() else None
            depth_zarr = vd / "depth"  if (vd / "depth"  / ".zarray").exists() else None
            video_path = vd / "color.mp4" if (vd / "color.mp4").exists() else None

            fps = 30.0
            if video_path:
                cap = cv2.VideoCapture(str(video_path))
                f = cap.get(cv2.CAP_PROP_FPS)
                fps = f if f > 0 else 30.0
                cap.release()

            if not color_zarr and not video_path:
                log.warning(f"  {ep_dir.name}: no RGB source found, skipping")
                continue

            n_color = len(_sorted_indices(color_zarr, 4)) if color_zarr else 0
            log.debug(f"  {ep_dir.name}: color={'zarr('+str(n_color)+')' if color_zarr else 'mp4'} "
                      f"depth={'zarr' if depth_zarr else 'none'} fps={fps:.1f}")

            episodes.append(REFLECTEpisode(
                task_name=_task_name(ep_dir.name),
                episode_id=ep_dir.name,
                episode_dir=ep_dir,
                video_path=video_path,
                color_zarr=color_zarr,
                depth_zarr=depth_zarr,
                fps=fps,
                task_description=_task_desc(ep_dir.name),
                failure_timestep=None,
                metadata={},
            ))

        log.info(f"Found {len(episodes)} episodes in {self.data_root}"
                 + (f" (task={task_filter})" if task_filter else ""))
        return episodes

    def iter_frames(self, episode: REFLECTEpisode, step: int = 1) -> Iterator:
        """Yield (frame_idx, fps, rgb uint8 H×W×3, depth uint16 H×W)."""
        if episode.color_zarr:
            yield from self._iter_zarr(episode, step)
        elif episode.video_path:
            log.info(f"  Reading from {episode.video_path.name} (no zarr)")
            yield from self._iter_video(episode, step)
        else:
            log.error(f"  No RGB source for {episode.episode_id}")

    def _iter_zarr(self, episode: REFLECTEpisode, step: int) -> Iterator:
        # Check imagecodecs
        try:
            import imagecodecs
        except ImportError:
            log.warning("imagecodecs not installed — falling back to color.mp4\n"
                        "  Fix: pip install imagecodecs")
            if episode.video_path:
                yield from self._iter_video(episode, step)
            return

        # Color metadata
        c_meta   = _read_meta(episode.color_zarr)
        c_dtype  = np.dtype(c_meta.get("dtype", "|u1"))
        c_chunks = tuple(c_meta.get("chunks", [1, 720, 1280, 3]))  # (1,H,W,3)
        c_idxs   = _sorted_indices(episode.color_zarr, ndim=4)

        # Depth metadata
        d_idxs_set = set()
        d_dtype = d_chunks = None
        if episode.depth_zarr:
            d_meta   = _read_meta(episode.depth_zarr)
            d_dtype  = np.dtype(d_meta.get("dtype", "<u2"))
            d_chunks = tuple(d_meta.get("chunks", [1, 720, 1280]))  # (1,H,W)
            d_idxs_set = set(_sorted_indices(episode.depth_zarr, ndim=3))

        log.info(f"  Zarr: {len(c_idxs)} color frames, shape={c_chunks[1:]}, "
                 f"depth frames={len(d_idxs_set)}")

        for i, frame_idx in enumerate(c_idxs):
            if i % step != 0:
                continue

            # Read color (JPEG-XL compressed)
            chunk = episode.color_zarr / f"{frame_idx}.0.0.0"
            if not chunk.exists():
                continue
            try:
                decoded = imagecodecs.jpegxl_decode(chunk.read_bytes())
                rgb = np.frombuffer(decoded, dtype=c_dtype).reshape(c_chunks[1:])
            except Exception as e:
                log.debug(f"  Color decode failed frame {frame_idx}: {e}")
                continue

            # Read depth (raw uint16)
            depth = np.zeros(rgb.shape[:2], dtype=np.uint16)
            if episode.depth_zarr and frame_idx in d_idxs_set:
                dp = episode.depth_zarr / f"{frame_idx}.0.0"
                if dp.exists():
                    try:
                        raw = np.frombuffer(dp.read_bytes(), dtype=d_dtype)
                        depth = raw.reshape(d_chunks)[0]
                    except Exception as e:
                        log.debug(f"  Depth read failed frame {frame_idx}: {e}")

            yield frame_idx, episode.fps, rgb, depth

    def _iter_video(self, episode: REFLECTEpisode, step: int) -> Iterator:
        cap = cv2.VideoCapture(str(episode.video_path))
        frame_idx = 0
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                depth = np.zeros(rgb.shape[:2], dtype=np.uint16)
                yield frame_idx, episode.fps, rgb, depth
            frame_idx += 1
        cap.release()


class SyntheticREFLECTDataset:
    """Generates synthetic RGBD frames for testing without real data."""

    SYNTHETIC_TASKS = [
        {"task_name": "sauteeCarrot",
         "task_description": "robot sauteing carrot slices in a saucepan",
         "phases": ["PICK_AND_PLACE", "PREPARING", "CUTTING", "PICK_AND_PLACE", "COOK_COOL"]},
        {"task_name": "boilWater",
         "task_description": "robot boiling water in a pot on the stove",
         "phases": ["PICK_AND_PLACE", "POURING", "PICK_AND_PLACE", "COOK_COOL"]},
        {"task_name": "makeCoffee",
         "task_description": "robot making coffee and serving it on the table",
         "phases": ["PICK_AND_PLACE", "POURING", "COOK_COOL", "PICK_AND_PLACE"]},
    ]

    def __init__(self, n_episodes=3, frames_per_episode=120, fps=10.0,
                 resolution=(480, 640), seed=42):
        self.n_episodes = n_episodes
        self.frames_per_episode = frames_per_episode
        self.fps = fps
        self.h, self.w = resolution
        random.seed(seed); np.random.seed(seed)

    def list_episodes(self, task_filter=None):
        eps = [
            REFLECTEpisode(
                task_name=t["task_name"], episode_id=f"synthetic_{i:03d}",
                episode_dir=Path("synthetic"), video_path=None,
                color_zarr=None, depth_zarr=None, fps=self.fps,
                task_description=t["task_description"], failure_timestep=None,
                metadata={"synthetic": True, "phases": t["phases"]},
            )
            for i, t in enumerate((self.SYNTHETIC_TASKS * 10)[:self.n_episodes])
        ]
        if task_filter:
            eps = [e for e in eps if e.task_name.lower().startswith(task_filter.lower())]
        return eps

    def iter_frames(self, episode, step=1):
        phases = episode.metadata.get("phases", ["PICK_AND_PLACE"])
        fpf = self.frames_per_episode // len(phases)
        frame_idx = 0
        for phase in phases:
            for local_f in range(fpf):
                if frame_idx % step == 0:
                    t = local_f / fpf
                    yield frame_idx, self.fps, self._rgb(phase, t), self._depth()
                frame_idx += 1

    def _rgb(self, phase, t):
        colors = {"CUTTING":(220,200,180),"POURING":(180,210,230),"MIXING":(200,220,200),
                  "PREPARING":(230,215,195),"PICK_AND_PLACE":(210,200,210),"COOK_COOL":(240,180,160)}
        img = np.full((self.h,self.w,3), colors.get(phase,(200,200,200)), dtype=np.uint8)
        img = np.clip(img.astype(np.int16)+(np.random.randn(self.h,self.w,3)*15).astype(np.int16),
                      0, 255).astype(np.uint8)
        ax=int(self.w*(0.3+0.4*t)); ay=int(self.h*(0.2+0.3*np.sin(t*np.pi)))
        cv2.rectangle(img,(ax,ay),(ax+40,ay+80),(240,240,240),-1)
        cv2.circle(img,(int(self.w*0.55),int(self.h*0.6)),25,(100,160,100),-1)
        cv2.rectangle(img,(0,int(self.h*0.65)),(self.w,self.h),(160,140,120),-1)
        return img

    def _depth(self):
        d = np.full((self.h,self.w),1500,dtype=np.float32)
        d[int(self.h*0.65):,:] = 800
        return np.clip(d+np.random.randn(self.h,self.w)*10,100,5000).astype(np.uint16)