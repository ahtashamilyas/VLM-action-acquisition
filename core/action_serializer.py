import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.vlm_client import AKGAction

log = logging.getLogger(__name__)


def actions_to_dict(
    actions: list[AKGAction],
    task_description: str,
    source_video: Optional[str] = None,
    total_frames: int = 0,
    video_fps: float = 30.0,
    schema_version: str = "1.0",
) -> dict:
    """Convert AKGAction list → serialisable dict (simplified: actions + intervals only)."""
    duration_s = total_frames / video_fps if total_frames > 0 else 0.0

    action_list = []
    for i, a in enumerate(actions):
        entry = {
            "id": i,
            "segment_id": a.segment_id,
            "action": a.action,
            "start_time_s": round(a.start_time_s, 3),
            "end_time_s": round(a.end_time_s, 3),
            "duration_s": round(a.end_time_s - a.start_time_s, 3),
            "confidence": round(a.confidence, 2),
        }
        action_list.append(entry)

    from collections import Counter
    counts = Counter(a.action for a in actions)
    mean_conf = (sum(a.confidence for a in actions) / len(actions)) if actions else 0.0

    return {
        "task_description": task_description,
        "source_video": str(source_video) if source_video else None,
        "duration_s": round(duration_s, 3),
        "total_actions": len(actions),
        "actions": action_list,
        "summary": {
            "action_counts": dict(counts),
            "mean_confidence": round(mean_conf, 2),
        }
    }


def save_json(
    actions: list[AKGAction],
    output_path: str,
    task_description: str,
    source_video: Optional[str] = None,
    total_frames: int = 0,
    video_fps: float = 30.0,
) -> dict:
    """Serialise actions to JSON file (simplified output)."""
    data = actions_to_dict(
        actions,
        task_description=task_description,
        source_video=source_video,
        total_frames=total_frames,
        video_fps=video_fps,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    log.info(f"Saved {len(actions)} actions → {out}")
    return data


def print_summary(data: dict):
    """Print simplified action detection summary."""
    print("  ── ACTION DETECTION RESULTS ──")
    print(f"  Task      : {data['task_description']}")
    print(f"  Source    : {data.get('source_video', 'N/A')}")
    print(f"  Duration  : {data['duration_s']:.1f}s")
    print(f"  Actions   : {data['total_actions']}")
    print(f"  Mean conf : {data['summary']['mean_confidence']:.2f}")
    print()
    for a in data["actions"]:
        bar = "█" * int(a["confidence"] * 10)
        print(f"  [{a['id']:02d}] {a['start_time_s']:6.2f}s → {a['end_time_s']:6.2f}s  "
              f"{a['action']:<15} (conf={a['confidence']:.2f}) {bar}")
    print("=" * 60 + "\n")
