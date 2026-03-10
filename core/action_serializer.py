"""
Action Sequence Serializer
===========================
Converts a list of AKGAction objects into the final JSON output format,
optionally enriched with REFLECT-compatible metadata.

Output schema (v1.0):
{
  "schema_version": "1.0",
  "task_description": "...",
  "source_video": "...",
  "processed_at": "2026-03-10T...",
  "total_frames": 1234,
  "duration_s": 41.1,
  "action_sequence": [
    {
      "id": 0,
      "segment_id": 0,
      "action_core": "PICK_AND_PLACE",
      "sub_action": "transport",
      "start_time_s": 0.0,
      "end_time_s": 3.2,
      "duration_s": 3.2,
      "objects_involved": ["pot", "stove"],
      "contact_type": "grasp",
      "spatial_relation": "above stove",
      "depth_context": "...",
      "confidence": 0.91,
      "reasoning": "..."
    },
    ...
  ],
  "summary": {
    "action_counts": {"PICK_AND_PLACE": 3, "CUTTING": 1, ...},
    "mean_confidence": 0.85,
    "coverage_s": 38.5,
    "low_confidence_segments": [...]
  }
}
"""

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
    include_reasoning: bool = True,
) -> dict:
    """Convert AKGAction list → serialisable dict."""
    duration_s = total_frames / video_fps if total_frames > 0 else 0.0

    action_list = []
    for i, a in enumerate(actions):
        entry = {
            "id": i,
            "segment_id": a.segment_id,
            "action_core": a.action_core,
            "sub_action": a.sub_action,
            "start_time_s": round(a.start_time_s, 3),
            "end_time_s": round(a.end_time_s, 3),
            "duration_s": round(a.end_time_s - a.start_time_s, 3),
            "objects_involved": a.objects_involved,
            "contact_type": a.contact_type,
            "spatial_relation": a.spatial_relation,
            "depth_context": a.depth_context,
            "confidence": round(a.confidence, 4),
        }
        if include_reasoning and a.reasoning:
            entry["reasoning"] = a.reasoning
        action_list.append(entry)

    # Build summary
    from collections import Counter
    counts = Counter(a.action_core for a in actions)
    mean_conf = (sum(a.confidence for a in actions) / len(actions)) if actions else 0.0
    coverage = sum(a.end_time_s - a.start_time_s for a in actions)
    low_conf = [
        {"segment_id": a.segment_id, "action_core": a.action_core, "confidence": a.confidence}
        for a in actions if a.confidence < 0.6
    ]

    return {
        "schema_version": schema_version,
        "task_description": task_description,
        "source_video": str(source_video) if source_video else None,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "total_frames": total_frames,
        "duration_s": round(duration_s, 3),
        "action_sequence": action_list,
        "summary": {
            "total_actions": len(actions),
            "action_counts": dict(counts),
            "mean_confidence": round(mean_conf, 4),
            "coverage_s": round(coverage, 3),
            "low_confidence_segments": low_conf,
        }
    }


def save_json(
    actions: list[AKGAction],
    output_path: str,
    task_description: str,
    source_video: Optional[str] = None,
    total_frames: int = 0,
    video_fps: float = 30.0,
    include_reasoning: bool = True,
) -> dict:
    """Serialise actions to JSON file and return the dict."""
    data = actions_to_dict(
        actions,
        task_description=task_description,
        source_video=source_video,
        total_frames=total_frames,
        video_fps=video_fps,
        include_reasoning=include_reasoning,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    log.info(f"Saved {len(actions)} actions → {out}")
    return data


def print_summary(data: dict):
    """Pretty-print the action sequence summary to stdout."""
    s = data["summary"]
    print("\n" + "═" * 60)
    print("  REFLECT ACTION EXTRACTION RESULTS")
    print("═" * 60)
    print(f"  Task      : {data['task_description']}")
    print(f"  Source    : {data.get('source_video', 'N/A')}")
    print(f"  Duration  : {data['duration_s']:.1f}s  ({data['total_frames']} frames)")
    print(f"  Actions   : {s['total_actions']}  (coverage: {s['coverage_s']:.1f}s)")
    print(f"  Mean conf : {s['mean_confidence']:.3f}")
    print()
    print("  ── Action Sequence ──")
    for a in data["action_sequence"]:
        bar = "█" * int(a["confidence"] * 10)
        print(f"  [{a['id']:02d}] {a['start_time_s']:6.2f}s → {a['end_time_s']:6.2f}s  "
              f"{a['action_core']:<18} {a['sub_action']:<15} "
              f"conf={a['confidence']:.2f} {bar}")
    if s["low_confidence_segments"]:
        print(f"\n  ⚠  {len(s['low_confidence_segments'])} low-confidence segment(s)")
    print("═" * 60 + "\n")
