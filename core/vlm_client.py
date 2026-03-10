"""
VLM Client — Ollama Backend
============================
Sends keyframe bundles to an Ollama vision model and parses the response
into AKG-grounded action records.

Prompt Engineering follows the NVIDIA VLM Prompt Engineering Guide (2025):
  - Structured system prompt with AKG ontology
  - Multi-image input (one per keyframe in segment)
  - Chain-of-thought reasoning before JSON output
  - Few-shot examples for Action Core disambiguation
  - Depth grid injected as additional context text
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

from config import AKG_ACTION_CORES, ActionConfig, OllamaConfig
from core.keyframe_extraction import ActionSegment, KeyframeBundle
from core.rgbd_preprocessing import pil_to_base64

log = logging.getLogger(__name__)


# ── Output Schema ─────────────────────────────────────────────────────────────

@dataclass
class AKGAction:
    """Single AKG-grounded action extracted from a video segment."""
    segment_id: int
    action_core: str                # one of AKG_ACTION_CORES
    sub_action: str                 # free-text verb (e.g. "slice", "dice")
    start_time_s: float
    end_time_s: float
    objects_involved: list[str]
    contact_type: str               # e.g. "grasp", "pour", "none"
    spatial_relation: str           # e.g. "above cutting board"
    depth_context: str              # depth_text summary
    confidence: float               # 0.0 – 1.0
    reasoning: str                  # VLM chain-of-thought (if enabled)
    raw_vlm_response: str = field(default="", repr=False)


# ── Few-Shot Examples ─────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = [
    {
        "description": "Robot arm moves downward with gripper closed around a knife, blade contacts a carrot on cutting board. Depth shows knife tip at 0.35m.",
        "expected": {
            "action_core": "CUTTING",
            "sub_action": "slice",
            "objects_involved": ["knife", "carrot", "cutting_board"],
            "contact_type": "blade_contact",
            "spatial_relation": "above cutting_board",
            "confidence": 0.95
        }
    },
    {
        "description": "Robot gripper holds a pot and moves it laterally from sink toward stove burner. Depth shows pot at 0.6m, constant height.",
        "expected": {
            "action_core": "PICK_AND_PLACE",
            "sub_action": "transport",
            "objects_involved": ["pot", "stove_burner"],
            "contact_type": "grasp",
            "spatial_relation": "above stove",
            "confidence": 0.90
        }
    },
]


# ── Prompt Builder ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a robot action analyser specialised in kitchen manipulation tasks.

You observe keyframes from an RGBD video of a robot performing a meal preparation task.
Each image shows both RGB (left half) and depth/HHA encoding (right half) side by side.

Your task is to identify the robotic manipulation action being performed and classify it
into exactly ONE of the following AKG Action Cores:

  CUTTING      – any slicing, dicing, chopping action with a blade tool
  POURING      – transferring liquid or granular material between containers
  MIXING       – stirring, whisking, blending, or combining ingredients
  PREPARING    – peeling, washing, measuring, opening packaging, setting up
  PICK_AND_PLACE – grasping an object and placing it at a new location
  COOK_COOL    – placing on heat source / in oven / microwave, or cooling

CRITICAL RULES:
1. Output ONLY valid JSON — no preamble, no markdown fences, no extra text.
2. If chain-of-thought is requested, put it in the "reasoning" field INSIDE the JSON.
3. confidence must be a float between 0.0 and 1.0.
4. objects_involved must be a JSON array of strings.
5. If you cannot determine the action, use action_core = "PICK_AND_PLACE" with confidence < 0.4.
"""


def build_user_prompt(
    bundles: list[KeyframeBundle],
    seg: ActionSegment,
    task_description: str,
    cfg: ActionConfig,
    n_few_shot: int = 0,
) -> str:
    """Build the user-turn prompt text (without images — those go in the API call)."""
    lines = [
        f"Task context: {task_description}",
        f"Segment: frames {seg.start_frame}–{seg.end_frame} "
        f"(duration: {seg.end_frame - seg.start_frame} frames, "
        f"mean motion: {seg.mean_velocity:.4f}, "
        f"mean depth-change: {seg.depth_change:.4f}m)",
        "",
        "Keyframe depth summaries:",
    ]
    for i, b in enumerate(bundles):
        lines.append(f"  Frame {b.frame_idx} @ {b.timestamp_s:.2f}s: {b.depth_text}")

    if n_few_shot > 0:
        lines += ["", "--- Reference examples (do NOT copy, use as style guide) ---"]
        for ex in FEW_SHOT_EXAMPLES[:n_few_shot]:
            lines.append(f"Scene: {ex['description']}")
            lines.append(f"Expected output: {json.dumps(ex['expected'])}")

    lines += [
        "",
        "Now analyse the provided keyframe images for this segment.",
    ]

    if cfg.chain_of_thought:
        lines.append(
            'First reason step-by-step about: (1) what objects are visible, '
            '(2) what the gripper/end-effector is doing, '
            '(3) what depth context reveals about spatial relationships. '
            'Then output the JSON.'
        )

    lines += [
        "",
        "Output JSON schema:",
        json.dumps({
            "action_core": "<one of: " + " | ".join(AKG_ACTION_CORES) + ">",
            "sub_action": "<specific verb>",
            "objects_involved": ["<object1>", "<object2>"],
            "contact_type": "<grasp | blade_contact | pour | none | other>",
            "spatial_relation": "<spatial description>",
            "depth_context": "<key depth observation>",
            "confidence": 0.0,
            "reasoning": "<chain-of-thought if applicable>"
        }, indent=2),
    ]

    return "\n".join(lines)


# ── Ollama API Client ─────────────────────────────────────────────────────────

class OllamaVLMClient:
    """
    Thin wrapper around the Ollama /api/chat endpoint for vision models.
    Compatible with OpenWebUI-hosted Ollama servers.
    """

    def __init__(self, cfg: OllamaConfig):
        self.cfg = cfg
        self.endpoint = f"{cfg.base_url.rstrip('/')}/api/chat"

    def _call(self, messages: list[dict], retries: int = 2) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.max_tokens,
            }
        }
        for attempt in range(retries + 1):
            try:
                resp = requests.post(
                    self.endpoint,
                    json=payload,
                    timeout=self.cfg.timeout
                )
                resp.raise_for_status()
                data = resp.json()
                return data["message"]["content"]
            except requests.exceptions.ConnectionError as e:
                log.error(f"Ollama connection failed: {e}")
                raise
            except Exception as e:
                if attempt < retries:
                    log.warning(f"VLM call failed (attempt {attempt+1}): {e} — retrying…")
                    time.sleep(2 ** attempt)
                else:
                    raise

    def query_segment(
        self,
        bundles: list[KeyframeBundle],
        seg: ActionSegment,
        task_description: str,
        action_cfg: ActionConfig,
    ) -> str:
        """
        Send keyframe images + prompt to Ollama and return the raw response string.
        """
        user_text = build_user_prompt(
            bundles, seg, task_description, action_cfg,
            n_few_shot=action_cfg.few_shot_examples
        )

        # Build images list (base64)
        images = [pil_to_base64(b.fused_pil) for b in bundles]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_text,
                "images": images,          # Ollama vision API format
            }
        ]

        log.debug(f"Querying VLM for segment {seg.start_frame}–{seg.end_frame} "
                  f"with {len(images)} image(s)")
        return self._call(messages)


# ── Response Parser ───────────────────────────────────────────────────────────

def parse_vlm_response(
    raw: str,
    seg: ActionSegment,
    seg_id: int,
    bundles: list[KeyframeBundle],
    action_cfg: ActionConfig,
) -> Optional[AKGAction]:
    """
    Parse the raw VLM string into an AKGAction.
    Handles: clean JSON, JSON embedded in markdown fences, partial JSON.
    """
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    # Find the first { ... } block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        log.warning(f"No JSON found in VLM response for segment {seg_id}: {raw[:200]}")
        return None

    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse error for segment {seg_id}: {e}\nRaw: {raw[:300]}")
        return None

    # Validate action_core
    action_core = str(parsed.get("action_core", "PICK_AND_PLACE")).upper()
    if action_core not in AKG_ACTION_CORES:
        log.warning(f"Unknown action_core '{action_core}' — defaulting to PICK_AND_PLACE")
        action_core = "PICK_AND_PLACE"

    confidence = float(parsed.get("confidence", 0.5))
    if confidence < action_cfg.min_confidence:
        log.info(f"Segment {seg_id}: confidence {confidence:.2f} below threshold — skipping")
        return None

    start_t = bundles[0].timestamp_s if bundles else 0.0
    end_t = bundles[-1].timestamp_s if bundles else 0.0

    return AKGAction(
        segment_id=seg_id,
        action_core=action_core,
        sub_action=str(parsed.get("sub_action", "")),
        start_time_s=start_t,
        end_time_s=end_t,
        objects_involved=list(parsed.get("objects_involved", [])),
        contact_type=str(parsed.get("contact_type", "unknown")),
        spatial_relation=str(parsed.get("spatial_relation", "")),
        depth_context=str(parsed.get("depth_context", "")),
        confidence=confidence,
        reasoning=str(parsed.get("reasoning", "")),
        raw_vlm_response=raw,
    )
