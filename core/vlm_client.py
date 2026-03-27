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
    Client for the university OpenWebUI server.

    Matches the API format from supervisor credentials:
      - Endpoint : POST {base_url}/api/chat/completions   (OpenAI-compatible)
      - Auth     : Authorization: <user_id>  (university email, no "Bearer" prefix)
      - Images   : sent as OpenAI-style content parts inside the user message

    Reference chatbot() function provided by supervisor:
        headers = {'Authorization': user_id, 'Content-Type': 'application/json'}
        data    = {"model": model, "messages": [{"role": "user", "content": query}]}
    """

    def __init__(self, cfg: OllamaConfig):
        self.cfg = cfg
        self.endpoint = cfg.chat_endpoint
        self.headers = {
            "Authorization": cfg.auth_header(),  # Bearer sk-xxxx (API key) or email fallback
            "Content-Type": "application/json",
        }
        auth_type = "API key" if cfg.api_key else "email (fallback)"
        log.info(f"OpenWebUI client → {self.endpoint}  model={cfg.model}  auth={auth_type}")

    def _build_messages(self, system_prompt: str, user_text: str, images_b64: list[str]) -> list[dict]:
        """
        Build messages for OpenWebUI /api/chat/completions.

        Two formats tried in order (controlled by self.cfg.image_format):
          "content_parts"  : OpenAI-style [{type:image_url,...},{type:text,...}]
          "ollama_images"  : Ollama native  {"content": text, "images": [b64, ...]}

        OpenWebUI generally accepts content_parts, but some model backends
        (especially older llava via Ollama) need the native images field.
        """
        fmt = getattr(self.cfg, "image_format", "content_parts")

        if fmt == "ollama_images":
            # Ollama-native format: images as plain base64 list, text as string
            return [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_text,
                    "images": images_b64,
                }
            ]
        else:
            # OpenAI content-parts format (default)
            content_parts = []
            for b64 in images_b64:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })
            content_parts.append({"type": "text", "text": user_text})
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_parts},
            ]

    def _call(self, messages: list[dict], retries: int = 2) -> str:
        """POST to the OpenWebUI /api/chat/completions endpoint."""
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        last_exc = None
        for attempt in range(retries + 1):
            try:
                resp = requests.post(
                    self.endpoint,
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=self.cfg.timeout,
                )

                # Always log server error body — crucial for debugging 400/422
                if not resp.ok:
                    log.error(
                        f"Server returned {resp.status_code}. "
                        f"Response body: {resp.text[:800]}"
                    )
                resp.raise_for_status()

                result = resp.json()
                # OpenAI-compatible: choices[0].message.content
                return result["choices"][0]["message"]["content"]

            except requests.exceptions.ConnectionError as e:
                log.error(f"Cannot reach server at {self.endpoint}: {e}")
                raise
            except KeyError as e:
                log.error(f"Unexpected response format (missing {e}): {resp.text[:300]}")
                raise
            except Exception as e:
                last_exc = e
                if attempt < retries:
                    log.warning(f"VLM call failed (attempt {attempt+1}/{retries}): {e} — retrying…")
                    time.sleep(2 ** attempt)
                else:
                    raise last_exc

    def query_segment(
        self,
        bundles: list[KeyframeBundle],
        seg: ActionSegment,
        task_description: str,
        action_cfg: ActionConfig,
    ) -> str:
        """
        Send keyframe images + structured AKG prompt to OpenWebUI/Ollama.
        Returns raw VLM response string.
        """
        user_text = build_user_prompt(
            bundles, seg, task_description, action_cfg,
            n_few_shot=action_cfg.few_shot_examples
        )
        images_b64 = [pil_to_base64(b.fused_pil) for b in bundles]

        messages = self._build_messages(SYSTEM_PROMPT, user_text, images_b64)

        log.info(f"  → Querying {self.cfg.model} | segment {seg.start_frame}–{seg.end_frame} "
                 f"| {len(images_b64)} keyframe(s)")
        return self._call(messages)

    def test_connection(self) -> bool:
        """
        Quick connectivity test — sends a plain text message with no images.
        Call this before running the full pipeline to verify credentials.
        """
        try:
            payload = {
                "model": self.cfg.model,
                "messages": [{"role": "user", "content": "Reply with the single word: OK"}],
                "stream": False,
                "max_tokens": 10,
            }
            resp = requests.post(
                self.endpoint,
                headers=self.headers,
                data=json.dumps(payload),
                timeout=15,
            )
            resp.raise_for_status()
            result = resp.json()
            reply = result["choices"][0]["message"]["content"].strip()
            log.info(f"Connection test passed. Server replied: '{reply}'")
            return True
        except Exception as e:
            log.error(f"Connection test FAILED: {e}")
            return False


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