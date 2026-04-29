"""
VLM Client — Ollama Backend
============================
Sends keyframe bundles to an Ollama vision model for low-level action detection.

Returns the detected action and confidence score.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

from config import ActionConfig, OllamaConfig
from core.keyframe_extraction import ActionSegment, KeyframeBundle
from core.rgbd_preprocessing import pil_to_base64

log = logging.getLogger(__name__)


# ── Output Schema ─────────────────────────────────────────────────────────────

@dataclass
class AKGAction:
    """Single action detected from a video segment."""
    segment_id: int
    action: str                     # action name/type (e.g., "pick", "place", "move")
    start_time_s: float
    end_time_s: float
    confidence: float               # 0.0 – 1.0
    raw_vlm_response: str = field(default="", repr=False)


# ── Few-Shot Examples ─────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = []  # Removed: not needed for simple action classification


# ── Prompt Builder ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a robot action analyser.

You receive N keyframes (images) from a single video segment, labelled Frame 0, Frame 1, … Frame N-1.
Each frame may show a DIFFERENT low-level action as the robot transitions through a motion.

Possible actions: reach, grasp, lift, move, align, orient, position_over_container, insert_into_container, place_on_object, release, drop, adjust, remove_obstacle

RULES:
1. Output ONLY a valid JSON array — one entry per frame — no preamble, no markdown fences, no extra text.
2. Each entry must have exactly: "frame_idx" (int), "action" (str), "confidence" (float 0.0–1.0).
3. Choose the action that best describes what the robot is doing IN THAT FRAME.
4. If two consecutive frames show the same action, that is fine — repeat it.
5. If uncertain about a frame, use confidence < 0.5.
"""


def build_user_prompt(
    bundles: list[KeyframeBundle],
    seg: ActionSegment,
    task_description: str,
    cfg: ActionConfig,
    n_few_shot: int = 0,
) -> str:
    """Build the user prompt for per-keyframe action labelling."""
    frame_lines = [
        f"  Frame {i}: t={b.timestamp_s:.3f}s"
        for i, b in enumerate(bundles)
    ]
    example_output = json.dumps(
        [{"frame_idx": i, "action": "<action>", "confidence": 0.0} for i in range(len(bundles))],
        indent=2,
    )
    lines = [
        f"Task: {task_description}",
        f"Segment: {bundles[0].timestamp_s:.3f}s – {bundles[-1].timestamp_s:.3f}s",
        "",
        "Keyframes (images are attached in order):",
        *frame_lines,
        "",
        "Label each frame with the low-level action the robot is performing at that moment.",
        "",
        "Output ONLY this JSON array (one entry per frame):",
        example_output,
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
    """Parse raw VLM response into AKGAction (simplified: action + confidence only)."""
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

    action = str(parsed.get("action", "unknown")).lower().strip()
    confidence = float(parsed.get("confidence", 0.5))
    
    if confidence < action_cfg.min_confidence:
        log.info(f"Segment {seg_id}: confidence {confidence:.2f} below threshold — skipping")
        return None

    start_t = bundles[0].timestamp_s if bundles else 0.0
    end_t = bundles[-1].timestamp_s if bundles else 0.0

    return AKGAction(
        segment_id=seg_id,
        action=action,
        start_time_s=start_t,
        end_time_s=end_t,
        confidence=confidence,
        raw_vlm_response=raw,
    )
