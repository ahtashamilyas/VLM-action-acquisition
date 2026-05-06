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
        fmt = getattr(self.cfg, "image_format", "content_parts")

        if fmt == "ollama_images":
            return [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_text,
                    "images": images_b64,
                }
            ]
        else:
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
                if not resp.ok:
                    log.error(
                        f"Server returned {resp.status_code}. "
                        f"Response body: {resp.text[:800]}"
                    )
                resp.raise_for_status()

                result = resp.json()
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
) -> list[AKGAction]:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            parsed = None
    else:
        parsed = None
    if parsed is None:
        match_obj = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match_obj:
            try:
                obj = json.loads(match_obj.group())
                parsed = [{"frame_idx": 0, "action": obj.get("action", "unknown"),
                           "confidence": obj.get("confidence", 0.5)}]
            except json.JSONDecodeError:
                pass

    if not parsed:
        log.warning(f"No JSON found in VLM response for segment {seg_id}: {raw[:200]}")
        return []

    # Build a timestamp-indexed lookup: frame_idx → KeyframeBundle
    bundle_map = {b.frame_idx: b for b in bundles}
    # Also support positional indexing if VLM uses 0,1,2 rather than real frame_idx
    bundle_list = list(bundles)

    def _bundle_for(entry: dict, pos: int) -> KeyframeBundle:
        fidx = entry.get("frame_idx", pos)
        return bundle_map.get(fidx, bundle_list[min(pos, len(bundle_list) - 1)])

    # Collapse consecutive same-action runs → one AKGAction each
    actions: list[AKGAction] = []
    run_start_pos = 0
    run_action = None
    run_confidences: list[float] = []

    def _flush_run(run_action, run_start_pos, run_end_pos, run_confidences, sub_id):
        b_start = _bundle_for(parsed[run_start_pos], run_start_pos)
        b_end   = _bundle_for(parsed[run_end_pos],   run_end_pos)
        conf = sum(run_confidences) / len(run_confidences)
        if conf < action_cfg.min_confidence:
            log.info(f"Segment {seg_id} sub-action '{run_action}': "
                     f"confidence {conf:.2f} below threshold — skipping")
            return None
        return AKGAction(
            segment_id=seg_id * 100 + sub_id,   # unique id: seg*100 + sub-action index
            action=run_action,
            start_time_s=b_start.timestamp_s,
            end_time_s=b_end.timestamp_s,
            confidence=round(conf, 2),
            raw_vlm_response=raw,
        )

    sub_id = 0
    for pos, entry in enumerate(parsed):
        action_name = str(entry.get("action", "unknown")).lower().strip()
        confidence  = float(entry.get("confidence", 0.5))

        if run_action is None:
            run_action = action_name
            run_start_pos = pos
            run_confidences = [confidence]
        elif action_name == run_action:
            run_confidences.append(confidence)
        else:
            # flush previous run
            result = _flush_run(run_action, run_start_pos, pos - 1, run_confidences, sub_id)
            if result:
                actions.append(result)
                sub_id += 1
            # start new run
            run_action = action_name
            run_start_pos = pos
            run_confidences = [confidence]

    # flush final run
    if run_action is not None:
        result = _flush_run(run_action, run_start_pos, len(parsed) - 1, run_confidences, sub_id)
        if result:
            actions.append(result)

    # Fix zero-duration actions (start == end, i.e. single-keyframe runs).
    # Extend each such action to the midpoint between its neighbours so the
    # output intervals tile the full segment span with no gaps and no zeros.
    for i, a in enumerate(actions):
        if a.start_time_s < a.end_time_s:
            continue  # already has a span — leave it alone
        prev_end = actions[i - 1].end_time_s if i > 0 else bundles[0].timestamp_s
        next_start = actions[i + 1].start_time_s if i < len(actions) - 1 else bundles[-1].timestamp_s
        a.start_time_s = round((prev_end + a.start_time_s) / 2, 3)
        a.end_time_s   = round((a.end_time_s + next_start) / 2, 3)

    return actions
