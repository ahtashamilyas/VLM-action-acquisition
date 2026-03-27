"""
REFLECT Pipeline Configuration
================================
Central config for the RGBD → Keyframes → VLM → AKG Actions pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional


# ── AKG Action Core Taxonomy (Kümpel et al. 2025) ──────────────────────────
AKG_ACTION_CORES = [
    "CUTTING",
    "POURING",
    "MIXING",
    "PREPARING",
    "PICK_AND_PLACE",
    "COOK_COOL",
]

# ── VLM Backend ─────────────────────────────────────────────────────────────
@dataclass
class OllamaConfig:
    base_url: str = "http://192.168.200.10:3000"   # ← change to your OpenWebUI host
    user_id: str = "o_yfodc2@uni-bremen.de" 
    api_key: str = "sk-f7b49d8820cd49f69922e849647d4b32"
    model: str = "openchat:7b"                    # ← any vision model available in Ollama
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: int = 600

    def auth_header(self) -> str:
        if self.api_key and self.api_key.strip():
            return f"Bearer {self.api_key.strip()}"
        return self.user_id

    @property
    def chat_endpoint(self) -> str:
        url = self.base_url.rstrip("/")
        if url.startswith("https://"):
            import socket, ssl
            hp = url.replace("https://", "").split("/")[0]
            h = hp.split(":")[0]
            p = int(hp.split(":")[1]) if ":" in hp else 443
            try:
                ctx = ssl.create_default_context()
                with socket.create_connection((h, p), timeout=3) as s:
                    with ctx.wrap_socket(s, server_hostname=h): pass
            except Exception:
                url = "http://" + url[8:]
        return f"{url}/api/chat/completions"                           # seconds


# ── RGBD Preprocessing ──────────────────────────────────────────────────────
@dataclass
class RGBDConfig:
    # Depth encoding strategy: "hha" | "depth_as_text" | "depth_channel"
    depth_strategy: str = "depth_as_text"
    # Depth scale factor (convert raw depth units to metres)
    depth_scale: float = 0.001             # typical for RealSense (mm → m)
    # Clip depth range in metres
    depth_min: float = 0.1
    depth_max: float = 3.0
    # Camera intrinsics (RealSense D435 defaults — override for your sensor)
    fx: float = 615.0
    fy: float = 615.0
    cx: float = 320.0
    cy: float = 240.0


# ── Keyframe Extraction ─────────────────────────────────────────────────────
@dataclass
class KeyframeConfig:
    # Velocity-based boundary detection
    velocity_threshold: float = 0.05      # normalised optical flow magnitude
    min_action_gap_frames: int = 10       # minimum frames between two actions
    max_keyframes_per_action: int = 3     # frames sampled inside each segment
    # Fallback: uniform sampling if velocity detection finds too few boundaries
    fallback_fps: float = 1.0            # frames per second for uniform fallback
    # Object detection for context extraction
    use_object_detection: bool = True


# ── Action Extraction ───────────────────────────────────────────────────────
@dataclass
class ActionConfig:
    # Output schema version
    schema_version: str = "1.0"
    # Confidence threshold — discard actions below this
    min_confidence: float = 0.3
    # Enable chain-of-thought reasoning in VLM prompt
    chain_of_thought: bool = True
    # Few-shot examples to include (0 = zero-shot)
    few_shot_examples: int = 2


# ── Top-level Pipeline Config ────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    rgbd: RGBDConfig = field(default_factory=RGBDConfig)
    keyframe: KeyframeConfig = field(default_factory=KeyframeConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    # Task description injected into VLM prompt
    task_description: str = "a kitchen robot performing a meal preparation task"
    # Output directory for JSON results
    output_dir: str = "output"
    # Verbosity
    verbose: bool = True
