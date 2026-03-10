# REFLECT RGBD → VLM → AKG Action Extraction Pipeline

Full end-to-end pipeline for acquiring robot actions from RGBD video frames
using a local Ollama vision-language model, outputting AKG-grounded JSON.

```
RGBD video  →  Depth Fusion (HHA)  →  Keyframe Extraction  →  Ollama VLM  →  AKG JSON
```

---

## Project Structure

```
reflect_pipeline/
├── pipeline.py                  ← Main entry point (CLI)
├── config.py                    ← All configuration dataclasses
├── core/
│   ├── rgbd_preprocessing.py    ← HHA encoding, depth-as-text, frame fusion
│   ├── keyframe_extraction.py   ← Optical-flow velocity boundary detection
│   ├── vlm_client.py            ← Ollama API client + AKG prompt engineering
│   └── action_serializer.py     ← JSON output schema + pretty-print summary
├── data/
│   └── reflect_loader.py        ← REFLECT dataset loader + synthetic generator
├── tests/
│   └── test_pipeline.py         ← 18 unit tests (all stages)
└── output/                      ← JSON results written here
```

---

## Quick Start

### 1. Requirements

```bash
# Python packages (all pre-installed in standard ROS2 / robotics environments)
pip install opencv-python numpy scipy Pillow requests
```

### 2. Configure Ollama

Edit `config.py`:

```python
@dataclass
class OllamaConfig:
    base_url: str = "http://YOUR_OPENWEBUI_HOST:11434"  # ← your server
    model: str = "llava:13b"                             # ← any vision model
```

Or pass via CLI flags: `--ollama_url http://... --model llava:13b`

**Recommended Ollama vision models** (pull before running):
- `llava:13b` — best accuracy/speed balance
- `llava:34b` — higher accuracy, slower
- `bakllava:7b` — fastest, lower accuracy
- `llava-llama3:8b` — good general-purpose choice

```bash
ollama pull llava:13b
```

### 3. Run

#### Synthetic data (test without any real input)
```bash
python pipeline.py --synthetic --episodes 3
```

#### REFLECT benchmark dataset
```bash
# Download from: https://www.cs.columbia.edu/~liuzeyi/reflect_data/
python pipeline.py --reflect_data /path/to/reflect_data/ --task boilWater
```

#### Your own RGB video
```bash
python pipeline.py --video /path/to/video.mp4 --task_desc "robot sautéing vegetables"
```

#### RGB video + depth frames
```bash
python pipeline.py \
  --video /path/to/rgb_video.mp4 \
  --depth_dir /path/to/depth_frames/ \
  --depth_strategy hha \
  --task_desc "robot boiling water in a pot"
```

#### Directory of PNG frames
```bash
python pipeline.py \
  --rgb_dir /path/to/rgb_frames/ \
  --depth_dir /path/to/depth_frames/ \
  --task_desc "robot making coffee"
```

#### Test preprocessing without VLM
```bash
python pipeline.py --synthetic --dry_run
```

---

## Configuration Reference

All settings are in `config.py` as Python dataclasses. Key options:

| Config class | Key parameter | Default | Description |
|---|---|---|---|
| `OllamaConfig` | `base_url` | `localhost:11434` | Ollama / OpenWebUI URL |
| `OllamaConfig` | `model` | `llava:13b` | Vision model name |
| `RGBDConfig` | `depth_strategy` | `"hha"` | `hha` / `depth_channel` / `depth_as_text` |
| `RGBDConfig` | `depth_scale` | `0.001` | Raw depth → metres (mm→m for RealSense) |
| `KeyframeConfig` | `velocity_threshold` | `0.05` | Motion boundary sensitivity |
| `KeyframeConfig` | `min_action_gap_frames` | `10` | Minimum frames between actions |
| `KeyframeConfig` | `max_keyframes_per_action` | `3` | Keyframes sent to VLM per segment |
| `ActionConfig` | `min_confidence` | `0.3` | Discard actions below this confidence |
| `ActionConfig` | `chain_of_thought` | `True` | Enable CoT reasoning in prompt |
| `ActionConfig` | `few_shot_examples` | `2` | Few-shot examples in prompt (0=zero-shot) |

---

## Output Format

Each episode produces a JSON file in `output/`:

```json
{
  "schema_version": "1.0",
  "task_description": "robot sautéing carrot slices in a saucepan",
  "source_video": "sauteeCarrot_episode1.mp4",
  "processed_at": "2026-03-10T11:00:00Z",
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
      "objects_involved": ["knife", "cutting_board"],
      "contact_type": "grasp",
      "spatial_relation": "above cutting board",
      "depth_context": "knife tip at 0.35m",
      "confidence": 0.91,
      "reasoning": "End-effector moves toward cutting board..."
    }
  ],
  "summary": {
    "total_actions": 5,
    "action_counts": {"PICK_AND_PLACE": 2, "CUTTING": 1, "COOK_COOL": 1, "PREPARING": 1},
    "mean_confidence": 0.87,
    "coverage_s": 38.5,
    "low_confidence_segments": []
  }
}
```

### AKG Action Core Taxonomy (Kümpel et al. 2025)

| Action Core | Description | Example sub-actions |
|---|---|---|
| `CUTTING` | Slicing/dicing/chopping with a blade | slice, dice, chop, mince |
| `POURING` | Transferring liquid or granular material | pour, fill, drain, sprinkle |
| `MIXING` | Combining/blending ingredients | stir, whisk, blend, fold |
| `PREPARING` | Setup actions | peel, wash, measure, open, arrange |
| `PICK_AND_PLACE` | Grasp and relocate an object | grasp, transport, place, stack |
| `COOK_COOL` | Apply/remove heat | heat, boil, fry, microwave, cool |

---

## RGBD Depth Strategies

### `hha` (recommended)
Encodes depth as an HHA pseudo-colour image (Horizontal disparity, Height above ground, Angle with gravity) placed side-by-side with the RGB frame. No VLM architectural change needed.

### `depth_as_text`
Extracts a 3×3 grid of median depth values as a text string prepended to the VLM prompt. Works with any Ollama vision model, even those not expecting RGBD.

### `depth_channel`
Blends normalised depth into the blue channel of the RGB image. Simplest approach, lowest geometric fidelity.

---

## Pipeline Stages

### Stage 1 — RGBD Preprocessing
- Converts raw 16-bit depth (millimetres) to metric depth map
- Computes HHA encoding or depth text summary
- Generates surface normal map for each frame

### Stage 2 — Keyframe Extraction
- Computes Farneback dense optical flow between consecutive frames
- Detects velocity minima (action boundaries) using peak detection
- Samples 1–3 representative keyframes per action segment

### Stage 3 — VLM Inference (Ollama)
- Sends keyframe images + structured prompt to Ollama vision model
- System prompt encodes AKG ontology and output schema
- Chain-of-thought reasoning before JSON output
- Parses and validates JSON response

---

## Running Tests

```bash
python tests/test_pipeline.py
```

Expected output: 18 tests, all pass (≈17s, no network needed).

---

## Extending the Pipeline

### Adding a new depth strategy
1. Add a new branch in `core/rgbd_preprocessing.py::fuse_rgbd_frame()`
2. Add the strategy name to `RGBDConfig.depth_strategy` type hint

### Changing the VLM backend to NVIDIA Cosmos Nemotron
Replace `OllamaVLMClient` with an `NVIDIAVLMClient` using the same interface:
```python
# core/vlm_client_nvidia.py
import requests

class NVIDIAVLMClient:
    def __init__(self, api_key, model="nvidia/cosmos-nemotron-34b"):
        self.api_key = api_key
        self.model = model

    def query_segment(self, bundles, seg, task_description, action_cfg):
        # Use NVIDIA NIM API (OpenAI-compatible)
        ...
```

### Integrating with CRAM / ROS2
The output JSON maps directly to CRAM plan steps. Each `action_core` entry
corresponds to a CRAM designator:
```lisp
;; PICK_AND_PLACE → (perform (a motion (type grasping) ...))
;; CUTTING        → (perform (an action (type cutting) ...))
```

---

## References

- Liu et al. (2023). REFLECT. CoRL 2023. arXiv:2306.15724
- Kümpel et al. (2025). Actionable Knowledge Graphs. Frontiers in Robotics and AI
- Wang et al. (2024). SeeDo. arXiv:2410.08792
- NVIDIA (2025). VLM Prompt Engineering Guide. developer.nvidia.com
