<div align="center">
  <h1>VLM Action Acquisition</h1>
  <a href="https://www.researchgate.net/publication/405824105_Acquiring_Low-Level_Robot_Actions_from_RGBD_Video_Using_Vision-Language_Models"><img src="https://img.shields.io/badge/Paper-blue?style=flat-square"></a>
</div>

## 📖 Introduction
This project processes RGB-D robot trajectories and utilizes Vision-Language Models (VLMs) to extract and serialize structured action steps. The pipeline handles keyframe extraction, RGB-D preprocessing, keyframe extraction from velocity signals, VLM inference with structured ontology prompts, and AKG grounding.

| **Four-Stage Pipeline** | **Optical-Flow Velocity** |
|:---:|:---:|
| ![Pipeline](./data/pipeline_architecture.png) | ![Velocity](./data/optical_flow_velocity.png) |

## ⚙️ Installation
Install the required dependencies from `requirement.txt`:

```bash
pip install -r requirement.txt
```

## Usage

You can run the full extraction pipeline with the following command (replace the credentials with your own):

```bash
python pipeline.py \
  --reflect_data ./real_data \
  --frame_step 3 \
  --ollama_url <Local Hosted server> \
  --user_id <your_email> \
  --api_key <your_api_key> \
  --model qwen3-vl:latest
```
