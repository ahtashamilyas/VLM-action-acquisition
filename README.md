<div align="center">
  <h1>VLM Action Acquisition</h1>
  <a href="#"><img src="https://img.shields.io/badge/Paper-Coming_Soon-blue?style=for-the-badge&logo=arxiv" alt="Paper"></a>
</div>

## 📖 Introduction
This project processes RGB-D robot trajectories and utilizes Vision-Language Models (VLMs) to extract and serialize structured action steps. The pipeline handles keyframe extraction, RGB-D preprocessing, and action generation via models like Qwen-VL.

## ⚙️ Installation
Install the required dependencies from `requirement.txt`:

```bash
pip install -r requirement.txt
```

## 🚀 Usage

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

## 📄 Paper
Paper link: *Coming soon!*
