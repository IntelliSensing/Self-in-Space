<div align="center">

# ✨Self-in-Space✨: Benchmarking Self-Awareness and Spatial Cognition in UAV Embodied Intelligence

<p align="center">
    <a href="https://choucisan.github.io">Zhishan Zou</a><sup>1</sup>,
    <a href="https://github.com/sunguoyan17-alt">Guoyan Sun</a><sup>1</sup>,
    <a href="https://trentonwei.github.io">Zhiwei Wei</a><sup>2</sup>,
    <a href="https://jianchengpan.space">Jiancheng Pan</a><sup>3</sup>,
    <a href="https://github.com/Davidup1">Yujie Li</a><sup>1</sup>,
    <a href="https://teacher.bupt.edu.cn/pengmugen/zh_CN/index.htm">Mugen Peng</a><sup>1</sup>,
    <a href="https://teacher.bupt.edu.cn/xuwenjia/zh_CN/index.htm">Wenjia Xu</a><sup>1&dagger;</sup>
    <br>
    <sup>&dagger;</sup>Corresponding author
    <br>
    <sup>1</sup>Beijing University of Posts and Telecommunications &nbsp;
    <sup>2</sup>Hunan Normal University &nbsp;
    <sup>3</sup>Tsinghua University
    <br>
    ACM MM 2026
</p>

<a href="https://arxiv.org/abs/2607.12477"><img src="https://img.shields.io/badge/arXiv-2607.12477-b31b1b?style=for-the-badge&logo=arxiv" alt="arXiv"></a> &nbsp;
<a href="https://choucisan.github.io/publications/self-in-space"><img src="https://img.shields.io/badge/%F0%9F%8C%90_Website-Project_Page-blue?style=for-the-badge" alt="Website"></a> &nbsp;
<a href="https://github.com/IntelliSensing/Self-in-Space"><img src="https://img.shields.io/badge/GitHub-Self--in--Space-181717?style=for-the-badge&logo=github" alt="GitHub"></a> &nbsp;
<a href="https://huggingface.co/datasets/choucsan/SIS-Bench"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-SIS--Bench-yellow?style=for-the-badge" alt="Hugging Face"></a> &nbsp;
<a href="https://www.modelscope.cn/collections/choucisan/Self-in-Space"><img src="https://img.shields.io/badge/ModelScope-Collection-5E4AF5?style=for-the-badge&logo=modelscope" alt="ModelScope"></a> &nbsp;
<a href="https://choosealicense.com/licenses/apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-green?style=for-the-badge" alt="License"></a>

![Teaser](images/teaser.jpeg)

</div>

<strong>Self-in-Space:</strong> We study spatial intelligence in embodied UAV scenarios from two complementary perspectives — <strong>Space</strong> and <strong>Self</strong>. We introduce SIS-Bench, SIS-Motion-54K, and SIS-Motion: a benchmark, a training dataset, and a motion-aware model for spatial reasoning.


## 📢 News
* 🎉 [2026/7/17] Our open-source models and datasets have accumulated over 1 million downloads.
* 🎉 [2026/7/16] Our open-source models and datasets have accumulated over 500k downloads.
* 🔥 [2026/7/14] Our [paper](https://arxiv.org/abs/2607.12477) is available on arXiv.
* 😊 [2026/7/13] We release our benchmarking dataset [SIS-Bench](https://huggingface.co/datasets/choucsan/SIS-Bench) on Hugging Face and [ModelScope](https://www.modelscope.cn/datasets/choucisan/SIS-Bench).
* 😊 [2026/7/13] We release our training dataset [SIS-Motion-54K](https://huggingface.co/datasets/choucsan/SIS-Motion-54K) on Hugging Face and [ModelScope](https://www.modelscope.cn/datasets/choucisan/SIS-Motion-54K-Dataset).
* 😊 [2026/7/13] We release our downstream task dataset [OpenUAV-QA](https://huggingface.co/datasets/choucsan/OpenUAV-QA) on Hugging Face and [ModelScope](https://www.modelscope.cn/datasets/choucisan/OpenUAV-QA).
* 😊 [2026/7/13] We release our model [SIS-Motion](https://huggingface.co/choucsan/SIS-Motion) on Hugging Face and [ModelScope](https://www.modelscope.cn/models/choucisan/SIS-Motion).
* 🎉 [2026/7/10] Our paper is accepted by ACM MM 2026!

## 🌟 Overview

### 🍏 SIS-Bench
![Teaser](images/bench-tasks.webp)
<strong>SIS-Bench:</strong> We introduce a benchmark of 4,856 QA pairs from 1,646 real-world UAV videos, evaluating embodied spatial intelligence across two dimensions (spatial cognition and self-awareness) and three cognitive levels (perception, memory, and reasoning).

### 🏗️ Pipeline
![Teaser](images/pipeline.webp)
<strong>Pipeline:</strong> SIS-Bench is built through a four-stage task-conditioned protocol: heterogeneous video processing, task-specific annotation, LLM-assisted QA construction, and dual-expert verification, ensuring scalability and reliability across 13 tasks.


### ✈️ SIS-Motion
![Teaser](images/sis-motion-framework.webp)
<strong>SIS-Motion:</strong> A motion-aware extension of a video MLLM that fuses optical-flow-based motion cues with visual features, jointly capturing environmental context and agent dynamics to improve both spatial cognition and self-awareness.


## 🧩 SIS-Motion Architecture

SIS-Motion extends `Qwen/Qwen2.5-VL-7B-Instruct` with a frozen VideoFlow MOFNet optical-flow estimator, a trainable visual-flow connector, and LoRA adapters on the language model. The base Qwen weights and vision encoder remain frozen.

```text
video frames
├── Qwen2.5-VL vision encoder (frozen) ───────────────> appearance tokens
└── VideoFlow MOFNet optical-flow estimator (frozen)
    └── forward/backward optical flow
        └── forward flow (dx, dy)
            └── pseudo-images [magnitude, dx, dy]
                └── shared Qwen vision encoder ──────> motion tokens

motion tokens -> LayerNorm -> Linear(3584, 2048) -> GELU
              -> Linear(2048, 3584)
appearance tokens + projected motion tokens -> Qwen2.5-VL language model + LoRA
```

MOFNet predicts four flow channels: forward `(dx, dy)` and backward `(dx, dy)`. The current fusion path uses only the two forward-flow channels to construct the three-channel pseudo-image. `MOF_kitti.pth` therefore contains the optical-flow estimator, not the connector.

| Component | Parameters | Training state | Included in SIS-Motion package |
|---|---:|---|:---:|
| Qwen2.5-VL-7B base model | See base model | Frozen base; LoRA on LLM | No |
| VideoFlow MOFNet | 13,453,240 | Frozen | Yes |
| Visual-flow connector | 14,692,864 | Trainable | Yes |
| LoRA adapter (`r=32`, `alpha=64`) | 20,185,088 | Trainable | Yes |
| **Total trainable** | **34,877,952** | Connector + LoRA | Yes |


## ⚙️ Getting Started

### 🎲 Clone

```bash
git clone https://github.com/IntelliSensing/Self-in-Space.git
cd Self-in-Space
```

### 🎹 Installation

The project requires Python 3.10, PyTorch 2.6 / CUDA 12.4, and an NVIDIA GPU with a compatible driver. LoRA and SIS-Motion setup installs FlashAttention from a compatible wheel when available; a matching CUDA toolkit, C++ compiler, and `CUDA_HOME` are required only when it must build from source. Evaluation and training use separate environments (vLLM vs. Transformers). Both Conda and uv are supported.

With Conda:

```bash
# Install one or more profiles
bash scripts/setup_conda.sh eval      # vLLM evaluation → conda env: sis-motion-eval
bash scripts/setup_conda.sh lora      # LoRA training/eval → conda env: sis-motion-lora
bash scripts/setup_conda.sh motion    # SIS-Motion training/eval → conda env: sis-motion-motion

conda activate sis-motion-motion
```

With [uv](https://docs.astral.sh/uv/):

```bash
bash scripts/setup_uv.sh eval         # → .venv-eval/
bash scripts/setup_uv.sh lora         # → .venv-lora/
bash scripts/setup_uv.sh motion       # → .venv-motion/

source .venv-motion/bin/activate
```

| Environment | Purpose | Key Components | FlashAttn |
|---|---|---|---|
| eval | vLLM inference benchmark | vllm==0.8.5, torch==2.6.0+cu124 | No |
| lora | LoRA training + Transformers eval | deepspeed, peft, flash-attn | Yes |
| motion | SIS-Motion training + eval | deepspeed, peft, flash-attn, VideoFlow | Yes |

To use a different CUDA stack, edit the definitions under `environments/` directly.

### 📊 Data Preparation

All data paths are defined in [`configs/data_registry.json`](configs/data_registry.json) and resolve to `data/` by default.

```bash
# Download all datasets
python scripts/download_data.py

# Or download selectively
python scripts/download_data.py sis_bench
python scripts/download_data.py sis_motion_54k
python scripts/download_data.py openuav_qa
```

For an external data directory:

```bash
export SIS_DATA_ROOT=/path/to/self-in-space-data
python scripts/data_registry.py validate
```

Expected layout after download:

```text
data/
├── SIS-Bench/
│   ├── SIS-Bench.jsonl              # 4,856 QA pairs
│   └── video/
│       ├── AirScape/                # 1,156 videos
│       ├── UrbanVideo/              # 427 videos
│       └── VisDrone/                # 63 videos
├── SIS-Motion-54K/
│   ├── SIS-Motion-54K.jsonl
│   └── AirScape_dataset/            # training videos
└── OpenUAV-QA/
    ├── TravelUAV_test.jsonl         # 3,895 QA pairs
    └── TravelUAV_dataset/           # frame sequences
```

SIS-Bench and SIS-Motion-54K accept raw videos (`.mp4`, `.mov`, `.avi`, `.mkv`, or `.webm`) and ordered frame directories containing `.jpg`, `.jpeg`, or `.png` files. If a JSONL entry references `sample.mp4`, a same-level `sample/` frame directory is also accepted. OpenUAV-QA uses ordered image-frame directories. No additional registration is needed when the data follows the layout above.

| Resource | Eval | LoRA | Motion | Frame directories | Raw video |
|---|:---:|:---:|:---:|:---:|:---:|
| SIS-Bench evaluation | Yes | Yes | Yes | Yes | Yes |
| SIS-Motion-54K training | — | Yes | Yes | Yes | Yes |
| OpenUAV-QA evaluation | — | Yes | Yes | Yes | — |
| SIS-Motion inference | — | — | Yes | Yes | Yes |

### 📊 Benchmark Evaluation (vLLM)

Evaluate an open-source model supported by the adapters under `eval/src/uav/utils/` on SIS-Bench. Qwen2.5-VL is the default verified vLLM path:

```bash
conda activate sis-motion-eval

MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct \
TENSOR_PARALLEL_SIZE=1 \
bash scripts/eval.sh
```

Change `MODEL_ID` and `TENSOR_PARALLEL_SIZE` for different models and GPU counts. Use `DATA_FILE` and `FRAMES_DIR` to evaluate an unregistered local dataset.

> **Note**: vLLM and Flash Attention version differences may lead to inconsistent evaluation results. Ensure all models under comparison are evaluated in the same environment.

### 🚀 LoRA Baseline Training

Fine-tune visual-only LoRA adapters on SIS-Motion-54K:

```bash
conda activate sis-motion-lora

DATASET_USE=sis_motion_54k \
CUDA_VISIBLE_DEVICES=0,1 \
OUTPUT_DIR=$PWD/lora/output/qwen3vl-4b-lora \
bash scripts/train_lora.sh
```

Defaults to `Qwen/Qwen3-VL-4B-Instruct`. Override with `MODEL_NAME_OR_PATH`. Training uses DeepSpeed ZeRO-2 on all visible GPUs; set `CUDA_VISIBLE_DEVICES` to select devices.

### 📊 LoRA Baseline Evaluation

Evaluate the trained LoRA checkpoint:

```bash
conda activate sis-motion-lora

MODEL_PATH=$PWD/lora/output/qwen3vl-4b-lora \
bash scripts/eval_lora.sh
```

### 🚀 SIS-Motion Training

Download [`MOF_kitti.pth`](https://drive.google.com/drive/folders/16YqDD_IQpzrVWvDHI9xK3kO0MaXnNIGx) and place it at `checkpoints/VideoFlow/MOF_kitti.pth`.

Train the motion-aware adapter (frozen Qwen base and VideoFlow, trainable connector and LoRA):

```bash
conda activate sis-motion-motion

DATASET_USE=sis_motion_54k \
CUDA_VISIBLE_DEVICES=0,1,2 \
PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
OUTPUT_DIR=$PWD/motion/output/sis-motion \
bash scripts/train_motion.sh
```

Set `VIDEOFLOW_CKPT` to override the checkpoint path. Training uses DeepSpeed ZeRO-2 on all visible GPUs.

For SwanLab experiment tracking:

```bash
SWANLAB_ENABLED=1 bash scripts/train_motion.sh
```

### 📊 SIS-Motion Evaluation

Ensure the SIS-Motion model package is placed under `model/` (or available on Hugging Face):

```bash
conda activate sis-motion-motion

# From Hugging Face
MODEL_PATH=choucsan/SIS-Motion \
bash scripts/eval_motion.sh

# Local model/ directory (auto-detected)
bash scripts/eval_motion.sh

# Training output (auto-detects latest checkpoint)
MODEL_DIR=$PWD/motion/output/sis-motion \
bash scripts/eval_motion.sh
```

The evaluator reads `base_model_name_or_path` from `adapter_config.json` and downloads the Qwen2.5-VL-7B base weights automatically.

The Hugging Face model repository is intentionally lightweight and does not duplicate the Qwen2.5-VL-7B base weights:

```text
model/
├── adapter_config.json
├── adapter_model.safetensors
├── connector_weights.pt
├── MOF_kitti.pth
└── sis_motion_config.json
```

### 📊 Downstream Evaluation (OpenUAV-QA)

```bash
# LoRA baseline
conda activate sis-motion-lora
BENCHMARK=openuav_qa \
MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct \
bash scripts/eval_lora.sh

# SIS-Motion
conda activate sis-motion-motion
BENCHMARK=openuav_qa \
MODEL_PATH=choucsan/SIS-Motion \
bash scripts/eval_motion.sh
```

> **Note**: The paper pre-processes videos with the same sampling strategy as Qwen2.5-VL to reduce GPU memory pressure. See the paper appendix for details. This codebase reproduces the paper setting only; using raw MP4 inputs may introduce discrepancies. Use a consistent input format when comparing against baselines.


## 📚 Citation

```bibtex
@misc{zou2026sis,
      title={Self in Space: Benchmarking Self-Awareness and Spatial Cognition in UAV Embodied Intelligence},
      author={Zhishan Zou and Guoyan Sun and Zhiwei Wei and Jiancheng Pan and Yujie Li and Mugen Peng and Wenjia Xu},
      year={2026},
      eprint={2607.12477},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.12477},
}
```



## 📖 References and Acknowledgements

This project builds on the following open-source models, libraries, and data
resources:

- **Code**: [Qwen2.5-VL and Qwen3-VL](https://github.com/QwenLM/Qwen3-VL),
  [Spatial-MLLM](https://github.com/THU-SI/Spatial-MLLM),
  [VideoFlow](https://github.com/XiaoyuShi97/VideoFlow)
- **Infra**: [vLLM](https://github.com/vllm-project/vllm),
  [Transformers](https://github.com/huggingface/transformers),
  [PEFT](https://github.com/huggingface/peft),
  [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
- **Data**: [AirScape](https://huggingface.co/datasets/EmbodiedCity/AirScape-Dataset),
  [UrbanVideo-Bench](https://huggingface.co/datasets/EmbodiedCity/UrbanVideo-Bench),
  [VisDrone](https://github.com/VisDrone/VisDrone-Dataset),
  [OpenUAV](https://github.com/prince687028/TravelUAV)




## 📮 Contact

For questions, corrections, or collaboration requests:

[choucisan@gmail.com](mailto:choucisan@gmail.com)
