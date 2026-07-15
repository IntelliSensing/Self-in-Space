from dataclasses import dataclass, field
from typing import Optional

import transformers

@dataclass
class ModelArguments:
    model_type: str = field(default="motion-mllm")
    videoflow_checkpoints_path: Optional[str] = field(default="checkpoints/VideoFlow/MOF_kitti.pth")
    connector_type: str = field(default="visual_flow")

    pretrained_model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-7B-Instruct")
    tune_mm_llm: bool = field(default=True)
    tune_mm_vision: bool = field(default=False)
    tune_mm_connector: bool = field(default=True)

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=512 * 28 * 28)
    video_min_frame_pixels: int = field(default=128 * 28 * 28)
    video_max_pixels: int = field(default=512 * 28 * 28)
    video_min_pixels: int = field(default=128 * 28 * 28)
    video_fps: Optional[float] = field(default=2.0)
    # If set, treat the input video frames as if they were sampled at this FPS (nominal FPS).
    # Used to compute the temporal spacing (second_per_grid_ts) for RoPE, especially when videos
    # are already provided as pre-extracted frames and the original FPS is unknown/unreliable.
    video_frame_fps: Optional[int] = field(default=None)
    data_packing: bool = field(default=False)
    data_flatten: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    # LoRA
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.0)
