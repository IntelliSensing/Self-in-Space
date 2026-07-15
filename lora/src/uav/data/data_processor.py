
import json
import random
import logging
import re
import time
import itertools
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple, Any
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import av
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers.video_utils import VideoMetadata

import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_3

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _make_abs_paths(base: Path, files: str) -> str:
    return f"{(base / files).resolve()}"


def update_processor_pixels(processor, data_args):
    logger = logging.getLogger(__name__)

    # --- Image Processor ---
    ip = processor.image_processor
    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"ip.size: {ip.size}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels
        rank0_print(f"✅ Updated image_processor min_pixels to {data_args.min_pixels}")
        rank0_print(f"✅ Updated image_processor max_pixels to {data_args.max_pixels}")

    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = data_args.min_pixels
        ip.size["longest_edge"] = data_args.max_pixels
        rank0_print(
            f"✅ Updated image_processor size['shortest_edge'] to {data_args.min_pixels}"
        )
        rank0_print(
            f"✅ Updated image_processor size['longest_edge'] to {data_args.max_pixels}"
        )

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        rank0_print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args.video_min_pixels
            vp.max_pixels = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated video_processor min_pixels to {data_args.video_min_pixels}"
            )
            rank0_print(
                f"✅ Updated video_processor max_pixels to {data_args.video_max_pixels}"
            )

        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
            rank0_print(
                f"✅ Updated video_processor min_frames to {data_args.video_min_frames}"
            )
            rank0_print(
                f"✅ Updated video_processor max_frames to {data_args.video_max_frames}"
            )

        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps
            rank0_print(f"✅ Updated video_processor fps to {data_args.video_fps}")

        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = data_args.video_min_pixels
            vp.size["longest_edge"] = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
            )
            rank0_print(
                f"✅ Updated Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}"
            )

        rank0_print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

    return processor


def _build_messages(item: Dict[str, Any], base_path: Path) -> List[Dict[str, Any]]:
    # Extract and normalize images and videos
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    # Build media pools with absolute paths
    image_pool = [
        {"type": "image", "image": _make_abs_paths(base_path, img)} for img in images
    ]
    video_pool = [
        {"type": "video", "video": _make_abs_paths(base_path, vid)} for vid in videos
    ]

    messages = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError(
                            "Number of <video> placeholders exceeds the number of provided videos"
                        )
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            # Assistant messages contain only text
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # Check for unused media files
    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    if video_pool:
        raise ValueError(
            f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)"
        )

    return messages


def _load_frames_from_dir(dir_path, max_frames=16):
    """Load PIL Image list from a frame directory, sorted by filename for temporal order."""
    frame_files = sorted(
        [f for f in dir_path.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    )
    if len(frame_files) > max_frames:
        indices = np.linspace(0, len(frame_files) - 1, max_frames, dtype=int)
        frame_files = [frame_files[i] for i in indices]
    return [Image.open(str(f)).convert("RGB") for f in frame_files]


def _load_frames_from_video(video_path, max_frames=16):
    """Decode uniformly sampled RGB frames from a video file."""
    max_frames = max_frames or 32
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        total_frames = int(stream.frames or 0)
        if total_frames <= 0 and stream.duration and stream.average_rate:
            total_frames = int(float(stream.duration * stream.time_base * stream.average_rate))

        target_indices = None
        if total_frames > max_frames:
            target_indices = set(np.linspace(0, total_frames - 1, max_frames, dtype=int))

        frames = []
        for index, frame in enumerate(container.decode(stream)):
            if target_indices is None or index in target_indices:
                frames.append(frame.to_image().convert("RGB"))
            if target_indices and index >= max(target_indices):
                break

    if not frames:
        raise ValueError(f"No decodable frames found in {video_path}")
    if target_indices is None and len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in indices]
    return frames


def _load_video_frames(video_path, max_frames=16):
    if not video_path.exists():
        frame_path = video_path.with_suffix("")
        if frame_path.is_dir():
            video_path = frame_path
        elif not video_path.suffix:
            for suffix in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
                candidate = video_path.with_suffix(suffix)
                if candidate.is_file():
                    video_path = candidate
                    break
            else:
                raise FileNotFoundError(f"Video input not found: {video_path}")
        else:
            raise FileNotFoundError(f"Video input not found: {video_path}")
    if video_path.is_dir():
        return _load_frames_from_dir(video_path, max_frames=max_frames)
    if video_path.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
        return _load_frames_from_video(video_path, max_frames=max_frames)
    raise ValueError(f"Unsupported video input: {video_path}")


def _build_video_processor_kwargs(messages, default_fps: float) -> Dict[str, Any]:
    """
    Qwen3-VL will otherwise try to re-sample pre-selected frame lists using fps-based logic.
    When we already provide sampled frames, pass explicit metadata and disable re-sampling.
    """
    video_metadata = []

    for message in messages:
        for content in message.get("content", []):
            if content.get("type") != "video" or "video" not in content:
                continue

            video_value = content["video"]
            if isinstance(video_value, (str, Path)):
                return {}
            if not isinstance(video_value, Sequence) or len(video_value) == 0:
                return {}

            first_frame = video_value[0]
            if not isinstance(first_frame, (Image.Image, np.ndarray, torch.Tensor)):
                return {}

            video_metadata.append(
                VideoMetadata(
                    total_num_frames=len(video_value),
                    fps=default_fps,
                    frames_indices=list(range(len(video_value))),
                )
            )

    if not video_metadata:
        return {}

    return {
        "do_sample_frames": False,
        "video_metadata": video_metadata,
    }


def _build_uav_messages(item, base_path):
    """Build chat messages from UAV JSONL format (supports mc and open qa_types)."""
    # 1. Resolve a frame directory or video file.
    video_path = Path(item["video_path"])
    if video_path.is_absolute():
        video_input = video_path
    else:
        # Relative path: e.g. "AirScape_dataset/AirScape_Train_0732.mp4"
        item_data_path = item.get("data_path", "")
        effective_base = Path(item_data_path) if item_data_path else base_path
        video_input = (effective_base / video_path).resolve()

    # 2. Load frames as PIL Image list (use video_max_frames from data_args if available)
    max_frames = item.get("video_max_frames", 32)
    frames = _load_video_frames(video_input, max_frames=max_frames)

    # 3. Build question/answer based on qa_type
    qa_type = item.get("qa_type", "mc")
    question = item["question"]

    if qa_type == "mc" and "options" in item:
        # Multiple-choice: show options, answer is letter (A/B/C/D)
        options_text = "\n".join(f"{k}. {v}" for k, v in sorted(item["options"].items()))
        user_text = f"{question}\n{options_text}\nPlease select the correct answer."
        assistant_text = item["answer"]
    else:
        # Open-ended (open_qa, motion_intention): answer is descriptive text
        user_text = question
        assistant_text = item["answer"]

    # 4. Build messages in the format expected by processor.apply_chat_template
    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": user_text},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": assistant_text},
        ]},
    ]
    return messages


def preprocess_qwen_visual(
    sources,
    processor,
) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = Path(source.get("data_path", ""))

    # UAV raw format: has "question" field (mc has "options", open/motion do not)
    if "question" in source and ("options" in source or "qa_type" in source):
        messages = _build_uav_messages(source, base_path)
    else:
        messages = _build_messages(source, base_path)

    video_processor = getattr(processor, "video_processor", None)
    video_kwargs = _build_video_processor_kwargs(
        messages,
        default_fps=float(getattr(video_processor, "fps", 2.0) or 2.0),
    )
    full_result = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        **video_kwargs,
    )

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)

    # Dynamically resolve role/boundary token IDs (works across Qwen2.5/Qwen3)
    _assistant_token_id = processor.tokenizer.convert_tokens_to_ids("assistant")
    _im_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    pos = 0
    while pos < L:
        if input_ids_flat[pos] == _assistant_token_id:
            ans_start = pos + 2  # skip "assistant" + "\n"
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != _im_end_token_id:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[
                    0, ans_start : ans_end + 2
                ]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    return full_result


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, processor, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif data_args.model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
        else:
            raise ValueError(f"model_type: {data_args.model_type} not supported")

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                rank0_print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                if isinstance(ann, list):
                    for sub_ann in ann:
                        sub_ann["data_path"] = data["data_path"]
                        sub_ann["video_max_frames"] = data_args.video_max_frames
                else:
                    ann["data_path"] = data["data_path"]
                    ann["video_max_frames"] = data_args.video_max_frames
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")


        rank0_print("Formatting inputs...Skip in lazy mode")
        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.list_data_dict = list_data_dict

        if data_args.data_packing:
            self.item_fn = self._get_packed_item
        else:
            self.item_fn = self._get_item

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            if "conversations" in sample:
                img_tokens = 128 if "image" in sample else 0
                length_list.append(
                    sum(len(conv["value"].split()) for conv in sample["conversations"])
                    + img_tokens
                )
            else:
                # UAV format: estimate from question + options + answer
                text_len = len(sample.get("question", "").split())
                for v in sample.get("options", {}).values():
                    text_len += len(v.split())
                text_len += len(sample.get("answer", "").split())
                length_list.append(text_len + 128)  # add video token estimate
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            if "conversations" in sample:
                cur_len = sum(
                    len(conv["value"].split()) for conv in sample["conversations"]
                )
                cur_len = (
                    cur_len if ("image" in sample) or ("video" in sample) else -cur_len
                )
            else:
                # UAV format: always has video
                cur_len = len(sample.get("question", "").split())
                for v in sample.get("options", {}).values():
                    cur_len += len(v.split())
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sources = self.list_data_dict[i]
                if isinstance(sources, dict):
                    sources = [sources]
                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                sources = self.list_data_dict[next_index]
                if isinstance(sources, dict):
                    sources = [sources]

                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sources = self.list_data_dict[i]
            if isinstance(sources, dict):
                sources = [sources]
            sample = self.item_fn(sources)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        data_dict = preprocess_qwen_visual(
            sources,
            self.processor,
        )

        seq_len = data_dict["input_ids"][0].size(0)

        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        if "video_grid_thw" in data_dict:
            video_grid_thw = data_dict.get("video_grid_thw")
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw = [video_grid_thw]
            second_per_grid_ts = [
                self.processor.video_processor.temporal_patch_size
                / self.processor.video_processor.fps
            ] * len(video_grid_thw)
        else:
            video_grid_thw = None
            second_per_grid_ts = None

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]

        text = self.processor.tokenizer.decode(
            data_dict["input_ids"][0], skip_special_tokens=False
        )

        labels = data_dict["labels"][0]
        labels = [
            tid if tid != -100 else self.processor.tokenizer.pad_token_id
            for tid in labels
        ]
        label = self.processor.tokenizer.decode(labels, skip_special_tokens=False)

        return data_dict

    def _get_packed_item(self, sources) -> Dict[str, torch.Tensor]:

        if isinstance(sources, dict):
            if isinstance(sources, dict):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            return self._get_item(sources)

        if isinstance(sources, list):
            data_list = []
            new_data_dict = {}
            for source in sources:
                if isinstance(source, dict):
                    source = [source]
                assert (
                    len(source) == 1
                ), f"Don't know why it is wrapped to a list.\n {source}"  # FIXME
                data_list.append(self._get_item(source))

            input_ids = torch.cat([d["input_ids"] for d in data_list], dim=1)
            labels = torch.cat([d["labels"] for d in data_list], dim=1)
            position_ids = torch.cat([d["position_ids"] for d in data_list], dim=2)
            attention_mask = [
                d["attention_mask"][0] for d in data_list if "attention_mask" in d
            ]
            new_data_dict = {
                "input_ids": input_ids,
                "labels": labels,
                "position_ids": position_ids,
                "attention_mask": attention_mask if attention_mask else None,
            }

            if any("pixel_values" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values": torch.cat(
                            [
                                d["pixel_values"]
                                for d in data_list
                                if "pixel_values" in d
                            ],
                            dim=0,
                        ),
                        "image_grid_thw": torch.cat(
                            [
                                d["image_grid_thw"]
                                for d in data_list
                                if "image_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )

            if any("pixel_values_videos" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values_videos": torch.cat(
                            [
                                d["pixel_values_videos"]
                                for d in data_list
                                if "pixel_values_videos" in d
                            ],
                            dim=0,
                        ),
                        "video_grid_thw": torch.cat(
                            [
                                d["video_grid_thw"]
                                for d in data_list
                                if "video_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )
            return new_data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch


def make_supervised_data_module(processor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(processor, data_args=data_args)
    if data_args.data_flatten or data_args.data_packing:
        data_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass
