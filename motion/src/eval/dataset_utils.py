"""SIS-Bench dataset adapter for SIS-Motion evaluation."""
import json
import os
from os.path import join
from pathlib import Path
from typing import List, Optional

import numpy as np
import av
import torch
from PIL import Image
from torch.utils.data import Dataset


UAV_PROMPT_TEMPLATES = {
    # Matches training format in data_processor.py _build_uav_messages()
    "default": """{question}
{options_str}
Please select the correct answer.""",

    "motion": """{question}
{options_str}
Please first describe the drone's motion intention, then select the correct answer.""",
}


class SISBenchDataset(Dataset):
    """
    Dataset for SIS-Bench evaluation with SIS-Motion.

    Returns raw PIL frames + metadata. The caller handles processor encoding
    and video_tchw construction.
    """

    def __init__(
        self,
        data_file: str,
        frames_dir: str,
        max_frames: int = 32,
        min_frames: int = 4,
        prompt_style: str = "default",
    ):
        self.data_file = data_file
        self.frames_dir = frames_dir
        self.max_frames = max_frames
        self.min_frames = min_frames

        if prompt_style not in UAV_PROMPT_TEMPLATES:
            raise ValueError(f"Unknown prompt_style '{prompt_style}', choose from: {list(UAV_PROMPT_TEMPLATES.keys())}")
        self.prompt_template = UAV_PROMPT_TEMPLATES[prompt_style]

        self.data = []
        with open(data_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    item["id"] = item.get("question_id", item.get("id"))
                    self.data.append(item)

        print(f"Loaded {len(self.data)} questions from {data_file}")
        self._validate_frames()

    def _resolve_video_input(self, item: dict) -> Optional[Path]:
        """Resolve either a raw video file or a pre-extracted frame directory."""
        media_root = Path(self.frames_dir)
        references = [item.get("video_name"), item.get("video_path")]
        for reference in dict.fromkeys(ref for ref in references if ref):
            relative_path = Path(reference)
            candidates = [relative_path if relative_path.is_absolute() else media_root / relative_path]
            if not relative_path.is_absolute():
                if relative_path.parts and relative_path.parts[0] == media_root.name:
                    candidates.append(media_root.joinpath(*relative_path.parts[1:]))
                candidates.extend(
                    media_root / subdir / relative_path
                    for subdir in ("video", "TravelUAV_dataset", "AirScape_dataset")
                )

            for candidate in dict.fromkeys(candidates):
                if candidate.is_file() or candidate.is_dir():
                    return candidate
                frame_path = candidate.with_suffix("")
                if frame_path.is_dir():
                    return frame_path
                if not candidate.suffix:
                    for suffix in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
                        video_path = candidate.with_suffix(suffix)
                        if video_path.is_file():
                            return video_path
        return None

    def _video_frame_count(self, video_path: Path) -> int:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            frame_count = int(stream.frames or 0)
            if frame_count <= 0 and stream.duration and stream.average_rate:
                frame_count = int(
                    float(stream.duration * stream.time_base * stream.average_rate)
                )
            return frame_count

    def _validate_frames(self):
        valid_data = []
        missing = 0
        for item in self.data:
            video_input = self._resolve_video_input(item)
            if video_input is None:
                missing += 1
                continue

            if video_input.is_dir():
                frame_count = len([
                    f for f in video_input.iterdir()
                    if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
                ])
            else:
                frame_count = self._video_frame_count(video_input)

            # Some containers do not expose a frame count until decoding.
            if frame_count == 0 or frame_count >= self.min_frames:
                valid_data.append(item)
            else:
                missing += 1
        if missing > 0:
            print(f"[WARN] Skipped {missing} samples (missing frames or < {self.min_frames} frames)")
        self.data = valid_data
        print(f"Valid samples: {len(self.data)}")

    def filter_finish_ids(self, finish_ids: List[str]):
        finish_set = set(finish_ids)
        self.data = [item for item in self.data if item["id"] not in finish_set]

    def filter_by_task_types(self, task_types: List[str]):
        self.data = [item for item in self.data if item.get("task_type") in task_types]

    def __len__(self):
        return len(self.data)

    def _load_frames_from_video(self, video_path: Path):
        """Decode uniformly sampled frames from a video file."""
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            total_frames = int(stream.frames or 0)
            if total_frames <= 0 and stream.duration and stream.average_rate:
                total_frames = int(
                    float(stream.duration * stream.time_base * stream.average_rate)
                )

            target_indices = None
            last_target = None
            if total_frames > self.max_frames:
                target_indices = set(
                    np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
                )
                last_target = max(target_indices)

            frames = []
            for index, frame in enumerate(container.decode(stream)):
                if target_indices is None or index in target_indices:
                    frames.append(frame.to_image().convert("RGB"))
                if last_target is not None and index >= last_target:
                    break

        if not frames:
            raise ValueError(f"No decodable frames found in {video_path}")
        if target_indices is None and len(frames) > self.max_frames:
            indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
            frames = [frames[i] for i in indices]
        return frames

    def _load_frames(self, item: dict):
        """Load and sample video frames as PIL images + raw tensor."""
        video_input = self._resolve_video_input(item)
        if video_input is None:
            raise FileNotFoundError(
                f"Video not found for {item.get('video_name') or item.get('video_path')}"
            )

        if video_input.is_file():
            pil_frames = self._load_frames_from_video(video_input)
        else:
            frame_files = sorted([
                f for f in video_input.iterdir()
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
            ])
            if len(frame_files) > self.max_frames:
                indices = np.linspace(0, len(frame_files) - 1, self.max_frames, dtype=int)
                frame_files = [frame_files[i] for i in indices]
            pil_frames = [Image.open(frame).convert("RGB") for frame in frame_files]

        raw_tensors = [
            torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
            for img in pil_frames
        ]

        # video_tchw: [T, 3, H, W] — pre-resize to motion encoder target resolution
        video_tchw = torch.stack(raw_tensors, dim=0)
        _motion_h, _motion_w = 320, 480
        if video_tchw.shape[2] != _motion_h or video_tchw.shape[3] != _motion_w:
            video_tchw = torch.nn.functional.interpolate(
                video_tchw, size=(_motion_h, _motion_w), mode="bilinear", align_corners=False,
            )

        return pil_frames, video_tchw

    def _format_options(self, options: dict) -> str:
        parts = []
        for key in sorted(options.keys()):
            parts.append(f"{key}. {options[key]}")
        return "\n".join(parts)

    def __getitem__(self, idx):
        item = self.data[idx]

        pil_frames, video_tchw = self._load_frames(item)

        options_str = self._format_options(item["options"])
        prompt_text = self.prompt_template.format(
            question=item["question"],
            options_str=options_str,
        )

        # Build Qwen2.5-VL chat messages
        video_content = [{"type": "video", "video": pil_frames}]
        text_content = [{"type": "text", "text": prompt_text}]
        messages = [{"role": "user", "content": video_content + text_content}]

        metadata = {
            "id": item["id"],
            "question_id": item.get("question_id", item["id"]),
            "video_name": item["video_name"],
            "task_type": item.get("task_type", "unknown"),
            "split": item.get("split", ""),
            "question": item["question"],
            "options": item["options"],
            "answer": item.get("answer", ""),
            "num_frames": len(pil_frames),
        }

        return {
            "messages": messages,
            "video_tchw": video_tchw,
            "metadata": metadata,
        }
