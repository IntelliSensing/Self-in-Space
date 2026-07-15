"""SIS-Bench dataset adapter for the vLLM evaluation pipeline."""
import json
import os
from os.path import join
from pathlib import Path
from typing import List, Optional

import av
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def resize_keep_aspect_ratio(img: Image.Image, max_size: int) -> Image.Image:
    """
    Resize image while maintaining aspect ratio.
    The longer side will be scaled to max_size.

    Args:
        img: PIL Image
        max_size: Maximum size for the longer side

    Returns:
        Resized PIL Image with preserved aspect ratio
    """
    width, height = img.size
    if width >= height:
        # Landscape or square
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        # Portrait
        new_height = max_size
        new_width = int(width * max_size / height)

    return img.resize((new_width, new_height), Image.BICUBIC)


def resize_to_pixel_range(img: Image.Image, min_pixels: int, max_pixels: int) -> Image.Image:
    """
    Resize image to fit within a pixel range while maintaining aspect ratio.

    Args:
        img: PIL Image
        min_pixels: Minimum number of pixels per image
        max_pixels: Maximum number of pixels per image

    Returns:
        Resized PIL Image within the specified pixel range
    """
    width, height = img.size
    current_pixels = width * height

    # Check if current size is within range
    if min_pixels <= current_pixels <= max_pixels:
        return img

    # Calculate target pixel count
    if current_pixels < min_pixels:
        target_pixels = min_pixels
    else:
        target_pixels = max_pixels

    # Calculate scaling factor
    scale_factor = (target_pixels / current_pixels) ** 0.5
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Ensure we don't have zero or negative dimensions
    new_width = max(1, new_width)
    new_height = max(1, new_height)

    return img.resize((new_width, new_height), Image.BICUBIC)


# Prompt templates for SIS-Bench multiple-choice questions
UAV_PROMPT_TEMPLATES = {
    # Default: generic VQA prompt
    "default": """Watch the video carefully and answer the following multiple choice question.

Question:
{question}

Options:
{options_str}

Please select the correct answer from the options above. Only output the letter (A, B, C, or D) corresponding to your answer.""",

    # Motion CoT: matches training prompt in data_processor.py _build_uav_messages()
    "motion": """{question}
{options_str}
Please first describe the drone's motion intention, then select the correct answer.""",
}


class SISBenchDataset(Dataset):
    """
    Dataset for SIS-Bench evaluation.

    Loads questions from JSONL and accepts raw videos or pre-extracted frames.
    Returns data in the format expected by benchvl's vLLM inference pipeline.
    """

    def __init__(
        self,
        data_file: str,
        frames_dir: str,
        max_frames: int = 32,
        image_size: Optional[int] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        prompt_style: str = "default",
    ):
        """
        Args:
            data_file: Path to SIS-Bench.jsonl
            frames_dir: Path to raw videos or pre-extracted frames
            max_frames: Maximum number of frames to use per video
            image_size: Optional size to resize images to (deprecated, use min_pixels/max_pixels)
            min_pixels: Minimum number of pixels per image for vision encoder
            max_pixels: Maximum number of pixels per image for vision encoder
            prompt_style: Prompt template style ("default" or "motion")
        """
        self.data_file = data_file
        self.frames_dir = frames_dir
        self.max_frames = max_frames
        self.image_size = image_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        if prompt_style not in UAV_PROMPT_TEMPLATES:
            raise ValueError(f"Unknown prompt_style '{prompt_style}', choose from: {list(UAV_PROMPT_TEMPLATES.keys())}")
        self.prompt_template = UAV_PROMPT_TEMPLATES[prompt_style]

        # Load metadata for frame counts
        metadata_path = join(frames_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        # Load data from JSONL file
        self.data = []
        with open(data_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    # Add unique id based on question_id (or fallback to id)
                    item["id"] = item.get("question_id", item.get("id"))
                    self.data.append(item)

        print(f"Loaded {len(self.data)} questions from {data_file}")

        # Validate frames exist for loaded data
        self._validate_frames()

    def _validate_frames(self):
        """Check that a raw video or frame directory exists for every sample."""
        valid_data = []
        missing_count = 0

        for item in self.data:
            video_input = self._resolve_video_input(item)
            if video_input is None:
                missing_count += 1
            elif video_input.is_file():
                valid_data.append(item)
            else:
                frame_files = [
                    f for f in video_input.iterdir()
                    if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
                ]
                if frame_files:
                    valid_data.append(item)
                else:
                    missing_count += 1

        if missing_count > 0:
            print(f"[WARN] Skipped {missing_count} samples due to missing frames")

        self.data = valid_data

    def filter_finish_ids(self, finish_ids: List[str]):
        """Filter out already completed samples."""
        self.data = [item for item in self.data if item["id"] not in finish_ids]

    def filter_by_task_types(self, task_types: List[str]):
        """Only keep samples matching the specified task types."""
        self.data = [item for item in self.data if item.get("task_type") in task_types]

    def __len__(self):
        return len(self.data)

    def _resolve_video_input(self, item: dict) -> Optional[Path]:
        media_root = Path(self.frames_dir)
        references = [item.get("video_name"), item.get("video_path")]
        for reference in dict.fromkeys(ref for ref in references if ref):
            relative_path = Path(reference)
            candidates = [relative_path if relative_path.is_absolute() else media_root / relative_path]
            if not relative_path.is_absolute():
                # Accept either the dataset parent or the media directory itself
                # as frames_dir. This avoids duplicating prefixes such as
                # TravelUAV_dataset/TravelUAV_dataset/...
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

    def _load_video_images(self, video_path: Path) -> List[Image.Image]:
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

            images = []
            for index, frame in enumerate(container.decode(stream)):
                if target_indices is None or index in target_indices:
                    images.append(frame.to_image().convert("RGB"))
                if last_target is not None and index >= last_target:
                    break

        if not images:
            raise ValueError(f"No decodable frames found in {video_path}")
        if target_indices is None and len(images) > self.max_frames:
            indices = np.linspace(0, len(images) - 1, self.max_frames, dtype=int)
            images = [images[i] for i in indices]
        return images

    def _load_images(self, item: dict) -> List[Image.Image]:
        video_input = self._resolve_video_input(item)
        if video_input is None:
            raise FileNotFoundError(
                f"Video not found for {item.get('video_name') or item.get('video_path')}"
            )

        if video_input.is_file():
            images = self._load_video_images(video_input)
        else:
            frame_files = sorted([
                f for f in video_input.iterdir()
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
            ])
            if len(frame_files) > self.max_frames:
                indices = np.linspace(0, len(frame_files) - 1, self.max_frames, dtype=int)
                frame_files = [frame_files[i] for i in indices]
            images = [Image.open(frame).convert("RGB") for frame in frame_files]

        resized_images = []
        for image in images:
            if self.image_size is not None:
                image = resize_keep_aspect_ratio(image, self.image_size)
            elif self.min_pixels is not None and self.max_pixels is not None:
                image = resize_to_pixel_range(image, self.min_pixels, self.max_pixels)
            resized_images.append(image)
        return resized_images

    def _format_options(self, options: dict) -> str:
        """Format options dictionary into a string."""
        option_strs = []
        for key in sorted(options.keys()):
            option_strs.append(f"{key}. {options[key]}")
        return "\n".join(option_strs)

    def __getitem__(self, idx):
        item = self.data[idx]

        images = self._load_images(item)

        # Format options
        options_str = self._format_options(item["options"])

        # Build prompt
        prompt_text = self.prompt_template.format(
            question=item["question"],
            options_str=options_str
        )

        # Build messages in the format expected by benchvl
        # Using processed images (not raw paths) to ensure size constraints
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]

        # Build metadata
        metadata = {
            "id": item["id"],
            "question_id": item.get("question_id", item["id"]),
            "video_name": item["video_name"],
            "task_type": item.get("task_type", "unknown"),
            "split": item.get("split", ""),
            "question": item["question"],
            "options": item["options"],
            "answer": item.get("answer", ""),
            "num_frames": len(images),
        }

        return {
            "images": images,
            "messages": messages,
            "metadata": metadata,
        }
