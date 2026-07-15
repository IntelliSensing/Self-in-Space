"""
Baseline Qwen-VL and LoRA evaluation on SIS-Bench.

No motion encoder — pure vision-language model for fair comparison.

Usage:
    python -m src.eval.eval \
        --model_path output/qwen3vl-4b-lora \
        --data_file data/SIS-Bench/SIS-Bench.jsonl \
        --frames_dir data/SIS-Bench/video \
        --result_dir results/qwen3vl-4b-lora
"""
import argparse
import contextlib
import copy
import json
import os
import re
import sys
import torch.multiprocessing as mp
from os.path import dirname, join
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)
from transformers.video_utils import VideoMetadata

# Add repo root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.eval.dataset_utils import SISBenchDataset


MODEL_CLASS_MAP = {
    "qwen2vl": Qwen2VLForConditionalGeneration,
    "qwen2.5vl": Qwen2_5_VLForConditionalGeneration,
    "qwen3vl": Qwen3VLForConditionalGeneration,
    "qwen3vl_moe": Qwen3VLMoeForConditionalGeneration,
}


def build_presampled_video_metadata(frames, fps: float):
    return [
        VideoMetadata(
            total_num_frames=len(frames),
            fps=fps,
            frames_indices=list(range(len(frames))),
        )
    ]


def extract_answer(response: str):
    """Extract A/B/C/D answer from model response."""
    text = response.strip()
    if not text:
        return None

    # Strip <think>...</think> blocks
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if not cleaned and '<think>' in text:
        cleaned = re.sub(r'<think>.*', '', text, flags=re.DOTALL).strip()

    check = cleaned if cleaned else text

    # Direct answer at start
    m = re.match(r'^([A-D])\b', check)
    if m:
        return m.group(1)

    # Explicit patterns
    for pat in [
        r'(?:the\s+)?answer\s*(?:is|:)\s*([A-D])\b',
        r'答案\s*(?:是|为|：|:)\s*([A-D])\b',
        r'选\s*([A-D])\b',
    ]:
        m = re.search(pat, check, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # Standalone A-D on its own line
    m = re.search(r'(?:^|\n)\s*([A-D])\s*[.\):\n]?\s*$', check, re.MULTILINE)
    if m:
        return m.group(1)

    # Last A-D with word boundary
    matches = re.findall(r'\b([A-D])\b', check)
    if matches:
        return matches[-1]

    return None


def resolve_model_family(model_name_or_path: str) -> str:
    config_model_type = None
    try:
        config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)
        config_model_type = getattr(config, "model_type", None)
    except Exception:
        config_model_type = None

    if config_model_type == "qwen3_vl":
        return "qwen3vl"
    if config_model_type == "qwen3_vl_moe":
        return "qwen3vl_moe"
    if config_model_type == "qwen2_5_vl":
        return "qwen2.5vl"
    if config_model_type == "qwen2_vl":
        return "qwen2vl"

    model_ref = model_name_or_path.rstrip("/")
    basename = Path(model_ref).name.lower()
    full_name = model_ref.lower()
    searchable = [basename, full_name]

    if any("qwen3-vl" in candidate for candidate in searchable):
        if re.search(r"(?:^|-)a\d+b(?:-|$)", basename):
            return "qwen3vl_moe"
        return "qwen3vl"
    if any("qwen2.5-vl" in candidate for candidate in searchable):
        return "qwen2.5vl"
    if any("qwen2-vl" in candidate for candidate in searchable):
        return "qwen2vl"

    raise ValueError(
        f"Unsupported model path '{model_name_or_path}'. "
        "Expected a Qwen2-VL, Qwen2.5-VL, or Qwen3-VL checkpoint."
    )


def get_model_class(model_family: str):
    try:
        return MODEL_CLASS_MAP[model_family]
    except KeyError as exc:
        raise ValueError(f"Unsupported model family: {model_family}") from exc


def get_model_load_kwargs(device: str):
    is_cuda_device = device == "auto" or device.startswith("cuda")
    return {
        "attn_implementation": "flash_attention_2" if is_cuda_device else "sdpa",
        "torch_dtype": torch.bfloat16 if is_cuda_device else torch.float32,
    }


def load_model(model_path, device="cuda"):
    """Load Qwen-VL model with optional LoRA adapter."""
    print(f"Loading model from {model_path}...")

    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    load_path = model_path

    if os.path.exists(adapter_config_path):
        # LoRA checkpoint: load base model then apply adapter
        with open(adapter_config_path) as f:
            adapter_cfg = json.load(f)
        base_model_path = adapter_cfg["base_model_name_or_path"]
        print(f"LoRA checkpoint detected. Base model: {base_model_path}")
        load_path = base_model_path
    else:
        base_model_path = model_path

    model_family = resolve_model_family(load_path)
    model_class = get_model_class(model_family)
    load_kwargs = get_model_load_kwargs(device)
    print(f"Resolved model family: {model_family}")

    model = model_class.from_pretrained(
        base_model_path,
        **load_kwargs,
    )

    if os.path.exists(adapter_config_path):
        # Load LoRA adapter
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        print("LoRA adapter merged.")
    processor_path = base_model_path
    if os.path.isdir(model_path):
        processor_markers = [
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer_config.json",
        ]
        if any(os.path.exists(os.path.join(model_path, marker)) for marker in processor_markers):
            processor_path = model_path

    processor = AutoProcessor.from_pretrained(processor_path)

    if device != "auto":
        model = model.to(device)

    # Align the eval processor with Qwen2.5-VL visual-token granularity.
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        vp.min_pixels = 128 * 28 * 28    # 100,352
        vp.max_pixels = 512 * 28 * 28    # 401,408
        vp.fps = 2.0
        if hasattr(vp, "min_frames"):
            vp.min_frames = 8
        if hasattr(vp, "max_frames"):
            vp.max_frames = 32
        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = 128 * 28 * 28
            vp.size["longest_edge"] = 512 * 28 * 28
        print(f"Video processor: min_pixels={vp.min_pixels}, max_pixels={vp.max_pixels}, "
              f"fps={vp.fps}, min_frames={getattr(vp, 'min_frames', 'N/A')}, "
              f"max_frames={getattr(vp, 'max_frames', 'N/A')}")

    model.eval()
    print(f"Model loaded. Family: {model_family}, device: {device}, dtype: {model.dtype}")
    return model, processor


@torch.no_grad()
def run_inference(model, processor, dataset, args):
    """Run inference on the dataset sample by sample."""
    results = []

    rank = getattr(args, '_rank', None)
    show_tqdm = rank is None or rank == 0
    desc = f"GPU {rank}" if rank is not None else "Evaluating"

    for i in tqdm(range(len(dataset)), desc=desc, disable=not show_tqdm):
        sample = dataset[i]
        messages = sample["messages"]
        metadata = sample["metadata"]

        try:
            # Apply chat template to get model inputs
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Process inputs through the processor
            video_frames = sample["messages"][0]["content"][0]["video"]
            inputs = processor(
                text=[text],
                videos=[video_frames],
                do_sample_frames=False,
                video_metadata=build_presampled_video_metadata(
                    video_frames,
                    float(getattr(processor.video_processor, "fps", 2.0) or 2.0),
                ),
                return_tensors="pt",
                padding=True,
            )

            # Move to device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            # Generate with autocast to match training bf16 behavior
            autocast_context = (
                torch.amp.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else contextlib.nullcontext()
            )
            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": not args.greedy,
            }
            if not args.greedy:
                generation_kwargs.update(temperature=0.7, top_p=0.9)
            with autocast_context:
                output_ids = model.generate(
                    **inputs,
                    **generation_kwargs,
                )

            # Decode only the generated tokens
            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, input_len:]
            response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        except torch.OutOfMemoryError:
            print(f"\n[OOM] Skipping sample {metadata['id']}, video too large")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            response = ""

        result = {
            "id": metadata["id"],
            "response": response,
            "metadata": metadata,
        }
        results.append(result)

        # Stream write
        if args.result_path:
            with open(args.result_path, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()

        if i % 50 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def compute_accuracy(results):
    """Compute overall, per-category, and per-split accuracy."""
    correct = 0
    total = 0
    category_stats = {}
    split_stats = {}

    for item in results:
        expected = item["metadata"].get("answer", "")
        response = item["response"]
        task_type = item["metadata"].get("task_type", "unknown")
        split = item["metadata"].get("split", "")

        if task_type not in category_stats:
            category_stats[task_type] = {"correct": 0, "total": 0}
        if split and split not in split_stats:
            split_stats[split] = {"correct": 0, "total": 0}

        extracted = extract_answer(response)
        is_correct = extracted is not None and extracted.upper() == expected.upper()

        total += 1
        category_stats[task_type]["total"] += 1
        if is_correct:
            correct += 1
            category_stats[task_type]["correct"] += 1
        if split:
            split_stats[split]["total"] += 1
            if is_correct:
                split_stats[split]["correct"] += 1

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results (Baseline: Qwen-VL + optional LoRA)")
    print("=" * 60)

    print(f"\n{'Category':<35} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print("-" * 60)
    for cat in sorted(category_stats.keys()):
        s = category_stats[cat]
        acc = 100 * s["correct"] / s["total"] if s["total"] > 0 else 0
        print(f"{cat:<35} {s['correct']:>8} {s['total']:>8} {acc:>7.2f}%")

    if split_stats:
        print(f"\n{'Split':<35} {'Correct':>8} {'Total':>8} {'Acc':>8}")
        print("-" * 60)
        for sp in sorted(split_stats.keys()):
            s = split_stats[sp]
            acc = 100 * s["correct"] / s["total"] if s["total"] > 0 else 0
            print(f"{sp:<35} {s['correct']:>8} {s['total']:>8} {acc:>7.2f}%")

    overall_acc = 100 * correct / total if total > 0 else 0
    print("-" * 60)
    print(f"{'Overall':<35} {correct:>8} {total:>8} {overall_acc:>7.2f}%")
    print("=" * 60)

    return {
        "overall": {"correct": correct, "total": total, "accuracy": overall_acc},
        "per_category": {
            cat: {**s, "accuracy": 100 * s["correct"] / s["total"] if s["total"] > 0 else 0}
            for cat, s in category_stats.items()
        },
        "per_split": {
            sp: {**s, "accuracy": 100 * s["correct"] / s["total"] if s["total"] > 0 else 0}
            for sp, s in split_stats.items()
        } if split_stats else {},
    }


def eval_worker(rank, world_size, args):
    """Worker function for multi-GPU parallel evaluation."""
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    # Each worker creates its own dataset
    ds = SISBenchDataset(
        data_file=args.data_file,
        frames_dir=args.frames_dir,
        max_frames=args.max_frames,
        min_frames=args.min_frames,
        prompt_style=args.prompt_style,
    )
    ds.filter_finish_ids(args._finish_ids)
    if args.task_types:
        ds.filter_by_task_types(args.task_types)

    # Shard: each rank processes every world_size-th sample
    ds.data = ds.data[rank::world_size]

    if len(ds) == 0:
        print(f"[GPU {rank}] No samples to process")
        return

    print(f"[GPU {rank}] Processing {len(ds)} samples on {device}")

    # Load model on this specific GPU
    model, processor = load_model(args.model_path, device=device)

    # Write to rank-specific result file
    worker_args = copy.copy(args)
    worker_args.result_path = args.result_path.replace('.jsonl', f'_rank{rank}.jsonl')
    worker_args._rank = rank
    open(worker_args.result_path, 'w').close()

    run_inference(model, processor, ds, worker_args)
    print(f"[GPU {rank}] Finished.")


def merge_rank_results(result_path, world_size):
    """Merge per-rank result files into the main result file."""
    all_new = []
    for rank in range(world_size):
        rank_path = result_path.replace('.jsonl', f'_rank{rank}.jsonl')
        if os.path.exists(rank_path):
            with open(rank_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_new.append(line)
            os.remove(rank_path)

    if all_new:
        with open(result_path, 'a') as f:
            for line in all_new:
                f.write(line + '\n')
    print(f"Merged {len(all_new)} new results from {world_size} GPUs")


def main():
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Evaluate a Qwen-VL model on SIS-Bench")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to trained LoRA checkpoint or HuggingFace model ID")
    parser.add_argument('--data_file', type=str,
                        default=str(repo_root / "data/SIS-Bench/SIS-Bench.jsonl"),
                        help="Path to SIS-Bench JSONL data file")
    parser.add_argument('--frames_dir', type=str,
                        default=str(repo_root / "data/SIS-Bench/video"),
                        help="Path to SIS-Bench videos or frame directories")
    parser.add_argument('--result_dir', type=str,
                        default="results/qwen3vl-4b-lora",
                        help="Directory to save results")
    parser.add_argument('--max_frames', type=int, default=32,
                        help="Maximum frames per video")
    parser.add_argument('--min_frames', type=int, default=4,
                        help="Minimum frames per video")
    parser.add_argument('--max_new_tokens', type=int, default=128,
                        help="Maximum tokens to generate")
    parser.add_argument('--greedy', action='store_true',
                        help="Use greedy decoding")
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite existing results")
    parser.add_argument('--task_types', type=str, nargs='+', default=None,
                        help="Task types to evaluate (default: all)")
    parser.add_argument('--prompt_style', type=str, default='default',
                        choices=['default', 'strict', 'motion'],
                        help="Prompt template style")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device for inference ('auto' for multi-GPU)")
    args = parser.parse_args()

    # Setup result path
    model_name = os.path.basename(os.path.normpath(args.model_path))
    result_save_path = join(args.result_dir, model_name, "uav.jsonl")
    os.makedirs(dirname(result_save_path), exist_ok=True)
    args.result_path = result_save_path

    # Resume support
    finish_ids = []
    if not args.overwrite and os.path.exists(result_save_path):
        with open(result_save_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    finish_ids.append(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Resume from {len(finish_ids)} finished samples")

    if args.overwrite or len(finish_ids) == 0:
        # Clear file for fresh start
        open(result_save_path, "w").close()

    # Load dataset
    ds = SISBenchDataset(
        data_file=args.data_file,
        frames_dir=args.frames_dir,
        max_frames=args.max_frames,
        min_frames=args.min_frames,
        prompt_style=args.prompt_style,
    )
    ds.filter_finish_ids(finish_ids)
    if args.task_types:
        ds.filter_by_task_types(args.task_types)

    if len(ds) == 0:
        print("All samples already processed!")
    else:
        num_gpus = torch.cuda.device_count()
        if args.device == "auto" and num_gpus > 1:
            # Multi-GPU: load one model per GPU, process shards in parallel
            print(f"Processing {len(ds)} samples across {num_gpus} GPUs (~{len(ds) // num_gpus} per GPU)...")
            args._finish_ids = finish_ids
            mp.spawn(eval_worker, args=(num_gpus, args), nprocs=num_gpus, join=True)
            merge_rank_results(result_save_path, num_gpus)
        else:
            # Single GPU
            if args.device == "auto":
                args.device = "cuda:0" if num_gpus > 0 else "cpu"
            print(f"Processing {len(ds)} samples...")
            model, processor = load_model(args.model_path, device=args.device)
            run_inference(model, processor, ds, args)

    # Load all results and compute accuracy
    with open(result_save_path) as f:
        all_results = [json.loads(line.strip()) for line in f if line.strip()]

    if all_results:
        # Save as JSON
        json_path = result_save_path.replace(".jsonl", ".json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)

        # Compute and save accuracy
        accuracy_stats = compute_accuracy(all_results)
        stats_path = result_save_path.replace(".jsonl", "_accuracy.json")
        with open(stats_path, "w") as f:
            json.dump(accuracy_stats, f, indent=4, ensure_ascii=False)

        print(f"\nResults: {result_save_path}")
        print(f"Accuracy: {stats_path}")


if __name__ == "__main__":
    main()
