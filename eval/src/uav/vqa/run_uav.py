"""
SIS-Bench inference script using the vLLM evaluation pipeline.

Example usage:
    python -m uav.vqa.run_uav \
        --model_id Qwen/Qwen2.5-VL-3B-Instruct \
        --data_file /path/to/SIS-Bench.jsonl \
        --frames_dir /path/to/SIS-Bench/video
"""
import argparse
import json
import os
import re
from dataclasses import asdict
from os.path import dirname, join

from torch.utils.data import Dataset
from tqdm import tqdm
from vllm import EngineArgs, LLM
from vllm.lora.request import LoRARequest

import uav
from uav.utils import build_model_config, ModelConfig
from uav.utils.vllm import get_sampling_params
from uav.vqa.uav_dataset import SISBenchDataset


def extract_answer(response: str) -> str | None:
    """Extract A/B/C/D answer from model response.

    Handles:
      1. Direct answers: "C", "C.", "C)"
      2. <think>...</think> wrapped: "<think>...</think>\n\nD"
      3. Empty think blocks: "<think>\n</think>\n\nD"
      4. Plain-text reasoning followed by answer
      5. Patterns like "answer is B", "选B"
    """
    text = response.strip()
    if not text:
        return None

    # 1) Strip <think>...</think> blocks (complete)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Handle incomplete <think> (truncated without </think>)
    if not cleaned and '<think>' in text:
        cleaned = re.sub(r'<think>.*', '', text, flags=re.DOTALL).strip()

    check = cleaned if cleaned else text

    # 2) If what remains starts with A-D (possibly after whitespace), that's the answer
    m = re.match(r'^([A-D])\b', check)
    if m:
        return m.group(1)

    # 3) Look for explicit answer patterns (case-insensitive)
    #    "the answer is B", "Answer: C", "答案是A", "选D"
    for pat in [
        r'(?:the\s+)?answer\s*(?:is|:)\s*([A-D])\b',
        r'答案\s*(?:是|为|：|:)\s*([A-D])\b',
        r'选\s*([A-D])\b',
    ]:
        m = re.search(pat, check, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # 4) Look for standalone A-D on its own line (common in CoT outputs)
    m = re.search(r'(?:^|\n)\s*([A-D])\s*[.\):\n]?\s*$', check, re.MULTILINE)
    if m:
        return m.group(1)

    # 5) Last standalone A-D with word boundary (fallback)
    matches = re.findall(r'\b([A-D])\b', check)
    if matches:
        return matches[-1]

    return None


def create_batch_inputs(ds: Dataset, model_config: ModelConfig, args):
    """Generate batches of inputs for vLLM inference."""
    for i in range(0, len(ds), args.batch_size):
        batch_data = [ds[j] for j in range(i, min(i + args.batch_size, len(ds)))]
        batch_inputs = []
        batch_metadata = []

        for data_dict in batch_data:
            messages = data_dict["messages"]
            images = data_dict["images"]

            result = model_config.get_prompt_from_question(messages)
            if isinstance(result, tuple) and len(result) == 2:
                # 模型配置已经返回了 (prompt_text, multimodal_inputs)
                prompt_text, multimodal_inputs = result
                inputs = {"prompt": prompt_text, **multimodal_inputs}
            else:
                # 否则，使用默认的处理方式
                prompt_text = result
                inputs = {
                    "prompt": prompt_text,
                    "multi_modal_data": {"image": images}
                }

            batch_inputs.append(inputs)
            batch_metadata.append(data_dict["metadata"])

        yield batch_inputs, batch_metadata


def main():
    repo_root = os.path.abspath(join(dirname(__file__), "../../../.."))
    data_root = os.environ.get("SIS_DATA_ROOT", join(repo_root, "data"))
    parser = argparse.ArgumentParser(description="Run SIS-Bench evaluation with vLLM")
    parser.add_argument('--model_id', type=str, required=True,
                        help="HuggingFace model ID (e.g., Qwen/Qwen2.5-VL-3B-Instruct)")
    parser.add_argument('--data_file', type=str,
                        default=join(data_root, "SIS-Bench", "SIS-Bench.jsonl"),
                        help="Path to SIS-Bench JSONL data file")
    parser.add_argument('--frames_dir', type=str,
                        default=join(data_root, "SIS-Bench", "video"),
                        help="Path to SIS-Bench videos or frame directories")
    parser.add_argument('--result_dir', type=str,
                        default=f"{dirname(uav.__file__)}/../../results/uav",
                        help="Directory to save results")
    parser.add_argument('--max_model_len', type=int, default=32768,
                        help="Maximum model context length")
    parser.add_argument('--max_tokens', type=int, default=128,
                        help="Maximum tokens to generate (short for multiple choice)")
    parser.add_argument('--max_frames', type=int, default=32,
                        help="Maximum number of frames per video")
    parser.add_argument('--image_size', type=int, default=None,
                        help="Resize images to this size (optional)")
    parser.add_argument('--min_pixels', type=int, default=256 * 28 * 28,
                        help="Minimum pixels per image for vision encoder")
    parser.add_argument('--max_pixels', type=int, default=512 * 28 * 28,
                        help="Maximum pixels per image for vision encoder")
    parser.add_argument('--tensor_parallel_size', type=int, default=None,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument('--enforce_eager', action='store_true',
                        help="Disable CUDA graph for debugging")
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite existing results")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for inference")
    parser.add_argument('--greedy', action='store_true',
                        help="Use greedy decoding (temperature=0)")
    parser.add_argument('--lora_path', type=str, default=None,
                        help="Path to LoRA adapter directory (optional)")
    parser.add_argument('--max_lora_rank', type=int, default=64,
                        help="Maximum LoRA rank (must >= training rank)")
    parser.add_argument('--task_types', type=str, nargs='+', default=None,
                        help="Task types to evaluate (default: all). E.g., --task_types action_prediction action_recall")
    parser.add_argument('--prompt_style', type=str, default='default', choices=['default', 'motion'],
                        help="Prompt style: 'default' (generic VQA) or 'motion' (matches training CoT prompt)")
    args = parser.parse_args()

    # Setup result path
    reformat_model_id = args.model_id.replace('/', '--')
    if args.lora_path:
        lora_name = os.path.basename(os.path.normpath(args.lora_path))
        reformat_model_id = f"{reformat_model_id}+lora-{lora_name}"
    if args.image_size is not None:
        reformat_model_id = reformat_model_id + f"_{args.image_size}"
    if args.task_types:
        reformat_model_id = reformat_model_id + f"_tasks-{'+'.join(sorted(args.task_types))}"
    result_save_path = join(args.result_dir, reformat_model_id, "uav.jsonl")
    os.makedirs(dirname(result_save_path), exist_ok=True)

    # Load already finished IDs for resume support
    finish_ids = []
    if not args.overwrite and os.path.exists(result_save_path):
        with open(result_save_path) as f:
            for line in f.readlines():
                data_dict = json.loads(line.strip())
                finish_ids.append(data_dict["id"])

    if len(finish_ids) > 0:
        print(f"Resume from {len(finish_ids)} finished samples")
    else:
        print(f"Start from scratch")

    # Load dataset
    ds = SISBenchDataset(
        data_file=args.data_file,
        frames_dir=args.frames_dir,
        max_frames=args.max_frames,
        image_size=args.image_size,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        prompt_style=args.prompt_style,
    )
    ds.filter_finish_ids(finish_ids)

    if args.task_types:
        ds.filter_by_task_types(args.task_types)

    if len(ds) == 0:
        print("All samples already processed!")
    else:
        print(f"Processing {len(ds)} samples...")

        # Build model config
        model_config = build_model_config(
            model_id=args.model_id,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
            max_num_frames=args.max_frames,
            video_min_pixels=args.min_pixels,
            video_max_pixels=args.max_pixels,
        )

        # Setup engine args
        engine_args = model_config.default_engine_args
        if "model" not in engine_args:
            engine_args["model"] = args.model_id

        # The current vLLM adapter represents sampled video frames as image items.
        engine_args["limit_mm_per_prompt"] = {"image": args.max_frames}

        if args.tensor_parallel_size is not None:
            engine_args["tensor_parallel_size"] = args.tensor_parallel_size
        if args.enforce_eager:
            engine_args["enforce_eager"] = args.enforce_eager

        # Enable LoRA support if lora_path is provided
        if args.lora_path:
            engine_args["enable_lora"] = True
            engine_args["max_lora_rank"] = args.max_lora_rank

        engine_args = asdict(EngineArgs(**engine_args))
        print(f"Engine arguments: {engine_args}")

        # Initialize vLLM
        vlm_model = LLM(**engine_args)

        # Get sampling params
        override_params = dict(max_tokens=args.max_tokens)
        if args.greedy:
            # Greedy decoding: temperature=0
            override_params["temperature"] = 0.0
            override_params["top_p"] = 1.0
            override_params["top_k"] = -1
        sampling_params = get_sampling_params(
            model_id=args.model_id,
            override_sampling_params=override_params,
        )
        print(f"Sampling params: {sampling_params}")

        # Setup LoRA request if applicable
        lora_request = None
        if args.lora_path:
            lora_request = LoRARequest("uav-lora", 1, args.lora_path)

        # Run inference
        with open(result_save_path, "w" if len(finish_ids) == 0 else "a") as f:
            progress_bar = tqdm(total=len(ds))
            for batch_inputs, batch_metadata in create_batch_inputs(ds, model_config, args):
                outputs = vlm_model.generate(
                    batch_inputs,
                    sampling_params=sampling_params,
                    lora_request=lora_request,
                    use_tqdm=False
                )

                for idx, output in enumerate(outputs):
                    generated_text = output.outputs[0].text
                    dump_dict = {
                        "id": batch_metadata[idx]["id"],
                        "response": generated_text,
                        "metadata": batch_metadata[idx],
                    }
                    f.write(json.dumps(dump_dict, ensure_ascii=False) + "\n")
                    f.flush()
                    progress_bar.update(1)

            progress_bar.close()

    # Convert to JSON format as well
    with open(result_save_path) as f:
        finished_data = [json.loads(line.strip()) for line in f.readlines()]

    with open(result_save_path.replace(".jsonl", ".json"), "w") as f:
        json.dump(finished_data, f, indent=4, ensure_ascii=False)

    # Calculate accuracy (overall, per category, and per split)
    correct = 0
    total = 0
    category_stats = {}  # {category: {"correct": int, "total": int}}
    split_stats = {}     # {split: {"correct": int, "total": int}}

    for item in finished_data:
        expected = item["metadata"].get("answer", "")
        response = item["response"].strip()
        task_type = item["metadata"].get("task_type", "unknown")
        split = item["metadata"].get("split", "")

        # Initialize category stats
        if task_type not in category_stats:
            category_stats[task_type] = {"correct": 0, "total": 0}

        # Initialize split stats
        if split and split not in split_stats:
            split_stats[split] = {"correct": 0, "total": 0}

        extracted = extract_answer(response)

        # Check if correct
        is_correct = extracted is not None and extracted.upper() == expected.upper()
        if is_correct:
            correct += 1

        total += 1
        category_stats[task_type]["total"] += 1
        if is_correct:
            category_stats[task_type]["correct"] += 1

        if split:
            split_stats[split]["total"] += 1
            if is_correct:
                split_stats[split]["correct"] += 1

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    # Per-category accuracy
    print("\nPer-category Accuracy:")
    print("-" * 60)
    print(f"{'Category':<30} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 60)

    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        acc = 100 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"{category:<30} {stats['correct']:>8} {stats['total']:>8} {acc:>9.2f}%")

    print("-" * 60)

    # Per-split accuracy (only if split field exists)
    if split_stats:
        print("\nPer-split Accuracy:")
        print("-" * 60)
        print(f"{'Split':<30} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
        print("-" * 60)
        for sp in sorted(split_stats.keys()):
            stats = split_stats[sp]
            acc = 100 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"{sp:<30} {stats['correct']:>8} {stats['total']:>8} {acc:>9.2f}%")
        print("-" * 60)

    # Overall accuracy
    overall_acc = 100 * correct / total if total > 0 else 0
    print(f"{'Overall':<30} {correct:>8} {total:>8} {overall_acc:>9.2f}%")
    print("=" * 60)

    # Save accuracy stats to JSON
    accuracy_stats = {
        "overall": {"correct": correct, "total": total, "accuracy": overall_acc},
        "per_category": {
            cat: {
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": 100 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            }
            for cat, stats in category_stats.items()
        }
    }
    if split_stats:
        accuracy_stats["per_split"] = {
            sp: {
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": 100 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            }
            for sp, stats in split_stats.items()
        }
    stats_path = result_save_path.replace(".jsonl", "_accuracy.json")
    with open(stats_path, "w") as f:
        json.dump(accuracy_stats, f, indent=4, ensure_ascii=False)

    print(f"\nResults saved to: {result_save_path}")
    print(f"Accuracy stats saved to: {stats_path}")


if __name__ == '__main__':
    main()
