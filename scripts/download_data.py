#!/usr/bin/env python3
"""Download registered SIS-Motion datasets from Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "configs" / "data_registry.json"
DATASETS = {
    "sis_bench": ("SIS_BENCH_REPO_ID", "SIS-Bench"),
    "sis_motion_54k": ("SIS_MOTION_54K_REPO_ID", "SIS-Motion-54K"),
    "openuav_qa": ("OPENUAV_QA_REPO_ID", "OpenUAV-QA"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        default=None,
        metavar="DATASET",
        help="Dataset ID to download (default: all)",
    )
    parser.add_argument("--revision", help="Override the registered revision")
    parser.add_argument("--dry-run", action="store_true", help="Print downloads without network access")
    args = parser.parse_args()

    with REGISTRY_PATH.open(encoding="utf-8") as handle:
        registry = json.load(handle)

    hf_config = registry["huggingface"]
    data_root = Path(
        os.environ.get("SIS_DATA_ROOT", REPO_ROOT / "data")
    ).expanduser().resolve()
    revision = args.revision or hf_config.get("revision", "main")

    downloads = []
    selected_datasets = args.datasets or list(DATASETS)
    for dataset in selected_datasets:
        if dataset not in DATASETS:
            parser.error(
                f"unknown dataset {dataset!r}; choose from {', '.join(sorted(DATASETS))}"
            )
        env_name, directory = DATASETS[dataset]
        repo_id = os.environ.get(env_name) or hf_config["repositories"].get(dataset)
        if not repo_id:
            raise SystemExit(
                f"No Hugging Face repository configured for {dataset}. "
                f"Set {env_name} or configs/data_registry.json."
            )
        downloads.append((dataset, repo_id, data_root / directory))

    for dataset, repo_id, destination in downloads:
        print(f"{dataset}: {repo_id}@{revision} -> {destination}")
        if args.dry_run:
            continue
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=destination,
        )


if __name__ == "__main__":
    main()
