#!/usr/bin/env python3
"""Resolve and validate paths from the shared SIS-Motion data registry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "configs" / "data_registry.json"


def load_registry() -> dict[str, Any]:
    with REGISTRY_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def data_root() -> Path:
    return Path(os.environ.get("SIS_DATA_ROOT", REPO_ROOT / "data")).expanduser().resolve()


def lookup(registry: dict[str, Any], key: str) -> Any:
    value: Any = registry
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Unknown registry key: {key}")
        value = value[part]
    return value


def resolve(value: Any) -> Any:
    if isinstance(value, str):
        return str((data_root() / value).resolve())
    if isinstance(value, dict):
        return {key: resolve(item) for key, item in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    get_parser = subparsers.add_parser("get", help="Print one resolved registry value")
    get_parser.add_argument("key", help="Dotted key, for example bench.sis_bench.annotation")

    subparsers.add_parser("show", help="Print the resolved registry")
    subparsers.add_parser("validate", help="Report missing registered files and directories")
    args = parser.parse_args()

    registry = load_registry()
    if args.command == "get":
        value = resolve(lookup(registry, args.key))
        if isinstance(value, (dict, list)):
            print(json.dumps(value, indent=2, ensure_ascii=False))
        else:
            print(value)
        return

    resolved = resolve({key: registry[key] for key in ("bench", "train")})
    if args.command == "show":
        print(json.dumps(resolved, indent=2, ensure_ascii=False))
        return

    missing = []
    for group in ("bench", "train"):
        for name, entry in resolved[group].items():
            for kind, path in entry.items():
                if not Path(path).exists():
                    missing.append(f"{group}.{name}.{kind}: {path}")
    if missing:
        print("Missing registered data (expected before the Hugging Face download):")
        for item in missing:
            print(f"  - {item}")
        raise SystemExit(1)
    print("All registered benchmark and training data paths exist.")


if __name__ == "__main__":
    main()
