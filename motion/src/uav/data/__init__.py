import os
import re
from pathlib import Path


DATA_ROOT = Path(
    os.environ.get("SIS_DATA_ROOT", Path(__file__).resolve().parents[4] / "data")
).expanduser().resolve()

SIS_MOTION_54K_ROOT = DATA_ROOT / "SIS-Motion-54K"

data_dict = {
    "sis_motion_54k": {
        "annotation_path": str(SIS_MOTION_54K_ROOT / "SIS-Motion-54K.jsonl"),
        "data_path": str(SIS_MOTION_54K_ROOT),
    },
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name not in data_dict:
            available = ", ".join(sorted(data_dict))
            raise ValueError(f"Unknown dataset {dataset_name!r}; available: {available}")
        config = data_dict[dataset_name].copy()
        config["sampling_rate"] = sampling_rate
        config_list.append(config)
    return config_list
