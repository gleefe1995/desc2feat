"""Explicit, serializable experiment configuration. JSON needs no extra dependency."""
import copy
import json
from pathlib import Path

DEFAULT_CONFIG = {
    "model": {
        "num_keypoints": 1000, "descriptor_dim": 256, "fine_dim": 64,
        "num_heads": 8, "num_layers": 4, "aggregation": 4,
        "backbone_blocks": [2, 4, 14], "nms_radius": 2,
        "detector_radius": 2, "detector_temperature": 0.1,
        "score_threshold": 0.0, "border": 4,
        "coarse_temperature": 0.1, "match_threshold": 0.01,
        "coarse_chunk_size": 0, "fine_radius": 4,
        "fine_temperature": 0.1, "local_temperature": 0.1,
        "train_fine_samples": 512, "train_gt_fraction": 0.5,
    },
    "loss": {},
    "data": {"train_manifest": None, "val_manifest": None, "resize": 832,
             "pad_multiple": 32, "workers": 4},
    "train": {"epochs": 30, "batch_size": 2, "lr": 1e-4,
              "weight_decay": 1e-4, "accumulation_steps": 1, "amp": True,
              "grad_clip": 1.0, "seed": 42, "log_every": 20, "save_every": 1},
}


def load_config(path=None):
    from .losses import DEFAULT_LOSS_CONFIG
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["loss"].update(DEFAULT_LOSS_CONFIG)
    if path is None:
        return config
    path = Path(path)
    if path.suffix.lower() == ".json":
        update = json.loads(path.read_text())
    else:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("YAML config requires PyYAML; JSON configs work without it") from exc
        update = yaml.safe_load(path.read_text())
    if not isinstance(update, dict):
        raise ValueError("Config must be a mapping")
    for section, values in update.items():
        if section not in config or not isinstance(values, dict):
            raise ValueError(f"Unknown or invalid config section: {section}")
        config[section].update(values)
    return config
