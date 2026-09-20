"""Image-pair manifests and pixel-center-consistent preprocessing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


def resize_transform(old_hw: tuple[int, int], new_hw: tuple[int, int]) -> torch.Tensor:
    """Map original pixel centers to resized pixel centers (align_corners=False)."""
    sy, sx = new_hw[0] / old_hw[0], new_hw[1] / old_hw[1]
    return torch.tensor([[sx, 0., (sx - 1.) / 2.],
                         [0., sy, (sy - 1.) / 2.], [0., 0., 1.]], dtype=torch.float32)


def transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    result = homogeneous @ transform.transpose(-1, -2)
    return result[..., :2] / result[..., 2:3]


def load_image(path: str | Path, resize: int | None = 832) -> dict[str, torch.Tensor]:
    with Image.open(path) as im:
        array = np.asarray(im.convert("L"), dtype=np.float32).copy() / 255.
    original_hw = tuple(array.shape)
    tensor = torch.from_numpy(array)[None]
    if resize is not None and int(resize) > 0:
        scale = int(resize) / max(original_hw)
        new_hw = tuple(max(1, round(side * scale)) for side in original_hw)
        if new_hw != original_hw:
            tensor = F.interpolate(tensor[None], size=new_hw, mode="bilinear",
                                   align_corners=False, antialias=True)[0]
    else:
        new_hw = original_hw
    return {"image": tensor, "original_hw": torch.tensor(original_hw),
            "resize_transform": resize_transform(original_hw, new_hw)}


def _array(value: Any, root: Path, default_key: str | None = None) -> torch.Tensor:
    if isinstance(value, (list, tuple, np.ndarray)):
        return torch.as_tensor(value, dtype=torch.float32)
    if isinstance(value, dict):
        path, key = root / value["path"], value.get("key", default_key)
    elif isinstance(value, str):
        name, _, key = value.partition("#")
        path, key = root / name, key or default_key
    else:
        raise TypeError("Geometry must be an inline array or path (optionally path#key).")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as content:
            if key in content:
                array = content[key]
            elif len(content.files) == 1:
                array = content[content.files[0]]
            else:
                raise ValueError(f"Specify the array key for {path}; available: {content.files}")
    elif suffix in (".h5", ".hdf5"):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("Install h5py to read MegaDepth .h5 depth maps.") from exc
        with h5py.File(path, "r") as content:
            array = np.asarray(content[key or "depth"])
    else:
        raise ValueError(f"Unsupported geometry file: {path}")
    return torch.as_tensor(np.asarray(array).copy(), dtype=torch.float32)


class PairDataset(Dataset):
    """JSONL pairs with either homography or metric depth/camera supervision."""

    def __init__(self, manifest: str | Path, resize: int | None = 832,
                 require_geometry: bool = True, load_depth: bool = True):
        self.manifest = Path(manifest).resolve()
        self.root, self.resize = self.manifest.parent, resize
        with self.manifest.open(encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        if not self.rows:
            raise ValueError(f"Empty pair manifest: {self.manifest}")
        self.require_geometry, self.load_depth = require_geometry, load_depth

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        sample: dict[str, Any] = {"pair_id": str(row.get("pair_id", index))}
        for side in (0, 1):
            info = load_image(self.root / row[f"image{side}"], self.resize)
            sample[f"image{side}"] = info["image"]
            sample[f"original_hw{side}"] = info["original_hw"]
            sample[f"resize_transform{side}"] = info["resize_transform"]
        a0, a1 = sample["resize_transform0"], sample["resize_transform1"]
        if "H_0to1" in row:
            h = _array(row["H_0to1"], self.root, "H_0to1")
            if h.shape != (3, 3):
                raise ValueError("H_0to1 must have shape (3, 3).")
            sample["H_0to1"] = a1 @ h @ torch.linalg.inv(a0)
        elif all(key in row for key in ("K0", "K1", "T_0to1")) and (
                not self.load_depth or all(key in row for key in ("depth0", "depth1"))):
            for side, a in ((0, a0), (1, a1)):
                if self.load_depth:
                    depth = _array(row[f"depth{side}"], self.root, "depth").squeeze()
                    expected = tuple(sample[f"original_hw{side}"].tolist())
                    if tuple(depth.shape) != expected:
                        raise ValueError(f"depth{side} shape {tuple(depth.shape)} != image shape {expected}")
                    depth = torch.where(torch.isfinite(depth) & (depth > 0), depth, 0.)
                    sample[f"depth{side}"] = F.interpolate(
                        depth[None, None], size=sample[f"image{side}"].shape[-2:],
                        mode="nearest-exact")[0, 0]
                k = _array(row[f"K{side}"], self.root, f"K{side}")
                if k.shape != (3, 3):
                    raise ValueError(f"K{side} must have shape (3, 3).")
                sample[f"K{side}"] = a @ k
            transform = _array(row["T_0to1"], self.root, "T_0to1")
            if transform.shape != (4, 4):
                raise ValueError("T_0to1 must have shape (4, 4), mapping camera 0 to camera 1.")
            sample["T_0to1"] = transform
        elif self.require_geometry:
            raise ValueError(f"Pair {sample['pair_id']} needs H_0to1 or depth0/depth1/K0/K1/T_0to1.")
        return sample


def collate_pairs(samples: list[dict[str, Any]], pad_multiple: int = 32) -> dict[str, Any]:
    """Pad each side independently, retaining valid-pixel masks and resize metadata."""
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    if pad_multiple < 1:
        raise ValueError("pad_multiple must be positive.")
    schemas = [{key for key in item if key in ("H_0to1", "depth0", "depth1", "K0", "K1", "T_0to1")}
               for item in samples]
    if any(schema != schemas[0] for schema in schemas):
        raise ValueError("A batch must use one geometry type; separate homography and depth manifests.")
    batch: dict[str, Any] = {"pair_id": [sample.get("pair_id", str(i)) for i, sample in enumerate(samples)]}
    for side in (0, 1):
        images = [sample[f"image{side}"] for sample in samples]
        height = (max(im.shape[-2] for im in images) + pad_multiple - 1) // pad_multiple * pad_multiple
        width = (max(im.shape[-1] for im in images) + pad_multiple - 1) // pad_multiple * pad_multiple
        masks, padded, depths = [], [], []
        for sample, im in zip(samples, images):
            h, w = im.shape[-2:]
            pad = (0, width - w, 0, height - h)
            padded.append(F.pad(im, pad))
            masks.append(F.pad(torch.ones(h, w, dtype=torch.bool), pad))
            if f"depth{side}" in sample:
                depths.append(F.pad(sample[f"depth{side}"], pad))
        batch[f"image{side}"] = torch.stack(padded)
        batch[f"mask{side}"] = torch.stack(masks)
        batch[f"hw{side}"] = torch.tensor([im.shape[-2:] for im in images])
        if depths:
            batch[f"depth{side}"] = torch.stack(depths)
    for key in samples[0]:
        if key not in batch and key not in ("image0", "image1", "depth0", "depth1"):
            if all(key in sample and isinstance(sample[key], torch.Tensor) for sample in samples):
                batch[key] = torch.stack([sample[key] for sample in samples])
    return batch


class SyntheticPairs(Dataset):
    """Deterministic textured translations for plumbing checks, never an accuracy benchmark."""

    def __init__(self, length: int = 8, size: int = 64, seed: int = 42):
        if size < 32:
            raise ValueError("Synthetic image size must be >= 32.")
        self.length, self.size, self.seed = length, size, seed

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(self.seed + index)
        size = self.size
        noise = torch.rand((1, 1, size, size), generator=generator)
        smooth = F.avg_pool2d(noise, kernel_size=5, stride=1, padding=2)
        image0 = (.35 * noise + .65 * smooth)[0]
        dx, dy = int(torch.randint(-5, 6, (), generator=generator)), int(torch.randint(-5, 6, (), generator=generator))
        h = torch.eye(3)
        h[0, 2], h[1, 2] = dx, dy
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        grid = torch.stack((2. * (x - dx + .5) / size - 1.,
                            2. * (y - dy + .5) / size - 1.), dim=-1).float()
        image1 = F.grid_sample(image0[None], grid[None], align_corners=False)[0]
        return {"image0": image0, "image1": image1, "H_0to1": h,
                "original_hw0": torch.tensor((size, size)), "original_hw1": torch.tensor((size, size)),
                "resize_transform0": torch.eye(3), "resize_transform1": torch.eye(3),
                "pair_id": f"synthetic-{index}"}


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}
