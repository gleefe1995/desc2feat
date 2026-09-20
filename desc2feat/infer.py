"""Match an image pair and save coordinates in the original image pixel frame."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .config import load_config
from .data import collate_pairs, load_image, to_device, transform_points
from .model import Desc2Feat


def load_model(checkpoint_path=None, config_path=None, device="cpu", allow_random=False, deploy=False):
    if checkpoint_path is None and not allow_random:
        raise ValueError("A trained --checkpoint is required; use --allow-random only to test the pipeline.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False) if checkpoint_path else None
    cfg = load_config(config_path) if config_path else (checkpoint.get("config") if checkpoint else None)
    cfg = cfg or load_config(None)
    model = Desc2Feat(cfg["model"])
    if checkpoint:
        model.load_state_dict(checkpoint.get("model", checkpoint.get("state_dict", checkpoint)), strict=True)
    model = model.to(device).eval()
    if deploy:
        model.switch_to_deploy()
    return model, cfg


def image_pair(image0, image1, resize=832, pad_multiple=32):
    sample = {"pair_id": f"{Path(image0).name}:{Path(image1).name}"}
    for side, path in enumerate((image0, image1)):
        info = load_image(path, resize)
        sample[f"image{side}"] = info["image"]
        sample[f"original_hw{side}"] = info["original_hw"]
        sample[f"resize_transform{side}"] = info["resize_transform"]
    return collate_pairs([sample], pad_multiple)


def original_matches(result, batch, batch_index=0):
    batch_ids = result.get("b_ids", torch.zeros(len(result["mconf"]), device=result["mconf"].device, dtype=torch.long))
    chosen = batch_ids == batch_index
    point0 = transform_points(result["mkpts0_f"][chosen].float(), torch.linalg.inv(batch["resize_transform0"][batch_index].float()))
    point1 = transform_points(result["mkpts1_f"][chosen].float(), torch.linalg.inv(batch["resize_transform1"][batch_index].float()))
    return {"keypoints0": point0.detach().cpu().numpy(), "keypoints1": point1.detach().cpu().numpy(),
            "confidence": result["mconf"][chosen].float().detach().cpu().numpy()}


def draw_matches(path0, path1, matches, output, max_matches=150):
    with Image.open(path0) as im:
        im0 = im.convert("RGB")
    with Image.open(path1) as im:
        im1 = im.convert("RGB")
    canvas = Image.new("RGB", (im0.width + im1.width, max(im0.height, im1.height)), (20, 20, 20))
    canvas.paste(im0, (0, 0))
    canvas.paste(im1, (im0.width, 0))
    draw = ImageDraw.Draw(canvas)
    order = np.argsort(matches["confidence"])[::-1][:max_matches]
    for index in order:
        x0, y0 = matches["keypoints0"][index]
        x1, y1 = matches["keypoints1"][index]
        score = float(matches["confidence"][index])
        color = (int(255 * (1. - score)), int(255 * score), 80)
        draw.line((float(x0), float(y0), float(x1) + im0.width, float(y1)), fill=color, width=1)
        for x, y in ((float(x0), float(y0)), (float(x1) + im0.width, float(y1))):
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image0")
    parser.add_argument("image1")
    parser.add_argument("--checkpoint")
    parser.add_argument("--config")
    parser.add_argument("--output", default="matches.npz")
    parser.add_argument("--visualization")
    parser.add_argument("--resize", type=int, help="Long side; 0 preserves original dimensions")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-random", action="store_true")
    parser.add_argument("--deploy", action="store_true", help="Fuse RepVGG branches for inference")
    args = parser.parse_args(argv)
    if not args.checkpoint and not args.allow_random:
        parser.error("--checkpoint is required unless --allow-random is explicitly set")
    device = torch.device(args.device)
    model, cfg = load_model(args.checkpoint, args.config, device, args.allow_random, args.deploy)
    resize = args.resize if args.resize is not None else cfg.get("data", {}).get("resize", 832)
    batch = to_device(image_pair(args.image0, args.image1, resize, cfg.get("data", {}).get("pad_multiple", 32)), device)
    with torch.inference_mode():
        result = model(batch)
    matches = original_matches(result, batch)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        np.savez_compressed(handle, **matches, original_hw0=batch["original_hw0"][0].cpu().numpy(),
                            original_hw1=batch["original_hw1"][0].cpu().numpy())
    if args.visualization:
        draw_matches(args.image0, args.image1, matches, args.visualization)
    print(json.dumps({"matches": len(matches["confidence"]), "output": str(destination),
                      "coordinate_frame": "original image pixel centers", "random_weights": not bool(args.checkpoint)}))


if __name__ == "__main__":
    main()
