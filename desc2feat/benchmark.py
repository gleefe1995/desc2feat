"""Measure synchronized model latency on one fixed pair, excluding I/O and resize."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from .data import SyntheticPairs, collate_pairs, to_device
from .infer import image_pair, load_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image0")
    parser.add_argument("--image1")
    parser.add_argument("--checkpoint")
    parser.add_argument("--config")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resize", type=int, default=832)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--allow-random", action="store_true")
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--amp", action="store_true", help="CUDA FP16 autocast")
    parser.add_argument("--output")
    parser.add_argument("--profile", action="store_true", help="Run a separate profiler pass for stage CPU/device time")
    args = parser.parse_args(argv)
    if bool(args.image0) != bool(args.image1):
        parser.error("Provide both --image0 and --image1")
    if not args.checkpoint and not args.allow_random:
        parser.error("--checkpoint is required unless --allow-random is explicitly set")
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup nonnegative")
    if args.resize < 32 and not args.image0:
        parser.error("Synthetic benchmark resize must be >= 32")
    device = torch.device(args.device)
    if args.amp and device.type != "cuda":
        parser.error("--amp requires a CUDA device")
    model, cfg = load_model(args.checkpoint, args.config, device, args.allow_random, args.deploy)
    pad = cfg.get("data", {}).get("pad_multiple", 32)
    batch = image_pair(args.image0, args.image1, args.resize, pad) if args.image0 else collate_pairs([SyntheticPairs(1, args.resize)[0]], pad)
    batch = to_device(batch, device)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    timings = []
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=args.amp):
        for _ in range(args.warmup):
            model(batch)
        synchronize()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.iterations):
            synchronize()
            started = time.perf_counter()
            result = model(batch)
            synchronize()
            timings.append((time.perf_counter() - started) * 1000.)
    pred0 = result.get("pred0", {})
    queries = pred0.get("keypoints")
    query_slots = int(queries.shape[1]) if queries is not None else cfg["model"].get("num_keypoints")
    valid_queries = int(pred0["valid"].sum()) if "valid" in pred0 else None
    coarse_hw = [batch["image1"].shape[-2] // 8, batch["image1"].shape[-1] // 8]
    target_cells = coarse_hw[0] * coarse_hw[1]
    report = {"device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "torch": torch.__version__, "threads": torch.get_num_threads(), "dtype": "float16_autocast" if args.amp else "float32",
              "deploy": args.deploy, "warmup": args.warmup, "iterations": args.iterations,
              "image0_shape": list(batch["image0"].shape), "image1_shape": list(batch["image1"].shape),
              "coarse1_hw_stride8": coarse_hw, "query_slots": query_slots, "valid_queries": valid_queries,
              "score_elements": query_slots * target_cells if query_slots else None,
              "median_ms": float(np.median(timings)), "p90_ms": float(np.percentile(timings, 90)),
              "min_ms": float(min(timings)), "matches": int(result["mconf"].numel()),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "random_weights": not bool(args.checkpoint), "scope": "model forward, including extraction and matching; excludes I/O, resize, host-to-device and checkpoint loading"}
    if args.profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=args.amp):
            with torch.profiler.profile(activities=activities) as profile:
                model(batch)
                synchronize()
        report["profile_stages"] = {entry.key: {
            "cpu_total_ms": entry.cpu_time_total / 1000.,
            "device_total_ms": getattr(entry, "device_time_total", 0.) / 1000.
        } for entry in profile.key_averages() if entry.key.startswith("desc2feat.")}
        report["profile_note"] = "Separate instrumented forward; stage totals include nested operations and profiling overhead. CUDA device time is kernel activity, not synchronized wall time."
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        from pathlib import Path
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
