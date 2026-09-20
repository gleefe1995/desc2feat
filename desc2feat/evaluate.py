"""Evaluate pair matching and geometry, explicitly retaining zero-match failures."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PairDataset, collate_pairs, to_device, transform_points
from .infer import load_model, original_matches


def error_auc(errors, thresholds):
    """Integral of empirical recall; missing estimates have infinite error."""
    errors = np.asarray([0.] + sorted(float(error) for error in errors), dtype=np.float64)
    recall = np.linspace(0., 1., len(errors))
    result = {}
    for threshold in thresholds:
        stop = np.searchsorted(errors, threshold, side="right")
        x = np.concatenate((errors[:stop], [threshold]))
        y = np.concatenate((recall[:stop], [recall[max(stop - 1, 0)]]))
        result[str(threshold)] = float(np.trapz(y, x) / threshold)
    return result


def _project(points, homography):
    value = np.concatenate((points, np.ones((len(points), 1))), axis=1) @ homography.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return value[:, :2] / value[:, 2:3]


def _homography_error(points0, points1, homography, hw0, cv2, ransac_px):
    if cv2 is None or len(points0) < 4:
        return float("inf")
    estimate, _ = cv2.findHomography(points0, points1, cv2.RANSAC, ransac_px)
    if estimate is None or not np.isfinite(estimate).all():
        return float("inf")
    height, width = hw0
    corners = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=float)
    value = np.linalg.norm(_project(corners, estimate) - _project(corners, homography), axis=1).mean()
    return float(value) if np.isfinite(value) else float("inf")


def _normalize(points, intrinsic):
    return _project(points, np.linalg.inv(intrinsic))


def _pose_error(points0, points1, k0, k1, transform, cv2, ransac_px):
    if len(points0) < 5:
        return float("inf"), float("inf")
    p0, p1 = _normalize(points0, k0), _normalize(points1, k1)
    focal = np.mean([abs(k0[0, 0]), abs(k0[1, 1]), abs(k1[0, 0]), abs(k1[1, 1])])
    if focal <= 0 or np.linalg.norm(transform[:3, 3]) < 1e-9:
        return float("inf"), float("inf")
    essential, inlier_mask = cv2.findEssentialMat(p0, p1, np.eye(3), method=cv2.RANSAC,
                                                prob=.999, threshold=ransac_px / focal)
    if essential is None:
        return float("inf"), float("inf")
    best, best_count = None, -1
    for candidate in np.split(essential, len(essential) // 3):
        count, rotation, translation, _ = cv2.recoverPose(candidate, p0, p1, np.eye(3), mask=inlier_mask.copy())
        if count > best_count:
            best, best_count = (rotation, translation[:, 0]), count
    if best is None or best_count < 5:
        return float("inf"), float("inf")
    rotation, translation = best
    rotation_error = np.rad2deg(np.arccos(np.clip((np.trace(rotation.T @ transform[:3, :3]) - 1.) / 2., -1., 1.)))
    target = transform[:3, 3]
    cosine = np.dot(translation, target) / (np.linalg.norm(translation) * np.linalg.norm(target))
    translation_error = np.rad2deg(np.arccos(np.clip(abs(cosine), -1., 1.)))
    return float(rotation_error), float(translation_error)


def _epipolar_error(points0, points1, k0, k1, transform):
    translation = transform[:3, 3]
    tx = np.array([[0., -translation[2], translation[1]], [translation[2], 0., -translation[0]],
                   [-translation[1], translation[0], 0.]])
    fundamental = np.linalg.inv(k1).T @ tx @ transform[:3, :3] @ np.linalg.inv(k0)
    x0, x1 = np.c_[points0, np.ones(len(points0))], np.c_[points1, np.ones(len(points1))]
    line1, line0 = x0 @ fundamental.T, x1 @ fundamental
    residual = abs(np.sum(x1 * line1, axis=1))
    denominator0, denominator1 = np.linalg.norm(line0[:, :2], axis=1), np.linalg.norm(line1[:, :2], axis=1)
    distance = residual / np.maximum(np.minimum(denominator0, denominator1), 1e-12)
    distance[(denominator0 < 1e-12) | (denominator1 < 1e-12)] = np.inf
    return distance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--checkpoint")
    parser.add_argument("--config")
    parser.add_argument("--output", default="evaluation.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resize", type=int)
    parser.add_argument("--ransac-px", type=float, default=1.)
    parser.add_argument("--allow-random", action="store_true")
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--max-pairs", type=int)
    args = parser.parse_args(argv)
    if not args.checkpoint and not args.allow_random:
        parser.error("--checkpoint is required unless --allow-random is explicitly set")
    if args.max_pairs is not None and args.max_pairs < 1:
        parser.error("max-pairs must be positive")
    if args.ransac_px <= 0:
        parser.error("ransac-px must be positive")
    try:
        import cv2
        cv2.setRNGSeed(42)
    except ImportError:
        cv2 = None
    device = torch.device(args.device)
    model, cfg = load_model(args.checkpoint, args.config, device, args.allow_random, args.deploy)
    resize = args.resize if args.resize is not None else cfg.get("data", {}).get("resize", 832)
    dataset = PairDataset(args.manifest, resize, load_depth=False)
    from functools import partial
    loader = DataLoader(dataset, batch_size=1, collate_fn=partial(collate_pairs, pad_multiple=cfg.get("data", {}).get("pad_multiple", 32)))
    records, geometry_errors, pair_precision = [], {"homography": [], "pose": []}, {"homography": [], "pose": []}
    total_matches, correct_counts = {"homography": 0, "pose": 0}, {"homography": np.zeros(3, dtype=int), "pose": np.zeros(3, dtype=int)}
    for index, batch in enumerate(loader):
        if args.max_pairs is not None and index >= args.max_pairs:
            break
        batch = to_device(batch, device)
        with torch.inference_mode():
            result = model(batch)
        matches = original_matches(result, batch)
        points0, points1 = matches["keypoints0"], matches["keypoints1"]
        count = len(points0)
        record = {"pair_id": batch["pair_id"][0], "matches": count}
        if "H_0to1" in batch:
            kind = "homography"
            a0 = batch["resize_transform0"][0].cpu().numpy()
            a1 = batch["resize_transform1"][0].cpu().numpy()
            homography = np.linalg.inv(a1) @ batch["H_0to1"][0].cpu().numpy() @ a0
            distances = np.linalg.norm(_project(points0, homography) - points1, axis=1)
            try:
                error = _homography_error(points0, points1, homography, batch["original_hw0"][0].tolist(), cv2, args.ransac_px)
            except Exception as exc:
                if cv2 is None or not isinstance(exc, cv2.error):
                    raise
                error = float("inf")
            record["corner_error_px"] = error if math.isfinite(error) else None
        else:
            kind = "pose"
            if cv2 is None:
                raise ImportError("Pose evaluation requires opencv-python; install the evaluation extra.")
            a0, a1 = batch["resize_transform0"][0].cpu().numpy(), batch["resize_transform1"][0].cpu().numpy()
            k0 = np.linalg.inv(a0) @ batch["K0"][0].cpu().numpy()
            k1 = np.linalg.inv(a1) @ batch["K1"][0].cpu().numpy()
            transform = batch["T_0to1"][0].cpu().numpy()
            distances = _epipolar_error(points0, points1, k0, k1, transform)
            try:
                rotation_error, translation_error = _pose_error(points0, points1, k0, k1, transform, cv2, args.ransac_px)
            except cv2.error:
                rotation_error = translation_error = float("inf")
            error = max(rotation_error, translation_error)
            record["rotation_error_deg"] = rotation_error if math.isfinite(rotation_error) else None
            record["translation_error_deg"] = translation_error if math.isfinite(translation_error) else None
        geometry_errors[kind].append(error)
        correct = np.array([np.sum(distances < threshold) for threshold in (1, 3, 5)])
        precision = correct / max(count, 1)
        pair_precision[kind].append(precision)
        total_matches[kind] += count
        correct_counts[kind] += correct
        record.update({"kind": kind, "precision_at_1_3_5_px": precision.tolist(), "estimation_failed": not math.isfinite(error)})
        records.append(record)
    if not records:
        raise ValueError("Evaluation contains no pairs.")
    summary = {"pairs": len(records), "zero_match_pairs": sum(record["matches"] == 0 for record in records),
               "mean_matches": float(np.mean([record["matches"] for record in records])),
               "random_weights": not bool(args.checkpoint), "opencv_available": cv2 is not None,
               "coordinate_frame": "original image pixels", "ransac_threshold_px": args.ransac_px}
    for kind in ("homography", "pose"):
        errors = geometry_errors[kind]
        if errors:
            summary[kind] = {"pairs": len(errors), "failures": sum(not math.isfinite(error) for error in errors),
                             "pair_mean_precision_at_1_3_5_px": np.mean(pair_precision[kind], axis=0).tolist(),
                             "pooled_precision_at_1_3_5_px": (correct_counts[kind] / max(total_matches[kind], 1)).tolist(),
                             "auc": error_auc(errors, (3, 5, 10) if kind == "homography" else (5, 10, 20)) if cv2 is not None else None}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with destination.with_suffix(".pairs.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
