"""Pixel-coordinate geometry with one explicit ``align_corners=False`` convention.

Coordinates are always (x, y), where integer values denote image pixel centers.
Feature cell (j, i) corresponds to ((j+.5)*W/Wf-.5, (i+.5)*H/Hf-.5).
Visibility decisions are discrete; projected coordinates retain autograd history.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def _size_xy(points: Tensor, hw: Sequence[int]) -> Tensor:
    if len(hw) != 2 or min(hw) <= 0:
        raise ValueError(f"Expected positive (height, width), got {hw}")
    return points.new_tensor((hw[1], hw[0]))


def pixel_to_grid(points: Tensor, hw: Sequence[int]) -> Tensor:
    """Convert pixel (x,y) to grid_sample coordinates, including half pixels."""
    if not points.is_floating_point():
        points = points.float()
    return 2 * (points + 0.5) / _size_xy(points, hw) - 1


def grid_to_pixel(grid: Tensor, hw: Sequence[int]) -> Tensor:
    """Inverse of :func:`pixel_to_grid`."""
    if not grid.is_floating_point():
        grid = grid.float()
    return (grid + 1) * _size_xy(grid, hw) / 2 - 0.5


def sample_map(
    feature: Tensor,
    points: Tensor,
    image_hw: Sequence[int] | None = None,
    *,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> Tensor:
    """Sample BCHW at BN2 image pixel coordinates and return BNC.

    ``image_hw`` is the coordinate system's size, which can differ from the
    feature resolution. For example, sample a stride-eight map at (3.5,3.5)
    using the full image size to obtain its upper-left cell exactly.
    """
    if feature.ndim != 4 or points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("sample_map expects BCHW features and BN2 points")
    if feature.shape[0] != points.shape[0]:
        raise ValueError("Feature and point batch dimensions must agree")
    if points.shape[1] == 0:
        return feature.new_empty((feature.shape[0], 0, feature.shape[1])) + feature.sum() * 0 + points.sum() * 0
    # Do not quantize normalized coordinates to fp16/bfloat16: a small grid
    # rounding error can move a sample substantially in a large image.
    sample_dtype = torch.float64 if feature.dtype == torch.float64 or points.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=feature.device.type, enabled=False):
        grid = pixel_to_grid(points.to(sample_dtype), image_hw or feature.shape[-2:]).unsqueeze(2)
        sampled = F.grid_sample(feature.to(sample_dtype), grid, mode=mode, padding_mode=padding_mode, align_corners=False)
    return sampled.squeeze(-1).transpose(1, 2)


def coarse_grid(
    hw: Sequence[int],
    image_hw: Sequence[int],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return HW-by-2 cell centers in full-image pixels, row-major order."""
    if len(hw) != 2 or min(hw) <= 0:
        raise ValueError("Feature height and width must be positive")
    y, x = torch.meshgrid(
        torch.arange(hw[0], device=device, dtype=dtype),
        torch.arange(hw[1], device=device, dtype=dtype),
        indexing="ij",
    )
    cells = torch.stack((x, y), dim=-1).reshape(-1, 2)
    scale = cells.new_tensor((image_hw[1] / hw[1], image_hw[0] / hw[0]))
    return (cells + 0.5) * scale - 0.5


def points_in_bounds(points: Tensor, hw: Sequence[int]) -> Tensor:
    """Require finite coordinates inside the image's pixel-center rectangle."""
    return torch.isfinite(points).all(-1) & (points >= 0).all(-1) & (points <= _size_xy(points, hw) - 1).all(-1)


def _mask_at(batch: Mapping[str, Tensor], side: str, points: Tensor, hw: Sequence[int]) -> Tensor:
    valid = points_in_bounds(points, hw)
    mask = batch.get("mask" + side)
    if mask is not None:
        if mask.ndim == 3:
            mask = mask[:, None]
        # Conservative validity: all bilinear contributors must be valid.
        value = sample_map(mask.to(points.dtype), torch.nan_to_num(points), hw)[..., 0]
        valid = valid & (value > 1 - 1e-5)
    return valid


def _image_hw(batch: Mapping[str, Tensor], side: str) -> tuple[int, int]:
    image = batch.get("image" + side, batch.get("depth" + side))
    if image is None:
        raise KeyError(f"Geometry requires image{side} or depth{side} to define its coordinate frame")
    return tuple(image.shape[-2:])


def _sample_depth(depth: Tensor, points: Tensor, hw: Sequence[int]) -> tuple[Tensor, Tensor]:
    if depth.ndim == 3:
        depth = depth[:, None]
    depth = depth.to(points.dtype)
    finite = torch.isfinite(depth) & (depth > 0)
    clean = torch.where(finite, depth, torch.zeros_like(depth))
    values = sample_map(clean, torch.nan_to_num(points), hw)[..., 0]
    valid = sample_map(finite.to(points.dtype), torch.nan_to_num(points), hw)[..., 0] > 1 - 1e-5
    return values, valid & (values > 0)


def warp_keypoints_with_visibility(
    points: Tensor,
    batch: Mapping[str, Tensor],
    direction: str = "0to1",
    *,
    depth_relative_threshold: float = 0.2,
) -> tuple[Tensor, Tensor, Tensor]:
    """Project points, returning (coordinates, visible, known_outside).

    A homography or depth+intrinsics+rigid pose must be supplied. Depth visibility
    requires positive, finite source/target depth and relative consistency below
    ``depth_relative_threshold`` (the eLoFTR default is 0.2). Holes, masks and
    depth-inconsistent projections are ignored, not assigned to a dustbin.
    ``known_outside`` means a calculable forward projection outside the target
    rectangle. Source depth uses bilinear sampling so detector reprojection
    gradients can also pass through its depth interpolation.
    """
    if direction not in ("0to1", "1to0"):
        raise ValueError("direction must be '0to1' or '1to0'")
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("Expected BN2 keypoints")
    if depth_relative_threshold <= 0:
        raise ValueError("depth_relative_threshold must be positive")
    src, dst = ("0", "1") if direction == "0to1" else ("1", "0")
    src_hw, dst_hw = _image_hw(batch, src), _image_hw(batch, dst)
    # Avoid mixed-precision inverse/projection while preserving double precision
    # for numerical checks and retaining coordinate gradients.
    work_points = points if points.dtype == torch.float64 else points.float()
    with torch.autocast(device_type=points.device.type, enabled=False):
        finite_source = torch.isfinite(work_points).all(-1)
        clean_points = torch.nan_to_num(work_points)
        source_valid = finite_source & _mask_at(batch, src, clean_points, src_hw)
        ones = torch.ones_like(clean_points[..., :1])
        homogeneous = torch.cat((clean_points, ones), dim=-1)
        homography = batch.get("H_0to1")
        if homography is not None:
            homography = homography.to(work_points)
            if direction == "1to0":
                homography = torch.linalg.inv(homography)
            projected = homogeneous @ homography.transpose(-1, -2)
            denominator = projected[..., 2:3]
            calculable = source_valid & torch.isfinite(projected).all(-1) & (denominator[..., 0].abs() > 1e-8)
            safe_denominator = torch.where(denominator.abs() > 1e-8, denominator, torch.ones_like(denominator))
            warped = projected[..., :2] / safe_denominator
            valid = calculable & _mask_at(batch, dst, warped, dst_hw)
        else:
            required = ("depth0", "depth1", "K0", "K1", "T_0to1")
            missing = [key for key in required if key not in batch]
            if missing:
                raise KeyError(f"Provide H_0to1 or depth geometry; missing {missing}")
            depth, depth_valid = _sample_depth(batch["depth" + src], clean_points, src_hw)
            Ksrc, Kdst = batch["K" + src].to(work_points), batch["K" + dst].to(work_points)
            transform = batch["T_0to1"].to(work_points)
            if direction == "1to0":
                transform = torch.linalg.inv(transform)
            rays = torch.linalg.solve(Ksrc, homogeneous.transpose(-1, -2)).transpose(-1, -2)
            xyz = rays * depth[..., None]
            xyz_target = xyz @ transform[:, :3, :3].transpose(-1, -2) + transform[:, None, :3, 3]
            projected = xyz_target @ Kdst.transpose(-1, -2)
            denominator = projected[..., 2:3]
            calculable = source_valid & depth_valid & (xyz_target[..., 2] > 1e-8) & torch.isfinite(projected).all(-1) & (denominator[..., 0].abs() > 1e-8)
            safe_denominator = torch.where(denominator.abs() > 1e-8, denominator, torch.ones_like(denominator))
            warped = projected[..., :2] / safe_denominator
            target_depth, target_depth_valid = _sample_depth(batch["depth" + dst], warped, dst_hw)
            depth_error = (target_depth - xyz_target[..., 2]).abs() / target_depth.clamp_min(1e-8)
            valid = calculable & _mask_at(batch, dst, warped, dst_hw) & target_depth_valid & (depth_error < depth_relative_threshold)
        outside = calculable & ~points_in_bounds(warped, dst_hw)
        # Invalid geometry is returned finite so downstream gather/grid_sample
        # does not propagate NaNs before its visibility mask is applied.
        warped = torch.nan_to_num(warped)
    return warped, valid.detach(), outside.detach()


def warp_keypoints(
    points: Tensor,
    batch: Mapping[str, Tensor],
    direction: str = "0to1",
    *,
    depth_relative_threshold: float = 0.2,
) -> tuple[Tensor, Tensor]:
    """Return differentiable projected pixel coordinates and a visibility mask."""
    warped, valid, _ = warp_keypoints_with_visibility(
        points, batch, direction, depth_relative_threshold=depth_relative_threshold
    )
    return warped, valid
