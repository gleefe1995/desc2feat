"""ALIKE detector/descriptor and eLoFTR matching loss *families*, adapted.

This is an explicit sparse-query implementation, not a claim of numerically
identical upstream reproduction. See ``docs/losses.md`` for each adaptation.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .geometry import coarse_grid, sample_map, warp_keypoints_with_visibility


# ALIKE_training/training/train.py weights/temperatures, plus eLoFTR's
# src/config/default.py focal and regression settings. New implementation
# controls (chunk size, sparse query supervision) are named explicitly.
DEFAULT_LOSS_CONFIG = {
    "peaky_weight": 0.5,
    "reprojection_weight": 1.0,
    "repeatability_weight": 1.0,
    "descriptor_weight": 5.0,
    "coarse_weight": 1.0,
    "fine_weight": 1.0,
    "local_weight": 0.5,
    "score_threshold": 0.1,
    "detector_match_radius": 5.0,
    "descriptor_temperature": 0.1,
    "repeatability_temperature": 0.1,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "pos_weight": 1.0,
    "neg_weight": 1.0,
    "coarse_sparse_supervision": True,
    "fine_sparse_supervision": True,
    "local_correct_threshold": 1.0,
    "descriptor_chunk_size": 128,
    "depth_relative_threshold": 0.2,
}


def _zero(tensor: Tensor) -> Tensor:
    # Masked log probabilities may contain -inf. Keep a finite, connected zero.
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    return value[mask].mean() if mask.any() else _zero(value)


def _log_softmax_masked(logits: Tensor, mask: Tensor) -> Tensor:
    # Finite sentinels make completely masked rows harmless before filtering.
    return F.log_softmax(logits.float().masked_fill(~mask, -1e9), dim=-1)


def _interpolation_indices(points: Tensor, feature_hw, image_hw):
    """Four border-clamped feature cells and their bilinear pixel weights."""
    h, w = feature_hw
    scale = points.new_tensor((w / image_hw[1], h / image_hw[0]))
    position = (points + 0.5) * scale - 0.5
    x, y = position[:, 0].clamp(0, w - 1), position[:, 1].clamp(0, h - 1)
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = (x0 + 1).clamp(max=w - 1), (y0 + 1).clamp(max=h - 1)
    dx, dy = x - x0, y - y0
    ids = torch.stack((y0 * w + x0, y0 * w + x1, y1 * w + x0, y1 * w + x1), dim=-1)
    weights = torch.stack(((1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy), dim=-1)
    return ids, weights


def _target_map_valid(batch, side, hw, device, dtype):
    batch_size = batch["image" + side].shape[0]
    mask = batch.get("mask" + side)
    if mask is None:
        return torch.ones((batch_size, hw[0] * hw[1]), device=device, dtype=torch.bool)
    if mask.ndim == 3:
        mask = mask[:, None]
    image_hw = batch["image" + side].shape[-2:]
    centers = coarse_grid(hw, image_hw, device=device, dtype=dtype)[None].expand(batch_size, -1, -1)
    return sample_map(mask.to(dtype), centers, image_hw)[..., 0] > 1 - 1e-5


class Desc2FeatLoss(nn.Module):
    """Return ``loss`` and seven named, unweighted differentiable components.

    Pass the flat ``config['loss']`` mapping. All defaults are public in
    :data:`DEFAULT_LOSS_CONFIG`; unknown settings raise rather than being ignored.
    Both predictions are required whenever any ALIKE-family term is enabled.
    """

    def __init__(self, config: Mapping | None = None):
        super().__init__()
        config = dict(config or {})
        unknown = set(config) - set(DEFAULT_LOSS_CONFIG)
        if unknown:
            raise ValueError(f"Unknown loss configuration keys: {sorted(unknown)}")
        self.config = {**DEFAULT_LOSS_CONFIG, **config}
        for key in ("descriptor_temperature", "repeatability_temperature", "detector_match_radius", "local_correct_threshold", "depth_relative_threshold"):
            if self.config[key] <= 0:
                raise ValueError(f"{key} must be positive")
        if int(self.config["descriptor_chunk_size"]) < 1:
            raise ValueError("descriptor_chunk_size must be positive")
        for key, value in self.config.items():
            if key.endswith("_weight") and value < 0:
                raise ValueError(f"{key} must be nonnegative")
        if not 0 <= self.config["focal_alpha"] <= 1 or self.config["focal_gamma"] < 0:
            raise ValueError("Focal alpha must be in [0,1] and gamma nonnegative")

    def _peaky(self, predictions):
        numerator = _zero(predictions[0]["dispersion"])
        count = 0
        for pred in predictions:
            mask = pred["valid"] & (pred["scores"].detach() > self.config["score_threshold"])
            numerator = numerator + pred["dispersion"][mask].sum()
            count += mask.sum()
        return numerator / torch.as_tensor(count, device=numerator.device).clamp_min(1)

    def _reprojection(self, pred0, pred1, warp0, warp1, visible0, visible1):
        p0, p1 = pred0["keypoints"], pred1["keypoints"]
        numerator = _zero(p0) + _zero(p1)
        count = 0
        threshold = self.config["score_threshold"]
        valid0 = visible0 & pred0["valid"] & (pred0["scores"].detach() > threshold)
        valid1 = visible1 & pred1["valid"] & (pred1["scores"].detach() > threshold)
        for b in range(p0.shape[0]):
            ids0, ids1 = valid0[b].nonzero(as_tuple=True)[0], valid1[b].nonzero(as_tuple=True)[0]
            if ids0.numel() == 0 or ids1.numel() == 0:
                continue
            # Only assignment is detached. The gathered reprojection errors
            # below retain gradients through both detector locations and warps.
            with torch.no_grad():
                distance = (
                    torch.cdist(warp0[b, ids0].float(), p1[b, ids1].float())
                    + torch.cdist(p0[b, ids0].float(), warp1[b, ids1].float())
                ) / 2
                nearest1 = distance.argmin(dim=1)
                nearest0 = distance.argmin(dim=0)
                row = torch.arange(ids0.numel(), device=p0.device)
                mutual = nearest0[nearest1] == row
                mutual &= distance[row, nearest1] < self.config["detector_match_radius"]
                selected0, selected1 = ids0[mutual], ids1[nearest1[mutual]]
            errors = (
                (warp0[b, selected0] - p1[b, selected1]).abs().sum(-1)
                + (p0[b, selected0] - warp1[b, selected1]).abs().sum(-1)
            ) / 2
            numerator = numerator + errors.sum()
            count += errors.numel()
        return numerator / max(count, 1)

    def _descriptor_and_repeatability(self, pred_src, pred_dst, batch, dst, warped, visible, outside):
        """Directional NRE and descriptor-guided score repeatability sums."""
        source_desc = F.normalize(pred_src["descriptors"].float(), dim=-1)
        target_map = F.normalize(pred_dst["descriptor_map"].float(), dim=1)
        feature_hw = target_map.shape[-2:]
        image_hw = batch["image" + dst].shape[-2:]
        target_valid = _target_map_valid(batch, dst, feature_hw, target_map.device, target_map.dtype)
        nre_sum = _zero(source_desc) + _zero(target_map)
        rep_sum = _zero(pred_src["scores"]) + _zero(pred_dst["score_map"])
        nre_count = rep_count = 0
        chunk_size = int(self.config["descriptor_chunk_size"])
        compute_nre = self.config["descriptor_weight"] > 0
        compute_rep = self.config["repeatability_weight"] > 0
        for b in range(source_desc.shape[0]):
            if not target_valid[b].any():
                continue
            source_valid = pred_src["valid"][b]
            vis = source_valid & visible[b]
            out = source_valid & outside[b]
            ids = (vis | out).nonzero(as_tuple=True)[0]
            target = target_map[b].flatten(1)
            repeat_numerator = _zero(pred_src["scores"][b])
            repeat_denominator = _zero(pred_src["scores"][b])
            repeat_count = 0
            for start in range(0, ids.numel(), chunk_size):
                chosen = ids[start : start + chunk_size]
                similarity = source_desc[b, chosen] @ target
                is_visible, is_outside = vis[chosen], out[chosen]
                if is_visible.any():
                    visible_ids = chosen[is_visible]
                    points = warped[b, visible_ids].detach()
                    local_ids, local_weights = _interpolation_indices(points, feature_hw, image_hw)
                    # Geometry with no valid target feature support is unknown.
                    support = target_valid[b][local_ids] & (local_weights > 0)
                    supervised = support.any(-1)
                    local_weights = local_weights * support
                    local_weights = local_weights / local_weights.sum(-1, keepdim=True).clamp_min(1e-12)
                    local_sim = similarity[is_visible].gather(1, local_ids)
                    if compute_nre:
                        log_distribution = _log_softmax_masked(
                            similarity[is_visible] / self.config["descriptor_temperature"],
                            target_valid[b][None].expand(is_visible.sum(), -1),
                        )
                        gathered_log = log_distribution.gather(1, local_ids)
                        weight_logs = local_weights.clamp_min(1e-30).log().masked_fill(~support, -1e9)
                        # log of interpolated probability, not interpolated log
                        # probability: this is the neural reprojection objective.
                        log_probability = torch.logsumexp(gathered_log + weight_logs, dim=-1)
                        nre_sum = nre_sum - log_probability[supervised].sum()
                        nre_count += int(supervised.sum())
                    if compute_rep:
                        reliability = (
                            ((local_sim.detach() - 1) / self.config["repeatability_temperature"]).exp()
                            * local_weights
                        ).sum(-1).clamp(0, 1)
                        target_score = sample_map(
                            pred_dst["score_map"][b : b + 1], points[None], image_hw
                        )[0, :, 0]
                        weight = pred_src["scores"][b, visible_ids] * target_score
                        weight = weight * supervised
                        repeat_numerator = repeat_numerator + ((1 - reliability) * weight).sum()
                        repeat_denominator = repeat_denominator + weight.sum()
                        repeat_count += int(supervised.sum())
                if compute_nre and is_outside.any():
                    # ALIKE NRE uses cosine=1 as the fixed out-of-view bin, and
                    # includes that bin for out-of-view examples only.
                    logits = similarity[is_outside] / self.config["descriptor_temperature"]
                    logits = logits.masked_fill(~target_valid[b][None], -1e9)
                    dustbin = logits.new_full((logits.shape[0], 1), 1 / self.config["descriptor_temperature"])
                    log_probability = F.log_softmax(torch.cat((logits, dustbin), dim=-1), dim=-1)[:, -1]
                    nre_sum = nre_sum - log_probability.sum()
                    nre_count += logits.shape[0]
            if compute_rep and repeat_count:
                # Normalize within each image/direction, then weight the batch
                # reduction by its number of visible keypoints, as in ALIKE.
                rep_sum = rep_sum + repeat_numerator / repeat_denominator.clamp_min(1e-12) * repeat_count
                rep_count += repeat_count
        return nre_sum, nre_count, rep_sum, rep_count

    def _positive_focal(self, log_probability: Tensor):
        log_probability = log_probability.float().clamp(min=-80, max=-1e-6)
        probability = log_probability.exp()
        return -self.config["focal_alpha"] * (1 - probability).pow(self.config["focal_gamma"]) * log_probability

    def _negative_focal(self, log_probability: Tensor):
        probability = log_probability.float().exp().clamp(min=1e-6, max=1 - 1e-6)
        # Upstream eLoFTR uses alpha for both signs (not 1-alpha for negatives).
        return -self.config["focal_alpha"] * probability.pow(self.config["focal_gamma"]) * torch.log1p(-probability)

    def _coarse(self, output, pred0, warped, visible, outside):
        log_probs = output["coarse_log_probs"]
        b, k, length = log_probs.shape
        if k == 0 or length == 0:
            return _zero(log_probs)
        ht, wt = output["target_hw"]
        if length != ht * wt:
            raise ValueError("coarse_log_probs target size does not match target_hw")
        image_hw = output["image_hw1"]
        with torch.no_grad():
            scale = warped.new_tensor((wt / image_hw[1], ht / image_hw[0]))
            cells = ((warped.detach() + 0.5) * scale).floor().long()
            cells[..., 0].clamp_(0, wt - 1)
            cells[..., 1].clamp_(0, ht - 1)
            targets = cells[..., 1] * wt + cells[..., 0]
            source_valid = output.get("source_valid", pred0["valid"]) & pred0["valid"]
            target_valid = output.get("target_valid")
            if target_valid is None:
                target_valid = torch.ones((b, length), device=log_probs.device, dtype=torch.bool)
            positive = source_valid & visible & target_valid.gather(1, targets)
            coarse_mask = output.get("coarse_mask")
            if coarse_mask is not None:
                positive &= coarse_mask.gather(2, targets[..., None])[..., 0]
        positive_log = log_probs.gather(2, targets[..., None])[..., 0]
        positive &= torch.isfinite(positive_log.detach())
        value = self.config["pos_weight"] * _masked_mean(self._positive_focal(positive_log), positive)
        if not self.config["coarse_sparse_supervision"]:
            # This opt-in dense-negative path intentionally allocates a mask;
            # the default sparse-positive path needs no extra B*K*L label map.
            valid = (source_valid & (visible | outside))[..., None] & target_valid[:, None]
            if coarse_mask is not None:
                valid &= coarse_mask
            negative = valid.clone()
            negative.scatter_(2, targets[..., None], ~positive[..., None])
            negative &= valid
            value = value + self.config["neg_weight"] * _masked_mean(self._negative_focal(log_probs), negative)
        return value

    def _fine(self, output, warped, visible):
        fine = output.get("fine")
        if fine is None:
            zero = _zero(output["coarse_log_probs"])
            return zero, zero
        logits = fine["logits"]
        zero = _zero(logits) + _zero(fine["keypoints"])
        if "local_logits" in fine:
            zero = zero + _zero(fine["local_logits"])
        if logits.shape[0] == 0:
            return zero, zero
        batch_ids, query_ids = fine["batch_ids"], fine["query_ids"]
        target = warped[batch_ids, query_ids].detach()
        geometry_valid = visible[batch_ids, query_ids]
        points, valid = fine["window_points"].detach(), fine["valid"]
        with torch.no_grad():
            side = math.isqrt(points.shape[1])
            if side * side != points.shape[1]:
                raise ValueError("Fine window must be a square grid")
            minimum, maximum = points.amin(dim=1), points.amax(dim=1)
            spacing = ((maximum - minimum) / max(side - 1, 1)).clamp_min(1e-6)
            inside = ((target >= minimum - spacing / 2) & (target <= maximum + spacing / 2)).all(-1)
            distance = (points - target[:, None]).square().sum(-1).masked_fill(~valid, float("inf"))
            closest = distance.argmin(-1)
            supervised = geometry_valid & inside & valid.any(-1) & valid.gather(1, closest[:, None])[:, 0]
        log_probs = _log_softmax_masked(logits, valid)
        selected_log = log_probs.gather(1, closest[:, None])[:, 0]
        fine_loss = self.config["pos_weight"] * _masked_mean(self._positive_focal(selected_log), supervised)
        if not self.config["fine_sparse_supervision"]:
            negative = valid & supervised[:, None]
            negative = negative.clone()
            negative.scatter_(1, closest[:, None], False)
            fine_loss = fine_loss + self.config["neg_weight"] * _masked_mean(self._negative_focal(log_probs), negative)
        local_points, local_valid = fine["local_points"].detach(), fine["local_valid"]
        with torch.no_grad():
            center = local_points[:, 4]
            local_spacing = ((local_points.amax(1) - local_points.amin(1)) / 2).clamp_min(1e-6)
            offset = (target - center) / local_spacing
            local_supervised = geometry_valid & local_valid[:, 4] & (offset.abs().amax(-1) < self.config["local_correct_threshold"])
            # A local crop intersecting padding still cannot supervise an
            # expectation outside its available sample-coordinate support.
            valid_min = local_points.masked_fill(~local_valid[..., None], float("inf")).amin(1)
            valid_max = local_points.masked_fill(~local_valid[..., None], -float("inf")).amax(1)
            local_supervised &= ((target >= valid_min) & (target <= valid_max)).all(-1)
        error = ((fine["keypoints"].float() - target) / local_spacing).square().sum(-1)
        local_loss = _masked_mean(error, local_supervised) + zero
        return fine_loss + zero, local_loss

    def forward(self, output: Mapping, batch: Mapping) -> dict[str, Tensor]:
        # Descriptor distributions, geometry and logarithms require float32
        # even when the caller wraps the training step in mixed precision.
        with torch.autocast(device_type=output["pred0"]["keypoints"].device.type, enabled=False):
            return self._forward(output, batch)

    def _forward(self, output: Mapping, batch: Mapping) -> dict[str, Tensor]:
        pred0 = output["pred0"]
        zero = _zero(pred0["score_map"]) + _zero(pred0["descriptor_map"])
        components = {key: zero for key in ("peaky", "reprojection", "repeatability", "descriptor", "coarse", "fine", "local")}
        kwargs = {"depth_relative_threshold": self.config["depth_relative_threshold"]}
        warp0, visible0, outside0 = warp_keypoints_with_visibility(pred0["keypoints"], batch, **kwargs)
        visible0 &= pred0["valid"]
        outside0 &= pred0["valid"]
        alike_enabled = any(self.config[key + "_weight"] > 0 for key in ("peaky", "reprojection", "repeatability", "descriptor"))
        if alike_enabled:
            if "pred1" not in output or output["pred1"] is None:
                raise ValueError("ALIKE training losses require output['pred1']; run the model in training mode")
            pred1 = output["pred1"]
            warp1, visible1, outside1 = warp_keypoints_with_visibility(pred1["keypoints"], batch, "1to0", **kwargs)
            visible1 &= pred1["valid"]
            outside1 &= pred1["valid"]
            if self.config["peaky_weight"] > 0:
                components["peaky"] = self._peaky((pred0, pred1))
            if self.config["reprojection_weight"] > 0:
                components["reprojection"] = self._reprojection(pred0, pred1, warp0, warp1, visible0, visible1)
            if self.config["descriptor_weight"] > 0 or self.config["repeatability_weight"] > 0:
                nre0, n0, rep0, r0 = self._descriptor_and_repeatability(pred0, pred1, batch, "1", warp0, visible0, outside0)
                nre1, n1, rep1, r1 = self._descriptor_and_repeatability(pred1, pred0, batch, "0", warp1, visible1, outside1)
                components["descriptor"] = (nre0 + nre1) / max(n0 + n1, 1)
                components["repeatability"] = (rep0 + rep1) / max(r0 + r1, 1)
        if self.config["coarse_weight"] > 0:
            components["coarse"] = self._coarse(output, pred0, warp0, visible0, outside0)
        if self.config["fine_weight"] > 0 or self.config["local_weight"] > 0:
            components["fine"], components["local"] = self._fine(output, warp0, visible0)
        loss = sum(self.config[key + "_weight"] * value for key, value in components.items())
        return {"loss": loss, **components}
