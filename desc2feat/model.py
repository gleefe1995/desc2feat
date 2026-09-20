"""eLoFTR RepVGG + ALIKE-style differentiable queries + asymmetric matching.

Coordinates are pixel centers (x,y). Every interpolation uses align_corners=False.
Top-k/NMS and discrete match assignments are detached; local detector positions,
descriptor sampling and the refinement distributions remain differentiable.
"""
import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .config import DEFAULT_CONFIG
from .geometry import sample_map, coarse_grid, warp_keypoints
from .vendor.repvgg import RepVGGBlock


def offsets(radius, device, dtype):
    axis = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((x, y), -1).reshape(-1, 2)


def masked_log_softmax(logits, mask, dim):
    # Finite sentinels avoid NaNs for empty images/padded query rows.
    logits = logits.float().masked_fill(~mask, -1e9)
    return F.log_softmax(logits, dim=dim).masked_fill(~mask, -1e9)


class Backbone(nn.Module):
    """Same layer names/shapes as eLoFTR's training RepVGG backbone by default."""
    def __init__(self, blocks):
        super().__init__()
        self.layer0 = RepVGGBlock(1, 64, 3, stride=2, padding=1)
        cin = 64
        for stage, (cout, count) in enumerate(zip((64, 128, 256), blocks), 1):
            modules = []
            for i in range(count):
                modules.append(RepVGGBlock(cin, cout, 3, stride=2 if stage > 1 and i == 0 else 1, padding=1))
                cin = cout
            setattr(self, f"layer{stage}", nn.ModuleList(modules))

    def forward(self, image):
        x = self.layer0(image)
        maps = []
        for stage in (self.layer1, self.layer2, self.layer3):
            for block in stage:
                x = block(x)
            maps.append(x)
        return maps


class Pyramid(nn.Module):
    """Shared lightweight FPN: fine descriptors stay at 1/2 resolution."""
    def __init__(self, dim):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, dim, 1) for c in (64, 128, 256)])
        self.refine = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1), nn.GroupNorm(8, dim), nn.GELU()) for _ in range(2)])

    def forward(self, maps):
        x = self.lateral[2](maps[2])
        for i in (1, 0):
            x = F.interpolate(x, size=maps[i].shape[-2:], mode="bilinear", align_corners=False)
            x = self.refine[i](x + self.lateral[i](maps[i]))
        return x


class SoftDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, score_map, descriptor_map, mask):
        c = self.config
        b, _, h, w = score_map.shape
        r, nr = c["detector_radius"], c["nms_radius"]
        k = min(c["num_keypoints"], h * w)
        with torch.no_grad():
            scores = score_map.detach()[:, 0]
            border = max(c["border"], r + 1)
            valid_pixels = mask.clone()
            # Erode around both padding and physical borders.
            invalid = F.max_pool2d((~mask).float()[:, None], 2 * border + 1, 1, border)[:, 0] > 0
            valid_pixels &= ~invalid
            valid_pixels[:, :border] = False
            valid_pixels[:, -border:] = False
            valid_pixels[:, :, :border] = False
            valid_pixels[:, :, -border:] = False
            maximum = F.max_pool2d(scores[:, None], 2 * nr + 1, 1, nr)[:, 0]
            maxima = (scores == maximum) & valid_pixels & (scores > c["score_threshold"])
            values, ids = scores.masked_fill(~maxima, -1).flatten(1).topk(k, dim=1)
            valid = values >= 0
            centers = torch.stack((ids % w, ids // w), -1).float()
        delta = offsets(r, score_map.device, torch.float32)
        locations = centers[:, :, None] + delta
        patch = sample_map(score_map, locations.reshape(b, -1, 2)).reshape(b, k, -1)
        prob = F.softmax(patch.float() / c["detector_temperature"], -1)
        residual = prob @ delta
        points = centers + residual
        variance = ((delta[None, None] - residual[:, :, None]) / max(r, 1)).square().sum(-1)
        dispersion = (prob * variance).sum(-1)
        descriptors = F.normalize(sample_map(descriptor_map, points, (h, w)), dim=-1)
        return {"keypoints": points, "valid": valid,
                "scores": sample_map(score_map, points)[..., 0],
                "dispersion": dispersion, "score_map": score_map,
                "descriptor_map": descriptor_map, "descriptors": descriptors}


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.merge = nn.Linear(dim, dim, bias=False)
        self.ffn = nn.Sequential(nn.Linear(dim * 2, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.outnorm = nn.LayerNorm(dim)

    def rope(self, tensor, points):
        # Continuous 2-D RoPE on actual sparse coordinates; no fake 1-D grid.
        d = tensor.shape[-1]
        frequency = torch.exp(-math.log(10000) * torch.arange(d // 4, device=tensor.device).float() / (d // 4))
        phase = (points.float()[..., :, None] / 8 * frequency).flatten(-2)
        phase = phase[:, None]
        a, b = tensor.float()[..., 0::2], tensor.float()[..., 1::2]
        return torch.stack((a * phase.cos() - b * phase.sin(), a * phase.sin() + b * phase.cos()), -1).flatten(-2).to(tensor.dtype)

    def forward(self, query, context, query_mask, context_mask, query_xy=None, context_xy=None):
        b, n, c = query.shape
        q = self.q(self.norm(query)).reshape(b, n, self.heads, c // self.heads).transpose(1, 2)
        k = self.k(self.norm(context)).reshape(b, -1, self.heads, c // self.heads).transpose(1, 2)
        v = self.v(self.norm(context)).reshape(b, -1, self.heads, c // self.heads).transpose(1, 2)
        if query_xy is not None:
            q, k = self.rope(q, query_xy), self.rope(k, context_xy)
        # A sentinel key handles all-masked batches on all SDPA backends.
        safe_mask = context_mask.clone()
        empty = ~safe_mask.any(-1)
        safe_mask[:, 0] |= empty
        v = v * context_mask[:, None, :, None]
        message = F.scaled_dot_product_attention(q, k, v, attn_mask=safe_mask[:, None, None])
        message = message.transpose(1, 2).reshape(b, n, c)
        message = self.outnorm(self.ffn(torch.cat((query, self.merge(message)), -1)))
        return query + message * query_mask[..., None] * (~empty)[:, None, None]


class AsymmetricTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.aggregation = config["aggregation"]
        self.layers = nn.ModuleList([nn.ModuleList([
            AttentionBlock(config["descriptor_dim"], config["num_heads"]),
            AttentionBlock(config["descriptor_dim"], config["num_heads"])]) for _ in range(config["num_layers"])])

    def forward(self, query, dense, query_xy, query_mask, target_mask, image_hw):
        b, c, h, w = dense.shape
        a = self.aggregation
        ph, pw = max(1, math.ceil(h / a)), max(1, math.ceil(w / a))
        # Fixed bins; divisible-by-32 inputs make all bins exact 4x4 by default.
        pooled_mask = F.adaptive_max_pool2d(target_mask[:, None].float(), (ph, pw))[:, 0].bool()
        target_xy = coarse_grid((ph, pw), image_hw, device=dense.device, dtype=torch.float32)[None].expand(b, -1, -1)
        for self_attention, cross_attention in self.layers:
            weight = F.adaptive_avg_pool2d(target_mask[:, None].float(), (ph, pw))
            pooled = F.adaptive_avg_pool2d(dense * target_mask[:, None], (ph, pw)) / weight.clamp_min(1e-6)
            pooled = pooled.flatten(2).transpose(1, 2)
            old = pooled
            query = self_attention(query, query, query_mask, query_mask, query_xy, query_xy)
            pooled = self_attention(pooled, pooled, pooled_mask.flatten(1), pooled_mask.flatten(1), target_xy, target_xy)
            q_before = query
            query = cross_attention(query, pooled, query_mask, pooled_mask.flatten(1))
            pooled = cross_attention(pooled, q_before, pooled_mask.flatten(1), query_mask)
            message = (pooled - old).transpose(1, 2).reshape(b, c, ph, pw)
            dense = dense + F.interpolate(message, size=(h, w), mode="bilinear", align_corners=False) * target_mask[:, None]
        return query, dense


def coarse_match(query, target, source_valid, target_valid, temperature, threshold, chunk_size=0):
    """Exact dual-softmax, optionally streamed in target chunks at inference.

    Streaming stores O(K*chunk + K + L) intermediate scalars, at the expense
    of computing correlations twice. This is an ablation, not a speed claim.
    """
    query, target = F.normalize(query.float(), dim=-1), F.normalize(target.float(), dim=-1)
    b, k, _ = query.shape
    length = target.shape[1]

    def logits_chunk(start, end):
        with torch.autocast(device_type=query.device.type, enabled=False):
            logits = (query @ target[:, start:end].transpose(1, 2)) / temperature
        mask = source_valid[:, :, None] & target_valid[:, None, start:end]
        return logits.masked_fill(~mask, -1e9), mask

    if not chunk_size:
        logits, mask = logits_chunk(0, length)
        log_conf = masked_log_softmax(logits, mask, -1) + masked_log_softmax(logits, mask, -2)
        values, indices = log_conf.max(-1)
        reverse = log_conf.argmax(-2)
        mutual = reverse.gather(1, indices) == torch.arange(k, device=query.device)[None]
    else:
        row_lse = query.new_full((b, k), -torch.inf)
        for start in range(0, length, chunk_size):
            logits, _ = logits_chunk(start, start + chunk_size)
            row_lse = torch.logaddexp(row_lse, torch.logsumexp(logits, -1))
        values = query.new_full((b, k), -torch.inf)
        indices = torch.zeros((b, k), device=query.device, dtype=torch.long)
        reverse_parts = []
        for start in range(0, length, chunk_size):
            logits, mask = logits_chunk(start, start + chunk_size)
            confidence = (logits - row_lse[..., None] + masked_log_softmax(logits, mask, -2)).masked_fill(~mask, -2e9)
            val, idx = confidence.max(-1)
            update = val > values
            indices = torch.where(update, idx + start, indices)
            values = torch.maximum(val, values)
            reverse_parts.append(confidence.argmax(-2))
        reverse = torch.cat(reverse_parts, -1)
        mutual = reverse.gather(1, indices) == torch.arange(k, device=query.device)[None]
        log_conf = None
    confidence = values.exp()
    keep = mutual & source_valid & target_valid.gather(1, indices) & (confidence > threshold)
    return log_conf, indices, confidence, keep


class FineRefiner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d, f = config["descriptor_dim"], config["fine_dim"]
        self.context = nn.Linear(d, f)
        self.pixel_proj = nn.Linear(f, f)
        self.local_proj = nn.Linear(f, f)

    def sample(self, fmap, batch_ids, points, hw):
        # Sample only selected patches; never unfold or duplicate full feature maps.
        result = fmap.new_zeros((*points.shape[:-1], fmap.shape[1]))
        for batch_index in range(fmap.shape[0]):
            sel = batch_ids == batch_index
            sampled = sample_map(fmap[batch_index:batch_index + 1], points[sel].reshape(1, -1, 2), hw)
            result[sel] = sampled.reshape(*points[sel].shape[:-1], fmap.shape[1]).to(result.dtype)
        return result

    def valid_points(self, mask, batch_ids, points, hw):
        h, w = hw
        inside = (points[..., 0] >= 0) & (points[..., 0] <= w - 1) & (points[..., 1] >= 0) & (points[..., 1] <= h - 1)
        sampled = self.sample(mask[:, None].float(), batch_ids, points, hw)[..., 0] > .999
        return inside & sampled

    def forward(self, feature0, feature1, query, points0, centers, batch_ids, query_ids, mask1, hw0, hw1):
        c = self.config
        source = self.sample(feature0, batch_ids, points0[:, None], hw0)[:, 0]
        source = source + self.context(query[batch_ids, query_ids])
        window = centers[:, None] + offsets(c["fine_radius"], centers.device, centers.dtype)
        target = self.sample(feature1, batch_ids, window, hw1)
        valid = self.valid_points(mask1, batch_ids, window, hw1)
        q = F.normalize(self.pixel_proj(source).float(), dim=-1)
        t = F.normalize(self.pixel_proj(target).float(), dim=-1)
        logits = (q[:, None] * t).sum(-1) / c["fine_temperature"]
        logits = logits.masked_fill(~valid, -1e9)
        best = logits.detach().argmax(-1)
        best_point = window[torch.arange(len(centers), device=centers.device), best]
        local_points = best_point[:, None] + offsets(1, centers.device, centers.dtype)
        local_target = self.sample(feature1, batch_ids, local_points, hw1)
        local_valid = self.valid_points(mask1, batch_ids, local_points, hw1)
        q = F.normalize(self.local_proj(source).float(), dim=-1)
        t = F.normalize(self.local_proj(local_target).float(), dim=-1)
        local_logits = ((q[:, None] * t).sum(-1) / c["local_temperature"]).masked_fill(~local_valid, -1e9)
        probability = F.softmax(local_logits, -1)
        points1 = (probability[..., None] * local_points).sum(-2)
        return {"batch_ids": batch_ids, "query_ids": query_ids, "centers": centers,
                "logits": logits, "window_points": window, "valid": valid,
                "keypoints": points1, "local_logits": local_logits,
                "local_points": local_points, "local_valid": local_valid}


class Desc2Feat(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = deepcopy(DEFAULT_CONFIG["model"])
        if config:
            unknown = set(config) - set(self.config)
            if unknown:
                raise ValueError(f"Unknown model config keys: {sorted(unknown)}")
            self.config.update(config)
        c = self.config
        if c["descriptor_dim"] % (4 * c["num_heads"]) or c["fine_dim"] % 8:
            raise ValueError("descriptor_dim must be divisible by 4*num_heads; fine_dim by 8")
        if c["num_keypoints"] < 1 or c["fine_radius"] < 4 or c["aggregation"] < 1:
            raise ValueError("Need positive K/aggregation and fine_radius>=4 for stride-8 coverage")
        if len(c["backbone_blocks"]) != 3 or min(c["backbone_blocks"]) < 1:
            raise ValueError("backbone_blocks needs three positive stage depths")
        if not 0 <= c["train_gt_fraction"] <= 1 or c["train_fine_samples"] < 1:
            raise ValueError("Invalid training fine sampling configuration")
        if min(c[name] for name in ("detector_temperature", "coarse_temperature", "fine_temperature", "local_temperature")) <= 0:
            raise ValueError("All temperatures must be positive")
        if c["coarse_chunk_size"] < 0 or c["num_layers"] < 0 or min(c["nms_radius"], c["detector_radius"], c["border"]) < 0:
            raise ValueError("Layer counts, radii, border and chunk size cannot be negative")
        self.backbone = Backbone(c["backbone_blocks"])
        self.pyramid = Pyramid(c["fine_dim"])
        self.coarse_head = nn.Conv2d(256, c["descriptor_dim"], 1)
        # PixelShuffle predicts separate scores for each full-resolution pixel phase.
        self.score_head = nn.Sequential(nn.Conv2d(c["fine_dim"], c["fine_dim"], 3, padding=1),
                                        nn.GELU(), nn.Conv2d(c["fine_dim"], 4, 1), nn.PixelShuffle(2))
        self.detector = SoftDetector(c)
        self.transformer = AsymmetricTransformer(c)
        self.fine = FineRefiner(c)

    def extract(self, image, mask, detect=True):
        maps = self.backbone(image)
        coarse = self.coarse_head(maps[-1])
        fine = self.pyramid(maps)
        pred = None
        if detect:
            score = torch.sigmoid(self.score_head(fine))
            pred = self.detector(score, coarse, mask)
        return coarse, fine, pred

    @torch.no_grad()
    def training_pairs(self, batch, pred, indices, keep, target_hw):
        # A mixture of actual predictions and geometry-selected examples gives the
        # untrained fine head valid targets without replacing the inference path.
        b_ids, q_ids = keep.nonzero(as_tuple=True)
        target_ids = indices[b_ids, q_ids]
        limit = self.config["train_fine_samples"]
        n_gt = int(limit * self.config["train_gt_fraction"])
        n_pred = min(len(b_ids), limit - n_gt)
        sel = torch.randperm(len(b_ids), device=indices.device)[:n_pred]
        b_ids, q_ids, target_ids = b_ids[sel], q_ids[sel], target_ids[sel]
        if n_gt == 0:
            return b_ids, q_ids, target_ids
        warped, valid = warp_keypoints(pred["keypoints"].detach(), batch)
        valid = valid & pred["valid"]
        gb, gq = valid.nonzero(as_tuple=True)
        sel = torch.randperm(len(gb), device=indices.device)[:limit - n_pred]
        gb, gq = gb[sel], gq[sel]
        h, w = target_hw
        ih, iw = batch["image1"].shape[-2:]
        xy = ((warped[gb, gq] + .5) * warped.new_tensor([w / iw, h / ih])).floor().long()
        gid = xy[:, 1].clamp(0, h - 1) * w + xy[:, 0].clamp(0, w - 1)
        return torch.cat((b_ids, gb)), torch.cat((q_ids, gq)), torch.cat((target_ids, gid))

    def forward(self, batch, return_training_outputs=False):
        """Run matching; return_training_outputs is exclusively for supervised loss.

        That flag enables target detection and GT-injected fine crops even in
        eval mode. Never use its returned matches for accuracy metrics.
        """
        supervised = self.training or return_training_outputs
        image0, image1 = batch["image0"], batch["image1"]
        for image in (image0, image1):
            if image.ndim != 4 or image.shape[1] != 1 or min(image.shape[-2:]) < 32 or any(s % 32 for s in image.shape[-2:]):
                raise ValueError("Images must be Bx1xHxW grayscale, H/W>=32 and divisible by 32; use data padding")
        if image0.shape[0] != image1.shape[0]:
            raise ValueError("Paired batch sizes must agree")
        mask0 = batch.get("mask0", torch.ones_like(image0[:, 0], dtype=torch.bool)).bool()
        mask1 = batch.get("mask1", torch.ones_like(image1[:, 0], dtype=torch.bool)).bool()
        with torch.profiler.record_function("desc2feat.backbone_detector"):
            if image0.shape[-2:] == image1.shape[-2:]:
                maps = self.backbone(torch.cat((image0, image1)))
                c0, c1 = self.coarse_head(maps[-1]).chunk(2)
                f0, f1 = self.pyramid(maps).chunk(2)
                p0 = self.detector(torch.sigmoid(self.score_head(f0)), c0, mask0)
                p1 = self.detector(torch.sigmoid(self.score_head(f1)), c1, mask1) if supervised else None
            else:
                c0, f0, p0 = self.extract(image0, mask0, True)
                c1, f1, p1 = self.extract(image1, mask1, supervised)
        b, _, h, w = c1.shape
        coarse_mask = F.interpolate(mask1[:, None].float(), (h, w), mode="area")[:, 0] > .999
        query = sample_map(c0, p0["keypoints"], image0.shape[-2:])
        with torch.profiler.record_function("desc2feat.transformer"):
            query, c1 = self.transformer(query, c1, p0["keypoints"], p0["valid"], coarse_mask, image1.shape[-2:])
        with torch.profiler.record_function("desc2feat.coarse"):
            log_conf, indices, confidence, keep = coarse_match(query, c1.flatten(2).transpose(1, 2),
                p0["valid"], coarse_mask.flatten(1), self.config["coarse_temperature"], self.config["match_threshold"],
                0 if supervised else self.config["coarse_chunk_size"])
        mb, mq = keep.nonzero(as_tuple=True)
        mt = indices[mb, mq]
        out = {"pred0": p0, "pred1": p1, "coarse_log_probs": log_conf,
               "supervised_outputs": supervised,
               "source_valid": p0["valid"], "target_valid": coarse_mask.flatten(1),
               "target_hw": (h, w), "image_hw1": image1.shape[-2:],
               "coarse_indices": indices, "coarse_confidence": confidence}
        if supervised:
            fb, fq, ft = self.training_pairs(batch, p0, indices, keep, (h, w))
        else:
            fb, fq, ft = mb, mq, mt
        grid = coarse_grid((h, w), image1.shape[-2:], device=image1.device, dtype=torch.float32)
        centers = grid[ft]
        with torch.profiler.record_function("desc2feat.fine"):
            fine = self.fine(f0, f1, query, p0["keypoints"][fb, fq], centers, fb, fq, mask1, image0.shape[-2:], image1.shape[-2:])
        out["fine"] = fine
        # During training these are the sampled pairs (may contain GT injection),
        # and are never used as evaluation metrics. Evaluation always calls eval().
        valid_fine = fine["valid"].any(-1) & fine["local_valid"].any(-1)
        final_points = fine["keypoints"]
        valid_fine &= self.fine.valid_points(mask1, fb, final_points[:, None], image1.shape[-2:])[:, 0]
        out.update({"mkpts0_f": p0["keypoints"][fb, fq][valid_fine],
                    "mkpts1_f": final_points[valid_fine],
                    "mconf": confidence[fb, fq][valid_fine], "b_ids": fb[valid_fine]})
        return out

    def load_eloftr_backbone(self, path):
        """Warm start the original training backbone only, reporting every mismatch."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        own = self.backbone.state_dict()
        selected = {}
        skipped = []
        for name, value in state.items():
            key = name
            for prefix in ("module.", "matcher.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            if key.startswith("backbone."):
                key = key[len("backbone."):]
                if key in own and own[key].shape == value.shape:
                    selected[key] = value
                else:
                    skipped.append(name)
        if not selected:
            raise ValueError("No compatible training RepVGG weights; deploy/fused eLoFTR checkpoints require training weights")
        missing, unexpected = self.backbone.load_state_dict(selected, strict=False)
        return {"loaded_tensors": len(selected), "missing": list(missing),
                "unexpected": list(unexpected), "skipped": skipped,
                "scope": "backbone only; detector/transformer/fine heads need training"}

    def switch_to_deploy(self):
        if self.training:
            raise RuntimeError("Call eval() before RepVGG fusion")
        for module in self.backbone.modules():
            if isinstance(module, RepVGGBlock):
                module.switch_to_deploy()
        return self
