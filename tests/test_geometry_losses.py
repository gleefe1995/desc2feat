"""Numerical invariants for coordinate supervision and all seven loss terms."""

import math
import unittest

import torch
from torch.nn import functional as F

from desc2feat.geometry import (
    coarse_grid,
    grid_to_pixel,
    pixel_to_grid,
    sample_map,
    warp_keypoints,
    warp_keypoints_with_visibility,
)
from desc2feat.losses import DEFAULT_LOSS_CONFIG, Desc2FeatLoss


def image_batch(height=16, width=24):
    return {
        "image0": torch.zeros(1, 1, height, width),
        "image1": torch.zeros(1, 1, height, width),
        "mask0": torch.ones(1, height, width, dtype=torch.bool),
        "mask1": torch.ones(1, height, width, dtype=torch.bool),
        "H_0to1": torch.eye(3)[None],
    }


def prediction(points, h=16, w=24):
    score_logits = torch.randn(1, 1, h, w, requires_grad=True)
    descriptor_logits = torch.randn(1, 4, h // 8, w // 8, requires_grad=True)
    score_map = score_logits.sigmoid()
    descriptor_map = F.normalize(descriptor_logits, dim=1)
    return {
        "keypoints": points,
        "valid": torch.ones(points.shape[:2], dtype=torch.bool),
        "scores": sample_map(score_map, points)[..., 0],
        "dispersion": torch.full(points.shape[:2], 0.3, requires_grad=True),
        "score_map": score_map,
        "descriptor_map": descriptor_map,
        "descriptors": F.normalize(sample_map(descriptor_map, points, (h, w)), dim=-1),
    }


def example_output():
    p0 = torch.tensor([[[3.7, 3.8], [11.8, 4.0]]], requires_grad=True)
    p1 = torch.tensor([[[3.9, 3.8], [11.6, 3.9]]], requires_grad=True)
    pred0, pred1 = prediction(p0), prediction(p1)
    coarse_logits = torch.randn(1, 2, 6, requires_grad=True)
    centers = torch.tensor([[3.5, 3.5], [11.5, 3.5]])
    y, x = torch.meshgrid(torch.arange(-4, 5), torch.arange(-4, 5), indexing="ij")
    offsets = torch.stack((x, y), -1).reshape(-1, 2).float()
    y, x = torch.meshgrid(torch.arange(-1, 2), torch.arange(-1, 2), indexing="ij")
    local_offsets = torch.stack((x, y), -1).reshape(-1, 2).float()
    local_points = centers[:, None] + local_offsets
    fine_logits = torch.randn(2, 81, requires_grad=True)
    local_logits = torch.randn(2, 9, requires_grad=True)
    return {
        "pred0": pred0,
        "pred1": pred1,
        "coarse_log_probs": F.log_softmax(coarse_logits, -1) + F.log_softmax(coarse_logits, -2),
        "target_hw": (2, 3),
        "image_hw1": (16, 24),
        "source_valid": pred0["valid"],
        "target_valid": torch.ones(1, 6, dtype=torch.bool),
        "fine": {
            "batch_ids": torch.zeros(2, dtype=torch.long),
            "query_ids": torch.arange(2),
            "centers": centers,
            "logits": fine_logits,
            "window_points": centers[:, None] + offsets,
            "valid": torch.ones(2, 81, dtype=torch.bool),
            "keypoints": (F.softmax(local_logits, -1)[..., None] * local_points).sum(1),
            "local_logits": local_logits,
            "local_points": local_points,
            "local_valid": torch.ones(2, 9, dtype=torch.bool),
        },
    }


class GeometryTests(unittest.TestCase):
    def test_half_pixel_round_trip_non_square(self):
        points = torch.tensor([[[0.0, 0.0], [12.0, 6.0], [2.25, 3.75]]], dtype=torch.float64)
        grid = pixel_to_grid(points, (7, 13))
        torch.testing.assert_close(grid[0, 0], torch.tensor([-12 / 13, -6 / 7], dtype=torch.float64))
        torch.testing.assert_close(grid_to_pixel(grid, (7, 13)), points)

    def test_pixel_center_sampling_and_coarse_centers(self):
        feature = torch.arange(15.0).reshape(1, 1, 3, 5)
        centers = coarse_grid((3, 5), (24, 40))[None]
        torch.testing.assert_close(centers[0, 0], torch.tensor([3.5, 3.5]))
        torch.testing.assert_close(centers[0, -1], torch.tensor([35.5, 19.5]))
        torch.testing.assert_close(sample_map(feature, centers, (24, 40))[0, :, 0], feature.flatten())

    def test_subpixel_bilinear_value_and_gradient(self):
        y, x = torch.meshgrid(torch.arange(7.0), torch.arange(13.0), indexing="ij")
        feature = (x + 2 * y)[None, None]
        points = torch.tensor([[[2.25, 3.5]]], requires_grad=True)
        value = sample_map(feature, points)
        torch.testing.assert_close(value[0, 0, 0], torch.tensor(9.25))
        value.sum().backward()
        torch.testing.assert_close(points.grad, torch.tensor([[[1.0, 2.0]]]))

    def test_mixed_precision_sampling_preserves_coordinate_precision(self):
        feature = torch.arange(832.0).reshape(1, 1, 1, 832).expand(1, 1, 4, -1)
        # Values in this range are exact in fp16, so grid precision is isolated.
        points = torch.tensor([[[500.125, 1.5]]], requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            value = sample_map(feature.half(), points)
        self.assertEqual(value.dtype, torch.float32)
        self.assertAlmostEqual(value.item(), 500.125, places=3)
        value.sum().backward()
        self.assertAlmostEqual(points.grad[0, 0, 0].item(), 1.0, places=5)

    def test_homography_reverse_masks_and_coordinate_gradients(self):
        batch = image_batch()
        batch["H_0to1"][0, 0, 2] = 2
        batch["H_0to1"][0, 1, 2] = 1
        points = torch.tensor([[[4.2, 5.1], [23.0, 7.0]]], requires_grad=True)
        warped, visible, outside = warp_keypoints_with_visibility(points, batch)
        torch.testing.assert_close(warped[0, 0], torch.tensor([6.2, 6.1]))
        self.assertEqual(visible.tolist(), [[True, False]])
        self.assertEqual(outside.tolist(), [[False, True]])
        back, back_valid = warp_keypoints(warped[:, :1], batch, "1to0")
        torch.testing.assert_close(back, points[:, :1])
        self.assertTrue(back_valid.item())
        warped[visible].sum().backward()
        torch.testing.assert_close(points.grad, torch.tensor([[[1.0, 1.0], [0.0, 0.0]]]))
        batch["mask1"][0, 6:8, 6:8] = False
        _, visible, outside = warp_keypoints_with_visibility(points.detach(), batch)
        self.assertFalse(visible.any())
        self.assertFalse(outside[0, 0])  # padding/unknown support is not dustbin

    def test_depth_consistency_distinguishes_holes_and_outside(self):
        batch = image_batch(8, 12)
        del batch["H_0to1"]
        batch.update({
            "depth0": torch.full((1, 8, 12), 2.0),
            "depth1": torch.full((1, 8, 12), 2.0),
            "K0": torch.eye(3)[None], "K1": torch.eye(3)[None],
            "T_0to1": torch.eye(4)[None],
        })
        points = torch.tensor([[[2.0, 2.0], [4.0, 2.0], [6.0, 2.0]]], requires_grad=True)
        batch["depth0"][0, 2, 4] = 0
        batch["depth1"][0, 2, 6] = 4  # projected depth2 disagrees with observed4
        warped, visible, outside = warp_keypoints_with_visibility(points, batch)
        self.assertEqual(visible.tolist(), [[True, False, False]])
        self.assertFalse(outside.any())
        warped[visible].sum().backward()
        self.assertTrue(torch.isfinite(points.grad).all())
        self.assertGreater(points.grad[0, 0].abs().sum().item(), 0)
        batch["T_0to1"][0, 0, 3] = 30  # true calculable out-of-view projection
        _, visible, outside = warp_keypoints_with_visibility(points.detach(), batch)
        self.assertFalse(visible.any())
        self.assertEqual(outside.tolist(), [[True, False, True]])

    def test_empty_point_arrays(self):
        points = torch.empty(1, 0, 2, requires_grad=True)
        feature = torch.ones(1, 3, 8, 12, requires_grad=True)
        sampled = sample_map(feature, points)
        self.assertEqual(sampled.shape, (1, 0, 3))
        sampled.sum().backward()
        self.assertIsNotNone(feature.grad)
        warped, valid = warp_keypoints(points, image_batch(8, 12))
        self.assertEqual(warped.shape, points.shape)
        self.assertEqual(valid.shape, (1, 0))


class LossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)

    def test_all_seven_components_finite_and_backward(self):
        output = example_output()
        loss = Desc2FeatLoss()(output, image_batch())
        self.assertEqual(set(loss), {"loss", "peaky", "reprojection", "repeatability", "descriptor", "coarse", "fine", "local"})
        for key, value in loss.items():
            self.assertTrue(torch.isfinite(value), key)
            self.assertGreaterEqual(value.item(), 0, key)
        expected = sum(DEFAULT_LOSS_CONFIG[key + "_weight"] * value for key, value in loss.items() if key != "loss")
        torch.testing.assert_close(loss["loss"], expected)
        loss["loss"].backward()
        self.assertTrue(torch.isfinite(output["pred0"]["keypoints"].grad).all())
        self.assertGreater(output["fine"]["local_logits"].grad.abs().sum().item(), 0)
        self.assertGreater(output["fine"]["logits"].grad.abs().sum().item(), 0)

    def test_reprojection_keeps_detector_coordinate_gradients(self):
        output = example_output()
        losses = Desc2FeatLoss()(output, image_batch())
        self.assertGreater(losses["reprojection"].item(), 0)
        losses["reprojection"].backward()
        for side in ("pred0", "pred1"):
            self.assertGreater(output[side]["keypoints"].grad.abs().sum().item(), 0)

    def test_all_masked_no_false_ground_truth_and_finite_backward(self):
        output, batch = example_output(), image_batch()
        output["pred0"]["valid"].zero_()
        output["pred1"]["valid"].zero_()
        output["target_valid"].zero_()
        batch["mask0"].zero_()
        batch["mask1"].zero_()
        output["fine"]["valid"].zero_()
        output["fine"]["local_valid"].zero_()
        output["coarse_log_probs"] = output["coarse_log_probs"].masked_fill(torch.ones_like(output["coarse_log_probs"], dtype=torch.bool), -float("inf"))
        loss = Desc2FeatLoss()(output, batch)
        for key, value in loss.items():
            self.assertTrue(torch.isfinite(value), key)
            self.assertEqual(value.item(), 0, key)
        loss["loss"].backward()
        self.assertTrue(torch.isfinite(output["pred0"]["keypoints"].grad).all())

    def test_empty_fine_batch_is_connected_zero(self):
        output = example_output()
        output["fine"] = {key: value[:0] for key, value in output["fine"].items()}
        losses = Desc2FeatLoss()(output, image_batch())
        self.assertEqual(losses["fine"].item(), 0)
        self.assertEqual(losses["local"].item(), 0)
        losses["loss"].backward()

    def test_out_of_view_nre_has_explicit_dustbin(self):
        batch = image_batch(8, 16)
        batch["H_0to1"][0, 0, 2] = 20
        points = torch.tensor([[[3.5, 3.5]]], requires_grad=True)
        pred0, pred1 = prediction(points, 8, 16), prediction(points.clone(), 8, 16)
        # Both target cells have cosine1, and dustbin has cosine1: p(bin)=1/3.
        pred0["descriptors"] = torch.ones(1, 1, 4, requires_grad=True)
        pred1["descriptor_map"] = torch.ones(1, 4, 1, 2, requires_grad=True)
        warped, visible, outside = warp_keypoints_with_visibility(points, batch)
        total, count, _, _ = Desc2FeatLoss()._descriptor_and_repeatability(
            pred0, pred1, batch, "1", warped, visible, outside
        )
        self.assertEqual(count, 1)
        self.assertAlmostEqual(total.item(), math.log(3), places=5)
        total.backward()
        self.assertIsNotNone(pred0["descriptors"].grad)

    def test_in_view_nre_interpolates_probability_not_log_probability(self):
        batch = image_batch(8, 16)
        points = torch.tensor([[[7.5, 3.5]]], requires_grad=True)  # equal cell weights
        pred0, pred1 = prediction(points, 8, 16), prediction(points.clone(), 8, 16)
        pred0["descriptors"] = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
        pred1["descriptor_map"] = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]], requires_grad=True)
        warped, visible, outside = warp_keypoints_with_visibility(points, batch)
        total, count, _, _ = Desc2FeatLoss()._descriptor_and_repeatability(
            pred0, pred1, batch, "1", warped, visible, outside
        )
        self.assertEqual(count, 1)
        # 0.5 * p(cell0) + 0.5 * p(cell1) is exactly 0.5.
        self.assertAlmostEqual(total.item(), math.log(2), places=5)

    def test_sparse_coarse_labels_use_half_pixel_cell_centers(self):
        output = example_output()
        criterion = Desc2FeatLoss()
        warped = torch.tensor([[[7.49, 3.5], [7.51, 3.5]]])
        visible = torch.ones(1, 2, dtype=torch.bool)
        outside = torch.zeros_like(visible)
        expected = criterion._positive_focal(torch.stack((output["coarse_log_probs"][0, 0, 0], output["coarse_log_probs"][0, 1, 1]))).mean()
        actual = criterion._coarse(output, output["pred0"], warped, visible, outside)
        torch.testing.assert_close(actual, expected)

    def test_dense_negative_option_finite(self):
        output = example_output()
        loss = Desc2FeatLoss({"coarse_sparse_supervision": False, "fine_sparse_supervision": False})(output, image_batch())
        self.assertTrue(torch.isfinite(loss["loss"]))
        loss["loss"].backward()

    def test_unknown_configuration_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown loss"):
            Desc2FeatLoss({"decriptor_weight": 0})


if __name__ == "__main__":
    unittest.main()
