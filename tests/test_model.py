import tempfile
import unittest
from pathlib import Path

import torch

from desc2feat import Desc2Feat, load_config
from desc2feat.model import coarse_match, SoftDetector
from desc2feat.geometry import sample_map


def tiny_config():
    cfg = load_config()["model"]
    cfg.update(num_keypoints=24, descriptor_dim=32, fine_dim=16, num_heads=2,
               num_layers=1, backbone_blocks=[1, 1, 1], train_fine_samples=24,
               coarse_chunk_size=17, match_threshold=0.)
    return cfg


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(17)

    def test_streamed_matching_equals_full_with_padding(self):
        q, t = torch.randn(2, 13, 32), torch.randn(2, 41, 32)
        qm, tm = torch.ones(2, 13, dtype=torch.bool), torch.ones(2, 41, dtype=torch.bool)
        qm[0, -3:] = False
        tm[1, 23:] = False
        full = coarse_match(q, t, qm, tm, .1, .001, 0)
        stream = coarse_match(q, t, qm, tm, .1, .001, 7)
        self.assertTrue(torch.equal(full[1], stream[1]))
        torch.testing.assert_close(full[2], stream[2], atol=2e-7, rtol=1e-5)
        self.assertTrue(torch.equal(full[3], stream[3]))
        tm.zero_()
        self.assertFalse(coarse_match(q, t, qm, tm, .1, 0, 7)[3].any())

    def test_different_image_shapes_and_empty_batch_member(self):
        model = Desc2Feat(tiny_config()).eval()
        batch = {"image0": torch.rand(2, 1, 64, 96), "image1": torch.rand(2, 1, 96, 64),
                 "mask0": torch.ones(2, 64, 96, dtype=torch.bool)}
        batch["mask0"][1] = False
        with torch.no_grad():
            out = model(batch)
        self.assertIsNone(out["pred1"])
        self.assertTrue(torch.isfinite(out["mkpts1_f"]).all())
        self.assertGreater(len(out["mkpts1_f"]), 0)
        self.assertTrue((out["b_ids"] == 0).all())
        batch["mask0"].zero_()
        with torch.no_grad():
            out = model(batch)
        self.assertEqual(tuple(out["mkpts0_f"].shape), (0, 2))
        self.assertEqual(tuple(out["mkpts1_f"].shape), (0, 2))

    def test_match_loss_reaches_detector_coordinates(self):
        model = Desc2Feat(tiny_config()).train()
        image = torch.rand(1, 1, 64, 64)
        batch = {"image0": image, "image1": image.clone(), "H_0to1": torch.eye(3)[None]}
        out = model(batch)
        p = out["pred0"]["keypoints"]
        p.retain_grad()
        ids = ((p.detach() + .5) / 8).long().clamp(0, 7)
        target = ids[..., 1] * 8 + ids[..., 0]
        selected = out["coarse_log_probs"].gather(-1, target[..., None])[..., 0]
        (-selected[out["source_valid"]].mean()).backward()
        self.assertGreater(p.grad.abs().sum().item(), 0)
        gradient = model.score_head[2].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)

    def test_detector_sampling_matches_pixel_coordinates(self):
        cfg = tiny_config()
        detector = SoftDetector(cfg)
        y, x = torch.meshgrid(torch.arange(32.), torch.arange(48.), indexing="ij")
        score = torch.exp(-((x - 16.2)**2 + (y - 12.3)**2) / 3)[None, None].requires_grad_()
        desc = torch.stack((x, y, x + y), 0)[None]
        pred = detector(score, desc, torch.ones(1, 32, 48, dtype=torch.bool))
        self.assertTrue(pred["valid"][0, 0])
        expected = sample_map(score, pred["keypoints"])[..., 0]
        torch.testing.assert_close(expected, pred["scores"])
        pred["keypoints"].sum().backward()
        self.assertGreater(score.grad.abs().sum().item(), 0)

    def test_inference_does_not_read_ground_truth(self):
        model = Desc2Feat(tiny_config()).eval()
        batch = {"image0": torch.rand(1, 1, 64, 64), "image1": torch.rand(1, 1, 64, 64)}
        with torch.no_grad():
            original = model(batch)
            batch["H_0to1"] = torch.full((1, 3, 3), float("nan"))
            changed = model(batch)
        self.assertFalse(original["supervised_outputs"])
        torch.testing.assert_close(original["mkpts0_f"], changed["mkpts0_f"])
        torch.testing.assert_close(original["mkpts1_f"], changed["mkpts1_f"])

    def test_zero_gt_fraction_disables_injection(self):
        cfg = tiny_config()
        cfg.update(train_gt_fraction=0., match_threshold=1.)
        model = Desc2Feat(cfg).train()
        batch = {"image0": torch.rand(1, 1, 64, 64), "image1": torch.rand(1, 1, 64, 64),
                 "H_0to1": torch.eye(3)[None]}
        out = model(batch)
        self.assertEqual(len(out["fine"]["batch_ids"]), 0)

    def test_deploy_fusion_and_backbone_checkpoint(self):
        model = Desc2Feat(tiny_config()).eval()
        image = torch.rand(1, 1, 64, 64)
        with torch.no_grad():
            before = model.backbone(image)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eloftr.pt"
            state = {"matcher.backbone." + k: v for k, v in model.backbone.state_dict().items()}
            torch.save({"state_dict": state}, path)
            report = model.load_eloftr_backbone(path)
            self.assertFalse(report["missing"])
        model.switch_to_deploy()
        with torch.no_grad():
            after = model.backbone(image)
        for left, right in zip(before, after):
            torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
