import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from desc2feat.data import PairDataset, SyntheticPairs, collate_pairs, resize_transform, transform_points


class DataTest(unittest.TestCase):
    def test_halfpixel_transform_and_inverse(self):
        a = resize_transform((10, 20), (5, 8))
        points = torch.tensor([[0., 0.], [19., 9.], [6.5, 3.25]])
        expected = (points + .5) * torch.tensor([.4, .5]) - .5
        self.assertTrue(torch.allclose(transform_points(points, a), expected))
        self.assertTrue(torch.allclose(transform_points(expected, torch.linalg.inv(a)), points, atol=2e-6))

    def test_manifest_homography_and_padding(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.fromarray(np.zeros((20, 40), dtype=np.uint8)).save(root / "a.png")
            Image.fromarray(np.zeros((30, 40), dtype=np.uint8)).save(root / "b.png")
            h = np.eye(3, dtype=float)
            h[0, 2] = 3
            (root / "pairs.jsonl").write_text(json.dumps({"image0": "a.png", "image1": "b.png", "H_0to1": h.tolist()}))
            sample = PairDataset(root / "pairs.jsonl", resize=64)[0]
            a0, a1 = sample["resize_transform0"], sample["resize_transform1"]
            self.assertTrue(torch.allclose(sample["H_0to1"], a1 @ torch.tensor(h, dtype=torch.float32) @ torch.linalg.inv(a0)))
            batch = collate_pairs([sample])
            self.assertEqual(batch["image0"].shape, (1, 1, 32, 64))
            self.assertEqual(batch["image1"].shape, (1, 1, 64, 64))
            self.assertEqual(int(batch["mask1"].sum()), 48 * 64)
            self.assertFalse(batch["mask1"][0, 48:].any())

    def test_depth_camera_resize(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.fromarray(np.zeros((20, 40), dtype=np.uint8)).save(root / "im.png")
            np.save(root / "depth.npy", np.full((20, 40), 2., dtype=np.float32))
            k = torch.tensor([[30., 0., 19.5], [0., 30., 9.5], [0., 0., 1.]])
            row = {"image0": "im.png", "image1": "im.png", "depth0": "depth.npy", "depth1": "depth.npy",
                   "K0": k.tolist(), "K1": k.tolist(), "T_0to1": np.eye(4).tolist()}
            (root / "pairs.jsonl").write_text(json.dumps(row))
            sample = PairDataset(root / "pairs.jsonl", resize=32)[0]
            self.assertTrue(torch.allclose(sample["K0"], sample["resize_transform0"] @ k))
            self.assertEqual(sample["depth0"].shape, (16, 32))
            self.assertTrue((sample["depth0"] == 2.).all())
            batch = collate_pairs([sample])
            self.assertTrue((batch["depth0"][0, 16:] == 0).all())

    def test_pose_evaluation_without_depth(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(root / "im.png")
            row = {"image0": "im.png", "image1": "im.png", "K0": np.eye(3).tolist(),
                   "K1": np.eye(3).tolist(), "T_0to1": np.eye(4).tolist()}
            (root / "pairs.jsonl").write_text(json.dumps(row))
            sample = PairDataset(root / "pairs.jsonl", resize=32, load_depth=False)[0]
            self.assertIn("T_0to1", sample)
            self.assertNotIn("depth0", sample)
            with self.assertRaisesRegex(ValueError, "needs"):
                PairDataset(root / "pairs.jsonl", resize=32)[0]

    def test_deterministic_synthetic_warp(self):
        sample = SyntheticPairs(length=2, size=64, seed=10)[0]
        again = SyntheticPairs(length=2, size=64, seed=10)[0]
        self.assertTrue(torch.equal(sample["image0"], again["image0"]))
        point0 = torch.tensor([[20., 20.], [30., 40.]])
        point1 = transform_points(point0, sample["H_0to1"]).long()
        for source, target in zip(point0.long(), point1):
            self.assertAlmostEqual(float(sample["image0"][0, source[1], source[0]]),
                                   float(sample["image1"][0, target[1], target[0]]), places=5)

    def test_reject_mixed_geometry_batch(self):
        sample0 = SyntheticPairs()[0]
        sample1 = dict(sample0)
        sample1.pop("H_0to1")
        with self.assertRaisesRegex(ValueError, "one geometry type"):
            collate_pairs([sample0, sample1])


if __name__ == "__main__":
    unittest.main()
