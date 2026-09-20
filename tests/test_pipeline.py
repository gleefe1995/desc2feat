import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from desc2feat.evaluate import _pose_error, error_auc
from desc2feat.prepare_megadepth import convert


class PipelineTest(unittest.TestCase):
    def test_converter_filters_and_inverts_reverse_pose(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            camera0, camera1 = np.eye(4), np.eye(4)
            camera0[0, 3], camera1[0, 3] = 2., 5.
            pairs = np.empty((2, 3), dtype=object)
            pairs[0] = [(0, 1), .7, None]
            pairs[1] = [(0, 1), .1, None]
            np.savez(root / "001.npz", image_paths=np.array(["a.jpg", "b.jpg"]),
                     depth_paths=np.array(["a.h5", "b.h5"]), intrinsics=np.stack([np.eye(3)] * 2),
                     poses=np.stack([camera0, camera1]), pair_infos=pairs)
            (root / "split.txt").write_text("001\n")
            report = convert(root, root / "split.txt", root / "data", root / "pairs.jsonl", bidirectional=True)
            rows = [json.loads(line) for line in (root / "pairs.jsonl").read_text().splitlines()]
            self.assertEqual(report["pairs"], 2)
            self.assertEqual(rows[0]["T_0to1"][0][3], 3.)
            self.assertEqual(rows[1]["T_0to1"][0][3], -3.)
            self.assertEqual(rows[0]["image0"], rows[1]["image1"])
            np.savez(root / "001.npz", image_paths=np.array(["a.jpg", "b.jpg"]),
                     intrinsics=np.stack([np.eye(3)] * 2), poses=np.stack([camera0, camera1]), pair_infos=pairs)
            convert(root, root / "split.txt", root / "data", root / "poses.jsonl", pose_only=True)
            pose_row = json.loads((root / "poses.jsonl").read_text())
            self.assertNotIn("depth0", pose_row)
            self.assertEqual(pose_row["T_0to1"][0][3], 3.)

    def test_auc_retains_estimation_failures(self):
        self.assertEqual(error_auc([float("inf"), float("inf")], [5])["5"], 0.)
        self.assertEqual(error_auc([0., 0.], [5])["5"], 1.)
        self.assertEqual(error_auc([0., float("inf")], [5])["5"], .5)

    def test_pose_known_camera_geometry(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is optional")
        rng = np.random.default_rng(7)
        xyz = rng.uniform([-1., -1., 3.], [1., 1., 8.], size=(100, 3))
        k = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
        transform = np.eye(4)
        transform[:3, 3] = [.5, .1, .2]
        moved = xyz + transform[:3, 3]
        points0, points1 = xyz @ k.T, moved @ k.T
        points0, points1 = points0[:, :2] / points0[:, 2:], points1[:, :2] / points1[:, 2:]
        cv2.setRNGSeed(42)
        rotation_error, translation_error = _pose_error(points0, points1, k, k, transform, cv2, 1.)
        self.assertLess(rotation_error, .1)
        self.assertLess(translation_error, .1)


if __name__ == "__main__":
    unittest.main()
