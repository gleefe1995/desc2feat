"""Boundary checks for masking, streaming, and mixed precision contracts."""
import unittest

import torch

from desc2feat.config import DEFAULT_CONFIG
from desc2feat.model import AttentionBlock, FineRefiner, coarse_match


class ReviewBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_streaming_matches_full_with_padding_and_empty_items(self):
        torch.manual_seed(45)
        query = torch.randn(4, 17, 32)
        target = torch.randn(4, 101, 32)
        source_valid = torch.rand(4, 17) > .2
        target_valid = torch.rand(4, 101) > .3
        source_valid[0] = False
        target_valid[1] = False
        full = coarse_match(query, target, source_valid, target_valid, .1, 0)
        for chunk_size in (1, 9, 51, 150):
            with self.subTest(chunk_size=chunk_size):
                streamed = coarse_match(query, target, source_valid, target_valid, .1, 0, chunk_size)
                self.assertTrue(torch.equal(full[1], streamed[1]))
                torch.testing.assert_close(full[2], streamed[2], atol=1e-7, rtol=1e-5)
                self.assertTrue(torch.equal(full[3], streamed[3]))
        self.assertFalse(full[3][:2].any())

    def test_streaming_preserves_first_index_on_exact_ties(self):
        query = torch.ones(1, 3, 8)
        target = torch.ones(1, 11, 8)
        source_valid = torch.ones(1, 3, dtype=torch.bool)
        target_valid = torch.ones(1, 11, dtype=torch.bool)
        full = coarse_match(query, target, source_valid, target_valid, .1, 0)
        streamed = coarse_match(query, target, source_valid, target_valid, .1, 0, 3)
        self.assertTrue(torch.equal(full[1], streamed[1]))
        self.assertTrue(torch.equal(full[3], streamed[3]))
        self.assertEqual(int(full[3].sum()), 1)

    def test_coarse_normalization_is_stable_under_autocast(self):
        torch.manual_seed(19)
        query = torch.randn(2, 7, 16)
        target = torch.randn(2, 23, 16)
        source_valid = torch.ones(2, 7, dtype=torch.bool)
        target_valid = torch.ones(2, 23, dtype=torch.bool)
        target_valid[1] = False
        expected = coarse_match(query, target, source_valid, target_valid, .1, 0)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = coarse_match(query, target, source_valid, target_valid, .1, 0)
        torch.testing.assert_close(actual[0], expected[0])
        self.assertTrue(torch.equal(actual[3], expected[3]))

    def test_sdpa_empty_context_and_invalid_queries_preserve_input(self):
        torch.manual_seed(12)
        block = AttentionBlock(32, 4)
        query = torch.randn(2, 5, 32, requires_grad=True)
        context = torch.randn(2, 7, 32, requires_grad=True)
        query_mask = torch.ones(2, 5, dtype=torch.bool)
        context_mask = torch.ones(2, 7, dtype=torch.bool)
        context_mask[0] = False
        query_mask[1, -1] = False
        result = block(query, context, query_mask, context_mask)
        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(result[0], query[0])
        torch.testing.assert_close(result[1, -1], query[1, -1])
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertTrue(torch.isfinite(context.grad).all())

    def test_fine_patch_sampling_handles_an_empty_batch_item(self):
        refiner = FineRefiner(DEFAULT_CONFIG["model"])
        feature = torch.randn(2, 4, 16, 16, requires_grad=True)
        points = torch.full((1, 9, 2), 12.5, requires_grad=True)
        sampled = refiner.sample(feature, torch.zeros(1, dtype=torch.long), points, (32, 32))
        self.assertEqual(tuple(sampled.shape), (1, 9, 4))
        sampled.sum().backward()
        self.assertTrue(torch.isfinite(feature.grad).all())
        self.assertTrue(torch.isfinite(points.grad).all())
        self.assertEqual(float(feature.grad[1].abs().sum()), 0)

    def test_fine_patch_sampling_handles_no_matches(self):
        refiner = FineRefiner(DEFAULT_CONFIG["model"])
        feature = torch.randn(2, 4, 16, 16)
        sampled = refiner.sample(feature, torch.empty(0, dtype=torch.long), torch.empty(0, 9, 2), (32, 32))
        self.assertEqual(tuple(sampled.shape), (0, 9, 4))


if __name__ == "__main__":
    unittest.main()
