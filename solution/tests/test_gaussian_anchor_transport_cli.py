import unittest
from types import SimpleNamespace

import numpy as np

from solution.radio_map.learning.gaussian_anchor_transport_cli import (
    _pair_tokens,
    build_parser,
)


class GaussianAnchorTransportCliTests(unittest.TestCase):
    class Cache:
        feature_names = tuple(f"f{i}" for i in range(2))

        def values(self, split, mode="real"):
            del mode
            if split == "train":
                values = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
            else:
                values = np.full((2, 3, 2), 100.0, np.float32)
            return values, np.ones(values.shape[:2], dtype=bool)

    def test_pair_tokens_bind_target_anchor_and_difference(self):
        neighbors = np.asarray(
            [[1, 2], [0, 3], [1, 3], [2, 0]], dtype=np.int64
        )
        rows = np.asarray([0, 2], dtype=np.int64)
        values, mask = _pair_tokens(
            self.Cache(), None, neighbors, rows, rows, "train", "real"
        )
        self.assertEqual(values.shape, (2, 2, 3, 6))
        target = self.Cache().values("train")[0][0]
        anchor = self.Cache().values("train")[0][1]
        np.testing.assert_array_equal(values[0, 0, :, :2], target)
        np.testing.assert_array_equal(values[0, 0, :, 2:4], anchor)
        np.testing.assert_array_equal(values[0, 0, :, 4:], target - anchor)
        self.assertTrue(mask.all())

    def test_zero_pair_context_only_removes_map_values(self):
        neighbors = np.asarray([[1, 2], [0, 2]], dtype=np.int64)
        rows = np.asarray([0], dtype=np.int64)
        values, mask = _pair_tokens(
            self.Cache(), None, neighbors, rows, rows, "train", "zero"
        )
        self.assertEqual(np.count_nonzero(values), 0)
        self.assertTrue(mask.all())

    def test_direct_anchor_target_path_is_bound_as_fourth_feature_block(self):
        class Direct:
            feature_names = ("d0", "d1")

            def values(self, split):
                self.assert_split = split
                tokens = np.full((4, 2, 3, 2), 7.0, np.float32)
                mask = np.ones((4, 2, 3), dtype=bool)
                neighbors = np.asarray(
                    [[1, 2], [0, 3], [1, 3], [2, 0]], dtype=np.int64
                )
                distances = np.ones((4, 2), np.float32)
                return tokens, mask, neighbors, distances

        neighbors = np.asarray(
            [[1, 2], [0, 3], [1, 3], [2, 0]], dtype=np.int64
        )
        rows = np.asarray([0, 2], dtype=np.int64)
        values, mask = _pair_tokens(
            self.Cache(), Direct(), neighbors, rows, rows, "train", "real"
        )
        self.assertEqual(values.shape, (2, 2, 3, 8))
        np.testing.assert_array_equal(values[..., 6:], 7.0)
        self.assertTrue(mask.all())

    def test_parser_exposes_independent_o4_commands(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "evaluate",
                "--data-dir", "data",
                "--cache-dir", "cache",
                "--base-checkpoint", "base.pt",
                "--refiner-checkpoint", "power.pt",
                "--supervision-cache", "supervision",
                "--gaussian-token-cache", "tokens",
                "--checkpoint", "o4.pt",
                "--output", "report.json",
            ]
        )
        self.assertEqual(args.anchor_count, 4)
        self.assertGreaterEqual(args.minimum_gain, 0.01)


if __name__ == "__main__":
    unittest.main()
