from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from solution.tests.test_geometry import create_binary_ply


class AnchorMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.positions = np.array(
            [
                [0.0, 0.0, 1.5],
                [1.0, 0.0, 1.5],
                [2.0, 0.0, 1.5],
                [3.0, 0.0, 1.5],
            ],
            dtype=np.float64,
        )
        self.latents = np.arange(24, dtype=np.float32).reshape(4, 3, 2).astype(
            np.complex64
        )
        self.source_indices = np.array([10, 11, 12, 13], dtype=np.int64)

    def test_query_excludes_target_and_returns_global_indices(self) -> None:
        from solution.radio_map.anchors import AnchorMemory

        memory = AnchorMemory(
            self.positions,
            self.latents,
            bs_position=np.array([0.0, -2.0, 5.0]),
            source_indices=self.source_indices,
        )
        first = memory.query(
            self.positions[[1, 2]], k=2, exclude_source_indices=np.array([11, 12])
        )
        second = memory.query(
            self.positions[[1, 2]], k=2, exclude_source_indices=np.array([11, 12])
        )

        self.assertNotIn(11, first.source_indices[0])
        self.assertNotIn(12, first.source_indices[1])
        np.testing.assert_array_equal(first.source_indices, second.source_indices)
        np.testing.assert_allclose(first.distances[:, 0], 1.0)
        self.assertEqual(first.latents.shape, (2, 2, 3, 2))

    def test_pair_features_match_relative_geometry(self) -> None:
        from solution.radio_map.anchors import AnchorMemory, PAIR_FEATURE_NAMES

        memory = AnchorMemory(
            self.positions,
            self.latents,
            bs_position=np.array([0.0, -2.0, 5.0]),
            source_indices=self.source_indices,
        )
        result = memory.query(np.array([[0.1, 0.0, 1.5]]), k=1)

        self.assertEqual(result.source_indices[0, 0], 10)
        np.testing.assert_allclose(result.relative_positions[0, 0], [-0.1, 0.0, 0.0])
        self.assertAlmostEqual(result.distances[0, 0], 0.1)
        self.assertEqual(result.pair_features.shape[-1], len(PAIR_FEATURE_NAMES))
        np.testing.assert_allclose(result.pair_features[0, 0, :4], [-0.1, 0, 0, 0.1])

    def test_requesting_too_many_anchors_is_rejected(self) -> None:
        from solution.radio_map.anchors import AnchorMemory

        memory = AnchorMemory(
            self.positions, self.latents, bs_position=np.zeros(3)
        )

        with self.assertRaisesRegex(ValueError, "k"):
            memory.query(self.positions[:1], k=4, exclude_source_indices=np.array([0]))


class PriorBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        ply = root / "map.ply"
        create_binary_ply(ply)
        from solution.radio_map.geometry import PlyPointCloud, build_geometry_prior

        self.prior = build_geometry_prior(
            PlyPointCloud.open(ply), resolution=1.0, height_layers=4, batch_size=3
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_prior_batch_contains_all_geometry_injection_points(self) -> None:
        from solution.radio_map.anchors import AnchorMemory, build_prior_batch

        positions = np.array(
            [[0.2, 0.2, 1.5], [1.2, 0.2, 1.5], [2.2, 2.2, 1.5]]
        )
        latents = np.arange(15, dtype=np.float32).reshape(3, 5).astype(np.complex64)
        memory = AnchorMemory(
            positions,
            latents,
            bs_position=np.array([0.2, 0.2, 1.5]),
            source_indices=np.array([20, 21, 22]),
        )
        batch = build_prior_batch(
            memory,
            target_positions=positions[[1]],
            geometry=self.prior,
            k=2,
            exclude_source_indices=np.array([21]),
            patch_size=3,
            corridor_samples=4,
            anchor_corridor_samples=3,
        )
        info = batch["prior_info"]
        channel_count = len(self.prior.feature_names)

        self.assertEqual(batch["position"].shape, (1, 3))
        self.assertEqual(info["local_map_patch"].shape, (1, channel_count, 3, 3))
        self.assertEqual(info["bs_map_patch"].shape, (1, channel_count, 3, 3))
        self.assertEqual(info["corridor_features"].shape, (1, 4, channel_count + 2))
        self.assertEqual(info["anchor_indices"].shape, (1, 2))
        self.assertEqual(info["anchor_latents"].shape, (1, 2, 5))
        self.assertEqual(
            info["anchor_path_features"].shape, (1, 2, 3, channel_count + 2)
        )
        self.assertNotIn(21, info["anchor_indices"][0])
        for value in info.values():
            self.assertTrue(np.isfinite(value).all())


if __name__ == "__main__":
    unittest.main()
