from __future__ import annotations

import unittest

import numpy as np


class SpatialSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        x, y = np.meshgrid(np.arange(6), np.arange(5), indexing="ij")
        self.positions = np.column_stack(
            (x.ravel(), y.ravel(), np.full(x.size, 1.5))
        ).astype(np.float64)

    def test_coverage_split_is_deterministic_disjoint_and_complete(self) -> None:
        from solution.radio_map.splits import coverage_split

        first = coverage_split(
            self.positions, validation_fraction=0.2, grid_size=2.0, seed=17
        )
        second = coverage_split(
            self.positions, validation_fraction=0.2, grid_size=2.0, seed=17
        )

        np.testing.assert_array_equal(first.train, second.train)
        np.testing.assert_array_equal(first.validation, second.validation)
        self.assertEqual(len(first.validation), 6)
        self.assertEqual(
            set(first.train) | set(first.validation), set(range(len(self.positions)))
        )
        self.assertFalse(set(first.train) & set(first.validation))

    def test_block_split_holds_out_high_coordinate_region(self) -> None:
        from solution.radio_map.splits import block_split

        split = block_split(self.positions, axis=0, validation_fraction=0.2)

        self.assertEqual(len(split.validation), 6)
        self.assertGreaterEqual(
            self.positions[split.validation, 0].min(),
            self.positions[split.train, 0].max(),
        )

    def test_nearest_anchor_distances_match_known_geometry(self) -> None:
        from solution.radio_map.splits import nearest_anchor_distances

        anchors = np.array([[0.0, 0.0, 1.5], [2.0, 0.0, 1.5]])
        queries = np.array([[0.0, 0.0, 1.5], [1.0, 0.0, 1.5]])

        distances = nearest_anchor_distances(anchors, queries)

        np.testing.assert_allclose(distances, [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
