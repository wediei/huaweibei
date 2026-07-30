from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from solution.radio_map.building_prior import build_building_prior
from solution.radio_map.data import RoundDataset
from solution.radio_map.geometry import PlyPointCloud, build_geometry_prior
from solution.tests.fixtures import create_round_dir


class BuildingPriorTests(unittest.TestCase):
    def test_ground_suppression_is_parallel_and_finite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = create_round_dir(Path(temporary))
            source = build_geometry_prior(
                PlyPointCloud.open(data_dir / "Round1_Map.ply"),
                height_layers=4,
                batch_size=1,
            )
            original = source.features.copy()
            prior = build_building_prior(
                source,
                user_height=1.5,
                elevated_thresholds=(2.5, 5.0),
                density_windows_metres=(3.0,),
            )
            self.assertTrue(np.isfinite(prior.features).all())
            self.assertIn("elevated_2p5_occupancy", prior.feature_names)
            self.assertEqual(prior.metadata["format_version"], 1)
            np.testing.assert_array_equal(source.features, original)
            sampled = prior.sample_points(
                np.asarray(RoundDataset.open(data_dir).train_pos)
            )
            self.assertEqual(sampled.shape[1], len(prior.feature_names))


if __name__ == "__main__":
    unittest.main()
