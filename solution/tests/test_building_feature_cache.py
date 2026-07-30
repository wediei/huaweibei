from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path

import numpy as np

from solution.radio_map.building_prior import build_building_prior
from solution.radio_map.data import RoundDataset
from solution.radio_map.geometry import PlyPointCloud, build_geometry_prior
from solution.radio_map.learning.building_feature_cache import (
    BuildingFeatureCache,
    coverage_feature_cache,
)
from solution.tests.fixtures import create_round_dir


class BuildingFeatureCacheTests(unittest.TestCase):
    def test_cache_is_fold_normalized_hashed_and_switchable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = create_round_dir(root)
            dataset = RoundDataset.open(data_dir)
            source = build_geometry_prior(
                PlyPointCloud.open(data_dir / "Round1_Map.ply"),
                height_layers=4,
                batch_size=1,
            )
            building = build_building_prior(
                source,
                elevated_thresholds=(2.5, 5.0),
                density_windows_metres=(3.0,),
            )
            source_path = root / "source.npz"
            building_path = root / "building.npz"
            source.save(source_path)
            building.save(building_path)
            cache = coverage_feature_cache(
                dataset,
                source,
                building,
                root / "features",
                validation_fraction=0.34,
                grid_size=2.0,
                path_samples=8,
                source_paths={"source": source_path, "building": building_path},
            )
            self.assertEqual(cache.train_features.shape[0], 6)
            self.assertEqual(cache.test_features.shape[0], 2)
            self.assertTrue(np.isfinite(cache.train_features).all())
            self.assertTrue(np.all(cache.values("train", mode="zero") == 0))
            shuffled = cache.values("test", mode="shuffle", seed=1)
            self.assertEqual(shuffled.shape, cache.test_features.shape)
            self.assertEqual(
                BuildingFeatureCache.load(root / "features").fingerprint,
                cache.fingerprint,
            )
            del cache, dataset
            gc.collect()


if __name__ == "__main__":
    unittest.main()
