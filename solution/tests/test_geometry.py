from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


def create_binary_ply(path: Path) -> np.ndarray:
    values = np.array(
        [
            (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (0.2, 0.2, 2.0, 1.0, 0.0, 0.0),
            (1.2, 0.2, 1.0, 0.0, 0.0, 1.0),
            (0.2, 1.2, 3.0, 1.0, 0.0, 0.0),
            (1.2, 1.2, 2.5, 0.0, 0.0, 1.0),
            (2.2, 2.2, 4.0, 0.0, 1.0, 0.0),
        ],
        dtype=[
            ("x", "<f8"),
            ("y", "<f8"),
            ("z", "<f8"),
            ("nx", "<f8"),
            ("ny", "<f8"),
            ("nz", "<f8"),
        ],
    )
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment synthetic fixture\n"
        f"element vertex {len(values)}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "property double nx\n"
        "property double ny\n"
        "property double nz\n"
        "end_header\n"
    ).encode("ascii")
    path.write_bytes(header + values.tobytes())
    return values


class PlyPointCloudTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = self.root / "map.ply"
        self.values = create_binary_ply(self.path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_binary_ply_is_memory_mapped_and_batched(self) -> None:
        from solution.radio_map.geometry import PlyPointCloud

        cloud = PlyPointCloud.open(self.path)
        batches = list(cloud.iter_batches(batch_size=4))

        self.assertIsInstance(cloud.vertices, np.memmap)
        self.assertEqual(cloud.vertex_count, 6)
        self.assertEqual([len(batch[0]) for batch in batches], [4, 2])
        np.testing.assert_allclose(batches[0][0][0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(batches[0][1][0], [0.0, 0.0, 1.0])

    def test_truncated_ply_is_rejected(self) -> None:
        from solution.radio_map.geometry import PlyPointCloud

        truncated = self.root / "truncated.ply"
        truncated.write_bytes(self.path.read_bytes()[:-8])

        with self.assertRaisesRegex(ValueError, "truncated"):
            PlyPointCloud.open(truncated)


class GeometryPriorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = self.root / "map.ply"
        create_binary_ply(self.path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_grid_aggregates_height_density_and_normals(self) -> None:
        from solution.radio_map.geometry import PlyPointCloud, build_geometry_prior

        prior = build_geometry_prior(
            PlyPointCloud.open(self.path),
            resolution=1.0,
            height_layers=4,
            batch_size=2,
        )
        index = {name: i for i, name in enumerate(prior.feature_names)}

        self.assertEqual(prior.features.shape, (13, 3, 3))
        self.assertEqual(prior.metadata["point_count"], 6)
        self.assertAlmostEqual(prior.features[index["occupancy"], 0, 0], 1.0)
        self.assertAlmostEqual(prior.features[index["z_min"], 0, 0], 0.0)
        self.assertAlmostEqual(prior.features[index["z_max"], 0, 0], 2.0)
        self.assertAlmostEqual(prior.features[index["height_range"], 0, 0], 2.0)
        self.assertAlmostEqual(prior.features[index["mean_normal_z"], 0, 0], 0.5)
        self.assertAlmostEqual(prior.features[index["wall_strength"], 0, 0], 0.5)

    def test_save_load_point_patch_and_corridor_share_coordinates(self) -> None:
        from solution.radio_map.geometry import (
            GeometryPrior,
            PlyPointCloud,
            build_geometry_prior,
        )

        prior = build_geometry_prior(
            PlyPointCloud.open(self.path), resolution=1.0, height_layers=4
        )
        cache = self.root / "cache" / "geometry.npz"
        prior.save(cache)
        loaded = GeometryPrior.load(cache)
        sampled = loaded.sample_points(np.array([[0.2, 0.2, 1.5]]))
        patch = loaded.extract_patch(np.array([0.2, 0.2, 1.5]), size=3)
        corridor = loaded.corridor_features(
            np.array([0.2, 0.2, 1.5]),
            np.array([2.2, 2.2, 1.5]),
            sample_count=3,
        )

        np.testing.assert_allclose(loaded.features, prior.features)
        np.testing.assert_allclose(patch[:, 1, 1], sampled[0])
        self.assertEqual(corridor.shape, (3, len(prior.feature_names) + 2))
        self.assertAlmostEqual(corridor[0, 0], 0.0)
        self.assertAlmostEqual(corridor[-1, 0], 1.0)
        self.assertAlmostEqual(corridor[-1, 1], np.sqrt(8.0), places=6)


if __name__ == "__main__":
    unittest.main()
