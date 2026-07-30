from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from solution.radio_map.learning.gaussian_geometry import (
    GaussianScene,
    GaussianSplatConfig,
)


class GaussianSceneTests(unittest.TestCase):
    def _scene(self) -> GaussianScene:
        return GaussianScene(
            centers=np.array([[0, 0, 0], [2, 0, 0]], np.float32),
            normals=np.array([[0, 0, 1], [0, 0, 1]], np.float32),
            tangent_scales=np.array([0.8, 0.8], np.float32),
            normal_scales=np.array([0.1, 0.1], np.float32),
            opacities=np.array([0.8, 0.8], np.float32),
            counts=np.array([4, 4], np.int32),
            metadata={"backend": "test"},
        )

    def test_anisotropic_kernel_uses_surface_normal(self) -> None:
        scene = self._scene()
        _, xy_index = scene._query(
            np.array([[0.2, 0.0, 0.0]], np.float32), 1
        )
        _, z_index = scene._query(
            np.array([[0.0, 0.0, 0.2]], np.float32), 1
        )
        xy = scene._kernel_response(
            np.array([[0.2, 0.0, 0.0]], np.float32), xy_index
        )
        z = scene._kernel_response(
            np.array([[0.0, 0.0, 0.2]], np.float32), z_index
        )
        self.assertGreater(float(xy[0, 0]), float(z[0, 0]))

    def test_local_and_path_splat_contract(self) -> None:
        scene = self._scene()
        queries = np.array(
            [[0.1, 0.0, 0.0], [1.9, 0.0, 0.0]], np.float32
        )
        local = scene.local_features(queries, k=2)
        profile = scene.path_splat_profiles(
            np.zeros_like(queries),
            queries,
            GaussianSplatConfig(
                local_k=2,
                path_k=2,
                path_samples=16,
                path_batch_size=2,
            ),
        )
        self.assertEqual(local.shape, (2, 16))
        self.assertEqual(profile.shape, (2, 16))
        self.assertTrue(np.isfinite(local).all())
        self.assertTrue(((profile >= 0) & (profile <= 1)).all())

    def test_save_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.npz"
            expected = self._scene()
            expected.save(path)
            actual = GaussianScene.load(path)
            np.testing.assert_array_equal(actual.centers, expected.centers)
            np.testing.assert_array_equal(
                actual.inverse_covariances,
                expected.inverse_covariances,
            )
            self.assertEqual(actual.metadata, expected.metadata)


if __name__ == "__main__":
    unittest.main()
