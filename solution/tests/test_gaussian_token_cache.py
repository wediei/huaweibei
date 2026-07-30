from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from solution.radio_map.learning.gaussian_geometry import GaussianScene
from solution.radio_map.learning.gaussian_token_cache import (
    GaussianTokenCache,
    GaussianTokenConfig,
    build_gaussian_path_tokens,
    save_gaussian_token_cache,
)


def _scene() -> GaussianScene:
    centers = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.1],
            [4.0, 0.0, 0.2],
            [2.0, 0.5, 2.0],
            [3.0, -0.5, 3.0],
            [4.0, 0.5, 4.0],
        ],
        dtype=np.float32,
    )
    normals = np.asarray(
        [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    return GaussianScene(
        centers,
        normals,
        np.full(6, 1.0, dtype=np.float32),
        np.full(6, 0.4, dtype=np.float32),
        np.full(6, 0.5, dtype=np.float32),
        np.arange(1, 7, dtype=np.int32),
        {"source_map_sha256": "a" * 64},
    )


class GaussianTokenTests(unittest.TestCase):
    def test_path_tokens_are_deterministic_and_ground_suppressed(self) -> None:
        scene = _scene()
        starts = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)
        ends = np.asarray([[5.0, 0.0, 1.0]], dtype=np.float32)
        config = GaussianTokenConfig(
            tokens_per_path=4,
            candidate_k=4,
            path_samples=5,
            elevated_quota=2,
            ground_height=0.5,
        )
        first = build_gaussian_path_tokens(scene, starts, ends, config)
        second = build_gaussian_path_tokens(scene, starts, ends, config)
        np.testing.assert_array_equal(first.tokens, second.tokens)
        np.testing.assert_array_equal(first.mask, second.mask)
        self.assertEqual(first.tokens.shape, (1, 4, len(first.feature_names)))
        surface_index = first.feature_names.index("surface_elevated")
        self.assertGreaterEqual(
            int((first.tokens[0, first.mask[0], surface_index] > 0.5).sum()),
            2,
        )
        fraction_index = first.feature_names.index("path_fraction")
        fractions = first.tokens[0, first.mask[0], fraction_index]
        self.assertTrue(np.all(fractions[:-1] <= fractions[1:]))
        self.assertTrue(np.isfinite(first.tokens).all())

    def test_cache_real_zero_shuffle_are_reproducible_and_authenticated(self) -> None:
        scene = _scene()
        config = GaussianTokenConfig(4, 4, 5, 2, 0.5)
        train = build_gaussian_path_tokens(
            scene,
            np.zeros((3, 3), dtype=np.float32),
            np.asarray([[2, 0, 1], [3, 0, 1], [4, 0, 1]], dtype=np.float32),
            config,
        )
        test = build_gaussian_path_tokens(
            scene,
            np.zeros((2, 3), dtype=np.float32),
            np.asarray([[2.5, 0, 1], [3.5, 0, 1]], dtype=np.float32),
            config,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tokens"
            save_gaussian_token_cache(
                path,
                train,
                test,
                fold_fingerprint="fold-a",
                map_sha256="a" * 64,
                train_fit_indices=np.asarray([0, 1], dtype=np.int64),
            )
            cache = GaussianTokenCache.load(path, "fold-a", "a" * 64)
            real, real_mask = cache.values("train", "real", seed=9)
            zero, zero_mask = cache.values("train", "zero", seed=9)
            shuffled_a, shuffled_mask_a = cache.values("train", "shuffle", seed=9)
            shuffled_b, shuffled_mask_b = cache.values("train", "shuffle", seed=9)
            np.testing.assert_array_equal(real_mask, zero_mask)
            np.testing.assert_array_equal(zero, np.zeros_like(zero))
            np.testing.assert_array_equal(shuffled_a, shuffled_b)
            np.testing.assert_array_equal(shuffled_mask_a, shuffled_mask_b)
            self.assertFalse(np.array_equal(real, shuffled_a))
            with self.assertRaisesRegex(ValueError, "fold"):
                GaussianTokenCache.load(path, "fold-b", "a" * 64)
            with self.assertRaisesRegex(ValueError, "map"):
                GaussianTokenCache.load(path, "fold-a", "b" * 64)
            del real, real_mask, zero, zero_mask
            del shuffled_a, shuffled_b, shuffled_mask_a, shuffled_mask_b
            cache.close()


if __name__ == "__main__":
    unittest.main()
