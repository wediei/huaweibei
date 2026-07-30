import tempfile
import unittest
from pathlib import Path

from solution.radio_map.geometry import PlyPointCloud, build_geometry_prior
from solution.radio_map.learning.cli import main as learning_main
from solution.radio_map.learning.gaussian_anchor_path_cache import (
    GaussianAnchorPathCache,
)
from solution.radio_map.learning.gaussian_anchor_path_cli import (
    main as anchor_path_main,
)
from solution.radio_map.learning.gaussian_geometry import _sha256_file
from solution.radio_map.learning.gaussian_token_cli import main as token_main
from solution.radio_map.learning.cache import FoldCacheManifest
from solution.tests.fixtures import create_round_dir


class GaussianAnchorPathCacheTests(unittest.TestCase):
    def test_prepare_binds_direct_paths_to_fold_neighbors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = create_round_dir(root)
            geometry = root / "geometry.npz"
            build_geometry_prior(
                PlyPointCloud.open(data / "Round1_Map.ply"),
                height_layers=4,
                batch_size=1,
            ).save(geometry)
            fold = root / "fold"
            self.assertEqual(
                learning_main(
                    [
                        "prepare-cache",
                        "--data-dir", str(data),
                        "--geometry-cache", str(geometry),
                        "--output-dir", str(fold),
                        "--validation-fraction", "0.34",
                        "--grid-size", "2.0",
                        "--k-max", "3",
                        "--anchor-count", "2",
                        "--patch", "3",
                        "--corridor", "2",
                        "--anchor-corridor", "2",
                        "--min-anchors", "1",
                        "--channel-batch-size", "1",
                    ]
                ),
                0,
            )
            scene = root / "scene.npz"
            bs_tokens = root / "bs_tokens"
            self.assertEqual(
                token_main(
                    [
                        "prepare",
                        "--data-dir", str(data),
                        "--cache-dir", str(fold),
                        "--scene-cache", str(scene),
                        "--output-dir", str(bs_tokens),
                        "--tokens-per-path", "4",
                        "--candidate-k", "4",
                        "--path-samples", "5",
                        "--elevated-quota", "2",
                    ]
                ),
                0,
            )
            direct = root / "direct"
            self.assertEqual(
                anchor_path_main(
                    [
                        "prepare",
                        "--data-dir", str(data),
                        "--cache-dir", str(fold),
                        "--scene-cache", str(scene),
                        "--output-dir", str(direct),
                        "--anchor-count", "2",
                        "--tokens-per-path", "4",
                        "--candidate-k", "4",
                        "--path-samples", "5",
                        "--elevated-quota", "2",
                    ]
                ),
                0,
            )
            manifest = FoldCacheManifest.load(fold / "manifest.json")
            cache = GaussianAnchorPathCache.load(
                direct,
                fold_fingerprint=manifest.fingerprint,
                map_sha256=_sha256_file(data / "Round1_Map.ply"),
                anchor_count=2,
                anchor_source="all_official_train",
            )
            train_tokens, train_mask, train_indices, _ = cache.values("train")
            test_tokens, test_mask, test_indices, _ = cache.values("test")
            self.assertEqual(train_tokens.shape, (6, 2, 4, 27))
            self.assertEqual(test_tokens.shape, (2, 2, 4, 27))
            self.assertEqual(train_indices.shape, (6, 2))
            self.assertEqual(test_indices.shape, (2, 2))
            self.assertTrue(train_mask.any())
            self.assertTrue(test_mask.any())
            cache.close()


if __name__ == "__main__":
    unittest.main()
