from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

import numpy as np

from solution.tests.fixtures import create_round_dir


class FoldCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        from solution.radio_map.data import RoundDataset
        from solution.radio_map.geometry import PlyPointCloud, build_geometry_prior
        from solution.radio_map.learning.cache import FoldCacheConfig, prepare_fold_cache
        from solution.radio_map.splits import SplitIndices

        self.temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.data_dir = create_round_dir(Path(self.temporary.name))
        self.dataset = RoundDataset.open(self.data_dir)
        self.geometry = build_geometry_prior(PlyPointCloud.open(self.dataset.map_path), height_layers=4)
        self.split = SplitIndices(np.array([0, 1, 2, 3]), np.array([4, 5]))
        self.config = FoldCacheConfig(k_max=3, anchor_count=3, patch=3, corridor=4, anchor_corridor=3, encode_batch_size=2, min_anchors=2)
        self.cache = prepare_fold_cache(self.dataset, self.geometry, self.split, self.config, Path(self.temporary.name) / "cache")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_manifest_rejects_changed_support_hash(self) -> None:
        from solution.radio_map.learning.cache import FoldCacheManifest, validate_cache

        manifest = FoldCacheManifest.load(self.cache / "manifest.json")
        altered = dataclasses.replace(manifest, support_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_cache(altered, self.cache)

    def test_required_files_dtypes_and_shapes(self) -> None:
        expected = {
            "latents.npy": (np.complex64, (6, 7)),
            "neighbor_indices.npy": (np.int64, (6, 3)),
            "neighbor_distances.npy": (np.float32, (6, 3)),
            "pair_features.npy": (np.float32, (6, 3, 14)),
            "target_point_features.npy": (np.float32, (6, 13)),
            "target_patches.npy": (np.float16, (6, 13, 3, 3)),
            "bs_patch.npy": (np.float16, (13, 3, 3)),
            "bs_target_corridors.npy": (np.float16, (6, 4, 15)),
            "anchor_corridors.npy": (np.float16, (6, 3, 3, 15)),
        }
        for name, (dtype, shape) in expected.items():
            array = np.load(self.cache / name, mmap_mode="r")
            self.assertEqual(array.dtype, dtype)
            self.assertEqual(array.shape, shape)

    def test_adapter_fit_and_statistics_are_train_only(self) -> None:
        from solution.radio_map.learning.latent_adapter import FixedSupportLatentAdapter
        from solution.radio_map.transforms import AntennaLayout

        adapter = FixedSupportLatentAdapter.load(self.cache / "adapter.npz", AntennaLayout(self.dataset.config, self.config.layout_order))
        np.testing.assert_array_equal(adapter.fitted_indices, self.split.train)
        with np.load(self.cache / "normalization.npz", allow_pickle=False) as values:
            cached_position_mean = values["position_mean"].copy()
        np.testing.assert_allclose(cached_position_mean, self.dataset.train_pos[self.split.train].mean(axis=0))

    def test_manifest_rejects_changed_source_hash(self) -> None:
        from solution.radio_map.learning.cache import FoldCacheManifest, validate_cache

        manifest = FoldCacheManifest.load(self.cache / "manifest.json")
        changed = dict(manifest.source_file_sha256)
        changed["Round1_Train_Pos.npy"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_cache(dataclasses.replace(manifest, source_file_sha256=changed), self.cache)

    def test_validation_only_channel_position_and_geometry_changes_do_not_fit_fold_state(self) -> None:
        """Only train rows may determine adapter support or any normalization array."""
        from solution.radio_map.data import RoundDataset
        from solution.radio_map.geometry import GeometryPrior
        from solution.radio_map.learning.cache import prepare_fold_cache

        other_dir = create_round_dir(Path(self.temporary.name) / "other")
        positions = np.load(other_dir / "Round1_Train_Pos.npy")
        channels = np.load(other_dir / "Round1_Train_Channel.npy")
        positions[self.split.validation] += 100.0
        channels[self.split.validation] *= np.complex64(17 + 3j)
        np.save(other_dir / "Round1_Train_Pos.npy", positions)
        np.save(other_dir / "Round1_Train_Channel.npy", channels)
        names = self.geometry.feature_names
        values = np.arange(13 * 3 * 3, dtype=np.float32).reshape(13, 3, 3)
        clean_geometry = GeometryPrior(values, names, (0.0, 0.0), 1.0, {})
        changed_values = values.copy()
        changed_values[:, :2, 2] += 10000.0  # cells belonging only to validation positions
        changed_geometry = GeometryPrior(changed_values, names, (0.0, 0.0), 1.0, {})
        clean = prepare_fold_cache(self.dataset, clean_geometry, self.split, self.config, Path(self.temporary.name) / "clean_fold")
        changed = prepare_fold_cache(RoundDataset.open(other_dir), changed_geometry, self.split, self.config, Path(self.temporary.name) / "changed_fold")
        with np.load(clean / "normalization.npz", allow_pickle=False) as left, np.load(changed / "normalization.npz", allow_pickle=False) as right:
            self.assertEqual(left.files, right.files)
            for name in left.files:
                np.testing.assert_array_equal(left[name], right[name])
        with np.load(clean / "adapter.npz", allow_pickle=False) as left, np.load(changed / "adapter.npz", allow_pickle=False) as right:
            np.testing.assert_array_equal(left["support_indices"], right["support_indices"])
            np.testing.assert_array_equal(left["mean"], right["mean"])
            np.testing.assert_array_equal(left["rms"], right["rms"])

    def test_every_target_is_excluded_from_train_anchor_pool(self) -> None:
        neighbors = np.load(self.cache / "neighbor_indices.npy", mmap_mode="r")
        for source_index, row in enumerate(neighbors):
            self.assertNotIn(source_index, row.tolist())

    def test_noncontiguous_train_split_uses_global_neighbor_distances_for_corridor_stats(self) -> None:
        from solution.radio_map.learning.cache import prepare_fold_cache
        from solution.radio_map.splits import SplitIndices

        split = SplitIndices(np.array([0, 2, 4, 5]), np.array([1, 3]))
        cache = prepare_fold_cache(self.dataset, self.geometry, split, self.config, Path(self.temporary.name) / "noncontiguous")
        neighbors = np.load(cache / "neighbor_distances.npy", mmap_mode="r")
        bs = np.asarray(self.dataset.config.bs_position)
        expected = np.concatenate((
            np.linalg.norm(self.dataset.train_pos[split.train] - bs, axis=1),
            np.asarray(neighbors[split.train]).reshape(-1),
        ))
        with np.load(cache / "normalization.npz", allow_pickle=False) as stats:
            self.assertAlmostEqual(float(stats["corridor_length_mean"][0]), float(expected.mean()), places=6)
            self.assertAlmostEqual(float(stats["corridor_length_std"][0]), float(expected.std().clip(min=1e-6)), places=6)
        adapter = np.load(cache / "adapter.npz", allow_pickle=False)
        try:
            np.testing.assert_array_equal(adapter["fitted_indices"], split.train)
        finally:
            adapter.close()

    def test_seed_is_manifested_and_reusing_directory_with_new_seed_is_rejected(self) -> None:
        from solution.radio_map.learning.cache import FoldCacheManifest, prepare_fold_cache

        manifest = FoldCacheManifest.load(self.cache / "manifest.json")
        self.assertEqual(manifest.seed, self.config.seed)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            prepare_fold_cache(self.dataset, self.geometry, self.split, dataclasses.replace(self.config, seed=self.config.seed + 1), self.cache)

    def test_code_fingerprint_is_combined_and_current_code_is_checked(self) -> None:
        from solution.radio_map.learning.cache import FoldCacheManifest, _sha256_file, validate_cache

        manifest = FoldCacheManifest.load(self.cache / "manifest.json")
        self.assertNotEqual(manifest.code_fingerprint, _sha256_file(Path(__file__).parents[1] / "radio_map" / "learning" / "cache.py"))
        fields = manifest.to_dict()
        fields.pop("fingerprint")
        fields["code_fingerprint"] = "0" * 64
        altered = FoldCacheManifest.create(**fields)
        with self.assertRaisesRegex(ValueError, "code|fingerprint"):
            validate_cache(altered, self.cache)

    def test_artifact_payload_tampering_is_rejected_before_shape_checks(self) -> None:
        from solution.radio_map.learning.cache import FoldCacheManifest, validate_cache

        target = self.cache / "pair_features.npy"
        with target.open("r+b") as handle:
            handle.seek(-1, 2)
            last = handle.read(1)
            handle.seek(-1, 2)
            handle.write(bytes([last[0] ^ 1]))
        manifest = FoldCacheManifest.load(self.cache / "manifest.json")
        with self.assertRaisesRegex(ValueError, "fingerprint|artifact|hash"):
            validate_cache(manifest, self.cache)


if __name__ == "__main__":
    unittest.main()
