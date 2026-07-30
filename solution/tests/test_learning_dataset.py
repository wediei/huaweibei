from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from solution.tests.fixtures import create_round_dir


class CachedAnchorDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        from solution.radio_map.data import RoundDataset
        from solution.radio_map.geometry import PlyPointCloud, build_geometry_prior
        from solution.radio_map.learning.cache import FoldCacheConfig, prepare_fold_cache
        from solution.radio_map.splits import SplitIndices

        self.temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        data_dir = create_round_dir(Path(self.temporary.name))
        self.dataset = RoundDataset.open(data_dir)
        geometry = build_geometry_prior(PlyPointCloud.open(self.dataset.map_path), height_layers=4)
        self.split = SplitIndices(np.array([0, 1, 2, 3]), np.array([4, 5]))
        config = FoldCacheConfig(k_max=3, anchor_count=3, patch=3, corridor=4, anchor_corridor=3, encode_batch_size=2, min_anchors=2, dropout=0.75)
        self.cache = prepare_fold_cache(self.dataset, geometry, self.split, config, Path(self.temporary.name) / "cache")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_dataset_never_returns_target_as_anchor(self) -> None:
        from solution.radio_map.learning.dataset import CachedAnchorDataset

        sample = CachedAnchorDataset(self.cache, self.split.validation, training=False, use_geometry=True)[0]
        self.assertNotIn(sample["source_index"], sample["anchor_indices"].tolist())

    def test_validation_is_deterministic_and_geometry_off_is_zero(self) -> None:
        from solution.radio_map.learning.dataset import CachedAnchorDataset

        dataset = CachedAnchorDataset(self.cache, self.split.validation, training=False, use_geometry=False)
        first, second = dataset[0], dataset[0]
        self.assertTrue(torch.equal(first["anchor_indices"], second["anchor_indices"]))
        self.assertTrue(torch.equal(first["target_patch"], torch.zeros_like(first["target_patch"])))
        self.assertFalse(first["bs_target_corridor_mask"].any())
        self.assertFalse(first["anchor_corridor_mask"].any())

    def test_collate_pads_anchor_axis(self) -> None:
        from solution.radio_map.learning.dataset import CachedAnchorDataset, collate_anchor_batch

        dataset = CachedAnchorDataset(self.cache, self.split.train[:2], training=True, use_geometry=True)
        dataset.set_epoch(1)
        first, second = dataset[0], dataset[1]
        # Exercise true variable-K padding independently of a particular RNG draw.
        second = dict(second)
        for name in ("anchor_latents", "anchor_distances", "anchor_mask", "anchor_indices", "pair_features", "anchor_corridors", "anchor_corridor_mask"):
            second[name] = second[name][:1]
        batch = collate_anchor_batch([first, second])
        self.assertEqual(batch["anchor_indices"].shape[0], 2)
        self.assertEqual(batch["anchor_latents"].dtype, torch.complex64)
        self.assertEqual(batch["anchor_mask"].dtype, torch.bool)
        self.assertTrue(torch.equal(batch["anchor_indices"][1, 1:], torch.full_like(batch["anchor_indices"][1, 1:], -1)))
        self.assertTrue(torch.isinf(batch["anchor_distances"][1, 1:]).all())
        self.assertFalse(batch["anchor_mask"][1, 1:].any())

    def test_dropout_keeps_minimum_and_raw_channel_is_per_item(self) -> None:
        from solution.radio_map.learning.dataset import CachedAnchorDataset

        dataset = CachedAnchorDataset(self.cache, self.split.train, training=True, use_geometry=True)
        calls: list[np.ndarray] = []
        original = dataset.dataset.channel_batch
        def counted(indices: object) -> np.ndarray:
            calls.append(np.asarray(indices))
            return original(indices)  # type: ignore[arg-type]
        object.__setattr__(dataset.dataset, "channel_batch", counted)
        dataset.set_epoch(4)
        first = dataset[0]
        dataset.set_epoch(4)
        second = dataset[0]
        self.assertGreaterEqual(first["anchor_indices"].numel(), 2)
        self.assertTrue(torch.equal(first["anchor_indices"], second["anchor_indices"]))
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call.shape == (1,) for call in calls))
        self.assertFalse(any("Channel" in path.name for path in self.cache.iterdir()))

    def test_batch_matches_task4_contract(self) -> None:
        from solution.radio_map.learning.dataset import CachedAnchorDataset, collate_anchor_batch
        from solution.radio_map.learning.anchor_mixer import SupportAwareAnchorMixer, SupportAwareAnchorMixerConfig
        from solution.radio_map.learning.latent_adapter import FixedSupportLatentAdapter
        from solution.radio_map.transforms import AntennaLayout

        dataset = CachedAnchorDataset(self.cache, self.split.validation, training=False, use_geometry=True)
        batch = collate_anchor_batch([dataset[0], dataset[1]])
        adapter = FixedSupportLatentAdapter.load(self.cache / "adapter.npz", AntennaLayout(dataset.dataset.config, tuple(dataset.manifest.layout_order)))
        model = SupportAwareAnchorMixer(SupportAwareAnchorMixerConfig(latent_size=adapter.coefficient_count, d_model=16, group_dim=8, geometry_dim=8, low_rank=2, num_frequencies=2, use_geometry=True), torch.from_numpy(adapter.group_ids))
        self.assertEqual(model(batch).latent.shape, batch["target_latent"].shape)

    def test_dropout_rng_uses_manifest_seed(self) -> None:
        from solution.radio_map.learning.cache import FoldCacheConfig, prepare_fold_cache
        from solution.radio_map.learning.dataset import CachedAnchorDataset
        from solution.radio_map.geometry import PlyPointCloud, build_geometry_prior

        geometry = build_geometry_prior(PlyPointCloud.open(self.dataset.map_path), height_layers=4)
        changed_cache = prepare_fold_cache(self.dataset, geometry, self.split, FoldCacheConfig(k_max=3, anchor_count=3, patch=3, corridor=4, anchor_corridor=3, encode_batch_size=2, min_anchors=2, dropout=0.75, seed=777), Path(self.temporary.name) / "seed_777")
        original = CachedAnchorDataset(self.cache, self.split.train, training=True, use_geometry=False)
        changed = CachedAnchorDataset(changed_cache, self.split.train, training=True, use_geometry=False)
        original.set_epoch(3)
        changed.set_epoch(3)
        self.assertEqual(original.manifest.seed, 42)
        self.assertEqual(changed.manifest.seed, 777)
        self.assertTrue(any(not torch.equal(original[index]["anchor_indices"], changed[index]["anchor_indices"]) for index in range(len(original))))


if __name__ == "__main__":
    unittest.main()
