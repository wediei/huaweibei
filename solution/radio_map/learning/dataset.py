"""Torch dataset over a :mod:`cache` fold without copying raw channels."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data import RoundDataset
from ..anchors import AnchorMemory
from ..geometry import GeometryPrior
from .cache import FoldCacheManifest, _geometry_hash, quantize_geometry_values, validate_cache


def load_persisted_split(cache_dir: str | Path, *, validate: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Load the exact cache-bound fold; a seed is never used to recreate it."""

    cache = Path(cache_dir)
    if validate:
        manifest = FoldCacheManifest.load(cache / "manifest.json")
        validate_cache(manifest, cache)
    train = np.asarray(np.load(cache / "train_indices.npy", allow_pickle=False), dtype=np.int64)
    validation = np.asarray(np.load(cache / "validation_indices.npy", allow_pickle=False), dtype=np.int64)
    return train, validation


class CoordinateBatchContext:
    """Validated fold state reused across all arbitrary-coordinate batches."""

    def __init__(self, cache_dir: str | Path, manifest: FoldCacheManifest, dataset: RoundDataset,
                 train_indices: np.ndarray, *, use_geometry: bool, geometry: GeometryPrior | None = None) -> None:
        self.cache_dir, self.manifest, self.dataset = Path(cache_dir), manifest, dataset
        self.train_indices, self.use_geometry, self.geometry = np.asarray(train_indices, dtype=np.int64), bool(use_geometry), geometry
        if self.use_geometry:
            if geometry is None:
                raise ValueError("geometry is required when geometry features are enabled")
            if _geometry_hash(geometry) != manifest.geometry_sha256:
                raise ValueError("geometry hash does not match cache manifest")
        with np.load(self.cache_dir / "normalization.npz", allow_pickle=False) as values:
            self.stats = {name: np.asarray(values[name]) for name in values.files}
        self.latents = np.load(self.cache_dir / "latents.npy", mmap_mode="r", allow_pickle=False)
        # KNN needs only positions; retain a tiny dummy value instead of a full
        # fancy-indexed latent copy. Query-selected latent rows are read below.
        self.memory = AnchorMemory(dataset.train_pos[self.train_indices], np.zeros((len(self.train_indices), 1), np.complex64), np.asarray(dataset.config.bs_position), self.train_indices)

    @classmethod
    def open(cls, cache_dir: str | Path, *, use_geometry: bool, geometry: GeometryPrior | None = None) -> "CoordinateBatchContext":
        cache = Path(cache_dir)
        manifest = FoldCacheManifest.load(cache / "manifest.json")
        validate_cache(manifest, cache)
        dataset = RoundDataset.open(manifest.data_dir)
        train, _ = load_persisted_split(cache, validate=False)
        return cls(cache, manifest, dataset, train, use_geometry=use_geometry, geometry=geometry)

    def _normal(self, values: np.ndarray) -> np.ndarray:
        raw = np.asarray(values, dtype=np.float32)
        occupancy = self.stats["geometry_occupancy"].astype(bool)
        output = (raw - self.stats["geometry_mean"].reshape((-1,) + (1,) * (raw.ndim - 1))) / self.stats["geometry_std"].reshape((-1,) + (1,) * (raw.ndim - 1))
        output[occupancy] = raw[occupancy]
        return output

    @staticmethod
    def _tensor(value: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.from_numpy(np.asarray(value).copy()).to(dtype)

    def build(self, positions: np.ndarray) -> dict[str, torch.Tensor]:
        targets = np.asarray(positions, dtype=np.float64)
        if targets.ndim != 2 or targets.shape[1] != 3 or not len(targets) or not np.isfinite(targets).all():
            raise ValueError("positions must be a nonempty finite array shaped (samples, 3)")
        query = self.memory.query(targets, self.manifest.anchor_count)
        bs, positions32 = np.asarray(self.dataset.config.bs_position, dtype=np.float32), targets.astype(np.float32)
        pair = (query.pair_features - self.stats["pair_mean"]) / self.stats["pair_std"]
        patch_shape = tuple(self.manifest.shape["target_patches.npy"][1:])
        corridor_shape = tuple(self.manifest.shape["bs_target_corridors.npy"][1:])
        anchor_shape = (len(targets), self.manifest.anchor_count) + tuple(self.manifest.shape["anchor_corridors.npy"][2:])
        if self.use_geometry:
            assert self.geometry is not None
            point = quantize_geometry_values(self._normal(self.geometry.sample_points(targets).T).T)
            patches = np.stack([quantize_geometry_values(self._normal(self.geometry.extract_patch(position, self.manifest.patch))) for position in targets])
            bs_patch = quantize_geometry_values(self._normal(self.geometry.extract_patch(bs, self.manifest.patch)))
            corridors = np.stack([self.geometry.corridor_features(bs, position, self.manifest.corridor) for position in targets])
            corridors[:, :, 1] = (corridors[:, :, 1] - self.stats["corridor_length_mean"][0]) / self.stats["corridor_length_std"][0]
            corridors[:, :, 2:] = self._normal(corridors[:, :, 2:].transpose(2, 0, 1)).transpose(1, 2, 0)
            corridors = quantize_geometry_values(corridors)
            anchor_paths = np.empty(anchor_shape, dtype=np.float32)
            for row, target in enumerate(targets):
                for slot, source in enumerate(query.source_indices[row]):
                    path = self.geometry.corridor_features(self.dataset.train_pos[source], target, self.manifest.anchor_corridor)
                    path[:, 1] = (path[:, 1] - self.stats["corridor_length_mean"][0]) / self.stats["corridor_length_std"][0]
                    path[:, 2:] = self._normal(path[:, 2:].T).T
                    anchor_paths[row, slot] = quantize_geometry_values(path)
        else:
            point = np.zeros((len(targets), 13), np.float32); patches = np.zeros((len(targets),) + patch_shape, np.float32)
            bs_patch = np.zeros(patch_shape, np.float32); corridors = np.zeros((len(targets),) + corridor_shape, np.float32); anchor_paths = np.zeros(anchor_shape, np.float32)
        return {
            "anchor_latents": self._tensor(self.latents[query.source_indices], torch.complex64), "anchor_distances": self._tensor(query.distances, torch.float32),
            "anchor_mask": torch.ones((len(targets), self.manifest.anchor_count), dtype=torch.bool), "anchor_indices": self._tensor(query.source_indices, torch.long),
            "pair_features": self._tensor(pair, torch.float32), "target_positions": self._tensor(positions32, torch.float32),
            "target_positions_standardized": self._tensor((positions32 - self.stats["position_mean"]) / self.stats["position_std"], torch.float32),
            "target_bs_relative": self._tensor(positions32 - bs, torch.float32), "target_bs_relative_standardized": self._tensor((positions32 - bs - self.stats["bs_relative_mean"]) / self.stats["bs_relative_std"], torch.float32),
            "target_point_features": self._tensor(point, torch.float32), "target_patch": self._tensor(patches, torch.float32),
            "bs_patch": self._tensor(np.broadcast_to(bs_patch, (len(targets),) + bs_patch.shape), torch.float32), "bs_target_corridor": self._tensor(corridors, torch.float32),
            "bs_target_corridor_mask": torch.full((len(targets), corridor_shape[0]), self.use_geometry, dtype=torch.bool), "anchor_corridors": self._tensor(anchor_paths, torch.float32),
            "anchor_corridor_mask": torch.full(anchor_shape[:-1], self.use_geometry, dtype=torch.bool),
        }


def build_coordinate_batch(cache_dir: str | Path, positions: np.ndarray, *, use_geometry: bool, geometry: GeometryPrior | None = None) -> dict[str, torch.Tensor]:
    """Compatibility wrapper for one-off callers; CLI inference reuses a context."""

    return CoordinateBatchContext.open(cache_dir, use_geometry=use_geometry, geometry=geometry).build(positions)


class CachedAnchorDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        cache_dir: str | Path,
        target_indices: Sequence[int] | np.ndarray,
        training: bool,
        use_geometry: bool,
        *,
        allow_cache_code_mismatch: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.manifest = FoldCacheManifest.load(self.cache_dir / "manifest.json")
        validate_cache(
            self.manifest,
            self.cache_dir,
            allow_code_mismatch=allow_cache_code_mismatch,
        )
        self.dataset = RoundDataset.open(self.manifest.data_dir)
        self.indices = np.asarray(target_indices, dtype=np.int64)
        if self.indices.ndim != 1 or np.any(self.indices < 0) or np.any(self.indices >= self.manifest.shape["latents.npy"][0]):
            raise IndexError("target_indices outside cache")
        self.training, self.use_geometry, self.epoch = bool(training), bool(use_geometry), 0
        self.latents = np.load(self.cache_dir / "latents.npy", mmap_mode="r")
        self.neighbor_indices = np.load(self.cache_dir / "neighbor_indices.npy", mmap_mode="r")
        self.neighbor_distances = np.load(self.cache_dir / "neighbor_distances.npy", mmap_mode="r")
        self.pair_features = np.load(self.cache_dir / "pair_features.npy", mmap_mode="r")
        self.point = np.load(self.cache_dir / "target_point_features.npy", mmap_mode="r")
        self.patches = np.load(self.cache_dir / "target_patches.npy", mmap_mode="r")
        self.bs_patch = np.load(self.cache_dir / "bs_patch.npy", mmap_mode="r")
        self.corridors = np.load(self.cache_dir / "bs_target_corridors.npy", mmap_mode="r")
        self.anchor_corridors = np.load(self.cache_dir / "anchor_corridors.npy", mmap_mode="r")
        with np.load(self.cache_dir / "normalization.npz", allow_pickle=False) as stats:
            self.position_mean, self.position_std = stats["position_mean"], stats["position_std"]
            self.relative_mean, self.relative_std = stats["bs_relative_mean"], stats["bs_relative_std"]

    def __len__(self) -> int:
        return len(self.indices)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _slots(self, source_index: int) -> np.ndarray:
        count = self.manifest.anchor_count
        slots = np.arange(count, dtype=np.int64)
        if not self.training or self.manifest.dropout == 0:
            return slots
        rng = np.random.default_rng(np.random.SeedSequence([self.manifest.seed, self.epoch, source_index]))
        retained = slots[rng.random(count) >= self.manifest.dropout]
        min_keep = min(self.manifest.anchor_count, self.manifest.min_anchors)
        if len(retained) < min_keep:
            retained = np.sort(rng.choice(slots, size=min_keep, replace=False))
        return retained

    @staticmethod
    def _tensor(value: np.ndarray, dtype: torch.dtype | None = None) -> torch.Tensor:
        result = torch.from_numpy(np.array(value, copy=True))
        return result if dtype is None else result.to(dtype)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        source = int(self.indices[item])
        slots = self._slots(source)
        anchors = np.asarray(self.neighbor_indices[source, slots], dtype=np.int64)
        if source in anchors:
            raise RuntimeError("cache violates target exclusion")
        position = np.asarray(self.dataset.train_pos[source], dtype=np.float32)
        relative = position - np.asarray(self.dataset.config.bs_position, dtype=np.float32)
        patch_shape = tuple(self.manifest.shape["target_patches.npy"][1:])
        corridor_shape = tuple(self.manifest.shape["bs_target_corridors.npy"][1:])
        anchor_corridor_shape = (len(slots),) + tuple(self.manifest.shape["anchor_corridors.npy"][2:])
        if self.use_geometry:
            point, patch, bs_patch = self.point[source], self.patches[source], self.bs_patch
            corridor, anchor_corridor = self.corridors[source], self.anchor_corridors[source, slots]
        else:
            point = np.zeros(13, np.float32); patch = np.zeros(patch_shape, np.float16); bs_patch = np.zeros(patch_shape, np.float16)
            corridor = np.zeros(corridor_shape, np.float16); anchor_corridor = np.zeros(anchor_corridor_shape, np.float16)
        return {
            "anchor_latents": self._tensor(self.latents[anchors], torch.complex64),
            "anchor_distances": self._tensor(self.neighbor_distances[source, slots], torch.float32),
            "anchor_mask": torch.ones(len(slots), dtype=torch.bool),
            "anchor_indices": self._tensor(anchors, torch.long),
            "pair_features": self._tensor(self.pair_features[source, slots], torch.float32),
            "target_positions": self._tensor(position, torch.float32),
            "target_positions_standardized": self._tensor((position - self.position_mean) / self.position_std, torch.float32),
            "target_bs_relative": self._tensor(relative, torch.float32),
            "target_bs_relative_standardized": self._tensor((relative - self.relative_mean) / self.relative_std, torch.float32),
            "target_point_features": self._tensor(point, torch.float32), "target_patch": self._tensor(patch, torch.float32), "bs_patch": self._tensor(bs_patch, torch.float32),
            "bs_target_corridor": self._tensor(corridor, torch.float32), "bs_target_corridor_mask": torch.full((corridor_shape[0],), self.use_geometry, dtype=torch.bool),
            "anchor_corridors": self._tensor(anchor_corridor, torch.float32), "anchor_corridor_mask": torch.full(anchor_corridor_shape[:-1], self.use_geometry, dtype=torch.bool),
            "target_latent": self._tensor(self.latents[source], torch.complex64),
            "target_channel": self._tensor(self.dataset.channel_batch([source])[0], torch.complex64),
            "source_index": torch.tensor(source, dtype=torch.long),
        }


def collate_anchor_batch(samples: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    max_k = max(sample["anchor_indices"].numel() for sample in samples)
    anchor_names = {"anchor_latents", "anchor_distances", "anchor_mask", "anchor_indices", "pair_features", "anchor_corridors", "anchor_corridor_mask"}
    output: dict[str, torch.Tensor] = {}
    for name in samples[0]:
        if name not in anchor_names:
            output[name] = torch.stack([sample[name] for sample in samples])
            continue
        padded = []
        for sample in samples:
            value, k = sample[name], sample["anchor_indices"].numel()
            if k == max_k:
                padded.append(value); continue
            fill = -1 if name == "anchor_indices" else float("inf") if name == "anchor_distances" else False if value.dtype == torch.bool else 0
            padded_value = torch.full((max_k,) + tuple(value.shape[1:]), fill, dtype=value.dtype)
            padded_value[:k] = value
            padded.append(padded_value)
        output[name] = torch.stack(padded)
    return output
