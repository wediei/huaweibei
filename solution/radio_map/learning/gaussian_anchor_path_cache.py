"""Authenticated direct Anchor-to-Target Gaussian path cache."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ..data import RoundDataset
from .cache import FoldCacheManifest, validate_cache
from .gaussian_geometry import GaussianScene, _sha256_file
from .gaussian_token_cache import (
    FEATURE_NAMES,
    GaussianTokenConfig,
    build_gaussian_path_tokens,
)


FILES = (
    "train_tokens.npy",
    "train_mask.npy",
    "train_neighbor_indices.npy",
    "train_neighbor_distances.npy",
    "test_tokens.npy",
    "test_mask.npy",
    "test_neighbor_indices.npy",
    "test_neighbor_distances.npy",
    "normalization_mean.npy",
    "normalization_std.npy",
)


def _indices_hash(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype="<i8").tobytes()
    ).hexdigest()


class GaussianAnchorPathCache:
    def __init__(self, path: Path, manifest: dict) -> None:
        self.path = path
        self.manifest = manifest
        self.anchor_count = int(manifest["anchor_count"])
        self.anchor_source = str(manifest["anchor_source"])
        self.feature_names = tuple(manifest["feature_names"])
        self.fingerprint = hashlib.sha256(
            json.dumps(
                manifest, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        self._arrays = {
            name.removesuffix(".npy"): np.load(
                path / name, mmap_mode="r", allow_pickle=False
            )
            for name in FILES
        }

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        fold_fingerprint: str,
        map_sha256: str,
        anchor_count: int,
        anchor_source: str,
    ) -> "GaussianAnchorPathCache":
        source = Path(path)
        try:
            manifest = json.loads(
                (source / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("invalid direct Gaussian path cache") from error
        if (
            manifest.get("format_version") != 1
            or manifest.get("kind") != "gaussian_anchor_target_path_cache"
            or manifest.get("fold_fingerprint") != fold_fingerprint
            or manifest.get("map_sha256") != map_sha256
            or int(manifest.get("anchor_count", -1)) != int(anchor_count)
            or manifest.get("anchor_source") != anchor_source
            or tuple(manifest.get("feature_names", ())) != FEATURE_NAMES
            or set(manifest.get("sha256", {})) != set(FILES)
            or set(manifest.get("shape", {})) != set(FILES)
            or set(manifest.get("dtype", {})) != set(FILES)
        ):
            raise ValueError("direct Gaussian path cache identity differs")
        for name in FILES:
            file_path = source / name
            if (
                not file_path.is_file()
                or _sha256_file(file_path) != manifest["sha256"][name]
            ):
                raise ValueError(f"direct Gaussian path hash differs: {name}")
            values = np.load(file_path, mmap_mode="r", allow_pickle=False)
            if (
                list(values.shape) != manifest["shape"][name]
                or str(values.dtype) != manifest["dtype"][name]
            ):
                raise ValueError(f"direct Gaussian path schema differs: {name}")
        return cls(source, manifest)

    def values(self, split: str):
        if split not in ("train", "test"):
            raise ValueError("split must be train or test")
        return (
            self._arrays[f"{split}_tokens"],
            self._arrays[f"{split}_mask"],
            self._arrays[f"{split}_neighbor_indices"],
            self._arrays[f"{split}_neighbor_distances"],
        )

    def normalization(self) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of the feature normalization used by this cache."""

        return (
            np.asarray(self._arrays["normalization_mean"]).copy(),
            np.asarray(self._arrays["normalization_std"]).copy(),
        )

    def close(self) -> None:
        for values in self._arrays.values():
            mapping = getattr(values, "_mmap", None)
            if mapping is not None:
                mapping.close()
        self._arrays.clear()


def _neighbors(
    dataset: RoundDataset,
    fold_path: Path,
    anchor_count: int,
    anchor_source: str,
):
    train_indices = np.asarray(
        np.load(fold_path / "neighbor_indices.npy", allow_pickle=False),
        dtype=np.int64,
    )[:, :anchor_count]
    train_distances = np.asarray(
        np.load(fold_path / "neighbor_distances.npy", allow_pickle=False),
        dtype=np.float32,
    )[:, :anchor_count]
    if anchor_source == "all_official_train":
        pool = np.arange(len(dataset.train_pos), dtype=np.int64)
    elif anchor_source == "fold":
        pool = np.asarray(
            np.load(fold_path / "train_indices.npy", allow_pickle=False),
            dtype=np.int64,
        )
    else:
        raise ValueError("anchor source must be fold or all_official_train")
    distance, local = cKDTree(dataset.train_pos[pool]).query(
        dataset.test_pos, k=min(anchor_count, len(pool))
    )
    local = np.asarray(local, dtype=np.int64)
    distance = np.asarray(distance, dtype=np.float32)
    if local.ndim == 1:
        local = local[:, None]
        distance = distance[:, None]
    return train_indices, train_distances, pool[local], distance


def _paths(
    scene: GaussianScene,
    source_positions: np.ndarray,
    target_positions: np.ndarray,
    neighbor_indices: np.ndarray,
    config: GaussianTokenConfig,
):
    count, anchors = neighbor_indices.shape
    starts = source_positions[neighbor_indices.reshape(-1)]
    ends = np.repeat(target_positions, anchors, axis=0)
    paths = build_gaussian_path_tokens(scene, starts, ends, config)
    return (
        paths.tokens.reshape(count, anchors, config.tokens_per_path, -1),
        paths.mask.reshape(count, anchors, config.tokens_per_path),
    )


def build_gaussian_anchor_path_cache(
    *,
    data_dir: str | Path,
    fold_cache: str | Path,
    scene_cache: str | Path,
    output_dir: str | Path,
    anchor_count: int,
    anchor_source: str,
    token_config: GaussianTokenConfig,
) -> dict:
    fold_path = Path(fold_cache)
    manifest = FoldCacheManifest.load(fold_path / "manifest.json")
    validate_cache(manifest, fold_path, allow_code_mismatch=True)
    dataset = RoundDataset.open(data_dir)
    if dataset.data_dir != Path(manifest.data_dir).resolve():
        raise ValueError("data directory differs from fold cache")
    if anchor_count < 1 or anchor_count > manifest.anchor_count:
        raise ValueError("anchor count outside persisted fold cache")
    scene = GaussianScene.load(scene_cache)
    map_sha256 = _sha256_file(dataset.map_path)
    if scene.metadata.get("source_map_sha256") != map_sha256:
        raise ValueError("Gaussian scene was built from another official map")
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("refusing to overwrite direct Gaussian path cache")
    destination.mkdir(parents=True, exist_ok=True)
    train_i, train_d, test_i, test_d = _neighbors(
        dataset, fold_path, anchor_count, anchor_source
    )
    train_tokens, train_mask = _paths(
        scene,
        np.asarray(dataset.train_pos, dtype=np.float32),
        np.asarray(dataset.train_pos, dtype=np.float32),
        train_i,
        token_config,
    )
    test_tokens, test_mask = _paths(
        scene,
        np.asarray(dataset.train_pos, dtype=np.float32),
        np.asarray(dataset.test_pos, dtype=np.float32),
        test_i,
        token_config,
    )
    fit_indices = np.asarray(
        np.load(fold_path / "train_indices.npy", allow_pickle=False),
        dtype=np.int64,
    )
    fit_values = train_tokens[fit_indices][train_mask[fit_indices]]
    mean = fit_values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = fit_values.std(axis=0, dtype=np.float64).clip(1e-6).astype(np.float32)
    for index, name in enumerate(FEATURE_NAMES):
        if name.startswith("surface_") or name == "map_present":
            mean[index], std[index] = 0.0, 1.0
    normalized_train = np.zeros_like(train_tokens)
    normalized_test = np.zeros_like(test_tokens)
    normalized_train[train_mask] = (train_tokens[train_mask] - mean) / std
    normalized_test[test_mask] = (test_tokens[test_mask] - mean) / std
    arrays = {
        "train_tokens.npy": normalized_train,
        "train_mask.npy": train_mask,
        "train_neighbor_indices.npy": train_i,
        "train_neighbor_distances.npy": train_d,
        "test_tokens.npy": normalized_test,
        "test_mask.npy": test_mask,
        "test_neighbor_indices.npy": test_i,
        "test_neighbor_distances.npy": test_d,
        "normalization_mean.npy": mean,
        "normalization_std.npy": std,
    }
    for name, values in arrays.items():
        np.save(destination / name, values, allow_pickle=False)
    report = {
        "format_version": 1,
        "kind": "gaussian_anchor_target_path_cache",
        "fold_fingerprint": manifest.fingerprint,
        "map_sha256": map_sha256,
        "anchor_count": int(anchor_count),
        "anchor_source": anchor_source,
        "feature_names": list(FEATURE_NAMES),
        "token_config": asdict(token_config),
        "fit_indices_sha256": _indices_hash(fit_indices),
        "shape": {name: list(value.shape) for name, value in arrays.items()},
        "dtype": {name: str(value.dtype) for name, value in arrays.items()},
        "sha256": {
            name: _sha256_file(destination / name) for name in arrays
        },
    }
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination, delete=False
    )
    temporary = Path(handle.name)
    try:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.close()
        os.replace(temporary, destination / "manifest.json")
    finally:
        if not handle.closed:
            handle.close()
        if temporary.exists():
            temporary.unlink()
    return report
