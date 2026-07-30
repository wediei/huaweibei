"""Versioned train/test cache for deployable ground-suppressed map features."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..building_prior_cli import _path_features
from ..data import RoundDataset
from ..geometry import GeometryPrior
from ..splits import coverage_split


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


@dataclass(frozen=True)
class BuildingFeatureCache:
    root: Path
    train_features: np.ndarray
    test_features: np.ndarray
    feature_names: tuple[str, ...]
    manifest: dict[str, object]

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        dataset: RoundDataset | None = None,
        fold_manifest_fingerprint: str | None = None,
    ) -> "BuildingFeatureCache":
        root = Path(root)
        try:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("invalid building feature cache manifest") from error
        required = {
            "format_version",
            "kind",
            "feature_names",
            "feature_dim",
            "train_shape",
            "test_shape",
            "train_features_sha256",
            "test_features_sha256",
            "fold_manifest_fingerprint",
            "source_sha256",
            "normalization",
        }
        if type(manifest) is not dict or set(manifest) != required:
            raise ValueError("invalid building feature cache schema")
        if manifest["format_version"] != 1 or manifest["kind"] != "building_features":
            raise ValueError("unsupported building feature cache")
        train_path, test_path = root / "train_features.npy", root / "test_features.npy"
        if (
            not train_path.is_file()
            or not test_path.is_file()
            or sha256_file(train_path) != manifest["train_features_sha256"]
            or sha256_file(test_path) != manifest["test_features_sha256"]
        ):
            raise ValueError("building feature cache hash mismatch")
        train = np.load(train_path, mmap_mode="r", allow_pickle=False)
        test = np.load(test_path, mmap_mode="r", allow_pickle=False)
        if (
            list(train.shape) != manifest["train_shape"]
            or list(test.shape) != manifest["test_shape"]
            or train.dtype != np.float32
            or test.dtype != np.float32
            or not np.isfinite(train).all()
            or not np.isfinite(test).all()
        ):
            raise ValueError("invalid building feature arrays")
        if dataset is not None and (
            len(train) != len(dataset.train_pos) or len(test) != len(dataset.test_pos)
        ):
            raise ValueError("building feature cache differs from dataset size")
        if (
            fold_manifest_fingerprint is not None
            and manifest["fold_manifest_fingerprint"] != fold_manifest_fingerprint
        ):
            raise ValueError("building feature cache fold differs")
        names = tuple(str(value) for value in manifest["feature_names"])
        if len(names) != train.shape[1] or manifest["feature_dim"] != train.shape[1]:
            raise ValueError("building feature names differ from array width")
        return cls(root, train, test, names, manifest)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.manifest, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def values(
        self,
        split: str,
        *,
        mode: str = "real",
        seed: int = 42,
    ) -> np.ndarray:
        source = self.train_features if split == "train" else self.test_features
        if split not in ("train", "test"):
            raise ValueError("split must be train or test")
        if mode == "real":
            return source
        if mode == "zero":
            return np.zeros(source.shape, dtype=np.float32)
        if mode == "shuffle":
            permutation = np.random.default_rng(seed).permutation(len(source))
            return np.asarray(source[permutation], dtype=np.float32)
        raise ValueError("map mode must be real, zero, or shuffle")


def build_feature_cache(
    dataset: RoundDataset,
    source_geometry: GeometryPrior,
    building_prior: GeometryPrior,
    output_dir: str | Path,
    *,
    train_indices: np.ndarray,
    fold_manifest_fingerprint: str,
    source_paths: dict[str, str | Path],
    path_samples: int = 64,
) -> BuildingFeatureCache:
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if (
        train_indices.ndim != 1
        or len(train_indices) == 0
        or len(np.unique(train_indices)) != len(train_indices)
        or int(train_indices.min()) < 0
        or int(train_indices.max()) >= len(dataset.train_pos)
    ):
        raise ValueError("invalid feature-cache fit indices")
    bs = np.asarray(dataset.config.bs_position, dtype=np.float64)

    def raw(positions: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
        point = building_prior.sample_points(positions).astype(np.float64)
        path, path_names = _path_features(
            source_geometry, building_prior, bs, positions, path_samples
        )
        return (
            np.concatenate((point, path), axis=1),
            tuple(building_prior.feature_names) + tuple(path_names),
        )

    raw_train, names = raw(np.asarray(dataset.train_pos, dtype=np.float64))
    raw_test, test_names = raw(np.asarray(dataset.test_pos, dtype=np.float64))
    if names != test_names:
        raise RuntimeError("train/test building feature names differ")
    mean = raw_train[train_indices].mean(axis=0, dtype=np.float64)
    scale = raw_train[train_indices].std(axis=0, dtype=np.float64)
    scale[scale < 1e-6] = 1.0
    train = np.clip((raw_train - mean) / scale, -20.0, 20.0).astype(np.float32)
    test = np.clip((raw_test - mean) / scale, -20.0, 20.0).astype(np.float32)
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("non-finite normalized building features")
    output = Path(output_dir)
    if output.exists():
        raise ValueError("building feature output directory already exists")
    output.mkdir(parents=True)
    np.save(output / "train_features.npy", train)
    np.save(output / "test_features.npy", test)
    manifest: dict[str, object] = {
        "format_version": 1,
        "kind": "building_features",
        "feature_names": list(names),
        "feature_dim": int(train.shape[1]),
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "train_features_sha256": sha256_file(output / "train_features.npy"),
        "test_features_sha256": sha256_file(output / "test_features.npy"),
        "fold_manifest_fingerprint": str(fold_manifest_fingerprint),
        "source_sha256": {
            name: sha256_file(path) for name, path in sorted(source_paths.items())
        },
        "normalization": {
            "fit_indices_sha256": _array_hash(train_indices.astype("<i8")),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "clip": 20.0,
            "path_samples": int(path_samples),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return BuildingFeatureCache.load(output)


def coverage_feature_cache(
    dataset: RoundDataset,
    source_geometry: GeometryPrior,
    building_prior: GeometryPrior,
    output_dir: str | Path,
    *,
    validation_fraction: float = 0.1,
    grid_size: float = 20.0,
    seed: int = 42,
    path_samples: int = 64,
    source_paths: dict[str, str | Path],
) -> BuildingFeatureCache:
    split = coverage_split(
        dataset.train_pos, validation_fraction, grid_size, seed
    )
    fingerprint = hashlib.sha256(
        np.ascontiguousarray(split.train.astype("<i8")).view(np.uint8)
    ).hexdigest()
    return build_feature_cache(
        dataset,
        source_geometry,
        building_prior,
        output_dir,
        train_indices=split.train,
        fold_manifest_fingerprint=f"coverage:{fingerprint}",
        source_paths=source_paths,
        path_samples=path_samples,
    )
