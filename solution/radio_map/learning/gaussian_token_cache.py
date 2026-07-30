"""Sequence-valued official-map Gaussian propagation tokens."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from tqdm.auto import tqdm

from .gaussian_geometry import GaussianScene


FEATURE_NAMES = (
    "center_from_start_x",
    "center_from_start_y",
    "center_from_start_z",
    "center_from_end_x",
    "center_from_end_y",
    "center_from_end_z",
    "normal_x",
    "normal_y",
    "normal_z",
    "tangent_scale",
    "normal_scale",
    "opacity",
    "log_count",
    "path_fraction",
    "lateral_distance_ratio",
    "kernel_response",
    "transmittance",
    "log_transmittance",
    "excess_length_ratio",
    "incidence_alignment",
    "departure_alignment",
    "wall_strength",
    "surface_ground",
    "surface_wall",
    "surface_roof",
    "surface_elevated",
    "map_present",
)


@dataclass(frozen=True)
class GaussianTokenConfig:
    tokens_per_path: int = 48
    candidate_k: int = 12
    path_samples: int = 16
    elevated_quota: int = 16
    ground_height: float = 1.5

    def __post_init__(self) -> None:
        for name in ("tokens_per_path", "candidate_k", "path_samples"):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 2
            ):
                raise ValueError(f"{name} must be an integer >= 2")
        if (
            not isinstance(self.elevated_quota, int)
            or isinstance(self.elevated_quota, bool)
            or self.elevated_quota < 0
            or self.elevated_quota > self.tokens_per_path
        ):
            raise ValueError("elevated_quota must lie in [0, tokens_per_path]")
        if (
            not isinstance(self.ground_height, (int, float))
            or isinstance(self.ground_height, bool)
            or not math.isfinite(float(self.ground_height))
            or float(self.ground_height) < 0
        ):
            raise ValueError("ground_height must be finite and non-negative")


@dataclass(frozen=True)
class GaussianPathTokens:
    tokens: np.ndarray
    mask: np.ndarray
    gaussian_indices: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def __post_init__(self) -> None:
        count, width, feature_count = self.tokens.shape
        if (
            self.tokens.dtype != np.float32
            or self.mask.shape != (count, width)
            or self.mask.dtype != np.bool_
            or self.gaussian_indices.shape != (count, width)
            or self.gaussian_indices.dtype != np.int64
            or feature_count != len(self.feature_names)
            or not np.isfinite(self.tokens).all()
        ):
            raise ValueError("invalid Gaussian path token arrays")


def _validate_endpoints(
    starts: np.ndarray, ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(starts, dtype=np.float32)
    target = np.asarray(ends, dtype=np.float32)
    if (
        source.shape != target.shape
        or source.ndim != 2
        or source.shape[1] != 3
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("path endpoints must be finite with shape (paths,3)")
    return source, target


def build_gaussian_path_tokens(
    scene: GaussianScene,
    starts: np.ndarray,
    ends: np.ndarray,
    config: GaussianTokenConfig | None = None,
) -> GaussianPathTokens:
    """Select surface-aware Gaussian tokens along propagation corridors."""

    if not isinstance(scene, GaussianScene):
        raise TypeError("scene must be a GaussianScene")
    config = config or GaussianTokenConfig()
    source, target = _validate_endpoints(starts, ends)
    path_count = len(source)
    tokens = np.zeros(
        (path_count, config.tokens_per_path, len(FEATURE_NAMES)),
        dtype=np.float32,
    )
    mask = np.zeros((path_count, config.tokens_per_path), dtype=bool)
    gaussian_indices = np.full(
        (path_count, config.tokens_per_path), -1, dtype=np.int64
    )
    fractions = np.linspace(
        0.0, 1.0, config.path_samples, dtype=np.float32
    )
    scene_ground = float(np.min(scene.centers[:, 2]))
    eps = 1e-6
    all_samples = (
        source[:, None, :]
        + fractions[None, :, None] * (target - source)[:, None, :]
    )
    _, all_sampled_indices = scene.query_candidates(
        all_samples.reshape(-1, 3), config.candidate_k
    )
    all_sampled_indices = all_sampled_indices.reshape(
        path_count, config.path_samples, -1
    )
    for path_index, (start, end) in enumerate(
        tqdm(
            zip(source, target),
            total=path_count,
            desc="Gaussian path tokens",
            leave=False,
            dynamic_ncols=True,
        )
    ):
        delta = end - start
        length = float(np.linalg.norm(delta))
        safe_length = max(length, eps)
        candidates = np.unique(all_sampled_indices[path_index].reshape(-1))
        centers = scene.centers[candidates]
        if length <= eps:
            projection = np.zeros(len(candidates), dtype=np.float32)
        else:
            projection = np.clip(
                ((centers - start) @ delta) / (length * length),
                0.0,
                1.0,
            ).astype(np.float32)
        projected = start[None, :] + projection[:, None] * delta[None, :]
        lateral = np.linalg.norm(centers - projected, axis=1)
        response = scene.kernel_response(
            projected, candidates[:, None]
        )[:, 0]
        normal_z = np.abs(scene.normals[candidates, 2])
        ground = (
            (centers[:, 2] <= scene_ground + float(config.ground_height))
            & (normal_z >= 0.5)
        )
        elevated = ~ground
        score = response * (1.0 + 0.15 * np.log1p(scene.counts[candidates]))
        stable_order = np.lexsort((candidates, -score))
        elevated_order = stable_order[elevated[stable_order]]
        selected_elevated = elevated_order[: config.elevated_quota]
        already = set(int(value) for value in selected_elevated)
        remaining = np.asarray(
            [value for value in stable_order if int(value) not in already],
            dtype=np.int64,
        )
        selected_local = np.concatenate(
            (
                selected_elevated,
                remaining[
                    : max(
                        0,
                        config.tokens_per_path - len(selected_elevated),
                    )
                ],
            )
        )
        selected_local = selected_local[: config.tokens_per_path]
        selected_local = selected_local[
            np.argsort(projection[selected_local], kind="stable")
        ]
        selected = candidates[selected_local]
        count = len(selected)
        if count == 0:
            continue
        selected_centers = scene.centers[selected]
        selected_normals = scene.normals[selected]
        selected_projection = projection[selected_local]
        selected_lateral = lateral[selected_local]
        selected_response = response[selected_local]
        selected_ground = ground[selected_local]
        selected_wall = np.abs(selected_normals[:, 2]) < 0.5
        selected_roof = (~selected_ground) & (~selected_wall)
        selected_elevated_flag = ~selected_ground
        incoming = selected_centers - start[None, :]
        outgoing = end[None, :] - selected_centers
        incoming /= np.maximum(
            np.linalg.norm(incoming, axis=1, keepdims=True), eps
        )
        outgoing /= np.maximum(
            np.linalg.norm(outgoing, axis=1, keepdims=True), eps
        )
        incidence = np.abs(np.sum(selected_normals * incoming, axis=1))
        departure = np.abs(np.sum(selected_normals * outgoing, axis=1))
        excess = (
            np.linalg.norm(selected_centers - start[None, :], axis=1)
            + np.linalg.norm(end[None, :] - selected_centers, axis=1)
            - length
        ) / safe_length
        transmittance = np.ones(count, dtype=np.float32)
        if count > 1:
            transmittance[1:] = np.cumprod(
                1.0 - selected_response[:-1], dtype=np.float32
            )
        rows = np.column_stack(
            (
                (selected_centers - start[None, :]) / safe_length,
                (selected_centers - end[None, :]) / safe_length,
                selected_normals,
                scene.tangent_scales[selected],
                scene.normal_scales[selected],
                scene.opacities[selected],
                np.log1p(scene.counts[selected]),
                selected_projection,
                selected_lateral / safe_length,
                selected_response,
                transmittance,
                np.log(np.maximum(transmittance, 1e-8)),
                excess,
                incidence,
                departure,
                1.0 - np.abs(selected_normals[:, 2]),
                selected_ground.astype(np.float32),
                selected_wall.astype(np.float32),
                selected_roof.astype(np.float32),
                selected_elevated_flag.astype(np.float32),
                np.ones(count, dtype=np.float32),
            )
        ).astype(np.float32)
        tokens[path_index, :count] = rows
        mask[path_index, :count] = True
        gaussian_indices[path_index, :count] = selected
    return GaussianPathTokens(tokens, mask, gaussian_indices)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_indices(indices: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(indices, dtype="<i8").tobytes()
    ).hexdigest()


def _normalization(
    train: GaussianPathTokens, fit_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(fit_indices, dtype=np.int64)
    if (
        indices.ndim != 1
        or len(indices) == 0
        or int(indices.min()) < 0
        or int(indices.max()) >= len(train.tokens)
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("train_fit_indices must be unique valid rows")
    values = train.tokens[indices][train.mask[indices]]
    if not len(values):
        raise ValueError("training Gaussian tokens contain no valid rows")
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).clip(1e-6).astype(np.float32)
    for index, name in enumerate(FEATURE_NAMES):
        if name.startswith("surface_") or name == "map_present":
            mean[index] = 0.0
            std[index] = 1.0
    return mean, std


def _normalized(
    batch: GaussianPathTokens, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    values = np.zeros_like(batch.tokens)
    values[batch.mask] = (batch.tokens[batch.mask] - mean) / std
    return values


def save_gaussian_token_cache(
    path: str | Path,
    train: GaussianPathTokens,
    test: GaussianPathTokens,
    *,
    fold_fingerprint: str,
    map_sha256: str,
    train_fit_indices: np.ndarray,
    config: GaussianTokenConfig | None = None,
) -> Path:
    destination = Path(path)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("refusing to overwrite an existing Gaussian token cache")
    destination.mkdir(parents=True, exist_ok=True)
    if train.feature_names != test.feature_names:
        raise ValueError("train/test Gaussian token schemas differ")
    mean, std = _normalization(train, train_fit_indices)
    arrays = {
        "train_tokens.npy": _normalized(train, mean, std),
        "train_mask.npy": train.mask,
        "train_gaussian_indices.npy": train.gaussian_indices,
        "test_tokens.npy": _normalized(test, mean, std),
        "test_mask.npy": test.mask,
        "test_gaussian_indices.npy": test.gaussian_indices,
        "normalization_mean.npy": mean,
        "normalization_std.npy": std,
    }
    for name, values in arrays.items():
        np.save(destination / name, values, allow_pickle=False)
    manifest = {
        "format_version": 1,
        "kind": "gaussian_path_token_cache",
        "fold_fingerprint": str(fold_fingerprint),
        "map_sha256": str(map_sha256),
        "fit_indices_sha256": _sha256_indices(train_fit_indices),
        "feature_names": list(train.feature_names),
        "config": None if config is None else asdict(config),
        "shape": {
            name: list(values.shape) for name, values in arrays.items()
        },
        "dtype": {
            name: str(values.dtype) for name, values in arrays.items()
        },
        "sha256": {
            name: _sha256_file(destination / name) for name in arrays
        },
    }
    handle = tempfile.NamedTemporaryFile(
        prefix="manifest.", suffix=".tmp", dir=destination, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination / "manifest.json")
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


class GaussianTokenCache:
    def __init__(self, path: str | Path, manifest: dict[str, object]) -> None:
        self.path = Path(path)
        self.manifest = manifest
        self.feature_names = tuple(manifest["feature_names"])
        self.fingerprint = hashlib.sha256(
            json.dumps(
                manifest, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        self._arrays = {
            name: np.load(self.path / name, mmap_mode="r", allow_pickle=False)
            for name in manifest["shape"]
        }

    def close(self) -> None:
        for array in self._arrays.values():
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()
        self._arrays.clear()

    def __enter__(self) -> "GaussianTokenCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @classmethod
    def load(
        cls,
        path: str | Path,
        fold_fingerprint: str,
        map_sha256: str,
    ) -> "GaussianTokenCache":
        source = Path(path)
        try:
            manifest = json.loads(
                (source / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("invalid Gaussian token manifest") from error
        if (
            not isinstance(manifest, dict)
            or manifest.get("format_version") != 1
            or manifest.get("kind") != "gaussian_path_token_cache"
        ):
            raise ValueError("invalid Gaussian token manifest")
        if manifest.get("fold_fingerprint") != fold_fingerprint:
            raise ValueError("Gaussian token cache fold differs")
        if manifest.get("map_sha256") != map_sha256:
            raise ValueError("Gaussian token cache map differs")
        required = set(manifest.get("shape", {}))
        if (
            required
            != {
                "train_tokens.npy",
                "train_mask.npy",
                "train_gaussian_indices.npy",
                "test_tokens.npy",
                "test_mask.npy",
                "test_gaussian_indices.npy",
                "normalization_mean.npy",
                "normalization_std.npy",
            }
            or set(manifest.get("dtype", {})) != required
            or set(manifest.get("sha256", {})) != required
            or tuple(manifest.get("feature_names", ())) != FEATURE_NAMES
        ):
            raise ValueError("invalid Gaussian token cache schema")
        for name in required:
            file_path = source / name
            if (
                not file_path.is_file()
                or _sha256_file(file_path) != manifest["sha256"][name]
            ):
                raise ValueError(f"Gaussian token cache hash differs for {name}")
            array = np.load(file_path, mmap_mode="r", allow_pickle=False)
            if (
                list(array.shape) != manifest["shape"][name]
                or str(array.dtype) != manifest["dtype"][name]
            ):
                raise ValueError(
                    f"Gaussian token cache shape/dtype differs for {name}"
                )
        return cls(source, manifest)

    def values(
        self,
        split: Literal["train", "test"],
        mode: Literal["real", "zero", "shuffle"] = "real",
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        if split not in ("train", "test"):
            raise ValueError("split must be train or test")
        if mode not in ("real", "zero", "shuffle"):
            raise ValueError("mode must be real, zero, or shuffle")
        tokens = self._arrays[f"{split}_tokens.npy"]
        mask = self._arrays[f"{split}_mask.npy"]
        if mode == "real":
            return tokens, mask
        if mode == "zero":
            return np.zeros_like(tokens), np.asarray(mask)
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(tokens))
        if len(tokens) > 1 and np.array_equal(
            permutation, np.arange(len(tokens))
        ):
            permutation = np.roll(permutation, 1)
        return np.asarray(tokens[permutation]), np.asarray(mask[permutation])

    @property
    def gaussian_count(self) -> int:
        """Number of scene Gaussians addressable by the persisted paths."""

        maximum = -1
        for split in ("train", "test"):
            values = self._arrays[f"{split}_gaussian_indices.npy"]
            if values.size:
                maximum = max(maximum, int(np.max(values)))
        if maximum < 0:
            raise ValueError("Gaussian token cache contains no scene indices")
        return maximum + 1

    @property
    def normalization(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self._arrays["normalization_mean.npy"]).copy(),
            np.asarray(self._arrays["normalization_std.npy"]).copy(),
        )

    def gaussian_indices(
        self,
        split: Literal["train", "test"],
        mode: Literal["real", "zero", "shuffle"] = "real",
        seed: int = 0,
    ) -> np.ndarray:
        """Return Gaussian identities with the same row transform as values()."""

        if split not in ("train", "test"):
            raise ValueError("split must be train or test")
        if mode not in ("real", "zero", "shuffle"):
            raise ValueError("mode must be real, zero, or shuffle")
        values = self._arrays[f"{split}_gaussian_indices.npy"]
        if mode in ("real", "zero"):
            return values
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(values))
        if len(values) > 1 and np.array_equal(
            permutation, np.arange(len(values))
        ):
            permutation = np.roll(permutation, 1)
        return np.asarray(values[permutation])
