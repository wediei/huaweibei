"""Anisotropic 3-D Gaussian scene backend for radio-map completion.

The implementation borrows the explicit mean/covariance/opacity
parameterization used by 3D Gaussian Splatting and the attenuation-oriented
interpretation used by XFreq-GS.  It deliberately does not copy a camera
rasterizer or perform traditional ray tracing.  Instead, Gaussian kernels are
alpha-composited at samples along BS-UE and Anchor-UE propagation segments.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from ..anchors import AnchorMemory
from ..data import RoundDataset
from ..geometry import PlyPointCloud
from .cache import FoldCacheManifest, validate_cache


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class GaussianSceneConfig:
    """Fixed, auditable PLY-to-Gaussian initialization controls."""

    voxel_size: float = 1.0
    normal_scale: float = 0.20
    tangent_scale: float = 0.65
    density_scale: float = 4.0

    def __post_init__(self) -> None:
        for name in (
            "voxel_size", "normal_scale", "tangent_scale", "density_scale"
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number")


@dataclass(frozen=True)
class GaussianSplatConfig:
    """Local selection and propagation-segment splat resolution."""

    local_k: int = 24
    path_k: int = 8
    path_samples: int = 16
    path_batch_size: int = 256

    def __post_init__(self) -> None:
        for name in ("local_k", "path_k", "path_samples", "path_batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        if self.path_samples != 16:
            raise ValueError("path_samples must be 16 for V2 feature compatibility")


class GaussianScene:
    """A compact anisotropic Gaussian map in the original metre frame."""

    LOCAL_FEATURE_NAMES = (
        "log_response_sum",
        "response_max",
        "effective_gaussian_count",
        "weighted_distance_mean",
        "weighted_distance_std",
        "nearest_center_distance",
        "weighted_offset_x",
        "weighted_offset_y",
        "weighted_offset_z",
        "weighted_normal_x",
        "weighted_normal_y",
        "weighted_normal_z",
        "weighted_tangent_scale",
        "weighted_normal_scale",
        "weighted_opacity",
        "weighted_wall_strength",
    )

    def __init__(
        self,
        centers: np.ndarray,
        normals: np.ndarray,
        tangent_scales: np.ndarray,
        normal_scales: np.ndarray,
        opacities: np.ndarray,
        counts: np.ndarray,
        metadata: dict[str, object],
    ) -> None:
        self.centers = np.asarray(centers, dtype=np.float32)
        self.normals = np.asarray(normals, dtype=np.float32)
        self.tangent_scales = np.asarray(tangent_scales, dtype=np.float32)
        self.normal_scales = np.asarray(normal_scales, dtype=np.float32)
        self.opacities = np.asarray(opacities, dtype=np.float32)
        self.counts = np.asarray(counts, dtype=np.int32)
        count = len(self.centers)
        if (
            self.centers.shape != (count, 3)
            or self.normals.shape != (count, 3)
            or self.tangent_scales.shape != (count,)
            or self.normal_scales.shape != (count,)
            or self.opacities.shape != (count,)
            or self.counts.shape != (count,)
            or count < 2
        ):
            raise ValueError("invalid Gaussian scene array shapes")
        arrays = (
            self.centers, self.normals, self.tangent_scales,
            self.normal_scales, self.opacities,
        )
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("Gaussian scene contains non-finite values")
        if (
            (self.tangent_scales <= 0).any()
            or (self.normal_scales <= 0).any()
            or (self.opacities <= 0).any()
            or (self.opacities >= 1).any()
            or (self.counts < 1).any()
        ):
            raise ValueError("Gaussian scene contains invalid scale/opacity/count")
        lengths = np.linalg.norm(self.normals, axis=1)
        if not np.allclose(lengths, 1.0, atol=1e-4):
            raise ValueError("Gaussian normals must be unit vectors")
        self.metadata = dict(metadata)
        # For covariance with two equal tangent axes:
        # inv(Sigma) = I/t^2 + (1/n^2 - 1/t^2) normal normal^T.
        identity = np.eye(3, dtype=np.float32)[None, :, :]
        outer = self.normals[:, :, None] * self.normals[:, None, :]
        tangent_inverse = 1.0 / np.square(self.tangent_scales)
        normal_inverse = 1.0 / np.square(self.normal_scales)
        self.inverse_covariances = (
            tangent_inverse[:, None, None] * identity
            + (normal_inverse - tangent_inverse)[:, None, None] * outer
        ).astype(np.float32)
        self.tree = cKDTree(self.centers)

    @classmethod
    def from_ply(
        cls,
        path: str | Path,
        config: GaussianSceneConfig | None = None,
    ) -> "GaussianScene":
        config = config or GaussianSceneConfig()
        cloud = PlyPointCloud.open(path)
        position_parts: list[np.ndarray] = []
        normal_parts: list[np.ndarray] = []
        for positions, normals in tqdm(
            cloud.iter_batches(262144),
            total=math.ceil(cloud.vertex_count / 262144),
            desc="load PLY for 3DGS",
            dynamic_ncols=True,
        ):
            position_parts.append(np.asarray(positions, dtype=np.float32))
            normal_parts.append(np.asarray(normals, dtype=np.float32))
        points = np.concatenate(position_parts)
        normals = np.concatenate(normal_parts)
        finite = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1)
        points, normals = points[finite], normals[finite]
        if len(points) < 2:
            raise ValueError("PLY must contain at least two finite vertices")

        origin = points.min(axis=0)
        voxel_indices = np.floor(
            (points - origin) / float(config.voxel_size)
        ).astype(np.int64)
        _, inverse, counts = np.unique(
            voxel_indices, axis=0, return_inverse=True, return_counts=True
        )
        gaussian_count = len(counts)
        center_sums = np.zeros((gaussian_count, 3), dtype=np.float64)
        normal_sums = np.zeros((gaussian_count, 3), dtype=np.float64)
        np.add.at(center_sums, inverse, points)
        np.add.at(normal_sums, inverse, normals)
        centers = (center_sums / counts[:, None]).astype(np.float32)
        normal_lengths = np.linalg.norm(normal_sums, axis=1, keepdims=True)
        unit_normals = np.divide(
            normal_sums,
            normal_lengths,
            out=np.zeros_like(normal_sums),
            where=normal_lengths > 1e-12,
        ).astype(np.float32)
        missing = np.linalg.norm(unit_normals, axis=1) < 0.5
        unit_normals[missing] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        density_factor = 1.0 + 0.10 * np.log1p(counts.astype(np.float32))
        tangent_scales = (
            float(config.voxel_size)
            * float(config.tangent_scale)
            * density_factor
        ).astype(np.float32)
        normal_scales = np.full(
            gaussian_count,
            float(config.voxel_size) * float(config.normal_scale),
            dtype=np.float32,
        )
        opacities = (
            1.0 - np.exp(-counts.astype(np.float32) / float(config.density_scale))
        ).clip(1e-4, 1.0 - 1e-4)
        metadata = {
            "format_version": 1,
            "backend": "anisotropic_3d_gaussian_splat",
            "source_map": str(Path(path).resolve()),
            "source_map_sha256": _sha256_file(path),
            "source_vertex_count": int(cloud.vertex_count),
            "finite_vertex_count": int(len(points)),
            "gaussian_count": int(gaussian_count),
            "coordinate_frame": "original_metres",
            "scene_config": asdict(config),
            "reference_designs": [
                "3DGS mean-covariance-opacity parameterization",
                "XFreq-GS attenuation-oriented Gaussian field",
            ],
        }
        return cls(
            centers,
            unit_normals,
            tangent_scales,
            normal_scales,
            opacities,
            counts.astype(np.int32),
            metadata,
        )

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            centers=self.centers,
            normals=self.normals,
            tangent_scales=self.tangent_scales,
            normal_scales=self.normal_scales,
            opacities=self.opacities,
            counts=self.counts,
            metadata=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "GaussianScene":
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "centers", "normals", "tangent_scales", "normal_scales",
                "opacities", "counts", "metadata",
            }
            if set(archive.files) != required:
                raise ValueError("invalid Gaussian scene cache")
            return cls(
                archive["centers"],
                archive["normals"],
                archive["tangent_scales"],
                archive["normal_scales"],
                archive["opacities"],
                archive["counts"],
                json.loads(str(archive["metadata"].item())),
            )

    @property
    def gaussian_count(self) -> int:
        return len(self.centers)

    def _query(self, points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(points, dtype=np.float32)
        if (
            values.ndim != 2
            or values.shape[1] != 3
            or not np.isfinite(values).all()
        ):
            raise ValueError("query points must be finite with shape (samples, 3)")
        count = min(int(k), self.gaussian_count)
        distances, indices = self.tree.query(values, k=count, workers=-1)
        if count == 1:
            distances, indices = distances[:, None], indices[:, None]
        return (
            np.asarray(distances, dtype=np.float32),
            np.asarray(indices, dtype=np.int64),
        )

    def _kernel_response(
        self, points: np.ndarray, indices: np.ndarray
    ) -> np.ndarray:
        values = np.asarray(points, dtype=np.float32)
        selected = np.asarray(indices, dtype=np.int64)
        offsets = self.centers[selected] - values[:, None, :]
        inverse = self.inverse_covariances[selected]
        squared_mahalanobis = np.einsum(
            "nki,nkij,nkj->nk", offsets, inverse, offsets, optimize=True
        )
        exponent = np.clip(-0.5 * squared_mahalanobis, -60.0, 0.0)
        return (
            self.opacities[selected] * np.exp(exponent)
        ).clip(0.0, 1.0 - 1e-6).astype(np.float32)

    def query_candidates(
        self, points: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Public, read-only candidate query for sequence tokenizers."""

        return self._query(points, k)

    def kernel_response(
        self, points: np.ndarray, indices: np.ndarray
    ) -> np.ndarray:
        """Public anisotropic response with the same semantics as V2 features."""

        return self._kernel_response(points, indices)

    def local_features(self, queries: np.ndarray, k: int = 24) -> np.ndarray:
        query = np.asarray(queries, dtype=np.float32)
        distances, indices = self._query(query, k)
        response = self._kernel_response(query, indices)
        response_sum = response.sum(axis=1, keepdims=True)
        weights = response / np.maximum(response_sum, 1e-8)
        empty = response_sum[:, 0] <= 1e-8
        if empty.any():
            weights[empty] = 0.0
            weights[empty, 0] = 1.0
        offsets = self.centers[indices] - query[:, None, :]
        normal = self.normals[indices]
        distance_mean = (weights * distances).sum(axis=1)
        distance_std = np.sqrt(
            np.maximum(
                (weights * np.square(distances - distance_mean[:, None])).sum(
                    axis=1
                ),
                0.0,
            )
        )
        effective = 1.0 / np.maximum(np.square(weights).sum(axis=1), 1e-8)
        weighted = lambda values: (weights * values).sum(axis=1)
        features = np.column_stack(
            (
                np.log1p(response_sum[:, 0]),
                response.max(axis=1),
                effective,
                distance_mean,
                distance_std,
                distances[:, 0],
                weighted(offsets[:, :, 0]),
                weighted(offsets[:, :, 1]),
                weighted(offsets[:, :, 2]),
                weighted(normal[:, :, 0]),
                weighted(normal[:, :, 1]),
                weighted(normal[:, :, 2]),
                weighted(self.tangent_scales[indices]),
                weighted(self.normal_scales[indices]),
                weighted(self.opacities[indices]),
                weighted(1.0 - np.abs(normal[:, :, 2])),
            )
        )
        return np.asarray(features, dtype=np.float32)

    def path_splat_profiles(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
        config: GaussianSplatConfig | None = None,
    ) -> np.ndarray:
        """Alpha-composite anisotropic Gaussian responses along 3-D segments."""

        config = config or GaussianSplatConfig()
        source = np.asarray(starts, dtype=np.float32)
        target = np.asarray(ends, dtype=np.float32)
        if (
            source.shape != target.shape
            or source.ndim != 2
            or source.shape[1] != 3
            or not np.isfinite(source).all()
            or not np.isfinite(target).all()
        ):
            raise ValueError("path endpoints must be finite with shape (paths, 3)")
        output = np.empty(
            (len(source), config.path_samples), dtype=np.float32
        )
        fractions = np.linspace(
            0.0, 1.0, config.path_samples, dtype=np.float32
        )
        for begin in tqdm(
            range(0, len(source), config.path_batch_size),
            desc="3DGS path splats",
            leave=False,
            dynamic_ncols=True,
        ):
            stop = min(len(source), begin + config.path_batch_size)
            samples = (
                source[begin:stop, None, :]
                + fractions[None, :, None]
                * (target[begin:stop] - source[begin:stop])[:, None, :]
            )
            flattened = samples.reshape(-1, 3)
            _, indices = self._query(flattened, config.path_k)
            alpha = self._kernel_response(flattened, indices)
            # Front-to-back alpha compositing at each sample; this is the
            # Gaussian splat response, not a material-aware ray trace.
            composite = 1.0 - np.prod(1.0 - alpha, axis=1)
            output[begin:stop] = composite.reshape(
                stop - begin, config.path_samples
            )
        return output


def prepare_gaussian_feature_cache(
    data_dir: str | Path,
    fold_cache_dir: str | Path,
    scene_cache_path: str | Path,
    output_path: str | Path,
    *,
    scene_config: GaussianSceneConfig | None = None,
    splat_config: GaussianSplatConfig | None = None,
    rebuild_scene: bool = False,
) -> Path:
    """Create a Geometry-V2-compatible 32/16 dimensional Gaussian cache."""

    scene_config = scene_config or GaussianSceneConfig()
    splat_config = splat_config or GaussianSplatConfig()
    fold_cache = Path(fold_cache_dir)
    manifest = FoldCacheManifest.load(fold_cache / "manifest.json")
    validate_cache(manifest, fold_cache)
    dataset = RoundDataset.open(data_dir)
    if dataset.data_dir != Path(manifest.data_dir).resolve():
        raise ValueError("data directory differs from fold cache")
    scene_path = Path(scene_cache_path)
    if scene_path.is_file() and not rebuild_scene:
        scene = GaussianScene.load(scene_path)
        if scene.metadata.get("source_map_sha256") != _sha256_file(dataset.map_path):
            raise ValueError("Gaussian scene was built from a different map")
        if scene.metadata.get("scene_config") != asdict(scene_config):
            raise ValueError("Gaussian scene configuration differs")
    else:
        scene = GaussianScene.from_ply(dataset.map_path, scene_config)
        scene.save(scene_path)

    train_indices = np.asarray(
        np.load(fold_cache / "train_indices.npy", allow_pickle=False),
        dtype=np.int64,
    )
    neighbors = np.asarray(
        np.load(fold_cache / "neighbor_indices.npy", allow_pickle=False),
        dtype=np.int64,
    )
    train_positions = np.asarray(dataset.train_pos, dtype=np.float32)
    test_positions = np.asarray(dataset.test_pos, dtype=np.float32)
    bs = np.asarray(dataset.config.bs_position, dtype=np.float32)

    local_train = scene.local_features(train_positions, splat_config.local_k)
    local_test = scene.local_features(test_positions, splat_config.local_k)
    bs_train = np.broadcast_to(bs, train_positions.shape)
    bs_test = np.broadcast_to(bs, test_positions.shape)
    target_train_raw = np.concatenate(
        (
            local_train,
            scene.path_splat_profiles(bs_train, train_positions, splat_config),
        ),
        axis=1,
    )
    target_test_raw = np.concatenate(
        (
            local_test,
            scene.path_splat_profiles(bs_test, test_positions, splat_config),
        ),
        axis=1,
    )

    anchor_sources = train_positions[neighbors].reshape(-1, 3)
    anchor_targets = np.broadcast_to(
        train_positions[:, None, :], neighbors.shape + (3,)
    ).reshape(-1, 3)
    anchor_train_raw = scene.path_splat_profiles(
        anchor_sources, anchor_targets, splat_config
    ).reshape(len(train_positions), neighbors.shape[1], 16)

    memory = AnchorMemory(
        train_positions[train_indices],
        np.zeros((len(train_indices), 1), dtype=np.complex64),
        bs,
        train_indices,
    )
    test_query = memory.query(test_positions, manifest.anchor_count)
    test_sources = train_positions[test_query.source_indices].reshape(-1, 3)
    test_targets = np.broadcast_to(
        test_positions[:, None, :], test_query.source_indices.shape + (3,)
    ).reshape(-1, 3)
    anchor_test_raw = scene.path_splat_profiles(
        test_sources, test_targets, splat_config
    ).reshape(len(test_positions), manifest.anchor_count, 16)

    target_mean = target_train_raw[train_indices].mean(
        axis=0, dtype=np.float64
    ).astype(np.float32)
    target_std = target_train_raw[train_indices].std(
        axis=0, dtype=np.float64
    ).clip(1e-6).astype(np.float32)
    anchor_fit = anchor_train_raw[train_indices].reshape(-1, 16)
    anchor_mean = anchor_fit.mean(axis=0, dtype=np.float64).astype(np.float32)
    anchor_std = anchor_fit.std(axis=0, dtype=np.float64).clip(1e-6).astype(
        np.float32
    )
    normalize_target = lambda value: np.asarray(
        (value - target_mean) / target_std, dtype=np.float32
    )
    normalize_anchor = lambda value: np.asarray(
        (value - anchor_mean) / anchor_std, dtype=np.float32
    )
    target_train = normalize_target(target_train_raw)
    target_test = normalize_target(target_test_raw)
    anchor_train = normalize_anchor(anchor_train_raw)
    anchor_test = normalize_anchor(anchor_test_raw)
    if not all(
        np.isfinite(value).all()
        for value in (target_train, target_test, anchor_train, anchor_test)
    ):
        raise ValueError("Gaussian feature cache contains non-finite values")

    metadata = {
        "format_version": 1,
        "backend": "anisotropic_3d_gaussian_splat",
        "fold_manifest_fingerprint": manifest.fingerprint,
        "map_sha256": _sha256_file(dataset.map_path),
        "scene_cache": str(scene_path.resolve()),
        "scene_cache_sha256": _sha256_file(scene_path),
        "gaussian_count": scene.gaussian_count,
        "coordinate_frame": "original_metres",
        "scene_config": asdict(scene_config),
        "splat_config": asdict(splat_config),
        "target_feature_names": list(GaussianScene.LOCAL_FEATURE_NAMES)
        + [f"bs_path_alpha_{index:02d}" for index in range(16)],
        "anchor_feature_names": [
            f"anchor_path_alpha_{index:02d}" for index in range(16)
        ],
        "train_count": int(len(train_positions)),
        "test_count": int(len(test_positions)),
        "anchor_count_train": int(neighbors.shape[1]),
        "anchor_count_test": int(test_query.source_indices.shape[1]),
        "compatible_interface": "geometry_v2_32x16",
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        target_train=target_train,
        target_test=target_test,
        anchor_train=anchor_train,
        anchor_test=anchor_test,
        test_anchor_indices=test_query.source_indices.astype(np.int64),
        target_mean=target_mean,
        target_std=target_std,
        anchor_mean=anchor_mean,
        anchor_std=anchor_std,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return output
