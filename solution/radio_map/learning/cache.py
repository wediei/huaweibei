"""Leakage-safe, fingerprinted on-disk fold caches for anchor learning."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from ..anchors import AnchorMemory
from ..data import RoundDataset
from ..geometry import GeometryPrior
from ..splits import SplitIndices
from ..transforms import AntennaLayout
from .latent_adapter import FixedSupportLatentAdapter


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


_CACHE_SCHEMA_VERSION = "fold-cache-schema-v3"
_CODE_FINGERPRINT_SOURCES = (
    "radio_map/learning/cache.py", "radio_map/learning/dataset.py", "radio_map/learning/latent_adapter.py",
    "radio_map/learning/anchor_mixer.py", "radio_map/learning/model_components.py", "radio_map/anchors.py",
    "radio_map/geometry.py", "radio_map/transforms.py", "radio_map/codecs.py", "radio_map/data.py",
    "radio_map/splits.py", "radio_map/config.py",
)


def _code_fingerprint() -> str:
    """Version the complete artifact producer/consumer contract, not one module."""

    root = Path(__file__).resolve().parents[2]
    payload = {
        "schema": _CACHE_SCHEMA_VERSION,
        "sources": {relative: _sha256_file(root / relative) for relative in _CODE_FINGERPRINT_SOURCES},
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


@dataclass(frozen=True)
class FoldCacheConfig:
    layout_order: tuple[str, str, str] = ("P", "H", "V")
    support_fraction: float = 0.10
    delay_block: int = 8
    k_max: int = 64
    anchor_count: int = 32
    patch: int = 33
    corridor: int = 32
    anchor_corridor: int = 8
    encode_batch_size: int = 16
    dropout: float = 0.1
    min_anchors: int = 8
    seed: int = 42
    code_version: str = "task-7-schema-v3"

    def __post_init__(self) -> None:
        if tuple(self.layout_order) not in {("H", "V", "P"), ("H", "P", "V"), ("V", "H", "P"), ("V", "P", "H"), ("P", "H", "V"), ("P", "V", "H")}:
            raise ValueError("layout_order must be a permutation of H, V, P")
        for name in ("delay_block", "k_max", "anchor_count", "patch", "corridor", "anchor_corridor", "encode_batch_size", "min_anchors"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.patch % 2 != 1:
            raise ValueError("patch must be odd")
        if not 0 < self.support_fraction <= 1:
            raise ValueError("support_fraction must be in (0, 1]")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be a non-bool integer")
        if self.anchor_count > self.k_max:
            raise ValueError("anchor_count cannot exceed k_max")


@dataclass(frozen=True)
class FoldCacheManifest:
    """Self-authenticating JSON metadata; the fingerprint excludes itself."""

    fingerprint: str
    data_dir: str
    channel_sha256: str
    ply_sha256: str
    source_file_sha256: dict[str, str]
    train_indices_sha256: str
    validation_indices_sha256: str
    adapter_config: dict[str, Any]
    adapter_sha256: str
    support_sha256: str
    geometry_sha256: str
    normalization_sha256: str
    normalization_hashes: dict[str, str]
    layout_order: list[str]
    k_max: int
    anchor_count: int
    patch: int
    corridor: int
    anchor_corridor: int
    dropout: float
    min_anchors: int
    seed: int
    shape: dict[str, list[int]]
    dtype: dict[str, str]
    numpy_version: str
    torch_version: str
    code_fingerprint: str
    code_version: str
    artifact_sha256: dict[str, str]

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("manifest seed must be a non-bool integer")
        if set(self.artifact_sha256) != set(self.shape):
            raise ValueError("manifest artifact hashes must exactly match shaped artifacts")
        for name, value in self.artifact_sha256.items():
            if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"manifest artifact hash is invalid for {name}")

    def _without_fingerprint(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value.pop("fingerprint")
        return value

    @classmethod
    def create(cls, **kwargs: Any) -> "FoldCacheManifest":
        empty = cls(fingerprint="", **kwargs)
        return dataclasses.replace(empty, fingerprint=hashlib.sha256(_canonical(empty._without_fingerprint())).hexdigest())

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(_canonical(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> "FoldCacheManifest":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            return cls(**raw)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("invalid cache manifest") from error


_REQUIRED_FILES = (
    "manifest.json", "adapter.npz", "normalization.npz", "latents.npy", "neighbor_indices.npy",
    "neighbor_distances.npy", "pair_features.npy", "target_point_features.npy", "target_patches.npy",
    "bs_patch.npy", "bs_target_corridors.npy", "anchor_corridors.npy", "train_indices.npy",
    "validation_indices.npy",
)


def validate_cache(
    manifest: FoldCacheManifest,
    cache_dir: str | Path,
    *,
    allow_code_mismatch: bool = False,
) -> None:
    """Reject tampering, incomplete caches, and shape/dtype substitutions.

    ``allow_code_mismatch`` is reserved for read-only consumers of an
    authenticated historical cache.  It does not relax manifest, artifact,
    source-data, adapter, normalization, runtime, shape, dtype, or split
    checks.
    """

    cache = Path(cache_dir)
    expected = hashlib.sha256(_canonical(manifest._without_fingerprint())).hexdigest()
    if manifest.fingerprint != expected:
        raise ValueError("cache fingerprint does not match its manifest")
    missing = [name for name in _REQUIRED_FILES if not (cache / name).is_file()]
    if missing:
        raise ValueError("cache fingerprint cannot be trusted: missing " + ", ".join(missing))
    persisted = FoldCacheManifest.load(cache / "manifest.json")
    if persisted.fingerprint != manifest.fingerprint or persisted.to_dict() != manifest.to_dict():
        raise ValueError("cache fingerprint differs from manifest.json")
    if (
        manifest.code_fingerprint != _code_fingerprint()
        and not allow_code_mismatch
    ):
        raise ValueError("cache fingerprint invalid: code fingerprint differs")
    if manifest.numpy_version != np.__version__ or manifest.torch_version != torch.__version__:
        raise ValueError("cache fingerprint invalid: runtime version differs")
    for name, shape in manifest.shape.items():
        if _sha256_file(cache / name) != manifest.artifact_sha256[name]:
            raise ValueError(f"cache fingerprint invalid: artifact hash differs for {name}")
        array = np.load(cache / name, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != shape or str(array.dtype) != manifest.dtype[name]:
            raise ValueError(f"cache fingerprint invalid: {name} shape or dtype differs")
    if _sha256_file(cache / "adapter.npz") != manifest.adapter_sha256:
        raise ValueError("cache fingerprint invalid: adapter hash differs")
    if _sha256_file(cache / "normalization.npz") != manifest.normalization_sha256:
        raise ValueError("cache fingerprint invalid: normalization hash differs")
    data_dir = Path(manifest.data_dir)
    for name, expected_hash in manifest.source_file_sha256.items():
        source = data_dir / name
        if not source.is_file() or _sha256_file(source) != expected_hash:
            raise ValueError(f"cache fingerprint invalid: source hash differs for {name}")
    with np.load(cache / "adapter.npz", allow_pickle=False) as adapter:
        if _sha256_array(adapter["support_indices"].astype("<i8", copy=False)) != manifest.support_sha256:
            raise ValueError("cache fingerprint invalid: support hash differs")
    train = np.load(cache / "train_indices.npy", mmap_mode="r", allow_pickle=False)
    validation = np.load(cache / "validation_indices.npy", mmap_mode="r", allow_pickle=False)
    if (
        train.ndim != 1 or validation.ndim != 1 or not len(train)
        or train.dtype.kind not in "iu" or validation.dtype.kind not in "iu"
        or _sha256_array(np.asarray(train, dtype="<i8")) != manifest.train_indices_sha256
        or _sha256_array(np.asarray(validation, dtype="<i8")) != manifest.validation_indices_sha256
        or np.intersect1d(train, validation).size
    ):
        raise ValueError("cache fingerprint invalid: persisted split indices differ")


def _geometry_hash(geometry: GeometryPrior) -> str:
    metadata = {key: _jsonable(value) for key, value in geometry.metadata.items()}
    digest = hashlib.sha256()
    digest.update(_sha256_array(np.asarray(geometry.features, dtype=np.float32)).encode())
    digest.update(_canonical({"names": list(geometry.feature_names), "origin": geometry.origin_xy, "resolution": geometry.resolution, "metadata": metadata}))
    return digest.hexdigest()


def _mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
    return values.mean(axis=0).astype(np.float32), values.std(axis=0).clip(1e-6).astype(np.float32)


def _normalization(geometry: GeometryPrior, positions: np.ndarray, train: np.ndarray, bs: np.ndarray, pair_train: np.ndarray, corridor_lengths: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    names = geometry.feature_names
    occupancy = np.asarray([name == "occupancy" or name.endswith("_occupancy") for name in names], dtype=bool)
    geometry_values = geometry.sample_points(positions[train])
    geometry_mean, geometry_std = _mean_std(geometry_values)
    geometry_mean[occupancy] = 0.0
    geometry_std[occupancy] = 1.0
    pos_mean, pos_std = _mean_std(positions[train])
    rel_mean, rel_std = _mean_std(positions[train] - bs)
    pair_mean, pair_std = _mean_std(pair_train)
    length_mean, length_std = _mean_std(corridor_lengths.reshape(-1, 1))
    return {
        "position_mean": pos_mean, "position_std": pos_std,
        "bs_relative_mean": rel_mean, "bs_relative_std": rel_std,
        "geometry_mean": geometry_mean, "geometry_std": geometry_std,
        "geometry_occupancy": occupancy.astype(np.uint8),
        "pair_mean": pair_mean, "pair_std": pair_std,
        "corridor_length_mean": length_mean, "corridor_length_std": length_std,
    }, occupancy


def _normalize_geometry(values: np.ndarray, stats: dict[str, np.ndarray], occupancy: np.ndarray) -> np.ndarray:
    output = (np.asarray(values, dtype=np.float32) - stats["geometry_mean"].reshape((-1,) + (1,) * (np.asarray(values).ndim - 1)))
    output = output / stats["geometry_std"].reshape((-1,) + (1,) * (np.asarray(values).ndim - 1))
    output[occupancy] = np.asarray(values, dtype=np.float32)[occupancy]
    return output.astype(np.float32, copy=False)


def _finite_float16(values: np.ndarray) -> np.ndarray:
    """The geometry cache is float16, so persist bounded finite values only."""

    limit = np.finfo(np.float16).max
    return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=limit, neginf=-limit).clip(-limit, limit).astype(np.float16)


def quantize_geometry_values(values: np.ndarray) -> np.ndarray:
    """Match cache storage exactly: finite clipping and float16 round-trip."""

    return _finite_float16(values).astype(np.float32)


def _normalization_hashes(stats: dict[str, np.ndarray]) -> dict[str, str]:
    return {name: _sha256_array(value) for name, value in sorted(stats.items())}


def prepare_fold_cache(dataset: RoundDataset, geometry: GeometryPrior, split: SplitIndices, config: FoldCacheConfig, output_dir: str | Path) -> Path:
    """Fit fold-only state and write all large arrays as independently mmap-able NPY files."""

    output = Path(output_dir).resolve()
    train = np.asarray(split.train, dtype=np.int64)
    validation = np.asarray(split.validation, dtype=np.int64)
    p = int(dataset.train_pos.shape[0])
    if train.ndim != 1 or validation.ndim != 1 or not len(train) or np.intersect1d(train, validation).size:
        raise ValueError("split must contain nonempty disjoint train and validation indices")
    if np.any(train < 0) or np.any(validation < 0) or np.any(train >= p) or np.any(validation >= p):
        raise IndexError("split index outside dataset")
    if config.k_max > len(train) - 1:
        raise ValueError("k_max exceeds train anchor pool after target exclusion")
    layout = AntennaLayout(dataset.config, tuple(config.layout_order))
    adapter = FixedSupportLatentAdapter(layout, config.support_fraction, config.delay_block)
    adapter.fit(dataset, train, config.encode_batch_size)

    # Dummy values preserve AnchorMemory's tested tie/exclusion implementation without loading train latents.
    memory = AnchorMemory(dataset.train_pos[train], np.zeros((len(train), 1), np.complex64), np.asarray(dataset.config.bs_position), train)
    query = memory.query(dataset.train_pos, config.k_max, exclude_source_indices=np.arange(p, dtype=np.int64))
    pair_train = query.pair_features[train]
    bs = np.asarray(dataset.config.bs_position, dtype=np.float64)
    lengths = np.concatenate((
        np.linalg.norm(dataset.train_pos[train] - bs, axis=1),
        np.asarray(query.distances[train], dtype=np.float64).reshape(-1),
    ))
    stats, occupancy = _normalization(geometry, dataset.train_pos, train, bs, pair_train, lengths)

    source_hashes = {name: _sha256_file(dataset.data_dir / name) for name in ("Round1_Setup.json", "Round1_Train_Pos.npy", "Round1_Test_Pos.npy", "Round1_Train_Channel.npy", "Round1_Map.ply")}
    geometry_sha = _geometry_hash(geometry)
    code_sha = _code_fingerprint()
    shape = {
        "latents.npy": [p, adapter.coefficient_count], "neighbor_indices.npy": [p, config.k_max],
        "neighbor_distances.npy": [p, config.k_max], "pair_features.npy": [p, config.k_max, 14],
        "target_point_features.npy": [p, 13], "target_patches.npy": [p, 13, config.patch, config.patch],
        "bs_patch.npy": [13, config.patch, config.patch], "bs_target_corridors.npy": [p, config.corridor, 15],
        "anchor_corridors.npy": [p, config.k_max, config.anchor_corridor, 15],
        "train_indices.npy": [int(len(train))], "validation_indices.npy": [int(len(validation))],
    }
    dtype = {"latents.npy": "complex64", "neighbor_indices.npy": "int64", "neighbor_distances.npy": "float32", "pair_features.npy": "float32", "target_point_features.npy": "float32", "target_patches.npy": "float16", "bs_patch.npy": "float16", "bs_target_corridors.npy": "float16", "anchor_corridors.npy": "float16", "train_indices.npy": "int64", "validation_indices.npy": "int64"}
    # The normalization archive hash is represented by the deterministic arrays before it is written.
    stat_hashes = _normalization_hashes(stats)
    adapter_config = {"support_fraction": config.support_fraction, "delay_block": config.delay_block, "layout_order": list(config.layout_order), "fit_indices_sha256": adapter.fit_indices_sha256}
    # Existing caches must be exactly the cache this invocation would make.
    if output.exists():
        existing = FoldCacheManifest.load(output / "manifest.json") if (output / "manifest.json").is_file() else None
        if existing is None:
            raise ValueError("refusing to overwrite an existing cache directory")
        validate_cache(existing, output)
        comparable = {"data_dir": str(dataset.data_dir), "channel_sha256": source_hashes["Round1_Train_Channel.npy"], "ply_sha256": source_hashes["Round1_Map.ply"], "source_file_sha256": source_hashes, "train_indices_sha256": _sha256_array(train.astype("<i8")), "validation_indices_sha256": _sha256_array(validation.astype("<i8")), "adapter_config": adapter_config, "support_sha256": _sha256_array(adapter.support_indices.astype("<i8")), "geometry_sha256": geometry_sha, "normalization_hashes": stat_hashes, "layout_order": list(config.layout_order), "k_max": config.k_max, "anchor_count": config.anchor_count, "patch": config.patch, "corridor": config.corridor, "anchor_corridor": config.anchor_corridor, "dropout": config.dropout, "min_anchors": config.min_anchors, "seed": config.seed, "shape": shape, "dtype": dtype, "numpy_version": np.__version__, "torch_version": torch.__version__, "code_fingerprint": code_sha, "code_version": config.code_version}
        for key, value in comparable.items():
            if getattr(existing, key) != value:
                raise ValueError("existing cache fingerprint is incompatible with requested fold")
        return output
    output.mkdir(parents=True)
    adapter.save(output / "adapter.npz")
    np.savez_compressed(output / "normalization.npz", **stats)
    normalization_sha = _sha256_file(output / "normalization.npz")
    np.save(output / "train_indices.npy", train.astype(np.int64, copy=False))
    np.save(output / "validation_indices.npy", validation.astype(np.int64, copy=False))

    latent_out = np.lib.format.open_memmap(output / "latents.npy", mode="w+", dtype=np.complex64, shape=tuple(shape["latents.npy"]))
    for start in tqdm(
        range(0, p, config.encode_batch_size),
        total=(p + config.encode_batch_size - 1) // config.encode_batch_size,
        desc="encode channel latents",
        dynamic_ncols=True,
    ):
        stop = min(p, start + config.encode_batch_size)
        latent_out[start:stop] = adapter.encode_numpy(dataset.channel_batch(np.arange(start, stop)))
    latent_out.flush(); del latent_out
    indices_out = np.lib.format.open_memmap(output / "neighbor_indices.npy", mode="w+", dtype=np.int64, shape=tuple(shape["neighbor_indices.npy"])); indices_out[:] = query.source_indices; indices_out.flush(); del indices_out
    distance_out = np.lib.format.open_memmap(output / "neighbor_distances.npy", mode="w+", dtype=np.float32, shape=tuple(shape["neighbor_distances.npy"])); distance_out[:] = query.distances; distance_out.flush(); del distance_out
    pair_out = np.lib.format.open_memmap(output / "pair_features.npy", mode="w+", dtype=np.float32, shape=tuple(shape["pair_features.npy"])); pair_out[:] = (query.pair_features - stats["pair_mean"]) / stats["pair_std"]; pair_out.flush(); del pair_out
    point_out = np.lib.format.open_memmap(output / "target_point_features.npy", mode="w+", dtype=np.float32, shape=tuple(shape["target_point_features.npy"]))
    patch_out = np.lib.format.open_memmap(output / "target_patches.npy", mode="w+", dtype=np.float16, shape=tuple(shape["target_patches.npy"]))
    corridor_out = np.lib.format.open_memmap(output / "bs_target_corridors.npy", mode="w+", dtype=np.float16, shape=tuple(shape["bs_target_corridors.npy"]))
    anchor_corridor_out = np.lib.format.open_memmap(output / "anchor_corridors.npy", mode="w+", dtype=np.float16, shape=tuple(shape["anchor_corridors.npy"]))
    bs_patch = quantize_geometry_values(_normalize_geometry(geometry.extract_patch(bs, config.patch), stats, occupancy))
    bs_out = np.lib.format.open_memmap(output / "bs_patch.npy", mode="w+", dtype=np.float16, shape=tuple(shape["bs_patch.npy"])); bs_out[:] = bs_patch; bs_out.flush(); del bs_out
    for row in tqdm(
        range(p),
        desc="cache geometry features",
        dynamic_ncols=True,
    ):
        point_out[row] = quantize_geometry_values(_normalize_geometry(geometry.sample_points(dataset.train_pos[row:row + 1]).T, stats, occupancy)).reshape(13)
        patch_out[row] = quantize_geometry_values(_normalize_geometry(geometry.extract_patch(dataset.train_pos[row], config.patch), stats, occupancy))
        target_corridor = geometry.corridor_features(bs, dataset.train_pos[row], config.corridor)
        target_corridor[:, 1] = (target_corridor[:, 1] - stats["corridor_length_mean"][0]) / stats["corridor_length_std"][0]
        target_corridor[:, 2:] = _normalize_geometry(target_corridor[:, 2:].T, stats, occupancy).T
        corridor_out[row] = quantize_geometry_values(target_corridor)
        for anchor_slot, anchor_id in enumerate(query.source_indices[row]):
            path = geometry.corridor_features(dataset.train_pos[anchor_id], dataset.train_pos[row], config.anchor_corridor)
            path[:, 1] = (path[:, 1] - stats["corridor_length_mean"][0]) / stats["corridor_length_std"][0]
            path[:, 2:] = _normalize_geometry(path[:, 2:].T, stats, occupancy).T
            anchor_corridor_out[row, anchor_slot] = quantize_geometry_values(path)
    point_out.flush(); patch_out.flush(); corridor_out.flush(); anchor_corridor_out.flush()
    del point_out, patch_out, corridor_out, anchor_corridor_out
    artifact_sha256 = {name: _sha256_file(output / name) for name in shape}
    manifest = FoldCacheManifest.create(data_dir=str(dataset.data_dir), channel_sha256=source_hashes["Round1_Train_Channel.npy"], ply_sha256=source_hashes["Round1_Map.ply"], source_file_sha256=source_hashes, train_indices_sha256=_sha256_array(train.astype("<i8")), validation_indices_sha256=_sha256_array(validation.astype("<i8")), adapter_config=adapter_config, adapter_sha256=_sha256_file(output / "adapter.npz"), support_sha256=_sha256_array(adapter.support_indices.astype("<i8")), geometry_sha256=geometry_sha, normalization_sha256=normalization_sha, normalization_hashes=stat_hashes, layout_order=list(config.layout_order), k_max=config.k_max, anchor_count=config.anchor_count, patch=config.patch, corridor=config.corridor, anchor_corridor=config.anchor_corridor, dropout=config.dropout, min_anchors=config.min_anchors, seed=config.seed, shape=shape, dtype=dtype, numpy_version=np.__version__, torch_version=torch.__version__, code_fingerprint=code_sha, code_version=config.code_version, artifact_sha256=artifact_sha256)
    manifest.save(output / "manifest.json")
    validate_cache(manifest, output)
    return output
