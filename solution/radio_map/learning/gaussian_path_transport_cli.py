"""Train and deploy causal Gaussian-map path transport on a frozen coarse CSI."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from ..cli import validate_submission
from ..metrics import MetricAccumulator
from ..transforms import beam_delay, inverse_beam_delay
from .cli import _geometry, _loader
from .coarse_provider import CoarseSpec, FrozenCoarseProvider
from .dataset import CachedAnchorDataset, CoordinateBatchContext
from .gaussian_geometry import GaussianScene, _sha256_file
from .gaussian_path_transport import (
    GaussianPathTransport,
    GaussianPathTransportConfig,
)
from .gaussian_token_cache import GaussianTokenCache
from .path_transport_oracle_cli import fit_group_transport_parameters
from .path_transport import TransportParameters, apply_transport_numpy
from .torch_metrics import torch_competition_metrics


LABEL_NAMES = (
    "delta_h",
    "delta_v",
    "delta_delay",
    "log_amplitude",
    "phase_real",
    "phase_imag",
    "existence",
    "reliability",
)


def _json(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(name)


def _move(batch: dict[str, torch.Tensor], device: torch.device):
    return {name: value.to(device) for name, value in batch.items()}


def _coarse_spec(args: argparse.Namespace) -> CoarseSpec:
    return CoarseSpec(
        base_checkpoint=args.base_checkpoint,
        refiner_checkpoint=args.refiner_checkpoint,
        power_scale=args.power_scale,
        anchor_source=args.anchor_source,
        feature_version=args.feature_version,
        hidden_dim=args.refiner_hidden_dim,
        geometry_cache=args.geometry_cache,
    )


def _open_provider(args: argparse.Namespace):
    provider, manifest, dataset, adapter, train, validation = (
        FrozenCoarseProvider.load(
            _coarse_spec(args),
            args.data_dir,
            args.cache_dir,
            args.device,
        )
    )
    use_geometry = bool(provider.base.config.use_geometry)
    geometry = _geometry(args.geometry_cache, use_geometry)
    return (
        provider,
        manifest,
        dataset,
        adapter,
        train,
        validation,
        geometry,
    )


def _write_latents(
    path: Path,
    rows: int,
    width: int,
    batches,
    provider: FrozenCoarseProvider,
    device: torch.device,
    description: str,
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    values = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.complex64, shape=(rows, width)
    )
    offset = 0
    try:
        for source in tqdm(
            batches, desc=description, dynamic_ncols=True, unit="batch"
        ):
            batch = _move(
                {
                    name: value
                    for name, value in source.items()
                    if name
                    not in {"target_latent", "target_channel", "source_index"}
                },
                device,
            )
            latent = (
                provider.latent(batch)
                .detach()
                .cpu()
                .numpy()
                .astype(np.complex64, copy=False)
            )
            values[offset : offset + len(latent)] = latent
            offset += len(latent)
        if offset != rows:
            raise RuntimeError(f"{description} produced {offset}/{rows} rows")
        values.flush()
        del values
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _run_prepare(args: argparse.Namespace) -> int:
    device = _device(args.device)
    (
        provider,
        manifest,
        dataset,
        adapter,
        train,
        validation,
        geometry,
    ) = _open_provider(args)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("refusing to overwrite a non-empty O3 coarse cache")
    output.mkdir(parents=True, exist_ok=True)
    all_rows = np.arange(len(dataset.train_pos), dtype=np.int64)
    train_dataset = CachedAnchorDataset(
        args.cache_dir,
        all_rows,
        training=False,
        use_geometry=bool(provider.base.config.use_geometry),
        allow_cache_code_mismatch=True,
    )
    _write_latents(
        output / "coarse_train.npy",
        len(all_rows),
        adapter.coefficient_count,
        _loader(train_dataset, args.batch_size, False, args.seed),
        provider,
        device,
        "O3 frozen train coarse",
    )
    anchor_indices = (
        all_rows if args.anchor_source == "all_official_train" else train
    )
    context = CoordinateBatchContext(
        args.cache_dir,
        manifest,
        dataset,
        anchor_indices,
        use_geometry=bool(provider.base.config.use_geometry),
        geometry=geometry,
    )
    test_batches = (
        context.build(dataset.test_pos[start : start + args.batch_size])
        for start in range(0, len(dataset.test_pos), args.batch_size)
    )
    _write_latents(
        output / "coarse_test.npy",
        len(dataset.test_pos),
        adapter.coefficient_count,
        test_batches,
        provider,
        device,
        "O3 frozen test coarse",
    )

    coarse = np.load(output / "coarse_train.npy", mmap_mode="r")
    label_shape = (
        len(all_rows),
        dataset.config.m_p,
        dataset.config.n,
    )
    labels = {
        name: np.lib.format.open_memmap(
            output / f"{name}.npy",
            mode="w+",
            dtype=np.float32,
            shape=label_shape,
        )
        for name in LABEL_NAMES
    }
    for start in tqdm(
        range(0, len(all_rows), args.label_batch_size),
        desc="O3 transport supervision",
        dynamic_ncols=True,
        unit="batch",
    ):
        stop = min(len(all_rows), start + args.label_batch_size)
        source = adapter.beam_delay_numpy(np.asarray(coarse[start:stop]))
        target = beam_delay(
            dataset.channel_batch(all_rows[start:stop]), adapter.layout
        )
        fitted = fit_group_transport_parameters(
            source,
            target,
            args.max_h_shift,
            args.max_v_shift,
            args.max_delay_shift,
        )
        for name in LABEL_NAMES:
            labels[name][start:stop] = fitted[name]
    for values in labels.values():
        values.flush()
    del labels, coarse
    files = ["coarse_train.npy", "coarse_test.npy"] + [
        f"{name}.npy" for name in LABEL_NAMES
    ]
    report = {
        "format_version": 1,
        "kind": "gaussian_path_transport_supervision",
        "coarse_identity": provider.identity,
        "coarse_fingerprint": provider.fingerprint,
        "fold_fingerprint": manifest.fingerprint,
        "adapter_sha256": manifest.adapter_sha256,
        "train_indices_sha256": hashlib.sha256(
            np.asarray(train, dtype="<i8").tobytes()
        ).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(
            np.asarray(validation, dtype="<i8").tobytes()
        ).hexdigest(),
        "shape": {
            name: list(np.load(output / name, mmap_mode="r").shape)
            for name in files
        },
        "sha256": {name: _sha256_file(output / name) for name in files},
        "transport_limits": {
            "max_h_shift": args.max_h_shift,
            "max_v_shift": args.max_v_shift,
            "max_delay_shift": args.max_delay_shift,
        },
        "anchor_source_official_test": args.anchor_source,
    }
    _json(output / "prepare_report.json", report)
    return 0


def _load_supervision(path: str | Path, provider, manifest):
    source = Path(path)
    try:
        report = json.loads(
            (source / "prepare_report.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("invalid O3 supervision report") from error
    if (
        report.get("format_version") != 1
        or report.get("kind") != "gaussian_path_transport_supervision"
        or report.get("coarse_fingerprint") != provider.fingerprint
        or report.get("fold_fingerprint") != manifest.fingerprint
    ):
        raise ValueError("O3 supervision identity differs")
    arrays = {}
    for name, expected_hash in report["sha256"].items():
        file_path = source / name
        if _sha256_file(file_path) != expected_hash:
            raise ValueError(f"O3 supervision hash differs for {name}")
        arrays[name.removesuffix(".npy")] = np.load(
            file_path, mmap_mode="r", allow_pickle=False
        )
    return report, arrays


def _tokens(args, manifest, dataset):
    return GaussianTokenCache.load(
        args.gaussian_token_cache,
        manifest.fingerprint,
        _sha256_file(dataset.map_path),
    )


def _map_context_values(
    args,
    token_cache,
    dataset,
    split,
    mode,
    seed,
):
    if args.map_context == "target":
        return token_cache.values(split, mode, seed)
    if args.map_context not in ("anchor-delta", "multi-anchor"):
        raise ValueError("unknown map context")
    if args.map_anchor_count < 1:
        raise ValueError("map anchor count must be positive")
    target, target_mask = token_cache.values(split, "real", seed)
    train_tokens, train_mask = token_cache.values("train", "real", seed)
    if split == "train":
        neighbor_indices = np.asarray(
            np.load(
                Path(args.cache_dir) / "neighbor_indices.npy",
                mmap_mode="r",
                allow_pickle=False,
            ),
            dtype=np.int64,
        )
        neighbor_distances = np.asarray(
            np.load(
                Path(args.cache_dir) / "neighbor_distances.npy",
                mmap_mode="r",
                allow_pickle=False,
            ),
            dtype=np.float32,
        )
    else:
        if args.anchor_source == "all_official_train":
            pool = np.arange(len(dataset.train_pos), dtype=np.int64)
        else:
            pool = np.asarray(
                np.load(
                    Path(args.cache_dir) / "train_indices.npy",
                    allow_pickle=False,
                ),
                dtype=np.int64,
            )
        distance, local = cKDTree(dataset.train_pos[pool]).query(
            dataset.test_pos,
            k=min(args.map_anchor_count, len(pool)),
        )
        local = np.asarray(local, dtype=np.int64)
        distance = np.asarray(distance, dtype=np.float32)
        if local.ndim == 1:
            local = local[:, None]
            distance = distance[:, None]
        neighbor_indices = pool[local]
        neighbor_distances = distance
    if args.map_context == "anchor-delta":
        nearest = (
            neighbor_indices[:, 0]
            if neighbor_indices.ndim == 2
            else neighbor_indices
        )
        anchor = np.asarray(train_tokens[nearest])
        anchor_mask = np.asarray(train_mask[nearest])
        target_array = np.asarray(target)
        combined = np.concatenate(
            (target_array, anchor, target_array - anchor), axis=-1
        ).astype(np.float32, copy=False)
        combined_mask = np.asarray(target_mask) & anchor_mask
    else:
        count = min(args.map_anchor_count, neighbor_indices.shape[1])
        selected = neighbor_indices[:, :count]
        distances = neighbor_distances[:, :count]
        weights = 1.0 / np.maximum(distances, 1e-3)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
        target_array = np.asarray(target)
        target_role = np.ones(
            target_array.shape[:2] + (1,), dtype=np.float32
        )
        target_weight = np.ones_like(target_role)
        target_part = np.concatenate(
            (target_array, target_role, target_weight), axis=-1
        )
        anchor = np.asarray(train_tokens[selected])
        anchor_role = -np.ones(
            anchor.shape[:-1] + (1,), dtype=np.float32
        )
        anchor_weight = np.broadcast_to(
            weights[:, :, None, None],
            anchor.shape[:-1] + (1,),
        ).astype(np.float32)
        anchor_part = np.concatenate(
            (anchor, anchor_role, anchor_weight), axis=-1
        ).reshape(len(target_array), -1, target_part.shape[-1])
        combined = np.concatenate(
            (target_part, anchor_part), axis=1
        ).astype(np.float32, copy=False)
        combined_mask = np.concatenate(
            (
                np.asarray(target_mask),
                np.asarray(train_mask[selected]).reshape(
                    len(target_array), -1
                ),
            ),
            axis=1,
        )
    if mode == "real":
        return combined, combined_mask
    if mode == "zero":
        return np.zeros_like(combined), combined_mask
    if mode != "shuffle":
        raise ValueError("map mode must be real, zero, or shuffle")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(combined))
    if len(combined) > 1 and np.array_equal(
        permutation, np.arange(len(combined))
    ):
        permutation = np.roll(permutation, 1)
    return combined[permutation], combined_mask[permutation]


def _model_config(
    args, dataset, feature_dim, gaussian_count=0, radio_grid_sizes=()
):
    return GaussianPathTransportConfig(
        p_count=dataset.config.m_p,
        n_count=dataset.config.n,
        map_feature_dim=feature_dim,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        max_h_shift=args.max_h_shift,
        max_v_shift=args.max_v_shift,
        max_delay_shift=args.max_delay_shift,
        causal_map_residual=args.causal_map_residual,
        gaussian_count=int(gaussian_count),
        radio_feature_dim=int(args.radio_feature_dim),
        radio_grid_sizes=tuple(int(value) for value in radio_grid_sizes),
    )


def _radio_grid_mapping(args, token_cache, dataset):
    resolutions = tuple(float(value) for value in args.radio_grid_resolutions)
    if not resolutions:
        return (), None
    if args.gaussian_scene_cache is None:
        raise ValueError(
            "--gaussian-scene-cache is required with radio grid resolutions"
        )
    if (
        any(not math.isfinite(value) or value <= 0 for value in resolutions)
        or tuple(sorted(resolutions)) != resolutions
    ):
        raise ValueError(
            "radio grid resolutions must be positive and increasing"
        )
    scene = GaussianScene.load(args.gaussian_scene_cache)
    if scene.metadata.get("source_map_sha256") != _sha256_file(dataset.map_path):
        raise ValueError("Gaussian scene source map differs")
    count = token_cache.gaussian_count
    if count > scene.gaussian_count:
        raise ValueError("Gaussian token identity exceeds scene size")
    centers = np.asarray(scene.centers[:count], dtype=np.float64)
    origin = np.asarray(scene.centers, dtype=np.float64).min(axis=0)
    mapping = np.zeros((count + 1, len(resolutions)), dtype=np.int64)
    sizes = []
    for level, resolution in enumerate(resolutions):
        coordinates = np.floor(
            (centers - origin[None, :]) / resolution
        ).astype(np.int64)
        _, inverse = np.unique(
            coordinates, axis=0, return_inverse=True
        )
        mapping[1:, level] = inverse + 1
        sizes.append(int(inverse.max()) + 2)
    return tuple(sizes), torch.from_numpy(mapping)


def _parameter_loss(output, labels):
    parameters = output.parameters
    phase_norm = torch.sqrt(
        parameters.phase_real.square() + parameters.phase_imag.square()
    ).clamp_min(1e-6)
    phase_real = parameters.phase_real / phase_norm
    phase_imag = parameters.phase_imag / phase_norm
    phase = 1.0 - (
        phase_real * labels["phase_real"]
        + phase_imag * labels["phase_imag"]
    ).mean()
    terms = {
        "shift_h": F.smooth_l1_loss(
            parameters.delta_h / 2.0, labels["delta_h"] / 2.0
        ),
        "shift_v": F.smooth_l1_loss(
            parameters.delta_v / 2.0, labels["delta_v"] / 2.0
        ),
        "shift_delay": F.smooth_l1_loss(
            parameters.delta_delay / 8.0,
            labels["delta_delay"] / 8.0,
        ),
        "log_amplitude": F.smooth_l1_loss(
            parameters.log_amplitude,
            labels["log_amplitude"].clamp(-2.0, 2.0),
        ),
        "phase": phase,
        "reliability": F.mse_loss(
            parameters.reliability, labels["reliability"]
        ),
        "existence": F.mse_loss(
            parameters.existence, labels["existence"]
        ),
    }
    total = (
        terms["shift_h"]
        + terms["shift_v"]
        + terms["shift_delay"]
        + terms["log_amplitude"]
        + terms["phase"]
        + 0.5 * terms["reliability"]
        + 0.1 * terms["existence"]
    )
    return total, terms


def _direct_score_loss(
    output,
    target_channel,
    adapter,
    labels,
    *,
    scale,
    label_weight,
    trust_weight,
    nmse_objective,
):
    candidate = output.coarse + float(scale) * (
        output.transported - output.coarse
    )
    prediction = adapter.channel_from_beam_delay_torch(candidate)
    metrics = torch_competition_metrics(
        prediction,
        target_channel,
        adapter.layout.config,
        adapter.layout.order,
    )
    pas_loss = 0.4 * (1.0 - metrics.pas)
    pdp_loss = 0.4 * (1.0 - metrics.pdp)
    if nmse_objective == "log":
        nmse_loss = 0.2 * torch.log1p(metrics.nmse)
    elif nmse_objective == "official":
        nmse_loss = 0.2 * metrics.nmse / (1.0 + metrics.nmse)
    else:
        raise ValueError("nmse_objective must be log or official")
    parameter_loss, _ = _parameter_loss(output, labels)
    denominator = output.coarse.abs().square().mean().clamp_min(1e-8)
    trust = (
        (candidate - output.coarse).abs().square().mean() / denominator
    )
    total = (
        pas_loss
        + pdp_loss
        + nmse_loss
        + float(label_weight) * parameter_loss
        + float(trust_weight) * trust
    )
    return total, {
        "pas": metrics.pas,
        "pdp": metrics.pdp,
        "nmse": metrics.nmse,
        "score": metrics.score,
        "parameter_loss": parameter_loss,
        "trust": trust,
    }


def _rows_loader(indices: np.ndarray, batch_size: int, shuffle: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.as_tensor(indices, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


def _tensor_rows(array, rows, device, dtype=None):
    value = torch.from_numpy(np.asarray(array[rows]).copy())
    return value.to(device=device, dtype=dtype)


def _forward_rows(
    model,
    adapter,
    arrays,
    tokens,
    mask,
    rows,
    device,
    map_rows=None,
    gaussian_indices=None,
):
    selected = rows if map_rows is None else map_rows
    latent = _tensor_rows(arrays["coarse_train"], rows, device, torch.complex64)
    coarse = adapter.beam_delay_torch(latent)
    map_tokens = _tensor_rows(tokens, selected, device, torch.float32)
    map_mask = _tensor_rows(mask, selected, device, torch.bool)
    selected_indices = None
    if gaussian_indices is not None:
        selected_indices = _tensor_rows(
            gaussian_indices, selected, device, torch.long
        )
    return model(coarse, map_tokens, map_mask, selected_indices)


def _evaluate(
    model,
    adapter,
    dataset,
    arrays,
    token_cache,
    indices,
    device,
    batch_size,
    modes,
    scales,
    seed,
    args,
):
    results = {}
    model.eval()
    stage = (
        "O7"
        if model.config.radio_grid_sizes
        else ("O6" if model.config.gaussian_count else "O3")
    )
    for mode in modes:
        tokens, mask = _map_context_values(
            args, token_cache, dataset, "train", mode, seed
        )
        gaussian_indices = (
            token_cache.gaussian_indices("train", mode, seed)
            if model.config.gaussian_count
            else None
        )
        accumulators = {
            float(scale): MetricAccumulator(adapter.layout)
            for scale in scales
        }
        with torch.no_grad():
            for (row_tensor,) in tqdm(
                _rows_loader(indices, batch_size, False, seed),
                desc=f"{stage} validation {mode}",
                leave=False,
                dynamic_ncols=True,
            ):
                rows = row_tensor.numpy()
                output = _forward_rows(
                    model,
                    adapter,
                    arrays,
                    tokens,
                    mask,
                    rows,
                    device,
                    gaussian_indices=gaussian_indices,
                )
                coarse = output.coarse.detach().cpu().numpy()
                transported = output.transported.detach().cpu().numpy()
                target = dataset.channel_batch(rows)
                for scale, accumulator in accumulators.items():
                    candidate = coarse + scale * (transported - coarse)
                    accumulator.update(
                        inverse_beam_delay(candidate, adapter.layout), target
                    )
        results[mode] = {
            str(scale): accumulator.compute().to_dict()
            for scale, accumulator in accumulators.items()
        }
    return results


def _validation_parameter_loss(
    model,
    adapter,
    arrays,
    tokens,
    mask,
    gaussian_indices,
    indices,
    device,
    batch_size,
    seed,
):
    model.eval()
    total = 0.0
    seen = 0
    with torch.no_grad():
        for (row_tensor,) in _rows_loader(
            indices, batch_size, False, seed
        ):
            rows = row_tensor.numpy()
            output = _forward_rows(
                model,
                adapter,
                arrays,
                tokens,
                mask,
                rows,
                device,
                gaussian_indices=gaussian_indices,
            )
            labels = {
                name: _tensor_rows(
                    arrays[name], rows, device, torch.float32
                )
                for name in LABEL_NAMES
            }
            loss, _ = _parameter_loss(output, labels)
            total += float(loss) * len(rows)
            seen += len(rows)
    if not seen:
        raise ValueError("empty parameter validation split")
    return total / seen


def _checkpoint(
    path,
    model,
    config,
    provider,
    token_cache,
    metrics,
    scale,
    map_context,
    map_anchor_count,
):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "kind": "gaussian_path_transport",
        "model_config": asdict(config),
        "coarse_fingerprint": provider.fingerprint,
        "gaussian_token_fingerprint": token_cache.fingerprint,
        "map_context": str(map_context),
        "map_anchor_count": int(map_anchor_count),
        "state": model.state_dict(),
        "best_scale": float(scale),
        "metrics": metrics,
    }
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    destination.with_name(destination.name + ".sha256").write_text(
        _sha256_file(destination) + "\n", encoding="ascii"
    )


def _load_checkpoint(
    path,
    provider,
    token_cache,
    device,
    expected_map_context="target",
    expected_map_anchor_count=4,
):
    source = Path(path)
    sidecar = source.with_name(source.name + ".sha256")
    if (
        not source.is_file()
        or not sidecar.is_file()
        or sidecar.read_text(encoding="ascii").strip()
        != _sha256_file(source)
    ):
        raise ValueError("invalid O3 checkpoint hash")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != 1
        or payload.get("kind") != "gaussian_path_transport"
        or payload.get("coarse_fingerprint") != provider.fingerprint
        or payload.get("gaussian_token_fingerprint")
        != token_cache.fingerprint
        or payload.get("map_context", "target") != expected_map_context
        or int(payload.get("map_anchor_count", 4))
        != int(expected_map_anchor_count)
    ):
        raise ValueError("O3 checkpoint identity differs")
    model = GaussianPathTransport(
        GaussianPathTransportConfig(**payload["model_config"])
    ).to(device)
    model.load_state_dict(payload["state"])
    model.eval()
    return model, payload


def _run_train(args: argparse.Namespace) -> int:
    device = _device(args.device)
    _seed(args.seed)
    provider, manifest, dataset, adapter, train, validation, _ = _open_provider(
        args
    )
    _, arrays = _load_supervision(args.supervision_cache, provider, manifest)
    token_cache = _tokens(args, manifest, dataset)
    real_tokens, real_mask = _map_context_values(
        args, token_cache, dataset, "train", "real", args.seed
    )
    if args.radio_feature_dim and args.map_context != "target":
        raise ValueError(
            "learnable Gaussian radio field currently requires --map-context target"
        )
    radio_grid_sizes, gaussian_grid_ids = _radio_grid_mapping(
        args, token_cache, dataset
    )
    config = _model_config(
        args,
        dataset,
        real_tokens.shape[2],
        token_cache.gaussian_count if args.radio_feature_dim else 0,
        radio_grid_sizes,
    )
    normalization_mean, normalization_std = token_cache.normalization
    model = GaussianPathTransport(
        config,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        gaussian_grid_ids=gaussian_grid_ids,
    ).to(device)
    if args.init_checkpoint is not None:
        initialized, payload = _load_checkpoint(
            args.init_checkpoint,
            provider,
            token_cache,
            device,
            args.map_context,
            args.map_anchor_count,
        )
        if payload.get("model_config") != asdict(config):
            raise ValueError("initial checkpoint model config differs")
        model = initialized
    real_gaussian_indices = (
        token_cache.gaussian_indices("train", "real", args.seed)
        if config.gaussian_count
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    best_score = -math.inf
    best_selection = -math.inf
    bad_epochs = 0
    history = []
    stage = (
        "O7"
        if config.radio_grid_sizes
        else ("O6" if config.gaussian_count else "O3")
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        progress = tqdm(
            _rows_loader(train, args.batch_size, True, args.seed + epoch),
            desc=f"{stage} epoch {epoch}/{args.epochs}",
            dynamic_ncols=True,
        )
        for (row_tensor,) in progress:
            rows = row_tensor.numpy()
            optimizer.zero_grad(set_to_none=True)
            output = _forward_rows(
                model,
                adapter,
                arrays,
                real_tokens,
                real_mask,
                rows,
                device,
                gaussian_indices=real_gaussian_indices,
            )
            labels = {
                name: _tensor_rows(
                    arrays[name], rows, device, torch.float32
                )
                for name in LABEL_NAMES
            }
            if args.objective == "direct-score":
                target_channel = torch.from_numpy(
                    np.asarray(dataset.channel_batch(rows)).copy()
                ).to(device=device, dtype=torch.complex64)
                loss, train_terms = _direct_score_loss(
                    output,
                    target_channel,
                    adapter,
                    labels,
                    scale=args.training_scale,
                    label_weight=args.label_weight,
                    trust_weight=args.trust_weight,
                    nmse_objective=args.nmse_objective,
                )
                if args.causal_margin_weight > 0 and len(rows) > 1:
                    shuffled_output = _forward_rows(
                        model,
                        adapter,
                        arrays,
                        real_tokens,
                        real_mask,
                        rows,
                        device,
                        map_rows=np.roll(rows, 1),
                        gaussian_indices=real_gaussian_indices,
                    )
                    _, shuffled_terms = _direct_score_loss(
                        shuffled_output,
                        target_channel,
                        adapter,
                        labels,
                        scale=args.training_scale,
                        label_weight=0.0,
                        trust_weight=0.0,
                        nmse_objective=args.nmse_objective,
                    )
                    causal_margin = torch.relu(
                        args.causal_margin
                        - (
                            train_terms["score"]
                            - shuffled_terms["score"]
                        )
                    )
                    loss = loss + (
                        args.causal_margin_weight * causal_margin
                    )
                    train_terms["causal_margin"] = causal_margin
            else:
                loss, train_terms = _parameter_loss(output, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite O3 transport loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip_norm
            )
            optimizer.step()
            total_loss += float(loss.detach()) * len(rows)
            seen += len(rows)
            postfix = {"loss": f"{total_loss / seen:.4f}"}
            if "score" in train_terms:
                postfix["score"] = f"{float(train_terms['score'].detach()):.4f}"
            progress.set_postfix(**postfix)
        validation_metrics = _evaluate(
            model,
            adapter,
            dataset,
            arrays,
            token_cache,
            validation,
            device,
            args.batch_size,
            ("real",),
            args.validation_scales,
            args.seed,
            args,
        )["real"]
        best_scale, best_values = max(
            validation_metrics.items(), key=lambda item: item[1]["score"]
        )
        validation_parameter_loss = (
            _validation_parameter_loss(
                model,
                adapter,
                arrays,
                real_tokens,
                real_mask,
                real_gaussian_indices,
                validation,
                device,
                args.batch_size,
                args.seed,
            )
            if args.objective == "parameter"
            else None
        )
        record = {
            "epoch": epoch,
            "train_loss": total_loss / seen,
            "validation": validation_metrics,
            "best_scale": float(best_scale),
            "best_score": best_values["score"],
            "selection_metric": (
                "validation_parameter_loss"
                if args.objective == "parameter"
                else "validation_score"
            ),
            "validation_parameter_loss": validation_parameter_loss,
        }
        selection = (
            -validation_parameter_loss
            if args.objective == "parameter"
            else best_values["score"]
        )
        record["selection_value"] = selection
        best_score = max(best_score, best_values["score"])
        history.append(record)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            f"{stage} epoch {epoch}/{args.epochs} "
            f"loss={record['train_loss']:.6f} "
            f"val_score={best_values['score']:.6f} "
            f"scale={float(best_scale):g} best={best_score:.6f} "
            + (
                f"val_param={validation_parameter_loss:.6f}"
                if validation_parameter_loss is not None
                else "select=score"
            )
        )
        _checkpoint(
            run_dir / "last.pt",
            model,
            config,
            provider,
            token_cache,
            best_values,
            float(best_scale),
            args.map_context,
            args.map_anchor_count,
        )
        if selection > best_selection + args.min_delta:
            best_selection = selection
            bad_epochs = 0
            _checkpoint(
                run_dir / "best.pt",
                model,
                config,
                provider,
                token_cache,
                best_values,
                float(best_scale),
                args.map_context,
                args.map_anchor_count,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    _json(
        run_dir / "train_report.json",
        {
            "kind": (
                "gaussian_multiscale_radio_field_train"
                if config.radio_grid_sizes
                else (
                    "gaussian_radio_field_train"
                    if config.gaussian_count
                    else "gaussian_path_transport_train"
                )
            ),
            "epochs_completed": len(history),
            "best_score": best_score,
            "checkpoint_selection": (
                "minimum_validation_parameter_loss"
                if args.objective == "parameter"
                else "maximum_validation_score"
            ),
            "coarse_fingerprint": provider.fingerprint,
            "gaussian_token_fingerprint": token_cache.fingerprint,
            "model_config": asdict(config),
            "objective": args.objective,
            "training_scale": args.training_scale,
            "label_weight": args.label_weight,
            "trust_weight": args.trust_weight,
            "nmse_objective": args.nmse_objective,
            "causal_map_residual": args.causal_map_residual,
            "causal_margin": args.causal_margin,
            "causal_margin_weight": args.causal_margin_weight,
            "map_context": args.map_context,
            "map_anchor_count": args.map_anchor_count,
            "learnable_gaussian_radio_field": bool(config.gaussian_count),
            "radio_parameter_count": (
                (
                    sum(config.radio_grid_sizes)
                    if config.radio_grid_sizes
                    else config.gaussian_count + 1
                )
                * (config.radio_feature_dim + 1)
            ),
            "radio_grid_resolutions": list(args.radio_grid_resolutions),
            "init_checkpoint": (
                None
                if args.init_checkpoint is None
                else str(Path(args.init_checkpoint).resolve())
            ),
            "checkpoint": str((run_dir / "best.pt").resolve()),
        },
    )
    token_cache.close()
    return 0


def _run_oracle(args: argparse.Namespace) -> int:
    """Audit whether the persisted target-visible O2 labels are worth learning."""

    provider, manifest, dataset, adapter, _, validation, _ = _open_provider(
        args
    )
    _, arrays = _load_supervision(
        args.supervision_cache, provider, manifest
    )
    accumulators = {
        float(scale): MetricAccumulator(adapter.layout)
        for scale in args.validation_scales
    }
    for start in tqdm(
        range(0, len(validation), args.batch_size),
        desc="O3 label oracle",
        dynamic_ncols=True,
        unit="batch",
    ):
        rows = validation[start : start + args.batch_size]
        coarse = adapter.beam_delay_numpy(
            np.asarray(arrays["coarse_train"][rows])
        )
        parameters = TransportParameters(
            **{
                name: np.asarray(arrays[name][rows])
                for name in LABEL_NAMES
            }
        )
        transported = apply_transport_numpy(coarse, parameters)
        target = dataset.channel_batch(rows)
        for scale, accumulator in accumulators.items():
            candidate = coarse + scale * (transported - coarse)
            accumulator.update(
                inverse_beam_delay(candidate, adapter.layout), target
            )
    metrics = {
        str(scale): accumulator.compute().to_dict()
        for scale, accumulator in accumulators.items()
    }
    best_scale, best = max(
        metrics.items(), key=lambda item: item[1]["score"]
    )
    baseline = metrics.get("0.0") or metrics["0"]
    report = {
        "kind": "gaussian_path_transport_label_oracle",
        "target_visible": True,
        "deployable": False,
        "metrics": metrics,
        "best_scale": float(best_scale),
        "best": best,
        "gain": best["score"] - baseline["score"],
        "representation_promoted": bool(
            float(best_scale) >= args.minimum_scale
            and best["score"] - baseline["score"] >= args.minimum_gain
        ),
    }
    _json(args.output, report)
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    device = _device(args.device)
    provider, manifest, dataset, adapter, _, validation, _ = _open_provider(
        args
    )
    _, arrays = _load_supervision(args.supervision_cache, provider, manifest)
    token_cache = _tokens(args, manifest, dataset)
    model, payload = _load_checkpoint(
        args.checkpoint,
        provider,
        token_cache,
        device,
        args.map_context,
        args.map_anchor_count,
    )
    metrics = _evaluate(
        model,
        adapter,
        dataset,
        arrays,
        token_cache,
        validation,
        device,
        args.batch_size,
        ("real", "zero", "shuffle"),
        args.validation_scales,
        args.seed,
        args,
    )
    best = {
        mode: max(values.items(), key=lambda item: item[1]["score"])
        for mode, values in metrics.items()
    }
    baseline = metrics["real"].get("0.0") or metrics["real"].get("0")
    real_gain = best["real"][1]["score"] - baseline["score"]
    control_score = max(best["zero"][1]["score"], best["shuffle"][1]["score"])
    report = {
        "kind": (
            "gaussian_multiscale_radio_field_causal_evaluation"
            if model.config.radio_grid_sizes
            else (
                "gaussian_radio_field_causal_evaluation"
                if model.config.gaussian_count
                else "gaussian_path_transport_causal_evaluation"
            )
        ),
        "metrics": metrics,
        "best": {
            mode: {"scale": float(value[0]), **value[1]}
            for mode, value in best.items()
        },
        "baseline": baseline,
        "real_gain": real_gain,
        "real_over_best_control": best["real"][1]["score"] - control_score,
        "promoted": bool(
            real_gain >= args.minimum_gain
            and best["real"][1]["score"] - control_score
            >= args.minimum_control_margin
        ),
        "checkpoint_best_scale": payload["best_scale"],
    }
    _json(args.output, report)
    token_cache.close()
    return 0


def _run_infer(args: argparse.Namespace) -> int:
    device = _device(args.device)
    provider, manifest, dataset, adapter, _, _, _ = _open_provider(args)
    _, arrays = _load_supervision(args.supervision_cache, provider, manifest)
    token_cache = _tokens(args, manifest, dataset)
    model, payload = _load_checkpoint(
        args.checkpoint,
        provider,
        token_cache,
        device,
        args.map_context,
        args.map_anchor_count,
    )
    scale = payload["best_scale"] if args.scale is None else args.scale
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and non-negative")
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise ValueError("submission exists; use --overwrite deliberately")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    prediction = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.complex64,
        shape=(len(dataset.test_pos),) + dataset.config.channel_shape,
    )
    tokens, mask = _map_context_values(
        args, token_cache, dataset, "test", "real", args.seed
    )
    gaussian_indices = (
        token_cache.gaussian_indices("test", "real", args.seed)
        if model.config.gaussian_count
        else None
    )
    coarse_test = arrays["coarse_test"]
    try:
        model.eval()
        with torch.no_grad():
            for start in tqdm(
                range(0, len(dataset.test_pos), args.batch_size),
                desc=(
                    "O7 multiscale Gaussian Radio Field inference"
                    if model.config.radio_grid_sizes
                    else (
                        "O6 Gaussian Radio Field inference"
                        if model.config.gaussian_count
                        else "O3 official test inference"
                    )
                ),
                dynamic_ncols=True,
            ):
                stop = min(len(dataset.test_pos), start + args.batch_size)
                rows = np.arange(start, stop)
                latent = _tensor_rows(
                    coarse_test, rows, device, torch.complex64
                )
                coarse = adapter.beam_delay_torch(latent)
                candidate = model(
                    coarse,
                    _tensor_rows(tokens, rows, device, torch.float32),
                    _tensor_rows(mask, rows, device, torch.bool),
                    (
                        None
                        if gaussian_indices is None
                        else _tensor_rows(
                            gaussian_indices, rows, device, torch.long
                        )
                    ),
                ).transported
                blended = coarse + float(scale) * (candidate - coarse)
                prediction[start:stop] = inverse_beam_delay(
                    blended.cpu().numpy(), adapter.layout
                )
        prediction.flush()
        del prediction
        validate_submission(temporary, dataset, args.batch_size)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    _json(
        args.report or output.with_name("infer_report.json"),
        {
            "kind": (
                "gaussian_multiscale_radio_field_infer"
                if model.config.radio_grid_sizes
                else (
                    "gaussian_radio_field_infer"
                    if model.config.gaussian_count
                    else "gaussian_path_transport_infer"
                )
            ),
            "scale": float(scale),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "coarse_fingerprint": provider.fingerprint,
            "gaussian_token_fingerprint": token_cache.fingerprint,
            "submission": validate_submission(
                output, dataset, args.batch_size
            ),
        },
    )
    token_cache.close()
    return 0


def _common(parser):
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--refiner-checkpoint", required=True)
    parser.add_argument(
        "--geometry-cache",
        help="required automatically when the frozen base uses geometry",
    )
    parser.add_argument("--power-scale", type=float, default=1.25)
    parser.add_argument(
        "--anchor-source",
        choices=("fold", "all_official_train"),
        default="all_official_train",
    )
    parser.add_argument("--feature-version", type=int, default=1)
    parser.add_argument("--refiner-hidden-dim", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--map-context",
        choices=("target", "anchor-delta", "multi-anchor"),
        default="target",
    )
    parser.add_argument("--map-anchor-count", type=int, default=4)


def _model(parser):
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max-h-shift", type=float, default=2.0)
    parser.add_argument("--max-v-shift", type=float, default=2.0)
    parser.add_argument("--max-delay-shift", type=float, default=8.0)
    parser.add_argument(
        "--causal-map-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--radio-feature-dim",
        type=int,
        default=0,
        help=(
            "positive value enables per-scene-Gaussian trainable radio "
            "features and opacity, e.g. 32"
        ),
    )
    parser.add_argument(
        "--gaussian-scene-cache",
        help="fixed anisotropic Gaussian scene used to build shared radio grids",
    )
    parser.add_argument(
        "--radio-grid-resolutions",
        type=float,
        nargs="*",
        default=(),
        help="increasing metre resolutions, e.g. 4 8 16",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    _common(prepare)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--label-batch-size", type=int, default=4)
    prepare.add_argument("--max-h-shift", type=int, default=2)
    prepare.add_argument("--max-v-shift", type=int, default=2)
    prepare.add_argument("--max-delay-shift", type=int, default=8)
    prepare.set_defaults(handler=_run_prepare)

    train = commands.add_parser("train")
    _common(train)
    _model(train)
    train.add_argument("--supervision-cache", required=True)
    train.add_argument("--gaussian-token-cache", required=True)
    train.add_argument("--run-dir", required=True)
    train.add_argument(
        "--init-checkpoint",
        help="strictly compatible checkpoint used to initialize this stage",
    )
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--gradient-clip-norm", type=float, default=1.0)
    train.add_argument(
        "--objective",
        choices=("direct-score", "parameter"),
        default="direct-score",
    )
    train.add_argument("--training-scale", type=float, default=0.75)
    train.add_argument("--label-weight", type=float, default=0.05)
    train.add_argument("--trust-weight", type=float, default=0.001)
    train.add_argument("--causal-margin", type=float, default=0.002)
    train.add_argument(
        "--causal-margin-weight", type=float, default=0.2
    )
    train.add_argument(
        "--nmse-objective",
        choices=("log", "official"),
        default="log",
    )
    train.add_argument("--patience", type=int, default=8)
    train.add_argument("--min-delta", type=float, default=1e-5)
    train.add_argument(
        "--validation-scales",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    train.set_defaults(handler=_run_train)

    oracle = commands.add_parser("oracle")
    _common(oracle)
    oracle.add_argument("--supervision-cache", required=True)
    oracle.add_argument("--output", required=True)
    oracle.add_argument(
        "--validation-scales",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    oracle.add_argument("--minimum-gain", type=float, default=0.015)
    oracle.add_argument("--minimum-scale", type=float, default=0.25)
    oracle.set_defaults(handler=_run_oracle)

    evaluate = commands.add_parser("evaluate")
    _common(evaluate)
    evaluate.add_argument("--supervision-cache", required=True)
    evaluate.add_argument("--gaussian-token-cache", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument(
        "--validation-scales",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    evaluate.add_argument("--minimum-gain", type=float, default=0.005)
    evaluate.add_argument(
        "--minimum-control-margin", type=float, default=0.003
    )
    evaluate.set_defaults(handler=_run_evaluate)

    infer = commands.add_parser("infer")
    _common(infer)
    infer.add_argument("--supervision-cache", required=True)
    infer.add_argument("--gaussian-token-cache", required=True)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--scale", type=float)
    infer.add_argument("--output", required=True)
    infer.add_argument("--report")
    infer.add_argument("--overwrite", action="store_true")
    infer.set_defaults(handler=_run_infer)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
