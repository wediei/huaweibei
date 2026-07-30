"""Train/evaluate per-anchor Gaussian transport before anchor aggregation."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from ..cli import validate_submission
from ..metrics import MetricAccumulator
from ..transforms import inverse_beam_delay
from .gaussian_anchor_transport import (
    GaussianAnchorTransport,
    GaussianAnchorTransportConfig,
)
from .gaussian_anchor_path_cache import GaussianAnchorPathCache
from .gaussian_geometry import _sha256_file
from .gaussian_path_transport_cli import (
    _device,
    _json,
    _load_supervision,
    _open_provider,
    _rows_loader,
    _seed,
    _tensor_rows,
    _tokens,
)
from .torch_metrics import torch_competition_metrics


def _direct_cache(args, manifest, dataset):
    if args.anchor_path_cache is None:
        return None
    return GaussianAnchorPathCache.load(
        args.anchor_path_cache,
        fold_fingerprint=manifest.fingerprint,
        map_sha256=_sha256_file(dataset.map_path),
        anchor_count=args.anchor_count,
        anchor_source=args.anchor_source,
    )


def _anchor_arrays(args, dataset, split: str, direct_cache=None):
    latents = np.load(
        Path(args.cache_dir) / "latents.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    if direct_cache is not None:
        _, _, indices, distances = direct_cache.values(split)
        return latents, indices, distances
    if split == "train":
        indices = np.load(
            Path(args.cache_dir) / "neighbor_indices.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        distances = np.load(
            Path(args.cache_dir) / "neighbor_distances.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        return latents, indices[:, : args.anchor_count], distances[:, : args.anchor_count]
    if args.anchor_source == "all_official_train":
        pool = np.arange(len(dataset.train_pos), dtype=np.int64)
    else:
        pool = np.asarray(
            np.load(Path(args.cache_dir) / "train_indices.npy"),
            dtype=np.int64,
        )
    distance, local = cKDTree(dataset.train_pos[pool]).query(
        dataset.test_pos, k=min(args.anchor_count, len(pool))
    )
    local = np.asarray(local, dtype=np.int64)
    distance = np.asarray(distance, dtype=np.float32)
    if local.ndim == 1:
        local = local[:, None]
        distance = distance[:, None]
    return latents, pool[local], distance


def _pair_tokens(
    token_cache,
    direct_cache,
    neighbor_indices,
    rows,
    map_rows,
    split: str,
    mode: str,
):
    train_tokens, train_mask = token_cache.values("train", "real")
    target_tokens, target_mask = token_cache.values(
        "train" if split == "train" else "test", "real"
    )
    map_neighbors = np.asarray(neighbor_indices[map_rows], dtype=np.int64)
    target = np.asarray(target_tokens[map_rows], dtype=np.float32)
    anchors = np.asarray(train_tokens[map_neighbors], dtype=np.float32)
    expanded = np.broadcast_to(
        target[:, None, :, :],
        (len(rows), anchors.shape[1], target.shape[1], target.shape[2]),
    )
    pairs = np.concatenate((expanded, anchors, expanded - anchors), axis=-1)
    masks = (
        np.asarray(target_mask[map_rows])[:, None, :]
        & np.asarray(train_mask[map_neighbors])
    )
    if direct_cache is not None:
        direct_tokens, direct_mask, direct_indices, _ = direct_cache.values(
            split
        )
        if not np.array_equal(
            np.asarray(direct_indices[map_rows]), map_neighbors
        ):
            raise ValueError("direct Gaussian paths differ from map anchors")
        direct = np.asarray(direct_tokens[map_rows], dtype=np.float32)
        pairs = np.concatenate((pairs, direct), axis=-1)
        masks &= np.asarray(direct_mask[map_rows])
    if mode == "zero":
        pairs = np.zeros_like(pairs)
    elif mode != "real":
        raise ValueError("pair mode must be real or zero")
    return pairs.astype(np.float32, copy=False), masks


def _forward(
    model,
    adapter,
    arrays,
    token_cache,
    direct_cache,
    anchor_latents,
    neighbor_indices,
    neighbor_distances,
    rows,
    map_rows,
    split,
    mode,
    device,
):
    coarse_latent = _tensor_rows(
        arrays["coarse_train" if split == "train" else "coarse_test"],
        rows,
        device,
        torch.complex64,
    )
    coarse = adapter.beam_delay_torch(coarse_latent)
    selected = np.asarray(neighbor_indices[rows], dtype=np.int64)
    source_latents = _tensor_rows(
        anchor_latents, selected, device, torch.complex64
    )
    batch, anchors, width = source_latents.shape
    anchor_beam = adapter.beam_delay_torch(
        source_latents.reshape(batch * anchors, width)
    ).reshape(batch, anchors, *coarse.shape[1:])
    pairs, pair_mask = _pair_tokens(
        token_cache,
        direct_cache,
        neighbor_indices,
        rows,
        map_rows,
        split,
        mode,
    )
    distances = _tensor_rows(
        neighbor_distances, rows, device, torch.float32
    )
    anchor_mask = torch.isfinite(distances)
    return model(
        coarse,
        anchor_beam,
        torch.from_numpy(pairs).to(device),
        torch.from_numpy(pair_mask).to(device),
        distances,
        anchor_mask,
    )


def _loss(output, target, adapter, scale: float, trust_weight: float):
    candidate = output.coarse + float(scale) * (
        output.transported - output.coarse
    )
    prediction = adapter.channel_from_beam_delay_torch(candidate)
    metrics = torch_competition_metrics(
        prediction,
        target,
        adapter.layout.config,
        adapter.layout.order,
    )
    trust = (
        (candidate - output.coarse).abs().square().mean()
        / output.coarse.abs().square().mean().clamp_min(1e-8)
    )
    loss = (
        0.4 * (1.0 - metrics.pas)
        + 0.4 * (1.0 - metrics.pdp)
        + 0.2 * torch.log1p(metrics.nmse)
        + float(trust_weight) * trust
    )
    return loss, metrics.score, trust


def _evaluate(
    model,
    adapter,
    dataset,
    arrays,
    token_cache,
    direct_cache,
    anchor_latents,
    neighbor_indices,
    neighbor_distances,
    indices,
    args,
    device,
    modes=("real",),
):
    model.eval()
    results = {}
    for mode in modes:
        map_indices = np.asarray(indices, dtype=np.int64).copy()
        map_mode = mode
        if mode == "shuffle":
            map_indices = np.random.default_rng(args.seed).permutation(map_indices)
            map_mode = "real"
        accumulators = {
            float(scale): MetricAccumulator(adapter.layout)
            for scale in args.validation_scales
        }
        with torch.no_grad():
            for start in tqdm(
                range(0, len(indices), args.batch_size),
                desc=f"O4 validation {mode}",
                leave=False,
                dynamic_ncols=True,
            ):
                rows = np.asarray(indices[start : start + args.batch_size])
                maps = map_indices[start : start + len(rows)]
                output = _forward(
                    model,
                    adapter,
                    arrays,
                    token_cache,
                    direct_cache,
                    anchor_latents,
                    neighbor_indices,
                    neighbor_distances,
                    rows,
                    maps,
                    "train",
                    map_mode,
                    device,
                )
                coarse = output.coarse.cpu().numpy()
                moved = output.transported.cpu().numpy()
                target = dataset.channel_batch(rows)
                for scale, accumulator in accumulators.items():
                    candidate = coarse + scale * (moved - coarse)
                    accumulator.update(
                        inverse_beam_delay(candidate, adapter.layout), target
                    )
        results[mode] = {
            str(scale): accumulator.compute().to_dict()
            for scale, accumulator in accumulators.items()
        }
    return results


def _save_checkpoint(
    path, model, config, provider, token_cache, direct_cache, args, scale, metrics
):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "kind": "gaussian_anchor_transport",
        "model_config": asdict(config),
        "coarse_fingerprint": provider.fingerprint,
        "gaussian_token_fingerprint": token_cache.fingerprint,
        "anchor_path_fingerprint": (
            None if direct_cache is None else direct_cache.fingerprint
        ),
        "anchor_count": args.anchor_count,
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


def _load_checkpoint(path, provider, token_cache, direct_cache, args, device):
    source = Path(path)
    sidecar = source.with_name(source.name + ".sha256")
    if (
        not source.is_file()
        or not sidecar.is_file()
        or sidecar.read_text(encoding="ascii").strip() != _sha256_file(source)
    ):
        raise ValueError("invalid O4 checkpoint hash")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if (
        payload.get("format_version") != 1
        or payload.get("kind") != "gaussian_anchor_transport"
        or payload.get("coarse_fingerprint") != provider.fingerprint
        or payload.get("gaussian_token_fingerprint") != token_cache.fingerprint
        or payload.get("anchor_path_fingerprint") != (
            None if direct_cache is None else direct_cache.fingerprint
        )
        or int(payload.get("anchor_count", -1)) != args.anchor_count
    ):
        raise ValueError("O4 checkpoint identity differs")
    model = GaussianAnchorTransport(
        GaussianAnchorTransportConfig(**payload["model_config"])
    ).to(device)
    model.load_state_dict(payload["state"])
    model.eval()
    return model, payload


def _run_train(args):
    device = _device(args.device)
    _seed(args.seed)
    provider, manifest, dataset, adapter, train, validation, _ = _open_provider(args)
    _, arrays = _load_supervision(args.supervision_cache, provider, manifest)
    token_cache = _tokens(args, manifest, dataset)
    direct_cache = _direct_cache(args, manifest, dataset)
    anchor_latents, neighbor_indices, neighbor_distances = _anchor_arrays(
        args, dataset, "train", direct_cache
    )
    config = GaussianAnchorTransportConfig(
        p_count=dataset.config.m_p,
        n_count=dataset.config.n,
        map_feature_dim=(
            3 * len(token_cache.feature_names)
            + (0 if direct_cache is None else len(direct_cache.feature_names))
        ),
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        learned_anchor_fusion=args.learned_anchor_fusion,
    )
    model = GaussianAnchorTransport(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    best_score, bad_epochs, history = -math.inf, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = score_sum = margin_sum = 0.0
        seen = 0
        progress = tqdm(
            _rows_loader(train, args.batch_size, True, args.seed + epoch),
            desc=f"O4 epoch {epoch}/{args.epochs}",
            dynamic_ncols=True,
        )
        for (row_tensor,) in progress:
            rows = row_tensor.numpy()
            optimizer.zero_grad(set_to_none=True)
            output = _forward(
                model, adapter, arrays, token_cache, direct_cache, anchor_latents,
                neighbor_indices, neighbor_distances, rows, rows, "train",
                "real", device,
            )
            target = torch.from_numpy(
                np.asarray(dataset.channel_batch(rows)).copy()
            ).to(device=device, dtype=torch.complex64)
            loss, score, _ = _loss(
                output, target, adapter, args.training_scale, args.trust_weight
            )
            shuffled = _forward(
                model, adapter, arrays, token_cache, direct_cache, anchor_latents,
                neighbor_indices, neighbor_distances, rows, np.roll(rows, 1),
                "train", "real", device,
            )
            _, shuffled_score, _ = _loss(
                shuffled, target, adapter, args.training_scale, 0.0
            )
            margin = torch.relu(
                args.causal_margin - (score - shuffled_score)
            )
            loss = loss + args.causal_margin_weight * margin
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite O4 loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip_norm
            )
            optimizer.step()
            count = len(rows)
            loss_sum += float(loss.detach()) * count
            score_sum += float(score.detach()) * count
            margin_sum += float(margin.detach()) * count
            seen += count
            progress.set_postfix(
                loss=f"{loss_sum / seen:.4f}",
                score=f"{score_sum / seen:.4f}",
                margin=f"{margin_sum / seen:.4f}",
            )
        values = _evaluate(
            model, adapter, dataset, arrays, token_cache, direct_cache, anchor_latents,
            neighbor_indices, neighbor_distances, validation, args, device,
        )["real"]
        best_scale, best_values = max(
            values.items(), key=lambda item: item[1]["score"]
        )
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train_score": score_sum / seen,
            "causal_margin": margin_sum / seen,
            "validation": values,
            "best_scale": float(best_scale),
            "best_score": best_values["score"],
        }
        history.append(record)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            f"O4 epoch {epoch}/{args.epochs} loss={record['train_loss']:.6f} "
            f"train_score={record['train_score']:.6f} "
            f"margin={record['causal_margin']:.6f} "
            f"val_score={best_values['score']:.6f} scale={float(best_scale):g}"
        )
        _save_checkpoint(
            run_dir / "last.pt", model, config, provider, token_cache,
            direct_cache, args,
            best_scale, best_values,
        )
        if best_values["score"] > best_score + args.min_delta:
            best_score, bad_epochs = best_values["score"], 0
            _save_checkpoint(
                run_dir / "best.pt", model, config, provider, token_cache,
                direct_cache, args, best_scale, best_values,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    _json(
        run_dir / "train_report.json",
        {
            "kind": "gaussian_anchor_transport_train",
            "epochs_completed": len(history),
            "best_score": best_score,
            "anchor_count": args.anchor_count,
            "anchor_path_fingerprint": (
                None if direct_cache is None else direct_cache.fingerprint
            ),
            "model_config": asdict(config),
            "checkpoint": str((run_dir / "best.pt").resolve()),
        },
    )
    token_cache.close()
    if direct_cache is not None:
        direct_cache.close()
    return 0


def _run_evaluate(args):
    device = _device(args.device)
    provider, manifest, dataset, adapter, _, validation, _ = _open_provider(args)
    _, arrays = _load_supervision(args.supervision_cache, provider, manifest)
    token_cache = _tokens(args, manifest, dataset)
    direct_cache = _direct_cache(args, manifest, dataset)
    model, payload = _load_checkpoint(
        args.checkpoint, provider, token_cache, direct_cache, args, device
    )
    anchor_latents, indices, distances = _anchor_arrays(
        args, dataset, "train", direct_cache
    )
    metrics = _evaluate(
        model, adapter, dataset, arrays, token_cache, direct_cache,
        anchor_latents, indices,
        distances, validation, args, device, ("real", "zero", "shuffle"),
    )
    best = {
        mode: max(values.items(), key=lambda item: item[1]["score"])
        for mode, values in metrics.items()
    }
    baseline = metrics["real"].get("0.0") or metrics["real"]["0"]
    gain = best["real"][1]["score"] - baseline["score"]
    control = max(best["zero"][1]["score"], best["shuffle"][1]["score"])
    report = {
        "kind": "gaussian_anchor_transport_causal_evaluation",
        "metrics": metrics,
        "best": {
            mode: {"scale": float(value[0]), **value[1]}
            for mode, value in best.items()
        },
        "baseline": baseline,
        "real_gain": gain,
        "real_over_best_control": best["real"][1]["score"] - control,
        "promoted": bool(
            gain >= args.minimum_gain
            and best["real"][1]["score"] - control
            >= args.minimum_control_margin
        ),
        "checkpoint_best_scale": payload["best_scale"],
    }
    _json(args.output, report)
    token_cache.close()
    if direct_cache is not None:
        direct_cache.close()
    return 0


def _run_infer(args):
    device = _device(args.device)
    provider, manifest, dataset, adapter, _, _, _ = _open_provider(args)
    _, arrays = _load_supervision(args.supervision_cache, provider, manifest)
    token_cache = _tokens(args, manifest, dataset)
    direct_cache = _direct_cache(args, manifest, dataset)
    model, payload = _load_checkpoint(
        args.checkpoint, provider, token_cache, direct_cache, args, device
    )
    scale = payload["best_scale"] if args.scale is None else args.scale
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and non-negative")
    anchor_latents, indices, distances = _anchor_arrays(
        args, dataset, "test", direct_cache
    )
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise ValueError("submission exists; use --overwrite deliberately")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    prediction = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.complex64,
        shape=(len(dataset.test_pos),) + dataset.config.channel_shape,
    )
    try:
        model.eval()
        with torch.no_grad():
            for start in tqdm(
                range(0, len(dataset.test_pos), args.batch_size),
                desc="O4 official test inference",
                dynamic_ncols=True,
            ):
                rows = np.arange(start, min(start + args.batch_size, len(dataset.test_pos)))
                result = _forward(
                    model, adapter, arrays, token_cache, direct_cache,
                    anchor_latents,
                    indices, distances, rows, rows, "test", "real", device,
                )
                candidate = result.coarse + float(scale) * (
                    result.transported - result.coarse
                )
                prediction[rows] = inverse_beam_delay(
                    candidate.cpu().numpy(), adapter.layout
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
            "kind": "gaussian_anchor_transport_infer",
            "scale": float(scale),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "anchor_count": args.anchor_count,
            "submission": validate_submission(output, dataset, args.batch_size),
        },
    )
    token_cache.close()
    if direct_cache is not None:
        direct_cache.close()
    return 0


def _common(parser):
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--refiner-checkpoint", required=True)
    parser.add_argument("--geometry-cache")
    parser.add_argument("--power-scale", type=float, default=1.25)
    parser.add_argument(
        "--anchor-source",
        choices=("fold", "all_official_train"),
        default="all_official_train",
    )
    parser.add_argument("--feature-version", type=int, default=1)
    parser.add_argument("--refiner-hidden-dim", type=int, default=64)
    parser.add_argument("--supervision-cache", required=True)
    parser.add_argument("--gaussian-token-cache", required=True)
    parser.add_argument(
        "--anchor-path-cache",
        help="optional direct Anchor-to-Target Gaussian path cache",
    )
    parser.add_argument("--anchor-count", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    _common(train)
    train.add_argument("--run-dir", required=True)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--learning-rate", type=float, default=5e-5)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--gradient-clip-norm", type=float, default=1.0)
    train.add_argument("--training-scale", type=float, default=0.75)
    train.add_argument("--trust-weight", type=float, default=0.001)
    train.add_argument("--causal-margin", type=float, default=0.003)
    train.add_argument("--causal-margin-weight", type=float, default=1.0)
    train.add_argument("--d-model", type=int, default=96)
    train.add_argument("--heads", type=int, default=4)
    train.add_argument("--layers", type=int, default=2)
    train.add_argument(
        "--learned-anchor-fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    train.add_argument("--patience", type=int, default=8)
    train.add_argument("--min-delta", type=float, default=1e-5)
    train.add_argument(
        "--validation-scales", type=float, nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    train.set_defaults(handler=_run_train)

    evaluate = commands.add_parser("evaluate")
    _common(evaluate)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument(
        "--validation-scales", type=float, nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    evaluate.add_argument("--minimum-gain", type=float, default=0.01)
    evaluate.add_argument("--minimum-control-margin", type=float, default=0.003)
    evaluate.set_defaults(handler=_run_evaluate)

    infer = commands.add_parser("infer")
    _common(infer)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--report")
    infer.add_argument("--scale", type=float)
    infer.add_argument("--overwrite", action="store_true")
    infer.set_defaults(handler=_run_infer)
    return parser


def main(argv: Sequence[str] | None = None):
    args = build_parser().parse_args(argv)
    if args.anchor_count < 1:
        raise ValueError("anchor-count must be positive")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
