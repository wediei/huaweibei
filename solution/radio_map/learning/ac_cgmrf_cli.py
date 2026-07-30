"""Train, evaluate, gate, and conditionally infer AC-CGMRF beside frozen O4.1."""

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
from tqdm.auto import tqdm

from ..cli import validate_submission
from ..data import RoundDataset
from ..geometry import GeometryPrior
from ..metrics import MetricAccumulator
from ..splits import SplitIndices
from ..transforms import inverse_beam_delay
from .ac_cgmrf import ACCGMRF, ACCGMRFConfig, ACCGMRFOutput
from .ac_cgmrf_loss import ACCGMRFLossConfig, ac_cgmrf_loss
from .ac_cgmrf_validation import (
    aggregate_cross_validation,
    audit_anchor_exclusion,
    pilot_gate,
    require_promotion_report,
    spatial_block_folds,
    spatial_fold_manifest,
)
from .cache import FoldCacheConfig, FoldCacheManifest, prepare_fold_cache, validate_cache
from .dataset import load_persisted_split
from .gaussian_anchor_transport_cli import (
    _anchor_arrays,
    _direct_cache,
    _forward as _o41_forward,
    _load_checkpoint as _load_o41_checkpoint,
)
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


TASK_ID = "020-ac-cgmrf"


def _atomic_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    path.with_name(path.name + ".sha256").write_text(
        _sha256_file(path) + "\n", encoding="ascii"
    )


def _validate_sidecar(path: str | Path, label: str) -> str:
    source = Path(path)
    sidecar = source.with_name(source.name + ".sha256")
    if not source.is_file() or not sidecar.is_file():
        raise ValueError(f"{label} checkpoint or SHA-256 sidecar is missing")
    digest = _sha256_file(source)
    if sidecar.read_text(encoding="ascii").strip() != digest:
        raise ValueError(f"{label} checkpoint SHA-256 sidecar differs")
    return digest


def _open_stack(args):
    provider, manifest, dataset, adapter, train, validation, _ = _open_provider(args)
    _, arrays = _load_supervision(args.supervision_cache, provider, manifest)
    token_cache = _tokens(args, manifest, dataset)
    direct_cache = _direct_cache(args, manifest, dataset)
    if direct_cache is None:
        token_cache.close()
        raise ValueError("AC-CGMRF requires --anchor-path-cache")
    o41_model, o41_payload = _load_o41_checkpoint(
        args.o41_checkpoint,
        provider,
        token_cache,
        direct_cache,
        args,
        _device(args.device),
    )
    o41_model.requires_grad_(False)
    o41_model.eval()
    o41_scale = (
        float(o41_payload["best_scale"])
        if args.o41_scale is None
        else float(args.o41_scale)
    )
    if not math.isfinite(o41_scale) or o41_scale < 0:
        token_cache.close()
        direct_cache.close()
        raise ValueError("O4.1 scale must be finite and non-negative")
    return (
        provider,
        manifest,
        dataset,
        adapter,
        np.asarray(train, dtype=np.int64),
        np.asarray(validation, dtype=np.int64),
        arrays,
        token_cache,
        direct_cache,
        o41_model,
        o41_payload,
        o41_scale,
    )


def _condition_arrays(
    token_cache,
    direct_cache,
    neighbor_indices: np.ndarray,
    rows: np.ndarray,
    map_rows: np.ndarray,
    split: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    train_tokens, train_mask = token_cache.values("train", "real")
    target_tokens, target_mask = token_cache.values(split, "real")
    map_neighbors = np.asarray(neighbor_indices[map_rows], dtype=np.int64)
    target = np.asarray(target_tokens[map_rows], dtype=np.float32)
    anchors = np.asarray(train_tokens[map_neighbors], dtype=np.float32)
    direct_tokens, direct_mask, direct_indices, _ = direct_cache.values(split)
    if not np.array_equal(
        np.asarray(direct_indices[map_rows], dtype=np.int64), map_neighbors
    ):
        raise ValueError("Anchor-to-Target paths differ from selected map Anchors")
    direct = np.asarray(direct_tokens[map_rows], dtype=np.float32)
    mask = (
        np.asarray(target_mask[map_rows], dtype=bool)[:, None, :]
        & np.asarray(train_mask[map_neighbors], dtype=bool)
        & np.asarray(direct_mask[map_rows], dtype=bool)
    )
    if anchors.shape[-1] != target.shape[-1] or direct.shape[-1] != target.shape[-1]:
        raise ValueError("Gaussian path feature dimensions differ")
    return {
        "target_path_tokens": torch.from_numpy(target.copy()).to(device),
        "anchor_path_tokens": torch.from_numpy(anchors.copy()).to(device),
        "anchor_target_tokens": torch.from_numpy(direct.copy()).to(device),
        "path_mask": torch.from_numpy(mask.copy()).to(device),
    }


def _frozen_batch(
    o41_model,
    o41_scale: float,
    adapter,
    dataset,
    arrays,
    token_cache,
    direct_cache,
    anchor_latents,
    neighbor_indices,
    neighbor_distances,
    rows: np.ndarray,
    split: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        output = _o41_forward(
            o41_model,
            adapter,
            arrays,
            token_cache,
            direct_cache,
            anchor_latents,
            neighbor_indices,
            neighbor_distances,
            rows,
            rows,
            split,
            "real",
            device,
        )
        base = output.coarse + float(o41_scale) * (
            output.transported - output.coarse
        )
        reliability = output.parameters.reliability
        if output.anchor_weights.ndim == 2:
            reliability = (
                reliability
                * output.anchor_weights[:, :, None, None]
            ).sum(dim=1).mean(dim=(-2, -1))
        else:
            reliability = (
                reliability * output.anchor_weights
            ).sum(dim=1).mean(dim=(-2, -1))
        selected = np.asarray(neighbor_indices[rows], dtype=np.int64)
        source_latents = _tensor_rows(
            anchor_latents, selected, device, torch.complex64
        )
        batch, anchors, width = source_latents.shape
        anchor_beam = adapter.beam_delay_torch(
            source_latents.reshape(batch * anchors, width)
        ).reshape(batch, anchors, *base.shape[1:])
    target_positions = (
        dataset.train_pos[rows] if split == "train" else dataset.test_pos[rows]
    )
    return {
        "base_beam_delay": base.detach(),
        "anchor_beam_delay": anchor_beam.detach(),
        "anchor_positions": torch.from_numpy(
            np.asarray(dataset.train_pos[selected], dtype=np.float32).copy()
        ).to(device),
        "target_positions": torch.from_numpy(
            np.asarray(target_positions, dtype=np.float32).copy()
        ).to(device),
        "anchor_mask": torch.from_numpy(
            np.isfinite(np.asarray(neighbor_distances[rows])).copy()
        ).to(device),
        "base_reliability": reliability.detach().to(
            dtype=base.real.dtype
        ),
    }


def _model_output(
    model: ACCGMRF,
    frozen: dict[str, torch.Tensor],
    conditions: dict[str, torch.Tensor],
    mode: str,
    progress: float,
) -> ACCGMRFOutput:
    return model(
        **conditions,
        mode=mode,
        base_beam_delay=frozen["base_beam_delay"],
        anchor_beam_delay=frozen["anchor_beam_delay"],
        anchor_positions=frozen["anchor_positions"],
        target_positions=frozen["target_positions"],
        anchor_mask=frozen["anchor_mask"],
        base_reliability=frozen["base_reliability"],
        progress=progress,
    )


def _model_config(args, dataset, token_cache) -> ACCGMRFConfig:
    return ACCGMRFConfig(
        p_count=dataset.config.m_p,
        n_count=dataset.config.n,
        path_feature_dim=len(token_cache.feature_names),
        atom_count=args.atom_count,
        d_model=args.d_model,
        map_hidden_dim=args.map_hidden_dim,
        support_h=args.support_h,
        support_v=args.support_v,
        support_delay=args.support_delay,
        max_residual_ratio=args.max_residual_ratio,
        support_threshold=args.support_threshold,
    )


def _loss_config(args) -> ACCGMRFLossConfig:
    return ACCGMRFLossConfig(
        complex_weight=args.complex_weight,
        pas_weight=args.pas_weight,
        pdp_weight=args.pdp_weight,
        score_weight=args.score_weight,
        energy_weight=args.energy_weight,
        trust_weight=args.trust_weight,
        sparsity_weight=args.sparsity_weight,
        coupling_weight=args.coupling_weight,
        phase_continuity_weight=args.phase_continuity_weight,
        causal_margin_weight=args.causal_margin_weight,
        zero_margin=args.zero_margin,
        shuffle_margin=args.shuffle_margin,
    )


def _identity(model, frozen, conditions) -> float:
    with torch.no_grad():
        initial = _model_output(model, frozen, conditions, "real", 1.0)
        disabled = model(
            **conditions,
            mode="real",
            base_beam_delay=frozen["base_beam_delay"],
            anchor_beam_delay=frozen["anchor_beam_delay"],
            anchor_positions=frozen["anchor_positions"],
            target_positions=frozen["target_positions"],
            anchor_mask=frozen["anchor_mask"],
            base_reliability=frozen["base_reliability"],
            enabled=False,
            progress=1.0,
        )
        zero = _model_output(model, frozen, conditions, "zero", 1.0)
    disabled_error = (
        disabled.prediction - frozen["base_beam_delay"]
    ).abs().max()
    zero_error = (zero.prediction - frozen["base_beam_delay"]).abs().max()
    initial_error = (
        initial.prediction - frozen["base_beam_delay"]
    ).abs().max()
    return float(
        torch.maximum(initial_error, torch.maximum(disabled_error, zero_error)).cpu()
    )


def _save_checkpoint(
    path: Path,
    model: ACCGMRF,
    config: ACCGMRFConfig,
    provider,
    token_cache,
    direct_cache,
    args,
    o41_sha256: str,
    o41_scale: float,
    best_scale: float,
    metrics: dict[str, object],
) -> None:
    payload = {
        "format_version": 1,
        "kind": "ac_cgmrf",
        "task_id": TASK_ID,
        "model_config": asdict(config),
        "state": model.state_dict(),
        "coarse_fingerprint": provider.fingerprint,
        "gaussian_token_fingerprint": token_cache.fingerprint,
        "anchor_path_fingerprint": direct_cache.fingerprint,
        "o41_checkpoint_sha256": o41_sha256,
        "o41_scale": float(o41_scale),
        "anchor_count": int(args.anchor_count),
        "training_stage": args.stage,
        "best_scale": float(best_scale),
        "metrics": metrics,
    }
    _atomic_checkpoint(path, payload)


def _load_checkpoint(
    path: str | Path,
    provider,
    token_cache,
    direct_cache,
    args,
    device,
    o41_sha256: str,
    o41_scale: float,
):
    _validate_sidecar(path, "AC-CGMRF")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("format_version") != 1
        or payload.get("kind") != "ac_cgmrf"
        or payload.get("task_id") != TASK_ID
        or payload.get("coarse_fingerprint") != provider.fingerprint
        or payload.get("gaussian_token_fingerprint") != token_cache.fingerprint
        or payload.get("anchor_path_fingerprint") != direct_cache.fingerprint
        or payload.get("o41_checkpoint_sha256") != o41_sha256
        or float(payload.get("o41_scale", math.nan)) != float(o41_scale)
        or int(payload.get("anchor_count", -1)) != args.anchor_count
    ):
        raise ValueError("AC-CGMRF checkpoint identity differs from frozen inputs")
    model = ACCGMRF(ACCGMRFConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["state"])
    model.eval()
    return model, payload


def _evaluate(
    model,
    adapter,
    dataset,
    arrays,
    token_cache,
    direct_cache,
    o41_model,
    o41_scale,
    anchor_latents,
    neighbor_indices,
    neighbor_distances,
    indices,
    args,
    device,
) -> dict[str, object]:
    model.eval()
    rows_all = np.asarray(indices, dtype=np.int64)
    if len(rows_all) < 2:
        raise ValueError("causal validation needs at least two targets")
    shuffled_all = np.roll(rows_all, 1)
    metrics = {
        mode: {
            float(scale): MetricAccumulator(adapter.layout)
            for scale in args.validation_scales
        }
        for mode in ("real", "zero", "shuffle")
    }
    diagnostic_sums = {
        mode: {"support_novelty": 0.0, "trust_abs": 0.0, "energy_ratio": 0.0}
        for mode in metrics
    }
    seen = 0
    with torch.no_grad():
        for start in tqdm(
            range(0, len(rows_all), args.batch_size),
            desc="AC-CGMRF validation batches",
            leave=False,
            dynamic_ncols=True,
        ):
            rows = rows_all[start : start + args.batch_size]
            map_rows = shuffled_all[start : start + len(rows)]
            frozen = _frozen_batch(
                o41_model,
                o41_scale,
                adapter,
                dataset,
                arrays,
                token_cache,
                direct_cache,
                anchor_latents,
                neighbor_indices,
                neighbor_distances,
                rows,
                "train",
                device,
            )
            real_conditions = _condition_arrays(
                token_cache,
                direct_cache,
                neighbor_indices,
                rows,
                rows,
                "train",
                device,
            )
            shuffle_conditions = _condition_arrays(
                token_cache,
                direct_cache,
                neighbor_indices,
                rows,
                map_rows,
                "train",
                device,
            )
            outputs = {
                "real": _model_output(model, frozen, real_conditions, "real", 1.0),
                "zero": _model_output(model, frozen, real_conditions, "zero", 1.0),
                "shuffle": _model_output(
                    model, frozen, shuffle_conditions, "real", 1.0
                ),
            }
            target = dataset.channel_batch(rows)
            for mode, output in outputs.items():
                count = len(rows)
                diagnostic_sums[mode]["support_novelty"] += (
                    float(output.support_novelty.sum().cpu())
                )
                diagnostic_sums[mode]["trust_abs"] += float(
                    output.trust.abs().sum().cpu()
                )
                diagnostic_sums[mode]["energy_ratio"] += float(
                    output.diagnostics["residual_energy_ratio"].sum().cpu()
                )
                base = frozen["base_beam_delay"]
                for scale, accumulator in metrics[mode].items():
                    candidate = base + float(scale) * output.residual
                    accumulator.update(
                        inverse_beam_delay(candidate.cpu().numpy(), adapter.layout),
                        target,
                    )
            seen += len(rows)
    metric_values = {
        mode: {
            str(scale): accumulator.compute().to_dict()
            for scale, accumulator in values.items()
        }
        for mode, values in metrics.items()
    }
    diagnostics = {
        mode: {
            key: value / seen for key, value in sums.items()
        }
        for mode, sums in diagnostic_sums.items()
    }
    return {"metrics": metric_values, "diagnostics": diagnostics}


def _evaluation_report(raw: dict[str, object]) -> dict[str, object]:
    metrics = raw["metrics"]
    best = {
        mode: {
            "scale": float(scale),
            **values,
        }
        for mode, mode_values in metrics.items()
        for scale, values in [
            max(mode_values.items(), key=lambda item: item[1]["score"])
        ]
    }
    baseline = metrics["real"].get("0.0") or metrics["real"].get("0")
    if baseline is None:
        raise ValueError("validation scales must include 0 for frozen O4.1")
    report = {
        "kind": "ac_cgmrf_fold_evaluation",
        **raw,
        "best": best,
        "baseline": baseline,
        "support_novelty": raw["diagnostics"]["real"]["support_novelty"],
    }
    report["pilot_gate"] = pilot_gate(report)
    return report


def _run_train(args) -> int:
    if args.batch_size < 2:
        raise ValueError("batch-size must be at least two for shuffle control")
    device = _device(args.device)
    _seed(args.seed)
    promotion = None
    if args.stage == "full":
        if args.promotion_report is None:
            raise ValueError("full training requires --promotion-report")
        promotion = require_promotion_report(args.promotion_report)
    (
        provider,
        manifest,
        dataset,
        adapter,
        train,
        validation,
        arrays,
        token_cache,
        direct_cache,
        o41_model,
        _,
        o41_scale,
    ) = _open_stack(args)
    o41_sha256 = _validate_sidecar(args.o41_checkpoint, "O4.1")
    anchor_latents, neighbor_indices, neighbor_distances = _anchor_arrays(
        args, dataset, "train", direct_cache
    )
    leakage = audit_anchor_exclusion(train, validation, neighbor_indices)
    if not leakage.passed:
        raise ValueError("fold cache permits validation target/Anchor leakage")
    training_indices = (
        np.arange(len(dataset.train_pos), dtype=np.int64)
        if args.stage == "full"
        else train
    )
    config = _model_config(args, dataset, token_cache)
    model = ACCGMRF(config).to(device)
    loss_config = _loss_config(args)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        raise ValueError("run-dir already has metrics.jsonl; choose a new run directory")
    _json(
        run_dir / "config.json",
        {
            "task_id": TASK_ID,
            "model": asdict(config),
            "loss": asdict(loss_config),
            "arguments": {
                key: value
                for key, value in vars(args).items()
                if key != "handler" and isinstance(value, (str, int, float, bool, list, tuple, type(None)))
            },
            "fold_fingerprint": manifest.fingerprint,
            "o41_checkpoint_sha256": o41_sha256,
            "o41_scale": o41_scale,
            "leakage_audit": leakage.to_dict(),
        },
    )
    sample_rows = training_indices[: min(args.batch_size, len(training_indices))]
    sample_frozen = _frozen_batch(
        o41_model,
        o41_scale,
        adapter,
        dataset,
        arrays,
        token_cache,
        direct_cache,
        anchor_latents,
        neighbor_indices,
        neighbor_distances,
        sample_rows,
        "train",
        device,
    )
    sample_conditions = _condition_arrays(
        token_cache,
        direct_cache,
        neighbor_indices,
        sample_rows,
        sample_rows,
        "train",
        device,
    )
    identity_error = _identity(model, sample_frozen, sample_conditions)
    if identity_error > 1e-6:
        raise RuntimeError(f"O4.1 identity fallback failed: {identity_error}")

    best_score, bad_epochs, history = -math.inf, 0, []
    stable_scale = (
        float(promotion["best_stable_scale"]) if promotion is not None else None
    )
    epoch_progress = tqdm(
        range(1, args.epochs + 1),
        desc="AC-CGMRF epochs",
        dynamic_ncols=True,
    )
    for epoch in epoch_progress:
        model.train()
        model.set_phase_unlocked(epoch > args.phase_unlock_epoch)
        progress_value = min(1.0, epoch / max(1, args.energy_warmup_epochs))
        shuffle_lookup = np.arange(len(dataset.train_pos), dtype=np.int64)
        shuffle_lookup[training_indices] = np.roll(training_indices, 1)
        totals = {
            "loss": 0.0,
            "score": 0.0,
            "real_over_zero": 0.0,
            "real_over_shuffle": 0.0,
            "novelty": 0.0,
        }
        seen = 0
        batches = _rows_loader(
            training_indices, args.batch_size, True, args.seed + epoch
        )
        batch_progress = tqdm(
            batches,
            desc=f"epoch {epoch} batches",
            leave=False,
            dynamic_ncols=True,
        )
        for (row_tensor,) in batch_progress:
            rows = row_tensor.numpy()
            optimizer.zero_grad(set_to_none=True)
            frozen = _frozen_batch(
                o41_model,
                o41_scale,
                adapter,
                dataset,
                arrays,
                token_cache,
                direct_cache,
                anchor_latents,
                neighbor_indices,
                neighbor_distances,
                rows,
                "train",
                device,
            )
            real_conditions = _condition_arrays(
                token_cache,
                direct_cache,
                neighbor_indices,
                rows,
                rows,
                "train",
                device,
            )
            shuffle_conditions = _condition_arrays(
                token_cache,
                direct_cache,
                neighbor_indices,
                rows,
                shuffle_lookup[rows],
                "train",
                device,
            )
            real_output = _model_output(
                model, frozen, real_conditions, "real", progress_value
            )
            zero_output = _model_output(
                model, frozen, real_conditions, "zero", progress_value
            )
            shuffle_output = _model_output(
                model, frozen, shuffle_conditions, "real", progress_value
            )
            target = torch.from_numpy(
                np.asarray(dataset.channel_batch(rows)).copy()
            ).to(device=device, dtype=torch.complex64)
            values = ac_cgmrf_loss(
                real_output,
                zero_output,
                shuffle_output,
                target,
                adapter,
                loss_config,
            )
            if not torch.isfinite(values.total):
                raise FloatingPointError("non-finite AC-CGMRF loss")
            values.total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip_norm
            )
            optimizer.step()
            count = len(rows)
            totals["loss"] += float(values.total.detach()) * count
            totals["score"] += float(values.metrics.score.detach()) * count
            totals["real_over_zero"] += (
                float(values.real_over_zero.detach()) * count
            )
            totals["real_over_shuffle"] += (
                float(values.real_over_shuffle.detach()) * count
            )
            totals["novelty"] += (
                float(real_output.support_novelty.mean().detach()) * count
            )
            seen += count
            batch_progress.set_postfix(
                loss=f"{totals['loss'] / seen:.4f}",
                score=f"{totals['score'] / seen:.4f}",
                novelty=f"{totals['novelty'] / seen:.3f}",
            )
        raw_evaluation = _evaluate(
            model,
            adapter,
            dataset,
            arrays,
            token_cache,
            direct_cache,
            o41_model,
            o41_scale,
            anchor_latents,
            neighbor_indices,
            neighbor_distances,
            validation,
            args,
            device,
        )
        report = _evaluation_report(raw_evaluation)
        if stable_scale is None:
            selected_scale = float(report["best"]["real"]["scale"])
        else:
            selected_scale = stable_scale
        scale_key = str(selected_scale)
        if scale_key not in report["metrics"]["real"]:
            scale_key = next(
                key
                for key in report["metrics"]["real"]
                if float(key) == selected_scale
            )
        selected_metrics = report["metrics"]["real"][scale_key]
        record = {
            "epoch": epoch,
            "phase_unlocked": model.phase_unlocked,
            "progress": progress_value,
            "train": {key: value / seen for key, value in totals.items()},
            "validation": report,
            "selected_scale": selected_scale,
            "selected_score": selected_metrics["score"],
        }
        history.append(record)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        _save_checkpoint(
            run_dir / "last.pt",
            model,
            config,
            provider,
            token_cache,
            direct_cache,
            args,
            o41_sha256,
            o41_scale,
            selected_scale,
            selected_metrics,
        )
        if selected_metrics["score"] > best_score + args.min_delta:
            best_score, bad_epochs = selected_metrics["score"], 0
            _save_checkpoint(
                run_dir / "best.pt",
                model,
                config,
                provider,
                token_cache,
                direct_cache,
                args,
                o41_sha256,
                o41_scale,
                selected_scale,
                selected_metrics,
            )
        else:
            bad_epochs += 1
        epoch_progress.set_postfix(
            train=f"{totals['score'] / seen:.4f}",
            val=f"{selected_metrics['score']:.4f}",
            scale=f"{selected_scale:g}",
        )
        if bad_epochs >= args.patience:
            break
    _json(
        run_dir / "train_report.json",
        {
            "kind": "ac_cgmrf_train",
            "task_id": TASK_ID,
            "stage": args.stage,
            "epochs_completed": len(history),
            "best_score": best_score,
            "identity_max_abs_error": identity_error,
            "leakage_audit": leakage.to_dict(),
            "model_config": asdict(config),
            "loss_config": asdict(loss_config),
            "o41_checkpoint_sha256": o41_sha256,
            "o41_scale": o41_scale,
            "best_checkpoint": str((run_dir / "best.pt").resolve()),
            "best_checkpoint_sha256": _sha256_file(run_dir / "best.pt"),
            "last_checkpoint_sha256": _sha256_file(run_dir / "last.pt"),
            "promotion_report": args.promotion_report,
        },
    )
    token_cache.close()
    direct_cache.close()
    return 0


def _run_preflight(args) -> int:
    device = _device(args.device)
    _seed(args.seed)
    (
        provider,
        manifest,
        dataset,
        adapter,
        train,
        validation,
        arrays,
        token_cache,
        direct_cache,
        o41_model,
        _,
        o41_scale,
    ) = _open_stack(args)
    o41_sha256 = _validate_sidecar(args.o41_checkpoint, "O4.1")
    anchor_latents, neighbor_indices, neighbor_distances = _anchor_arrays(
        args, dataset, "train", direct_cache
    )
    leakage = audit_anchor_exclusion(train, validation, neighbor_indices)
    rows = validation[: min(args.batch_size, len(validation))]
    if len(rows) < 1:
        raise ValueError("preflight needs at least one validation target")
    frozen = _frozen_batch(
        o41_model,
        o41_scale,
        adapter,
        dataset,
        arrays,
        token_cache,
        direct_cache,
        anchor_latents,
        neighbor_indices,
        neighbor_distances,
        rows,
        "train",
        device,
    )
    conditions = _condition_arrays(
        token_cache,
        direct_cache,
        neighbor_indices,
        rows,
        rows,
        "train",
        device,
    )
    config = _model_config(args, dataset, token_cache)
    model = ACCGMRF(config).to(device)
    identity_error = _identity(model, frozen, conditions)
    output = _model_output(model, frozen, conditions, "real", 1.0)
    channel = adapter.channel_from_beam_delay_torch(output.prediction)
    gradient_probe = (
        output.raw_residual.abs().square().mean()
        + output.prediction.real.mean()
    )
    gradient_probe.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    gradient_finite = bool(
        gradients
        and all(torch.isfinite(value).all() for value in gradients)
        and any(float(value.abs().sum()) > 0 for value in gradients)
    )
    expected_shape = (len(rows),) + tuple(dataset.config.channel_shape)
    checks = {
        "o41_identity_max_abs_error": identity_error <= 1e-6,
        "channel_shape": tuple(channel.shape) == expected_shape,
        "channel_dtype_complex64": channel.dtype == torch.complex64,
        "channel_finite": bool(torch.isfinite(channel).all()),
        "splat_gradient_finite_nonzero": gradient_finite,
        "validation_targets_excluded_from_anchor_pool": leakage.passed,
        "anchor_count_k4": neighbor_indices.shape[1] == 4,
    }
    report = {
        "kind": "ac_cgmrf_preflight",
        "task_id": TASK_ID,
        "passed": all(checks.values()),
        "checks": checks,
        "identity_max_abs_error": identity_error,
        "channel_shape": list(channel.shape),
        "channel_dtype": str(channel.dtype),
        "leakage_audit": leakage.to_dict(),
        "fold_fingerprint": manifest.fingerprint,
        "coarse_fingerprint": provider.fingerprint,
        "gaussian_token_fingerprint": token_cache.fingerprint,
        "anchor_path_fingerprint": direct_cache.fingerprint,
        "o41_checkpoint_sha256": o41_sha256,
        "o41_scale": o41_scale,
        "model_config": asdict(config),
    }
    _json(args.output, report)
    token_cache.close()
    direct_cache.close()
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 2


def _run_evaluate(args) -> int:
    device = _device(args.device)
    (
        provider,
        _,
        dataset,
        adapter,
        train,
        validation,
        arrays,
        token_cache,
        direct_cache,
        o41_model,
        _,
        o41_scale,
    ) = _open_stack(args)
    o41_sha256 = _validate_sidecar(args.o41_checkpoint, "O4.1")
    model, payload = _load_checkpoint(
        args.checkpoint,
        provider,
        token_cache,
        direct_cache,
        args,
        device,
        o41_sha256,
        o41_scale,
    )
    anchor_latents, neighbor_indices, neighbor_distances = _anchor_arrays(
        args, dataset, "train", direct_cache
    )
    leakage = audit_anchor_exclusion(train, validation, neighbor_indices)
    if not leakage.passed:
        raise ValueError("fold cache permits validation target/Anchor leakage")
    raw = _evaluate(
        model,
        adapter,
        dataset,
        arrays,
        token_cache,
        direct_cache,
        o41_model,
        o41_scale,
        anchor_latents,
        neighbor_indices,
        neighbor_distances,
        validation,
        args,
        device,
    )
    report = _evaluation_report(raw)
    report.update(
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": _sha256_file(Path(args.checkpoint)),
            "checkpoint_best_scale": payload["best_scale"],
            "o41_checkpoint_sha256": o41_sha256,
            "leakage_audit": leakage.to_dict(),
        }
    )
    _json(args.output, report)
    token_cache.close()
    direct_cache.close()
    return 0


def _run_make_folds(args) -> int:
    dataset = RoundDataset.open(args.data_dir)
    folds = spatial_block_folds(
        dataset.train_pos, args.fold_count, args.block_size, args.seed
    )
    _json(
        args.output,
        spatial_fold_manifest(
            dataset.train_pos, folds, args.block_size, args.seed
        ),
    )
    return 0


def _run_prepare_fold_cache(args) -> int:
    dataset = RoundDataset.open(args.data_dir)
    geometry = GeometryPrior.load(args.geometry_cache)
    manifest = json.loads(Path(args.fold_manifest).read_text(encoding="utf-8"))
    if manifest.get("kind") != "ac_cgmrf_spatial_folds":
        raise ValueError("not an AC-CGMRF spatial fold manifest")
    fold = manifest["folds"][args.fold_index]
    split = SplitIndices(
        train=np.asarray(fold["train_indices"], dtype=np.int64),
        validation=np.asarray(fold["validation_indices"], dtype=np.int64),
    )
    config = FoldCacheConfig(
        layout_order=tuple(args.layout_order),
        support_fraction=args.support_fraction,
        delay_block=args.delay_block,
        k_max=args.k_max,
        anchor_count=args.anchor_count,
        patch=args.patch,
        corridor=args.corridor,
        anchor_corridor=args.anchor_corridor,
        encode_batch_size=args.channel_batch_size,
        dropout=args.dropout,
        min_anchors=args.min_anchors,
        seed=args.seed,
        code_version="020-ac-cgmrf-spatial-v1",
    )
    output = prepare_fold_cache(dataset, geometry, split, config, args.output_dir)
    cache_manifest = FoldCacheManifest.load(output / "manifest.json")
    validate_cache(cache_manifest, output)
    train, validation = load_persisted_split(output, validate=False)
    neighbors = np.load(output / "neighbor_indices.npy", mmap_mode="r")
    leakage = audit_anchor_exclusion(train, validation, neighbors[:, : args.anchor_count])
    if not leakage.passed:
        raise RuntimeError("prepared fold failed Anchor leakage audit")
    _json(
        output / "ac_cgmrf_fold_report.json",
        {
            "kind": "ac_cgmrf_fold_cache",
            "fold_index": args.fold_index,
            "fold_fingerprint": cache_manifest.fingerprint,
            "leakage_audit": leakage.to_dict(),
        },
    )
    return 0


def _run_aggregate_cv(args) -> int:
    reports = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in args.fold_reports
    ]
    result = aggregate_cross_validation(reports)
    result["fold_reports"] = [str(Path(path).resolve()) for path in args.fold_reports]
    _json(args.output, result)
    return 0


def _run_infer(args) -> int:
    promotion = require_promotion_report(args.promotion_report)
    device = _device(args.device)
    (
        provider,
        _,
        dataset,
        adapter,
        _,
        _,
        arrays,
        token_cache,
        direct_cache,
        o41_model,
        _,
        o41_scale,
    ) = _open_stack(args)
    o41_sha256 = _validate_sidecar(args.o41_checkpoint, "O4.1")
    model, payload = _load_checkpoint(
        args.checkpoint,
        provider,
        token_cache,
        direct_cache,
        args,
        device,
        o41_sha256,
        o41_scale,
    )
    if payload.get("training_stage") != "full":
        raise ValueError("official inference requires a full-training checkpoint")
    scale = float(promotion["best_stable_scale"])
    anchor_latents, neighbor_indices, neighbor_distances = _anchor_arrays(
        args, dataset, "test", direct_cache
    )
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
    try:
        with torch.no_grad():
            for start in tqdm(
                range(0, len(dataset.test_pos), args.batch_size),
                desc="AC-CGMRF official test inference",
                dynamic_ncols=True,
            ):
                rows = np.arange(
                    start, min(start + args.batch_size, len(dataset.test_pos))
                )
                frozen = _frozen_batch(
                    o41_model,
                    o41_scale,
                    adapter,
                    dataset,
                    arrays,
                    token_cache,
                    direct_cache,
                    anchor_latents,
                    neighbor_indices,
                    neighbor_distances,
                    rows,
                    "test",
                    device,
                )
                conditions = _condition_arrays(
                    token_cache,
                    direct_cache,
                    neighbor_indices,
                    rows,
                    rows,
                    "test",
                    device,
                )
                result = _model_output(model, frozen, conditions, "real", 1.0)
                candidate = frozen["base_beam_delay"] + scale * result.residual
                values = inverse_beam_delay(candidate.cpu().numpy(), adapter.layout)
                if not np.isfinite(values).all():
                    raise FloatingPointError("non-finite AC-CGMRF inference")
                prediction[rows] = values.astype(np.complex64, copy=False)
        prediction.flush()
        del prediction
        validate_submission(temporary, dataset, args.batch_size)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    validation = validate_submission(output, dataset, args.batch_size)
    _json(
        args.report or output.with_name("infer_report.json"),
        {
            "kind": "ac_cgmrf_infer",
            "task_id": TASK_ID,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": _sha256_file(Path(args.checkpoint)),
            "o41_checkpoint_sha256": o41_sha256,
            "promotion_report": str(Path(args.promotion_report).resolve()),
            "scale": scale,
            "submission": validation,
            "submission_sha256": _sha256_file(output),
        },
    )
    token_cache.close()
    direct_cache.close()
    return 0


def _common(parser) -> None:
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
    parser.add_argument("--anchor-path-cache", required=True)
    parser.add_argument("--o41-checkpoint", required=True)
    parser.add_argument("--o41-scale", type=float)
    parser.add_argument("--anchor-count", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)


def _validation_scales(parser) -> None:
    parser.add_argument(
        "--validation-scales",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )


def _model_arguments(parser) -> None:
    parser.add_argument("--atom-count", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--map-hidden-dim", type=int, default=96)
    parser.add_argument("--support-h", type=int, default=3)
    parser.add_argument("--support-v", type=int, default=2)
    parser.add_argument("--support-delay", type=int, default=4)
    parser.add_argument("--max-residual-ratio", type=float, default=0.35)
    parser.add_argument("--support-threshold", type=float, default=1e-3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    folds = commands.add_parser("make-folds")
    folds.add_argument("--data-dir", required=True)
    folds.add_argument("--fold-count", type=int, default=5)
    folds.add_argument("--block-size", type=float, default=20.0)
    folds.add_argument("--seed", type=int, default=42)
    folds.add_argument("--output", required=True)
    folds.set_defaults(handler=_run_make_folds)

    prepare = commands.add_parser("prepare-fold-cache")
    prepare.add_argument("--data-dir", required=True)
    prepare.add_argument("--geometry-cache", required=True)
    prepare.add_argument("--fold-manifest", required=True)
    prepare.add_argument("--fold-index", type=int, required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--layout-order", nargs=3, default=("H", "V", "P"))
    prepare.add_argument("--support-fraction", type=float, default=0.02)
    prepare.add_argument("--delay-block", type=int, default=8)
    prepare.add_argument("--k-max", type=int, default=16)
    prepare.add_argument("--anchor-count", type=int, default=4)
    prepare.add_argument("--patch", type=int, default=5)
    prepare.add_argument("--corridor", type=int, default=16)
    prepare.add_argument("--anchor-corridor", type=int, default=8)
    prepare.add_argument("--channel-batch-size", type=int, default=4)
    prepare.add_argument("--dropout", type=float, default=0.25)
    prepare.add_argument("--min-anchors", type=int, default=2)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.set_defaults(handler=_run_prepare_fold_cache)

    preflight = commands.add_parser("preflight")
    _common(preflight)
    _model_arguments(preflight)
    preflight.add_argument("--output", required=True)
    preflight.set_defaults(handler=_run_preflight)

    train = commands.add_parser("train")
    _common(train)
    _validation_scales(train)
    _model_arguments(train)
    train.add_argument("--run-dir", required=True)
    train.add_argument("--stage", choices=("pilot", "full"), default="pilot")
    train.add_argument("--promotion-report")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--learning-rate", type=float, default=5e-5)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--gradient-clip-norm", type=float, default=1.0)
    train.add_argument("--phase-unlock-epoch", type=int, default=5)
    train.add_argument("--energy-warmup-epochs", type=int, default=8)
    train.add_argument("--complex-weight", type=float, default=0.20)
    train.add_argument("--pas-weight", type=float, default=0.20)
    train.add_argument("--pdp-weight", type=float, default=0.20)
    train.add_argument("--score-weight", type=float, default=0.40)
    train.add_argument("--energy-weight", type=float, default=0.002)
    train.add_argument("--trust-weight", type=float, default=0.001)
    train.add_argument("--sparsity-weight", type=float, default=0.001)
    train.add_argument("--coupling-weight", type=float, default=0.0005)
    train.add_argument("--phase-continuity-weight", type=float, default=0.0005)
    train.add_argument("--causal-margin-weight", type=float, default=1.0)
    train.add_argument("--zero-margin", type=float, default=0.01)
    train.add_argument("--shuffle-margin", type=float, default=0.01)
    train.add_argument("--patience", type=int, default=8)
    train.add_argument("--min-delta", type=float, default=1e-5)
    train.set_defaults(handler=_run_train)

    evaluate = commands.add_parser("evaluate")
    _common(evaluate)
    _validation_scales(evaluate)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(handler=_run_evaluate)

    aggregate = commands.add_parser("aggregate-cv")
    aggregate.add_argument("--fold-reports", nargs="+", required=True)
    aggregate.add_argument("--output", required=True)
    aggregate.set_defaults(handler=_run_aggregate_cv)

    infer = commands.add_parser("infer")
    _common(infer)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--promotion-report", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--report")
    infer.add_argument("--overwrite", action="store_true")
    infer.set_defaults(handler=_run_infer)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(args, "anchor_count") and args.anchor_count != 4:
        raise ValueError("AC-CGMRF freezes O4.1 K=4 Anchor selection")
    if hasattr(args, "validation_scales"):
        scales = tuple(float(value) for value in args.validation_scales)
        if 0.0 not in scales or any(not math.isfinite(value) or value < 0 for value in scales):
            raise ValueError("validation-scales must be finite, non-negative, and include 0")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
