"""Command line entry points for the staged E2E-CGPF protocol."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from ..data import RoundDataset
from ..splits import block_split
from .e2e_cgpf import (
    E2ECGPFConfig,
    GaussianFieldConfig,
    PathNetworkConfig,
    RendererConfig,
)
from .e2e_cgpf_losses import E2ECGPFLossConfig
from .e2e_cgpf_runtime import (
    E2ECGPFTrainer,
    TrainingConfig,
    build_model,
    capacity_stage,
    default_stages,
    load_model_checkpoint,
    make_loader,
    set_deterministic_seed,
    sha256_file,
    write_hash_sidecar,
    write_json,
)


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _model_config(args: argparse.Namespace, map_mode: str | None = None) -> E2ECGPFConfig:
    if args.model_config:
        values = _load_json(args.model_config)
        values["seed"] = args.seed
        if map_mode is not None:
            values["map_mode"] = map_mode
        return E2ECGPFConfig.from_dict(values)
    config = E2ECGPFConfig(
        map_mode=map_mode or getattr(args, "map_mode", "real"),
        seed=args.seed,
    )
    config.validate()
    return config


def _training_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        field_learning_rate=args.field_learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        accumulation_steps=args.accumulation_steps,
        num_workers=args.num_workers,
        seed=args.seed,
        causal_every_batches=args.causal_every_batches,
        densify_start_epoch=args.densify_start_epoch,
        densify_interval=args.densify_interval,
        structure_rollback_tolerance=args.structure_rollback_tolerance,
        device=args.device,
    )


def _loss_config(args: argparse.Namespace, map_mode: str) -> E2ECGPFLossConfig:
    causal_weight = args.causal_weight if map_mode == "real" else 0.0
    return E2ECGPFLossConfig(
        complex_weight=args.complex_weight,
        pas_weight=args.pas_weight,
        pdp_weight=args.pdp_weight,
        nmse_weight=args.nmse_weight,
        score_weight=args.score_weight,
        phase_weight=args.phase_weight,
        path_sparsity_weight=args.path_sparsity_weight,
        path_diversity_weight=args.path_diversity_weight,
        causal_weight=causal_weight,
        causal_margin=args.causal_margin,
        minimum_effective_paths=args.minimum_effective_paths,
    )


def _open_official_data(path: str | Path) -> RoundDataset:
    source = RoundDataset.open(path)
    actual = int(source.train_pos.shape[0])
    if actual != 2000:
        raise ValueError(
            f"E2E-CGPF is bound to the verified Round1 P_Train=2000 arrays, got {actual}"
        )
    return source


def _run_training(
    source: RoundDataset,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
    *,
    map_mode: str,
    stages: Sequence[Any],
) -> dict[str, Any]:
    set_deterministic_seed(args.seed)
    model = build_model(source, train_indices, _model_config(args, map_mode))
    training_config = _training_config(args)
    trainer = E2ECGPFTrainer(
        model,
        training_config,
        _loss_config(args, map_mode),
        output_dir,
        train_indices=train_indices,
        validation_indices=validation_indices,
    )
    train_loader = make_loader(
        source, train_indices, training_config, shuffle=True
    )
    validation_loader = make_loader(
        source, validation_indices, training_config, shuffle=False
    )
    result = trainer.fit(train_loader, validation_loader, stages)
    result.update(
        {
            "map_mode": map_mode,
            "output_dir": str(output_dir),
            "train_count": int(train_indices.size),
            "validation_count": int(validation_indices.size),
        }
    )
    return result


def _capacity(args: argparse.Namespace) -> int:
    source = _open_official_data(args.data_dir)
    if not 2 <= args.subset_size <= len(source.train_pos):
        raise ValueError("subset_size must be between 2 and P_Train")
    rng = np.random.default_rng(args.seed)
    subset = np.sort(
        rng.choice(len(source.train_pos), size=args.subset_size, replace=False)
    ).astype(np.int64)
    output_dir = Path(args.output_dir)
    result = _run_training(
        source,
        subset,
        subset,
        output_dir,
        args,
        map_mode="real",
        stages=capacity_stage(args.epochs),
    )
    metrics = result["best_metrics"]
    finite = all(
        math.isfinite(float(metrics[name])) for name in ("score", "pas", "pdp", "nmse")
    )
    stable_paths = float(metrics["active_paths_mean"]) >= 2.0
    nonzero = float(metrics["prediction_energy_ratio"]) > 1e-8
    target_dependent = (
        int(metrics["target_path_state_unique"]) > 1
        or float(metrics["target_delay_mean_std"]) > 1e-8
        or float(metrics["target_angle_mean_std"]) > 1e-8
        or float(metrics["target_path_energy_std"]) > 1e-8
    )
    passed = (
        finite
        and float(metrics["score"]) >= 0.80
        and stable_paths
        and nonzero
        and target_dependent
    )
    report = {
        "kind": "representation_capacity",
        "passed": passed,
        "threshold": 0.80,
        "metrics": metrics,
        "checks": {
            "finite_complex_pas_pdp_nmse": finite,
            "multiple_stable_paths": stable_paths,
            "not_all_zero": nonzero,
            "target_dependent_paths": target_dependent,
            "complex_loss": float(metrics["nmse"]),
        },
        "subset_indices": subset.tolist(),
        "warning": "This is a train-subset representation test, not deployment evidence.",
        **result,
    }
    write_json(output_dir / "capacity_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


def _baseline_metrics(path: str | None, fallback_score: float) -> dict[str, Any]:
    if path is None:
        return {
            "score": fallback_score,
            "pas": None,
            "pdp": None,
            "nmse": None,
            "component_provenance_closed": False,
        }
    raw = _load_json(path)
    for key in ("metrics", "best_metrics", "validation"):
        if isinstance(raw.get(key), dict) and "score" in raw[key]:
            raw = raw[key]
            break
    return {
        "score": float(raw["score"]),
        "pas": float(raw["pas"]),
        "pdp": float(raw["pdp"]),
        "nmse": float(raw["nmse"]),
        "component_provenance_closed": True,
    }


def _component_gate(real: dict[str, Any], baseline: dict[str, Any]) -> bool:
    if not baseline["component_provenance_closed"]:
        return False
    return (
        float(real["pas"]) >= float(baseline["pas"])
        and float(real["pdp"]) >= float(baseline["pdp"])
        and float(real["nmse"]) <= float(baseline["nmse"])
    )


def _spatial_variants(
    source: RoundDataset,
    train: np.ndarray,
    validation: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    stages = default_stages(args.epochs_b, args.epochs_c, args.epochs_d)
    for map_mode in ("real", "zero", "shuffle"):
        results[map_mode] = _run_training(
            source,
            train,
            validation,
            output_dir / map_mode,
            args,
            map_mode=map_mode,
            stages=stages,
        )
    return results


def _single_fold(args: argparse.Namespace) -> int:
    _require_passed_report(args.capacity_report, "representation_capacity")
    source = _open_official_data(args.data_dir)
    split = block_split(
        np.asarray(source.train_pos),
        axis=args.axis,
        validation_fraction=args.validation_fraction,
        side=args.side,
    )
    output_dir = Path(args.output_dir)
    results = _spatial_variants(
        source, split.train, split.validation, output_dir, args
    )
    baseline = _baseline_metrics(args.baseline_report, args.o41_score)
    real = results["real"]["best_metrics"]
    zero = results["zero"]["best_metrics"]
    shuffle = results["shuffle"]["best_metrics"]
    real_over_zero = float(real["score"]) - float(zero["score"])
    real_over_shuffle = float(real["score"]) - float(shuffle["score"])
    component_ok = _component_gate(real, baseline)
    target_dependent = (
        int(real["target_path_state_unique"]) > 1
        or float(real["target_delay_mean_std"]) > 1e-8
        or float(real["target_angle_mean_std"]) > 1e-8
        or float(real["target_path_energy_std"]) > 1e-8
    )
    passed = (
        float(real["score"]) >= float(baseline["score"]) + 0.01
        and real_over_zero >= 0.005
        and real_over_shuffle >= 0.005
        and component_ok
        and target_dependent
    )
    report = {
        "kind": "single_spatial_fold",
        "passed": passed,
        "fold": {"axis": args.axis, "side": args.side},
        "baseline": baseline,
        "results": results,
        "real_over_zero": real_over_zero,
        "real_over_shuffle": real_over_shuffle,
        "component_gate": component_ok,
        "target_dependent_paths": target_dependent,
    }
    write_json(output_dir / "single_fold_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


def _cross_validate(args: argparse.Namespace) -> int:
    _require_passed_report(args.capacity_report, "representation_capacity")
    _require_passed_report(args.single_fold_report, "single_spatial_fold")
    source = _open_official_data(args.data_dir)
    output_dir = Path(args.output_dir)
    baseline = _baseline_metrics(args.baseline_report, args.o41_score)
    folds: list[dict[str, Any]] = []
    for axis in range(3):
        for side in ("low", "high"):
            fold_name = f"axis{axis}_{side}"
            split = block_split(
                np.asarray(source.train_pos),
                axis=axis,
                validation_fraction=args.validation_fraction,
                side=side,
            )
            results = _spatial_variants(
                source,
                split.train,
                split.validation,
                output_dir / fold_name,
                args,
            )
            real = results["real"]["best_metrics"]
            zero = results["zero"]["best_metrics"]
            shuffle = results["shuffle"]["best_metrics"]
            folds.append(
                {
                    "name": fold_name,
                    "results": results,
                    "gain": float(real["score"]) - float(baseline["score"]),
                    "real_over_zero": float(real["score"]) - float(zero["score"]),
                    "real_over_shuffle": float(real["score"])
                    - float(shuffle["score"]),
                    "component_gate": _component_gate(real, baseline),
                    "target_dependent_paths": (
                        int(real["target_path_state_unique"]) > 1
                        or float(real["target_delay_mean_std"]) > 1e-8
                        or float(real["target_angle_mean_std"]) > 1e-8
                        or float(real["target_path_energy_std"]) > 1e-8
                    ),
                }
            )
            write_json(output_dir / fold_name / "fold_report.json", folds[-1])
    gains = [fold["gain"] for fold in folds]
    zero_margins = [fold["real_over_zero"] for fold in folds]
    shuffle_margins = [fold["real_over_shuffle"] for fold in folds]
    passed = (
        all(gain > 0.0 for gain in gains)
        and float(np.mean(gains)) >= 0.03
        and max(gains) >= 0.05
        and min(zero_margins) >= 0.01
        and min(shuffle_margins) >= 0.01
        and all(fold["component_gate"] for fold in folds)
        and all(fold["target_dependent_paths"] for fold in folds)
    )
    report = {
        "kind": "full_spatial_cross_validation",
        "passed": passed,
        "baseline": baseline,
        "folds": folds,
        "gain_mean": float(np.mean(gains)),
        "gain_best": max(gains),
        "gain_min": min(gains),
        "real_over_zero_min": min(zero_margins),
        "real_over_shuffle_min": min(shuffle_margins),
    }
    write_json(output_dir / "cross_validation_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


def _require_passed_report(path: str | Path, expected_kind: str) -> dict[str, Any]:
    report = _load_json(path)
    if report.get("kind") != expected_kind or report.get("passed") is not True:
        raise RuntimeError(
            f"protocol stage is locked until {expected_kind} passes"
        )
    return report


def _require_passed_gate(path: str | Path) -> dict[str, Any]:
    return _require_passed_report(path, "full_spatial_cross_validation")


def _full_train(args: argparse.Namespace) -> int:
    _require_passed_gate(args.gate_report)
    source = _open_official_data(args.data_dir)
    all_indices = np.arange(len(source.train_pos), dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    monitor = np.sort(
        rng.choice(all_indices, size=min(args.monitor_size, len(all_indices)), replace=False)
    )
    output_dir = Path(args.output_dir)
    result = _run_training(
        source,
        all_indices,
        monitor,
        output_dir,
        args,
        map_mode="real",
        stages=default_stages(args.epochs_b, args.epochs_c, args.epochs_d),
    )
    report = {
        "kind": "full_training",
        "passed_spatial_gate": True,
        "uses_all_2000_training_positions": result["train_count"] == 2000,
        **result,
    }
    write_json(output_dir / "full_training_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


@torch.no_grad()
def _infer(args: argparse.Namespace) -> int:
    _require_passed_gate(args.gate_report)
    source = _open_official_data(args.data_dir)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, payload = load_model_checkpoint(args.checkpoint, device)
    if len(np.asarray(payload["train_indices"])) != 2000:
        raise RuntimeError("official inference requires a checkpoint trained on all 2000 positions")
    if model.config.map_mode != "real":
        raise RuntimeError("official inference requires the real-map model")
    model.eval()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=np.complex64,
        shape=(len(source.test_pos), *source.config.channel_shape),
    )
    for start in tqdm(
        range(0, len(source.test_pos), args.batch_size),
        desc="official inference",
        dynamic_ncols=True,
    ):
        stop = min(len(source.test_pos), start + args.batch_size)
        positions = torch.from_numpy(
            np.array(source.test_pos[start:stop], dtype=np.float32, copy=True)
        ).to(device)
        channel, _ = model(positions)
        output[start:stop] = channel.cpu().numpy().astype(np.complex64, copy=False)
    output.flush()
    del output
    verification = np.load(destination, mmap_mode="r")
    finite = True
    for start in range(0, len(verification), 16):
        block = verification[start : start + 16]
        finite = finite and bool(
            np.isfinite(block.real).all() and np.isfinite(block.imag).all()
        )
    expected_shape = (500, 256, 4, 192)
    if verification.shape != expected_shape or verification.dtype != np.complex64 or not finite:
        raise RuntimeError("official NPY failed shape, dtype, or finite validation")
    write_hash_sidecar(destination)
    report = {
        "kind": "official_inference",
        "output": str(destination),
        "shape": list(verification.shape),
        "dtype": str(verification.dtype),
        "finite": finite,
        "sha256": sha256_file(destination),
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "automatic_upload": False,
    }
    write_json(destination.with_name("inference_report.json"), report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def _smoke(args: argparse.Namespace) -> int:
    source = _open_official_data(args.data_dir)
    indices = np.asarray([0, 1], dtype=np.int64)
    config = E2ECGPFConfig(
        field=GaussianFieldConfig(
            initial_count=8,
            max_count=16,
            min_count=4,
            material_dim=8,
            densify_count=2,
            prune_count=2,
        ),
        paths=PathNetworkConfig(
            hidden_dim=32,
            query_dim=16,
            modes_per_gaussian=2,
            selected_gaussians=4,
            polarization_rank=1,
            path_type_dim=4,
            fourier_bands=3,
        ),
        renderer=RendererConfig(path_chunk_size=4),
        seed=args.seed,
    )
    set_deterministic_seed(args.seed)
    smoke_device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    model = build_model(source, indices, config).to(smoke_device)
    device = next(model.parameters()).device
    positions = torch.from_numpy(
        np.array(source.train_pos[indices], dtype=np.float32, copy=True)
    ).to(device)
    prediction, paths = model(positions)
    loss = prediction.abs().square().mean()
    loss.backward()
    gradient_checks = {}
    for name in (
        "center_delta",
        "log_scale",
        "orientation",
        "opacity_logit",
        "material_code",
    ):
        gradient = getattr(model.field, name).grad
        gradient_checks[name] = bool(
            gradient is not None
            and torch.isfinite(gradient).all()
            and float(gradient.abs().sum()) > 0.0
        )
    path_gradient = model.path_network.head.weight.grad
    gradient_checks["path_network"] = bool(
        path_gradient is not None
        and torch.isfinite(path_gradient).all()
        and float(path_gradient.abs().sum()) > 0.0
    )
    report = {
        "shape": list(prediction.shape),
        "dtype": str(prediction.dtype),
        "device": str(prediction.device),
        "finite": bool(
            torch.isfinite(prediction.real).all()
            and torch.isfinite(prediction.imag).all()
        ),
        "gradients": gradient_checks,
        "paths": paths.path_count,
        "o41_forward_dependency": False,
        "task020_dependency": False,
    }
    passed = (
        tuple(prediction.shape) == (2, 256, 4, 192)
        and torch.is_complex(prediction)
        and report["finite"]
        and all(gradient_checks.values())
    )
    report["passed"] = passed
    write_json(Path(args.output_dir) / "engineering_smoke_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


def _add_model_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-config")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--field-learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--causal-every-batches", type=int, default=4)
    parser.add_argument("--densify-start-epoch", type=int, default=20)
    parser.add_argument("--densify-interval", type=int, default=5)
    parser.add_argument("--structure-rollback-tolerance", type=float, default=0.02)
    parser.add_argument("--complex-weight", type=float, default=1.0)
    parser.add_argument("--pas-weight", type=float, default=0.25)
    parser.add_argument("--pdp-weight", type=float, default=0.25)
    parser.add_argument("--nmse-weight", type=float, default=0.25)
    parser.add_argument("--score-weight", type=float, default=0.5)
    parser.add_argument("--phase-weight", type=float, default=0.1)
    parser.add_argument("--path-sparsity-weight", type=float, default=1e-3)
    parser.add_argument("--path-diversity-weight", type=float, default=1e-2)
    parser.add_argument("--minimum-effective-paths", type=float, default=4.0)
    parser.add_argument("--causal-weight", type=float, default=0.05)
    parser.add_argument("--causal-margin", type=float, default=0.005)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="run production-shape engineering smoke")
    smoke.add_argument("--data-dir", required=True)
    smoke.add_argument("--output-dir", required=True)
    smoke.add_argument("--device", default="auto")
    smoke.add_argument("--seed", type=int, default=42)
    smoke.set_defaults(handler=_smoke)

    capacity = subparsers.add_parser("capacity", help="run the hard representation gate")
    capacity.add_argument("--data-dir", required=True)
    capacity.add_argument("--output-dir", required=True)
    capacity.add_argument("--subset-size", type=int, default=32)
    capacity.add_argument("--epochs", type=int, default=300)
    capacity.add_argument("--device", default="auto")
    capacity.add_argument("--seed", type=int, default=42)
    _add_model_training_arguments(capacity)
    capacity.set_defaults(handler=_capacity)

    single = subparsers.add_parser("single-fold", help="train real/zero/shuffle on one fold")
    single.add_argument("--data-dir", required=True)
    single.add_argument("--output-dir", required=True)
    single.add_argument("--axis", type=int, choices=(0, 1, 2), default=0)
    single.add_argument("--side", choices=("low", "high"), default="high")
    single.add_argument("--validation-fraction", type=float, default=0.2)
    single.add_argument("--epochs-b", type=int, default=20)
    single.add_argument("--epochs-c", type=int, default=40)
    single.add_argument("--epochs-d", type=int, default=40)
    single.add_argument("--baseline-report")
    single.add_argument("--capacity-report", required=True)
    single.add_argument("--o41-score", type=float, default=0.552317)
    single.add_argument("--device", default="auto")
    single.add_argument("--seed", type=int, default=42)
    _add_model_training_arguments(single)
    single.set_defaults(handler=_single_fold)

    cross = subparsers.add_parser("cross-validate", help="run all six strict spatial folds")
    cross.add_argument("--data-dir", required=True)
    cross.add_argument("--output-dir", required=True)
    cross.add_argument("--validation-fraction", type=float, default=0.2)
    cross.add_argument("--epochs-b", type=int, default=20)
    cross.add_argument("--epochs-c", type=int, default=40)
    cross.add_argument("--epochs-d", type=int, default=40)
    cross.add_argument("--baseline-report")
    cross.add_argument("--capacity-report", required=True)
    cross.add_argument("--single-fold-report", required=True)
    cross.add_argument("--o41-score", type=float, default=0.552317)
    cross.add_argument("--device", default="auto")
    cross.add_argument("--seed", type=int, default=42)
    _add_model_training_arguments(cross)
    cross.set_defaults(handler=_cross_validate)

    full = subparsers.add_parser("full-train", help="train on all 2000 after gate pass")
    full.add_argument("--data-dir", required=True)
    full.add_argument("--output-dir", required=True)
    full.add_argument("--gate-report", required=True)
    full.add_argument("--monitor-size", type=int, default=64)
    full.add_argument("--epochs-b", type=int, default=20)
    full.add_argument("--epochs-c", type=int, default=40)
    full.add_argument("--epochs-d", type=int, default=40)
    full.add_argument("--device", default="auto")
    full.add_argument("--seed", type=int, default=42)
    _add_model_training_arguments(full)
    full.set_defaults(handler=_full_train)

    infer = subparsers.add_parser("infer", help="generate gated official-format NPY")
    infer.add_argument("--data-dir", required=True)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--gate-report", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--batch-size", type=int, default=1)
    infer.add_argument("--device", default="auto")
    infer.set_defaults(handler=_infer)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
