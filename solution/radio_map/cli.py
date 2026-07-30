"""Command-line entry points for the Round1 foundation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .analysis import coverage_report, layout_report
from .baselines import InverseDistanceRegressor, NearestAnchorRegressor
from .data import RoundDataset
from .geometry import PlyPointCloud, build_geometry_prior
from .metrics import MetricAccumulator
from .oracle import codec_oracle_report
from .splits import coverage_split, nearest_anchor_distances
from .transforms import AntennaLayout


def _write_json(path: str | Path, report: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_submission(
    path: str | Path,
    dataset: RoundDataset,
    batch_size: int = 4,
) -> dict[str, object]:
    """Strictly validate a Round1 submission without loading it in full."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"submission file does not exist: {path}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    submission = np.load(path, mmap_mode="r", allow_pickle=False)
    expected_shape = (dataset.config.p_test,) + dataset.config.channel_shape
    if submission.shape != expected_shape:
        raise ValueError(
            f"submission shape mismatch: expected {expected_shape}, got {submission.shape}"
        )
    if submission.dtype != np.dtype(np.complex64):
        raise ValueError(
            f"submission dtype must be complex64, got {submission.dtype}"
        )
    finite = True
    file_hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            file_hasher.update(chunk)

    sample_hashes: set[str] = set()
    duplicate_sample_count = 0
    zero_sample_count = 0
    value_count = 0
    real_sum = 0.0
    imag_sum = 0.0
    power_sum = 0.0
    peak_magnitude = 0.0
    sample_mean_power: list[float] = []
    for start in range(0, len(submission), batch_size):
        batch = np.asarray(submission[start : start + batch_size])
        if not np.isfinite(batch).all():
            finite = False
            break
        real = np.asarray(batch.real, dtype=np.float64)
        imag = np.asarray(batch.imag, dtype=np.float64)
        power = np.square(real) + np.square(imag)
        value_count += int(batch.size)
        real_sum += float(real.sum(dtype=np.float64))
        imag_sum += float(imag.sum(dtype=np.float64))
        power_sum += float(power.sum(dtype=np.float64))
        peak_magnitude = max(peak_magnitude, float(np.sqrt(power.max(initial=0.0))))
        sample_mean_power.extend(
            np.mean(power.reshape(len(batch), -1), axis=1, dtype=np.float64).tolist()
        )
        for sample in batch:
            contiguous = np.ascontiguousarray(sample)
            digest = hashlib.sha256(contiguous.view(np.uint8)).hexdigest()
            if digest in sample_hashes:
                duplicate_sample_count += 1
            else:
                sample_hashes.add(digest)
            if not np.any(contiguous):
                zero_sample_count += 1
    if not finite:
        raise ValueError("submission contains NaN or Inf")
    quantiles = np.quantile(
        np.asarray(sample_mean_power, dtype=np.float64),
        [0.0, 0.25, 0.5, 0.75, 1.0],
    )
    warnings: list[str] = []
    if zero_sample_count:
        warnings.append(f"{zero_sample_count} all-zero sample(s)")
    if duplicate_sample_count:
        warnings.append(f"{duplicate_sample_count} duplicate sample(s)")
    if power_sum == 0.0:
        warnings.append("submission has zero global power")
    return {
        "path": str(path),
        "shape": [int(value) for value in submission.shape],
        "dtype": str(submission.dtype),
        "finite": True,
        "bytes": int(path.stat().st_size),
        "sha256": file_hasher.hexdigest(),
        "zero_sample_count": int(zero_sample_count),
        "duplicate_sample_count": int(duplicate_sample_count),
        "unique_sample_count": int(len(sample_hashes)),
        "real_mean": float(real_sum / value_count),
        "imag_mean": float(imag_sum / value_count),
        "global_mean_power": float(power_sum / value_count),
        "peak_magnitude": float(peak_magnitude),
        "sample_mean_power_quantiles": {
            "min": float(quantiles[0]),
            "q25": float(quantiles[1]),
            "median": float(quantiles[2]),
            "q75": float(quantiles[3]),
            "max": float(quantiles[4]),
        },
        "warnings": warnings,
    }


def _run_audit(args: argparse.Namespace) -> int:
    report = RoundDataset.open(args.data_dir).audit().to_dict()
    _write_json(args.output, report)
    return 0


def _run_analyze(args: argparse.Namespace) -> int:
    dataset = RoundDataset.open(args.data_dir)
    report = {
        "audit": dataset.audit().to_dict(),
        "coverage": coverage_report(dataset),
        "layout": layout_report(dataset, args.sample_count, args.seed),
    }
    _write_json(args.output, report)
    return 0


def _evaluate_model(
    model: object,
    dataset: RoundDataset,
    validation_indices: np.ndarray,
    layout: AntennaLayout,
    batch_size: int,
) -> dict[str, float]:
    accumulator = MetricAccumulator(layout, dataset.config.weights)
    start_time = time.perf_counter()
    for start in range(0, len(validation_indices), batch_size):
        indices = validation_indices[start : start + batch_size]
        prediction = model.predict(dataset.train_pos[indices])
        target = dataset.channel_batch(indices)
        accumulator.update(prediction, target)
    elapsed = time.perf_counter() - start_time
    result = accumulator.compute().to_dict()
    result["runtime_seconds"] = float(elapsed)
    return result


def _run_baseline(args: argparse.Namespace) -> int:
    dataset = RoundDataset.open(args.data_dir)
    split = coverage_split(
        dataset.train_pos,
        validation_fraction=args.validation_fraction,
        grid_size=args.grid_size,
        seed=args.seed,
    )
    layout = AntennaLayout(dataset.config, tuple(args.layout_order))
    anchor_positions = np.asarray(dataset.train_pos[split.train], dtype=np.float64)
    distances = nearest_anchor_distances(
        anchor_positions, dataset.train_pos[split.validation]
    )
    nearest = NearestAnchorRegressor().fit(
        anchor_positions, dataset.train_channel, channel_indices=split.train
    )
    inverse_distance = InverseDistanceRegressor(
        k=args.k, power=args.power
    ).fit(anchor_positions, dataset.train_channel, channel_indices=split.train)
    report = {
        "configuration": {
            "validation_fraction": float(args.validation_fraction),
            "grid_size": float(args.grid_size),
            "seed": int(args.seed),
            "batch_size": int(args.batch_size),
            "layout_order": args.layout_order,
            "inverse_distance_k": int(args.k),
            "inverse_distance_power": float(args.power),
        },
        "split": {
            "train": split.train.tolist(),
            "validation": split.validation.tolist(),
        },
        "coverage": {
            "nearest_distance_min": float(distances.min()),
            "nearest_distance_median": float(np.median(distances)),
            "nearest_distance_max": float(distances.max()),
            "within_5m_fraction": float(np.mean(distances <= 5.0)),
        },
        "models": {
            "nearest": _evaluate_model(
                nearest,
                dataset,
                split.validation,
                layout,
                args.batch_size,
            ),
            "inverse_distance": _evaluate_model(
                inverse_distance,
                dataset,
                split.validation,
                layout,
                args.batch_size,
            ),
        },
    }
    _write_json(args.output, report)
    return 0


def _run_validate_submission(args: argparse.Namespace) -> int:
    report = validate_submission(
        args.submission, RoundDataset.open(args.data_dir), args.batch_size
    )
    if args.output:
        _write_json(args.output, report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if len(result) != 5:
        raise argparse.ArgumentTypeError("expected exactly five Tucker ranks")
    return result


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not result:
        raise argparse.ArgumentTypeError("at least one value is required")
    return result


def _run_codec_oracle(args: argparse.Namespace) -> int:
    dataset = RoundDataset.open(args.data_dir)
    report = codec_oracle_report(
        dataset,
        validation_fraction=args.validation_fraction,
        grid_size=args.grid_size,
        seed=args.seed,
        batch_size=args.batch_size,
        layout_order=tuple(args.layout_order),
        tucker_ranks=args.tucker_ranks,
        support_fractions=args.support_fractions,
        fit_samples=args.fit_samples,
        validation_samples=args.validation_samples,
    )
    _write_json(args.output, report)
    return 0


def _run_build_geometry(args: argparse.Namespace) -> int:
    dataset = RoundDataset.open(args.data_dir)
    cloud = PlyPointCloud.open(dataset.map_path)
    started = time.perf_counter()
    prior = build_geometry_prior(
        cloud,
        resolution=args.resolution,
        height_layers=args.height_layers,
        batch_size=args.batch_size,
    )
    prior.save(args.cache)
    report = {
        "cache_path": str(Path(args.cache).resolve()),
        "feature_count": len(prior.feature_names),
        "feature_names": list(prior.feature_names),
        "feature_shape": [int(value) for value in prior.features.shape],
        "origin_xy": list(prior.origin_xy),
        "resolution": float(prior.resolution),
        "metadata": prior.metadata,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    _write_json(args.output, report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="audit official Round1 files")
    audit.add_argument("--data-dir", required=True)
    audit.add_argument("--output", required=True)
    audit.set_defaults(handler=_run_audit)

    analyze = subparsers.add_parser("analyze", help="analyze coverage and layouts")
    analyze.add_argument("--data-dir", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--sample-count", type=int, default=20)
    analyze.add_argument("--seed", type=int, default=42)
    analyze.set_defaults(handler=_run_analyze)

    baseline = subparsers.add_parser("baseline", help="evaluate local baselines")
    baseline.add_argument("--data-dir", required=True)
    baseline.add_argument("--output", required=True)
    baseline.add_argument("--validation-fraction", type=float, default=0.1)
    baseline.add_argument("--grid-size", type=float, default=20.0)
    baseline.add_argument("--seed", type=int, default=42)
    baseline.add_argument("--batch-size", type=int, default=2)
    baseline.add_argument("--layout-order", choices=("HVP", "HPV", "VHP", "VPH", "PHV", "PVH"), default="HVP")
    baseline.add_argument("--k", type=int, default=4)
    baseline.add_argument("--power", type=float, default=2.0)
    baseline.set_defaults(handler=_run_baseline)

    oracle = subparsers.add_parser(
        "codec-oracle", help="evaluate leakage-free low-rank and sparse codecs"
    )
    oracle.add_argument("--data-dir", required=True)
    oracle.add_argument("--output", required=True)
    oracle.add_argument("--validation-fraction", type=float, default=0.1)
    oracle.add_argument("--grid-size", type=float, default=20.0)
    oracle.add_argument("--seed", type=int, default=42)
    oracle.add_argument("--batch-size", type=int, default=2)
    oracle.add_argument(
        "--layout-order",
        choices=("HVP", "HPV", "VHP", "VPH", "PHV", "PVH"),
        default="PHV",
    )
    oracle.add_argument(
        "--tucker-ranks", type=_parse_int_tuple, default=(8, 4, 2, 4, 32)
    )
    oracle.add_argument(
        "--support-fractions",
        type=_parse_float_tuple,
        default=(0.01, 0.025, 0.05, 0.1),
    )
    oracle.add_argument("--fit-samples", type=int)
    oracle.add_argument("--validation-samples", type=int)
    oracle.set_defaults(handler=_run_codec_oracle)

    geometry = subparsers.add_parser(
        "build-geometry", help="build a cached 2.5-D prior from the official PLY"
    )
    geometry.add_argument("--data-dir", required=True)
    geometry.add_argument("--cache", required=True)
    geometry.add_argument("--output", required=True)
    geometry.add_argument("--resolution", type=float, default=1.0)
    geometry.add_argument("--height-layers", type=int, default=4)
    geometry.add_argument("--batch-size", type=int, default=131072)
    geometry.set_defaults(handler=_run_build_geometry)

    validate = subparsers.add_parser(
        "validate-submission", help="validate a submission .npy file"
    )
    validate.add_argument("--data-dir", required=True)
    validate.add_argument("--submission", required=True)
    validate.add_argument("--batch-size", type=int, default=4)
    validate.add_argument("--output")
    validate.set_defaults(handler=_run_validate_submission)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
