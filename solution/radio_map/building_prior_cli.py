"""Build and audit a switchable, ground-suppressed map prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .building_prior import build_building_prior
from .data import RoundDataset
from .geometry import GeometryPrior
from .splits import coverage_split


def _write(path: str | Path, value: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_build(args: argparse.Namespace) -> int:
    source = GeometryPrior.load(args.geometry_cache)
    prior = build_building_prior(
        source,
        user_height=args.user_height,
        elevated_thresholds=args.thresholds,
        density_windows_metres=args.density_windows,
    )
    prior.save(args.output_cache)
    occupancy = {
        name: float(prior.features[index].mean())
        for index, name in enumerate(prior.feature_names)
        if name.endswith("_occupancy")
    }
    _write(
        args.report,
        {
            "output_cache": str(Path(args.output_cache).resolve()),
            "feature_names": list(prior.feature_names),
            "feature_shape": list(prior.features.shape),
            "occupancy_fractions": occupancy,
            "metadata": prior.metadata,
        },
    )
    return 0


def _channel_power(dataset: RoundDataset, batch_size: int) -> np.ndarray:
    output = np.empty(len(dataset.train_channel), dtype=np.float64)
    for start in range(0, len(output), batch_size):
        batch = np.asarray(dataset.train_channel[start : start + batch_size])
        output[start : start + len(batch)] = np.sum(
            np.abs(batch) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
    return output


def _path_features(
    source: GeometryPrior,
    building: GeometryPrior,
    bs: np.ndarray,
    positions: np.ndarray,
    sample_count: int,
) -> tuple[np.ndarray, list[str]]:
    source_names = {name: i for i, name in enumerate(source.feature_names)}
    building_names = {name: i for i, name in enumerate(building.feature_names)}
    occupancy_names = [
        name for name in building.feature_names if name.endswith("_occupancy")
    ]
    names = [
        *(f"path_{name}_fraction" for name in occupancy_names),
        "path_los_blocked_fraction",
        "path_los_max_height_excess",
        "path_los_first_blocked_fraction",
    ]
    output = np.empty((len(positions), len(names)), dtype=np.float64)
    progress = np.linspace(0.0, 1.0, sample_count)
    for row, target in enumerate(positions):
        coordinates = bs[None] + progress[:, None] * (target - bs)[None]
        source_values = source.sample_points(coordinates)
        building_values = building.sample_points(coordinates)
        elevated_fractions = [
            float(building_values[:, building_names[name]].mean())
            for name in occupancy_names
        ]
        excess = source_values[:, source_names["z_max"]] - coordinates[:, 2]
        blocked = (excess > 0.5) & (progress > 0.02) & (progress < 0.98)
        blocked_progress = progress[blocked]
        output[row] = (
            *elevated_fractions,
            float(blocked.mean()),
            float(excess.max()),
            float(blocked_progress.min()) if len(blocked_progress) else 1.0,
        )
    return output, names


def _design(
    positions: np.ndarray,
    bs: np.ndarray,
    point_features: np.ndarray | None,
    path_features: np.ndarray | None,
) -> np.ndarray:
    relative = positions - bs
    distance = np.linalg.norm(relative, axis=1, keepdims=True)
    azimuth = np.arctan2(relative[:, 1], relative[:, 0])[:, None]
    coordinate = np.column_stack(
        (
            relative[:, :2],
            distance,
            np.sin(azimuth),
            np.cos(azimuth),
        )
    )
    values = [coordinate]
    if point_features is not None:
        values.append(point_features)
    if path_features is not None:
        values.append(path_features)
    return np.concatenate(values, axis=1)


def _ridge_report(
    x: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    regularization: float,
) -> dict[str, float]:
    mean = x[train].mean(axis=0)
    scale = x[train].std(axis=0)
    scale[scale < 1e-8] = 1.0
    x_train = (x[train] - mean) / scale
    x_validation = (x[validation] - mean) / scale
    x_train = np.column_stack((np.ones(len(x_train)), x_train))
    x_validation = np.column_stack((np.ones(len(x_validation)), x_validation))
    penalty = np.eye(x_train.shape[1], dtype=np.float64) * regularization
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        x_train.T @ x_train + penalty, x_train.T @ target[train]
    )
    prediction = x_validation @ coefficients
    truth = target[validation]
    residual = truth - prediction
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / max(denominator, 1e-30)
    correlation = (
        0.0
        if np.std(prediction) == 0.0 or np.std(truth) == 0.0
        else float(np.corrcoef(prediction, truth)[0, 1])
    )
    return {
        "r2": r2,
        "mae": float(np.mean(np.abs(residual))),
        "correlation": correlation,
    }


def _run_audit(args: argparse.Namespace) -> int:
    dataset = RoundDataset.open(args.data_dir)
    source = GeometryPrior.load(args.geometry_cache)
    building = GeometryPrior.load(args.building_cache)
    positions = np.asarray(dataset.train_pos, dtype=np.float64)
    bs = np.asarray(dataset.config.bs_position, dtype=np.float64)
    point = building.sample_points(positions).astype(np.float64)
    path, path_names = _path_features(
        source, building, bs, positions, args.path_samples
    )
    powers = _channel_power(dataset, args.batch_size)
    positive = powers[powers > 0.0]
    floor = max(float(positive.min()) * 0.1, 1e-30)
    target = np.log10(np.maximum(powers, floor))
    split = coverage_split(
        positions, args.validation_fraction, args.grid_size, args.seed
    )
    coordinate = _design(positions, bs, None, None)
    point_only = _design(positions, bs, point, None)
    path_only = _design(positions, bs, None, path)
    real = _design(positions, bs, point, path)
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(positions))
    shuffled = _design(positions, bs, point[permutation], path[permutation])
    def compare(train_indices: np.ndarray, validation_indices: np.ndarray):
        reports = {
            "coordinate_only": _ridge_report(
                coordinate, target, train_indices, validation_indices, args.regularization
            ),
            "coordinate_plus_point_map": _ridge_report(
                point_only, target, train_indices, validation_indices, args.regularization
            ),
            "coordinate_plus_path_map": _ridge_report(
                path_only, target, train_indices, validation_indices, args.regularization
            ),
            "real_map": _ridge_report(
                real, target, train_indices, validation_indices, args.regularization
            ),
            "shuffled_map": _ridge_report(
                shuffled, target, train_indices, validation_indices, args.regularization
            ),
        }
        return {
            "models": reports,
            "gates": {
                "real_minus_coordinate_r2": float(
                    reports["real_map"]["r2"] - reports["coordinate_only"]["r2"]
                ),
                "real_minus_shuffle_r2": float(
                    reports["real_map"]["r2"] - reports["shuffled_map"]["r2"]
                ),
                "map_has_independent_signal": bool(
                    reports["real_map"]["r2"] > reports["coordinate_only"]["r2"]
                    and reports["real_map"]["r2"] > reports["shuffled_map"]["r2"]
                ),
            },
            "train_count": int(len(train_indices)),
            "validation_count": int(len(validation_indices)),
        }

    nonzero_train = split.train[powers[split.train] > 0.0]
    nonzero_validation = split.validation[powers[split.validation] > 0.0]
    all_report = compare(split.train, split.validation)
    nonzero_report = compare(nonzero_train, nonzero_validation)
    feature_names = [
        "relative_x", "relative_y", "distance_to_bs", "sin_azimuth", "cos_azimuth",
        *building.feature_names,
        *path_names,
    ]
    _write(
        args.output,
        {
            "configuration": {
                "seed": args.seed,
                "validation_fraction": args.validation_fraction,
                "grid_size": args.grid_size,
                "regularization": args.regularization,
                "path_samples": args.path_samples,
                "power_floor": floor,
            },
            "split": {
                "train_count": int(len(split.train)),
                "validation_count": int(len(split.validation)),
                "overlap_count": int(
                    np.intersect1d(split.train, split.validation).size
                ),
            },
            "feature_names": feature_names,
            "models": all_report["models"],
            "gates": all_report["gates"],
            "protocols": {
                "all_labels": all_report,
                "nonzero_labels": nonzero_report,
            },
            "zero_power_count": int(np.sum(powers == 0.0)),
        },
    )
    return 0


def _floats(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not values:
        raise argparse.ArgumentTypeError("at least one number is required")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--geometry-cache", required=True)
    build.add_argument("--output-cache", required=True)
    build.add_argument("--report", required=True)
    build.add_argument("--user-height", type=float, default=1.5)
    build.add_argument("--thresholds", type=_floats, default=(2.5, 5.0, 10.0, 15.0))
    build.add_argument("--density-windows", type=_floats, default=(5.0, 15.0, 31.0))
    build.set_defaults(handler=_run_build)
    audit = commands.add_parser("audit")
    audit.add_argument("--data-dir", required=True)
    audit.add_argument("--geometry-cache", required=True)
    audit.add_argument("--building-cache", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--validation-fraction", type=float, default=0.1)
    audit.add_argument("--grid-size", type=float, default=20.0)
    audit.add_argument("--seed", type=int, default=42)
    audit.add_argument("--batch-size", type=int, default=8)
    audit.add_argument("--path-samples", type=int, default=64)
    audit.add_argument("--regularization", type=float, default=1.0)
    audit.set_defaults(handler=_run_audit)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
