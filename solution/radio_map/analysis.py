"""Deterministic dataset coverage and antenna-layout diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

from .data import RoundDataset
from .splits import nearest_anchor_distances
from .transforms import (
    AntennaLayout,
    beam_delay,
    candidate_orders,
    inverse_beam_delay,
)


def _coordinate_ranges(positions: np.ndarray) -> dict[str, list[float]]:
    return {
        axis: [float(positions[:, index].min()), float(positions[:, index].max())]
        for index, axis in enumerate(("x", "y", "z"))
    }


def _inside_convex_hull(train_xy: np.ndarray, test_xy: np.ndarray) -> np.ndarray:
    try:
        hull = ConvexHull(train_xy)
    except QhullError:
        lower = train_xy.min(axis=0)
        upper = train_xy.max(axis=0)
        return np.all((test_xy >= lower) & (test_xy <= upper), axis=1)
    equations = hull.equations
    return np.all(
        test_xy @ equations[:, :-1].T + equations[:, -1] <= 1e-9,
        axis=1,
    )


def coverage_report(dataset: RoundDataset) -> dict[str, object]:
    """Describe official train/test spatial coverage in metre coordinates."""

    train = np.asarray(dataset.train_pos, dtype=np.float64)
    test = np.asarray(dataset.test_pos, dtype=np.float64)
    distances = nearest_anchor_distances(train, test)
    quantile_levels = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    quantiles = np.quantile(distances, quantile_levels)
    inside = _inside_convex_hull(train[:, :2], test[:, :2])
    train_constant_z = bool(np.allclose(train[:, 2], train[0, 2]))
    test_constant_z = bool(np.allclose(test[:, 2], test[0, 2]))
    return {
        "coordinate_unit": "metre",
        "train_count": int(len(train)),
        "test_count": int(len(test)),
        "train_ranges": _coordinate_ranges(train),
        "test_ranges": _coordinate_ranges(test),
        "constant_z": {
            "train": train_constant_z,
            "test": test_constant_z,
            "same_height": bool(
                train_constant_z
                and test_constant_z
                and np.isclose(train[0, 2], test[0, 2])
            ),
        },
        "test_inside_train_xy_convex_hull": {
            "count": int(inside.sum()),
            "fraction": float(inside.mean()),
        },
        "test_to_train_nearest_distance": {
            f"q{int(level * 100):02d}": float(value)
            for level, value in zip(quantile_levels, quantiles)
        },
        "within_5m_fraction": float(np.mean(distances <= 5.0)),
    }


def _top_energy_fraction(power: np.ndarray, fraction: float) -> float:
    flat = power.reshape(len(power), -1)
    count = max(1, int(np.ceil(flat.shape[1] * fraction)))
    top = np.partition(flat, flat.shape[1] - count, axis=1)[:, -count:]
    total = flat.sum(axis=1, dtype=np.float64)
    captured = top.sum(axis=1, dtype=np.float64)
    ratio = np.divide(captured, total, out=np.ones_like(captured), where=total > 0)
    return float(ratio.mean())


def _row_power_cosine(first: np.ndarray, second: np.ndarray) -> float:
    first = first.reshape(len(first), -1).astype(np.float64, copy=False)
    second = second.reshape(len(second), -1).astype(np.float64, copy=False)
    numerator = np.einsum("ij,ij->i", first, second)
    denominator = np.sqrt(
        np.einsum("ij,ij->i", first, first)
        * np.einsum("ij,ij->i", second, second)
    )
    cosine = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
    )
    cosine[(denominator == 0)] = 1.0
    return float(np.mean(np.clip(cosine, 0.0, 1.0)))


def layout_report(
    dataset: RoundDataset,
    sample_count: int = 20,
    seed: int = 42,
) -> dict[str, object]:
    """Compare all antenna flattening orders on identical channels and neighbors."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    sample_count = min(int(sample_count), len(dataset.train_pos))
    rng = np.random.default_rng(seed)
    sample_indices = np.sort(
        rng.choice(len(dataset.train_pos), size=sample_count, replace=False)
    ).astype(np.int64)
    tree = cKDTree(np.asarray(dataset.train_pos, dtype=np.float64))
    _, neighbor_indices = tree.query(dataset.train_pos[sample_indices], k=2)
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)[:, 1]
    channels = dataset.channel_batch(sample_indices)
    neighbor_channels = dataset.channel_batch(neighbor_indices)

    candidates: list[dict[str, object]] = []
    for order in candidate_orders():
        layout = AntennaLayout(dataset.config, order)
        transformed = beam_delay(channels, layout)
        reconstructed = inverse_beam_delay(transformed, layout)
        neighbor_transformed = beam_delay(neighbor_channels, layout)
        power = np.abs(transformed) ** 2
        neighbor_power = np.abs(neighbor_transformed) ** 2
        candidates.append(
            {
                "order": "".join(order),
                "roundtrip_max_abs_error": float(
                    np.max(np.abs(reconstructed - channels))
                ),
                "top_energy_fraction": {
                    "top_1_percent": _top_energy_fraction(power, 0.01),
                    "top_5_percent": _top_energy_fraction(power, 0.05),
                    "top_10_percent": _top_energy_fraction(power, 0.10),
                },
                "nearest_neighbor_power_cosine": _row_power_cosine(
                    power, neighbor_power
                ),
            }
        )
    return {
        "sample_count": sample_count,
        "seed": int(seed),
        "sample_indices": sample_indices.tolist(),
        "neighbor_indices": neighbor_indices.tolist(),
        "candidates": candidates,
    }

