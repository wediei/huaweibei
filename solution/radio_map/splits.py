"""Spatial validation splits and coverage diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray


def _validate_positions(positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError("positions must have shape (samples, dimensions>=2)")
    if len(positions) < 2:
        raise ValueError("at least two positions are required")
    if not np.isfinite(positions).all():
        raise ValueError("positions must contain only finite values")
    return positions


def _validation_count(sample_count: int, validation_fraction: float) -> int:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    return min(sample_count - 1, max(1, int(round(sample_count * validation_fraction))))


def coverage_split(
    positions: np.ndarray,
    validation_fraction: float = 0.2,
    grid_size: float = 20.0,
    seed: int = 42,
) -> SplitIndices:
    """Hold out samples across occupied XY cells while retaining anchor coverage."""

    positions = _validate_positions(positions)
    if grid_size <= 0.0:
        raise ValueError("grid_size must be positive")
    target_count = _validation_count(len(positions), validation_fraction)
    xy = positions[:, :2]
    cells = np.floor((xy - xy.min(axis=0)) / grid_size).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    rng = np.random.default_rng(seed)

    cell_members: list[np.ndarray] = []
    validation: list[int] = []
    for cell_id in range(int(inverse.max()) + 1):
        members = np.flatnonzero(inverse == cell_id)
        members = rng.permutation(members)
        cell_members.append(members)
        initial_count = min(
            len(members) - 1,
            int(np.floor(len(members) * validation_fraction)),
        )
        validation.extend(int(index) for index in members[:initial_count])

    if len(validation) > target_count:
        validation = list(rng.permutation(validation)[:target_count])
    elif len(validation) < target_count:
        selected = set(validation)
        eligible: list[int] = []
        for members in cell_members:
            retained = sum(int(index) not in selected for index in members)
            if retained <= 1:
                continue
            eligible.extend(
                int(index) for index in members if int(index) not in selected
            )
        for index in rng.permutation(eligible):
            if len(validation) >= target_count:
                break
            cell_id = int(inverse[index])
            selected_in_cell = sum(
                int(member) in selected for member in cell_members[cell_id]
            )
            if selected_in_cell < len(cell_members[cell_id]) - 1:
                selected.add(int(index))
                validation.append(int(index))

    if len(validation) != target_count:
        raise ValueError(
            "requested validation size cannot retain at least one anchor in every cell; "
            "increase grid_size or reduce validation_fraction"
        )
    validation_array = np.sort(np.asarray(validation, dtype=np.int64))
    train_mask = np.ones(len(positions), dtype=bool)
    train_mask[validation_array] = False
    return SplitIndices(
        train=np.flatnonzero(train_mask).astype(np.int64),
        validation=validation_array,
    )


def block_split(
    positions: np.ndarray,
    axis: int = 0,
    validation_fraction: float = 0.2,
    side: str = "high",
) -> SplitIndices:
    """Hold out a contiguous coordinate extreme for extrapolation testing."""

    positions = _validate_positions(positions)
    if not 0 <= axis < positions.shape[1]:
        raise ValueError(f"axis must be in [0, {positions.shape[1]})")
    if side not in {"high", "low"}:
        raise ValueError("side must be 'high' or 'low'")
    count = _validation_count(len(positions), validation_fraction)
    order = np.argsort(positions[:, axis], kind="stable")
    validation = order[-count:] if side == "high" else order[:count]
    validation = np.sort(validation.astype(np.int64))
    train_mask = np.ones(len(positions), dtype=bool)
    train_mask[validation] = False
    return SplitIndices(
        train=np.flatnonzero(train_mask).astype(np.int64), validation=validation
    )


def nearest_anchor_distances(
    anchor_positions: np.ndarray,
    query_positions: np.ndarray,
) -> np.ndarray:
    anchors = _validate_positions(anchor_positions)
    queries = np.asarray(query_positions, dtype=np.float64)
    if queries.ndim != 2 or queries.shape[1] != anchors.shape[1]:
        raise ValueError("query positions must match anchor coordinate dimensions")
    distances, _ = cKDTree(anchors).query(queries, k=1)
    return np.asarray(distances, dtype=np.float64)

