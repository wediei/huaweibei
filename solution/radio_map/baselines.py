"""Memory-conscious local interpolation baselines."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from scipy.spatial import cKDTree


class _LocalRegressor:
    def fit(
        self,
        positions: np.ndarray,
        channels: np.ndarray,
        channel_indices: np.ndarray | None = None,
    ) -> "_LocalRegressor":
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] < 2 or len(positions) == 0:
            raise ValueError("positions must have shape (nonzero samples, dimensions>=2)")
        if not np.isfinite(positions).all():
            raise ValueError("positions must contain only finite values")
        if np.asarray(channels).ndim != 4 or not np.iscomplexobj(channels):
            raise ValueError("channels must be a complex array with shape (P, M, N, S)")

        if channel_indices is None:
            if len(channels) != len(positions):
                raise ValueError("channels and positions must have equal sample counts")
            channel_indices = np.arange(len(positions), dtype=np.int64)
        else:
            channel_indices = np.asarray(channel_indices, dtype=np.int64)
            if channel_indices.shape != (len(positions),):
                raise ValueError("channel_indices must have one entry per anchor position")
            if channel_indices.min() < 0 or channel_indices.max() >= len(channels):
                raise IndexError("channel_indices address outside the channel source")

        self.positions_ = positions
        self.channels_ = channels
        self.channel_indices_ = channel_indices
        self.tree_ = cKDTree(positions)
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "tree_"):
            raise RuntimeError("regressor must be fitted before prediction")

    def _validate_queries(self, positions: np.ndarray) -> np.ndarray:
        self._check_fitted()
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != self.positions_.shape[1]:
            raise ValueError("query positions must match anchor coordinate dimensions")
        return positions

    def predict_batches(
        self, positions: np.ndarray, batch_size: int = 1
    ) -> Iterator[np.ndarray]:
        positions = self._validate_queries(positions)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(positions), batch_size):
            yield self.predict(positions[start : start + batch_size])


class NearestAnchorRegressor(_LocalRegressor):
    """Copy the full complex channel of the nearest training position."""

    def predict(self, positions: np.ndarray) -> np.ndarray:
        positions = self._validate_queries(positions)
        _, local_indices = self.tree_.query(positions, k=1)
        source_indices = self.channel_indices_[np.asarray(local_indices, dtype=np.int64)]
        return np.asarray(self.channels_[source_indices], dtype=np.complex64)


class InverseDistanceRegressor(_LocalRegressor):
    """Interpolate complex channels from k nearest anchors by inverse distance."""

    def __init__(self, k: int = 4, power: float = 2.0, eps: float = 1e-6) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        if power <= 0.0 or eps <= 0.0:
            raise ValueError("power and eps must be positive")
        self.k = int(k)
        self.power = float(power)
        self.eps = float(eps)

    def predict(self, positions: np.ndarray) -> np.ndarray:
        positions = self._validate_queries(positions)
        neighbor_count = min(self.k, len(self.positions_))
        distances, local_indices = self.tree_.query(positions, k=neighbor_count)
        distances = np.asarray(distances, dtype=np.float64).reshape(
            len(positions), neighbor_count
        )
        local_indices = np.asarray(local_indices, dtype=np.int64).reshape(
            len(positions), neighbor_count
        )
        source_indices = self.channel_indices_[local_indices]
        neighbors = np.asarray(self.channels_[source_indices], dtype=np.complex64)

        prediction = np.empty(
            (len(positions),) + tuple(self.channels_.shape[1:]), dtype=np.complex64
        )
        exact_rows = distances[:, 0] <= self.eps
        if np.any(exact_rows):
            prediction[exact_rows] = neighbors[exact_rows, 0]
        interpolate_rows = ~exact_rows
        if np.any(interpolate_rows):
            selected_distances = distances[interpolate_rows]
            weights = 1.0 / np.maximum(selected_distances, self.eps) ** self.power
            weights /= weights.sum(axis=1, keepdims=True)
            selected_neighbors = neighbors[interpolate_rows]
            prediction[interpolate_rows] = np.einsum(
                "bk,bk...->b...", weights, selected_neighbors, optimize=True
            ).astype(np.complex64)
        return prediction

