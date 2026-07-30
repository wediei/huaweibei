"""Leakage-safe radio Anchor memory and unified geometry prior batches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .geometry import GeometryPrior


PAIR_FEATURE_NAMES = (
    "relative_x",
    "relative_y",
    "relative_z",
    "anchor_distance",
    "direction_x",
    "direction_y",
    "direction_z",
    "anchor_bs_distance",
    "target_bs_distance",
    "bs_distance_delta",
    "azimuth_delta_sin",
    "azimuth_delta_cos",
    "elevation_delta_sin",
    "elevation_delta_cos",
)


@dataclass(frozen=True)
class AnchorQuery:
    target_positions: np.ndarray
    local_indices: np.ndarray
    source_indices: np.ndarray
    distances: np.ndarray
    relative_positions: np.ndarray
    pair_features: np.ndarray
    latents: np.ndarray


class AnchorMemory:
    """Spatial index over channel latents with auditable global source indices."""

    def __init__(
        self,
        positions: np.ndarray,
        latents: np.ndarray,
        bs_position: np.ndarray,
        source_indices: np.ndarray | None = None,
    ) -> None:
        positions = np.asarray(positions, dtype=np.float64)
        latents = np.asarray(latents)
        bs_position = np.asarray(bs_position, dtype=np.float64).reshape(-1)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
            raise ValueError("anchor positions must have shape (nonzero samples, 3)")
        if not np.isfinite(positions).all():
            raise ValueError("anchor positions must be finite")
        if latents.ndim < 2 or len(latents) != len(positions):
            raise ValueError("latents must have one non-scalar entry per anchor")
        if bs_position.shape != (3,) or not np.isfinite(bs_position).all():
            raise ValueError("bs_position must contain three finite coordinates")
        if source_indices is None:
            source_indices = np.arange(len(positions), dtype=np.int64)
        else:
            source_indices = np.asarray(source_indices, dtype=np.int64)
        if source_indices.shape != (len(positions),):
            raise ValueError("source_indices must have one entry per anchor")
        if len(np.unique(source_indices)) != len(source_indices):
            raise ValueError("source_indices must be unique")
        self.positions = positions
        self.latents = latents
        self.bs_position = bs_position
        self.source_indices = source_indices
        self.tree = cKDTree(positions)

    def query(
        self,
        target_positions: np.ndarray,
        k: int,
        exclude_source_indices: np.ndarray | None = None,
    ) -> AnchorQuery:
        target_positions = np.asarray(target_positions, dtype=np.float64)
        if target_positions.ndim != 2 or target_positions.shape[1] != 3:
            raise ValueError("target positions must have shape (samples, 3)")
        if not np.isfinite(target_positions).all():
            raise ValueError("target positions must be finite")
        if k <= 0:
            raise ValueError("k must be positive")
        if exclude_source_indices is None:
            excluded = np.full(len(target_positions), -1, dtype=np.int64)
            maximum_k = len(self.positions)
            candidate_count = min(len(self.positions), k)
        else:
            excluded = np.asarray(exclude_source_indices, dtype=np.int64)
            if excluded.shape != (len(target_positions),):
                raise ValueError(
                    "exclude_source_indices must have one entry per target"
                )
            maximum_k = len(self.positions) - 1
            candidate_count = min(len(self.positions), k + 1)
        if k > maximum_k:
            raise ValueError(
                f"k={k} exceeds the {maximum_k} anchors available after exclusion"
            )

        candidate_distances, candidate_local = self.tree.query(
            target_positions, k=candidate_count
        )
        candidate_distances = np.asarray(candidate_distances, dtype=np.float64).reshape(
            len(target_positions), candidate_count
        )
        candidate_local = np.asarray(candidate_local, dtype=np.int64).reshape(
            len(target_positions), candidate_count
        )
        local_indices = np.empty((len(target_positions), k), dtype=np.int64)
        distances = np.empty((len(target_positions), k), dtype=np.float64)
        for row in range(len(target_positions)):
            local = candidate_local[row]
            distance = candidate_distances[row]
            keep = self.source_indices[local] != excluded[row]
            local = local[keep]
            distance = distance[keep]
            order = np.lexsort((self.source_indices[local], distance))
            local = local[order][:k]
            distance = distance[order][:k]
            if len(local) != k:
                raise RuntimeError("insufficient anchors after target exclusion")
            local_indices[row] = local
            distances[row] = distance

        anchor_positions = self.positions[local_indices]
        relative = anchor_positions - target_positions[:, None, :]
        direction = np.divide(
            relative,
            distances[:, :, None],
            out=np.zeros_like(relative),
            where=distances[:, :, None] > 0.0,
        )
        anchor_bs_vector = anchor_positions - self.bs_position[None, None, :]
        target_bs_vector = target_positions - self.bs_position[None, :]
        anchor_bs_distance = np.linalg.norm(anchor_bs_vector, axis=-1)
        target_bs_distance = np.linalg.norm(target_bs_vector, axis=-1)
        anchor_azimuth = np.arctan2(anchor_bs_vector[..., 1], anchor_bs_vector[..., 0])
        target_azimuth = np.arctan2(
            target_bs_vector[:, 1], target_bs_vector[:, 0]
        )
        azimuth_delta = anchor_azimuth - target_azimuth[:, None]
        anchor_horizontal = np.linalg.norm(anchor_bs_vector[..., :2], axis=-1)
        target_horizontal = np.linalg.norm(target_bs_vector[:, :2], axis=-1)
        anchor_elevation = np.arctan2(anchor_bs_vector[..., 2], anchor_horizontal)
        target_elevation = np.arctan2(target_bs_vector[:, 2], target_horizontal)
        elevation_delta = anchor_elevation - target_elevation[:, None]
        pair_features = np.concatenate(
            (
                relative,
                distances[:, :, None],
                direction,
                anchor_bs_distance[:, :, None],
                np.broadcast_to(
                    target_bs_distance[:, None, None],
                    (len(target_positions), k, 1),
                ),
                (anchor_bs_distance - target_bs_distance[:, None])[:, :, None],
                np.sin(azimuth_delta)[:, :, None],
                np.cos(azimuth_delta)[:, :, None],
                np.sin(elevation_delta)[:, :, None],
                np.cos(elevation_delta)[:, :, None],
            ),
            axis=-1,
        ).astype(np.float32)
        return AnchorQuery(
            target_positions=target_positions.astype(np.float32),
            local_indices=local_indices,
            source_indices=self.source_indices[local_indices],
            distances=distances.astype(np.float32),
            relative_positions=relative.astype(np.float32),
            pair_features=pair_features,
            latents=np.asarray(self.latents[local_indices]),
        )


def build_prior_batch(
    memory: AnchorMemory,
    target_positions: np.ndarray,
    geometry: GeometryPrior,
    k: int,
    exclude_source_indices: np.ndarray | None = None,
    patch_size: int = 9,
    corridor_samples: int = 32,
    anchor_corridor_samples: int = 8,
) -> dict[str, object]:
    """Build the single prior dictionary consumed by future model adapters."""

    query = memory.query(target_positions, k, exclude_source_indices)
    targets = np.asarray(target_positions, dtype=np.float64)
    batch_count = len(targets)
    bs_positions = np.broadcast_to(memory.bs_position, (batch_count, 3)).copy()
    target_map_features = geometry.sample_points(targets)
    bs_map_features = geometry.sample_points(bs_positions)
    local_patches = np.stack(
        [geometry.extract_patch(position, patch_size) for position in targets]
    )
    bs_patch = geometry.extract_patch(memory.bs_position, patch_size)
    bs_patches = np.broadcast_to(
        bs_patch, (batch_count,) + bs_patch.shape
    ).copy()
    corridors = np.stack(
        [
            geometry.corridor_features(
                memory.bs_position, target, corridor_samples
            )
            for target in targets
        ]
    )
    anchor_corridors = np.empty(
        (
            batch_count,
            k,
            anchor_corridor_samples,
            len(geometry.feature_names) + 2,
        ),
        dtype=np.float32,
    )
    anchor_positions = memory.positions[query.local_indices]
    for batch_index in range(batch_count):
        for anchor_index in range(k):
            anchor_corridors[batch_index, anchor_index] = geometry.corridor_features(
                anchor_positions[batch_index, anchor_index],
                targets[batch_index],
                anchor_corridor_samples,
            )
    return {
        "position": targets.astype(np.float32),
        "prior_info": {
            "bs_position": bs_positions.astype(np.float32),
            "target_map_features": target_map_features,
            "bs_map_features": bs_map_features,
            "local_map_patch": local_patches,
            "bs_map_patch": bs_patches,
            "corridor_features": corridors,
            "anchor_indices": query.source_indices,
            "anchor_distances": query.distances,
            "anchor_relative_pos": query.relative_positions,
            "anchor_pair_features": query.pair_features,
            "anchor_latents": query.latents,
            "anchor_path_features": anchor_corridors,
        },
    }
