"""Ground-suppressed building priors derived from the auditable 2.5-D grid."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    uniform_filter,
)

from .geometry import GeometryPrior


def build_building_prior(
    geometry: GeometryPrior,
    *,
    user_height: float = 1.5,
    elevated_thresholds: Iterable[float] = (2.5, 5.0, 10.0, 15.0),
    density_windows_metres: Iterable[float] = (5.0, 15.0, 31.0),
) -> GeometryPrior:
    """Create a parallel cache that suppresses the near-continuous ground layer."""

    thresholds = tuple(float(value) for value in elevated_thresholds)
    windows = tuple(float(value) for value in density_windows_metres)
    if not thresholds or any(value <= user_height for value in thresholds):
        raise ValueError("elevated thresholds must all exceed user_height")
    if tuple(sorted(set(thresholds))) != thresholds:
        raise ValueError("elevated thresholds must be unique and increasing")
    if not windows or any(value <= 0.0 for value in windows):
        raise ValueError("density windows must be positive")
    names = {name: index for index, name in enumerate(geometry.feature_names)}
    if "z_max" not in names or "occupancy" not in names:
        raise ValueError("source geometry must contain z_max and occupancy")
    z_max = np.asarray(geometry.features[names["z_max"]], dtype=np.float32)
    raw_occupancy = geometry.features[names["occupancy"]] > 0.0
    features: list[np.ndarray] = [
        np.maximum(z_max - float(user_height), 0.0),
    ]
    feature_names = ["height_above_user"]
    for threshold in thresholds:
        elevated = raw_occupancy & (z_max >= threshold)
        boundary = binary_dilation(elevated) ^ binary_erosion(elevated)
        features.extend(
            (
                elevated.astype(np.float32),
                distance_transform_edt(
                    ~elevated, sampling=geometry.resolution
                ).astype(np.float32),
                distance_transform_edt(
                    ~boundary, sampling=geometry.resolution
                ).astype(np.float32),
            )
        )
        label = f"{threshold:g}".replace(".", "p")
        feature_names.extend(
            (
                f"elevated_{label}_occupancy",
                f"distance_to_elevated_{label}",
                f"distance_to_elevated_{label}_edge",
            )
        )
    base_elevated = raw_occupancy & (z_max >= thresholds[0])
    for metres in windows:
        cells = max(1, int(round(metres / geometry.resolution)))
        if cells % 2 == 0:
            cells += 1
        features.append(
            uniform_filter(
                base_elevated.astype(np.float32),
                size=cells,
                mode="nearest",
            )
        )
        label = f"{metres:g}".replace(".", "p")
        feature_names.append(f"elevated_density_{label}m")
    metadata = dict(geometry.metadata)
    metadata.update(
        {
            "kind": "ground_suppressed_building_prior",
            "format_version": 1,
            "source_geometry_feature_names": list(geometry.feature_names),
            "user_height": float(user_height),
            "elevated_thresholds": list(thresholds),
            "density_windows_metres": list(windows),
        }
    )
    return GeometryPrior(
        features=np.stack(features).astype(np.float32),
        feature_names=tuple(feature_names),
        origin_xy=geometry.origin_xy,
        resolution=geometry.resolution,
        metadata=metadata,
    )
