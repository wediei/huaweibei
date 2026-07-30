"""Memory-mapped PLY reading and unified 2.5-D geometry priors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy.ndimage import distance_transform_edt


_PLY_DTYPES = {
    "char": "i1",
    "uchar": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "float": "<f4",
    "double": "<f8",
    "int8": "i1",
    "uint8": "u1",
    "int16": "<i2",
    "uint16": "<u2",
    "int32": "<i4",
    "uint32": "<u4",
    "float32": "<f4",
    "float64": "<f8",
}


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PlyPointCloud:
    path: Path
    vertex_count: int
    header_bytes: int
    dtype: np.dtype
    vertices: np.memmap

    @classmethod
    def open(cls, path: str | Path) -> "PlyPointCloud":
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PLY file does not exist: {path}")
        with path.open("rb") as handle:
            first = handle.readline()
            if first.rstrip(b"\r\n") != b"ply":
                raise ValueError("not a PLY file")
            format_name: str | None = None
            vertex_count: int | None = None
            vertex_properties: list[tuple[str, str]] = []
            current_element: str | None = None
            while True:
                raw_line = handle.readline()
                if not raw_line:
                    raise ValueError("PLY header is missing end_header")
                try:
                    line = raw_line.decode("ascii").strip()
                except UnicodeDecodeError as error:
                    raise ValueError("PLY header must be ASCII") from error
                tokens = line.split()
                if not tokens or tokens[0] == "comment":
                    continue
                if tokens[0] == "format":
                    format_name = tokens[1]
                elif tokens[0] == "element":
                    current_element = tokens[1]
                    if current_element == "vertex":
                        vertex_count = int(tokens[2])
                elif tokens[0] == "property" and current_element == "vertex":
                    if len(tokens) != 3 or tokens[1] == "list":
                        raise ValueError("list-valued vertex properties are unsupported")
                    if tokens[1] not in _PLY_DTYPES:
                        raise ValueError(f"unsupported PLY property type: {tokens[1]}")
                    vertex_properties.append((tokens[2], _PLY_DTYPES[tokens[1]]))
                elif tokens[0] == "end_header":
                    header_bytes = handle.tell()
                    break

        if format_name != "binary_little_endian":
            raise ValueError(
                "only binary_little_endian PLY is supported, "
                f"got {format_name!r}"
            )
        if vertex_count is None or vertex_count <= 0:
            raise ValueError("PLY header must declare a positive vertex count")
        names = {name for name, _ in vertex_properties}
        required = {"x", "y", "z", "nx", "ny", "nz"}
        if not required.issubset(names):
            raise ValueError(
                f"PLY vertex properties must include {sorted(required)}, got {sorted(names)}"
            )
        dtype = np.dtype(vertex_properties)
        required_bytes = header_bytes + vertex_count * dtype.itemsize
        if path.stat().st_size < required_bytes:
            raise ValueError(
                f"PLY vertex payload is truncated: need at least {required_bytes} bytes, "
                f"got {path.stat().st_size}"
            )
        vertices = np.memmap(
            path,
            dtype=dtype,
            mode="r",
            offset=header_bytes,
            shape=(vertex_count,),
        )
        return cls(path, vertex_count, header_bytes, dtype, vertices)

    def iter_batches(
        self, batch_size: int = 131072
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, self.vertex_count, batch_size):
            records = self.vertices[start : start + batch_size]
            positions = np.column_stack(
                (records["x"], records["y"], records["z"])
            ).astype(np.float64, copy=False)
            normals = np.column_stack(
                (records["nx"], records["ny"], records["nz"])
            ).astype(np.float64, copy=False)
            yield positions, normals


@dataclass(frozen=True)
class GeometryPrior:
    features: np.ndarray
    feature_names: tuple[str, ...]
    origin_xy: tuple[float, float]
    resolution: float
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        if self.features.ndim != 3:
            raise ValueError("geometry features must have shape (channels, height, width)")
        if self.features.shape[0] != len(self.feature_names):
            raise ValueError("feature_names must match the channel count")
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(self.metadata)
        metadata.update(
            {
                "feature_names": list(self.feature_names),
                "origin_xy": list(self.origin_xy),
                "resolution": float(self.resolution),
            }
        )
        np.savez_compressed(
            path,
            features=np.asarray(self.features, dtype=np.float32),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "GeometryPrior":
        with np.load(Path(path), allow_pickle=False) as archive:
            features = np.asarray(archive["features"], dtype=np.float32)
            metadata = json.loads(str(archive["metadata"].item()))
        feature_names = tuple(str(name) for name in metadata.pop("feature_names"))
        origin_xy = tuple(float(value) for value in metadata.pop("origin_xy"))
        resolution = float(metadata.pop("resolution"))
        return cls(features, feature_names, origin_xy, resolution, metadata)

    def _cell_indices(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] < 2:
            raise ValueError("positions must have shape (samples, dimensions>=2)")
        ix = np.floor((positions[:, 0] - self.origin_xy[0]) / self.resolution)
        iy = np.floor((positions[:, 1] - self.origin_xy[1]) / self.resolution)
        return ix.astype(np.int64), iy.astype(np.int64)

    def sample_points(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float64)
        ix, iy = self._cell_indices(positions)
        output = np.zeros((len(positions), len(self.feature_names)), dtype=np.float32)
        valid = (
            (ix >= 0)
            & (ix < self.features.shape[2])
            & (iy >= 0)
            & (iy < self.features.shape[1])
        )
        output[valid] = self.features[:, iy[valid], ix[valid]].T
        return output

    def extract_patch(self, position: np.ndarray, size: int) -> np.ndarray:
        if size <= 0 or size % 2 == 0:
            raise ValueError("patch size must be a positive odd integer")
        position = np.asarray(position, dtype=np.float64).reshape(1, -1)
        ix, iy = self._cell_indices(position)
        center_x, center_y = int(ix[0]), int(iy[0])
        radius = size // 2
        output = np.zeros(
            (len(self.feature_names), size, size), dtype=np.float32
        )
        source_x0 = max(0, center_x - radius)
        source_x1 = min(self.features.shape[2], center_x + radius + 1)
        source_y0 = max(0, center_y - radius)
        source_y1 = min(self.features.shape[1], center_y + radius + 1)
        if source_x0 >= source_x1 or source_y0 >= source_y1:
            return output
        target_x0 = source_x0 - (center_x - radius)
        target_y0 = source_y0 - (center_y - radius)
        target_x1 = target_x0 + source_x1 - source_x0
        target_y1 = target_y0 + source_y1 - source_y0
        output[:, target_y0:target_y1, target_x0:target_x1] = self.features[
            :, source_y0:source_y1, source_x0:source_x1
        ]
        return output

    def corridor_features(
        self,
        start_position: np.ndarray,
        end_position: np.ndarray,
        sample_count: int,
    ) -> np.ndarray:
        if sample_count < 2:
            raise ValueError("corridor sample_count must be at least two")
        start = np.asarray(start_position, dtype=np.float64).reshape(-1)
        end = np.asarray(end_position, dtype=np.float64).reshape(-1)
        if start.shape != end.shape or len(start) < 2:
            raise ValueError("corridor endpoints must have matching coordinate shapes")
        progress = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
        positions = start[None, :] + progress[:, None] * (end - start)[None, :]
        sampled = self.sample_points(positions)
        length = float(np.linalg.norm(end - start))
        return np.column_stack((progress, progress * length, sampled)).astype(
            np.float32
        )


def build_geometry_prior(
    cloud: PlyPointCloud,
    resolution: float = 1.0,
    height_layers: int = 4,
    batch_size: int = 131072,
) -> GeometryPrior:
    """Rasterize positions and normals into a single auditable geometry grid."""

    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    if height_layers <= 0:
        raise ValueError("height_layers must be positive")
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    valid_count = 0
    invalid_count = 0
    for positions, normals in cloud.iter_batches(batch_size):
        valid = np.isfinite(positions).all(axis=1) & np.isfinite(normals).all(axis=1)
        invalid_count += int((~valid).sum())
        if np.any(valid):
            minimum = np.minimum(minimum, positions[valid].min(axis=0))
            maximum = np.maximum(maximum, positions[valid].max(axis=0))
            valid_count += int(valid.sum())
    if valid_count == 0:
        raise ValueError("PLY contains no finite position/normal records")

    origin = np.floor(minimum[:2] / resolution) * resolution
    width = int(np.floor((maximum[0] - origin[0]) / resolution)) + 1
    height = int(np.floor((maximum[1] - origin[1]) / resolution)) + 1
    cell_count = width * height
    counts = np.zeros(cell_count, dtype=np.int64)
    z_min = np.full(cell_count, np.inf, dtype=np.float64)
    z_max = np.full(cell_count, -np.inf, dtype=np.float64)
    normal_z_sum = np.zeros(cell_count, dtype=np.float64)
    wall_sum = np.zeros(cell_count, dtype=np.float64)
    layer_counts = np.zeros((height_layers, cell_count), dtype=np.int64)
    if np.isclose(maximum[2], minimum[2]):
        height_edges = np.linspace(
            minimum[2] - 0.5, maximum[2] + 0.5, height_layers + 1
        )
    else:
        height_edges = np.linspace(minimum[2], maximum[2], height_layers + 1)

    for positions, normals in cloud.iter_batches(batch_size):
        valid = np.isfinite(positions).all(axis=1) & np.isfinite(normals).all(axis=1)
        positions = positions[valid]
        normals = normals[valid]
        if len(positions) == 0:
            continue
        ix = np.floor((positions[:, 0] - origin[0]) / resolution).astype(np.int64)
        iy = np.floor((positions[:, 1] - origin[1]) / resolution).astype(np.int64)
        ix = np.clip(ix, 0, width - 1)
        iy = np.clip(iy, 0, height - 1)
        flat = iy * width + ix
        np.add.at(counts, flat, 1)
        np.minimum.at(z_min, flat, positions[:, 2])
        np.maximum.at(z_max, flat, positions[:, 2])
        normal_length = np.linalg.norm(normals, axis=1)
        normalized_z = np.divide(
            normals[:, 2],
            normal_length,
            out=np.zeros(len(normals), dtype=np.float64),
            where=normal_length > 0.0,
        )
        np.add.at(normal_z_sum, flat, normalized_z)
        np.add.at(wall_sum, flat, 1.0 - np.clip(np.abs(normalized_z), 0.0, 1.0))
        bins = np.searchsorted(height_edges, positions[:, 2], side="right") - 1
        bins = np.clip(bins, 0, height_layers - 1)
        np.add.at(layer_counts, (bins, flat), 1)

    occupancy = counts.reshape(height, width) > 0
    if not np.any(occupancy):
        raise ValueError("geometry grid contains no occupied cells")
    _, nearest = distance_transform_edt(~occupancy, return_indices=True)
    z_min_grid = z_min.reshape(height, width)
    z_max_grid = z_max.reshape(height, width)
    filled_z_min = z_min_grid[tuple(nearest)]
    filled_z_max = z_max_grid[tuple(nearest)]
    count_grid = counts.reshape(height, width)
    max_count = int(count_grid.max())
    log_density = np.log1p(count_grid) / np.log1p(max_count)
    mean_normal_z = np.divide(
        normal_z_sum.reshape(height, width),
        count_grid,
        out=np.zeros((height, width), dtype=np.float64),
        where=count_grid > 0,
    )
    wall_strength = np.divide(
        wall_sum.reshape(height, width),
        count_grid,
        out=np.zeros((height, width), dtype=np.float64),
        where=count_grid > 0,
    )
    gradient_y = (
        np.gradient(filled_z_max, resolution, axis=0)
        if height > 1
        else np.zeros_like(filled_z_max)
    )
    gradient_x = (
        np.gradient(filled_z_max, resolution, axis=1)
        if width > 1
        else np.zeros_like(filled_z_max)
    )
    height_edge = np.hypot(gradient_x, gradient_y)
    distance = distance_transform_edt(~occupancy, sampling=resolution)
    layer_occupancy = np.divide(
        layer_counts.reshape(height_layers, height, width),
        count_grid[None, :, :],
        out=np.zeros((height_layers, height, width), dtype=np.float64),
        where=count_grid[None, :, :] > 0,
    )
    feature_names = (
        "occupancy",
        "log_density",
        "z_min",
        "z_max",
        "height_range",
        "mean_normal_z",
        "wall_strength",
        "height_edge",
        "distance_to_occupancy",
    ) + tuple(f"height_layer_{index}_occupancy" for index in range(height_layers))
    features = np.concatenate(
        (
            np.stack(
                (
                    occupancy.astype(np.float64),
                    log_density,
                    filled_z_min,
                    filled_z_max,
                    filled_z_max - filled_z_min,
                    mean_normal_z,
                    wall_strength,
                    height_edge,
                    distance,
                ),
                axis=0,
            ),
            layer_occupancy,
        ),
        axis=0,
    ).astype(np.float32)
    metadata: dict[str, object] = {
        "source_path": str(cloud.path),
        "source_sha256": _sha256_file(cloud.path),
        "source_vertex_count": int(cloud.vertex_count),
        "point_count": int(valid_count),
        "invalid_point_count": int(invalid_count),
        "world_min_xyz": minimum.tolist(),
        "world_max_xyz": maximum.tolist(),
        "grid_shape": [height, width],
        "height_edges": height_edges.tolist(),
        "height_layers": int(height_layers),
    }
    return GeometryPrior(
        features=features,
        feature_names=feature_names,
        origin_xy=(float(origin[0]), float(origin[1])),
        resolution=float(resolution),
        metadata=metadata,
    )

