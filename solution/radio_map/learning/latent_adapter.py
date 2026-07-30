"""Fold-fitted normalized fixed-support Beam-Delay latents."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..codecs import GlobalSupportCodec
from ..transforms import AntennaLayout, beam_delay, inverse_beam_delay


class FixedSupportLatentAdapter:
    """Encode fixed Beam-Delay support coefficients with fold-only statistics."""

    def __init__(
        self,
        layout: AntennaLayout,
        support_fraction: float = 0.10,
        delay_block: int = 8,
    ) -> None:
        self.layout = layout
        self.support_fraction = float(support_fraction)
        self.delay_block = int(delay_block)
        self.total_coefficients = int(np.prod(layout.structured_tail))
        if not 0.0 < self.support_fraction <= 1.0:
            raise ValueError("support_fraction must be in (0, 1]")
        if self.delay_block <= 0:
            raise ValueError("delay_block must be positive")
        self.coefficient_count = int(
            np.ceil(self.total_coefficients * self.support_fraction)
        )

    def _check_fitted(self) -> None:
        if not hasattr(self, "support_indices"):
            raise RuntimeError("adapter must be fitted before encode/decode")

    def _validate_source(self, channel_source: Any) -> Any:
        source = getattr(channel_source, "train_channel", channel_source)
        shape = getattr(source, "shape", None)
        dtype = getattr(source, "dtype", None)
        expected_tail = self.layout.config.channel_shape
        if shape is None or len(shape) != 4 or tuple(shape[1:]) != expected_tail:
            raise ValueError(
                "channel source must have shape "
                f"(P,{expected_tail[0]},{expected_tail[1]},{expected_tail[2]})"
            )
        if dtype is None or not np.issubdtype(np.dtype(dtype), np.complexfloating):
            raise TypeError("channel source must be complex-valued")
        return source

    @staticmethod
    def _validate_fit_indices(indices: np.ndarray, source_count: int) -> np.ndarray:
        validated = np.asarray(indices, dtype=np.int64)
        if validated.ndim != 1 or not len(validated):
            raise ValueError("fit_indices must be a non-empty one-dimensional array")
        if int(validated.min()) < 0 or int(validated.max()) >= source_count:
            raise IndexError("fit index outside channel source")
        if len(np.unique(validated)) != len(validated):
            raise ValueError("fit_indices must not contain duplicates")
        return validated

    @staticmethod
    def _indices_sha256(indices: np.ndarray) -> str:
        canonical = np.asarray(indices, dtype="<i8")
        return hashlib.sha256(canonical.tobytes()).hexdigest()

    def fit(
        self,
        channel_source: Any,
        fit_indices: np.ndarray,
        batch_size: int,
    ) -> "FixedSupportLatentAdapter":
        """Learn support and normalization exclusively from the given fold."""

        source = self._validate_source(channel_source)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = self._validate_fit_indices(fit_indices, int(source.shape[0]))

        support_codec = GlobalSupportCodec(self.layout, self.coefficient_count)
        support_codec.fit(source, train_indices=indices, batch_size=batch_size)
        support_indices = support_codec.support_indices.copy()

        latent_sum = np.zeros(self.coefficient_count, dtype=np.complex128)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            transformed = beam_delay(
                np.asarray(source[batch_indices], dtype=np.complex64), self.layout
            ).reshape(len(batch_indices), -1)
            latent_sum += transformed[:, support_indices].sum(axis=0, dtype=np.complex128)
        mean = latent_sum / len(indices)

        squared_deviation_sum = np.zeros(self.coefficient_count, dtype=np.float64)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            transformed = beam_delay(
                np.asarray(source[batch_indices], dtype=np.complex64), self.layout
            ).reshape(len(batch_indices), -1)
            deviation = transformed[:, support_indices] - mean
            squared_deviation_sum += np.sum(
                np.abs(deviation) ** 2, axis=0, dtype=np.float64
            )

        self.support_indices = np.asarray(support_indices, dtype=np.int64)
        self.mean = np.asarray(mean, dtype=np.complex64)
        self.rms = np.asarray(
            np.sqrt(squared_deviation_sum / len(indices)).clip(min=1e-6),
            dtype=np.float32,
        )
        self.latent_coordinates = np.stack(
            np.unravel_index(self.support_indices, self.layout.structured_tail), axis=1
        ).astype(np.int64, copy=False)
        block_count = int(np.ceil(self.layout.config.s / self.delay_block))
        coordinates = self.latent_coordinates
        self.group_ids = np.asarray(
            (coordinates[:, 2] * self.layout.config.n + coordinates[:, 3])
            * block_count
            + coordinates[:, 4] // self.delay_block,
            dtype=np.int64,
        )
        self.fitted_indices = indices.copy()
        self.fit_sample_count = int(len(indices))
        self.fit_indices_sha256 = self._indices_sha256(indices)
        self.source_shape = tuple(int(value) for value in source.shape)
        return self

    def encode_numpy(self, channels: np.ndarray) -> np.ndarray:
        self._check_fitted()
        values = np.asarray(channels)
        if values.ndim != 4 or tuple(values.shape[1:]) != self.layout.config.channel_shape:
            raise ValueError("channels must have an explicit batch dimension and layout tail")
        if not np.iscomplexobj(values):
            raise TypeError("channels must be complex-valued")
        transformed = beam_delay(values, self.layout).reshape(len(values), -1)
        latent = transformed[:, self.support_indices]
        return np.asarray((latent - self.mean) / self.rms, dtype=np.complex64)

    def decode_numpy(self, normalized_latent: np.ndarray) -> np.ndarray:
        self._check_fitted()
        latent = np.asarray(normalized_latent)
        if latent.ndim != 2 or latent.shape[1] != self.coefficient_count:
            raise ValueError(
                f"latent must have shape (B,{self.coefficient_count}), got {latent.shape}"
            )
        if not np.iscomplexobj(latent):
            raise TypeError("latent must be complex-valued")
        raw_latent = latent * self.rms + self.mean
        flat = np.zeros((len(latent), self.total_coefficients), dtype=np.complex64)
        flat[:, self.support_indices] = raw_latent.astype(np.complex64, copy=False)
        structured = flat.reshape((len(latent),) + self.layout.structured_tail)
        return inverse_beam_delay(structured, self.layout)

    def beam_delay_numpy(self, normalized_latent: np.ndarray) -> np.ndarray:
        """Expand normalized fixed-support latents directly in Beam-Delay space."""

        self._check_fitted()
        latent = np.asarray(normalized_latent)
        if latent.ndim != 2 or latent.shape[1] != self.coefficient_count:
            raise ValueError(
                f"latent must have shape (B,{self.coefficient_count}), got {latent.shape}"
            )
        if not np.iscomplexobj(latent):
            raise TypeError("latent must be complex-valued")
        raw_latent = latent * self.rms + self.mean
        flat = np.zeros((len(latent), self.total_coefficients), dtype=np.complex64)
        flat[:, self.support_indices] = raw_latent.astype(np.complex64, copy=False)
        return flat.reshape((len(latent),) + self.layout.structured_tail)

    def beam_delay_torch(self, normalized_latent: torch.Tensor) -> torch.Tensor:
        """Differentiably expand normalized latents without redundant FFTs."""

        self._check_fitted()
        if (
            normalized_latent.ndim != 2
            or normalized_latent.shape[1] != self.coefficient_count
        ):
            raise ValueError(
                f"latent must have shape (B,{self.coefficient_count}), "
                f"got {tuple(normalized_latent.shape)}"
            )
        if not torch.is_complex(normalized_latent):
            raise TypeError("latent must be complex-valued")
        device = normalized_latent.device
        mean = torch.as_tensor(
            self.mean, device=device, dtype=normalized_latent.dtype
        )
        rms = torch.as_tensor(
            self.rms, device=device, dtype=normalized_latent.real.dtype
        )
        indices = torch.as_tensor(
            self.support_indices, device=device, dtype=torch.long
        )
        raw_latent = normalized_latent * rms + mean
        flat = torch.zeros(
            (len(normalized_latent), self.total_coefficients),
            device=device,
            dtype=normalized_latent.dtype,
        )
        flat = flat.scatter(
            1,
            indices[None, :].expand(len(normalized_latent), -1),
            raw_latent,
        )
        return flat.reshape(
            (len(normalized_latent),) + self.layout.structured_tail
        )

    def channel_from_beam_delay_torch(
        self, beam_delay_values: torch.Tensor
    ) -> torch.Tensor:
        """Differentiably invert canonical ``B,H,V,P,N,D`` Beam-Delay."""

        expected = self.layout.structured_tail
        if (
            beam_delay_values.ndim != 6
            or tuple(beam_delay_values.shape[1:]) != expected
        ):
            raise ValueError(
                "beam-delay tensor must have shape "
                f"(B,{','.join(str(value) for value in expected)})"
            )
        if not torch.is_complex(beam_delay_values):
            raise TypeError("beam-delay tensor must be complex-valued")
        frequency = torch.fft.fft(
            beam_delay_values, dim=-1, norm="ortho"
        )
        spatial = torch.fft.ifft2(
            frequency, dim=(1, 2), norm="ortho"
        )
        standard_axes = {"H": 1, "V": 2, "P": 3}
        permutation = (0,) + tuple(
            standard_axes[label] for label in self.layout.order
        ) + (4, 5)
        ordered = spatial.permute(permutation)
        return ordered.reshape(
            len(beam_delay_values), *self.layout.config.channel_shape
        )

    def decode_torch(self, normalized_latent: torch.Tensor) -> torch.Tensor:
        """Differentiably decode on the input tensor's device."""

        self._check_fitted()
        if normalized_latent.ndim != 2 or normalized_latent.shape[1] != self.coefficient_count:
            raise ValueError(
                f"latent must have shape (B,{self.coefficient_count}), "
                f"got {tuple(normalized_latent.shape)}"
            )
        if not torch.is_complex(normalized_latent):
            raise TypeError("latent must be complex-valued")
        device = normalized_latent.device
        real_dtype = normalized_latent.real.dtype
        mean = torch.as_tensor(self.mean, device=device, dtype=normalized_latent.dtype)
        rms = torch.as_tensor(self.rms, device=device, dtype=real_dtype)
        indices = torch.as_tensor(self.support_indices, device=device, dtype=torch.long)
        raw_latent = normalized_latent * rms + mean
        flat = torch.zeros(
            (normalized_latent.shape[0], self.total_coefficients),
            device=device,
            dtype=normalized_latent.dtype,
        )
        scatter_indices = indices.unsqueeze(0).expand(normalized_latent.shape[0], -1)
        flat = flat.scatter(1, scatter_indices, raw_latent)
        structured = flat.reshape((normalized_latent.shape[0],) + self.layout.structured_tail)
        return self.channel_from_beam_delay_torch(structured)

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        destination = Path(path)
        metadata = {
            "format_version": 1,
            "support_fraction": self.support_fraction,
            "delay_block": self.delay_block,
            "layout_order": list(self.layout.order),
            "fit_indices_sha256": self.fit_indices_sha256,
            "source_shape": list(self.source_shape),
            "coefficient_count": self.coefficient_count,
        }
        np.savez_compressed(
            destination,
            support_indices=self.support_indices,
            mean=self.mean,
            rms=self.rms,
            group_ids=self.group_ids,
            latent_coordinates=self.latent_coordinates,
            fitted_indices=self.fitted_indices,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )

    @classmethod
    def load(
        cls, path: str | Path, layout: AntennaLayout
    ) -> "FixedSupportLatentAdapter":
        """Load a fully validated pickle-free adapter archive.

        Every persisted invariant is checked here so corrupt archives cannot
        surface later as an indexing or decode failure.
        """

        required_arrays = (
            "metadata_json",
            "support_indices",
            "mean",
            "rms",
            "group_ids",
            "latent_coordinates",
            "fitted_indices",
        )
        try:
            stored = np.load(Path(path), allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError("invalid adapter archive: cannot read NPZ") from error
        with stored:
            missing = set(required_arrays).difference(stored.files)
            if missing:
                raise ValueError(
                    "invalid adapter archive: missing " + ", ".join(sorted(missing))
                )
            try:
                metadata_value = stored["metadata_json"]
                if metadata_value.ndim != 0:
                    raise ValueError("metadata_json must be a scalar")
                metadata = json.loads(str(metadata_value.item()))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError("invalid adapter archive: invalid metadata") from error
            if not isinstance(metadata, dict):
                raise ValueError("invalid adapter archive: metadata must be an object")
            format_version = metadata.get("format_version")
            if type(format_version) is not int:
                raise ValueError(
                    "invalid adapter archive: format_version must be a non-bool integer"
                )
            if format_version != 1:
                raise ValueError("invalid adapter archive: unsupported format_version")
            layout_order = metadata.get("layout_order")
            if (
                not isinstance(layout_order, list)
                or len(layout_order) != 3
                or not all(isinstance(value, str) for value in layout_order)
            ):
                raise ValueError(
                    "invalid adapter archive: layout_order must be a list of three strings"
                )
            if tuple(layout_order) != layout.order:
                raise ValueError("invalid adapter archive: layout order mismatch")
            delay_block = metadata.get("delay_block")
            if type(delay_block) is not int:
                raise ValueError(
                    "invalid adapter archive: delay_block must be a non-bool integer"
                )
            support_fraction = metadata.get("support_fraction")
            if (
                type(support_fraction) not in (int, float)
                or not math.isfinite(support_fraction)
            ):
                raise ValueError(
                    "invalid adapter archive: support_fraction must be a finite number"
                )
            coefficient_count = metadata.get("coefficient_count")
            if type(coefficient_count) is not int:
                raise ValueError(
                    "invalid adapter archive: coefficient_count must be a non-bool integer"
                )
            source_shape_value = metadata.get("source_shape")
            if (
                not isinstance(source_shape_value, list)
                or len(source_shape_value) != 4
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in source_shape_value
                )
            ):
                raise ValueError("invalid adapter archive: source_shape must be four integers")
            source_shape = tuple(source_shape_value)
            if source_shape[0] <= 0 or tuple(source_shape[1:]) != layout.config.channel_shape:
                raise ValueError("invalid adapter archive: source_shape layout mismatch")
            try:
                adapter = cls(
                    layout,
                    support_fraction=support_fraction,
                    delay_block=delay_block,
                )
            except (TypeError, ValueError) as error:
                raise ValueError("invalid adapter archive: invalid configuration") from error
            if coefficient_count != adapter.coefficient_count:
                raise ValueError("invalid adapter archive: coefficient_count mismatch")

            support_indices = np.asarray(stored["support_indices"])
            if (
                support_indices.dtype.kind not in "iu"
                or support_indices.shape != (adapter.coefficient_count,)
            ):
                raise ValueError("invalid adapter archive: support_indices shape or dtype")
            support_indices = support_indices.astype(np.int64, copy=False)
            if (
                np.any(support_indices < 0)
                or np.any(support_indices >= adapter.total_coefficients)
                or len(np.unique(support_indices)) != len(support_indices)
            ):
                raise ValueError("invalid adapter archive: invalid support_indices")

            mean = np.asarray(stored["mean"])
            if mean.shape != (adapter.coefficient_count,) or not np.iscomplexobj(mean):
                raise ValueError("invalid adapter archive: mean shape or dtype")
            mean = mean.astype(np.complex64, copy=False)
            if not np.isfinite(mean).all():
                raise ValueError("invalid adapter archive: mean must be finite")

            rms = np.asarray(stored["rms"])
            if (
                rms.shape != (adapter.coefficient_count,)
                or not np.issubdtype(rms.dtype, np.floating)
            ):
                raise ValueError("invalid adapter archive: rms shape or dtype")
            rms = rms.astype(np.float32, copy=False)
            if not np.isfinite(rms).all() or np.any(rms <= 0):
                raise ValueError("invalid adapter archive: rms must be finite and positive")

            latent_coordinates = np.asarray(stored["latent_coordinates"])
            expected_coordinates = np.stack(
                np.unravel_index(support_indices, layout.structured_tail), axis=1
            ).astype(np.int64, copy=False)
            if (
                latent_coordinates.dtype.kind not in "iu"
                or latent_coordinates.shape != (adapter.coefficient_count, 5)
                or not np.array_equal(latent_coordinates, expected_coordinates)
            ):
                raise ValueError("invalid adapter archive: latent_coordinates mismatch")
            latent_coordinates = latent_coordinates.astype(np.int64, copy=False)

            group_ids = np.asarray(stored["group_ids"])
            block_count = int(np.ceil(layout.config.s / adapter.delay_block))
            expected_group_ids = np.asarray(
                (expected_coordinates[:, 2] * layout.config.n + expected_coordinates[:, 3])
                * block_count
                + expected_coordinates[:, 4] // adapter.delay_block,
                dtype=np.int64,
            )
            if (
                group_ids.dtype.kind not in "iu"
                or group_ids.shape != (adapter.coefficient_count,)
                or not np.array_equal(group_ids, expected_group_ids)
            ):
                raise ValueError("invalid adapter archive: group_ids mismatch")
            group_ids = group_ids.astype(np.int64, copy=False)

            fitted_indices = np.asarray(stored["fitted_indices"])
            if fitted_indices.dtype.kind not in "iu" or fitted_indices.ndim != 1:
                raise ValueError("invalid adapter archive: fitted_indices shape or dtype")
            fitted_indices = fitted_indices.astype(np.int64, copy=False)
            if (
                not len(fitted_indices)
                or np.any(fitted_indices < 0)
                or np.any(fitted_indices >= source_shape[0])
                or len(np.unique(fitted_indices)) != len(fitted_indices)
            ):
                raise ValueError("invalid adapter archive: invalid fitted_indices")
            fit_indices_sha256 = metadata.get("fit_indices_sha256")
            if (
                not isinstance(fit_indices_sha256, str)
                or len(fit_indices_sha256) != 64
                or any(character not in "0123456789abcdef" for character in fit_indices_sha256)
                or cls._indices_sha256(fitted_indices) != fit_indices_sha256
            ):
                raise ValueError("invalid adapter archive: fitted_indices SHA256 mismatch")

        adapter.support_indices = support_indices
        adapter.mean = mean
        adapter.rms = rms
        adapter.group_ids = group_ids
        adapter.latent_coordinates = latent_coordinates
        adapter.fitted_indices = fitted_indices
        adapter.fit_sample_count = int(len(fitted_indices))
        adapter.fit_indices_sha256 = fit_indices_sha256
        adapter.source_shape = source_shape
        return adapter
