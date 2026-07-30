from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

from .config import RoundConfig


AXES = ("H", "V", "P")


def candidate_orders() -> tuple[tuple[str, str, str], ...]:
    return tuple(permutations(AXES))


@dataclass(frozen=True)
class AntennaLayout:
    config: RoundConfig
    order: tuple[str, str, str] = ("H", "V", "P")

    def __post_init__(self) -> None:
        if len(self.order) != 3 or set(self.order) != set(AXES):
            raise ValueError(
                "antenna order must be a permutation of ('H','V','P'), "
                f"got {self.order}"
            )

    @property
    def structured_tail(self) -> tuple[int, int, int, int, int]:
        return (
            self.config.m_h,
            self.config.m_v,
            self.config.m_p,
            self.config.n,
            self.config.s,
        )

    def _axis_sizes(self) -> dict[str, int]:
        return {
            "H": self.config.m_h,
            "V": self.config.m_v,
            "P": self.config.m_p,
        }

    def to_structured(self, channel: np.ndarray) -> np.ndarray:
        array = np.asarray(channel)
        expected = self.config.channel_shape
        if array.ndim < 3 or tuple(array.shape[-3:]) != expected:
            raise ValueError(
                f"expected channel tail {expected}, got {array.shape[-3:]}"
            )

        leading = tuple(array.shape[:-3])
        leading_ndim = len(leading)
        sizes = self._axis_sizes()
        ordered_shape = tuple(sizes[label] for label in self.order)
        reshaped = array.reshape(
            leading + ordered_shape + (self.config.n, self.config.s)
        )
        ordered_axes = {label: leading_ndim + i for i, label in enumerate(self.order)}
        permutation = (
            tuple(range(leading_ndim))
            + tuple(ordered_axes[label] for label in AXES)
            + (leading_ndim + 3, leading_ndim + 4)
        )
        return np.transpose(reshaped, permutation)

    def from_structured(self, structured: np.ndarray) -> np.ndarray:
        array = np.asarray(structured)
        expected = self.structured_tail
        if array.ndim < 5 or tuple(array.shape[-5:]) != expected:
            raise ValueError(
                f"expected structured channel tail {expected}, got {array.shape[-5:]}"
            )

        leading = tuple(array.shape[:-5])
        leading_ndim = len(leading)
        standard_axes = {label: leading_ndim + i for i, label in enumerate(AXES)}
        permutation = (
            tuple(range(leading_ndim))
            + tuple(standard_axes[label] for label in self.order)
            + (leading_ndim + 3, leading_ndim + 4)
        )
        ordered = np.transpose(array, permutation)
        return ordered.reshape(
            leading + (self.config.m, self.config.n, self.config.s)
        )


def _preserve_complex_precision(result: np.ndarray, source: np.ndarray) -> np.ndarray:
    if np.asarray(source).dtype == np.dtype(np.complex64):
        return result.astype(np.complex64)
    return result


def beam_delay(channel: np.ndarray, layout: AntennaLayout) -> np.ndarray:
    structured = layout.to_structured(channel)
    angular = np.fft.fft2(structured, axes=(-5, -4), norm="ortho")
    transformed = np.fft.ifft(angular, axis=-1, norm="ortho")
    return _preserve_complex_precision(transformed, np.asarray(channel))


def inverse_beam_delay(
    beam_delay_channel: np.ndarray, layout: AntennaLayout
) -> np.ndarray:
    frequency = np.fft.fft(beam_delay_channel, axis=-1, norm="ortho")
    spatial = np.fft.ifft2(frequency, axes=(-5, -4), norm="ortho")
    flattened = layout.from_structured(spatial)
    return _preserve_complex_precision(flattened, np.asarray(beam_delay_channel))
