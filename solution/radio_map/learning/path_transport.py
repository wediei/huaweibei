"""Differentiable equivalent beam-index and delay transport.

No carrier frequency, wavelength, subcarrier spacing, or antenna spacing is
assumed here.  H/V values are index-domain shifts implemented exactly by the
Fourier shift theorem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import numpy as np
import torch


ArrayT = TypeVar("ArrayT")


@dataclass(frozen=True)
class TransportParameters:
    delta_h: Any = 0.0
    delta_v: Any = 0.0
    delta_delay: Any = 0.0
    log_amplitude: Any = 0.0
    phase_real: Any = 1.0
    phase_imag: Any = 0.0
    existence: Any = 1.0
    reliability: Any = 1.0


def _numpy_parameter(
    value: Any, source: np.ndarray, name: str
) -> np.ndarray:
    result = np.asarray(value, dtype=source.real.dtype)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    if source.ndim == 6 and result.shape == (
        source.shape[0],
        source.shape[3],
        source.shape[4],
    ):
        result = result.reshape(source.shape[0], 1, 1, source.shape[3], source.shape[4], 1)
    return result


def _torch_parameter(
    value: Any, source: torch.Tensor, name: str
) -> torch.Tensor:
    result = torch.as_tensor(
        value, device=source.device, dtype=source.real.dtype
    )
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    if source.ndim == 6 and tuple(result.shape) == (
        source.shape[0],
        source.shape[3],
        source.shape[4],
    ):
        result = result.reshape(source.shape[0], 1, 1, source.shape[3], source.shape[4], 1)
    return result


def fractional_circular_shift_numpy(
    values: np.ndarray,
    shift: Any,
    axis: int,
) -> np.ndarray:
    source = np.asarray(values)
    if not np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError("fractional shift requires a complex array")
    normalized_axis = int(axis) % source.ndim
    if np.asarray(shift).ndim == 0 and float(np.asarray(shift)) == 0.0:
        return source
    amount = _numpy_parameter(shift, source, "shift")
    frequency = np.fft.fftfreq(source.shape[normalized_axis]).astype(
        source.real.dtype, copy=False
    )
    frequency_shape = [1] * source.ndim
    frequency_shape[normalized_axis] = len(frequency)
    phase = np.exp(
        -2j * np.pi * amount * frequency.reshape(frequency_shape)
    )
    transformed = np.fft.fft(source, axis=normalized_axis)
    result = np.fft.ifft(
        transformed * phase, axis=normalized_axis
    )
    return result.astype(source.dtype, copy=False)


def fractional_circular_shift_torch(
    values: torch.Tensor,
    shift: Any,
    axis: int,
) -> torch.Tensor:
    if not torch.is_complex(values):
        raise TypeError("fractional shift requires a complex tensor")
    normalized_axis = int(axis) % values.ndim
    if not isinstance(shift, torch.Tensor) and float(shift) == 0.0:
        return values
    amount = _torch_parameter(shift, values, "shift")
    frequency = torch.fft.fftfreq(
        values.shape[normalized_axis],
        device=values.device,
        dtype=values.real.dtype,
    )
    frequency_shape = [1] * values.ndim
    frequency_shape[normalized_axis] = len(frequency)
    angle = -2.0 * math.pi * amount * frequency.reshape(frequency_shape)
    phase = torch.complex(torch.cos(angle), torch.sin(angle))
    return torch.fft.ifft(
        torch.fft.fft(values, dim=normalized_axis) * phase,
        dim=normalized_axis,
    )


def apply_transport_numpy(
    values: np.ndarray,
    parameters: TransportParameters,
) -> np.ndarray:
    source = np.asarray(values)
    if source.ndim != 6:
        raise ValueError("transport expects shape (B,H,V,P,N,D)")
    shifted = fractional_circular_shift_numpy(
        source, parameters.delta_h, axis=1
    )
    shifted = fractional_circular_shift_numpy(
        shifted, parameters.delta_v, axis=2
    )
    shifted = fractional_circular_shift_numpy(
        shifted, parameters.delta_delay, axis=5
    )
    log_amplitude = np.clip(
        _numpy_parameter(
            parameters.log_amplitude, source, "log_amplitude"
        ),
        -8.0,
        8.0,
    )
    phase_real = _numpy_parameter(
        parameters.phase_real, source, "phase_real"
    )
    phase_imag = _numpy_parameter(
        parameters.phase_imag, source, "phase_imag"
    )
    phase_norm = np.sqrt(phase_real**2 + phase_imag**2)
    phase = (phase_real + 1j * phase_imag) / np.maximum(
        phase_norm, np.finfo(source.real.dtype).eps
    )
    existence = np.clip(
        _numpy_parameter(parameters.existence, source, "existence"),
        0.0,
        1.0,
    )
    reliability = np.clip(
        _numpy_parameter(parameters.reliability, source, "reliability"),
        0.0,
        1.0,
    )
    candidate = shifted * np.exp(log_amplitude) * phase * existence
    return np.asarray(
        source + reliability * (candidate - source), dtype=source.dtype
    )


def apply_transport_torch(
    values: torch.Tensor,
    parameters: TransportParameters,
) -> torch.Tensor:
    if values.ndim != 6:
        raise ValueError("transport expects shape (B,H,V,P,N,D)")
    shifted = fractional_circular_shift_torch(
        values, parameters.delta_h, axis=1
    )
    shifted = fractional_circular_shift_torch(
        shifted, parameters.delta_v, axis=2
    )
    shifted = fractional_circular_shift_torch(
        shifted, parameters.delta_delay, axis=5
    )
    log_amplitude = _torch_parameter(
        parameters.log_amplitude, values, "log_amplitude"
    ).clamp(-8.0, 8.0)
    phase_real = _torch_parameter(
        parameters.phase_real, values, "phase_real"
    )
    phase_imag = _torch_parameter(
        parameters.phase_imag, values, "phase_imag"
    )
    phase_norm = torch.sqrt(phase_real.square() + phase_imag.square()).clamp_min(
        torch.finfo(values.real.dtype).eps
    )
    phase = torch.complex(phase_real / phase_norm, phase_imag / phase_norm)
    existence = _torch_parameter(
        parameters.existence, values, "existence"
    ).clamp(0.0, 1.0)
    reliability = _torch_parameter(
        parameters.reliability, values, "reliability"
    ).clamp(0.0, 1.0)
    candidate = shifted * torch.exp(log_amplitude) * phase * existence
    return values + reliability * (candidate - values)


def blend_transport(
    coarse: ArrayT,
    scale: float,
    operation: Callable[[], ArrayT],
) -> ArrayT:
    """Blend a transport candidate while hard-bypassing a disabled stage."""

    value = float(scale)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("transport scale must be finite and non-negative")
    if value == 0.0:
        return coarse
    transported = operation()
    return coarse + value * (transported - coarse)

