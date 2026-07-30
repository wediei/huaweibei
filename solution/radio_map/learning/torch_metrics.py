"""Differentiable PyTorch implementations of the competition metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import torch

if TYPE_CHECKING:
    from ..config import RoundConfig


@dataclass(frozen=True)
class TorchMetricValues:
    """Differentiable PAS, PDP, NMSE, and weighted competition score."""

    pas: torch.Tensor
    pdp: torch.Tensor
    nmse: torch.Tensor
    score: torch.Tensor


def _accumulation_dtype(tensor: torch.Tensor) -> torch.dtype:
    """Use CPU float64 parity while retaining float32 GPU training reductions."""

    return torch.float64 if tensor.device.type == "cpu" else torch.float32


def _normalise_vector_axes(ndim: int, vector_axes: Iterable[int]) -> tuple[int, ...]:
    axes = tuple(axis % ndim for axis in vector_axes)
    if not axes or len(set(axes)) != len(axes):
        raise ValueError("vector_axes must contain distinct tensor axes")
    return axes


def torch_to_structured(
    channel: torch.Tensor, config: RoundConfig, order: tuple[str, str, str]
) -> torch.Tensor:
    """Reshape flattened antenna axes to canonical ``(B, H, V, P, N, S)``."""

    if channel.ndim != 4:
        raise ValueError(
            "competition tensors must have shape (batch, M, N, S); "
            f"got {tuple(channel.shape)}"
        )
    if tuple(channel.shape[1:]) != config.channel_shape:
        raise ValueError(
            f"trailing tensor shape must be {config.channel_shape}, "
            f"got {tuple(channel.shape[1:])}"
        )
    if not torch.is_complex(channel):
        raise TypeError("competition tensors must be complex-valued")
    if len(order) != 3 or set(order) != {"H", "V", "P"}:
        raise ValueError(
            "antenna order must be a permutation of ('H', 'V', 'P'), "
            f"got {order}"
        )

    antenna_sizes = {"H": config.m_h, "V": config.m_v, "P": config.m_p}
    ordered_shape = tuple(antenna_sizes[label] for label in order)
    reshaped = channel.reshape(channel.shape[0], *ordered_shape, config.n, config.s)
    ordered_axes = {label: 1 + index for index, label in enumerate(order)}
    return reshaped.permute(
        0,
        ordered_axes["H"],
        ordered_axes["V"],
        ordered_axes["P"],
        4,
        5,
    )


def row_power_cosine(
    first: torch.Tensor, second: torch.Tensor, vector_axes: Iterable[int]
) -> torch.Tensor:
    """Return mean cosine of non-negative power vectors over ``vector_axes``.

    Matching zero vectors receive a score of one; a single zero vector receives
    zero.  The CPU accumulation dtype matches the NumPy evaluator's float64
    reductions, while GPU reductions stay float32 for practical training cost.
    """

    if first.shape != second.shape:
        raise ValueError(
            f"power tensors must have the same shape, got {tuple(first.shape)} "
            f"and {tuple(second.shape)}"
        )
    axes = _normalise_vector_axes(first.ndim, vector_axes)
    row_axes = tuple(axis for axis in range(first.ndim) if axis not in axes)
    permutation = row_axes + axes
    vector_size = 1
    for axis in axes:
        vector_size *= first.shape[axis]

    accumulation_dtype = _accumulation_dtype(first)
    first_rows = first.permute(permutation).reshape(-1, vector_size).to(accumulation_dtype)
    second_rows = (
        second.permute(permutation).reshape(-1, vector_size).to(accumulation_dtype)
    )
    numerator = (first_rows * second_rows).sum(dim=-1)
    first_norm_sq = first_rows.square().sum(dim=-1)
    second_norm_sq = second_rows.square().sum(dim=-1)
    # Do not multiply the squared norms: on CUDA/FP32 that product can
    # overflow even when each norm and the final cosine are finite, producing
    # non-finite backward gradients. Divide by the two norms sequentially.
    first_nonzero = first_norm_sq > 0
    second_nonzero = second_norm_sq > 0
    first_norm = torch.sqrt(torch.where(first_nonzero, first_norm_sq, torch.ones_like(first_norm_sq)))
    second_norm = torch.sqrt(torch.where(second_nonzero, second_norm_sq, torch.ones_like(second_norm_sq)))
    nonzero = first_nonzero & second_nonzero
    safe_first = torch.where(nonzero, first_norm, torch.ones_like(first_norm))
    safe_second = torch.where(nonzero, second_norm, torch.ones_like(second_norm))
    cosine = numerator / safe_first / safe_second
    cosine = torch.where(nonzero, cosine, torch.zeros_like(cosine))
    both_zero = (first_norm_sq == 0) & (second_norm_sq == 0)
    cosine = torch.where(both_zero, torch.ones_like(cosine), cosine)
    return cosine.clamp(0, 1).mean()


def torch_competition_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    config: RoundConfig,
    order: tuple[str, str, str],
) -> TorchMetricValues:
    """Compute differentiable competition metrics for a channel tensor batch."""

    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have the same shape, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.device != target.device:
        raise ValueError("prediction and target must be on the same device")

    pred_structured = torch_to_structured(prediction, config, order)
    target_structured = torch_to_structured(target, config, order)
    pred_pas = torch.fft.fft2(pred_structured, dim=(1, 2), norm="ortho").abs().square()
    target_pas = torch.fft.fft2(target_structured, dim=(1, 2), norm="ortho").abs().square()
    # One cosine row is defined per position/subcarrier/UE antenna.  The
    # complete BS-side PAS vector includes both spatial axes and polarization.
    pas = row_power_cosine(pred_pas, target_pas, vector_axes=(1, 2, 3))

    pred_pdp = torch.fft.ifft(prediction, dim=-1, norm="ortho").abs().square()
    target_pdp = torch.fft.ifft(target, dim=-1, norm="ortho").abs().square()
    pdp = row_power_cosine(pred_pdp, target_pdp, vector_axes=(-1,))

    accumulation_dtype = _accumulation_dtype(prediction)
    difference = prediction - target
    error_sum = difference.abs().square().to(accumulation_dtype).sum()
    power_sum = target.abs().square().to(accumulation_dtype).sum()
    nmse = error_sum / power_sum.clamp_min(1e-30)
    score = (
        config.weights[0] * pas
        + config.weights[1] * pdp
        + config.weights[2] / (1 + nmse)
    )
    return TorchMetricValues(pas=pas, pdp=pdp, nmse=nmse, score=score)
