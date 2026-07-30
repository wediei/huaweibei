"""Competition metrics with streaming accumulation support."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .transforms import AntennaLayout


@dataclass(frozen=True)
class MetricComponents:
    """Additive sufficient statistics for the competition metrics."""

    pas_sum: float
    pas_count: int
    pdp_sum: float
    pdp_count: int
    error_sum: float
    power_sum: float


@dataclass(frozen=True)
class CompetitionMetrics:
    """Final PAS, PDP, NMSE, and weighted competition score."""

    pas: float
    pdp: float
    nmse: float
    score: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _cosine_sum_count(first: np.ndarray, second: np.ndarray) -> tuple[float, int]:
    """Return the sum and count of row-wise non-negative cosine similarities."""

    first_rows = np.asarray(first, dtype=np.float64).reshape(-1, first.shape[-1])
    second_rows = np.asarray(second, dtype=np.float64).reshape(-1, second.shape[-1])
    numerator = np.einsum("ij,ij->i", first_rows, second_rows, dtype=np.float64)
    first_norm_sq = np.einsum("ij,ij->i", first_rows, first_rows, dtype=np.float64)
    second_norm_sq = np.einsum("ij,ij->i", second_rows, second_rows, dtype=np.float64)
    denominator = np.sqrt(first_norm_sq * second_norm_sq)

    cosine = np.zeros_like(numerator, dtype=np.float64)
    nonzero = denominator > 0.0
    cosine[nonzero] = numerator[nonzero] / denominator[nonzero]
    both_zero = (first_norm_sq == 0.0) & (second_norm_sq == 0.0)
    cosine[both_zero] = 1.0
    cosine = np.clip(cosine, 0.0, 1.0)
    return float(cosine.sum(dtype=np.float64)), int(cosine.size)


def metric_components(
    prediction: np.ndarray,
    target: np.ndarray,
    layout: AntennaLayout,
) -> MetricComponents:
    """Compute additive PAS/PDP/NMSE statistics for one batch."""

    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have the same shape, got "
            f"{prediction.shape} and {target.shape}"
        )
    if prediction.ndim != 4:
        raise ValueError(
            "competition tensors must have shape (batch, M, N, S); "
            f"got {prediction.shape}"
        )
    expected = layout.config.channel_shape
    if prediction.shape[1:] != expected:
        raise ValueError(
            f"trailing tensor shape must be {expected} for this layout, "
            f"got {prediction.shape[1:]}"
        )
    if not np.iscomplexobj(prediction) or not np.iscomplexobj(target):
        raise TypeError("prediction and target must be complex-valued arrays")

    pred_structured = layout.to_structured(prediction)
    target_structured = layout.to_structured(target)
    pred_angle = np.fft.fft2(pred_structured, axes=(1, 2), norm="ortho")
    target_angle = np.fft.fft2(target_structured, axes=(1, 2), norm="ortho")
    # The task defines one BS-side PAS cosine for every
    # (position, subcarrier, UE antenna).  Polarization belongs to the BS
    # antenna vector; it is not another averaging axis.  Each PAS vector
    # therefore contains H*V*P == M entries.
    pred_pas = np.transpose(np.abs(pred_angle) ** 2, (0, 4, 5, 1, 2, 3)).reshape(
        -1, layout.config.m
    )
    target_pas = np.transpose(
        np.abs(target_angle) ** 2, (0, 4, 5, 1, 2, 3)
    ).reshape(-1, layout.config.m)
    pas_sum, pas_count = _cosine_sum_count(pred_pas, target_pas)

    pred_delay = np.fft.ifft(prediction, axis=-1, norm="ortho")
    target_delay = np.fft.ifft(target, axis=-1, norm="ortho")
    pred_pdp = (np.abs(pred_delay) ** 2).reshape(-1, layout.config.s)
    target_pdp = (np.abs(target_delay) ** 2).reshape(-1, layout.config.s)
    pdp_sum, pdp_count = _cosine_sum_count(pred_pdp, target_pdp)

    difference = prediction - target
    error_sum = float(np.sum(np.abs(difference) ** 2, dtype=np.float64))
    power_sum = float(np.sum(np.abs(target) ** 2, dtype=np.float64))
    return MetricComponents(
        pas_sum=pas_sum,
        pas_count=pas_count,
        pdp_sum=pdp_sum,
        pdp_count=pdp_count,
        error_sum=error_sum,
        power_sum=power_sum,
    )


def _finalize_metrics(
    components: MetricComponents,
    weights: Iterable[float],
) -> CompetitionMetrics:
    weights_tuple = tuple(float(weight) for weight in weights)
    if len(weights_tuple) != 3:
        raise ValueError("weights must contain exactly three values: PAS, PDP, NMSE")
    if components.pas_count == 0 or components.pdp_count == 0:
        raise ValueError("cannot finalize metrics without at least one sample")

    pas = components.pas_sum / components.pas_count
    pdp = components.pdp_sum / components.pdp_count
    if components.power_sum == 0.0:
        nmse = 0.0 if components.error_sum == 0.0 else components.error_sum / 1e-30
    else:
        nmse = components.error_sum / components.power_sum
    score = weights_tuple[0] * pas + weights_tuple[1] * pdp + weights_tuple[2] / (
        1.0 + nmse
    )
    return CompetitionMetrics(pas=pas, pdp=pdp, nmse=nmse, score=score)


def competition_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    layout: AntennaLayout,
    weights: Iterable[float] = (0.4, 0.4, 0.2),
) -> CompetitionMetrics:
    """Compute all competition metrics for an in-memory batch."""

    return _finalize_metrics(metric_components(prediction, target, layout), weights)


class MetricAccumulator:
    """Accumulate competition metrics without retaining full predictions."""

    def __init__(
        self,
        layout: AntennaLayout,
        weights: Iterable[float] = (0.4, 0.4, 0.2),
    ) -> None:
        self.layout = layout
        self.weights = tuple(float(weight) for weight in weights)
        if len(self.weights) != 3:
            raise ValueError("weights must contain exactly three values")
        self._components = MetricComponents(0.0, 0, 0.0, 0, 0.0, 0.0)

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        current = metric_components(prediction, target, self.layout)
        total = self._components
        self._components = MetricComponents(
            pas_sum=total.pas_sum + current.pas_sum,
            pas_count=total.pas_count + current.pas_count,
            pdp_sum=total.pdp_sum + current.pdp_sum,
            pdp_count=total.pdp_count + current.pdp_count,
            error_sum=total.error_sum + current.error_sum,
            power_sum=total.power_sum + current.power_sum,
        )

    def compute(self) -> CompetitionMetrics:
        return _finalize_metrics(self._components, self.weights)
