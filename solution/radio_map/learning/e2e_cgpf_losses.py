"""Joint objectives for E2E-CGPF."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch

from ..config import RoundConfig
from .e2e_cgpf import PathModes, TrainableGaussianField
from .torch_metrics import TorchMetricValues, torch_competition_metrics


@dataclass(frozen=True)
class E2ECGPFLossConfig:
    complex_weight: float = 1.0
    pas_weight: float = 0.25
    pdp_weight: float = 0.25
    nmse_weight: float = 0.25
    score_weight: float = 0.5
    phase_weight: float = 0.1
    path_sparsity_weight: float = 1e-3
    path_diversity_weight: float = 1e-2
    field_center_weight: float = 1e-3
    field_scale_weight: float = 1e-4
    field_material_weight: float = 1e-5
    field_smooth_weight: float = 1e-4
    field_opacity_weight: float = 1e-4
    field_orientation_weight: float = 1e-4
    polarization_weight: float = 1e-3
    causal_weight: float = 0.0
    causal_margin: float = 0.005
    minimum_effective_paths: float = 4.0

    def validate(self) -> None:
        for field in dataclasses.fields(self):
            value = float(getattr(self, field.name))
            if not torch.isfinite(torch.tensor(value)):
                raise ValueError(f"{field.name} must be finite")
            if field.name != "causal_margin" and value < 0.0:
                raise ValueError(f"{field.name} must be non-negative")


@dataclass
class E2ECGPFLossValues:
    total: torch.Tensor
    complex: torch.Tensor
    pas: torch.Tensor
    pdp: torch.Tensor
    nmse: torch.Tensor
    score: torch.Tensor
    phase: torch.Tensor
    path_sparsity: torch.Tensor
    path_diversity: torch.Tensor
    polarization: torch.Tensor
    field: torch.Tensor
    causal: torch.Tensor
    real_over_zero: torch.Tensor
    real_over_shuffle: torch.Tensor

    def scalars(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach().cpu())
            for name in self.__dataclass_fields__
        }


def normalized_complex_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    if not torch.is_complex(prediction) or not torch.is_complex(target):
        raise TypeError("complex loss requires complex tensors")
    difference = (prediction - target).abs().square().sum()
    target_energy = target.abs().square().sum().clamp_min(1e-12)
    return difference / target_energy


def phase_coherence_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    inner = (prediction * target.conj()).sum()
    denominator = (
        prediction.abs().square().sum().clamp_min(1e-12).sqrt()
        * target.abs().square().sum().clamp_min(1e-12).sqrt()
    )
    return 1.0 - (inner.abs() / denominator).clamp(0.0, 1.0)


def _effective_path_penalty(
    paths: PathModes, minimum_effective_paths: float
) -> torch.Tensor:
    weight = paths.gate.clamp_min(0.0)
    effective = weight.sum(dim=1).square() / weight.square().sum(dim=1).clamp_min(
        1e-12
    )
    count_penalty = torch.relu(
        torch.as_tensor(
            minimum_effective_paths, device=weight.device, dtype=weight.dtype
        )
        - effective
    ).mean()
    type_spread = paths.path_type.float().std(dim=1, unbiased=False).mean()
    type_penalty = torch.relu(
        torch.as_tensor(0.05, device=weight.device, dtype=weight.dtype)
        - type_spread
    )
    type_magnitude = 1e-3 * paths.path_type.square().mean()
    return count_penalty + type_penalty + type_magnitude


def _polarization_penalty(paths: PathModes) -> torch.Tensor:
    energy = paths.polarization.abs().square().sum(dim=(-2, -1))
    return (energy - 1.0).square().mean()


def _causal_term(
    prediction: torch.Tensor,
    target: torch.Tensor,
    config: RoundConfig,
    order: tuple[str, str, str],
    zero_prediction: torch.Tensor | None,
    shuffle_prediction: torch.Tensor | None,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zero = prediction.real.new_zeros(())
    if zero_prediction is None or shuffle_prediction is None:
        return zero, zero, zero
    real_score = torch_competition_metrics(prediction, target, config, order).score
    zero_score = torch_competition_metrics(
        zero_prediction, target, config, order
    ).score
    shuffle_score = torch_competition_metrics(
        shuffle_prediction, target, config, order
    ).score
    real_over_zero = real_score - zero_score
    real_over_shuffle = real_score - shuffle_score
    causal = torch.relu(margin - real_over_zero) + torch.relu(
        margin - real_over_shuffle
    )
    return causal, real_over_zero, real_over_shuffle


def e2e_cgpf_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    paths: PathModes,
    field: TrainableGaussianField,
    round_config: RoundConfig,
    order: tuple[str, str, str],
    config: E2ECGPFLossConfig | None = None,
    *,
    zero_prediction: torch.Tensor | None = None,
    shuffle_prediction: torch.Tensor | None = None,
) -> tuple[E2ECGPFLossValues, TorchMetricValues]:
    loss_config = config or E2ECGPFLossConfig()
    loss_config.validate()
    metrics = torch_competition_metrics(prediction, target, round_config, order)
    complex_value = normalized_complex_loss(prediction, target)
    phase_value = phase_coherence_loss(prediction, target)
    path_sparsity = paths.gate.mean()
    path_diversity = _effective_path_penalty(
        paths, loss_config.minimum_effective_paths
    )
    polarization = _polarization_penalty(paths)
    field_values = field.regularization()
    field_total = (
        loss_config.field_center_weight * field_values["field_center"]
        + loss_config.field_scale_weight * field_values["field_scale"]
        + loss_config.field_material_weight * field_values["field_material"]
        + loss_config.field_smooth_weight * field_values["field_smooth"]
        + loss_config.field_opacity_weight * field_values["field_opacity"]
        + loss_config.field_orientation_weight
        * field_values["field_orientation"]
    )
    causal, real_over_zero, real_over_shuffle = _causal_term(
        prediction,
        target,
        round_config,
        order,
        zero_prediction,
        shuffle_prediction,
        loss_config.causal_margin,
    )
    total = (
        loss_config.complex_weight * complex_value
        + loss_config.pas_weight * (1.0 - metrics.pas)
        + loss_config.pdp_weight * (1.0 - metrics.pdp)
        + loss_config.nmse_weight * metrics.nmse
        + loss_config.score_weight * (1.0 - metrics.score)
        + loss_config.phase_weight * phase_value
        + loss_config.path_sparsity_weight * path_sparsity
        + loss_config.path_diversity_weight * path_diversity
        + loss_config.polarization_weight * polarization
        + field_total
        + loss_config.causal_weight * causal
    )
    values = E2ECGPFLossValues(
        total=total,
        complex=complex_value,
        pas=metrics.pas,
        pdp=metrics.pdp,
        nmse=metrics.nmse,
        score=metrics.score,
        phase=phase_value,
        path_sparsity=path_sparsity,
        path_diversity=path_diversity,
        polarization=polarization,
        field=field_total,
        causal=causal,
        real_over_zero=real_over_zero,
        real_over_shuffle=real_over_shuffle,
    )
    return values, metrics
