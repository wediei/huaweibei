"""Joint complex-channel, structural, causal, and regularization losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .ac_cgmrf import ACCGMRFOutput
from .torch_metrics import TorchMetricValues, torch_competition_metrics


@dataclass(frozen=True)
class ACCGMRFLossConfig:
    complex_weight: float = 0.20
    pas_weight: float = 0.20
    pdp_weight: float = 0.20
    score_weight: float = 0.40
    energy_weight: float = 0.002
    trust_weight: float = 0.001
    sparsity_weight: float = 0.001
    coupling_weight: float = 0.0005
    phase_continuity_weight: float = 0.0005
    causal_margin_weight: float = 1.0
    zero_margin: float = 0.01
    shuffle_margin: float = 0.01


@dataclass(frozen=True)
class ACCGMRFLossValues:
    total: torch.Tensor
    metrics: TorchMetricValues
    complex_nmse: torch.Tensor
    pas_loss: torch.Tensor
    pdp_loss: torch.Tensor
    score_loss: torch.Tensor
    residual_energy: torch.Tensor
    trust: torch.Tensor
    sparsity: torch.Tensor
    coupling: torch.Tensor
    phase_continuity: torch.Tensor
    real_over_zero: torch.Tensor
    real_over_shuffle: torch.Tensor
    zero_margin_loss: torch.Tensor
    shuffle_margin_loss: torch.Tensor


def _phase_continuity(output: ACCGMRFOutput) -> torch.Tensor:
    if output.atoms is None or output.atoms.delay_phase_ramp.shape[-1] < 2:
        return output.prediction.real.new_zeros(())
    ramp = output.atoms.delay_phase_ramp
    adjacent = ramp[..., 1:] * ramp[..., :-1].conj()
    expected = adjacent[..., :1].expand_as(adjacent)
    return (adjacent - expected).abs().square().mean()


def _coupling_regularizer(output: ACCGMRFOutput) -> torch.Tensor:
    if output.atoms is None:
        return output.prediction.real.new_zeros(())
    coupling = output.atoms.coupling
    magnitude = (coupling.abs() - 1.0).square().mean()
    phase = torch.angle(coupling)
    return magnitude + 0.1 * phase.square().mean()


def ac_cgmrf_loss(
    real_output: ACCGMRFOutput,
    zero_output: ACCGMRFOutput,
    shuffle_output: ACCGMRFOutput,
    target_channel: torch.Tensor,
    adapter: object,
    config: ACCGMRFLossConfig | None = None,
) -> ACCGMRFLossValues:
    """Compute all Spec-required terms from deployment-available conditions."""

    cfg = config or ACCGMRFLossConfig()
    real_channel = adapter.channel_from_beam_delay_torch(real_output.prediction)
    zero_channel = adapter.channel_from_beam_delay_torch(zero_output.prediction)
    shuffle_channel = adapter.channel_from_beam_delay_torch(
        shuffle_output.prediction
    )
    real_metrics = torch_competition_metrics(
        real_channel,
        target_channel,
        adapter.layout.config,
        adapter.layout.order,
    )
    zero_metrics = torch_competition_metrics(
        zero_channel,
        target_channel,
        adapter.layout.config,
        adapter.layout.order,
    )
    shuffle_metrics = torch_competition_metrics(
        shuffle_channel,
        target_channel,
        adapter.layout.config,
        adapter.layout.order,
    )
    complex_nmse = (
        (real_channel - target_channel).abs().square().sum()
        / target_channel.abs().square().sum().clamp_min(1e-12)
    )
    pas_loss = 1.0 - real_metrics.pas
    pdp_loss = 1.0 - real_metrics.pdp
    score_loss = 1.0 - real_metrics.score
    residual_energy = real_output.diagnostics[
        "residual_energy_ratio"
    ].mean()
    trust = real_output.trust.abs().mean()
    sparsity = (
        real_output.prediction.real.new_zeros(())
        if real_output.atoms is None
        else real_output.atoms.existence.mean()
    )
    coupling = _coupling_regularizer(real_output)
    phase_continuity = _phase_continuity(real_output)
    real_over_zero = real_metrics.score - zero_metrics.score
    real_over_shuffle = real_metrics.score - shuffle_metrics.score
    zero_margin_loss = torch.relu(
        real_metrics.score.new_tensor(cfg.zero_margin) - real_over_zero
    )
    shuffle_margin_loss = torch.relu(
        real_metrics.score.new_tensor(cfg.shuffle_margin)
        - real_over_shuffle
    )
    total = (
        cfg.complex_weight * complex_nmse
        + cfg.pas_weight * pas_loss
        + cfg.pdp_weight * pdp_loss
        + cfg.score_weight * score_loss
        + cfg.energy_weight * residual_energy
        + cfg.trust_weight * trust
        + cfg.sparsity_weight * sparsity
        + cfg.coupling_weight * coupling
        + cfg.phase_continuity_weight * phase_continuity
        + cfg.causal_margin_weight * (zero_margin_loss + shuffle_margin_loss)
    )
    return ACCGMRFLossValues(
        total=total,
        metrics=real_metrics,
        complex_nmse=complex_nmse,
        pas_loss=pas_loss,
        pdp_loss=pdp_loss,
        score_loss=score_loss,
        residual_energy=residual_energy,
        trust=trust,
        sparsity=sparsity,
        coupling=coupling,
        phase_continuity=phase_continuity,
        real_over_zero=real_over_zero,
        real_over_shuffle=real_over_shuffle,
        zero_margin_loss=zero_margin_loss,
        shuffle_margin_loss=shuffle_margin_loss,
    )
