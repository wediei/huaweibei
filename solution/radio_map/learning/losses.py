"""Differentiable objective for support-aware anchor completion."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .anchor_mixer import MixerOutput
from .torch_metrics import torch_competition_metrics


@dataclass(frozen=True)
class AnchorLossConfig:
    """Task objective and an explicit trust region around the nearest anchor."""

    pas_weight: float = 0.4
    pdp_weight: float = 0.4
    nmse_weight: float = 0.2
    latent_weight: float = 0.02
    # Keep the learned channel a local correction to the strong nearest-anchor
    # prior.  A value of 1e-3 was too weak on the full fold: latent MSE kept
    # decreasing while the official PAS/PDP score collapsed.
    nearest_weight: float = 0.05
    nmse_objective: str = "official"

    def __post_init__(self) -> None:
        if (self.pas_weight, self.pdp_weight, self.nmse_weight) != (0.4, 0.4, 0.2):
            raise ValueError("competition objective weights are fixed by design")
        for name in ("latent_weight", "nearest_weight"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be a finite non-negative number")
            if not torch.isfinite(torch.tensor(float(value))) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.nmse_objective not in {"official", "log"}:
            raise ValueError("nmse_objective must be 'official' or 'log'")


@dataclass(frozen=True)
class AnchorLossValues:
    """Objective total, its terms, and decoded official metrics."""

    total: torch.Tensor
    pas_loss: torch.Tensor
    pdp_loss: torch.Tensor
    nmse_loss: torch.Tensor
    latent_mse: torch.Tensor
    residual_energy: torch.Tensor
    pas: torch.Tensor
    pdp: torch.Tensor
    nmse: torch.Tensor
    score: torch.Tensor

    def detached_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name).detach().cpu()) for name in self.__dataclass_fields__}


def complex_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared complex magnitude, retaining gradients through both parts."""

    if prediction.shape != target.shape:
        raise ValueError("complex MSE inputs must have identical shapes")
    if not torch.is_complex(prediction) or not torch.is_complex(target):
        raise TypeError("complex MSE inputs must be complex tensors")
    if prediction.device != target.device:
        raise ValueError("complex MSE inputs must be on the same device")
    return (prediction - target).abs().square().mean()


def anchor_completion_loss(
    output: MixerOutput,
    target_latent: torch.Tensor,
    target_channel: torch.Tensor,
    adapter: object,
    config: AnchorLossConfig | None = None,
) -> AnchorLossValues:
    """Calculate the exact task objective from a decoded latent prediction.

    The adapter owns both the fixed support and antenna-layout ordering, so it
    is deliberately the only source used to decode and evaluate channels.
    """

    if not isinstance(output, MixerOutput):
        raise TypeError("output must be a MixerOutput")
    if config is None:
        config = AnchorLossConfig()
    if not isinstance(config, AnchorLossConfig):
        raise TypeError("config must be an AnchorLossConfig")
    if output.latent.shape != target_latent.shape or output.nearest_latent.shape != target_latent.shape:
        raise ValueError("output and target_latent must have matching shapes")
    if not torch.is_complex(target_latent) or not torch.is_complex(target_channel):
        raise TypeError("targets must be complex tensors")
    if output.latent.device != target_latent.device or target_channel.device != target_latent.device:
        raise ValueError("output and targets must be on the same device")
    decode = getattr(adapter, "decode_torch", None)
    layout = getattr(adapter, "layout", None)
    if not callable(decode) or layout is None:
        raise TypeError("adapter must provide decode_torch and layout")

    decoded = decode(output.latent)
    metrics = torch_competition_metrics(decoded, target_channel, layout.config, layout.order)
    # Optimize the actual score components. row_power_cosine evaluates the
    # cosine denominator without multiplying squared norms, keeping the
    # CUDA/FP32 backward pass finite.
    pas_loss = config.pas_weight * (1.0 - metrics.pas)
    pdp_loss = config.pdp_weight * (1.0 - metrics.pdp)
    if config.nmse_objective == "official":
        # Exactly the complement of the official NMSE score term.
        nmse_loss = config.nmse_weight * metrics.nmse / (1.0 + metrics.nmse)
    else:
        # Retained as an explicit reproducibility mode: this stronger NMSE
        # pressure produced the current best online checkpoint.
        nmse_loss = config.nmse_weight * torch.log1p(metrics.nmse)
    latent_mse = complex_mse(output.latent, target_latent)
    residual_energy = complex_mse(output.latent, output.nearest_latent)
    total = pas_loss + pdp_loss + nmse_loss + config.latent_weight * latent_mse + config.nearest_weight * residual_energy
    return AnchorLossValues(total, pas_loss, pdp_loss, nmse_loss, latent_mse, residual_energy, metrics.pas, metrics.pdp, metrics.nmse, metrics.score)
