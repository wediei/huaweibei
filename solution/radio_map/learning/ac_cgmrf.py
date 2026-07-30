"""Anchor-conditioned complex Gaussian multipath residual field.

The module is deliberately independent of the frozen O4.1 implementation.  It
consumes an already computed O4.1 Beam-Delay tensor and can only add a bounded
residual.  Disabling the branch, selecting ``zero`` mode, or using the
zero-initialized trust head returns the input bit-for-bit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


MapMode = Literal["real", "zero", "shuffle"]


@dataclass(frozen=True)
class ACCGMRFConfig:
    """Small shared-network configuration; no per-map-Gaussian parameters."""

    p_count: int
    n_count: int
    path_feature_dim: int
    atom_count: int = 24
    d_model: int = 96
    map_hidden_dim: int = 96
    support_h: int = 3
    support_v: int = 2
    support_delay: int = 4
    sigma_h_min: float = 0.35
    sigma_h_max: float = 2.5
    sigma_v_min: float = 0.35
    sigma_v_max: float = 2.0
    sigma_delay_min: float = 0.5
    sigma_delay_max: float = 4.0
    phase_residual_limit: float = math.pi / 2.0
    coupling_phase_limit: float = math.pi / 4.0
    max_residual_ratio: float = 0.35
    support_threshold: float = 1e-3

    def __post_init__(self) -> None:
        for name in (
            "p_count",
            "n_count",
            "path_feature_dim",
            "atom_count",
            "d_model",
            "map_hidden_dim",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("support_h", "support_v", "support_delay"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for lower_name, upper_name in (
            ("sigma_h_min", "sigma_h_max"),
            ("sigma_v_min", "sigma_v_max"),
            ("sigma_delay_min", "sigma_delay_max"),
        ):
            lower, upper = float(getattr(self, lower_name)), float(
                getattr(self, upper_name)
            )
            if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower <= upper):
                raise ValueError(f"invalid bounds {lower_name}/{upper_name}")
        for name in (
            "phase_residual_limit",
            "coupling_phase_limit",
            "max_residual_ratio",
            "support_threshold",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class MultipathAtoms:
    """Continuous atom parameters and analytic phase/coupling diagnostics."""

    centers: torch.Tensor
    sigmas: torch.Tensor
    complex_gain: torch.Tensor
    coupling: torch.Tensor
    phase_reference: torch.Tensor
    existence: torch.Tensor
    reliability: torch.Tensor
    delay_phase_ramp: torch.Tensor


@dataclass(frozen=True)
class ACCGMRFOutput:
    prediction: torch.Tensor
    residual: torch.Tensor
    raw_residual: torch.Tensor
    atoms: MultipathAtoms | None
    trust: torch.Tensor
    support_novelty: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


def continuous_delay_phase_ramp(
    delay_centers: torch.Tensor, subcarrier_count: int
) -> torch.Tensor:
    """Return the exact DFT phase ramp induced by continuous delay locations."""

    if subcarrier_count < 1:
        raise ValueError("subcarrier_count must be positive")
    frequency = torch.arange(
        subcarrier_count,
        device=delay_centers.device,
        dtype=delay_centers.dtype,
    )
    phase = (
        -2.0
        * math.pi
        * delay_centers[..., None]
        * frequency
        / float(subcarrier_count)
    )
    return torch.polar(torch.ones_like(phase), phase)


def _linear_parameter(
    raw: torch.Tensor, lower: float, upper: float
) -> torch.Tensor:
    return float(lower) + (float(upper) - float(lower)) * torch.sigmoid(raw)


def _complex_unit(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    magnitude = values.abs()
    return torch.where(
        magnitude > eps,
        values / magnitude.clamp_min(eps),
        torch.ones_like(values),
    )


def support_novelty_ratio(
    residual: torch.Tensor,
    anchor_beam_delay: torch.Tensor,
    anchor_mask: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Fraction of residual energy outside the measured Anchor support."""

    if residual.ndim != 6 or anchor_beam_delay.ndim != 7:
        raise ValueError("expected residual B,H,V,P,N,D and anchors B,K,H,V,P,N,D")
    valid = anchor_mask[:, :, None, None, None, None, None]
    anchor_power = torch.where(
        valid, anchor_beam_delay.abs().square(), torch.zeros_like(anchor_beam_delay.real)
    ).sum(dim=1)
    maximum = anchor_power.flatten(1).amax(dim=1).clamp_min(1e-12)
    occupied = anchor_power > float(threshold) * maximum[:, None, None, None, None, None]
    residual_power = residual.abs().square()
    total = residual_power.flatten(1).sum(dim=1)
    novel = torch.where(occupied, torch.zeros_like(residual_power), residual_power)
    return torch.where(
        total > 1e-12,
        novel.flatten(1).sum(dim=1) / total.clamp_min(1e-12),
        torch.zeros_like(total),
    )


def continuous_complex_splat(
    atoms: MultipathAtoms,
    grid_shape: tuple[int, int, int, int, int],
    support_radii: tuple[int, int, int],
) -> torch.Tensor:
    """Locally splat continuous atoms and coherently sum in the complex domain.

    Beam axes wrap because they are FFT bins.  Delay support is clipped rather
    than wrapped.  Only ``(2rh+1)*(2rv+1)*(2rd+1)`` cells per atom are
    materialized; there is no Atom-by-full-grid intermediate.
    """

    h_count, v_count, p_count, n_count, delay_count = grid_shape
    radius_h, radius_v, radius_d = support_radii
    if min(grid_shape) < 1 or min(support_radii) < 0:
        raise ValueError("invalid grid or support radii")
    batch, atom_count, coordinate_count = atoms.centers.shape
    if coordinate_count != 3 or atoms.sigmas.shape != atoms.centers.shape:
        raise ValueError("centers/sigmas must have shape B,A,3")
    if atoms.complex_gain.shape != (batch, atom_count):
        raise ValueError("complex_gain must have shape B,A")
    expected_coupling = (batch, atom_count, p_count, n_count)
    if (
        atoms.coupling.shape != expected_coupling
        or atoms.phase_reference.shape != expected_coupling
    ):
        raise ValueError("coupling/phase_reference shape differs from grid")
    if not (
        torch.is_complex(atoms.complex_gain)
        and torch.is_complex(atoms.coupling)
        and torch.is_complex(atoms.phase_reference)
    ):
        raise TypeError("complex atom coefficients are required")

    device, real_dtype = atoms.centers.device, atoms.centers.dtype
    h_offsets = torch.arange(-radius_h, radius_h + 1, device=device)
    v_offsets = torch.arange(-radius_v, radius_v + 1, device=device)
    d_offsets = torch.arange(-radius_d, radius_d + 1, device=device)
    offset_grid = torch.cartesian_prod(h_offsets, v_offsets, d_offsets)
    if offset_grid.ndim == 1:
        offset_grid = offset_grid[None, :]
    offset_grid = offset_grid.to(real_dtype)
    p_grid = torch.arange(p_count, device=device)
    n_grid = torch.arange(n_count, device=device)
    all_indices: list[torch.Tensor] = []
    all_values: list[torch.Tensor] = []

    for atom_index in range(atom_count):
        center = atoms.centers[:, atom_index]
        sigma = atoms.sigmas[:, atom_index].clamp_min(1e-4)
        base = torch.floor(center)
        unwrapped = base[:, None, :] + offset_grid[None, :, :]
        valid_delay = (unwrapped[..., 2] >= 0) & (
            unwrapped[..., 2] < delay_count
        )
        distance = (unwrapped - center[:, None, :]) / sigma[:, None, :]
        envelope = torch.exp(-0.5 * distance.square().sum(dim=-1))
        envelope = torch.where(valid_delay, envelope, torch.zeros_like(envelope))
        envelope = envelope / envelope.sum(dim=1, keepdim=True).clamp_min(1e-12)

        h_index = torch.remainder(unwrapped[..., 0].long(), h_count)
        v_index = torch.remainder(unwrapped[..., 1].long(), v_count)
        d_index = unwrapped[..., 2].long().clamp(0, delay_count - 1)
        spatial_linear = (
            ((h_index * v_count + v_index)[:, :, None, None] * p_count + p_grid[None, None, :, None])
            * n_count
            + n_grid[None, None, None, :]
        ) * delay_count + d_index[:, :, None, None]
        coefficient = (
            atoms.complex_gain[:, atom_index, None, None]
            * atoms.coupling[:, atom_index]
            * atoms.phase_reference[:, atom_index]
        )
        values = (
            envelope[:, :, None, None].to(coefficient.dtype)
            * coefficient[:, None, :, :]
        )
        all_indices.append(spatial_linear.reshape(batch, -1))
        all_values.append(values.reshape(batch, -1))
    output = torch.zeros(
        batch,
        h_count * v_count * p_count * n_count * delay_count,
        device=device,
        dtype=atoms.complex_gain.dtype,
    )
    output = output.scatter_add(
        1,
        torch.cat(all_indices, dim=1),
        torch.cat(all_values, dim=1),
    )
    return output.reshape(batch, h_count, v_count, p_count, n_count, delay_count)


class ACCGMRF(nn.Module):
    """Generate a bounded new-support residual beside a frozen O4.1 tensor."""

    def __init__(self, config: ACCGMRFConfig) -> None:
        super().__init__()
        self.config = config
        combined_path_dim = 4 * config.path_feature_dim
        self.map_encoder = nn.Sequential(
            nn.Linear(combined_path_dim, config.map_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.map_hidden_dim, config.d_model),
        )
        group_summary_dim = 3 * config.p_count * config.n_count
        geometry_dim = 11
        self.condition_encoder = nn.Sequential(
            nn.Linear(config.d_model + 2 * group_summary_dim + geometry_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
        )
        self.atom_embedding = nn.Parameter(
            torch.randn(config.atom_count, config.d_model) * 0.02
        )
        atom_width = 10 + 2 * config.p_count + 2 * config.n_count
        self.atom_head = nn.Sequential(
            nn.Linear(2 * config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, atom_width),
        )
        self.trust_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.SiLU(),
            nn.Linear(config.d_model // 2, 1),
        )
        nn.init.zeros_(self.trust_head[-1].weight)
        nn.init.zeros_(self.trust_head[-1].bias)
        self.phase_unlocked = True

    def set_phase_unlocked(self, enabled: bool) -> None:
        """Curriculum switch; it changes no checkpoint shapes."""

        self.phase_unlocked = bool(enabled)

    def _validate(
        self,
        base: torch.Tensor,
        anchors: torch.Tensor,
        anchor_positions: torch.Tensor,
        target_positions: torch.Tensor,
        target_tokens: torch.Tensor,
        anchor_tokens: torch.Tensor,
        anchor_target_tokens: torch.Tensor,
        path_mask: torch.Tensor,
        anchor_mask: torch.Tensor,
    ) -> None:
        if (
            base.ndim != 6
            or anchors.ndim != 7
            or not torch.is_complex(base)
            or not torch.is_complex(anchors)
            or anchors.shape[0] != base.shape[0]
            or anchors.shape[2:] != base.shape[1:]
        ):
            raise ValueError("invalid frozen base/Anchor Beam-Delay tensors")
        batch, anchor_count = anchors.shape[:2]
        if (
            base.shape[3:5] != (self.config.p_count, self.config.n_count)
            or anchor_positions.shape != (batch, anchor_count, 3)
            or target_positions.shape != (batch, 3)
            or target_tokens.ndim != 3
            or target_tokens.shape[0] != batch
            or target_tokens.shape[-1] != self.config.path_feature_dim
            or anchor_tokens.shape != (
                batch,
                anchor_count,
                target_tokens.shape[1],
                self.config.path_feature_dim,
            )
            or anchor_target_tokens.shape != anchor_tokens.shape
            or path_mask.shape != anchor_tokens.shape[:-1]
            or anchor_mask.shape != (batch, anchor_count)
            or path_mask.dtype != torch.bool
            or anchor_mask.dtype != torch.bool
            or not anchor_mask.any(dim=1).all()
        ):
            raise ValueError("invalid AC-CGMRF condition shapes")
        real_values = (
            anchor_positions,
            target_positions,
            target_tokens,
            anchor_tokens,
            anchor_target_tokens,
        )
        if not all(torch.isfinite(value).all() for value in real_values):
            raise ValueError("AC-CGMRF real conditions must be finite")
        if not torch.isfinite(base).all() or not torch.isfinite(anchors).all():
            raise ValueError("AC-CGMRF complex inputs must be finite")

    @staticmethod
    def _group_summary(values: torch.Tensor) -> torch.Tensor:
        """Return per-P/N log power and coherent normalized real/imag."""

        reduce_dims = tuple(range(1, values.ndim - 3)) + (values.ndim - 1,)
        power = values.abs().square().mean(dim=reduce_dims).clamp_min(1e-12)
        coherent = values.mean(dim=reduce_dims) / power.sqrt()
        return torch.cat(
            (torch.log1p(power), coherent.real, coherent.imag), dim=-1
        ).flatten(1)

    def _map_context(
        self,
        target_tokens: torch.Tensor,
        anchor_tokens: torch.Tensor,
        anchor_target_tokens: torch.Tensor,
        path_mask: torch.Tensor,
        anchor_distances: torch.Tensor,
        anchor_mask: torch.Tensor,
    ) -> torch.Tensor:
        expanded_target = target_tokens[:, None].expand_as(anchor_tokens)
        combined = torch.cat(
            (
                expanded_target,
                anchor_tokens,
                expanded_target - anchor_tokens,
                anchor_target_tokens,
            ),
            dim=-1,
        )
        encoded = self.map_encoder(combined)
        valid_path = path_mask & anchor_mask[:, :, None]
        per_anchor = (
            encoded
            * valid_path[..., None].to(encoded.dtype)
        ).sum(dim=2) / valid_path.sum(dim=2, keepdim=True).clamp_min(1)
        distance_weights = torch.where(
            anchor_mask,
            1.0 / anchor_distances.clamp_min(1e-3).square(),
            torch.zeros_like(anchor_distances),
        )
        distance_weights = distance_weights / distance_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-12)
        return (per_anchor * distance_weights[..., None]).sum(dim=1)

    def _phase_reference(
        self,
        centers: torch.Tensor,
        base: torch.Tensor,
        anchor_reference: torch.Tensor,
    ) -> torch.Tensor:
        batch, atom_count = centers.shape[:2]
        h_count, v_count, _, _, delay_count = base.shape[1:]
        h_index = torch.remainder(torch.round(centers[..., 0]).long(), h_count)
        v_index = torch.remainder(torch.round(centers[..., 1]).long(), v_count)
        d_index = torch.round(centers[..., 2]).long().clamp(0, delay_count - 1)
        batch_index = torch.arange(batch, device=base.device)[:, None].expand(
            batch, atom_count
        )
        anchor_sample = anchor_reference[
            batch_index, h_index, v_index, :, :, d_index
        ]
        base_sample = base[batch_index, h_index, v_index, :, :, d_index]
        threshold = (
            base.abs().flatten(1).amax(dim=1)[:, None, None, None] * 1e-6
        )
        reference = torch.where(
            anchor_sample.abs() > threshold,
            anchor_sample,
            torch.where(
                base_sample.abs() > threshold,
                base_sample,
                torch.ones_like(base_sample),
            ),
        )
        return _complex_unit(reference)

    def forward(
        self,
        base_beam_delay: torch.Tensor,
        anchor_beam_delay: torch.Tensor,
        anchor_positions: torch.Tensor,
        target_positions: torch.Tensor,
        target_path_tokens: torch.Tensor,
        anchor_path_tokens: torch.Tensor,
        anchor_target_tokens: torch.Tensor,
        path_mask: torch.Tensor,
        anchor_mask: torch.Tensor,
        mode: MapMode = "real",
        *,
        base_reliability: torch.Tensor | None = None,
        enabled: bool = True,
        progress: float = 1.0,
        map_permutation: torch.Tensor | None = None,
    ) -> ACCGMRFOutput:
        self._validate(
            base_beam_delay,
            anchor_beam_delay,
            anchor_positions,
            target_positions,
            target_path_tokens,
            anchor_path_tokens,
            anchor_target_tokens,
            path_mask,
            anchor_mask,
        )
        if mode not in ("real", "zero", "shuffle"):
            raise ValueError("mode must be real, zero, or shuffle")
        if not math.isfinite(float(progress)) or not 0 <= float(progress) <= 1:
            raise ValueError("progress must be in [0,1]")
        batch = len(base_beam_delay)
        if base_reliability is None:
            base_reliability = torch.zeros(
                batch,
                device=base_beam_delay.device,
                dtype=base_beam_delay.real.dtype,
            )
        elif (
            base_reliability.shape != (batch,)
            or not torch.isfinite(base_reliability).all()
        ):
            raise ValueError("base_reliability must be finite with shape B")
        zero = torch.zeros_like(base_beam_delay)
        zero_scalar = torch.zeros(
            batch, device=base_beam_delay.device, dtype=base_beam_delay.real.dtype
        )
        if not enabled or mode == "zero":
            return ACCGMRFOutput(
                prediction=base_beam_delay,
                residual=zero,
                raw_residual=zero,
                atoms=None,
                trust=zero_scalar,
                support_novelty=zero_scalar,
                diagnostics={
                    "residual_energy_ratio": zero_scalar,
                    "raw_energy_ratio": zero_scalar,
                    "trust_abs_mean": zero_scalar,
                },
            )

        if mode == "shuffle":
            if map_permutation is None:
                if batch < 2:
                    raise ValueError("shuffle mode needs at least two samples")
                map_permutation = torch.roll(
                    torch.arange(batch, device=base_beam_delay.device), 1
                )
            if (
                map_permutation.shape != (batch,)
                or map_permutation.dtype != torch.long
                or not torch.equal(
                    torch.sort(map_permutation).values,
                    torch.arange(batch, device=map_permutation.device),
                )
            ):
                raise ValueError("map_permutation must be a batch permutation")
            target_path_tokens = target_path_tokens[map_permutation]
            anchor_path_tokens = anchor_path_tokens[map_permutation]
            anchor_target_tokens = anchor_target_tokens[map_permutation]
            path_mask = path_mask[map_permutation]

        anchor_distances = torch.linalg.vector_norm(
            anchor_positions - target_positions[:, None, :], dim=-1
        )
        map_context = self._map_context(
            target_path_tokens,
            anchor_path_tokens,
            anchor_target_tokens,
            path_mask,
            anchor_distances,
            anchor_mask,
        )
        weights = torch.where(
            anchor_mask,
            1.0 / anchor_distances.clamp_min(1e-3).square(),
            torch.zeros_like(anchor_distances),
        )
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        anchor_reference = (
            anchor_beam_delay * weights[:, :, None, None, None, None, None]
        ).sum(dim=1)
        relative = anchor_positions - target_positions[:, None, :]
        weighted_relative = (relative * weights[..., None]).sum(dim=1)
        valid_distances = torch.where(
            anchor_mask, anchor_distances, torch.zeros_like(anchor_distances)
        )
        distance_count = anchor_mask.sum(dim=1).clamp_min(1)
        geometry = torch.cat(
            (
                target_positions,
                weighted_relative,
                torch.log1p(
                    valid_distances.sum(dim=1, keepdim=True)
                    / distance_count[:, None]
                ),
                torch.log1p(
                    torch.where(
                        anchor_mask,
                        anchor_distances,
                        torch.full_like(anchor_distances, float("inf")),
                    ).amin(dim=1, keepdim=True)
                ),
                torch.log1p(anchor_distances.masked_fill(~anchor_mask, 0).amax(dim=1, keepdim=True)),
                anchor_mask.to(base_beam_delay.real.dtype).mean(dim=1, keepdim=True),
                base_reliability[:, None].to(base_beam_delay.real.dtype),
            ),
            dim=-1,
        )
        condition = self.condition_encoder(
            torch.cat(
                (
                    map_context,
                    self._group_summary(base_beam_delay),
                    self._group_summary(anchor_reference),
                    geometry.to(map_context.dtype),
                ),
                dim=-1,
            )
        )
        atom_state = torch.cat(
            (
                condition[:, None, :].expand(-1, self.config.atom_count, -1),
                self.atom_embedding[None, :, :].expand(batch, -1, -1),
            ),
            dim=-1,
        )
        raw = self.atom_head(atom_state)
        h_count, v_count, p_count, n_count, delay_count = base_beam_delay.shape[1:]
        centers = torch.stack(
            (
                torch.sigmoid(raw[..., 0]) * max(h_count - 1, 0),
                torch.sigmoid(raw[..., 1]) * max(v_count - 1, 0),
                torch.sigmoid(raw[..., 2]) * max(delay_count - 1, 0),
            ),
            dim=-1,
        )
        sigmas = torch.stack(
            (
                _linear_parameter(
                    raw[..., 3],
                    self.config.sigma_h_min,
                    self.config.sigma_h_max,
                ),
                _linear_parameter(
                    raw[..., 4],
                    self.config.sigma_v_min,
                    self.config.sigma_v_max,
                ),
                _linear_parameter(
                    raw[..., 5],
                    self.config.sigma_delay_min,
                    self.config.sigma_delay_max,
                ),
            ),
            dim=-1,
        )
        existence = torch.sigmoid(raw[..., 8])
        reliability = torch.sigmoid(raw[..., 9])
        amplitude = torch.nn.functional.softplus(raw[..., 6] - 2.0)
        phase_scale = 1.0 if self.phase_unlocked else 0.0
        phase = (
            torch.tanh(raw[..., 7])
            * self.config.phase_residual_limit
            * phase_scale
        )
        complex_gain = torch.polar(
            amplitude * existence * reliability, phase
        )
        cursor = 10
        p_log_amplitude = 0.25 * torch.tanh(
            raw[..., cursor : cursor + p_count]
        )
        cursor += p_count
        p_phase = (
            torch.tanh(raw[..., cursor : cursor + p_count])
            * self.config.coupling_phase_limit
            * phase_scale
        )
        cursor += p_count
        n_log_amplitude = 0.25 * torch.tanh(
            raw[..., cursor : cursor + n_count]
        )
        cursor += n_count
        n_phase = (
            torch.tanh(raw[..., cursor : cursor + n_count])
            * self.config.coupling_phase_limit
            * phase_scale
        )
        p_factor = torch.polar(torch.exp(p_log_amplitude), p_phase)
        n_factor = torch.polar(torch.exp(n_log_amplitude), n_phase)
        coupling = p_factor[..., :, None] * n_factor[..., None, :]
        coupling = coupling / coupling.abs().square().mean(
            dim=(-2, -1), keepdim=True
        ).sqrt().clamp_min(1e-8)
        phase_reference = self._phase_reference(
            centers, base_beam_delay, anchor_reference
        )
        atoms = MultipathAtoms(
            centers=centers,
            sigmas=sigmas,
            complex_gain=complex_gain,
            coupling=coupling,
            phase_reference=phase_reference,
            existence=existence,
            reliability=reliability,
            delay_phase_ramp=continuous_delay_phase_ramp(
                centers[..., 2], delay_count
            ),
        )
        raw_residual = continuous_complex_splat(
            atoms,
            (h_count, v_count, p_count, n_count, delay_count),
            (
                self.config.support_h,
                self.config.support_v,
                self.config.support_delay,
            ),
        )
        base_energy = base_beam_delay.abs().square().flatten(1).mean(dim=1)
        raw_energy = raw_residual.abs().square().flatten(1).mean(dim=1)
        energy_limit = (
            float(progress)
            * self.config.max_residual_ratio
            * base_energy.sqrt()
        )
        energy_scale = torch.minimum(
            torch.ones_like(energy_limit),
            energy_limit / raw_energy.sqrt().clamp_min(1e-12),
        )
        bounded = raw_residual * energy_scale[:, None, None, None, None, None]
        trust = torch.tanh(self.trust_head(condition)).squeeze(-1)
        residual = bounded * trust[:, None, None, None, None, None]
        prediction = base_beam_delay + residual
        novelty = support_novelty_ratio(
            residual,
            anchor_beam_delay,
            anchor_mask,
            self.config.support_threshold,
        )
        residual_energy = residual.abs().square().flatten(1).mean(dim=1)
        return ACCGMRFOutput(
            prediction=prediction,
            residual=residual,
            raw_residual=raw_residual,
            atoms=atoms,
            trust=trust,
            support_novelty=novelty,
            diagnostics={
                "residual_energy_ratio": residual_energy
                / base_energy.clamp_min(1e-12),
                "raw_energy_ratio": raw_energy / base_energy.clamp_min(1e-12),
                "energy_scale": energy_scale,
                "trust_abs_mean": trust.abs(),
                "existence_mean": existence.mean(dim=1),
                "reliability_mean": reliability.mean(dim=1),
            },
        )
