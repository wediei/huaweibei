"""Per-anchor Gaussian-conditioned channel transport before anchor aggregation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .gaussian_path_transport import _CrossBlock, _group_features
from .path_transport import TransportParameters, apply_transport_torch


@dataclass(frozen=True)
class GaussianAnchorTransportConfig:
    p_count: int
    n_count: int
    map_feature_dim: int = 81
    d_model: int = 96
    heads: int = 4
    layers: int = 2
    max_h_shift: float = 2.0
    max_v_shift: float = 2.0
    max_delay_shift: float = 8.0
    learned_anchor_fusion: bool = False

    def __post_init__(self) -> None:
        for name in (
            "p_count",
            "n_count",
            "map_feature_dim",
            "d_model",
            "heads",
            "layers",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        for name in ("max_h_shift", "max_v_shift", "max_delay_shift"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not isinstance(self.learned_anchor_fusion, bool):
            raise ValueError("learned_anchor_fusion must be boolean")


@dataclass(frozen=True)
class GaussianAnchorTransportOutput:
    coarse: torch.Tensor
    transported: torch.Tensor
    anchor_delta: torch.Tensor
    parameters: TransportParameters
    anchor_weights: torch.Tensor
    attention: torch.Tensor


class GaussianAnchorTransport(nn.Module):
    """Transport every source anchor with its paired map path, then mix deltas."""

    GROUP_FEATURE_DIM = 11

    def __init__(self, config: GaussianAnchorTransportConfig) -> None:
        super().__init__()
        self.config = config
        groups = config.p_count * config.n_count
        self.group_embedding = nn.Parameter(
            torch.randn(groups, config.d_model) * 0.02
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(self.GROUP_FEATURE_DIM + 1, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.map_encoder = nn.Sequential(
            nn.Linear(config.map_feature_dim, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.null_map_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.blocks = nn.ModuleList(
            [_CrossBlock(config.d_model, config.heads) for _ in range(config.layers)]
        )
        self.parameter_head = nn.Linear(config.d_model, 8)
        nn.init.zeros_(self.parameter_head.weight)
        nn.init.zeros_(self.parameter_head.bias)
        if config.learned_anchor_fusion:
            self.anchor_norm = nn.LayerNorm(config.d_model)
            self.anchor_attention = nn.MultiheadAttention(
                config.d_model, config.heads, batch_first=True
            )
            self.fusion_head = nn.Linear(config.d_model, 1)
            nn.init.zeros_(self.fusion_head.weight)
            nn.init.zeros_(self.fusion_head.bias)

    def _raw_parameters(
        self,
        anchor_beam_delay: torch.Tensor,
        pair_tokens: torch.Tensor,
        pair_mask: torch.Tensor,
        anchor_distances: torch.Tensor,
        anchor_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch, anchors = anchor_beam_delay.shape[:2]
        flat_anchor = anchor_beam_delay.reshape(
            batch * anchors, *anchor_beam_delay.shape[2:]
        )
        features = _group_features(flat_anchor)
        groups = self.config.p_count * self.config.n_count
        distance = torch.log1p(anchor_distances).reshape(
            batch * anchors, 1, 1
        ).expand(-1, groups, -1)
        query_base = self.query_encoder(
            torch.cat(
                (features.reshape(batch * anchors, groups, -1), distance),
                dim=-1,
            )
        )
        query_base = query_base + self.group_embedding[None, :, :]
        flat_tokens = pair_tokens.reshape(
            batch * anchors, pair_tokens.shape[2], pair_tokens.shape[3]
        )
        flat_mask = pair_mask.reshape(batch * anchors, pair_mask.shape[2])
        context = self.map_encoder(flat_tokens)
        null = self.null_map_token.to(context.dtype).expand(
            batch * anchors, -1, -1
        )
        context = torch.cat((null, context), dim=1)
        context_mask = torch.cat(
            (
                torch.ones(
                    batch * anchors,
                    1,
                    dtype=torch.bool,
                    device=flat_mask.device,
                ),
                flat_mask,
            ),
            dim=1,
        )
        query = query_base
        attention = torch.empty(
            batch * anchors,
            groups,
            context.shape[1],
            device=context.device,
            dtype=context.dtype,
        )
        for block in self.blocks:
            query, attention = block(query, context, context_mask)

        # Remove every map-independent correction exactly.
        zero_context = self.map_encoder(torch.zeros_like(flat_tokens))
        zero_context = torch.cat((null, zero_context), dim=1)
        zero_query = query_base
        for block in self.blocks:
            zero_query, _ = block(zero_query, zero_context, context_mask)
        fusion_residual = None
        if self.config.learned_anchor_fusion:
            def compare(values):
                grouped = values.reshape(batch, anchors, groups, -1)
                grouped = grouped.permute(0, 2, 1, 3).reshape(
                    batch * groups, anchors, -1
                )
                normalized = self.anchor_norm(grouped)
                compared, _ = self.anchor_attention(
                    normalized,
                    normalized,
                    normalized,
                    key_padding_mask=~anchor_mask[:, None, :]
                    .expand(batch, groups, anchors)
                    .reshape(batch * groups, anchors),
                    need_weights=False,
                )
                return (grouped + compared).reshape(
                    batch, groups, anchors, -1
                ).permute(0, 2, 1, 3).reshape(
                    batch * anchors, groups, -1
                )

            query = compare(query)
            zero_query = compare(zero_query)
            fusion_residual = (
                self.fusion_head(query) - self.fusion_head(zero_query)
            ).reshape(batch, anchors, self.config.p_count, self.config.n_count)
        raw = self.parameter_head(query)
        raw = raw - self.parameter_head(zero_query)
        return (
            raw.reshape(
                batch,
                anchors,
                self.config.p_count,
                self.config.n_count,
                8,
            ),
            attention,
            fusion_residual,
        )

    def forward(
        self,
        coarse_beam_delay: torch.Tensor,
        anchor_beam_delay: torch.Tensor,
        pair_tokens: torch.Tensor,
        pair_mask: torch.Tensor,
        anchor_distances: torch.Tensor,
        anchor_mask: torch.Tensor,
    ) -> GaussianAnchorTransportOutput:
        if (
            coarse_beam_delay.ndim != 6
            or anchor_beam_delay.ndim != 7
            or not torch.is_complex(coarse_beam_delay)
            or not torch.is_complex(anchor_beam_delay)
            or anchor_beam_delay.shape[0] != coarse_beam_delay.shape[0]
            or anchor_beam_delay.shape[2:] != coarse_beam_delay.shape[1:]
        ):
            raise ValueError("coarse/anchor beam-delay shapes differ")
        batch, anchors = anchor_beam_delay.shape[:2]
        if (
            pair_tokens.shape[:2] != (batch, anchors)
            or pair_tokens.ndim != 4
            or pair_tokens.shape[-1] != self.config.map_feature_dim
            or pair_mask.shape != pair_tokens.shape[:3]
            or pair_mask.dtype != torch.bool
            or anchor_distances.shape != (batch, anchors)
            or anchor_mask.shape != (batch, anchors)
            or anchor_mask.dtype != torch.bool
            or not anchor_mask.any(dim=1).all()
        ):
            raise ValueError("invalid per-anchor map context")
        if (
            not torch.isfinite(pair_tokens).all()
            or not torch.isfinite(anchor_distances[anchor_mask]).all()
            or (anchor_distances[anchor_mask] < 0).any()
        ):
            raise ValueError("per-anchor inputs must be finite")

        raw, attention, fusion_residual = self._raw_parameters(
            anchor_beam_delay,
            pair_tokens.to(dtype=coarse_beam_delay.real.dtype),
            pair_mask,
            anchor_distances.to(dtype=coarse_beam_delay.real.dtype),
            anchor_mask,
        )
        parameters = TransportParameters(
            delta_h=torch.tanh(raw[..., 0]) * self.config.max_h_shift,
            delta_v=torch.tanh(raw[..., 1]) * self.config.max_v_shift,
            delta_delay=torch.tanh(raw[..., 2]) * self.config.max_delay_shift,
            log_amplitude=2.0 * torch.tanh(raw[..., 3]),
            phase_real=1.0 + 0.5 * torch.tanh(raw[..., 4]),
            phase_imag=0.5 * torch.tanh(raw[..., 5]),
            existence=(1.0 - 0.5 * torch.tanh(raw[..., 6])).clamp(0.0, 1.0),
            reliability=torch.sigmoid(raw[..., 7]),
        )
        if int(torch.count_nonzero(pair_tokens.detach()).item()) == 0:
            moved = anchor_beam_delay
        else:
            flat = anchor_beam_delay.reshape(
                batch * anchors, *anchor_beam_delay.shape[2:]
            )
            flat_parameters = TransportParameters(
                **{
                    name: getattr(parameters, name).reshape(
                        batch * anchors,
                        self.config.p_count,
                        self.config.n_count,
                    )
                    for name in parameters.__dataclass_fields__
                }
            )
            moved = apply_transport_torch(flat, flat_parameters).reshape_as(
                anchor_beam_delay
            )
        valid_distances = torch.where(
            anchor_mask,
            anchor_distances.to(dtype=coarse_beam_delay.real.dtype),
            torch.full_like(anchor_distances, float("inf")),
        )
        base_weights = torch.where(
            anchor_mask,
            1.0 / valid_distances.clamp_min(1e-3).square(),
            torch.zeros_like(valid_distances),
        )
        if fusion_residual is None:
            weights = base_weights / base_weights.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
            broadcast_weights = weights[:, :, None, None, None, None, None]
        else:
            logits = torch.log(base_weights.clamp_min(1e-12))[
                :, :, None, None
            ] + 2.0 * torch.tanh(fusion_residual)
            logits = logits.masked_fill(
                ~anchor_mask[:, :, None, None], float("-inf")
            )
            weights = torch.softmax(logits, dim=1)
            broadcast_weights = weights[
                :, :, None, None, :, :, None
            ]
        delta = (
            (moved - anchor_beam_delay)
            * broadcast_weights
        ).sum(dim=1)
        transported = coarse_beam_delay + delta
        return GaussianAnchorTransportOutput(
            coarse_beam_delay,
            transported,
            delta,
            parameters,
            weights,
            attention.reshape(batch, anchors, *attention.shape[1:]),
        )
