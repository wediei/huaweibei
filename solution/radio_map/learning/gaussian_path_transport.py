"""Small Gaussian-token cross-attention head for equivalent path transport."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .path_transport import TransportParameters, apply_transport_torch


@dataclass(frozen=True)
class GaussianPathTransportConfig:
    p_count: int
    n_count: int
    map_feature_dim: int
    d_model: int = 96
    heads: int = 4
    layers: int = 2
    max_h_shift: float = 2.0
    max_v_shift: float = 2.0
    max_delay_shift: float = 8.0
    causal_map_residual: bool = False
    gaussian_count: int = 0
    radio_feature_dim: int = 0
    radio_grid_sizes: tuple[int, ...] = ()

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
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        for name in ("max_h_shift", "max_v_shift", "max_delay_shift"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        if not isinstance(self.causal_map_residual, bool):
            raise ValueError("causal_map_residual must be boolean")
        for name in ("gaussian_count", "radio_feature_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (self.gaussian_count == 0) != (self.radio_feature_dim == 0):
            raise ValueError(
                "gaussian_count and radio_feature_dim must be enabled together"
            )
        if not isinstance(self.radio_grid_sizes, (tuple, list)):
            raise ValueError("radio_grid_sizes must be a sequence")
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 2
            for value in self.radio_grid_sizes
        ):
            raise ValueError("radio grid sizes must be integers >= 2")
        if self.radio_grid_sizes and not self.gaussian_count:
            raise ValueError("radio grids require an enabled Gaussian field")


@dataclass(frozen=True)
class GaussianTransportOutput:
    coarse: torch.Tensor
    transported: torch.Tensor
    parameters: TransportParameters
    attention: torch.Tensor
    group_features: torch.Tensor


class _CrossBlock(nn.Module):
    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, heads, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, weights = self.attention(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            key_padding_mask=~context_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        query = query + attended
        query = query + self.feed_forward(query)
        return query, weights


def _group_features(values: torch.Tensor) -> torch.Tensor:
    # B,H,V,P,N,D -> B,P,N,H,V,D
    grouped = values.permute(0, 3, 4, 1, 2, 5)
    power = grouped.abs().square()
    total = power.sum(dim=(3, 4, 5)).clamp_min(1e-12)
    weights = power / total[:, :, :, None, None, None]
    sizes = grouped.shape[3:]
    coordinates = [
        torch.linspace(
            -1.0,
            1.0,
            size,
            device=values.device,
            dtype=values.real.dtype,
        )
        for size in sizes
    ]
    h_grid = coordinates[0].reshape(1, 1, 1, sizes[0], 1, 1)
    v_grid = coordinates[1].reshape(1, 1, 1, 1, sizes[1], 1)
    d_grid = coordinates[2].reshape(1, 1, 1, 1, 1, sizes[2])
    grids = (h_grid, v_grid, d_grid)
    centroids = [(weights * grid).sum(dim=(3, 4, 5)) for grid in grids]
    spreads = [
        torch.sqrt(
            (
                weights
                * (grid - centroid[:, :, :, None, None, None]).square()
            )
            .sum(dim=(3, 4, 5))
            .clamp_min(0.0)
        )
        for grid, centroid in zip(grids, centroids)
    ]
    peak = power.amax(dim=(3, 4, 5))
    entropy = -(
        weights * weights.clamp_min(1e-12).log()
    ).sum(dim=(3, 4, 5)) / math.log(max(2, int(math.prod(sizes))))
    coherent = grouped.sum(dim=(3, 4, 5))
    coherent = coherent / coherent.abs().clamp_min(1e-8)
    return torch.stack(
        (
            torch.log1p(total),
            torch.log1p(peak),
            entropy,
            *centroids,
            *spreads,
            coherent.real,
            coherent.imag,
        ),
        dim=-1,
    )


class GaussianPathTransport(nn.Module):
    """Predict a few group transport parameters, never a full CSI tensor."""

    GROUP_FEATURE_DIM = 11

    def __init__(
        self,
        config: GaussianPathTransportConfig,
        normalization_mean: torch.Tensor | None = None,
        normalization_std: torch.Tensor | None = None,
        gaussian_grid_ids: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        group_count = config.p_count * config.n_count
        self.group_embedding = nn.Parameter(
            torch.randn(group_count, config.d_model) * 0.02
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(self.GROUP_FEATURE_DIM, config.d_model),
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
        if config.gaussian_count:
            if config.radio_grid_sizes:
                levels = len(config.radio_grid_sizes)
                if gaussian_grid_ids is None:
                    grid_ids = torch.zeros(
                        config.gaussian_count + 1, levels, dtype=torch.long
                    )
                else:
                    grid_ids = torch.as_tensor(
                        gaussian_grid_ids, dtype=torch.long
                    )
                if grid_ids.shape != (config.gaussian_count + 1, levels):
                    raise ValueError("Gaussian radio-grid mapping shape differs")
                for level, size in enumerate(config.radio_grid_sizes):
                    if (
                        (grid_ids[:, level] < 0).any()
                        or (grid_ids[:, level] >= size).any()
                    ):
                        raise ValueError("Gaussian radio-grid id is out of range")
                self.register_buffer(
                    "gaussian_grid_ids", grid_ids.clone(), persistent=True
                )
                self.radio_codes = nn.ModuleList(
                    [
                        nn.Embedding(
                            size, config.radio_feature_dim, padding_idx=0
                        )
                        for size in config.radio_grid_sizes
                    ]
                )
                self.radio_log_opacity = nn.ModuleList(
                    [
                        nn.Embedding(size, 1, padding_idx=0)
                        for size in config.radio_grid_sizes
                    ]
                )
                for table in self.radio_codes:
                    nn.init.zeros_(table.weight)
                for table in self.radio_log_opacity:
                    nn.init.zeros_(table.weight)
            else:
                self.radio_codes = nn.Embedding(
                    config.gaussian_count + 1,
                    config.radio_feature_dim,
                    padding_idx=0,
                )
                self.radio_log_opacity = nn.Embedding(
                    config.gaussian_count + 1, 1, padding_idx=0
                )
                # Unobserved test-only Gaussians must be neutral rather than
                # random. Training paths activate only the radio codes
                # supported by official training channels.
                nn.init.zeros_(self.radio_codes.weight)
                nn.init.zeros_(self.radio_log_opacity.weight)
            self.radio_token_projection = nn.Linear(
                config.radio_feature_dim, config.d_model, bias=False
            )
            self.radio_query_projection = nn.Linear(
                config.radio_feature_dim, config.d_model, bias=False
            )
            mean = (
                torch.zeros(config.map_feature_dim)
                if normalization_mean is None
                else torch.as_tensor(normalization_mean, dtype=torch.float32)
            )
            std = (
                torch.ones(config.map_feature_dim)
                if normalization_std is None
                else torch.as_tensor(normalization_std, dtype=torch.float32)
            )
            if mean.shape != (config.map_feature_dim,) or std.shape != mean.shape:
                raise ValueError("Gaussian token normalization shape differs")
            self.register_buffer("token_mean", mean.clone(), persistent=True)
            self.register_buffer("token_std", std.clone(), persistent=True)

    def _radio_field(
        self,
        map_tokens: torch.Tensor,
        map_mask: torch.Tensor,
        gaussian_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            gaussian_indices.shape != map_mask.shape
            or gaussian_indices.dtype != torch.long
            or (gaussian_indices[map_mask] < 0).any()
            or (gaussian_indices[map_mask] >= self.config.gaussian_count).any()
        ):
            raise ValueError("invalid Gaussian field indices")
        lookup = torch.where(
            map_mask, gaussian_indices + 1, torch.zeros_like(gaussian_indices)
        )
        if self.config.radio_grid_sizes:
            grid_ids = self.gaussian_grid_ids[lookup]
            codes = torch.stack(
                [
                    table(grid_ids[..., level])
                    for level, table in enumerate(self.radio_codes)
                ],
                dim=0,
            ).mean(dim=0)
            opacity = torch.stack(
                [
                    table(grid_ids[..., level]).squeeze(-1)
                    for level, table in enumerate(self.radio_log_opacity)
                ],
                dim=0,
            ).mean(dim=0)
        else:
            codes = self.radio_codes(lookup)
            opacity = self.radio_log_opacity(lookup).squeeze(-1)
        raw = (
            map_tokens.to(torch.float32) * self.token_std[None, None, :]
            + self.token_mean[None, None, :]
        )
        base_response = raw[..., 15].clamp(0.0, 1.0 - 1e-6)
        density_gain = torch.exp(2.0 * torch.tanh(opacity))
        adjusted_response = 1.0 - torch.pow(
            1.0 - base_response, density_gain
        )
        alpha = (
            adjusted_response
            * raw[..., 16].clamp(0.0, 1.0)
            * raw[..., 26].clamp(0.0, 1.0)
        )
        alpha = alpha * map_mask.to(alpha.dtype)
        weights = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)
        token_context = self.radio_token_projection(codes) * alpha[..., None]
        pooled = (codes * weights[..., None]).sum(dim=1)
        return token_context, self.radio_query_projection(pooled)

    def forward(
        self,
        coarse_beam_delay: torch.Tensor,
        map_tokens: torch.Tensor,
        map_mask: torch.Tensor,
        gaussian_indices: torch.Tensor | None = None,
    ) -> GaussianTransportOutput:
        if (
            coarse_beam_delay.ndim != 6
            or not torch.is_complex(coarse_beam_delay)
            or coarse_beam_delay.shape[3] != self.config.p_count
            or coarse_beam_delay.shape[4] != self.config.n_count
        ):
            raise ValueError(
                "coarse beam-delay must be complex (B,H,V,P,N,D)"
            )
        batch_size = coarse_beam_delay.shape[0]
        if (
            map_tokens.ndim != 3
            or map_tokens.shape[0] != batch_size
            or map_tokens.shape[2] != self.config.map_feature_dim
            or not torch.is_floating_point(map_tokens)
            or not torch.isfinite(map_tokens).all()
        ):
            raise ValueError("map token shape or dtype differs from config")
        if (
            map_mask.shape != map_tokens.shape[:2]
            or map_mask.dtype != torch.bool
        ):
            raise ValueError("map mask must be boolean with shape (B,M)")
        features = _group_features(coarse_beam_delay)
        group_count = self.config.p_count * self.config.n_count
        query_base = self.query_encoder(
            features.reshape(batch_size, group_count, -1)
        )
        query_base = query_base + self.group_embedding[None, :, :]
        context = self.map_encoder(
            map_tokens.to(dtype=coarse_beam_delay.real.dtype)
        )
        if self.config.gaussian_count:
            if gaussian_indices is None:
                raise ValueError("learnable radio field requires Gaussian indices")
            token_radio, pooled_radio = self._radio_field(
                map_tokens, map_mask, gaussian_indices
            )
            context = context + token_radio.to(context.dtype)
            query_base = query_base + pooled_radio[:, None, :].to(
                query_base.dtype
            )
        elif gaussian_indices is not None:
            raise ValueError("Gaussian indices supplied to a geometry-only model")
        null = self.null_map_token.to(context.dtype).expand(batch_size, -1, -1)
        context = torch.cat((null, context), dim=1)
        context_mask = torch.cat(
            (
                torch.ones(
                    batch_size, 1, dtype=torch.bool, device=map_mask.device
                ),
                map_mask,
            ),
            dim=1,
        )
        attention = torch.empty(
            batch_size,
            group_count,
            context.shape[1],
            device=context.device,
            dtype=context.dtype,
        )
        query = query_base
        for block in self.blocks:
            query, attention = block(query, context, context_mask)
        raw = self.parameter_head(query)
        if self.config.causal_map_residual:
            # Remove the exact zero-map branch.  Coarse/group embeddings can
            # no longer create a deployable correction without map evidence.
            zero_context = self.map_encoder(torch.zeros_like(map_tokens))
            zero_query_base = query_base
            if self.config.gaussian_count:
                zero_tokens = torch.zeros_like(map_tokens)
                zero_radio, zero_pooled = self._radio_field(
                    zero_tokens, map_mask, gaussian_indices
                )
                zero_context = zero_context + zero_radio.to(zero_context.dtype)
                zero_query_base = (
                    query_base
                    - pooled_radio[:, None, :].to(query_base.dtype)
                    + zero_pooled[:, None, :].to(query_base.dtype)
                )
            zero_context = torch.cat((null, zero_context), dim=1)
            zero_query = zero_query_base
            for block in self.blocks:
                zero_query, _ = block(
                    zero_query, zero_context, context_mask
                )
            raw = raw - self.parameter_head(zero_query)
        raw = raw.reshape(
            batch_size, self.config.p_count, self.config.n_count, 8
        )
        parameters = TransportParameters(
            delta_h=torch.tanh(raw[..., 0]) * self.config.max_h_shift,
            delta_v=torch.tanh(raw[..., 1]) * self.config.max_v_shift,
            delta_delay=torch.tanh(raw[..., 2])
            * self.config.max_delay_shift,
            log_amplitude=2.0 * torch.tanh(raw[..., 3]),
            phase_real=1.0 + 0.5 * torch.tanh(raw[..., 4]),
            phase_imag=0.5 * torch.tanh(raw[..., 5]),
            existence=(1.0 - 0.5 * torch.tanh(raw[..., 6])).clamp(0.0, 1.0),
            reliability=torch.sigmoid(raw[..., 7]),
        )
        if (
            self.config.causal_map_residual
            and int(torch.count_nonzero(raw.detach()).item()) == 0
        ):
            transported = coarse_beam_delay
        else:
            transported = apply_transport_torch(
                coarse_beam_delay, parameters
            )
        return GaussianTransportOutput(
            coarse=coarse_beam_delay,
            transported=transported,
            parameters=parameters,
            attention=attention,
            group_features=features,
        )


def apply_optional_gaussian_transport(
    model: Any,
    coarse_beam_delay: torch.Tensor,
    map_tokens: torch.Tensor | None,
    map_mask: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    value = float(scale)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("transport scale must be finite and non-negative")
    if value == 0.0:
        return coarse_beam_delay
    if map_tokens is None or map_mask is None:
        raise ValueError("enabled transport requires map tokens and mask")
    output = model(coarse_beam_delay, map_tokens, map_mask)
    return coarse_beam_delay + value * (
        output.transported - coarse_beam_delay
    )
