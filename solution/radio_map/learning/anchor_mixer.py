"""Support-aware complex latent mixing for fixed-support radio-map codecs.

The public batch contract is deliberately tensor-only so Task 5 can build it
without model-specific adapters.  Required fields are ``anchor_latents``
``(B,K,L)`` complex, ``anchor_distances`` ``(B,K)`` float,
``anchor_mask`` ``(B,K)`` bool, ``anchor_indices`` ``(B,K)`` integer stable
source IDs, ``pair_features`` ``(B,K,14)``, and the four
``(B,3)`` target position fields: ``target_positions``,
``target_positions_standardized``, ``target_bs_relative``, and
``target_bs_relative_standardized``.  If ``config.use_geometry`` is true it
also requires ``target_point_features`` ``(B,13)``, ``target_patch`` and
``bs_patch`` ``(B,13,H,W)``, ``bs_target_corridor`` ``(B,Lc,15)`` with
``bs_target_corridor_mask`` ``(B,Lc)``, and ``anchor_corridors``
``(B,K,La,15)`` with ``anchor_corridor_mask`` ``(B,K,La)``.  Geometry fields
are intentionally not read when geometry is disabled; their learned branches
are replaced with shape-stable zero features.

``anchor_latents`` are currently required to be ``complex64``; all real batch
features may use any floating dtype and are converted differentiably to its
``float32`` real dtype immediately before Fourier, linear, or convolutional
encoding.  Valid source IDs are nonnegative and unique per batch row.  For an
exact nearest-distance tie, the smallest valid source ID is selected, making
nearest-anchor behavior invariant to a consistent anchor-axis permutation.

``anchor_weights`` has shape ``(B,K,G)``.  Each non-empty fixed support group
normalizes across valid anchors; support groups with no coefficients have
exactly-zero weights.  The mixer never materializes a ``G x L`` assignment or
a dense ``2L`` output head: group weights are gathered with ``group_ids``.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import math

import torch
from torch import nn
from torch.nn import functional as F

from .model_components import CorridorEncoder, FourierFeatures, GroupSummaryEncoder, PatchEncoder


@dataclass(frozen=True)
class SupportAwareAnchorMixerConfig:
    """Dimensions and optional geometry controls for :class:`SupportAwareAnchorMixer`."""

    latent_size: int
    group_count: int | None = None
    d_model: int = 128
    group_dim: int = 64
    geometry_dim: int = 64
    low_rank: int = 32
    num_frequencies: int = 6
    use_geometry: bool = False

    def __post_init__(self) -> None:
        for name in ("latent_size", "d_model", "group_dim", "geometry_dim", "low_rank", "num_frequencies"):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.group_count is not None and (
            not isinstance(self.group_count, Integral)
            or isinstance(self.group_count, bool)
            or self.group_count < 1
        ):
            raise ValueError("group_count must be a positive integer or None")


@dataclass(frozen=True)
class MixerOutput:
    """Complex prediction and its explicitly inspectable support contributions."""

    latent: torch.Tensor
    nearest_latent: torch.Tensor
    anchor_weights: torch.Tensor
    alpha: torch.Tensor
    complex_gain: torch.Tensor
    low_rank_residual: torch.Tensor


class SupportAwareAnchorMixer(nn.Module):
    """Factorized group attention with a nearest-anchor-preserving initialization."""

    def __init__(self, config: SupportAwareAnchorMixerConfig, group_ids: torch.Tensor) -> None:
        super().__init__()
        if not isinstance(config, SupportAwareAnchorMixerConfig):
            raise TypeError("config must be a SupportAwareAnchorMixerConfig")
        ids = torch.as_tensor(group_ids)
        if ids.ndim != 1 or ids.numel() != config.latent_size:
            raise ValueError("group_ids must have shape (latent_size,)")
        if ids.dtype == torch.bool or ids.dtype.is_floating_point or torch.is_complex(ids):
            raise TypeError("group_ids must have an integer dtype")
        if (ids < 0).any():
            raise ValueError("group_ids must be nonnegative")
        inferred_groups = int(ids.max().item()) + 1
        group_count = inferred_groups if config.group_count is None else int(config.group_count)
        if inferred_groups > group_count:
            raise ValueError("group_ids must lie in [0, group_count)")

        self.config = config
        self.latent_size = int(config.latent_size)
        self.group_count = group_count
        self.register_buffer("group_ids", ids.to(dtype=torch.long), persistent=True)
        self.summary_encoder = GroupSummaryEncoder(self.group_ids, group_count)

        self.distance_features = FourierFeatures(1, config.num_frequencies)
        self.position_features = FourierFeatures(3, config.num_frequencies)
        self.anchor_geometry_dim = int(config.geometry_dim)
        self.target_geometry_dim = 5 * int(config.geometry_dim)
        anchor_input_dim = 14 + self.distance_features.single_output_dim + self.anchor_geometry_dim
        target_input_dim = 2 * self.position_features.output_dim + self.target_geometry_dim
        self.anchor_encoder = nn.Sequential(nn.Linear(anchor_input_dim, config.d_model), nn.SiLU(), nn.Linear(config.d_model, config.d_model), nn.SiLU())
        self.target_encoder = nn.Sequential(nn.Linear(target_input_dim, config.d_model), nn.SiLU(), nn.Linear(config.d_model, config.d_model), nn.SiLU())
        self.target_projection = nn.Linear(config.d_model, config.group_dim)
        self.anchor_projection = nn.Linear(config.d_model, config.group_dim)
        self.summary_projection = nn.Linear(6, config.group_dim)
        self.group_embedding = nn.Parameter(torch.empty(group_count, config.group_dim))
        nn.init.normal_(self.group_embedding, std=1.0 / math.sqrt(config.group_dim))
        self.distance_beta = nn.Parameter(torch.zeros(()))

        # Target and BS use the same geometry vocabulary.  Shared encoders are
        # intentionally invoked twice rather than duplicated by role.
        self.point_encoder = nn.Sequential(nn.Linear(13, config.geometry_dim), nn.SiLU(), nn.Linear(config.geometry_dim, config.geometry_dim))
        self.patch_encoder = PatchEncoder(13, config.geometry_dim)
        self.corridor_encoder = CorridorEncoder(15, config.geometry_dim)

        self.alpha_head = nn.Linear(config.d_model, group_count)
        self.gain_real_head = nn.Linear(config.d_model, group_count)
        self.gain_imag_head = nn.Linear(config.d_model, group_count)
        self.residual_real_head = nn.Linear(config.d_model, config.low_rank)
        self.residual_imag_head = nn.Linear(config.d_model, config.low_rank)
        self._zero_linear(self.alpha_head)
        self._zero_linear(self.gain_real_head)
        self._zero_linear(self.gain_imag_head)
        self._zero_linear(self.residual_real_head)
        self._zero_linear(self.residual_imag_head)
        basis_std = 1.0 / math.sqrt(self.latent_size)
        self.complex_basis = nn.Parameter(torch.complex(
            torch.randn(self.latent_size, config.low_rank) * basis_std,
            torch.randn(self.latent_size, config.low_rank) * basis_std,
        ))

    @staticmethod
    def _zero_linear(layer: nn.Linear) -> None:
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)

    @staticmethod
    def _float_tensor(batch: dict[str, torch.Tensor], name: str, shape: tuple[int, ...]) -> torch.Tensor:
        value = batch.get(name)
        if not isinstance(value, torch.Tensor) or not value.dtype.is_floating_point:
            raise TypeError(f"{name} must be a floating-point torch.Tensor")
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
        return value

    @staticmethod
    def _mask_tensor(batch: dict[str, torch.Tensor], name: str, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        value = batch.get(name)
        if not isinstance(value, torch.Tensor) or value.dtype != torch.bool or value.shape != shape or value.device != device:
            raise ValueError(f"{name} must be a bool tensor on the input device with shape {shape}")
        return value

    def _validate_core(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        if not isinstance(batch, dict):
            raise TypeError("batch must be a dictionary of tensors")
        latents = batch.get("anchor_latents")
        if not isinstance(latents, torch.Tensor) or latents.dtype != torch.complex64:
            raise TypeError("anchor_latents must be a complex64 torch.Tensor")
        if latents.ndim != 3 or latents.shape[-1] != self.latent_size:
            raise ValueError("anchor_latents must have shape (B, K, L) with L=latent_size")
        batch_size, anchor_count, _ = latents.shape
        if batch_size < 1 or anchor_count < 1:
            raise ValueError("anchor_latents must have non-empty B and K dimensions")
        mask = self._mask_tensor(batch, "anchor_mask", (batch_size, anchor_count), latents.device)
        if not mask.any(dim=1).all():
            raise ValueError("each batch item must contain at least one valid anchor")
        distances = self._float_tensor(batch, "anchor_distances", (batch_size, anchor_count))
        pair_features = self._float_tensor(batch, "pair_features", (batch_size, anchor_count, 14))
        source_ids = batch.get("anchor_indices")
        if (
            not isinstance(source_ids, torch.Tensor)
            or source_ids.ndim != 2
            or source_ids.shape != (batch_size, anchor_count)
            or source_ids.dtype == torch.bool
            or source_ids.dtype.is_floating_point
            or torch.is_complex(source_ids)
            or source_ids.device != latents.device
        ):
            raise ValueError("anchor_indices must be an integer tensor on the input device with shape (B, K)")
        if distances.device != latents.device or pair_features.device != latents.device:
            raise ValueError("anchor fields must be on the anchor_latents device")
        if not torch.isfinite(distances[mask]).all() or (distances[mask] < 0).any():
            raise ValueError("valid anchor_distances must be finite and nonnegative")
        if not torch.isfinite(pair_features[mask]).all() or not torch.isfinite(latents[mask]).all():
            raise ValueError("valid anchor_latents and pair_features must be finite")
        if (source_ids[mask] < 0).any():
            raise ValueError("valid anchor_indices must be nonnegative")
        for row in range(batch_size):
            valid_ids = source_ids[row, mask[row]]
            if valid_ids.unique().numel() != valid_ids.numel():
                raise ValueError("valid anchor_indices must be unique in each batch row")
        position_fields = tuple(
            self._float_tensor(batch, name, (batch_size, 3))
            for name in (
                "target_positions", "target_positions_standardized", "target_bs_relative", "target_bs_relative_standardized"
            )
        )
        if any(value.device != latents.device or not torch.isfinite(value).all() for value in position_fields):
            raise ValueError("target position fields must be finite and on the anchor_latents device")
        return latents, distances, mask, source_ids, pair_features, position_fields

    def _geometry_features(self, batch: dict[str, torch.Tensor], batch_size: int, anchor_count: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        zeros_target = torch.zeros(batch_size, self.target_geometry_dim, device=device, dtype=dtype)
        zeros_anchor = torch.zeros(batch_size, anchor_count, self.anchor_geometry_dim, device=device, dtype=dtype)
        if not self.config.use_geometry:
            return zeros_target, zeros_anchor
        point = self._float_tensor(batch, "target_point_features", (batch_size, 13))
        target_patch = batch.get("target_patch")
        bs_patch = batch.get("bs_patch")
        if not isinstance(target_patch, torch.Tensor) or not target_patch.dtype.is_floating_point or target_patch.ndim != 4 or target_patch.shape[:2] != (batch_size, 13) or target_patch.shape[2] < 1 or target_patch.shape[3] < 1:
            raise ValueError("target_patch must have shape (B, 13, H, W)")
        if not isinstance(bs_patch, torch.Tensor) or not bs_patch.dtype.is_floating_point or bs_patch.shape != target_patch.shape:
            raise ValueError("bs_patch must have the same shape as target_patch")
        corridor = batch.get("bs_target_corridor")
        anchor_corridor = batch.get("anchor_corridors")
        if not isinstance(corridor, torch.Tensor) or not corridor.dtype.is_floating_point or corridor.ndim != 3 or corridor.shape[0] != batch_size or corridor.shape[-1] != 15:
            raise ValueError("bs_target_corridor must have shape (B, Lc, 15)")
        if not isinstance(anchor_corridor, torch.Tensor) or not anchor_corridor.dtype.is_floating_point or anchor_corridor.ndim != 4 or anchor_corridor.shape[:2] != (batch_size, anchor_count) or anchor_corridor.shape[-1] != 15:
            raise ValueError("anchor_corridors must have shape (B, K, La, 15)")
        values = (point, target_patch, bs_patch, corridor, anchor_corridor)
        if any(value.device != device or not torch.isfinite(value).all() for value in values):
            raise ValueError("geometry values must be finite and on the anchor_latents device")
        corridor_mask = self._mask_tensor(batch, "bs_target_corridor_mask", corridor.shape[:2], device)
        anchor_mask = self._mask_tensor(batch, "anchor_corridor_mask", anchor_corridor.shape[:3], device)
        point = point.to(dtype=dtype)
        target_patch = target_patch.to(dtype=dtype)
        bs_patch = bs_patch.to(dtype=dtype)
        corridor = corridor.to(dtype=dtype)
        anchor_corridor = anchor_corridor.to(dtype=dtype)
        flat_corridor = anchor_corridor.reshape(batch_size * anchor_count, anchor_corridor.shape[2], 15)
        flat_mask = anchor_mask.reshape(batch_size * anchor_count, anchor_corridor.shape[2])
        anchor_features = self.corridor_encoder(flat_corridor, flat_mask).reshape(batch_size, anchor_count, -1)
        bs_center = bs_patch[:, :, bs_patch.shape[2] // 2, bs_patch.shape[3] // 2]
        target_features = torch.cat((
            self.point_encoder(point), self.point_encoder(bs_center),
            self.patch_encoder(target_patch), self.patch_encoder(bs_patch),
            self.corridor_encoder(corridor, corridor_mask),
        ), dim=-1)
        return target_features, anchor_features

    def forward(self, batch: dict[str, torch.Tensor]) -> MixerOutput:
        latents, distances, anchor_mask, source_ids, pair_features, positions = self._validate_core(batch)
        batch_size, anchor_count, _ = latents.shape
        real_dtype = latents.real.dtype
        raw_distances = distances
        distances = distances.to(dtype=real_dtype)
        pair_features = pair_features.to(dtype=real_dtype)
        positions = tuple(value.to(dtype=real_dtype) for value in positions)
        # Invalid padded rows are zeroed before any feature transform, allowing
        # arbitrary sentinel values (including infinities) outside the mask.
        safe_distances = torch.where(anchor_mask, distances, torch.zeros_like(distances))
        safe_pairs = torch.where(anchor_mask.unsqueeze(-1), pair_features, torch.zeros_like(pair_features))
        safe_latents = torch.where(anchor_mask.unsqueeze(-1), latents, torch.zeros_like(latents))
        target_geometry, anchor_geometry = self._geometry_features(batch, batch_size, anchor_count, latents.device, real_dtype)
        distance_encoding = self.distance_features(safe_distances.unsqueeze(-1))
        anchor_base = self.anchor_encoder(torch.cat((safe_pairs, distance_encoding, anchor_geometry), dim=-1))
        target_encoding = torch.cat((
            self.position_features(positions[0], positions[1]),
            self.position_features(positions[2], positions[3]),
            target_geometry,
        ), dim=-1)
        target_condition = self.target_encoder(target_encoding)

        summaries = self.summary_encoder(safe_latents)
        query = self.target_projection(target_condition)[:, None, :] + self.group_embedding[None, :, :]
        key = self.anchor_projection(anchor_base)[:, :, None, :] + self.summary_projection(summaries)
        logits = torch.einsum("bgd,bkgd->bkg", query, key) / math.sqrt(self.config.group_dim)
        logits = logits - F.softplus(self.distance_beta) * torch.log(safe_distances + 1e-3)[:, :, None]
        logits = logits.masked_fill(~anchor_mask[:, :, None], float("-inf"))
        weights = torch.softmax(logits, dim=1)
        weights = torch.where(self.summary_encoder.group_mask[None, None, :], weights, torch.zeros_like(weights))

        group_ids = self.group_ids.to(device=latents.device)
        coefficient_weights = weights.index_select(2, group_ids)
        mixed = (safe_latents * coefficient_weights).sum(dim=1)
        minimum_distances = raw_distances.masked_fill(~anchor_mask, float("inf")).amin(dim=1, keepdim=True)
        nearest_ties = anchor_mask & raw_distances.eq(minimum_distances)
        source_sentinel = torch.iinfo(source_ids.dtype).max
        nearest_indices = source_ids.masked_fill(~nearest_ties, source_sentinel).argmin(dim=1)
        nearest = latents[torch.arange(batch_size, device=latents.device), nearest_indices]

        # torch.complex does not accept BF16 components.  Keep encoder/head
        # autocast active, then promote only the real-valued correction heads
        # at the complex boundary to match the complex latent/basis precision.
        alpha = torch.tanh(self.alpha_head(target_condition)).to(dtype=real_dtype)
        gain_real = torch.tanh(self.gain_real_head(target_condition)).to(dtype=real_dtype)
        gain_imag = torch.tanh(self.gain_imag_head(target_condition)).to(dtype=real_dtype)
        complex_gain = 0.1 * torch.complex(
            gain_real, gain_imag
        )
        # A raw low-rank head can turn one high-gradient channel into an
        # unbounded complex correction.  Bounded coefficients preserve the
        # nearest-anchor initialization and keep the residual a correction,
        # rather than an unconstrained replacement for the channel.
        residual_real = (0.1 * torch.tanh(self.residual_real_head(target_condition))).to(dtype=real_dtype)
        residual_imag = (0.1 * torch.tanh(self.residual_imag_head(target_condition))).to(dtype=real_dtype)
        residual_coeff = torch.complex(residual_real, residual_imag)
        basis = self.complex_basis / (1.0 + self.complex_basis.abs())
        low_rank_residual = residual_coeff @ basis.transpose(0, 1)
        prediction = nearest + alpha.index_select(1, group_ids) * (mixed - nearest)
        prediction = prediction + complex_gain.index_select(1, group_ids) * mixed + low_rank_residual
        return MixerOutput(prediction, nearest, weights, alpha, complex_gain, low_rank_residual)
