"""Small, bounded PyTorch encoders shared by the anchor mixer."""

from __future__ import annotations

from numbers import Integral

import torch
from torch import nn


def _group_count(channels: int) -> int:
    """Choose a small GroupNorm group count that divides ``channels``."""

    for groups in range(min(8, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _require_float(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor) or not tensor.dtype.is_floating_point:
        raise TypeError(f"{name} must be a floating-point torch.Tensor")


class FourierFeatures(nn.Module):
    """Fixed power-of-two Fourier features for raw and standardized positions.

    Passing both tensors returns their concatenated encodings.  Passing only a
    raw position is supported for callers that have no separate standardized
    coordinate; its output has ``single_output_dim`` features.
    """

    def __init__(
        self, input_dim: int, num_frequencies: int = 6, include_input: bool = True
    ) -> None:
        super().__init__()
        if not isinstance(input_dim, Integral) or isinstance(input_dim, bool) or input_dim < 1:
            raise ValueError("input_dim must be a positive integer")
        if (
            not isinstance(num_frequencies, Integral)
            or isinstance(num_frequencies, bool)
            or num_frequencies < 1
        ):
            raise ValueError("num_frequencies must be a positive integer")
        self.input_dim = int(input_dim)
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)
        frequencies = torch.pow(2.0, torch.arange(self.num_frequencies, dtype=torch.float32))
        self.register_buffer("frequencies", frequencies, persistent=True)
        self.single_output_dim = self.input_dim * (
            2 * self.num_frequencies + int(self.include_input)
        )
        self.output_dim = 2 * self.single_output_dim

    def _encode(self, positions: torch.Tensor, name: str) -> torch.Tensor:
        _require_float(positions, name)
        if positions.ndim < 2 or positions.shape[-1] != self.input_dim:
            raise ValueError(
                f"{name} must have shape (..., {self.input_dim}), got {tuple(positions.shape)}"
            )
        frequencies = self.frequencies.to(dtype=positions.dtype, device=positions.device)
        angles = positions.unsqueeze(-1) * frequencies * torch.pi
        encoded = (torch.sin(angles), torch.cos(angles))
        pieces = [positions] if self.include_input else []
        pieces.extend(feature.flatten(start_dim=-2) for feature in encoded)
        return torch.cat(pieces, dim=-1)

    def forward(
        self, raw_positions: torch.Tensor, standardized_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode positions with fixed frequencies ``1, 2, 4, ...``."""

        raw = self._encode(raw_positions, "raw_positions")
        if standardized_positions is None:
            return raw
        if standardized_positions.shape != raw_positions.shape:
            raise ValueError(
                "standardized_positions must have the same shape as raw_positions, "
                f"got {tuple(standardized_positions.shape)} and {tuple(raw_positions.shape)}"
            )
        return torch.cat((raw, self._encode(standardized_positions, "standardized_positions")), dim=-1)


class PatchEncoder(nn.Module):
    """Three stride-two Conv2d blocks followed by global mean pooling."""

    def __init__(self, in_channels: int = 13, output_dim: int = 64, hidden_dim: int = 64) -> None:
        super().__init__()
        for name, value in (("in_channels", in_channels), ("output_dim", output_dim), ("hidden_dim", hidden_dim)):
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.in_channels = int(in_channels)
        self.output_dim = int(output_dim)
        hidden_dim = int(hidden_dim)
        norm_groups = _group_count(hidden_dim)
        layers: list[nn.Module] = []
        current_channels = self.in_channels
        for _ in range(3):
            layers.extend(
                (
                    nn.Conv2d(current_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(norm_groups, hidden_dim),
                    nn.SiLU(),
                )
            )
            current_channels = hidden_dim
        self.blocks = nn.Sequential(*layers)
        self.projection = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        _require_float(patches, "patches")
        if patches.ndim != 4 or patches.shape[1] != self.in_channels:
            raise ValueError(
                f"patches must have shape (B, {self.in_channels}, H, W), got {tuple(patches.shape)}"
            )
        return self.projection(self.blocks(patches).mean(dim=(-2, -1)))


class CorridorEncoder(nn.Module):
    """Point MLP and Conv1d encoder with padding-safe mean/max pooling."""

    def __init__(self, input_dim: int = 15, output_dim: int = 64, hidden_dim: int = 64) -> None:
        super().__init__()
        for name, value in (("input_dim", input_dim), ("output_dim", output_dim), ("hidden_dim", hidden_dim)):
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        hidden_dim = int(hidden_dim)
        norm_groups = _group_count(hidden_dim)
        self.point_mlp = nn.Sequential(nn.Linear(self.input_dim, hidden_dim), nn.SiLU())
        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(norm_groups, hidden_dim),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(norm_groups, hidden_dim),
                    nn.SiLU(),
                ),
            ]
        )
        self.projection = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, self.output_dim)
        )

    def forward(self, corridor: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        _require_float(corridor, "corridor")
        if corridor.ndim != 3 or corridor.shape[-1] != self.input_dim:
            raise ValueError(
                f"corridor must have shape (B, L, {self.input_dim}), got {tuple(corridor.shape)}"
            )
        batch_size, length, _ = corridor.shape
        if padding_mask is None:
            padding_mask = torch.ones(batch_size, length, dtype=torch.bool, device=corridor.device)
        elif (
            not isinstance(padding_mask, torch.Tensor)
            or padding_mask.dtype != torch.bool
            or padding_mask.shape != (batch_size, length)
            or padding_mask.device != corridor.device
        ):
            raise ValueError("padding_mask must be a bool tensor on the corridor device with shape (B, L)")

        mask = padding_mask.unsqueeze(-1)
        points = self.point_mlp(corridor.masked_fill(~mask, 0.0)).masked_fill(~mask, 0.0)
        sequence = points.transpose(1, 2)
        sequence_mask = padding_mask.unsqueeze(1)
        for block in self.conv_blocks:
            sequence = block(sequence).masked_fill(~sequence_mask, 0.0)
        features = sequence.transpose(1, 2)
        count = padding_mask.sum(dim=1, keepdim=True)
        mean = features.sum(dim=1) / count.clamp_min(1).to(features.dtype)
        maximum = features.masked_fill(~mask, float("-inf")).max(dim=1).values
        maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
        output = self.projection(torch.cat((mean, maximum), dim=-1))
        return output * (count > 0).to(output.dtype)


class GroupSummaryEncoder(nn.Module):
    """Aggregate complex coefficients into six phase-aware statistics per group."""

    def __init__(self, group_ids: torch.Tensor, group_count: int) -> None:
        super().__init__()
        if not isinstance(group_count, Integral) or isinstance(group_count, bool) or group_count < 1:
            raise ValueError("group_count must be a positive integer")
        group_ids = torch.as_tensor(group_ids)
        if group_ids.ndim != 1 or group_ids.numel() == 0:
            raise ValueError("group_ids must be a non-empty one-dimensional tensor")
        if group_ids.dtype == torch.bool or group_ids.dtype.is_floating_point or torch.is_complex(group_ids):
            raise TypeError("group_ids must have an integer dtype")
        if (group_ids < 0).any() or (group_ids >= group_count).any():
            raise ValueError("group_ids must lie in [0, group_count)")
        ids = group_ids.to(dtype=torch.long)
        self.group_count = int(group_count)
        self.register_buffer("group_ids", ids, persistent=True)
        self.register_buffer(
            "group_mask", torch.bincount(ids, minlength=self.group_count).gt(0), persistent=True
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if not isinstance(latents, torch.Tensor) or not torch.is_complex(latents):
            raise TypeError("latents must be a complex torch.Tensor")
        if latents.ndim != 3 or latents.shape[-1] != self.group_ids.numel():
            raise ValueError(
                "latents must have shape (B, K, L) with L matching group_ids; "
                f"got {tuple(latents.shape)} and L={self.group_ids.numel()}"
            )
        batch_size, anchor_count, latent_count = latents.shape
        values = latents.reshape(batch_size * anchor_count, latent_count)
        ids = self.group_ids.to(device=latents.device)
        groups = self.group_count
        counts = torch.bincount(ids, minlength=groups).to(device=latents.device, dtype=latents.real.dtype)

        def summed(source: torch.Tensor) -> torch.Tensor:
            result = source.new_zeros(source.shape[0], groups)
            return result.index_add(1, ids, source)

        magnitude = values.abs()
        energy = summed(magnitude.square())
        magnitude_sum = summed(magnitude)
        real_mean = summed(values.real) / counts.clamp_min(1)
        imaginary_mean = summed(values.imag) / counts.clamp_min(1)
        magnitude_mean = magnitude_sum / counts.clamp_min(1)
        max_magnitude = magnitude.new_zeros(magnitude.shape[0], groups).scatter_reduce(
            1,
            ids.expand(magnitude.shape[0], -1),
            magnitude,
            reduce="amax",
            include_self=True,
        )
        unit_phase = torch.where(
            magnitude > 0,
            values / magnitude.clamp_min(torch.finfo(magnitude.dtype).eps),
            torch.zeros_like(values),
        )
        coherence = torch.abs(summed(unit_phase.real) + 1j * summed(unit_phase.imag))
        coherence = coherence / counts.clamp_min(1)
        summary = torch.stack(
            (torch.log1p(energy), magnitude_mean, max_magnitude, real_mean, imaginary_mean, coherence),
            dim=-1,
        )
        summary = summary * self.group_mask.to(device=summary.device, dtype=summary.dtype).view(1, groups, 1)
        return summary.reshape(batch_size, anchor_count, groups, 6)
