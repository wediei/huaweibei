"""End-to-end complex Gaussian path field.

This module is intentionally independent from retained O4.1 predictors.  The
official PLY is used only to initialise a bounded trainable field; complete
complex channels are produced by explicit path parameters and coherent RF
rendering.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from ..config import RoundConfig
from ..geometry import PlyPointCloud

MapMode = Literal["real", "zero", "shuffle"]


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class GaussianFieldConfig:
    initial_count: int = 128
    max_count: int = 256
    min_count: int = 32
    material_dim: int = 16
    center_offset_limit: float = 8.0
    min_scale: float = 0.05
    max_scale: float = 12.0
    densify_count: int = 8
    prune_count: int = 8
    densify_threshold: float = 0.02
    prune_threshold: float = 1e-4
    statistics_decay: float = 0.95

    def validate(self) -> None:
        for name in (
            "initial_count",
            "max_count",
            "min_count",
            "material_dim",
            "densify_count",
            "prune_count",
        ):
            _require_positive_int(name, getattr(self, name))
        if not self.min_count <= self.initial_count <= self.max_count:
            raise ValueError("field counts must satisfy min <= initial <= max")
        if not 0.0 < self.min_scale < self.max_scale:
            raise ValueError("field scales must satisfy 0 < min < max")
        if self.center_offset_limit <= 0.0:
            raise ValueError("center_offset_limit must be positive")
        if not 0.0 <= self.statistics_decay < 1.0:
            raise ValueError("statistics_decay must be in [0,1)")


@dataclass(frozen=True)
class PathNetworkConfig:
    hidden_dim: int = 128
    query_dim: int = 64
    modes_per_gaussian: int = 4
    selected_gaussians: int = 32
    polarization_rank: int = 1
    path_type_dim: int = 8
    fourier_bands: int = 6
    max_delay_bins: float = 191.0
    angle_residual_limit: float = 0.35
    score_distance_weight: float = 0.25

    def validate(self) -> None:
        for name in (
            "hidden_dim",
            "query_dim",
            "modes_per_gaussian",
            "selected_gaussians",
            "polarization_rank",
            "path_type_dim",
            "fourier_bands",
        ):
            _require_positive_int(name, getattr(self, name))
        if self.max_delay_bins <= 0.0:
            raise ValueError("max_delay_bins must be positive")
        if not 0.0 < self.angle_residual_limit <= 2.0:
            raise ValueError("angle_residual_limit must be in (0,2]")


@dataclass(frozen=True)
class RendererConfig:
    path_chunk_size: int = 16

    def validate(self) -> None:
        _require_positive_int("path_chunk_size", self.path_chunk_size)


@dataclass(frozen=True)
class E2ECGPFConfig:
    field: GaussianFieldConfig = dataclasses.field(default_factory=GaussianFieldConfig)
    paths: PathNetworkConfig = dataclasses.field(default_factory=PathNetworkConfig)
    renderer: RendererConfig = dataclasses.field(default_factory=RendererConfig)
    map_mode: MapMode = "real"
    antenna_order: tuple[str, str, str] = ("H", "V", "P")
    seed: int = 42

    def validate(self) -> None:
        self.field.validate()
        self.paths.validate()
        self.renderer.validate()
        if self.map_mode not in {"real", "zero", "shuffle"}:
            raise ValueError("map_mode must be real, zero, or shuffle")
        if len(self.antenna_order) != 3 or set(self.antenna_order) != {"H", "V", "P"}:
            raise ValueError("antenna_order must be a permutation of H,V,P")
        if self.paths.selected_gaussians > self.field.max_count:
            raise ValueError("selected_gaussians cannot exceed max_count")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "E2ECGPFConfig":
        result = cls(
            field=GaussianFieldConfig(**values.get("field", {})),
            paths=PathNetworkConfig(**values.get("paths", {})),
            renderer=RendererConfig(**values.get("renderer", {})),
            map_mode=values.get("map_mode", "real"),
            antenna_order=tuple(values.get("antenna_order", ("H", "V", "P"))),
            seed=int(values.get("seed", 42)),
        )
        result.validate()
        return result


@dataclass
class GaussianState:
    centers: torch.Tensor
    scales: torch.Tensor
    normals: torch.Tensor
    opacity: torch.Tensor
    material: torch.Tensor
    active_mask: torch.Tensor
    slot_indices: torch.Tensor


@dataclass
class PathModes:
    delay_bins: torch.Tensor
    aod: torch.Tensor
    aoa: torch.Tensor
    complex_gain: torch.Tensor
    polarization: torch.Tensor
    existence: torch.Tensor
    reliability: torch.Tensor
    width: torch.Tensor
    path_type: torch.Tensor
    gate: torch.Tensor
    gaussian_indices: torch.Tensor
    selection_weight: torch.Tensor

    @property
    def path_count(self) -> int:
        return int(self.delay_bins.shape[1])


@dataclass(frozen=True)
class StructureEdit:
    active_before: int
    active_after: int
    densified: tuple[int, ...]
    pruned: tuple[int, ...]
    rolled_back: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _normalise(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return vector / vector.square().sum(dim=-1, keepdim=True).clamp_min(eps).sqrt()


def _rotate_by_quaternion(vector: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = _normalise(quaternion)
    xyz = quaternion[..., :3]
    real = quaternion[..., 3:4]
    first = torch.cross(xyz, vector, dim=-1)
    second = torch.cross(xyz, first, dim=-1)
    return vector + 2.0 * (real * first + second)


class TrainableGaussianField(nn.Module):
    """Bounded trainable Gaussian slots with deterministic structural control."""

    def __init__(
        self,
        initial_centers: torch.Tensor,
        initial_normals: torch.Tensor,
        config: GaussianFieldConfig,
        *,
        map_mode: MapMode = "real",
        seed: int = 42,
    ) -> None:
        super().__init__()
        config.validate()
        if initial_centers.shape != (config.initial_count, 3):
            raise ValueError("initial_centers has the wrong shape")
        if initial_normals.shape != initial_centers.shape:
            raise ValueError("initial_normals must match initial_centers")
        if map_mode not in {"real", "zero", "shuffle"}:
            raise ValueError("invalid map mode")

        centers = initial_centers.detach().to(dtype=torch.float32, device="cpu").clone()
        normals = _normalise(
            initial_normals.detach().to(dtype=torch.float32, device="cpu").clone()
        )
        scene_center = centers.mean(dim=0)
        if map_mode == "zero":
            centers[:] = scene_center
            normals.zero_()
            normals[:, 2] = 1.0
        elif map_mode == "shuffle":
            generator = torch.Generator(device="cpu").manual_seed(seed + 1701)
            permutation = torch.randperm(config.initial_count, generator=generator)
            centers = centers[permutation]
            normals = normals[permutation]

        max_count = config.max_count
        center_prior = scene_center.expand(max_count, 3).clone()
        normal_prior = torch.zeros(max_count, 3)
        normal_prior[:, 2] = 1.0
        center_prior[: config.initial_count] = centers
        normal_prior[: config.initial_count] = normals
        if config.initial_count < max_count:
            repeat = torch.arange(
                max_count - config.initial_count, dtype=torch.long
            ) % config.initial_count
            center_prior[config.initial_count :] = centers[repeat]
            normal_prior[config.initial_count :] = normals[repeat]

        active = torch.zeros(max_count, dtype=torch.bool)
        active[: config.initial_count] = True
        generator = torch.Generator(device="cpu").manual_seed(seed)
        material = torch.randn(
            max_count, config.material_dim, generator=generator
        ) * 0.02

        self.config = config
        self.map_mode = map_mode
        self.seed = int(seed)
        self.register_buffer("center_prior", center_prior)
        self.register_buffer("normal_prior", normal_prior)
        self.register_buffer("scene_center", scene_center)
        self.register_buffer("active_mask", active)
        self.register_buffer("shuffle_permutation", torch.randperm(max_count, generator=generator))
        self.register_buffer("ema_activation", torch.zeros(max_count))
        self.register_buffer("ema_contribution", torch.zeros(max_count))
        self.register_buffer("ema_residual", torch.zeros(max_count))
        self.register_buffer("ema_gradient", torch.zeros(max_count))
        self.register_buffer("age", torch.zeros(max_count, dtype=torch.long))
        self.register_buffer("structure_version", torch.zeros((), dtype=torch.long))

        self.center_delta = nn.Parameter(torch.zeros(max_count, 3))
        self.log_scale = nn.Parameter(torch.zeros(max_count, 3))
        self.orientation = nn.Parameter(
            torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(max_count, 4).clone()
        )
        self.opacity_logit = nn.Parameter(torch.full((max_count, 1), -0.5))
        self.material_code = nn.Parameter(material)

    @classmethod
    def from_ply(
        cls,
        path: str | Path,
        config: GaussianFieldConfig,
        *,
        map_mode: MapMode = "real",
        seed: int = 42,
    ) -> "TrainableGaussianField":
        cloud = PlyPointCloud.open(path)
        if cloud.vertex_count < config.initial_count:
            raise ValueError(
                f"PLY has {cloud.vertex_count} vertices, fewer than initial_count "
                f"{config.initial_count}"
            )
        rng = np.random.default_rng(seed)
        indices = np.sort(
            rng.choice(cloud.vertex_count, size=config.initial_count, replace=False)
        )
        records = cloud.vertices[indices]
        centers = np.column_stack((records["x"], records["y"], records["z"])).astype(
            np.float32
        )
        if not np.isfinite(centers).all():
            raise ValueError("sampled PLY centers contain non-finite values")
        normals = np.column_stack(
            (records["nx"], records["ny"], records["nz"])
        ).astype(np.float32)
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        invalid = (~np.isfinite(norm).all(axis=1)) | (norm[:, 0] < 1e-8)
        normals = normals / np.maximum(norm, 1e-8)
        normals[invalid] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        return cls(
            torch.from_numpy(centers),
            torch.from_numpy(normals),
            config,
            map_mode=map_mode,
            seed=seed,
        )

    @property
    def active_count(self) -> int:
        return int(self.active_mask.sum().item())

    def state(self, view: MapMode | None = None) -> GaussianState:
        # ``map_mode`` changes only the initial prior for separately trained
        # equal-capacity controls. Explicit views are causal ablations of an
        # already trained real-map model.
        view = "real" if view is None else view
        if view not in {"real", "zero", "shuffle"}:
            raise ValueError("invalid field view")
        offset = torch.tanh(self.center_delta) * self.config.center_offset_limit
        centers = self.center_prior + offset
        span = self.config.max_scale - self.config.min_scale
        scales = self.config.min_scale + torch.sigmoid(self.log_scale) * span
        normals = _normalise(
            _rotate_by_quaternion(self.normal_prior, self.orientation)
        )
        opacity = torch.sigmoid(self.opacity_logit).squeeze(-1)
        material = torch.tanh(self.material_code)
        active = self.active_mask
        slot_indices = torch.arange(
            self.config.max_count, device=active.device, dtype=torch.long
        )

        if view == "zero":
            centers = self.scene_center.expand_as(centers)
            scales = torch.ones_like(scales) * scales.detach().mean()
            normals = torch.zeros_like(normals)
            material = torch.zeros_like(material)
        elif view == "shuffle":
            permutation = self.shuffle_permutation
            centers = centers[permutation]
            scales = scales[permutation]
            normals = normals[permutation]
            opacity = opacity[permutation]
            material = material[permutation]
            active = active[permutation]
            slot_indices = slot_indices[permutation]
        return GaussianState(
            centers, scales, normals, opacity, material, active, slot_indices
        )

    def regularization(self) -> dict[str, torch.Tensor]:
        state = self.state("real")
        active = self.active_mask
        offset = torch.tanh(self.center_delta[active])
        material = self.material_code[active]
        orientation_norm = self.orientation[active].square().sum(dim=-1).sqrt()
        if self.active_count > 1:
            distance = torch.cdist(state.centers[active], state.centers[active])
            distance = distance + torch.eye(
                self.active_count, device=distance.device, dtype=distance.dtype
            ) * 1e6
            neighbor_count = min(4, self.active_count - 1)
            neighbor_distance, neighbor_index = torch.topk(
                distance, k=neighbor_count, dim=1, largest=False
            )
            neighbor_material = material[neighbor_index]
            smooth_weight = torch.exp(
                -neighbor_distance / state.scales[active].mean().clamp_min(1e-3)
            )
            field_smooth = (
                (material.unsqueeze(1) - neighbor_material)
                .square()
                .mean(dim=-1)
                .mul(smooth_weight)
                .mean()
            )
        else:
            field_smooth = material.new_zeros(())
        return {
            "field_center": offset.square().mean(),
            "field_scale": (
                torch.log(state.scales[active].clamp_min(1e-6)).square().mean()
            ),
            "field_material": material.square().mean(),
            "field_smooth": field_smooth,
            "field_opacity": state.opacity[active].mean(),
            "field_orientation": (orientation_norm - 1.0).square().mean(),
        }

    @torch.no_grad()
    def record_statistics(
        self,
        gaussian_indices: torch.Tensor,
        gate: torch.Tensor,
        residual: float | torch.Tensor,
    ) -> None:
        indices = gaussian_indices.detach().reshape(-1).to(self.active_mask.device)
        weights = gate.detach().abs().reshape(-1).to(self.active_mask.device)
        if indices.numel() != weights.numel():
            raise ValueError("indices and gates must have the same element count")
        count = torch.zeros_like(self.ema_activation)
        activation = torch.zeros_like(self.ema_activation)
        count.scatter_add_(0, indices, torch.ones_like(weights))
        activation.scatter_add_(0, indices, weights)
        mean_activation = activation / count.clamp_min(1.0)
        residual_value = float(
            residual.detach().cpu() if isinstance(residual, torch.Tensor) else residual
        )
        gradient = torch.zeros_like(self.ema_gradient)
        if self.center_delta.grad is not None:
            gradient = self.center_delta.grad.detach().norm(dim=-1)
        decay = self.config.statistics_decay
        touched = count > 0
        self.ema_activation[touched] = (
            decay * self.ema_activation[touched]
            + (1.0 - decay) * mean_activation[touched]
        )
        self.ema_contribution[touched] = (
            decay * self.ema_contribution[touched]
            + (1.0 - decay) * activation[touched]
        )
        self.ema_residual[touched] = (
            decay * self.ema_residual[touched]
            + (1.0 - decay) * residual_value
        )
        self.ema_gradient.mul_(decay).add_(gradient, alpha=1.0 - decay)
        self.age[self.active_mask] += 1

    @torch.no_grad()
    def structure_snapshot(self) -> dict[str, torch.Tensor]:
        names = (
            "center_delta",
            "log_scale",
            "orientation",
            "opacity_logit",
            "material_code",
            "active_mask",
            "ema_activation",
            "ema_contribution",
            "ema_residual",
            "ema_gradient",
            "age",
            "structure_version",
        )
        return {name: getattr(self, name).detach().cpu().clone() for name in names}

    @torch.no_grad()
    def restore_structure(self, snapshot: dict[str, torch.Tensor]) -> None:
        for name, value in snapshot.items():
            destination = getattr(self, name)
            if destination.shape != value.shape:
                raise ValueError(f"snapshot shape mismatch for {name}")
            destination.copy_(value.to(destination.device))

    @torch.no_grad()
    def densify_prune(self, step: int) -> StructureEdit:
        """Activate/deactivate fixed slots without changing optimizer shapes."""

        before = self.active_count
        active_indices = torch.nonzero(self.active_mask, as_tuple=False).flatten()
        inactive_indices = torch.nonzero(~self.active_mask, as_tuple=False).flatten()
        minimum_age = 2
        prune_score = self.ema_activation + self.ema_contribution
        eligible_prune = active_indices[
            (self.age[active_indices] >= minimum_age)
            & (prune_score[active_indices] < self.config.prune_threshold)
        ]
        maximum_prune = min(
            self.config.prune_count,
            max(0, before - self.config.min_count),
            int(eligible_prune.numel()),
        )
        if maximum_prune:
            order = torch.argsort(prune_score[eligible_prune])
            pruned = eligible_prune[order[:maximum_prune]]
            self.active_mask[pruned] = False
        else:
            pruned = active_indices.new_empty((0,))

        active_indices = torch.nonzero(self.active_mask, as_tuple=False).flatten()
        inactive_indices = torch.nonzero(~self.active_mask, as_tuple=False).flatten()
        densify_score = (
            self.ema_gradient
            + self.ema_residual
            + self.ema_contribution
        )
        eligible_parent = active_indices[
            densify_score[active_indices] >= self.config.densify_threshold
        ]
        maximum_densify = min(
            self.config.densify_count,
            self.config.max_count - self.active_count,
            int(inactive_indices.numel()),
            int(eligible_parent.numel()),
        )
        densified: list[int] = []
        if maximum_densify:
            parent_order = torch.argsort(
                densify_score[eligible_parent], descending=True
            )
            parents = eligible_parent[parent_order[:maximum_densify]]
            children = inactive_indices[:maximum_densify]
            generator = torch.Generator(device="cpu").manual_seed(self.seed + int(step))
            jitter = torch.randn(
                maximum_densify, 3, generator=generator
            ).to(self.center_delta.device) * 0.02
            for child, parent, child_jitter in zip(children, parents, jitter):
                child_index, parent_index = int(child), int(parent)
                self.center_delta[child_index].copy_(
                    self.center_delta[parent_index] + child_jitter
                )
                self.log_scale[child_index].copy_(
                    self.log_scale[parent_index] - math.log(1.25)
                )
                self.orientation[child_index].copy_(self.orientation[parent_index])
                self.opacity_logit[child_index].copy_(
                    self.opacity_logit[parent_index] - 0.5
                )
                self.material_code[child_index].copy_(
                    self.material_code[parent_index]
                )
                self.active_mask[child_index] = True
                self.age[child_index] = 0
                densified.append(child_index)
        if maximum_prune or maximum_densify:
            self.structure_version.add_(1)
        return StructureEdit(
            active_before=before,
            active_after=self.active_count,
            densified=tuple(densified),
            pruned=tuple(int(value) for value in pruned.tolist()),
        )


def _fourier_encode(values: torch.Tensor, bands: int) -> torch.Tensor:
    frequencies = (2.0 ** torch.arange(
        bands, device=values.device, dtype=values.dtype
    )) * math.pi
    angles = values.unsqueeze(-1) * frequencies
    return torch.cat(
        (values, torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)),
        dim=-1,
    )


def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = indices.shape[0]
    expanded = values.unsqueeze(0).expand(batch, *values.shape)
    gather_index = indices.reshape(batch, indices.shape[1], *([1] * (values.ndim - 1)))
    gather_index = gather_index.expand(batch, indices.shape[1], *values.shape[1:])
    return torch.gather(expanded, 1, gather_index)


class TargetConditionedPathNetwork(nn.Module):
    """Generate multiple soft RF path modes for each selected Gaussian."""

    def __init__(
        self,
        field_config: GaussianFieldConfig,
        config: PathNetworkConfig,
        *,
        m_p: int,
        n_p: int,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.m_p = int(m_p)
        self.n_p = int(n_p)
        query_input = 3 * (1 + 2 * config.fourier_bands)
        field_input = 3 + 3 + 3 + 1 + field_config.material_dim
        geometry_input = (
            query_input
            + field_input
            + 3 * 4
            + 4
        )
        self.query = nn.Sequential(
            nn.Linear(query_input, config.query_dim),
            nn.SiLU(),
            nn.Linear(config.query_dim, config.query_dim),
        )
        self.key = nn.Sequential(
            nn.Linear(field_input, config.query_dim),
            nn.SiLU(),
            nn.Linear(config.query_dim, config.query_dim),
        )
        self.trunk = nn.Sequential(
            nn.Linear(geometry_input, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )
        per_mode = (
            1
            + 2
            + 2
            + 2
            + 2 * m_p * config.polarization_rank
            + 2 * n_p * config.polarization_rank
            + 3
            + config.path_type_dim
        )
        self.head = nn.Linear(config.hidden_dim, per_mode)
        self.per_mode = per_mode
        self.mode_embedding = nn.Parameter(
            torch.randn(config.modes_per_gaussian, config.hidden_dim) * 0.01
        )

    def forward(
        self,
        targets: torch.Tensor,
        bs_position: torch.Tensor,
        state: GaussianState,
        *,
        position_center: torch.Tensor,
        position_scale: torch.Tensor,
    ) -> PathModes:
        if targets.ndim != 2 or targets.shape[1] != 3:
            raise ValueError("targets must have shape [B,3]")
        batch = targets.shape[0]
        center_norm = (state.centers - position_center) / position_scale
        target_norm = (targets - position_center) / position_scale
        scale_norm = state.scales / position_scale
        field_features = torch.cat(
            (
                center_norm,
                scale_norm,
                state.normals,
                state.opacity.unsqueeze(-1),
                state.material,
            ),
            dim=-1,
        )
        query_features = _fourier_encode(target_norm, self.config.fourier_bands)
        query = self.query(query_features)
        key = self.key(field_features)
        score = torch.einsum("bd,gd->bg", query, key) / math.sqrt(query.shape[-1])
        distance = torch.linalg.vector_norm(
            center_norm.unsqueeze(0) - target_norm.unsqueeze(1), dim=-1
        )
        score = (
            score
            - self.config.score_distance_weight * distance
            + torch.log(state.opacity.clamp_min(1e-6)).unsqueeze(0)
        )
        score = score.masked_fill(~state.active_mask.unsqueeze(0), -torch.inf)
        selected_count = min(
            self.config.selected_gaussians, int(state.active_mask.sum().item())
        )
        if selected_count < 1:
            raise RuntimeError("Gaussian field has no active slots")
        selected_score, indices = torch.topk(score, k=selected_count, dim=1)
        selection_weight = torch.softmax(selected_score, dim=1)
        selected_slot_indices = _gather(
            state.slot_indices.unsqueeze(-1), indices
        ).squeeze(-1)

        centers = _gather(state.centers, indices)
        scales = _gather(state.scales, indices)
        normals = _gather(state.normals, indices)
        opacity = _gather(state.opacity.unsqueeze(-1), indices).squeeze(-1)
        material = _gather(state.material, indices)
        selected_field = _gather(field_features, indices)

        bs = bs_position.reshape(1, 1, 3).expand(batch, selected_count, 3)
        target = targets.unsqueeze(1).expand(batch, selected_count, 3)
        bs_to_g = centers - bs
        g_to_target = target - centers
        direct = target - bs
        bs_distance = torch.linalg.vector_norm(bs_to_g, dim=-1, keepdim=True)
        target_distance = torch.linalg.vector_norm(
            g_to_target, dim=-1, keepdim=True
        )
        direct_distance = torch.linalg.vector_norm(direct, dim=-1, keepdim=True)
        geometry = torch.cat(
            (
                query_features.unsqueeze(1).expand(-1, selected_count, -1),
                selected_field,
                _normalise(bs_to_g),
                _normalise(g_to_target),
                _normalise(direct),
                normals,
                bs_distance / position_scale,
                target_distance / position_scale,
                direct_distance / position_scale,
                scales.mean(dim=-1, keepdim=True) / position_scale,
            ),
            dim=-1,
        )
        hidden = self.trunk(geometry)
        hidden = hidden.unsqueeze(2) + self.mode_embedding.reshape(
            1, 1, self.config.modes_per_gaussian, -1
        )
        raw = self.head(hidden).reshape(
            batch, selected_count, self.config.modes_per_gaussian, -1
        )

        cursor = 0

        def take(size: int) -> torch.Tensor:
            nonlocal cursor
            value = raw[..., cursor : cursor + size]
            cursor += size
            return value

        delay_raw = take(1).squeeze(-1)
        aod_delta = take(2)
        aoa_delta = take(2)
        gain_raw = take(2)
        left_raw = take(2 * self.m_p * self.config.polarization_rank)
        right_raw = take(2 * self.n_p * self.config.polarization_rank)
        existence = torch.sigmoid(take(1).squeeze(-1))
        reliability = torch.sigmoid(take(1).squeeze(-1))
        width = 0.01 + 0.5 * torch.sigmoid(take(1).squeeze(-1))
        path_type = torch.tanh(take(self.config.path_type_dim))
        if cursor != self.per_mode:
            raise AssertionError("path head packing mismatch")

        geometric_delay = (bs_distance + target_distance).squeeze(-1)
        geometric_delay = geometric_delay / position_scale
        geometric_delay = geometric_delay.unsqueeze(-1)
        delay_fraction = torch.sigmoid(delay_raw + 0.1 * geometric_delay)
        delay_bins = delay_fraction * self.config.max_delay_bins

        aod_base_3d = _normalise(bs_to_g)
        aoa_base_3d = _normalise(g_to_target)
        aod_base = torch.stack((aod_base_3d[..., 0], aod_base_3d[..., 2]), dim=-1)
        aoa_base = torch.stack((aoa_base_3d[..., 0], aoa_base_3d[..., 2]), dim=-1)
        limit = self.config.angle_residual_limit
        aod = torch.clamp(aod_base.unsqueeze(2) + limit * torch.tanh(aod_delta), -1.0, 1.0)
        aoa = torch.clamp(aoa_base.unsqueeze(2) + limit * torch.tanh(aoa_delta), -1.0, 1.0)

        amplitude = torch.nn.functional.softplus(gain_raw[..., 0])
        phase = math.pi * torch.tanh(gain_raw[..., 1])
        complex_gain = torch.polar(amplitude, phase)

        left = left_raw.reshape(
            batch,
            selected_count,
            self.config.modes_per_gaussian,
            self.config.polarization_rank,
            self.m_p,
            2,
        )
        right = right_raw.reshape(
            batch,
            selected_count,
            self.config.modes_per_gaussian,
            self.config.polarization_rank,
            self.n_p,
            2,
        )
        left_complex = torch.complex(left[..., 0], left[..., 1])
        right_complex = torch.complex(right[..., 0], right[..., 1])
        left_complex = left_complex / left_complex.abs().square().sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8).sqrt()
        right_complex = right_complex / right_complex.abs().square().sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8).sqrt()
        polarization = torch.einsum(
            "bgkri,bgkrj->bgkij", left_complex, right_complex.conj()
        )

        mode_weight = selection_weight.unsqueeze(-1).expand(
            -1, -1, self.config.modes_per_gaussian
        )
        opacity_weight = opacity.unsqueeze(-1)
        repeated_indices = selected_slot_indices.unsqueeze(-1).expand(
            -1, -1, self.config.modes_per_gaussian
        )

        def flatten(value: torch.Tensor, tail: tuple[int, ...] = ()) -> torch.Tensor:
            return value.reshape(batch, -1, *tail)

        existence_flat = flatten(existence)
        reliability_flat = flatten(reliability)
        selection_flat = flatten(mode_weight)
        opacity_flat = flatten(
            opacity_weight.expand(-1, -1, self.config.modes_per_gaussian)
        )
        gate_flat = (
            selection_flat * opacity_flat * existence_flat * reliability_flat
        )
        return PathModes(
            delay_bins=flatten(delay_bins),
            aod=flatten(aod, (2,)),
            aoa=flatten(aoa, (2,)),
            complex_gain=flatten(complex_gain),
            polarization=flatten(polarization, (self.m_p, self.n_p)),
            existence=existence_flat,
            reliability=reliability_flat,
            width=flatten(width),
            path_type=flatten(path_type, (self.config.path_type_dim,)),
            gate=gate_flat,
            gaussian_indices=flatten(repeated_indices),
            selection_weight=selection_flat,
        )


class ComplexMIMOOFDMRenderer(nn.Module):
    """Pure PyTorch coherent MIMO-OFDM renderer with bounded intermediates."""

    def __init__(
        self,
        round_config: RoundConfig,
        config: RendererConfig | None = None,
        *,
        antenna_order: tuple[str, str, str] = ("H", "V", "P"),
    ) -> None:
        super().__init__()
        self.round_config = round_config
        self.config = config or RendererConfig()
        self.config.validate()
        if tuple(antenna_order) != ("H", "V", "P"):
            raise ValueError(
                "E2E-CGPF currently requires canonical H,V,P antenna order"
            )
        self.antenna_order = tuple(antenna_order)
        subcarriers = torch.arange(round_config.s, dtype=torch.float32)
        self.register_buffer("subcarrier_indices", subcarriers)
        wrapped = torch.minimum(subcarriers, round_config.s - subcarriers)
        self.register_buffer("wrapped_frequency", wrapped / round_config.s)
        self.register_buffer(
            "bs_h", torch.arange(round_config.m_h, dtype=torch.float32)
        )
        self.register_buffer(
            "bs_v", torch.arange(round_config.m_v, dtype=torch.float32)
        )
        self.register_buffer(
            "ue_h", torch.arange(round_config.n_h, dtype=torch.float32)
        )
        self.register_buffer(
            "ue_v", torch.arange(round_config.n_v, dtype=torch.float32)
        )

    def _spatial_steering(
        self,
        direction: torch.Tensor,
        horizontal: torch.Tensor,
        vertical: torch.Tensor,
    ) -> torch.Tensor:
        phase = math.pi * (
            direction[..., 0, None, None] * horizontal.reshape(1, 1, -1, 1)
            + direction[..., 1, None, None] * vertical.reshape(1, 1, 1, -1)
        )
        return torch.polar(torch.ones_like(phase), phase)

    def forward(self, paths: PathModes) -> torch.Tensor:
        batch, path_count = paths.delay_bins.shape
        expected = (batch, path_count)
        for name in ("complex_gain", "existence", "reliability", "gate"):
            if tuple(getattr(paths, name).shape) != expected:
                raise ValueError(f"path field {name} has inconsistent shape")
        result = torch.zeros(
            batch,
            self.round_config.m,
            self.round_config.n,
            self.round_config.s,
            dtype=torch.complex64,
            device=paths.delay_bins.device,
        )
        chunk_size = self.config.path_chunk_size
        for start in range(0, path_count, chunk_size):
            stop = min(path_count, start + chunk_size)
            aod = paths.aod[:, start:stop].to(torch.float32)
            aoa = paths.aoa[:, start:stop].to(torch.float32)
            bs_spatial = self._spatial_steering(aod, self.bs_h, self.bs_v)
            ue_spatial = self._spatial_steering(aoa, self.ue_h, self.ue_v)
            polarization = paths.polarization[:, start:stop].to(torch.complex64)
            matrix = (
                bs_spatial[:, :, :, :, None, None, None, None]
                * polarization[:, :, None, None, :, None, None, :]
                * ue_spatial.conj()[:, :, None, None, None, :, :, None]
            )
            matrix = matrix.reshape(
                batch,
                stop - start,
                self.round_config.m,
                self.round_config.n,
            )
            coefficient = (
                paths.complex_gain[:, start:stop].to(torch.complex64)
                * paths.gate[:, start:stop].to(torch.float32)
            )
            matrix = matrix * coefficient[:, :, None, None]
            delay = paths.delay_bins[:, start:stop].to(torch.float32)
            phase = (
                -2.0
                * math.pi
                * delay.unsqueeze(-1)
                * self.subcarrier_indices.reshape(1, 1, -1)
                / self.round_config.s
            )
            frequency = torch.polar(torch.ones_like(phase), phase)
            width = paths.width[:, start:stop].to(torch.float32)
            envelope = torch.exp(
                -0.5
                * (
                    2.0
                    * math.pi
                    * width.unsqueeze(-1)
                    * self.wrapped_frequency.reshape(1, 1, -1)
                ).square()
            )
            frequency = frequency * envelope
            result = result + torch.einsum("bcmn,bcs->bmns", matrix, frequency)
        if tuple(result.shape[1:]) != self.round_config.channel_shape:
            raise AssertionError("renderer produced the wrong channel shape")
        return result


class E2ECGPF(nn.Module):
    """Joint trainable Gaussian field, path network, and RF renderer."""

    def __init__(
        self,
        field: TrainableGaussianField,
        round_config: RoundConfig,
        config: E2ECGPFConfig,
        position_center: torch.Tensor,
        position_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        config.validate()
        self.field = field
        self.round_config = round_config
        self.config = config
        self.path_network = TargetConditionedPathNetwork(
            config.field,
            config.paths,
            m_p=round_config.m_p,
            n_p=round_config.n_p,
        )
        self.renderer = ComplexMIMOOFDMRenderer(
            round_config,
            config.renderer,
            antenna_order=config.antenna_order,
        )
        center = torch.as_tensor(position_center, dtype=torch.float32)
        scale = torch.as_tensor(position_scale, dtype=torch.float32)
        if center.shape != (3,) or scale.numel() != 1 or float(scale) <= 0:
            raise ValueError("position normalization must be center[3] and positive scale")
        self.register_buffer("position_center", center)
        self.register_buffer("position_scale", scale.reshape(()))
        self.register_buffer(
            "bs_position", torch.tensor(round_config.bs_position, dtype=torch.float32)
        )
        self.stage = "full"

    def set_stage(self, stage: str) -> None:
        if stage not in {"geometry", "complex", "structural", "full"}:
            raise ValueError("unknown training stage")
        self.stage = stage

    def forward(
        self, targets: torch.Tensor, *, map_view: MapMode | None = None
    ) -> tuple[torch.Tensor, PathModes]:
        targets = targets.to(dtype=torch.float32)
        state = self.field.state(map_view)
        paths = self.path_network(
            targets,
            self.bs_position,
            state,
            position_center=self.position_center,
            position_scale=self.position_scale,
        )
        if self.stage == "geometry":
            paths.complex_gain = paths.complex_gain.abs().to(torch.complex64)
            identity = torch.zeros_like(paths.polarization)
            diagonal = min(self.round_config.m_p, self.round_config.n_p)
            for index in range(diagonal):
                identity[..., index, index] = 1.0 / diagonal
            paths.polarization = identity
        channel = self.renderer(paths)
        return channel, paths

    def model_manifest(self) -> dict[str, Any]:
        return {
            "architecture": "E2E-CGPF",
            "config": dataclasses.asdict(self.config),
            "round": dataclasses.asdict(self.round_config),
            "position_center": self.position_center.detach().cpu().tolist(),
            "position_scale": float(self.position_scale.detach().cpu()),
            "o41_forward_dependency": False,
            "task020_dependency": False,
        }


def path_diagnostics(paths: PathModes, field: TrainableGaussianField) -> dict[str, Any]:
    gate = paths.gate.detach()
    delay = paths.delay_bins.detach()
    aod = paths.aod.detach()
    aoa = paths.aoa.detach()
    gain = paths.complex_gain.detach()
    active = gate > 1e-4

    def quantiles(value: torch.Tensor) -> list[float]:
        flattened = value.float().reshape(-1)
        if flattened.numel() == 0:
            return [0.0, 0.0, 0.0]
        return [
            float(v)
            for v in torch.quantile(
                flattened, torch.tensor([0.1, 0.5, 0.9], device=value.device)
            ).cpu()
        ]

    per_target_energy = (gain.abs().square() * gate.square()).sum(dim=1)
    phase = torch.angle(gain)
    circular = torch.exp(torch.complex(torch.zeros_like(phase), phase)).mean()
    selected_unique = int(torch.unique(paths.gaussian_indices).numel())
    polarization_energy = paths.polarization.detach().abs().square().sum(
        dim=(-2, -1)
    )
    return {
        "field_active": field.active_count,
        "field_capacity": field.config.max_count,
        "field_structure_version": int(field.structure_version.item()),
        "paths_per_target": paths.path_count,
        "active_paths_mean": float(active.sum(dim=1).float().mean().cpu()),
        "active_paths_min": int(active.sum(dim=1).min().cpu()),
        "active_paths_max": int(active.sum(dim=1).max().cpu()),
        "delay_quantiles": quantiles(delay),
        "aod_quantiles": quantiles(aod),
        "aoa_quantiles": quantiles(aoa),
        "gate_quantiles": quantiles(gate),
        "path_energy_quantiles": quantiles(gain.abs().square() * gate.square()),
        "phase_quantiles": quantiles(phase),
        "existence_quantiles": quantiles(paths.existence.detach()),
        "reliability_quantiles": quantiles(paths.reliability.detach()),
        "width_quantiles": quantiles(paths.width.detach()),
        "polarization_energy_quantiles": quantiles(polarization_energy),
        "selected_gaussians_unique": selected_unique,
        "field_coverage_ratio": selected_unique / max(field.active_count, 1),
        "target_energy_std": float(per_target_energy.float().std(unbiased=False).cpu()),
        "phase_circular_concentration": float(circular.abs().cpu()),
        "polarization_matrix_rank_max": int(
            min(paths.polarization.shape[-2], paths.polarization.shape[-1])
        ),
    }
