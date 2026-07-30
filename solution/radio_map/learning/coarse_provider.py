"""Authenticated frozen coarse predictions shared by optional later stages.

This module deliberately wraps the existing base mixer and Power Refiner
instead of reimplementing either model.  Its main purpose is to give every
transport/generative experiment the same explicit coarse identity and an
exact disabled-stage return path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal, TypeVar

import torch
from torch import nn

from .anchor_mixer import MixerOutput
from .cli import _load_verified_trainer
from .power_refiner_cli import (
    PowerAwareRefiner,
    _adapter_sha256,
    _load_refiner,
    _sha256,
)


AnchorSource = Literal["fold", "all_official_train"]
SplitName = Literal["train", "validation", "official_test"]
TensorT = TypeVar("TensorT")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _optional_file_identity(path: str | Path | None, label: str) -> dict[str, str] | None:
    if path is None:
        return None
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{label} is missing: {source}")
    return {
        "path": str(source.resolve()),
        "sha256": _sha256(source),
    }


@dataclass(frozen=True)
class CoarseSpec:
    """Versioned identity of a frozen deployable coarse predictor."""

    kind: str = "power_refiner_v1"
    base_checkpoint: str | Path | None = None
    refiner_checkpoint: str | Path | None = None
    power_scale: float = 1.25
    anchor_source: AnchorSource = "fold"
    feature_version: int = 1
    hidden_dim: int = 64
    map_feature_dim: int = 0
    geometry_cache: str | Path | None = None
    map_mode: str = "real"
    format_version: int = 1

    def __post_init__(self) -> None:
        if self.format_version != 1:
            raise ValueError("unsupported coarse spec format_version")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("coarse kind must be a non-empty string")
        if (
            not isinstance(self.power_scale, (int, float))
            or isinstance(self.power_scale, bool)
            or not math.isfinite(float(self.power_scale))
            or float(self.power_scale) < 0.0
        ):
            raise ValueError("power_scale must be finite and non-negative")
        if self.anchor_source not in ("fold", "all_official_train"):
            raise ValueError("anchor_source must be fold or all_official_train")
        if self.feature_version not in (1, 2, 3):
            raise ValueError("feature_version must be 1, 2, or 3")
        if (
            not isinstance(self.hidden_dim, int)
            or isinstance(self.hidden_dim, bool)
            or self.hidden_dim < 8
        ):
            raise ValueError("hidden_dim must be an integer of at least 8")
        if (
            not isinstance(self.map_feature_dim, int)
            or isinstance(self.map_feature_dim, bool)
            or self.map_feature_dim < 0
        ):
            raise ValueError("map_feature_dim must be a non-negative integer")
        if self.feature_version == 3 and self.map_feature_dim <= 0:
            raise ValueError("feature_version 3 requires map_feature_dim")
        if self.feature_version < 3 and self.map_feature_dim != 0:
            raise ValueError("map_feature_dim is only valid for feature_version 3")
        if self.map_mode not in ("real", "zero", "shuffle"):
            raise ValueError("map_mode must be real, zero, or shuffle")

    def identity(
        self,
        adapter_sha256: str,
        fold_manifest_fingerprint: str,
    ) -> dict[str, Any]:
        if not isinstance(adapter_sha256, str) or not adapter_sha256:
            raise ValueError("adapter_sha256 must be a non-empty string")
        if (
            not isinstance(fold_manifest_fingerprint, str)
            or not fold_manifest_fingerprint
        ):
            raise ValueError("fold manifest fingerprint must be non-empty")
        return {
            "format_version": self.format_version,
            "kind": self.kind,
            "base_checkpoint": _optional_file_identity(
                self.base_checkpoint, "base checkpoint"
            ),
            "refiner_checkpoint": _optional_file_identity(
                self.refiner_checkpoint, "refiner checkpoint"
            ),
            "power_scale": float(self.power_scale),
            "anchor_source": self.anchor_source,
            "feature_version": self.feature_version,
            "hidden_dim": self.hidden_dim,
            "map_feature_dim": self.map_feature_dim,
            "geometry_cache": _optional_file_identity(
                self.geometry_cache, "geometry cache"
            ),
            "map_mode": self.map_mode,
            "adapter_sha256": adapter_sha256,
            "fold_manifest_fingerprint": fold_manifest_fingerprint,
        }

    def fingerprint(
        self,
        adapter_sha256: str,
        fold_manifest_fingerprint: str,
    ) -> str:
        return hashlib.sha256(
            _canonical_json(
                self.identity(adapter_sha256, fold_manifest_fingerprint)
            )
        ).hexdigest()


def apply_optional_correction(
    coarse: TensorT,
    scale: float,
    operation: Callable[[TensorT], TensorT],
) -> TensorT:
    """Apply an optional later stage with a hard, callback-free zero bypass."""

    value = float(scale)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("correction scale must be finite and non-negative")
    if value == 0.0:
        return coarse
    correction = operation(coarse)
    return coarse + value * correction


class FrozenCoarseProvider(nn.Module):
    """Run one authenticated base/refiner coarse path in evaluation mode."""

    def __init__(
        self,
        base: nn.Module,
        adapter: object,
        spec: CoarseSpec,
        manifest_fingerprint: str,
        refiner: PowerAwareRefiner | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Module):
            raise TypeError("base must be a torch module")
        if refiner is not None and refiner.base is not base:
            raise ValueError("refiner must wrap the exact provider base model")
        self.base = base
        self.refiner = refiner
        self.adapter = adapter
        self.spec = spec
        self.manifest_fingerprint = str(manifest_fingerprint)
        self.adapter_sha256 = _adapter_sha256(adapter)
        self.identity = spec.identity(
            self.adapter_sha256, self.manifest_fingerprint
        )
        self.fingerprint = hashlib.sha256(
            _canonical_json(self.identity)
        ).hexdigest()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    @classmethod
    def from_components(
        cls,
        base: nn.Module,
        adapter: object,
        spec: CoarseSpec,
        manifest_fingerprint: str,
        refiner: PowerAwareRefiner | None = None,
    ) -> "FrozenCoarseProvider":
        """Construct from already verified components (also useful for tests)."""

        return cls(base, adapter, spec, manifest_fingerprint, refiner)

    @classmethod
    def load(
        cls,
        spec: CoarseSpec,
        data_dir: str | Path,
        fold_cache: str | Path,
        device: str | torch.device = "cpu",
    ) -> tuple["FrozenCoarseProvider", Any, Any, Any, Any, Any]:
        """Load the existing verified base and optional Power Refiner assets.

        The extra return values are the existing manifest, dataset, adapter and
        persisted split indices, allowing later stages to reuse the same
        authenticated context without reopening private checkpoint formats.
        """

        if spec.base_checkpoint is None:
            raise ValueError("base_checkpoint is required when loading assets")
        args = SimpleNamespace(
            cache_dir=str(fold_cache),
            data_dir=str(data_dir),
            checkpoint=str(spec.base_checkpoint),
            device=str(device),
            # New optional heads are read-only consumers of the authenticated
            # historical fold.  Every artifact/source/split hash is still
            # checked; only the producer code fingerprint may predate them.
            allow_cache_code_mismatch=True,
        )
        trainer, manifest, dataset, adapter, train, validation = (
            _load_verified_trainer(args)
        )
        refiner = None
        if spec.refiner_checkpoint is not None:
            refiner = PowerAwareRefiner(
                trainer.model,
                adapter,
                hidden_dim=spec.hidden_dim,
                feature_version=spec.feature_version,
                map_feature_dim=spec.map_feature_dim,
            ).to(trainer.device)
            _load_refiner(
                spec.refiner_checkpoint,
                refiner,
                _sha256(spec.base_checkpoint),
                _adapter_sha256(adapter),
                manifest.fingerprint,
                None,
            )
        provider = cls.from_components(
            trainer.model,
            adapter,
            spec,
            manifest.fingerprint,
            refiner,
        ).to(trainer.device)
        return provider, manifest, dataset, adapter, train, validation

    def train(self, mode: bool = True) -> "FrozenCoarseProvider":
        # A provider is an immutable inference component even when a parent
        # pipeline enters training mode.
        super().train(False)
        self.base.eval()
        if self.refiner is not None:
            self.refiner.eval()
        return self

    def validate_anchor_source(self, split: SplitName) -> AnchorSource:
        if split not in ("train", "validation", "official_test"):
            raise ValueError("unknown split")
        # The spec records the deployable official-test policy.  Training and
        # validation are always forced to the persisted fold pool regardless
        # of that policy, so one authenticated identity covers both phases.
        if split in ("train", "validation"):
            return "fold"
        return self.spec.anchor_source

    def forward(self, batch: dict[str, torch.Tensor]) -> MixerOutput:
        with torch.no_grad():
            if self.refiner is None:
                return self.base(batch)
            refined = self.refiner(batch)
            unrefined_latent = refined.latent - refined.low_rank_residual
            scaled_latent = unrefined_latent + float(self.spec.power_scale) * (
                refined.latent - unrefined_latent
            )
            return dataclasses.replace(
                refined,
                latent=scaled_latent,
                low_rank_residual=scaled_latent - unrefined_latent,
            )

    def latent(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self(batch).latent
