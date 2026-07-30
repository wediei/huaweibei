"""Differentiable learning utilities for radio-map completion."""

from .latent_adapter import FixedSupportLatentAdapter
from .anchor_mixer import MixerOutput, SupportAwareAnchorMixer, SupportAwareAnchorMixerConfig
from .cache import FoldCacheConfig, FoldCacheManifest, prepare_fold_cache, validate_cache
from .dataset import CachedAnchorDataset, CoordinateBatchContext, build_coordinate_batch, collate_anchor_batch, load_persisted_split
from .losses import AnchorLossConfig, AnchorLossValues, anchor_completion_loss, complex_mse
from .trainer import Trainer, TrainerConfig
from .e2e_cgpf import (
    ComplexMIMOOFDMRenderer,
    E2ECGPF,
    E2ECGPFConfig,
    GaussianFieldConfig,
    PathModes,
    PathNetworkConfig,
    RendererConfig,
    TargetConditionedPathNetwork,
    TrainableGaussianField,
)
from .e2e_cgpf_losses import (
    E2ECGPFLossConfig,
    E2ECGPFLossValues,
    e2e_cgpf_loss,
)

__all__ = [
    "FixedSupportLatentAdapter",
    "MixerOutput",
    "SupportAwareAnchorMixer",
    "SupportAwareAnchorMixerConfig",
    "FoldCacheConfig",
    "FoldCacheManifest",
    "prepare_fold_cache",
    "validate_cache",
    "CachedAnchorDataset",
    "CoordinateBatchContext",
    "build_coordinate_batch",
    "collate_anchor_batch",
    "load_persisted_split",
    "AnchorLossConfig",
    "AnchorLossValues",
    "anchor_completion_loss",
    "complex_mse",
    "Trainer",
    "TrainerConfig",
    "ComplexMIMOOFDMRenderer",
    "E2ECGPF",
    "E2ECGPFConfig",
    "GaussianFieldConfig",
    "PathModes",
    "PathNetworkConfig",
    "RendererConfig",
    "TargetConditionedPathNetwork",
    "TrainableGaussianField",
    "E2ECGPFLossConfig",
    "E2ECGPFLossValues",
    "e2e_cgpf_loss",
]
