"""Core utilities for Round1 radio-map completion."""

from .config import RoundConfig
from .codecs import GlobalSupportCodec, SharedTuckerCodec, oracle_topk_reconstruction
from .baselines import InverseDistanceRegressor, NearestAnchorRegressor
from .anchors import AnchorMemory, AnchorQuery, PAIR_FEATURE_NAMES, build_prior_batch
from .data import DatasetAudit, RoundDataset
from .geometry import GeometryPrior, PlyPointCloud, build_geometry_prior
from .metrics import CompetitionMetrics, MetricAccumulator, competition_metrics
from .oracle import codec_oracle_report
from .transforms import (
    AntennaLayout,
    beam_delay,
    candidate_orders,
    inverse_beam_delay,
)
from .splits import SplitIndices, block_split, coverage_split, nearest_anchor_distances

__all__ = [
    "AntennaLayout",
    "AnchorMemory",
    "AnchorQuery",
    "CompetitionMetrics",
    "DatasetAudit",
    "InverseDistanceRegressor",
    "GlobalSupportCodec",
    "GeometryPrior",
    "MetricAccumulator",
    "NearestAnchorRegressor",
    "PlyPointCloud",
    "PAIR_FEATURE_NAMES",
    "RoundConfig",
    "RoundDataset",
    "SharedTuckerCodec",
    "SplitIndices",
    "beam_delay",
    "block_split",
    "build_geometry_prior",
    "build_prior_batch",
    "candidate_orders",
    "competition_metrics",
    "codec_oracle_report",
    "coverage_split",
    "inverse_beam_delay",
    "nearest_anchor_distances",
    "oracle_topk_reconstruction",
]
