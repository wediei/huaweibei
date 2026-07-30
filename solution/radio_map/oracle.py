"""Leakage-free codec reconstruction sweeps for model-selection decisions."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable

import numpy as np

from .codecs import (
    GlobalSupportCodec,
    SharedTuckerCodec,
    oracle_topk_reconstruction,
)
from .data import RoundDataset
from .metrics import MetricAccumulator
from .splits import coverage_split
from .transforms import AntennaLayout


def _indices_hash(indices: np.ndarray) -> str:
    canonical = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _bounded_subset(
    indices: np.ndarray,
    limit: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if limit is None or limit >= len(indices):
        return indices.copy()
    if limit <= 0:
        raise ValueError("sample limits must be positive when provided")
    return np.sort(rng.choice(indices, size=limit, replace=False)).astype(np.int64)


def _evaluate_reconstruction(
    reconstruct: Callable[[np.ndarray], np.ndarray],
    dataset: RoundDataset,
    indices: np.ndarray,
    layout: AntennaLayout,
    batch_size: int,
) -> dict[str, float]:
    accumulator = MetricAccumulator(layout, dataset.config.weights)
    started = time.perf_counter()
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        target = dataset.channel_batch(batch_indices)
        prediction = reconstruct(target)
        accumulator.update(prediction, target)
    metrics = accumulator.compute().to_dict()
    metrics["runtime_seconds"] = float(time.perf_counter() - started)
    return metrics


def codec_oracle_report(
    dataset: RoundDataset,
    *,
    validation_fraction: float,
    grid_size: float,
    seed: int,
    batch_size: int,
    layout_order: tuple[str, str, str],
    tucker_ranks: tuple[int, int, int, int, int],
    support_fractions: tuple[float, ...],
    fit_samples: int | None = None,
    validation_samples: int | None = None,
) -> dict[str, object]:
    """Fit codecs on anchors and evaluate reconstruction on disjoint targets."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not support_fractions or any(
        fraction <= 0.0 or fraction > 1.0 for fraction in support_fractions
    ):
        raise ValueError("support fractions must lie in (0, 1]")
    split = coverage_split(
        dataset.train_pos,
        validation_fraction=validation_fraction,
        grid_size=grid_size,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    fit_indices = _bounded_subset(split.train, fit_samples, rng)
    validation_indices = _bounded_subset(
        split.validation, validation_samples, rng
    )
    overlap = np.intersect1d(fit_indices, validation_indices)
    if len(overlap):
        raise RuntimeError("codec fit/validation index leakage detected")

    layout = AntennaLayout(dataset.config, layout_order)
    tucker = SharedTuckerCodec(layout, tucker_ranks)
    fit_started = time.perf_counter()
    tucker.fit(dataset.train_channel, fit_indices, batch_size=batch_size)
    tucker_fit_seconds = time.perf_counter() - fit_started
    retained_energy = []
    for rank, eigenvalues in zip(tucker.ranks, tucker.mode_energy):
        total = float(eigenvalues.sum())
        retained_energy.append(
            1.0 if total == 0.0 else float(eigenvalues[:rank].sum() / total)
        )
    tucker_report = {
        "ranks": list(tucker.ranks),
        "fit_seconds": float(tucker_fit_seconds),
        "mode_retained_energy": retained_energy,
        "compression": tucker.compression_report(
            amortized_samples=dataset.config.p_test
        ),
        "metrics": _evaluate_reconstruction(
            tucker.reconstruct,
            dataset,
            validation_indices,
            layout,
            batch_size,
        ),
    }

    total_coefficients = int(np.prod(layout.structured_tail))
    sparse_candidates: list[dict[str, object]] = []
    for fraction in support_fractions:
        coefficient_count = min(
            total_coefficients,
            max(1, int(round(total_coefficients * fraction))),
        )
        global_codec = GlobalSupportCodec(layout, coefficient_count)
        fit_started = time.perf_counter()
        global_codec.fit(
            dataset.train_channel, fit_indices, batch_size=batch_size
        )
        global_fit_seconds = time.perf_counter() - fit_started
        sparse_candidates.append(
            {
                "support_fraction": float(fraction),
                "coefficient_count": coefficient_count,
                "global_support_fit_seconds": float(global_fit_seconds),
                "compression": global_codec.compression_report(),
                "global_support_metrics": _evaluate_reconstruction(
                    global_codec.reconstruct,
                    dataset,
                    validation_indices,
                    layout,
                    batch_size,
                ),
                "per_sample_topk_oracle_metrics": _evaluate_reconstruction(
                    lambda batch, count=coefficient_count: oracle_topk_reconstruction(
                        batch, layout, count
                    ),
                    dataset,
                    validation_indices,
                    layout,
                    batch_size,
                ),
            }
        )

    return {
        "configuration": {
            "validation_fraction": float(validation_fraction),
            "grid_size": float(grid_size),
            "seed": int(seed),
            "batch_size": int(batch_size),
            "layout_order": "".join(layout_order),
            "fit_samples_limit": fit_samples,
            "validation_samples_limit": validation_samples,
        },
        "split": {
            "fit_indices": fit_indices.tolist(),
            "validation_indices": validation_indices.tolist(),
            "fit_indices_sha256": _indices_hash(fit_indices),
            "validation_indices_sha256": _indices_hash(validation_indices),
            "overlap_count": int(len(overlap)),
        },
        "tucker": tucker_report,
        "sparse_candidates": sparse_candidates,
        "decision_gate": {
            "deployable_codec_target_score": 0.98,
            "oracle_is_not_deployable": True,
        },
    }

