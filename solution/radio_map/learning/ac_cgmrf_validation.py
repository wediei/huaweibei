"""Strict spatial folds, Anchor leakage audits, and research gates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..splits import SplitIndices


@dataclass(frozen=True)
class LeakageAudit:
    train_count: int
    validation_count: int
    checked_anchor_count: int
    target_self_leaks: tuple[int, ...]
    validation_anchor_leaks: tuple[int, ...]

    @property
    def passed(self) -> bool:
        return not self.target_self_leaks and not self.validation_anchor_leaks

    def to_dict(self) -> dict[str, object]:
        return {
            "train_count": self.train_count,
            "validation_count": self.validation_count,
            "checked_anchor_count": self.checked_anchor_count,
            "target_self_leaks": list(self.target_self_leaks),
            "validation_anchor_leaks": list(self.validation_anchor_leaks),
            "passed": self.passed,
        }


def _validate_positions(positions: np.ndarray) -> np.ndarray:
    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or len(values) < 2:
        raise ValueError("positions must have shape P,D with P>=2 and D>=2")
    if not np.isfinite(values).all():
        raise ValueError("positions must be finite")
    return values


def spatial_block_folds(
    positions: np.ndarray,
    fold_count: int,
    block_size: float | Sequence[float],
    seed: int = 42,
) -> tuple[SplitIndices, ...]:
    """Assign whole coordinate blocks to balanced deterministic folds."""

    values = _validate_positions(positions)
    if not isinstance(fold_count, int) or isinstance(fold_count, bool) or fold_count < 2:
        raise ValueError("fold_count must be at least two")
    size = np.asarray(block_size, dtype=np.float64)
    if size.ndim == 0:
        size = np.full(values.shape[1], float(size))
    if size.shape != (values.shape[1],) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("block_size must be positive scalar or per-axis vector")
    cells = np.floor((values - values.min(axis=0, keepdims=True)) / size).astype(
        np.int64
    )
    unique, inverse = np.unique(cells, axis=0, return_inverse=True)
    if len(unique) < fold_count:
        raise ValueError("fewer spatial blocks than requested folds")
    groups = [np.flatnonzero(inverse == index) for index in range(len(unique))]
    rng = np.random.default_rng(seed)
    tie_break = rng.permutation(len(groups))
    order = sorted(
        range(len(groups)),
        key=lambda index: (-len(groups[index]), int(tie_break[index])),
    )
    assignments: list[list[np.ndarray]] = [[] for _ in range(fold_count)]
    counts = np.zeros(fold_count, dtype=np.int64)
    for group_index in order:
        target_fold = int(np.argmin(counts))
        assignments[target_fold].append(groups[group_index])
        counts[target_fold] += len(groups[group_index])
    all_indices = np.arange(len(values), dtype=np.int64)
    folds = []
    for groups_in_fold in assignments:
        validation = np.sort(np.concatenate(groups_in_fold)).astype(np.int64)
        train = np.setdiff1d(all_indices, validation, assume_unique=True)
        if len(train) == 0 or len(validation) == 0:
            raise ValueError("spatial fold is empty")
        folds.append(SplitIndices(train=train, validation=validation))
    return tuple(folds)


def spatial_fold_manifest(
    positions: np.ndarray,
    folds: Sequence[SplitIndices],
    block_size: float | Sequence[float],
    seed: int,
) -> dict[str, object]:
    values = _validate_positions(positions)
    covered = np.concatenate([np.asarray(fold.validation) for fold in folds])
    if not np.array_equal(np.sort(covered), np.arange(len(values))):
        raise ValueError("validation folds do not partition all positions")
    digest = hashlib.sha256(values.astype("<f8", copy=False).tobytes()).hexdigest()
    return {
        "kind": "ac_cgmrf_spatial_folds",
        "position_sha256": digest,
        "sample_count": len(values),
        "fold_count": len(folds),
        "block_size": np.asarray(block_size).tolist(),
        "seed": int(seed),
        "folds": [
            {
                "fold": index,
                "train_indices": np.asarray(fold.train, dtype=np.int64).tolist(),
                "validation_indices": np.asarray(
                    fold.validation, dtype=np.int64
                ).tolist(),
            }
            for index, fold in enumerate(folds)
        ],
    }


def audit_anchor_exclusion(
    train_indices: Iterable[int],
    validation_indices: Iterable[int],
    neighbor_indices: np.ndarray,
) -> LeakageAudit:
    """Require every validation Anchor to be in train and never be its target."""

    train = np.asarray(list(train_indices), dtype=np.int64)
    validation = np.asarray(list(validation_indices), dtype=np.int64)
    neighbors = np.asarray(neighbor_indices, dtype=np.int64)
    if train.ndim != 1 or validation.ndim != 1 or neighbors.ndim != 2:
        raise ValueError("invalid split or neighbor dimensions")
    if np.intersect1d(train, validation).size:
        raise ValueError("train and validation overlap")
    if np.any(validation < 0) or np.any(validation >= len(neighbors)):
        raise IndexError("validation target outside neighbor table")
    train_set = set(int(value) for value in train)
    self_leaks: list[int] = []
    validation_leaks: list[int] = []
    checked = 0
    for target in validation:
        row = neighbors[int(target)]
        checked += len(row)
        if int(target) in row:
            self_leaks.append(int(target))
        if any(int(anchor) not in train_set for anchor in row):
            validation_leaks.append(int(target))
    return LeakageAudit(
        train_count=len(train),
        validation_count=len(validation),
        checked_anchor_count=checked,
        target_self_leaks=tuple(self_leaks),
        validation_anchor_leaks=tuple(validation_leaks),
    )


def pilot_gate(
    report: dict[str, object],
    minimum_gain: float = 0.01,
    minimum_shuffle_margin: float = 0.005,
    minimum_support_novelty: float = 0.01,
) -> dict[str, object]:
    """Apply the single-fold stop conditions without rounding."""

    best = report["best"]
    baseline = report["baseline"]
    real = best["real"]
    shuffle = best["shuffle"]
    gain = float(real["score"]) - float(baseline["score"])
    shuffle_margin = float(real["score"]) - float(shuffle["score"])
    novelty = float(report.get("support_novelty", 0.0))
    checks = {
        "gain_at_least_0_01": gain >= minimum_gain,
        "real_over_shuffle_at_least_0_005": shuffle_margin
        >= minimum_shuffle_margin,
        "support_novelty_nontrivial": novelty >= minimum_support_novelty,
        "pas_not_lower": float(real["pas"]) >= float(baseline["pas"]),
        "pdp_not_lower": float(real["pdp"]) >= float(baseline["pdp"]),
        "nmse_not_worse": float(real["nmse"]) <= float(baseline["nmse"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "real_gain": gain,
        "real_over_shuffle": shuffle_margin,
        "support_novelty": novelty,
        "decision": "CONTINUE_TO_CV" if all(checks.values()) else "STOP",
    }


def _mean_metrics(metrics: Sequence[dict[str, float]]) -> dict[str, float]:
    keys = ("pas", "pdp", "nmse", "score")
    return {
        key: float(np.mean([float(item[key]) for item in metrics]))
        for key in keys
    }


def aggregate_cross_validation(
    reports: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Choose one common scale across folds and enforce all full-training gates."""

    if len(reports) < 2:
        raise ValueError("at least two spatial fold reports are required")
    common_scales = set(reports[0]["metrics"]["real"])
    for report in reports[1:]:
        common_scales &= set(report["metrics"]["real"])
    common_scales.discard("0")
    common_scales.discard("0.0")
    if not common_scales:
        raise ValueError("fold reports share no non-zero validation scale")
    baseline = [_baseline_metrics(report) for report in reports]
    scale_summaries: dict[str, dict[str, object]] = {}
    for scale in sorted(common_scales, key=float):
        real = [report["metrics"]["real"][scale] for report in reports]
        zero = [report["metrics"]["zero"][scale] for report in reports]
        shuffle = [report["metrics"]["shuffle"][scale] for report in reports]
        fold_gains = [
            float(item["score"]) - float(base["score"])
            for item, base in zip(real, baseline)
        ]
        scale_summaries[scale] = {
            "real": _mean_metrics(real),
            "zero": _mean_metrics(zero),
            "shuffle": _mean_metrics(shuffle),
            "baseline": _mean_metrics(baseline),
            "fold_gains": fold_gains,
            "mean_gain": float(np.mean(fold_gains)),
            "real_over_zero": float(
                np.mean(
                    [
                        float(item["score"]) - float(control["score"])
                        for item, control in zip(real, zero)
                    ]
                )
            ),
            "real_over_shuffle": float(
                np.mean(
                    [
                        float(item["score"]) - float(control["score"])
                        for item, control in zip(real, shuffle)
                    ]
                )
            ),
        }
    best_scale, stable = max(
        scale_summaries.items(), key=lambda item: item[1]["mean_gain"]
    )
    checks = {
        "all_folds_positive": all(value > 0 for value in stable["fold_gains"]),
        "mean_gain_at_least_0_03": stable["mean_gain"] >= 0.03,
        "stable_configuration_gain_at_least_0_04": stable["mean_gain"] >= 0.04,
        "real_over_zero_at_least_0_01": stable["real_over_zero"] >= 0.01,
        "real_over_shuffle_at_least_0_01": stable["real_over_shuffle"] >= 0.01,
        "pas_not_lower": stable["real"]["pas"] >= stable["baseline"]["pas"],
        "pdp_not_lower": stable["real"]["pdp"] >= stable["baseline"]["pdp"],
        "nmse_not_worse": stable["real"]["nmse"] <= stable["baseline"]["nmse"],
    }
    promoted = all(checks.values())
    return {
        "kind": "ac_cgmrf_cross_validation",
        "fold_count": len(reports),
        "best_stable_scale": float(best_scale),
        "stable": stable,
        "scales": scale_summaries,
        "checks": checks,
        "promoted": promoted,
        "decision": "ALLOW_FULL_TRAINING_AND_INFERENCE" if promoted else "STOP",
    }


def _baseline_metrics(report: dict[str, object]) -> dict[str, float]:
    baseline = report.get("baseline")
    if baseline is not None:
        return baseline
    real = report["metrics"]["real"]
    if "0.0" in real:
        return real["0.0"]
    if "0" in real:
        return real["0"]
    raise ValueError("fold report has no O4.1 scale=0 baseline")


def require_promotion_report(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if not source.is_file():
        raise ValueError("promotion report does not exist")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        payload.get("kind") != "ac_cgmrf_cross_validation"
        or payload.get("promoted") is not True
    ):
        raise ValueError("full spatial validation gates have not passed")
    return payload
