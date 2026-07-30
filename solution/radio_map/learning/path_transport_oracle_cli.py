"""O2 target-visible ceiling for low-dimensional path transport parameters."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm.auto import tqdm

from ..metrics import MetricAccumulator
from ..transforms import AntennaLayout, beam_delay, inverse_beam_delay
from .cli import _load_fold


_STAGES = (
    "identity",
    "common_phase",
    "delay_shift",
    "beam_delay_shift",
    "amplitude",
    "reliability",
)


def fit_group_transport_parameters(
    reference_beam_delay: np.ndarray,
    target_beam_delay: np.ndarray,
    max_h_shift: int,
    max_v_shift: int,
    max_delay_shift: int,
) -> dict[str, np.ndarray]:
    """Fit deployable-sized O2 labels, independently for every P/N group."""

    reference = np.asarray(reference_beam_delay)
    target = np.asarray(target_beam_delay)
    if (
        reference.shape != target.shape
        or reference.ndim != 6
        or not np.issubdtype(reference.dtype, np.complexfloating)
        or not np.issubdtype(target.dtype, np.complexfloating)
    ):
        raise ValueError("transport labels require matching complex B,H,V,P,N,D")
    shape = (reference.shape[0], reference.shape[3], reference.shape[4])
    output = {
        "delta_h": np.zeros(shape, dtype=np.float32),
        "delta_v": np.zeros(shape, dtype=np.float32),
        "delta_delay": np.zeros(shape, dtype=np.float32),
        "log_amplitude": np.zeros(shape, dtype=np.float32),
        "phase_real": np.ones(shape, dtype=np.float32),
        "phase_imag": np.zeros(shape, dtype=np.float32),
        "existence": np.ones(shape, dtype=np.float32),
        "reliability": np.zeros(shape, dtype=np.float32),
    }
    for batch_index in range(shape[0]):
        for p_index in range(shape[1]):
            for n_index in range(shape[2]):
                source = reference[batch_index, :, :, p_index, n_index, :]
                destination = target[batch_index, :, :, p_index, n_index, :]
                shift = _best_beam_delay_shift(
                    source,
                    destination,
                    max_h_shift,
                    max_v_shift,
                    max_delay_shift,
                )
                shifted = np.roll(source, shift=shift, axis=(0, 1, 2))
                gain = _complex_fit(shifted, destination)
                magnitude = abs(gain)
                unit = 1.0 + 0.0j if magnitude <= 1e-12 else gain / magnitude
                candidate = shifted * gain
                delta = candidate - source
                denominator = float(np.vdot(delta, delta).real)
                reliability = (
                    0.0
                    if denominator <= 1e-12
                    else float(
                        np.clip(
                            np.vdot(delta, destination - source).real
                            / denominator,
                            0.0,
                            1.0,
                        )
                    )
                )
                output["delta_h"][batch_index, p_index, n_index] = shift[0]
                output["delta_v"][batch_index, p_index, n_index] = shift[1]
                output["delta_delay"][batch_index, p_index, n_index] = shift[2]
                output["log_amplitude"][batch_index, p_index, n_index] = (
                    float(np.log(max(magnitude, 1e-8)))
                )
                output["phase_real"][batch_index, p_index, n_index] = unit.real
                output["phase_imag"][batch_index, p_index, n_index] = unit.imag
                output["existence"][batch_index, p_index, n_index] = (
                    0.0 if magnitude <= 1e-8 else 1.0
                )
                output["reliability"][batch_index, p_index, n_index] = reliability
    return output


def _complex_fit(reference: np.ndarray, target: np.ndarray) -> complex:
    denominator = float(np.vdot(reference, reference).real)
    if denominator <= np.finfo(np.float32).eps:
        return 0.0j if np.any(np.abs(target) > 0) else 1.0 + 0.0j
    return complex(np.vdot(reference, target) / denominator)


def _phase_only(reference: np.ndarray, target: np.ndarray) -> complex:
    value = _complex_fit(reference, target)
    magnitude = abs(value)
    return 1.0 + 0.0j if magnitude <= 1e-12 else value / magnitude


def _signed_bins(size: int) -> np.ndarray:
    values = np.arange(size, dtype=np.int64)
    return np.where(values <= size // 2, values, values - size)


def _best_delay_shift(
    reference: np.ndarray, target: np.ndarray, maximum: int
) -> int:
    if not np.any(reference) or not np.any(target):
        return 0
    cross_spectrum = np.sum(
        np.fft.fft(target, axis=-1)
        * np.conj(np.fft.fft(reference, axis=-1)),
        axis=(0, 1),
    )
    correlation = np.abs(np.fft.ifft(cross_spectrum))
    shifts = _signed_bins(reference.shape[-1])
    correlation[np.abs(shifts) > maximum] = -np.inf
    return int(shifts[int(np.argmax(correlation))])


def _best_beam_delay_shift(
    reference: np.ndarray,
    target: np.ndarray,
    max_h: int,
    max_v: int,
    max_delay: int,
) -> tuple[int, int, int]:
    if not np.any(reference) or not np.any(target):
        return 0, 0, 0
    correlation = np.abs(
        np.fft.ifftn(
            np.fft.fftn(target) * np.conj(np.fft.fftn(reference))
        )
    )
    h_shifts = _signed_bins(reference.shape[0])
    v_shifts = _signed_bins(reference.shape[1])
    d_shifts = _signed_bins(reference.shape[2])
    valid = (
        (np.abs(h_shifts)[:, None, None] <= max_h)
        & (np.abs(v_shifts)[None, :, None] <= max_v)
        & (np.abs(d_shifts)[None, None, :] <= max_delay)
    )
    correlation = np.where(valid, correlation, -np.inf)
    h_index, v_index, d_index = np.unravel_index(
        int(np.argmax(correlation)), correlation.shape
    )
    return (
        int(h_shifts[h_index]),
        int(v_shifts[v_index]),
        int(d_shifts[d_index]),
    )


def _oracle_stages(
    reference: np.ndarray,
    target: np.ndarray,
    max_h_shift: int,
    max_v_shift: int,
    max_delay_shift: int,
) -> tuple[dict[str, np.ndarray], list[tuple[int, int, int]]]:
    outputs = {
        name: np.zeros_like(reference) for name in _STAGES
    }
    outputs["identity"][...] = reference
    shifts: list[tuple[int, int, int]] = []
    b_count, _, _, p_count, n_count, _ = reference.shape
    for batch_index in range(b_count):
        for p_index in range(p_count):
            for n_index in range(n_count):
                source_group = reference[
                    batch_index, :, :, p_index, n_index, :
                ]
                target_group = target[
                    batch_index, :, :, p_index, n_index, :
                ]
                common = source_group * _phase_only(
                    source_group, target_group
                )
                outputs["common_phase"][
                    batch_index, :, :, p_index, n_index, :
                ] = common

                delay = _best_delay_shift(
                    source_group, target_group, max_delay_shift
                )
                delay_group = np.roll(source_group, delay, axis=2)
                delay_group *= _phase_only(delay_group, target_group)
                outputs["delay_shift"][
                    batch_index, :, :, p_index, n_index, :
                ] = delay_group

                shift = _best_beam_delay_shift(
                    source_group,
                    target_group,
                    max_h_shift,
                    max_v_shift,
                    max_delay_shift,
                )
                shifts.append(shift)
                shifted = np.roll(
                    source_group, shift=shift, axis=(0, 1, 2)
                )
                phase_group = shifted * _phase_only(shifted, target_group)
                outputs["beam_delay_shift"][
                    batch_index, :, :, p_index, n_index, :
                ] = phase_group

                amplitude_group = shifted * _complex_fit(
                    shifted, target_group
                )
                outputs["amplitude"][
                    batch_index, :, :, p_index, n_index, :
                ] = amplitude_group

                delta = amplitude_group - source_group
                denominator = float(np.vdot(delta, delta).real)
                reliability = (
                    0.0
                    if denominator <= 1e-12
                    else float(
                        np.clip(
                            np.vdot(
                                delta, target_group - source_group
                            ).real
                            / denominator,
                            0.0,
                            1.0,
                        )
                    )
                )
                outputs["reliability"][
                    batch_index, :, :, p_index, n_index, :
                ] = source_group + reliability * delta
    return outputs, shifts


def audit_transport_ceiling(
    reference_channels: np.ndarray,
    target_channels: np.ndarray,
    layout: AntennaLayout,
    max_h_shift: int = 2,
    max_v_shift: int = 2,
    max_delay_shift: int = 8,
    batch_size: int = 2,
) -> dict[str, object]:
    """Measure whether a small shift/gain family can explain target channels."""

    reference = np.asarray(reference_channels)
    target = np.asarray(target_channels)
    if (
        reference.shape != target.shape
        or reference.ndim != 4
        or tuple(reference.shape[1:]) != layout.config.channel_shape
        or not np.issubdtype(reference.dtype, np.complexfloating)
        or not np.issubdtype(target.dtype, np.complexfloating)
    ):
        raise ValueError("reference and target channels must match the layout")
    if not np.isfinite(reference).all() or not np.isfinite(target).all():
        raise ValueError("reference and target channels must be finite")
    for name, value in (
        ("max_h_shift", max_h_shift),
        ("max_v_shift", max_v_shift),
        ("max_delay_shift", max_delay_shift),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    accumulators = {
        name: MetricAccumulator(layout, layout.config.weights)
        for name in _STAGES
    }
    shifts: list[tuple[int, int, int]] = []
    for start in tqdm(
        range(0, len(reference), batch_size),
        desc="O2 transport ceiling",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    ):
        stop = min(len(reference), start + batch_size)
        reference_batch = reference[start:stop].astype(
            np.complex64, copy=False
        )
        target_batch = target[start:stop].astype(np.complex64, copy=False)
        reference_bd = beam_delay(reference_batch, layout)
        target_bd = beam_delay(target_batch, layout)
        outputs, batch_shifts = _oracle_stages(
            reference_bd,
            target_bd,
            max_h_shift,
            max_v_shift,
            max_delay_shift,
        )
        shifts.extend(batch_shifts)
        for name, transformed in outputs.items():
            prediction = inverse_beam_delay(transformed, layout)
            accumulators[name].update(prediction, target_batch)

    metrics = {
        name: accumulator.compute().to_dict()
        for name, accumulator in accumulators.items()
    }
    identity_score = float(metrics["identity"]["score"])
    previous = identity_score
    stages: dict[str, dict[str, float]] = {}
    for name in _STAGES:
        current = float(metrics[name]["score"])
        stages[name] = {
            **metrics[name],
            "gain_vs_identity": current - identity_score,
            "marginal_gain": current - previous,
        }
        previous = current
    shift_array = (
        np.asarray(shifts, dtype=np.float64)
        if shifts
        else np.zeros((0, 3), dtype=np.float64)
    )
    best_gain = max(
        float(values["gain_vs_identity"]) for values in stages.values()
    )
    return {
        "kind": "path_transport_o2_target_visible_ceiling",
        "target_visible": True,
        "deployable_prediction": False,
        "sample_count": int(len(reference)),
        "layout_order": list(layout.order),
        "limits": {
            "max_h_shift": max_h_shift,
            "max_v_shift": max_v_shift,
            "max_delay_shift": max_delay_shift,
        },
        "stages": stages,
        "shift_statistics": {
            "count": int(len(shift_array)),
            "mean": (
                [0.0, 0.0, 0.0]
                if not len(shift_array)
                else shift_array.mean(axis=0).tolist()
            ),
            "absolute_p95": (
                [0.0, 0.0, 0.0]
                if not len(shift_array)
                else np.quantile(np.abs(shift_array), 0.95, axis=0).tolist()
            ),
        },
        "best_gain_vs_identity": best_gain,
        "promoted": bool(best_gain >= 0.015),
    }


def _atomic_json(path: str | Path, value: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _run(args: argparse.Namespace) -> int:
    manifest, dataset, _, _, validation = _load_fold(
        args.cache_dir, args.data_dir
    )
    if args.limit_samples is not None:
        if args.limit_samples <= 0:
            raise ValueError("limit-samples must be positive")
        validation = validation[: args.limit_samples]
    coarse = np.load(args.coarse_validation, mmap_mode="r")
    expected = (len(validation),) + dataset.config.channel_shape
    if coarse.shape != expected or not np.issubdtype(
        coarse.dtype, np.complexfloating
    ):
        raise ValueError(
            f"coarse validation must have shape {expected} and complex dtype"
        )
    target = dataset.channel_batch(validation)
    layout = AntennaLayout(dataset.config, tuple(manifest.layout_order))
    report = audit_transport_ceiling(
        np.asarray(coarse, dtype=np.complex64),
        target,
        layout,
        args.max_h_shift,
        args.max_v_shift,
        args.max_delay_shift,
        args.batch_size,
    )
    report.update(
        {
            "cache_manifest_fingerprint": manifest.fingerprint,
            "adapter_sha256": manifest.adapter_sha256,
            "coarse_validation": str(Path(args.coarse_validation).resolve()),
        }
    )
    _atomic_json(args.output, report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit target-visible low-dimensional transport ceiling"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--coarse-validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-h-shift", type=int, default=2)
    parser.add_argument("--max-v-shift", type=int, default=2)
    parser.add_argument("--max-delay-shift", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit-samples", type=int)
    parser.set_defaults(handler=_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
