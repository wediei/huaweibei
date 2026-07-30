"""Training, checkpointing, diagnostics, and reports for E2E-CGPF."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ..config import RoundConfig
from ..data import RoundDataset
from ..metrics import MetricAccumulator
from ..transforms import AntennaLayout
from .e2e_cgpf import (
    E2ECGPF,
    E2ECGPFConfig,
    PathModes,
    TrainableGaussianField,
    path_diagnostics,
)
from .e2e_cgpf_losses import E2ECGPFLossConfig, e2e_cgpf_loss


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_indices(indices: Sequence[int] | np.ndarray) -> str:
    array = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def write_json(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def write_hash_sidecar(path: str | Path) -> Path:
    source = Path(path)
    sidecar = source.with_name(source.name + ".sha256")
    temporary = sidecar.with_name(sidecar.name + ".tmp")
    temporary.write_text(sha256_file(source) + "\n", encoding="ascii")
    os.replace(temporary, sidecar)
    return sidecar


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ChannelPositionDataset(Dataset[dict[str, torch.Tensor]]):
    """Index a memmapped official channel without caching complete folds."""

    def __init__(self, source: RoundDataset, indices: Sequence[int] | np.ndarray):
        self.source = source
        self.indices = np.asarray(indices, dtype=np.int64)
        if self.indices.ndim != 1 or self.indices.size == 0:
            raise ValueError("dataset indices must be a non-empty vector")
        if int(self.indices.min()) < 0 or int(self.indices.max()) >= len(source.train_pos):
            raise IndexError("dataset index outside the official training arrays")

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        source_index = int(self.indices[item])
        position = np.array(self.source.train_pos[source_index], dtype=np.float32, copy=True)
        channel = np.array(
            self.source.train_channel[source_index], dtype=np.complex64, copy=True
        )
        return {
            "source_index": torch.tensor(source_index, dtype=torch.long),
            "position": torch.from_numpy(position),
            "channel": torch.from_numpy(channel),
        }


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 1
    learning_rate: float = 1e-3
    field_learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    accumulation_steps: int = 1
    num_workers: int = 0
    seed: int = 42
    causal_every_batches: int = 4
    densify_start_epoch: int = 20
    densify_interval: int = 5
    structure_rollback_tolerance: float = 0.02
    device: str = "auto"

    def validate(self) -> None:
        for name in (
            "batch_size",
            "accumulation_steps",
            "causal_every_batches",
            "densify_interval",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_workers < 0 or self.densify_start_epoch < 0:
            raise ValueError("worker and epoch counts must be non-negative")
        for name in (
            "learning_rate",
            "field_learning_rate",
            "gradient_clip_norm",
        ):
            if not math.isfinite(float(getattr(self, name))) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.weight_decay < 0 or self.structure_rollback_tolerance < 0:
            raise ValueError("regularization/tolerance values must be non-negative")


@dataclass(frozen=True)
class StageSpec:
    name: str
    mode: str
    epochs: int
    allow_structure: bool = False

    def validate(self) -> None:
        if not self.name:
            raise ValueError("stage name cannot be empty")
        if self.mode not in {"geometry", "complex", "structural", "full"}:
            raise ValueError("invalid model stage mode")
        if isinstance(self.epochs, bool) or self.epochs <= 0:
            raise ValueError("stage epochs must be positive")


def default_stages(epochs_b: int, epochs_c: int, epochs_d: int) -> list[StageSpec]:
    return [
        StageSpec("B_geometry_power_delay", "geometry", epochs_b, False),
        StageSpec("C_complex_phase_polarization", "complex", epochs_c, False),
        StageSpec("D_joint_field_structure", "structural", epochs_d, True),
    ]


def capacity_stage(epochs: int) -> list[StageSpec]:
    return [StageSpec("A_representation_capacity", "full", epochs, False)]


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; activate the existing SSH environment")
    return device


def make_loader(
    source: RoundDataset,
    indices: Sequence[int] | np.ndarray,
    config: TrainingConfig,
    *,
    shuffle: bool,
) -> DataLoader[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        ChannelPositionDataset(source, indices),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        persistent_workers=config.num_workers > 0,
    )


def build_model(
    source: RoundDataset,
    train_indices: Sequence[int] | np.ndarray,
    config: E2ECGPFConfig,
) -> E2ECGPF:
    config.validate()
    indices = np.asarray(train_indices, dtype=np.int64)
    positions = np.asarray(source.train_pos[indices], dtype=np.float32)
    position_center = positions.mean(axis=0)
    span = float(np.linalg.norm(np.ptp(positions, axis=0)))
    position_scale = np.asarray(max(span, 1.0), dtype=np.float32)
    field = TrainableGaussianField.from_ply(
        source.map_path,
        config.field,
        map_mode=config.map_mode,
        seed=config.seed,
    )
    return E2ECGPF(
        field,
        source.config,
        config,
        torch.from_numpy(position_center),
        torch.from_numpy(position_scale),
    )


def _round_from_dict(values: dict[str, Any]) -> RoundConfig:
    return RoundConfig(
        p_train_declared=int(values["p_train_declared"]),
        p_test=int(values["p_test"]),
        m=int(values["m"]),
        m_h=int(values["m_h"]),
        m_v=int(values["m_v"]),
        m_p=int(values["m_p"]),
        n=int(values["n"]),
        n_h=int(values["n_h"]),
        n_v=int(values["n_v"]),
        n_p=int(values["n_p"]),
        s=int(values["s"]),
        q=int(values["q"]),
        bs_position=tuple(float(v) for v in values["bs_position"]),
        weights=tuple(float(v) for v in values["weights"]),
    )


def _atomic_torch_save(payload: Any, path: str | Path) -> Path:
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
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        write_hash_sidecar(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _verified_torch_load(path: str | Path, map_location: str | torch.device) -> Any:
    source = Path(path)
    sidecar = source.with_name(source.name + ".sha256")
    if not source.is_file() or not sidecar.is_file():
        raise FileNotFoundError("checkpoint and SHA-256 sidecar are both required")
    expected = sidecar.read_text(encoding="ascii").strip()
    actual = sha256_file(source)
    if expected != actual:
        raise ValueError("checkpoint SHA-256 mismatch")
    try:
        return torch.load(source, map_location=map_location, weights_only=False)
    except TypeError:
        # PyTorch 2.0.1 on the required SSH environment predates the current
        # default change but still loads this locally generated, hash-verified
        # resumable payload.
        return torch.load(source, map_location=map_location)


def load_model_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[E2ECGPF, dict[str, Any]]:
    payload = _verified_torch_load(path, map_location=device)
    if payload.get("format") != "e2e-cgpf-v1":
        raise ValueError("unsupported E2E-CGPF checkpoint format")
    config = E2ECGPFConfig.from_dict(payload["model_manifest"]["config"])
    round_config = _round_from_dict(payload["model_manifest"]["round"])
    state = payload["model"]
    centers = state["field.center_prior"][: config.field.initial_count].cpu()
    normals = state["field.normal_prior"][: config.field.initial_count].cpu()
    field = TrainableGaussianField(
        centers,
        normals,
        config.field,
        map_mode=config.map_mode,
        seed=config.seed,
    )
    model = E2ECGPF(
        field,
        round_config,
        config,
        torch.tensor(payload["model_manifest"]["position_center"]),
        torch.tensor(payload["model_manifest"]["position_scale"]),
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model, payload


class E2ECGPFTrainer:
    def __init__(
        self,
        model: E2ECGPF,
        training_config: TrainingConfig,
        loss_config: E2ECGPFLossConfig,
        output_dir: str | Path,
        *,
        train_indices: Sequence[int] | np.ndarray,
        validation_indices: Sequence[int] | np.ndarray,
    ) -> None:
        training_config.validate()
        loss_config.validate()
        self.model = model
        self.training_config = training_config
        self.loss_config = loss_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(training_config.device)
        self.model.to(self.device)
        self.train_indices = np.asarray(train_indices, dtype=np.int64)
        self.validation_indices = np.asarray(validation_indices, dtype=np.int64)
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": list(self.model.field.parameters()),
                    "lr": training_config.field_learning_rate,
                },
                {
                    "params": list(self.model.path_network.parameters()),
                    "lr": training_config.learning_rate,
                },
            ],
            weight_decay=training_config.weight_decay,
        )
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.epoch = -1
        self.best_score = -math.inf
        self.best_metrics: dict[str, Any] = {}
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.process_log = self.output_dir / "process.log"
        self.field_events = self.output_dir / "field_events.jsonl"

    def _log_text(self, message: str) -> None:
        timestamped = time.strftime("%Y-%m-%d %H:%M:%S") + " " + message
        with self.process_log.open("a", encoding="utf-8") as handle:
            handle.write(timestamped + "\n")
        print(message, flush=True)

    def _log_metric(self, record: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _checkpoint_payload(self, stage: StageSpec) -> dict[str, Any]:
        return {
            "format": "e2e-cgpf-v1",
            "model_manifest": self.model.model_manifest(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
            "epoch": self.epoch,
            "stage": dataclasses.asdict(stage),
            "best_score": self.best_score,
            "best_metrics": self.best_metrics,
            "training_config": dataclasses.asdict(self.training_config),
            "loss_config": dataclasses.asdict(self.loss_config),
            "train_indices": self.train_indices,
            "validation_indices": self.validation_indices,
            "train_indices_sha256": sha256_indices(self.train_indices),
            "validation_indices_sha256": sha256_indices(self.validation_indices),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
            },
        }

    def save_checkpoint(self, name: str, stage: StageSpec) -> Path:
        return _atomic_torch_save(
            self._checkpoint_payload(stage), self.output_dir / name
        )

    def _move(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            name: value.to(
                self.device,
                non_blocking=self.device.type == "cuda",
            )
            for name, value in batch.items()
        }

    def _causal_predictions(
        self, positions: torch.Tensor, batch_index: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if (
            self.loss_config.causal_weight <= 0.0
            or self.model.config.map_mode != "real"
            or batch_index % self.training_config.causal_every_batches != 0
        ):
            return None, None
        zero, _ = self.model(positions, map_view="zero")
        shuffle, _ = self.model(positions, map_view="shuffle")
        return zero, shuffle

    def train_epoch(
        self,
        loader: Iterable[dict[str, torch.Tensor]],
        stage: StageSpec,
        *,
        epoch_in_stage: int,
    ) -> dict[str, float]:
        self.model.train()
        self.model.set_stage(stage.mode)
        self.optimizer.zero_grad(set_to_none=True)
        running: dict[str, float] = {}
        batch_count = 0
        window = 0
        progress = tqdm(
            loader,
            desc=f"{stage.name} epoch {epoch_in_stage + 1}/{stage.epochs}",
            leave=True,
            dynamic_ncols=True,
        )
        for batch_index, source_batch in enumerate(progress):
            batch = self._move(source_batch)
            prediction, paths = self.model(batch["position"])
            zero_prediction, shuffle_prediction = self._causal_predictions(
                batch["position"], batch_index
            )
            values, _ = e2e_cgpf_loss(
                prediction,
                batch["channel"],
                paths,
                self.model.field,
                self.model.round_config,
                self.model.config.antenna_order,
                self.loss_config,
                zero_prediction=zero_prediction,
                shuffle_prediction=shuffle_prediction,
            )
            if not torch.isfinite(values.total):
                source = batch["source_index"].detach().cpu().tolist()
                raise FloatingPointError(
                    f"non-finite loss for source indices {source}: {values.scalars()}"
                )
            values.total.backward()
            self.model.field.record_statistics(
                paths.gaussian_indices, paths.gate, values.complex
            )
            window += 1
            batch_count += 1
            scalar_values = values.scalars()
            for name, value in scalar_values.items():
                running[name] = running.get(name, 0.0) + value
            self._log_metric(
                {
                    "kind": "batch",
                    "epoch": self.epoch + 1,
                    "stage": stage.name,
                    "batch": batch_index,
                    "source_indices": batch["source_index"].detach().cpu().tolist(),
                    **scalar_values,
                }
            )
            if window == self.training_config.accumulation_steps:
                self._optimizer_step(window)
                window = 0
            progress.set_postfix(
                loss=f"{running['total'] / batch_count:.4f}",
                score=f"{running['score'] / batch_count:.4f}",
                paths=f"{float((paths.gate > 1e-4).sum(dim=1).float().mean()):.1f}",
            )
        if batch_count == 0:
            raise ValueError("training loader yielded no batches")
        if window:
            self._optimizer_step(window)
        return {name: value / batch_count for name, value in running.items()}

    def _optimizer_step(self, window: int) -> None:
        if window <= 0:
            raise ValueError("gradient window must be positive")
        if window != 1:
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(window)
        finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in self.model.parameters()
        )
        if not finite:
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("non-finite model gradient")
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.training_config.gradient_clip_norm
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def validate(
        self,
        loader: Iterable[dict[str, torch.Tensor]],
        *,
        causal_views: bool = True,
    ) -> dict[str, Any]:
        self.model.eval()
        layout = AntennaLayout(
            self.model.round_config, self.model.config.antenna_order
        )
        accumulator = MetricAccumulator(
            layout, weights=self.model.round_config.weights
        )
        zero_accumulator = MetricAccumulator(
            layout, weights=self.model.round_config.weights
        )
        shuffle_accumulator = MetricAccumulator(
            layout, weights=self.model.round_config.weights
        )
        diagnostics: list[dict[str, Any]] = []
        per_target_path_energy: list[float] = []
        per_target_delay_mean: list[float] = []
        per_target_angle_mean: list[float] = []
        path_state_signatures: set[tuple[int, ...]] = set()
        output_energy = 0.0
        target_energy = 0.0
        count = 0
        progress = tqdm(loader, desc="validation", leave=False, dynamic_ncols=True)
        for source_batch in progress:
            batch = self._move(source_batch)
            prediction, paths = self.model(batch["position"])
            if not torch.isfinite(prediction.real).all() or not torch.isfinite(
                prediction.imag
            ).all():
                raise FloatingPointError("non-finite validation channel")
            target = batch["channel"]
            pred_numpy = prediction.detach().cpu().numpy().astype(
                np.complex64, copy=False
            )
            target_numpy = target.detach().cpu().numpy().astype(
                np.complex64, copy=False
            )
            accumulator.update(pred_numpy, target_numpy)
            if causal_views and self.model.config.map_mode == "real":
                zero, _ = self.model(batch["position"], map_view="zero")
                shuffle, _ = self.model(batch["position"], map_view="shuffle")
                zero_accumulator.update(
                    zero.detach().cpu().numpy().astype(np.complex64, copy=False),
                    target_numpy,
                )
                shuffle_accumulator.update(
                    shuffle.detach().cpu().numpy().astype(np.complex64, copy=False),
                    target_numpy,
                )
            diagnostics.append(path_diagnostics(paths, self.model.field))
            batch_path_energy = (
                paths.complex_gain.detach().abs().square()
                * paths.gate.detach().square()
            ).sum(dim=1)
            per_target_path_energy.extend(
                float(value) for value in batch_path_energy.cpu().tolist()
            )
            per_target_delay_mean.extend(
                float(value)
                for value in paths.delay_bins.detach().mean(dim=1).cpu().tolist()
            )
            angle_mean = torch.cat((paths.aod.detach(), paths.aoa.detach()), dim=-1)
            per_target_angle_mean.extend(
                float(value) for value in angle_mean.mean(dim=(1, 2)).cpu().tolist()
            )
            for row in paths.gaussian_indices.detach().cpu():
                path_state_signatures.add(tuple(int(value) for value in row.tolist()))
            output_energy += float(prediction.abs().square().sum().cpu())
            target_energy += float(target.abs().square().sum().cpu())
            count += int(target.shape[0])
        if count == 0:
            raise ValueError("validation loader yielded no batches")
        metrics = accumulator.compute().to_dict()
        if causal_views and self.model.config.map_mode == "real":
            zero_metrics = zero_accumulator.compute().to_dict()
            shuffle_metrics = shuffle_accumulator.compute().to_dict()
            metrics.update(
                {
                    "zero_score": zero_metrics["score"],
                    "shuffle_score": shuffle_metrics["score"],
                    "real_over_zero": metrics["score"] - zero_metrics["score"],
                    "real_over_shuffle": metrics["score"]
                    - shuffle_metrics["score"],
                }
            )
        active_path_values = [item["active_paths_mean"] for item in diagnostics]
        metrics.update(
            {
                "samples": count,
                "prediction_energy_ratio": output_energy / max(target_energy, 1e-30),
                "active_paths_mean": float(np.mean(active_path_values)),
                "active_paths_min": int(
                    min(item["active_paths_min"] for item in diagnostics)
                ),
                "target_path_energy_std": float(
                    np.std(np.asarray(per_target_path_energy, dtype=np.float64))
                ),
                "target_delay_mean_std": float(
                    np.std(np.asarray(per_target_delay_mean, dtype=np.float64))
                ),
                "target_angle_mean_std": float(
                    np.std(np.asarray(per_target_angle_mean, dtype=np.float64))
                ),
                "target_path_state_unique": len(path_state_signatures),
                "field_active": self.model.field.active_count,
                "field_capacity": self.model.field.config.max_count,
                "diagnostics": diagnostics[-1],
            }
        )
        return metrics

    def _maybe_edit_structure(
        self,
        stage: StageSpec,
        validation_loader: Iterable[dict[str, torch.Tensor]],
        previous_score: float,
    ) -> dict[str, Any] | None:
        if (
            not stage.allow_structure
            or self.epoch < self.training_config.densify_start_epoch
            or (self.epoch + 1) % self.training_config.densify_interval != 0
        ):
            return None
        snapshot = self.model.field.structure_snapshot()
        snapshot_path = self.output_dir / "field_snapshots" / f"epoch_{self.epoch + 1:04d}.pt"
        _atomic_torch_save(snapshot, snapshot_path)
        edit = self.model.field.densify_prune(self.epoch + 1)
        rolled_back = False
        post_metrics: dict[str, Any] | None = None
        if edit.active_after != edit.active_before:
            try:
                post_metrics = self.validate(validation_loader, causal_views=False)
                score = float(post_metrics["score"])
                if not math.isfinite(score) or score < (
                    previous_score
                    - self.training_config.structure_rollback_tolerance
                ):
                    self.model.field.restore_structure(snapshot)
                    rolled_back = True
            except (FloatingPointError, RuntimeError, ValueError):
                self.model.field.restore_structure(snapshot)
                rolled_back = True
        event = {
            "epoch": self.epoch + 1,
            **edit.to_dict(),
            "rolled_back": rolled_back,
            "snapshot": str(snapshot_path),
            "snapshot_sha256": sha256_file(snapshot_path),
            "post_edit_score": None
            if post_metrics is None
            else post_metrics.get("score"),
            "post_edit_metrics": post_metrics,
        }
        with self.field_events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        return event

    def fit(
        self,
        train_loader: Iterable[dict[str, torch.Tensor]],
        validation_loader: Iterable[dict[str, torch.Tensor]],
        stages: Sequence[StageSpec],
    ) -> dict[str, Any]:
        if not stages:
            raise ValueError("at least one training stage is required")
        for stage in stages:
            stage.validate()
        total_epochs = sum(stage.epochs for stage in stages)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, total_epochs)
        )
        write_json(
            self.output_dir / "config.json",
            {
                "model": self.model.model_manifest(),
                "training": dataclasses.asdict(self.training_config),
                "loss": dataclasses.asdict(self.loss_config),
                "stages": [dataclasses.asdict(stage) for stage in stages],
                "train_indices_sha256": sha256_indices(self.train_indices),
                "validation_indices_sha256": sha256_indices(
                    self.validation_indices
                ),
                "train_count": int(self.train_indices.size),
                "validation_count": int(self.validation_indices.size),
            },
        )
        overall = tqdm(
            total=total_epochs, desc="E2E-CGPF stages", leave=True, dynamic_ncols=True
        )
        try:
            for stage in stages:
                self._log_text(
                    f"stage_start name={stage.name} mode={stage.mode} epochs={stage.epochs}"
                )
                for epoch_in_stage in range(stage.epochs):
                    self.epoch += 1
                    train_metrics = self.train_epoch(
                        train_loader, stage, epoch_in_stage=epoch_in_stage
                    )
                    validation = self.validate(
                        validation_loader,
                        causal_views=self.loss_config.causal_weight > 0.0,
                    )
                    structure_event = self._maybe_edit_structure(
                        stage,
                        validation_loader,
                        float(validation["score"]),
                    )
                    if (
                        structure_event is not None
                        and not structure_event["rolled_back"]
                        and structure_event["post_edit_metrics"] is not None
                    ):
                        validation = structure_event["post_edit_metrics"]
                    score = float(validation["score"])
                    if score > self.best_score:
                        self.best_score = score
                        self.best_metrics = dict(validation)
                        self.save_checkpoint("best.pt", stage)
                    self.scheduler.step()
                    self.save_checkpoint("last.pt", stage)
                    record = {
                        "kind": "epoch",
                        "epoch": self.epoch,
                        "stage": stage.name,
                        "train": train_metrics,
                        "validation": validation,
                        "best_score": self.best_score,
                        "learning_rates": [
                            group["lr"] for group in self.optimizer.param_groups
                        ],
                        "structure_event": structure_event,
                    }
                    self._log_metric(record)
                    self._log_text(
                        f"epoch={self.epoch + 1} stage={stage.name} "
                        f"score={score:.6f} pas={validation['pas']:.6f} "
                        f"pdp={validation['pdp']:.6f} nmse={validation['nmse']:.6f} "
                        f"best={self.best_score:.6f} field={self.model.field.active_count}"
                    )
                    overall.update(1)
                    overall.set_postfix(score=f"{score:.4f}", best=f"{self.best_score:.4f}")
                write_json(
                    self.output_dir / f"{stage.name}_report.json",
                    {
                        "stage": dataclasses.asdict(stage),
                        "epoch": self.epoch,
                        "best_score": self.best_score,
                        "best_metrics": self.best_metrics,
                        "last_metrics": validation,
                    },
                )
        finally:
            overall.close()
        return {
            "best_score": self.best_score,
            "best_metrics": self.best_metrics,
            "last_metrics": validation,
            "epochs_completed": self.epoch + 1,
            "best_checkpoint": str(self.output_dir / "best.pt"),
            "last_checkpoint": str(self.output_dir / "last.pt"),
        }
