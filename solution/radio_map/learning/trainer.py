"""CPU-safe training, exact streaming validation, and verified checkpoints."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from ..metrics import MetricAccumulator
from .anchor_mixer import SupportAwareAnchorMixerConfig
from .losses import AnchorLossConfig, anchor_completion_loss


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_hash(adapter: object | None) -> str | None:
    if adapter is None:
        return None
    digest = hashlib.sha256()
    layout = getattr(adapter, "layout", None)
    if layout is None:
        raise TypeError("adapter must expose layout")
    digest.update(repr((layout.config, layout.order)).encode("utf-8"))
    for name in ("support_indices", "mean", "rms", "group_ids", "fitted_indices"):
        if not hasattr(adapter, name):
            raise RuntimeError("adapter must be fitted before checkpointing")
        value = np.ascontiguousarray(np.asarray(getattr(adapter, name)))
        digest.update(name.encode("ascii")); digest.update(str(value.dtype).encode("ascii")); digest.update(value.tobytes())
    return digest.hexdigest()


_CHECKPOINT_REQUIRED = {
    "format_version", "model", "optimizer", "scheduler", "epoch", "best_score",
    "best_metrics", "bad_epochs", "trainer_config", "loss_config", "model_config",
    "group_ids", "adapter_hash", "manifest_hash", "rng",
}


def _load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    """Verify a checkpoint sidecar and deserialize only into CPU memory."""

    try:
        source = Path(path)
        sidecar = source.with_name(source.name + ".sha256")
        if not source.is_file() or not sidecar.is_file():
            raise ValueError("checkpoint and its SHA-256 sidecar are required")
        expected = sidecar.read_text(encoding="ascii").strip()
        actual = _sha256_file(source)
        if len(expected) != 64 or expected != actual:
            raise ValueError("checkpoint SHA-256 mismatch")
        payload = torch.load(source, weights_only=False, map_location="cpu")
    except Exception as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError("invalid checkpoint payload") from error
    _validate_checkpoint_payload(payload)
    return dict(payload)


def _validate_checkpoint_payload(payload: object) -> None:
    """Reject malformed checkpoint structure before model_factory is invoked."""

    try:
        if type(payload) is not dict or set(payload) != _CHECKPOINT_REQUIRED:
            raise ValueError("invalid checkpoint payload")
        if type(payload["format_version"]) is not int or payload["format_version"] != 1:
            raise ValueError("unsupported checkpoint format")
        if not all(isinstance(payload[name], Mapping) for name in ("model", "optimizer", "scheduler")):
            raise ValueError("invalid checkpoint payload")
        if type(payload["epoch"]) is not int or payload["epoch"] < -1:
            raise ValueError("invalid checkpoint payload")
        if type(payload["bad_epochs"]) is not int or payload["bad_epochs"] < 0:
            raise ValueError("invalid checkpoint payload")
        best_score = payload["best_score"]
        if isinstance(best_score, bool) or not isinstance(best_score, Real) or math.isnan(float(best_score)):
            raise ValueError("invalid checkpoint payload")

        def valid_hash(value: object) -> bool:
            return value is None or (
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdefABCDEF" for character in value)
            )

        if not valid_hash(payload["adapter_hash"]) or not valid_hash(payload["manifest_hash"]):
            raise ValueError("invalid checkpoint payload")

        group_ids = payload["group_ids"]
        if not isinstance(group_ids, (torch.Tensor, np.ndarray)):
            raise ValueError("invalid checkpoint payload")
        ids = torch.as_tensor(group_ids)
        if (
            ids.ndim != 1
            or ids.dtype == torch.bool
            or ids.dtype.is_floating_point
            or torch.is_complex(ids)
            or bool((ids < 0).any())
        ):
            raise ValueError("invalid checkpoint payload")

        best_metrics = payload["best_metrics"]
        trainer_mapping = payload["trainer_config"]
        loss_mapping = payload["loss_config"]
        model_mapping = payload["model_config"]
        if not all(isinstance(value, Mapping) for value in (best_metrics, trainer_mapping, loss_mapping, model_mapping)):
            raise ValueError("invalid checkpoint payload")
        if not all(isinstance(key, str) for mapping in (best_metrics, trainer_mapping, loss_mapping, model_mapping) for key in mapping):
            raise ValueError("invalid checkpoint payload")
        if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) for value in best_metrics.values()):
            raise ValueError("invalid checkpoint payload")
        initial_best_sentinel = (
            float(best_score) == -math.inf
            and payload["epoch"] == -1
            and not best_metrics
            and payload["bad_epochs"] == 0
        )
        if not math.isfinite(float(best_score)) and not initial_best_sentinel:
            raise ValueError("invalid checkpoint payload")
        TrainerConfig(**dict(trainer_mapping))
        AnchorLossConfig(**dict(loss_mapping))
        model_fields = {item.name for item in dataclasses.fields(SupportAwareAnchorMixerConfig)}
        if set(model_mapping) != model_fields or type(model_mapping.get("use_geometry")) is not bool:
            raise ValueError("invalid checkpoint payload")
        mixer_config = SupportAwareAnchorMixerConfig(**dict(model_mapping))
        if mixer_config.latent_size != ids.numel():
            raise ValueError("invalid checkpoint payload")
        if mixer_config.group_count is not None and int(ids.max()) >= mixer_config.group_count:
            raise ValueError("invalid checkpoint payload")

        rng = payload["rng"]
        if not isinstance(rng, Mapping) or set(rng) != {"python", "numpy", "torch", "cuda"}:
            raise ValueError("invalid checkpoint payload")
        random.Random().setstate(rng["python"])
        np.random.RandomState().set_state(rng["numpy"])
        torch_state = rng["torch"]
        if not isinstance(torch_state, torch.Tensor) or torch_state.dtype != torch.uint8 or torch_state.device.type != "cpu" or torch_state.ndim != 1:
            raise ValueError("invalid checkpoint payload")
        torch.Generator(device="cpu").set_state(torch_state)
        cuda_states = rng["cuda"]
        if cuda_states is not None:
            if not isinstance(cuda_states, list):
                raise ValueError("invalid checkpoint payload")
            for state in cuda_states:
                if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.device.type != "cpu" or state.ndim != 1:
                    raise ValueError("invalid checkpoint payload")
    except Exception as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError("invalid checkpoint payload") from error


def _restore_rng(rng: Mapping[str, Any]) -> None:
    """Restore CPU and, where available, CUDA generators from CPU checkpoint data."""

    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"].detach().cpu())
    if torch.cuda.is_available() and rng["cuda"] is not None:
        if not isinstance(rng["cuda"], (list, tuple)):
            raise ValueError("invalid checkpoint CUDA RNG state")
        torch.cuda.set_rng_state_all([torch.as_tensor(state, dtype=torch.uint8).cpu() for state in rng["cuda"]])


def _optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Ensure deserialized optimizer tensors follow their model parameters."""

    def move(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value

    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            state[key] = move(value)


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    accumulation_steps: int = 4
    gradient_clip_norm: float = 1.0
    warmup_epochs: int = 5
    patience: int = 15
    log_path: str | None = None

    def __post_init__(self) -> None:
        for name in ("epochs", "accumulation_steps", "warmup_epochs", "patience"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "weight_decay", "gradient_clip_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite real number")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("optimizer settings must be non-negative, with positive learning rate/clip")
        if self.log_path is not None and not isinstance(self.log_path, str):
            raise ValueError("log_path must be a string or None")


@dataclass
class Trainer:
    """A trainer whose validation never retains the complete fold prediction."""

    model: nn.Module
    adapter: object | None
    config: TrainerConfig = field(default_factory=TrainerConfig)
    device: str | torch.device = "cpu"
    model_config: dict[str, Any] | None = None
    group_ids: torch.Tensor | None = None
    manifest_hash: str | None = None
    loss_config: AnchorLossConfig = field(default_factory=AnchorLossConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.model, nn.Module):
            raise TypeError("model must be a torch module")
        if not isinstance(self.config, TrainerConfig):
            raise TypeError("config must be a TrainerConfig")
        self.device = torch.device(self.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_multiplier)
        self.epoch = -1
        self.best_score = -math.inf
        self.best_metrics: dict[str, float] = {}
        self._bad_epochs = 0
        if self.group_ids is None and hasattr(self.model, "group_ids"):
            self.group_ids = getattr(self.model, "group_ids").detach().cpu().clone()
        elif self.group_ids is not None:
            self.group_ids = torch.as_tensor(self.group_ids, dtype=torch.long).cpu().clone()
        if self.model_config is None and hasattr(self.model, "config"):
            candidate = getattr(self.model, "config")
            self.model_config = dataclasses.asdict(candidate) if dataclasses.is_dataclass(candidate) else dict(candidate)
        self.model_config = dict(self.model_config or {})
        self.adapter_hash = _adapter_hash(self.adapter)

    def _lr_multiplier(self, epoch: int) -> float:
        if epoch < self.config.warmup_epochs:
            return float(epoch + 1) / self.config.warmup_epochs
        span = max(1, self.config.epochs - self.config.warmup_epochs)
        progress = min(1.0, (epoch - self.config.warmup_epochs) / span)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _autocast(self) -> Any:
        # The objective differentiates through complex FFT metrics.  On the
        # target 4090, BF16 autocast can turn otherwise finite full-fold
        # gradients into NaN; this model is small enough that FP32 is the
        # reliable default (smoke used only ~126 MB of VRAM).
        return contextlib.nullcontext()

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(batch, dict):
            raise TypeError("DataLoader must yield dictionaries")
        return {name: value.to(self.device, non_blocking=self.device.type == "cuda") if isinstance(value, torch.Tensor) else value for name, value in batch.items()}

    def _loss_for_batch(self, batch: dict[str, Any]) -> Any:
        if self.adapter is None:
            raise RuntimeError("an adapter is required for training and validation")
        output = self.model(batch)
        return output, anchor_completion_loss(output, batch["target_latent"], batch["target_channel"], self.adapter, self.loss_config)

    @staticmethod
    def _non_finite_value_names(values: Any) -> list[str]:
        return [
            name for name in values.__dataclass_fields__
            if isinstance(getattr(values, name), torch.Tensor)
            and not torch.isfinite(getattr(values, name)).all()
        ]

    @staticmethod
    def _set_dataset_epoch(loader: Iterable[Any], epoch: int) -> None:
        dataset = getattr(loader, "dataset", None)
        setter = getattr(dataset, "set_epoch", None)
        if callable(setter):
            setter(epoch)
        sampler_setter = getattr(getattr(loader, "sampler", None), "set_epoch", None)
        if callable(sampler_setter):
            sampler_setter(epoch)

    def validate(self, loader: Iterable[dict[str, Any]], correction_scale: float = 1.0) -> dict[str, float]:
        """Return exact whole-fold metrics using additive NumPy sufficient statistics."""

        if self.adapter is None:
            raise RuntimeError("an adapter is required for validation")
        if isinstance(correction_scale, bool) or not isinstance(correction_scale, Real) or not math.isfinite(float(correction_scale)) or correction_scale < 0:
            raise ValueError("correction_scale must be a finite non-negative number")
        self.model.eval()
        weights = self.adapter.layout.config.weights
        accumulator = MetricAccumulator(self.adapter.layout, weights=weights)
        nearest_accumulator = MetricAccumulator(self.adapter.layout, weights=weights)
        loss_sum = 0.0
        latent_mse_sum = 0.0
        residual_energy_sum = 0.0
        sample_count = 0
        with torch.no_grad():
            for source_batch in tqdm(loader, desc="validation", leave=False, dynamic_ncols=True):
                batch = self._move_batch(source_batch)
                with self._autocast():
                    output, values = self._loss_for_batch(batch)
                    if correction_scale != 1.0:
                        latent = output.nearest_latent + float(correction_scale) * (
                            output.latent - output.nearest_latent
                        )
                        output = dataclasses.replace(output, latent=latent)
                        values = anchor_completion_loss(
                            output, batch["target_latent"], batch["target_channel"],
                            self.adapter, self.loss_config,
                        )
                if not torch.isfinite(values.total) or not torch.isfinite(output.latent).all():
                    source = batch.get("source_index")
                    identifiers = [] if not isinstance(source, torch.Tensor) else source.detach().cpu().tolist()
                    raise FloatingPointError(f"non-finite validation value for source indices {identifiers}")
                prediction = self.adapter.decode_torch(output.latent)
                nearest = self.adapter.decode_torch(output.nearest_latent)
                target = batch["target_channel"]
                accumulator.update(prediction.detach().cpu().numpy().astype(np.complex64, copy=False), target.detach().cpu().numpy().astype(np.complex64, copy=False))
                nearest_accumulator.update(nearest.detach().cpu().numpy().astype(np.complex64, copy=False), target.detach().cpu().numpy().astype(np.complex64, copy=False))
                count = int(target.shape[0])
                loss_sum += float(values.total.detach().cpu()) * count
                latent_mse_sum += float(values.latent_mse.detach().cpu()) * count
                residual_energy_sum += float(values.residual_energy.detach().cpu()) * count
                sample_count += count
        if sample_count == 0:
            raise ValueError("validation loader yielded no samples")
        metrics, nearest_metrics = accumulator.compute(), nearest_accumulator.compute()
        result = metrics.to_dict()
        result.update({"nearest_score": nearest_metrics.score, "nearest_pas": nearest_metrics.pas, "nearest_pdp": nearest_metrics.pdp, "nearest_nmse": nearest_metrics.nmse, "gain": metrics.score - nearest_metrics.score, "loss": loss_sum / sample_count, "latent_mse": latent_mse_sum / sample_count, "residual_energy": residual_energy_sum / sample_count})
        return result

    def _optimizer_step(self, window_count: int) -> bool:
        """Average the actual accumulation window, clip, step, and clear grads."""

        if window_count < 1:
            raise ValueError("accumulation window must contain at least one batch")
        scale = 1.0 / window_count
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in self.model.parameters()
        ):
            self.optimizer.zero_grad(set_to_none=True)
            return False
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return True

    def fit(self, train_loader: Iterable[dict[str, Any]], validation_loader: Iterable[dict[str, Any]], checkpoint_path: str | Path | None = None, last_checkpoint_path: str | Path | None = None) -> list[dict[str, float]]:
        """Train by microbatches, validate each epoch, and early-stop on exact score."""

        records: list[dict[str, float]] = []
        log_handle = Path(self.config.log_path).open("a", encoding="utf-8") if self.config.log_path else None
        try:
            for epoch in range(self.epoch + 1, self.config.epochs):
                self._set_dataset_epoch(train_loader, epoch)
                self.model.train(); self.optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0; batch_count = 0; window_count = 0; skipped_steps = 0
                progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{self.config.epochs}", leave=True, dynamic_ncols=True)
                for source_batch in progress:
                    batch = self._move_batch(source_batch)
                    with self._autocast():
                        _, values = self._loss_for_batch(batch)
                    if not torch.isfinite(values.total):
                        source = batch.get("source_index")
                        identifiers = [] if not isinstance(source, torch.Tensor) else source.detach().cpu().tolist()
                        names = self._non_finite_value_names(values)
                        raise FloatingPointError(f"non-finite training values {names} for source indices {identifiers}")
                    values.total.backward()
                    total_loss += float(values.total.detach().cpu()); batch_count += 1; window_count += 1
                    progress.set_postfix(loss=f"{total_loss / batch_count:.4f}")
                    if window_count == self.config.accumulation_steps:
                        skipped_steps += int(not self._optimizer_step(window_count))
                        window_count = 0
                if batch_count == 0:
                    raise ValueError("training loader yielded no batches")
                if window_count:
                    skipped_steps += int(not self._optimizer_step(window_count))
                if skipped_steps:
                    print(f"epoch {epoch + 1}: skipped {skipped_steps} non-finite gradient update(s)", flush=True)
                self.scheduler.step()
                self.epoch = epoch
                validation = self.validate(validation_loader)
                improved = validation["score"] > self.best_score
                if improved:
                    self.best_score = validation["score"]; self.best_metrics = dict(validation); self._bad_epochs = 0
                    if checkpoint_path is not None:
                        self.save_checkpoint(checkpoint_path)
                else:
                    self._bad_epochs += 1
                record = {"epoch": float(epoch), "train_loss": total_loss / batch_count, "learning_rate": self.optimizer.param_groups[0]["lr"], **validation, "best_score": self.best_score}
                records.append(record)
                if log_handle is not None:
                    log_handle.write(json.dumps(record, sort_keys=True) + "\n"); log_handle.flush()
                print(
                    f"epoch {epoch + 1}/{self.config.epochs} "
                    f"train_loss={record['train_loss']:.6f} "
                    f"val={record['score']:.6f} "
                    f"pas={record.get('pas', float('nan')):.4f} "
                    f"pdp={record.get('pdp', float('nan')):.4f} "
                    f"nmse={record.get('nmse', float('nan')):.4f} "
                    f"gain={record.get('gain', float('nan')):+.6f} "
                    f"residual={record.get('residual_energy', float('nan')):.6f} "
                    f"lr={record['learning_rate']:.2e} "
                    f"best={self.best_score:.6f}",
                    flush=True,
                )
                if last_checkpoint_path is not None:
                    self.save_checkpoint(last_checkpoint_path)
                if self._bad_epochs >= self.config.patience:
                    break
        finally:
            if log_handle is not None:
                log_handle.close()
        return records

    def _checkpoint_payload(self) -> dict[str, Any]:
        return {
            "format_version": 1, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
            "epoch": self.epoch, "best_score": self.best_score, "best_metrics": self.best_metrics, "bad_epochs": self._bad_epochs,
            "trainer_config": dataclasses.asdict(self.config), "loss_config": dataclasses.asdict(self.loss_config), "model_config": self.model_config,
            "group_ids": self.group_ids, "adapter_hash": self.adapter_hash, "manifest_hash": self.manifest_hash,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        """Atomically save a resumable state then write its streamed SHA-256 sidecar."""

        destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent, delete=False)
        temporary = Path(handle.name); handle.close()
        try:
            torch.save(self._checkpoint_payload(), temporary)
            os.replace(temporary, destination)
            digest = _sha256_file(destination)
            sidecar = destination.with_name(destination.name + ".sha256")
            side_tmp = sidecar.with_name(sidecar.name + ".tmp")
            side_tmp.write_text(digest + "\n", encoding="ascii")
            os.replace(side_tmp, sidecar)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination

    @classmethod
    def load_checkpoint(cls, path: str | Path, model_factory: Callable[[dict[str, Any], torch.Tensor], nn.Module], device: str | torch.device = "cpu", adapter: object | None = None, expected_adapter_hash: str | None = None, expected_manifest_hash: str | None = None) -> "Trainer":
        """Verify the sidecar before deserializing, then restore complete state."""

        payload = _load_checkpoint_payload(path)
        stored_adapter_hash = payload.get("adapter_hash")
        actual_adapter_hash = _adapter_hash(adapter)
        if expected_adapter_hash is not None and stored_adapter_hash != expected_adapter_hash:
            raise ValueError("checkpoint adapter hash mismatch")
        if actual_adapter_hash is not None and stored_adapter_hash != actual_adapter_hash:
            raise ValueError("checkpoint adapter hash mismatch")
        if expected_manifest_hash is not None and payload.get("manifest_hash") != expected_manifest_hash:
            raise ValueError("checkpoint manifest hash mismatch")
        group_ids = torch.as_tensor(payload["group_ids"], dtype=torch.long).cpu()
        model = model_factory(dict(payload["model_config"]), group_ids)
        trainer = cls(model, adapter, TrainerConfig(**payload["trainer_config"]), device=device, model_config=dict(payload["model_config"]), group_ids=group_ids, manifest_hash=payload.get("manifest_hash"), loss_config=AnchorLossConfig(**payload["loss_config"]))
        trainer.model.load_state_dict(payload["model"]); trainer.optimizer.load_state_dict(payload["optimizer"]); _optimizer_state_to_device(trainer.optimizer, trainer.device); trainer.scheduler.load_state_dict(payload["scheduler"])
        trainer.epoch = int(payload["epoch"]); trainer.best_score = float(payload["best_score"]); trainer.best_metrics = dict(payload["best_metrics"]); trainer._bad_epochs = int(payload.get("bad_epochs", 0))
        _restore_rng(payload["rng"])
        return trainer
