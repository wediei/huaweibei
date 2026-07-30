"""Train and infer a compact power-aware refiner on top of a frozen coarse model."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..cli import validate_submission
from ..metrics import MetricAccumulator
from .anchor_mixer import MixerOutput
from .building_feature_cache import BuildingFeatureCache
from .cli import _geometry, _loader, _load_verified_trainer
from .dataset import CachedAnchorDataset, CoordinateBatchContext
from .losses import AnchorLossConfig, anchor_completion_loss


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class PowerAwareRefiner(nn.Module):
    """Learn group-wise amplitude/phase gates from aligned local channel templates."""

    def __init__(
        self,
        base: nn.Module,
        adapter: object,
        hidden_dim: int = 64,
        feature_version: int = 1,
        map_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        if hidden_dim < 8:
            raise ValueError("hidden_dim must be at least 8")
        if feature_version not in (1, 2, 3):
            raise ValueError("feature_version must be 1, 2, or 3")
        if feature_version == 3 and map_feature_dim <= 0:
            raise ValueError("feature_version 3 requires positive map_feature_dim")
        if feature_version < 3 and map_feature_dim != 0:
            raise ValueError("map_feature_dim is only valid for feature_version 3")
        self.feature_version = int(feature_version)
        self.map_feature_dim = int(map_feature_dim)
        self.feature_count = (
            9
            if self.feature_version == 1
            else 11 + (2 * self.map_feature_dim if self.feature_version == 3 else 0)
        )
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()
        group_ids = torch.as_tensor(adapter.group_ids, dtype=torch.long)
        group_count = int(group_ids.max().item()) + 1
        self.group_count = group_count
        self.register_buffer("group_ids", group_ids)
        self.register_buffer("latent_mean", torch.as_tensor(adapter.mean, dtype=torch.complex64))
        self.register_buffer("latent_rms", torch.as_tensor(adapter.rms, dtype=torch.float32))
        embedding_dim = 16
        self.group_embedding = nn.Parameter(torch.randn(group_count, embedding_dim) * 0.02)
        self.gate = nn.Sequential(
            nn.Linear(self.feature_count + embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.gate[-1].weight)
        with torch.no_grad():
            # Preserve the coarse model initially: little phase transport,
            # moderate magnitude-template access, and unit global amplitude.
            self.gate[-1].bias.copy_(torch.tensor((-2.2, -1.1, 0.0)))

    def train(self, mode: bool = True) -> "PowerAwareRefiner":
        super().train(mode)
        self.base.eval()
        return self

    def head_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if not name.startswith("base.")
        }

    def load_head_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected or any(not name.startswith("base.") for name in missing):
            raise ValueError("invalid power-refiner state")

    def _group_mean(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, latent_size = values.shape
        index = self.group_ids.to(values.device).reshape(1, latent_size).expand(batch_size, -1)
        total = torch.zeros(
            batch_size, self.group_count, device=values.device, dtype=values.dtype
        )
        total.scatter_add_(1, index, values)
        count = torch.bincount(
            self.group_ids, minlength=self.group_count
        ).to(device=values.device, dtype=values.dtype)
        return total / count.clamp_min(1)[None, :]

    def forward(self, batch: dict[str, torch.Tensor]) -> MixerOutput:
        with torch.no_grad():
            coarse = self.base(batch)
        anchors = batch["anchor_latents"]
        distances = batch["anchor_distances"]
        mask = batch["anchor_mask"]
        batch_size, anchor_count, latent_size = anchors.shape
        ids = self.group_ids.to(anchors.device)
        # Perform the learned transport in the fold-normalized latent domain.
        # Reconstructing physical coefficients here exposes the gate network to
        # the raw channel's extreme dynamic range and can overflow before the
        # already-stable adapter decoder/loss has a chance to rescale it.
        coarse_raw = coarse.latent
        nearest_raw = coarse.nearest_latent
        anchor_raw = anchors

        # Align every anchor to the coarse phase separately in each
        # polarization/UE/delay group before mixing its fine angular pattern.
        scatter_index = ids.reshape(1, 1, latent_size).expand(
            batch_size, anchor_count, -1
        )
        numerator = torch.zeros(
            batch_size, anchor_count, self.group_count,
            dtype=anchors.dtype, device=anchors.device,
        )
        phase_epsilon = torch.finfo(anchors.real.dtype).eps
        anchor_unit = anchor_raw / anchor_raw.abs().clamp_min(phase_epsilon)
        coarse_unit = coarse_raw / coarse_raw.abs().clamp_min(phase_epsilon)
        numerator.scatter_add_(
            2, scatter_index, anchor_unit.conj() * coarse_unit[:, None, :]
        )
        alignment = numerator / numerator.abs().clamp_min(phase_epsilon)
        aligned = anchor_raw * alignment.index_select(2, ids)

        # Training anchor dropout pads invalid distances with +Inf.  Mask them
        # before every arithmetic operation: a later 0 * Inf is still NaN.
        safe_distance = torch.where(
            mask, distances.clamp_min(1e-3), torch.ones_like(distances)
        )
        weights = coarse.anchor_weights * torch.where(
            mask[:, :, None],
            safe_distance[:, :, None].pow(-2.0),
            torch.zeros_like(safe_distance[:, :, None]),
        )
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        coefficient_weights = weights.index_select(2, ids)
        template = (aligned * coefficient_weights).sum(dim=1)
        template_magnitude = (aligned.abs() * coefficient_weights).sum(dim=1)

        eps = 1e-8
        # Raw channel scales span many orders of magnitude.  Never square them
        # for conditioning: log-amplitude statistics stay finite in FP32 and
        # express the ratios needed by the gates directly.
        coarse_energy = self._group_mean(torch.log1p(coarse_raw.abs()))
        template_energy = self._group_mean(torch.log1p(template.abs()))
        nearest_energy = self._group_mean(torch.log1p(nearest_raw.abs()))
        disagreement = self._group_mean(torch.log1p((template - coarse_raw).abs()))
        entropy = -(weights.clamp_min(eps) * weights.clamp_min(eps).log()).sum(dim=1)
        mean_distance = (weights * safe_distance[:, :, None]).sum(dim=1)
        minimum_distance = safe_distance.masked_fill(~mask, float("inf")).amin(dim=1)
        features = torch.stack(
            (
                coarse_energy,
                template_energy,
                nearest_energy,
                (template_energy - coarse_energy).clamp(-5, 5),
                disagreement,
                entropy / math.log(max(2, anchor_count)),
                torch.log1p(mean_distance),
                torch.log1p(minimum_distance)[:, None].expand(-1, self.group_count),
                coarse.alpha.detach(),
            ),
            dim=-1,
        )
        if self.feature_version >= 2:
            physical_anchor = (
                anchor_raw * self.latent_rms[None, None, :]
                + self.latent_mean[None, None, :]
            )
            anchor_log_energy = torch.log1p(physical_anchor.abs())
            group_index = ids.reshape(1, 1, latent_size).expand(
                batch_size, anchor_count, -1
            )
            anchor_group_sum = torch.zeros(
                batch_size,
                anchor_count,
                self.group_count,
                dtype=anchor_log_energy.dtype,
                device=anchor_log_energy.device,
            )
            anchor_group_sum.scatter_add_(2, group_index, anchor_log_energy)
            group_size = torch.bincount(
                ids, minlength=self.group_count
            ).to(device=anchors.device, dtype=anchor_log_energy.dtype)
            anchor_group_energy = anchor_group_sum / group_size.clamp_min(1)[None, None, :]
            reliability_weights = weights.detach()
            energy_mean = (reliability_weights * anchor_group_energy).sum(dim=1)
            energy_variance = (
                reliability_weights
                * (anchor_group_energy - energy_mean[:, None, :]).square()
            ).sum(dim=1)
            features = torch.cat(
                (features, energy_mean[..., None], torch.sqrt(energy_variance + eps)[..., None]),
                dim=-1,
            )
        if self.feature_version >= 3:
            map_features = batch.get("building_features")
            anchor_map_features = batch.get("anchor_building_features")
            if (
                not isinstance(map_features, torch.Tensor)
                or map_features.shape != (batch_size, self.map_feature_dim)
                or not torch.isfinite(map_features).all()
                or not isinstance(anchor_map_features, torch.Tensor)
                or anchor_map_features.shape
                != (batch_size, anchor_count, self.map_feature_dim)
                or not torch.isfinite(anchor_map_features).all()
            ):
                raise ValueError(
                    "feature_version 3 requires finite target/anchor building "
                    "features with matching map_feature_dim"
                )
            map_features = map_features.to(
                device=features.device, dtype=features.dtype
            )
            anchor_map_features = anchor_map_features.to(
                device=features.device, dtype=features.dtype
            )
            weighted_anchor_map = torch.einsum(
                "bkg,bkf->bgf", weights.detach(), anchor_map_features
            )
            map_target = map_features[:, None, :].expand(
                -1, self.group_count, -1
            )
            features = torch.cat(
                (
                    features,
                    map_target,
                    map_target - weighted_anchor_map,
                ),
                dim=-1,
            )
        embedding = self.group_embedding[None, :, :].expand(batch_size, -1, -1)
        controls = self.gate(torch.cat((features, embedding), dim=-1))
        phase_gate = 0.5 * torch.sigmoid(controls[..., 0])
        magnitude_gate = torch.sigmoid(controls[..., 1])
        amplitude_gain = torch.exp(0.35 * torch.tanh(controls[..., 2]))

        phase_blend = coarse_raw + phase_gate.index_select(1, ids) * (
            template - coarse_raw
        )
        phase_unit = phase_blend / phase_blend.abs().clamp_min(eps)
        magnitude = (
            (1.0 - magnitude_gate.index_select(1, ids)) * phase_blend.abs()
            + magnitude_gate.index_select(1, ids) * template_magnitude
        )
        refined_raw = (
            phase_unit
            * magnitude
            * amplitude_gain.index_select(1, ids)
        )
        refined = refined_raw
        return dataclasses.replace(
            coarse,
            latent=refined.to(dtype=coarse.latent.dtype),
            # Reuse this inspectable field for the exact refiner delta.  The
            # generic loss does not consume it, and inference/evaluation can
            # now sweep correction strength without another base forward.
            low_rank_residual=refined - coarse.latent,
        )


def _validate(refiner, loader, trainer, adapter, loss_config):
    refiner.eval()
    accumulator = MetricAccumulator(adapter.layout)
    loss_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for source in tqdm(loader, desc="power validation", leave=False, dynamic_ncols=True):
            batch = trainer._move_batch(source)
            output = refiner(batch)
            loss = anchor_completion_loss(
                output, batch["target_latent"], batch["target_channel"],
                adapter, loss_config,
            )
            prediction = adapter.decode_torch(output.latent)
            target = batch["target_channel"]
            accumulator.update(
                prediction.cpu().numpy().astype(np.complex64, copy=False),
                target.cpu().numpy().astype(np.complex64, copy=False),
            )
            count = int(target.shape[0])
            loss_sum += float(loss.total.cpu()) * count
            sample_count += count
    result = accumulator.compute().to_dict()
    result["loss"] = loss_sum / sample_count
    return result


class _BuildingFeatureDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        features: np.ndarray,
        anchor_feature_table: np.ndarray,
    ) -> None:
        self.base = base
        self.features = features
        self.anchor_feature_table = anchor_feature_table
        if len(base) != len(features):
            raise ValueError("base dataset and building features differ in length")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        item = dict(self.base[index])
        item["building_features"] = torch.as_tensor(
            np.asarray(self.features[index]), dtype=torch.float32
        )
        anchor_indices = np.asarray(item["anchor_indices"], dtype=np.int64)
        item["anchor_building_features"] = torch.as_tensor(
            np.asarray(self.anchor_feature_table[anchor_indices]),
            dtype=torch.float32,
        )
        return item

    def set_epoch(self, epoch: int) -> None:
        setter = getattr(self.base, "set_epoch", None)
        if callable(setter):
            setter(epoch)


def _load_building_features(args, dataset, manifest):
    if args.feature_version < 3:
        return None
    if not args.building_feature_cache:
        raise ValueError(
            "--building-feature-cache is required with --feature-version 3"
        )
    return BuildingFeatureCache.load(
        args.building_feature_cache,
        dataset=dataset,
        fold_manifest_fingerprint=manifest.fingerprint,
    )


def _attach_building_dataset(base, cache, source_indices, mode, seed):
    if cache is None:
        return base
    all_features = cache.values("train", mode=mode, seed=seed)
    return _BuildingFeatureDataset(
        base,
        np.asarray(all_features[source_indices]),
        np.asarray(all_features),
    )


def _adapter_sha256(adapter: object) -> str:
    digest = hashlib.sha256()
    for name in ("support_indices", "mean", "rms", "group_ids", "fitted_indices"):
        value = np.asarray(getattr(adapter, name))
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(value).view(np.uint8))
    return digest.hexdigest()


def _finite_state(state: dict[str, torch.Tensor]) -> bool:
    return all(
        isinstance(value, torch.Tensor)
        and (
            torch.isfinite(value.real).all().item()
            and (not torch.is_complex(value) or torch.isfinite(value.imag).all().item())
        )
        for value in state.values()
    )


def _save_refiner(
    path,
    refiner,
    args,
    base_hash,
    adapter_hash,
    manifest_fingerprint,
    building_feature_fingerprint,
    metrics,
):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = refiner.head_state_dict()
    if not _finite_state(state) or not all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in metrics.values()
    ):
        raise ValueError("refusing to save non-finite power-refiner state or metrics")
    payload = {
        "format_version": 2,
        "base_checkpoint_sha256": base_hash,
        "adapter_sha256": adapter_hash,
        "cache_manifest_fingerprint": manifest_fingerprint,
        "building_feature_cache_fingerprint": building_feature_fingerprint,
        "config": {
            "hidden_dim": int(args.hidden_dim),
            "feature_version": int(args.feature_version),
            "feature_count": int(refiner.feature_count),
            "map_feature_dim": int(refiner.map_feature_dim),
        },
        "state": state,
        "metrics": dict(metrics),
    }
    handle = tempfile.NamedTemporaryFile(
        prefix=destination.name + ".", suffix=".tmp",
        dir=destination.parent, delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        destination.with_name(destination.name + ".sha256").write_text(
            _sha256(destination) + "\n", encoding="ascii"
        )
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_refiner(
    path,
    refiner,
    base_hash,
    adapter_hash: str | None = None,
    manifest_fingerprint: str | None = None,
    building_feature_fingerprint: str | None = None,
):
    source = Path(path)
    sidecar = source.with_name(source.name + ".sha256")
    if not source.is_file() or not sidecar.is_file() or sidecar.read_text(
        encoding="ascii"
    ).strip() != _sha256(source):
        raise ValueError("invalid power-refiner checkpoint hash")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if type(payload) is not dict or payload.get("format_version") not in (1, 2):
        raise ValueError("invalid power-refiner checkpoint")
    if (
        payload.get("base_checkpoint_sha256") != base_hash
        or type(payload.get("state")) is not dict
        or not _finite_state(payload["state"])
    ):
        raise ValueError("invalid power-refiner checkpoint")
    if payload["format_version"] == 1:
        if refiner.feature_version != 1:
            raise ValueError("legacy checkpoint requires feature_version=1")
    else:
        config = payload.get("config")
        expected = {
            "hidden_dim": int(refiner.gate[0].out_features),
            "feature_version": int(refiner.feature_version),
            "feature_count": int(refiner.feature_count),
            "map_feature_dim": int(refiner.map_feature_dim),
        }
        if config != expected:
            raise ValueError("power-refiner checkpoint configuration differs")
        if adapter_hash is not None and payload.get("adapter_sha256") != adapter_hash:
            raise ValueError("power-refiner adapter hash differs")
        if (
            manifest_fingerprint is not None
            and payload.get("cache_manifest_fingerprint") != manifest_fingerprint
        ):
            raise ValueError("power-refiner cache manifest differs")
        if (
            building_feature_fingerprint is not None
            and payload.get("building_feature_cache_fingerprint")
            != building_feature_fingerprint
        ):
            raise ValueError("power-refiner building feature cache differs")
    refiner.load_head_state_dict(payload["state"])
    return payload


def _run_train(args):
    trainer, manifest, dataset, adapter, train, validation = _load_verified_trainer(args)
    building_cache = _load_building_features(args, dataset, manifest)
    map_dim = 0 if building_cache is None else len(building_cache.feature_names)
    refiner = PowerAwareRefiner(
        trainer.model, adapter, args.hidden_dim, args.feature_version, map_dim
    ).to(trainer.device)
    train_set = CachedAnchorDataset(
        args.cache_dir, train, True, bool(trainer.model.config.use_geometry)
    )
    validation_set = CachedAnchorDataset(
        args.cache_dir, validation, False, bool(trainer.model.config.use_geometry)
    )
    train_set = _attach_building_dataset(
        train_set, building_cache, train, args.map_mode, args.seed
    )
    validation_set = _attach_building_dataset(
        validation_set, building_cache, validation, args.map_mode, args.seed
    )
    loss_config = AnchorLossConfig(
        latent_weight=args.latent_weight,
        nearest_weight=args.nearest_weight,
        nmse_objective=args.nmse_objective,
    )
    parameters = [value for name, value in refiner.named_parameters() if not name.startswith("base.")]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best.pt"
    if checkpoint.exists():
        raise ValueError("power-refiner run directory already contains best.pt")
    log_path = run_dir / "metrics.jsonl"
    best_score = -math.inf
    bad_epochs = 0
    records = []
    started = time.perf_counter()
    base_hash = _sha256(args.checkpoint)
    adapter_hash = _adapter_sha256(adapter)
    base_metrics = trainer.validate(
        _loader(validation_set, args.batch_size, False, args.seed)
    )
    print(
        f"frozen coarse: score={base_metrics['score']:.6f} "
        f"pas={base_metrics['pas']:.4f} pdp={base_metrics['pdp']:.4f} "
        f"nmse={base_metrics['nmse']:.4f}",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(args.epochs):
            refiner.train()
            total = 0.0
            count = 0
            progress = tqdm(
                _loader(train_set, args.batch_size, True, args.seed + epoch),
                desc=f"power epoch {epoch + 1}/{args.epochs}",
                dynamic_ncols=True,
            )
            for source in progress:
                batch = trainer._move_batch(source)
                optimizer.zero_grad(set_to_none=True)
                output = refiner(batch)
                values = anchor_completion_loss(
                    output, batch["target_latent"], batch["target_channel"],
                    adapter, loss_config,
                )
                if not torch.isfinite(values.total):
                    diagnostics = values.detached_dict()
                    diagnostics["latent_finite"] = bool(
                        torch.isfinite(output.latent).all().item()
                    )
                    diagnostics["latent_abs_max"] = float(
                        torch.nan_to_num(output.latent.abs()).amax().detach().cpu()
                    )
                    raise FloatingPointError(
                        "non-finite power-refiner loss: "
                        + json.dumps(diagnostics, sort_keys=True)
                    )
                values.total.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                total += float(values.total.detach().cpu())
                count += 1
                progress.set_postfix(loss=f"{total / count:.4f}")
            metrics = _validate(
                refiner,
                _loader(validation_set, args.batch_size, False, args.seed),
                trainer, adapter, loss_config,
            )
            record = {
                "epoch": epoch + 1,
                "train_loss": total / count,
                **metrics,
                "best_score": max(best_score, metrics["score"]),
            }
            records.append(record)
            log.write(json.dumps(record, sort_keys=True) + "\n")
            log.flush()
            improved = metrics["score"] > best_score
            if improved:
                best_score = metrics["score"]
                bad_epochs = 0
                _save_refiner(
                    checkpoint,
                    refiner,
                    args,
                    base_hash,
                    adapter_hash,
                    manifest.fingerprint,
                    None if building_cache is None else building_cache.fingerprint,
                    metrics,
                )
            else:
                bad_epochs += 1
            print(
                f"power epoch {epoch + 1}/{args.epochs} "
                f"train={record['train_loss']:.6f} val={metrics['score']:.6f} "
                f"pas={metrics['pas']:.4f} pdp={metrics['pdp']:.4f} "
                f"nmse={metrics['nmse']:.4f} best={best_score:.6f}",
                flush=True,
            )
            if bad_epochs >= args.patience:
                break
    best_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    report = {
        "base_checkpoint": str(Path(args.checkpoint).resolve()),
        "base_checkpoint_sha256": base_hash,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "cache_manifest_fingerprint": manifest.fingerprint,
        "best_metrics": best_payload["metrics"],
        "coarse_metrics": base_metrics,
        "gain_vs_coarse": float(
            best_payload["metrics"]["score"] - base_metrics["score"]
        ),
        "epochs_completed": len(records),
        "runtime_seconds": time.perf_counter() - started,
        "config": {
            "hidden_dim": args.hidden_dim,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "latent_weight": args.latent_weight,
            "nearest_weight": args.nearest_weight,
            "nmse_objective": args.nmse_objective,
            "feature_version": args.feature_version,
            "feature_count": refiner.feature_count,
            "map_feature_dim": refiner.map_feature_dim,
            "map_mode": args.map_mode,
            "building_feature_cache_fingerprint": (
                None if building_cache is None else building_cache.fingerprint
            ),
        },
    }
    (run_dir / "train_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


def _run_infer(args):
    trainer, manifest, dataset, adapter, train, _ = _load_verified_trainer(args)
    building_cache = _load_building_features(args, dataset, manifest)
    map_dim = 0 if building_cache is None else len(building_cache.feature_names)
    refiner = PowerAwareRefiner(
        trainer.model, adapter, args.hidden_dim, args.feature_version, map_dim
    ).to(trainer.device)
    _load_refiner(
        args.refiner_checkpoint,
        refiner,
        _sha256(args.checkpoint),
        _adapter_sha256(adapter),
        manifest.fingerprint,
        None if building_cache is None else building_cache.fingerprint,
    )
    refiner.eval()
    geometry = _geometry(
        args.geometry_cache, bool(trainer.model.config.use_geometry)
    )
    anchor_indices = (
        np.arange(len(dataset.train_pos), dtype=np.int64)
        if args.all_train_anchors else train
    )
    context = CoordinateBatchContext(
        args.cache_dir, manifest, dataset, anchor_indices,
        use_geometry=bool(trainer.model.config.use_geometry), geometry=geometry,
    )
    positions = np.asarray(dataset.test_pos, dtype=np.float64)
    test_building_features = (
        None
        if building_cache is None
        else building_cache.values("test", mode=args.map_mode, seed=args.seed)
    )
    train_building_features = (
        None
        if building_cache is None
        else building_cache.values("train", mode=args.map_mode, seed=args.seed)
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise ValueError("submission already exists")
    temporary = output.with_name(output.name + ".tmp")
    array = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.complex64,
        shape=(len(positions),) + dataset.config.channel_shape,
    )
    try:
        with torch.no_grad():
            for start in tqdm(
                range(0, len(positions), args.batch_size),
                desc="power infer", dynamic_ncols=True,
            ):
                source_batch = context.build(
                    positions[start:start + args.batch_size]
                )
                if test_building_features is not None:
                    source_batch["building_features"] = torch.as_tensor(
                        np.asarray(
                            test_building_features[
                                start : start + args.batch_size
                            ]
                        ),
                        dtype=torch.float32,
                    )
                    anchor_ids = source_batch["anchor_indices"].cpu().numpy()
                    source_batch["anchor_building_features"] = torch.as_tensor(
                        np.asarray(train_building_features[anchor_ids]),
                        dtype=torch.float32,
                    )
                batch = trainer._move_batch(source_batch)
                refined = refiner(batch)
                coarse_latent = refined.latent - refined.low_rank_residual
                latent = coarse_latent + float(args.correction_scale) * (
                    refined.latent - coarse_latent
                )
                decoded = adapter.decode_torch(latent)
                values = decoded.cpu().numpy().astype(np.complex64, copy=False)
                array[start:start + len(values)] = values
        array.flush()
        del array
        validate_submission(temporary, dataset, args.batch_size)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    report = {
        "base_checkpoint": str(Path(args.checkpoint).resolve()),
        "refiner_checkpoint": str(Path(args.refiner_checkpoint).resolve()),
        "anchor_count_available": int(len(anchor_indices)),
        "correction_scale": float(args.correction_scale),
        "feature_version": int(args.feature_version),
        "map_mode": args.map_mode,
        "building_feature_cache_fingerprint": (
            None if building_cache is None else building_cache.fingerprint
        ),
        "submission": validate_submission(output, dataset, args.batch_size),
    }
    Path(args.report or output.with_name("power_infer_report.json")).write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


def _run_evaluate(args):
    trainer, manifest, dataset, adapter, _, validation = _load_verified_trainer(args)
    building_cache = _load_building_features(args, dataset, manifest)
    map_dim = 0 if building_cache is None else len(building_cache.feature_names)
    refiner = PowerAwareRefiner(
        trainer.model, adapter, args.hidden_dim, args.feature_version, map_dim
    ).to(trainer.device)
    _load_refiner(
        args.refiner_checkpoint,
        refiner,
        _sha256(args.checkpoint),
        _adapter_sha256(adapter),
        manifest.fingerprint,
        None if building_cache is None else building_cache.fingerprint,
    )
    refiner.eval()
    validation_set = CachedAnchorDataset(
        args.cache_dir, validation, False, bool(trainer.model.config.use_geometry)
    )
    validation_set = _attach_building_dataset(
        validation_set, building_cache, validation, args.map_mode, args.seed
    )
    scales = tuple(float(value) for value in args.correction_scales)
    if any(not math.isfinite(value) or value < 0 for value in scales):
        raise ValueError("correction scales must be finite and non-negative")
    prediction_scale = (
        None if args.prediction_scale is None else float(args.prediction_scale)
    )
    if args.prediction_output and prediction_scale is None:
        raise ValueError("--prediction-scale is required with --prediction-output")
    if prediction_scale is not None and prediction_scale not in scales:
        raise ValueError("--prediction-scale must be included in --correction-scales")
    prediction_output = None
    prediction_offset = 0
    if args.prediction_output:
        prediction_path = Path(args.prediction_output)
        if prediction_path.exists():
            raise ValueError("validation prediction output already exists")
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_output = np.lib.format.open_memmap(
            prediction_path,
            mode="w+",
            dtype=np.complex64,
            shape=(len(validation),) + tuple(dataset.config.channel_shape),
        )
    accumulators = {
        f"{scale:g}": MetricAccumulator(adapter.layout) for scale in scales
    }
    with torch.no_grad():
        for source in tqdm(
            _loader(validation_set, args.batch_size, False, args.seed),
            desc="power scale sweep", dynamic_ncols=True,
        ):
            batch = trainer._move_batch(source)
            refined = refiner(batch)
            coarse = refined.latent - refined.low_rank_residual
            target = batch["target_channel"].cpu().numpy().astype(
                np.complex64, copy=False
            )
            for scale in scales:
                latent = coarse + scale * (refined.latent - coarse)
                prediction = adapter.decode_torch(latent).cpu().numpy().astype(
                    np.complex64, copy=False
                )
                accumulators[f"{scale:g}"].update(prediction, target)
                if prediction_output is not None and scale == prediction_scale:
                    prediction_output[
                        prediction_offset : prediction_offset + len(prediction)
                    ] = prediction
            prediction_offset += len(target)
    if prediction_output is not None:
        prediction_output.flush()
        del prediction_output
    metrics = {
        name: accumulator.compute().to_dict()
        for name, accumulator in accumulators.items()
    }
    best_scale, best = max(metrics.items(), key=lambda item: item[1]["score"])
    report = {
        "base_checkpoint": str(Path(args.checkpoint).resolve()),
        "refiner_checkpoint": str(Path(args.refiner_checkpoint).resolve()),
        "cache_manifest_fingerprint": manifest.fingerprint,
        "best_scale": float(best_scale),
        "best_metrics": best,
        "scale_metrics": metrics,
        "sample_count": int(len(validation)),
        "feature_version": int(args.feature_version),
        "map_mode": args.map_mode,
        "building_feature_cache_fingerprint": (
            None if building_cache is None else building_cache.fingerprint
        ),
        "validation_prediction": (
            None
            if not args.prediction_output
            else {
                "path": str(Path(args.prediction_output).resolve()),
                "scale": prediction_scale,
                "sha256": _sha256(args.prediction_output),
                "shape": [len(validation), *dataset.config.channel_shape],
                "dtype": "complex64",
            }
        ),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"best scale={best_scale}: score={best['score']:.6f} "
        f"pas={best['pas']:.4f} pdp={best['pdp']:.4f} nmse={best['nmse']:.4f}",
        flush=True,
    )
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--data-dir", required=True)
    train.add_argument("--cache-dir", required=True)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--run-dir", required=True)
    train.add_argument("--device", default="cpu")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--patience", type=int, default=8)
    train.add_argument("--hidden-dim", type=int, default=64)
    train.add_argument("--feature-version", type=int, choices=(1, 2, 3), default=1)
    train.add_argument("--building-feature-cache")
    train.add_argument("--map-mode", choices=("real", "zero", "shuffle"), default="real")
    train.add_argument("--latent-weight", type=float, default=0.01)
    train.add_argument("--nearest-weight", type=float, default=0.02)
    train.add_argument("--nmse-objective", choices=("official", "log"), default="log")
    train.set_defaults(handler=_run_train)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--data-dir", required=True)
    evaluate.add_argument("--cache-dir", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--refiner-checkpoint", required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--hidden-dim", type=int, default=64)
    evaluate.add_argument("--feature-version", type=int, choices=(1, 2, 3), default=1)
    evaluate.add_argument("--building-feature-cache")
    evaluate.add_argument("--map-mode", choices=("real", "zero", "shuffle"), default="real")
    evaluate.add_argument(
        "--correction-scales", type=float, nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25),
    )
    evaluate.add_argument("--prediction-output")
    evaluate.add_argument("--prediction-scale", type=float)
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(handler=_run_evaluate)

    infer = commands.add_parser("infer")
    infer.add_argument("--data-dir", required=True)
    infer.add_argument("--cache-dir", required=True)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--refiner-checkpoint", required=True)
    infer.add_argument("--geometry-cache")
    infer.add_argument("--device", default="cpu")
    infer.add_argument("--batch-size", type=int, default=8)
    infer.add_argument("--seed", type=int, default=42)
    infer.add_argument("--hidden-dim", type=int, default=64)
    infer.add_argument("--feature-version", type=int, choices=(1, 2, 3), default=1)
    infer.add_argument("--building-feature-cache")
    infer.add_argument("--map-mode", choices=("real", "zero", "shuffle"), default="real")
    infer.add_argument("--correction-scale", type=float, default=1.0)
    infer.add_argument("--all-train-anchors", action="store_true")
    infer.add_argument("--output", required=True)
    infer.add_argument("--report")
    infer.add_argument("--overwrite", action="store_true")
    infer.set_defaults(handler=_run_infer)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
