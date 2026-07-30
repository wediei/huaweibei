"""Reproducible command surface for Support-Aware Anchor Mixer experiments.

All commands operate on a fingerprinted fold cache.  In particular, a split is
stored as integer arrays and authenticated by the cache manifest; it is never
re-derived from a seed during train/evaluate/infer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

from ..data import RoundDataset
from ..geometry import GeometryPrior
from ..splits import block_split, coverage_split
from ..transforms import AntennaLayout
from ..cli import validate_submission
from .anchor_mixer import SupportAwareAnchorMixer, SupportAwareAnchorMixerConfig
from .cache import FoldCacheConfig, FoldCacheManifest, prepare_fold_cache, validate_cache
from .dataset import CachedAnchorDataset, CoordinateBatchContext, collate_anchor_batch, load_persisted_split
from .latent_adapter import FixedSupportLatentAdapter
from .losses import AnchorLossConfig
from .trainer import Trainer, TrainerConfig


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: str | Path, report: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as handle:
        json.dump(report, handle, sort_keys=True, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)
    return destination


def _seed_everything(seed: int) -> None:
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _limited(indices: np.ndarray, limit: int | None, name: str) -> np.ndarray:
    if limit is None:
        return indices
    if limit <= 0:
        raise ValueError(f"{name} must be positive when provided")
    values = indices[:limit]
    if not len(values):
        raise ValueError(f"{name} leaves no samples")
    return values


def _load_fold(
    cache_dir: str | Path,
    data_dir: str | Path | None = None,
    *,
    allow_cache_code_mismatch: bool = False,
) -> tuple[FoldCacheManifest, RoundDataset, FixedSupportLatentAdapter, np.ndarray, np.ndarray]:
    cache = Path(cache_dir)
    manifest = FoldCacheManifest.load(cache / "manifest.json")
    validate_cache(
        manifest,
        cache,
        allow_code_mismatch=allow_cache_code_mismatch,
    )
    dataset = RoundDataset.open(manifest.data_dir)
    if data_dir is not None and Path(data_dir).resolve() != dataset.data_dir:
        raise ValueError("data-dir does not match the cache manifest")
    train, validation = load_persisted_split(cache, validate=False)
    if not np.array_equal(np.asarray(train, dtype=np.int64), np.asarray(np.load(cache / "train_indices.npy", allow_pickle=False), dtype=np.int64)):
        raise ValueError("persisted train split identity mismatch")
    layout = AntennaLayout(dataset.config, tuple(manifest.layout_order))
    adapter = FixedSupportLatentAdapter.load(cache / "adapter.npz", layout)
    if adapter.fit_indices_sha256 != manifest.train_indices_sha256 or not np.array_equal(adapter.fitted_indices, train):
        raise ValueError("adapter is not fitted to the persisted training split")
    return manifest, dataset, adapter, train, validation


def _model_factory(config: dict[str, Any], group_ids: torch.Tensor) -> SupportAwareAnchorMixer:
    return SupportAwareAnchorMixer(SupportAwareAnchorMixerConfig(**config), group_ids)


def _model_config(args: argparse.Namespace, adapter: FixedSupportLatentAdapter, use_geometry: bool) -> SupportAwareAnchorMixerConfig:
    return SupportAwareAnchorMixerConfig(
        latent_size=adapter.coefficient_count,
        group_count=int(np.max(adapter.group_ids)) + 1,
        d_model=args.d_model,
        group_dim=args.group_dim,
        geometry_dim=args.geometry_dim,
        low_rank=args.low_rank,
        num_frequencies=args.num_frequencies,
        use_geometry=use_geometry,
    )


def _geometry(path: str | None, required: bool) -> GeometryPrior | None:
    if not required:
        return None
    if not path:
        raise ValueError("--geometry-cache is required with --geometry")
    return GeometryPrior.load(path)


def _run_prepare_cache(args: argparse.Namespace) -> int:
    dataset = RoundDataset.open(args.data_dir)
    geometry = GeometryPrior.load(args.geometry_cache)
    if args.split_protocol == "coverage":
        split = coverage_split(
            dataset.train_pos,
            args.validation_fraction,
            args.grid_size,
            args.split_seed,
        )
        split_details = {
            "seed": args.split_seed,
            "validation_fraction": args.validation_fraction,
            "grid_size": args.grid_size,
        }
    else:
        split = block_split(
            dataset.train_pos,
            axis=args.block_axis,
            validation_fraction=args.validation_fraction,
            side=args.block_side,
        )
        split_details = {
            "validation_fraction": args.validation_fraction,
            "axis": args.block_axis,
            "side": args.block_side,
        }
    config = FoldCacheConfig(
        layout_order=tuple(args.layout_order), support_fraction=args.support_fraction, delay_block=args.delay_block,
        k_max=args.k_max, anchor_count=args.anchor_count, patch=args.patch, corridor=args.corridor,
        anchor_corridor=args.anchor_corridor, encode_batch_size=args.channel_batch_size, dropout=args.dropout,
        min_anchors=args.min_anchors, seed=args.split_seed, code_version="task-7-schema-v3",
    )
    cache = prepare_fold_cache(dataset, geometry, split, config, args.output_dir)
    manifest = FoldCacheManifest.load(cache / "manifest.json")
    validate_cache(manifest, cache)
    train, validation = load_persisted_split(cache, validate=False)
    expected_validation = min(len(dataset.train_pos) - 1, max(1, int(round(len(dataset.train_pos) * args.validation_fraction))))
    report = {
        "cache_dir": str(cache), "manifest_fingerprint": manifest.fingerprint, "adapter_sha256": manifest.adapter_sha256,
        "geometry_sha256": manifest.geometry_sha256, "protocol": f"{args.split_protocol}_split", "expected_train_count": int(len(dataset.train_pos) - expected_validation),
        "expected_validation_count": int(expected_validation), "actual_train_count": int(len(train)), "actual_validation_count": int(len(validation)),
        "train_count": int(len(train)), "validation_count": int(len(validation)),
        "overlap_count": int(np.intersect1d(train, validation).size), "train_indices_path": str(cache / "train_indices.npy"),
        "validation_indices_path": str(cache / "validation_indices.npy"), "split": split_details,
    }
    _atomic_json(cache / "prepare_report.json", report)
    return 0


class _EpochSeededSampler(Sampler[int]):
    """Stable per-epoch shuffle, so resume sees exactly the original order."""

    def __init__(self, data_source: CachedAnchorDataset, seed: int) -> None:
        self.data_source, self.seed, self.epoch = data_source, int(seed), 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.data_source), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.data_source)


def _loader(dataset: CachedAnchorDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader[dict[str, torch.Tensor]]:
    if batch_size <= 0:
        raise ValueError("batch-size must be positive")
    sampler = _EpochSeededSampler(dataset, seed) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=0, collate_fn=collate_anchor_batch)


def _run_train(args: argparse.Namespace) -> int:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    _seed_everything(args.seed)
    manifest, _, adapter, train, validation = _load_fold(args.cache_dir, args.data_dir)
    use_geometry = bool(args.geometry)
    train = _limited(train, args.limit_train_samples, "limit-train-samples")
    validation = _limited(validation, args.limit_validation_samples, "limit-validation-samples")
    train_set = CachedAnchorDataset(args.cache_dir, train, training=True, use_geometry=use_geometry)
    validation_set = CachedAnchorDataset(args.cache_dir, validation, training=False, use_geometry=use_geometry)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint, last_checkpoint = run_dir / "best.pt", run_dir / "last.pt"
    if (checkpoint.exists() or last_checkpoint.exists()) and not args.resume:
        raise ValueError("run checkpoints already exist; use --resume to continue the verified run")
    config = _model_config(args, adapter, use_geometry)
    trainer_config = TrainerConfig(epochs=args.epochs, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        accumulation_steps=args.accumulation_steps, gradient_clip_norm=args.gradient_clip_norm, warmup_epochs=args.warmup_epochs,
        patience=args.patience, log_path=str(run_dir / "metrics.jsonl"))
    loss_config = AnchorLossConfig(
        latent_weight=args.latent_weight,
        nearest_weight=args.nearest_weight,
        nmse_objective=args.nmse_objective,
    )
    if args.resume:
        trainer = Trainer.load_checkpoint(last_checkpoint, _model_factory, device=args.device, adapter=adapter,
            expected_adapter_hash=None, expected_manifest_hash=manifest.fingerprint)
        if trainer.model_config != config.__dict__ or trainer.config != trainer_config or trainer.loss_config != loss_config:
            raise ValueError("resume trainer/model/loss configuration differs from the verified last checkpoint")
        if trainer.epoch >= args.epochs - 1:
            raise ValueError("resume would run zero epochs; increase nothing is permitted, start a new run instead")
    else:
        model = SupportAwareAnchorMixer(config, torch.as_tensor(adapter.group_ids, dtype=torch.long))
        trainer = Trainer(model, adapter, trainer_config, device=args.device, manifest_hash=manifest.fingerprint,
            loss_config=loss_config)
    started = time.perf_counter()
    records = trainer.fit(_loader(train_set, args.batch_size, True, args.seed), _loader(validation_set, args.batch_size, False, args.seed), checkpoint, last_checkpoint)
    report = {
        "cache_manifest_fingerprint": manifest.fingerprint, "adapter_sha256": manifest.adapter_sha256,
        "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256_file(checkpoint), "last_checkpoint": str(last_checkpoint), "last_checkpoint_sha256": _sha256_file(last_checkpoint), "device": str(args.device),
        "use_geometry": use_geometry, "model_config": config.__dict__, "trainer_config": trainer_config.__dict__,
        "loss_config": loss_config.__dict__,
        "limit_train_samples": args.limit_train_samples, "limit_validation_samples": args.limit_validation_samples,
        "epochs_completed": len(records), "best_metrics": trainer.best_metrics, "runtime_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated() if torch.cuda.is_available() and str(args.device).startswith("cuda") else 0),
    }
    _atomic_json(run_dir / "train_report.json", report)
    return 0


def _load_verified_trainer(args: argparse.Namespace) -> tuple[Trainer, FoldCacheManifest, RoundDataset, FixedSupportLatentAdapter, np.ndarray, np.ndarray]:
    manifest, dataset, adapter, train, validation = _load_fold(
        args.cache_dir,
        args.data_dir,
        allow_cache_code_mismatch=bool(
            getattr(args, "allow_cache_code_mismatch", False)
        ),
    )
    trainer = Trainer.load_checkpoint(args.checkpoint, _model_factory, device=args.device, adapter=adapter,
        expected_manifest_hash=manifest.fingerprint)
    return trainer, manifest, dataset, adapter, train, validation


def _run_evaluate(args: argparse.Namespace) -> int:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    trainer, manifest, _, _, _, validation = _load_verified_trainer(args)
    indices = _limited(validation, args.limit_samples, "limit-samples")
    use_geometry = bool(trainer.model.config.use_geometry)
    dataset = CachedAnchorDataset(args.cache_dir, indices, False, use_geometry)
    scale_metrics = {
        f"{scale:g}": trainer.validate(
            _loader(dataset, args.batch_size, False, args.seed),
            correction_scale=scale,
        )
        for scale in args.correction_scales
    }
    best_scale_name, values = max(
        scale_metrics.items(), key=lambda item: item[1]["score"]
    )
    report = {"cache_manifest_fingerprint": manifest.fingerprint, "checkpoint": str(Path(args.checkpoint).resolve()), "metrics": values,
        "correction_scale": float(best_scale_name), "scale_metrics": scale_metrics,
        "sample_count": int(len(indices)), "use_geometry": use_geometry}
    _atomic_json(args.output or Path(args.checkpoint).parent / "evaluate_report.json", report)
    return 0


def _run_infer(args: argparse.Namespace) -> int:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if not np.isfinite(args.correction_scale) or args.correction_scale < 0:
        raise ValueError("correction-scale must be finite and non-negative")
    trainer, manifest, dataset, adapter, train, _ = _load_verified_trainer(args)
    use_geometry = bool(trainer.model.config.use_geometry)
    geometry = _geometry(args.geometry_cache, use_geometry)
    positions = np.asarray(dataset.test_pos, dtype=np.float64)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise ValueError("submission already exists; pass --overwrite only for a deliberate replacement")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    prediction = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.complex64, shape=(len(positions),) + dataset.config.channel_shape)
    try:
        trainer.model.eval()
        anchor_indices = (
            np.arange(len(dataset.train_pos), dtype=np.int64)
            if args.all_train_anchors
            else train
        )
        context = CoordinateBatchContext(args.cache_dir, manifest, dataset, anchor_indices, use_geometry=use_geometry, geometry=geometry)
        for start in tqdm(
            range(0, len(positions), args.batch_size),
            desc="test inference",
            dynamic_ncols=True,
        ):
            batch = context.build(positions[start:start + args.batch_size])
            batch = trainer._move_batch(batch)
            with torch.no_grad(), trainer._autocast():
                model_output = trainer.model(batch)
                latent = model_output.nearest_latent + float(args.correction_scale) * (
                    model_output.latent - model_output.nearest_latent
                )
                decoded = adapter.decode_torch(latent).detach().cpu().numpy().astype(np.complex64, copy=False)
            prediction[start:start + len(decoded)] = decoded
        prediction.flush(); del prediction
        validation = validate_submission(temporary, dataset, args.batch_size)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    report = {"cache_manifest_fingerprint": manifest.fingerprint, "checkpoint": str(Path(args.checkpoint).resolve()),
        "submission": validate_submission(output, dataset, args.batch_size), "test_count": int(len(positions)),
        "correction_scale": float(args.correction_scale),
        "anchor_source": "all_official_train" if args.all_train_anchors else "fold_train",
        "anchor_count_available": int(len(anchor_indices)),
        "test_positions_sha256": hashlib.sha256(np.ascontiguousarray(positions).tobytes()).hexdigest(),
        "train_dataset_rows_used_for_test_positions": False, "use_geometry": use_geometry}
    _atomic_json(args.report or output.with_name("infer_report.json"), report)
    return 0


def _add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--d-model", type=int, default=128); parser.add_argument("--group-dim", type=int, default=64)
    parser.add_argument("--geometry-dim", type=int, default=64); parser.add_argument("--low-rank", type=int, default=32)
    parser.add_argument("--num-frequencies", type=int, default=6)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-cache")
    prepare.add_argument("--data-dir", required=True); prepare.add_argument("--geometry-cache", required=True); prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--layout-order", choices=("HVP", "HPV", "VHP", "VPH", "PHV", "PVH"), default="PHV")
    prepare.add_argument("--support-fraction", type=float, default=0.10); prepare.add_argument("--delay-block", type=int, default=8)
    prepare.add_argument("--split-seed", type=int, default=42); prepare.add_argument("--validation-fraction", type=float, default=0.1); prepare.add_argument("--grid-size", type=float, default=20.0)
    prepare.add_argument("--split-protocol", choices=("coverage", "block"), default="coverage")
    prepare.add_argument("--block-axis", type=int, choices=(0, 1, 2), default=0)
    prepare.add_argument("--block-side", choices=("high", "low"), default="high")
    prepare.add_argument("--k-max", type=int, default=64); prepare.add_argument("--anchor-count", type=int, default=32); prepare.add_argument("--channel-batch-size", type=int, default=16)
    prepare.add_argument("--patch", type=int, default=33); prepare.add_argument("--corridor", type=int, default=32); prepare.add_argument("--anchor-corridor", type=int, default=8)
    prepare.add_argument("--dropout", type=float, default=0.1); prepare.add_argument("--min-anchors", type=int, default=8); prepare.set_defaults(handler=_run_prepare_cache)
    train = commands.add_parser("train")
    train.add_argument("--data-dir", required=True); train.add_argument("--cache-dir", required=True); train.add_argument("--run-dir", required=True); train.add_argument("--device", default="cpu")
    group = train.add_mutually_exclusive_group(); group.add_argument("--geometry", action="store_true"); group.add_argument("--no-geometry", dest="geometry", action="store_false"); train.set_defaults(geometry=False)
    train.add_argument("--seed", type=int, default=42); train.add_argument("--epochs", type=int, default=100); train.add_argument("--batch-size", type=int, default=4); train.add_argument("--learning-rate", type=float, default=2e-4); train.add_argument("--weight-decay", type=float, default=1e-4); train.add_argument("--accumulation-steps", type=int, default=4); train.add_argument("--gradient-clip-norm", type=float, default=1.0); train.add_argument("--warmup-epochs", type=int, default=5); train.add_argument("--patience", type=int, default=15); train.add_argument("--nmse-objective", choices=("official", "log"), default="official"); train.add_argument("--latent-weight", type=float, default=0.02); train.add_argument("--nearest-weight", type=float, default=0.05); train.add_argument("--limit-train-samples", type=int); train.add_argument("--limit-validation-samples", type=int); train.add_argument("--resume", action="store_true"); _add_common_model_args(train); train.set_defaults(handler=_run_train)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--data-dir", required=True); evaluate.add_argument("--cache-dir", required=True); evaluate.add_argument("--checkpoint", required=True); evaluate.add_argument("--device", default="cpu"); evaluate.add_argument("--batch-size", type=int, default=4); evaluate.add_argument("--seed", type=int, default=42); evaluate.add_argument("--limit-samples", type=int); evaluate.add_argument("--correction-scales", type=float, nargs="+", default=(1.0,)); evaluate.add_argument("--output"); evaluate.set_defaults(handler=_run_evaluate)
    infer = commands.add_parser("infer")
    infer.add_argument("--data-dir", required=True); infer.add_argument("--cache-dir", required=True); infer.add_argument("--checkpoint", required=True); infer.add_argument("--output", required=True); infer.add_argument("--device", default="cpu"); infer.add_argument("--batch-size", type=int, default=4); infer.add_argument("--geometry-cache"); infer.add_argument("--correction-scale", type=float, default=1.0); infer.add_argument("--all-train-anchors", action="store_true"); infer.add_argument("--report"); infer.add_argument("--overwrite", action="store_true"); infer.set_defaults(handler=_run_infer)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
