"""Prepare authenticated sequence-valued Gaussian map tokens."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from ..data import RoundDataset
from .cache import FoldCacheManifest, validate_cache
from .gaussian_geometry import (
    GaussianScene,
    GaussianSceneConfig,
    _sha256_file,
)
from .gaussian_token_cache import (
    GaussianTokenCache,
    GaussianTokenConfig,
    build_gaussian_path_tokens,
    save_gaussian_token_cache,
)


def _run_prepare(args: argparse.Namespace) -> int:
    fold_path = Path(args.cache_dir)
    manifest = FoldCacheManifest.load(fold_path / "manifest.json")
    validate_cache(manifest, fold_path, allow_code_mismatch=True)
    dataset = RoundDataset.open(args.data_dir)
    if dataset.data_dir != Path(manifest.data_dir).resolve():
        raise ValueError("data directory differs from fold cache")
    scene_config = GaussianSceneConfig(
        voxel_size=args.voxel_size,
        normal_scale=args.normal_scale,
        tangent_scale=args.tangent_scale,
        density_scale=args.density_scale,
    )
    scene_path = Path(args.scene_cache)
    if scene_path.is_file():
        scene = GaussianScene.load(scene_path)
        if scene.metadata.get("source_map_sha256") != _sha256_file(
            dataset.map_path
        ):
            raise ValueError("Gaussian scene was built from another map")
        if scene.metadata.get("scene_config") != asdict(scene_config):
            raise ValueError("Gaussian scene configuration differs")
    else:
        scene = GaussianScene.from_ply(dataset.map_path, scene_config)
        scene.save(scene_path)
    token_config = GaussianTokenConfig(
        tokens_per_path=args.tokens_per_path,
        candidate_k=args.candidate_k,
        path_samples=args.path_samples,
        elevated_quota=args.elevated_quota,
        ground_height=args.ground_height,
    )
    bs = np.asarray(dataset.config.bs_position, dtype=np.float32)
    train_positions = np.asarray(dataset.train_pos, dtype=np.float32)
    test_positions = np.asarray(dataset.test_pos, dtype=np.float32)
    train_tokens = build_gaussian_path_tokens(
        scene,
        np.broadcast_to(bs, train_positions.shape),
        train_positions,
        token_config,
    )
    test_tokens = build_gaussian_path_tokens(
        scene,
        np.broadcast_to(bs, test_positions.shape),
        test_positions,
        token_config,
    )
    train_indices = np.asarray(
        np.load(fold_path / "train_indices.npy", allow_pickle=False),
        dtype=np.int64,
    )
    save_gaussian_token_cache(
        args.output_dir,
        train_tokens,
        test_tokens,
        fold_fingerprint=manifest.fingerprint,
        map_sha256=_sha256_file(dataset.map_path),
        train_fit_indices=train_indices,
        config=token_config,
    )
    cache = GaussianTokenCache.load(
        args.output_dir,
        manifest.fingerprint,
        _sha256_file(dataset.map_path),
    )
    try:
        train_real, train_mask = cache.values("train", "real")
        test_real, test_mask = cache.values("test", "real")
        report = {
            "kind": "gaussian_path_token_prepare",
            "cache": str(Path(args.output_dir).resolve()),
            "cache_fingerprint": cache.fingerprint,
            "scene_cache": str(scene_path.resolve()),
            "scene_gaussian_count": scene.gaussian_count,
            "scene_config": asdict(scene_config),
            "token_config": asdict(token_config),
            "feature_names": list(cache.feature_names),
            "train_shape": list(train_real.shape),
            "test_shape": list(test_real.shape),
            "train_valid_ratio": float(train_mask.mean()),
            "test_valid_ratio": float(test_mask.mean()),
            "fold_fingerprint": manifest.fingerprint,
            "map_sha256": _sha256_file(dataset.map_path),
        }
    finally:
        del train_real, train_mask, test_real, test_mask
        cache.close()
    report_path = Path(args.output_dir) / "prepare_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare official-map Gaussian propagation tokens"
    )
    parser.add_argument("prepare", nargs="?")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--scene-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--voxel-size", type=float, default=1.0)
    parser.add_argument("--normal-scale", type=float, default=0.2)
    parser.add_argument("--tangent-scale", type=float, default=0.65)
    parser.add_argument("--density-scale", type=float, default=4.0)
    parser.add_argument("--tokens-per-path", type=int, default=48)
    parser.add_argument("--candidate-k", type=int, default=12)
    parser.add_argument("--path-samples", type=int, default=16)
    parser.add_argument("--elevated-quota", type=int, default=16)
    parser.add_argument("--ground-height", type=float, default=1.5)
    parser.set_defaults(handler=_run_prepare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
