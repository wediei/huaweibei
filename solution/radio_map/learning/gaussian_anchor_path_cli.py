"""Build direct Anchor-to-Target Gaussian propagation paths."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .gaussian_anchor_path_cache import build_gaussian_anchor_path_cache
from .gaussian_token_cache import GaussianTokenConfig


def _run(args):
    build_gaussian_anchor_path_cache(
        data_dir=args.data_dir,
        fold_cache=args.cache_dir,
        scene_cache=args.scene_cache,
        output_dir=args.output_dir,
        anchor_count=args.anchor_count,
        anchor_source=args.anchor_source,
        token_config=GaussianTokenConfig(
            tokens_per_path=args.tokens_per_path,
            candidate_k=args.candidate_k,
            path_samples=args.path_samples,
            elevated_quota=args.elevated_quota,
            ground_height=args.ground_height,
        ),
    )
    print(Path(args.output_dir, "manifest.json").resolve())
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepare", nargs="?")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--scene-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--anchor-count", type=int, default=4)
    parser.add_argument(
        "--anchor-source",
        choices=("fold", "all_official_train"),
        default="all_official_train",
    )
    parser.add_argument("--tokens-per-path", type=int, default=48)
    parser.add_argument("--candidate-k", type=int, default=12)
    parser.add_argument("--path-samples", type=int, default=16)
    parser.add_argument("--elevated-quota", type=int, default=16)
    parser.add_argument("--ground-height", type=float, default=1.5)
    parser.set_defaults(handler=_run)
    return parser


def main(argv: Sequence[str] | None = None):
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
