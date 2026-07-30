import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from solution.radio_map.learning.ac_cgmrf import ACCGMRF, ACCGMRFConfig
from solution.radio_map.learning.ac_cgmrf_cli import (
    _condition_arrays,
    _load_checkpoint,
    _save_checkpoint,
    build_parser,
    main,
)


class _TokenCache:
    feature_names = ("f0", "f1")
    fingerprint = "token-fingerprint"

    def values(self, split, mode="real"):
        del mode
        count = 5 if split == "train" else 2
        values = np.arange(count * 3 * 2, dtype=np.float32).reshape(
            count, 3, 2
        )
        return values, np.ones((count, 3), dtype=bool)


class _DirectCache:
    fingerprint = "direct-fingerprint"

    def __init__(self, neighbors):
        self.neighbors = neighbors

    def values(self, split):
        count = 5 if split == "train" else 2
        tokens = np.full((count, 2, 3, 2), 7.0, np.float32)
        mask = np.ones((count, 2, 3), dtype=bool)
        distances = np.ones((count, 2), np.float32)
        return tokens, mask, self.neighbors[:count], distances


class ACCGMRFCliTests(unittest.TestCase):
    def test_conditions_bind_shuffled_map_only(self):
        neighbors = np.asarray(
            [[1, 2], [0, 2], [1, 3], [2, 4], [3, 0]], dtype=np.int64
        )
        rows = np.asarray([0, 2], dtype=np.int64)
        map_rows = np.asarray([1, 3], dtype=np.int64)
        result = _condition_arrays(
            _TokenCache(),
            _DirectCache(neighbors),
            neighbors,
            rows,
            map_rows,
            "train",
            torch.device("cpu"),
        )
        expected_target = _TokenCache().values("train")[0][map_rows]
        expected_anchor = _TokenCache().values("train")[0][
            neighbors[map_rows]
        ]
        np.testing.assert_array_equal(
            result["target_path_tokens"].numpy(), expected_target
        )
        np.testing.assert_array_equal(
            result["anchor_path_tokens"].numpy(), expected_anchor
        )
        self.assertTrue(result["path_mask"].all())

    def test_parser_exposes_gated_research_commands_and_k4_guard(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "train",
                "--data-dir",
                "data",
                "--cache-dir",
                "cache",
                "--base-checkpoint",
                "base.pt",
                "--refiner-checkpoint",
                "power.pt",
                "--supervision-cache",
                "supervision",
                "--gaussian-token-cache",
                "tokens",
                "--anchor-path-cache",
                "paths",
                "--o41-checkpoint",
                "o41.pt",
                "--run-dir",
                "run",
            ]
        )
        self.assertEqual(args.anchor_count, 4)
        self.assertEqual(args.stage, "pilot")
        self.assertIn(0.0, args.validation_scales)
        with self.assertRaisesRegex(ValueError, "K=4"):
            main(
                [
                    "train",
                    "--data-dir",
                    "unused",
                    "--cache-dir",
                    "unused",
                    "--base-checkpoint",
                    "unused",
                    "--refiner-checkpoint",
                    "unused",
                    "--supervision-cache",
                    "unused",
                    "--gaussian-token-cache",
                    "unused",
                    "--anchor-path-cache",
                    "unused",
                    "--o41-checkpoint",
                    "unused",
                    "--run-dir",
                    "unused",
                    "--anchor-count",
                    "3",
                ]
            )

    def test_checkpoint_roundtrip_binds_o41_and_cache_identities(self):
        config = ACCGMRFConfig(
            p_count=1,
            n_count=2,
            path_feature_dim=2,
            atom_count=2,
            d_model=16,
            map_hidden_dim=12,
            support_h=1,
            support_v=1,
            support_delay=1,
        )
        model = ACCGMRF(config)
        provider = SimpleNamespace(fingerprint="coarse-fingerprint")
        token_cache = _TokenCache()
        direct_cache = SimpleNamespace(fingerprint="direct-fingerprint")
        args = SimpleNamespace(anchor_count=4, stage="pilot")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            _save_checkpoint(
                path,
                model,
                config,
                provider,
                token_cache,
                direct_cache,
                args,
                "o41-sha",
                0.5,
                1.0,
                {"score": 0.5},
            )
            loaded, payload = _load_checkpoint(
                path,
                provider,
                token_cache,
                direct_cache,
                args,
                torch.device("cpu"),
                "o41-sha",
                0.5,
            )
            self.assertIsInstance(loaded, ACCGMRF)
            self.assertEqual(payload["training_stage"], "pilot")
            self.assertTrue(path.with_name("best.pt.sha256").is_file())
            with self.assertRaisesRegex(ValueError, "identity"):
                _load_checkpoint(
                    path,
                    provider,
                    token_cache,
                    direct_cache,
                    args,
                    torch.device("cpu"),
                    "different-o41-sha",
                    0.5,
                )


if __name__ == "__main__":
    unittest.main()
