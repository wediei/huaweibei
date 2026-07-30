import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from solution.radio_map.learning.gaussian_path_transport import (
    GaussianPathTransport,
    GaussianPathTransportConfig,
)
from solution.radio_map.learning.gaussian_path_transport_cli import (
    _direct_score_loss,
    _load_checkpoint,
    _map_context_values,
    _parameter_loss,
    build_parser,
)


class GaussianPathTransportCliTests(unittest.TestCase):
    def test_anchor_delta_context_contains_target_anchor_and_difference(self):
        class Cache:
            def values(self, split, mode, seed=0):
                del mode, seed
                if split == "train":
                    values = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
                else:
                    values = np.full((1, 2, 4), 100.0, np.float32)
                return values, np.ones(values.shape[:2], dtype=bool)

        with tempfile.TemporaryDirectory() as directory:
            np.save(
                Path(directory) / "neighbor_indices.npy",
                np.asarray([[1], [0], [1]], dtype=np.int64),
            )
            np.save(
                Path(directory) / "neighbor_distances.npy",
                np.ones((3, 1), dtype=np.float32),
            )
            args = SimpleNamespace(
                map_context="anchor-delta",
                map_anchor_count=1,
                cache_dir=directory,
                anchor_source="all_official_train",
            )
            dataset = SimpleNamespace(
                train_pos=np.asarray(
                    [[0, 0, 0], [2, 0, 0], [4, 0, 0]], np.float64
                ),
                test_pos=np.asarray([[3.9, 0, 0]], np.float64),
            )
            values, mask = _map_context_values(
                args, Cache(), dataset, "train", "real", 42
            )
            self.assertEqual(values.shape, (3, 2, 12))
            np.testing.assert_array_equal(values[0, :, :4], Cache().values("train", "real")[0][0])
            np.testing.assert_array_equal(values[0, :, 4:8], Cache().values("train", "real")[0][1])
            np.testing.assert_array_equal(
                values[0, :, 8:],
                values[0, :, :4] - values[0, :, 4:8],
            )
            self.assertTrue(mask.all())

    def test_multi_anchor_context_has_roles_and_distance_weights(self):
        class Cache:
            def values(self, split, mode, seed=0):
                del mode, seed
                if split == "train":
                    values = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
                else:
                    values = np.full((1, 2, 4), 100.0, np.float32)
                return values, np.ones(values.shape[:2], dtype=bool)

        with tempfile.TemporaryDirectory() as directory:
            np.save(
                Path(directory) / "neighbor_indices.npy",
                np.asarray([[1, 2], [0, 2], [1, 0]], dtype=np.int64),
            )
            np.save(
                Path(directory) / "neighbor_distances.npy",
                np.asarray([[1, 3], [2, 2], [3, 1]], dtype=np.float32),
            )
            args = SimpleNamespace(
                map_context="multi-anchor",
                map_anchor_count=2,
                cache_dir=directory,
                anchor_source="all_official_train",
            )
            dataset = SimpleNamespace(
                train_pos=np.asarray(
                    [[0, 0, 0], [2, 0, 0], [4, 0, 0]], np.float64
                ),
                test_pos=np.asarray([[3.9, 0, 0]], np.float64),
            )
            values, mask = _map_context_values(
                args, Cache(), dataset, "train", "real", 42
            )
            self.assertEqual(values.shape, (3, 6, 6))
            self.assertEqual(mask.shape, (3, 6))
            np.testing.assert_array_equal(values[0, :2, 4], 1.0)
            np.testing.assert_array_equal(values[0, 2:, 4], -1.0)
            np.testing.assert_allclose(values[0, 2:4, 5], 0.75)
            np.testing.assert_allclose(values[0, 4:6, 5], 0.25)
            self.assertTrue(mask.all())

    def test_commands_and_causal_defaults_are_exposed(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        args = parser.parse_args(
            [
                "evaluate",
                "--data-dir",
                "data",
                "--cache-dir",
                "fold",
                "--base-checkpoint",
                "base.pt",
                "--refiner-checkpoint",
                "power.pt",
                "--supervision-cache",
                "supervision",
                "--gaussian-token-cache",
                "tokens",
                "--checkpoint",
                "o3.pt",
                "--output",
                "report.json",
            ]
        )
        self.assertEqual(args.validation_scales[0], 0.0)
        self.assertGreater(args.minimum_gain, 0.0)
        self.assertGreater(args.minimum_control_margin, 0.0)
        oracle = parser.parse_args(
            [
                "oracle",
                "--data-dir",
                "data",
                "--cache-dir",
                "fold",
                "--base-checkpoint",
                "base.pt",
                "--refiner-checkpoint",
                "power.pt",
                "--supervision-cache",
                "supervision",
                "--output",
                "oracle.json",
            ]
        )
        self.assertEqual(oracle.validation_scales[0], 0.0)
        self.assertGreaterEqual(oracle.minimum_scale, 0.25)

    def test_parameter_loss_is_finite_and_backpropagates(self):
        config = GaussianPathTransportConfig(
            p_count=1, n_count=2, map_feature_dim=3, d_model=16, heads=2
        )
        model = GaussianPathTransport(config)
        coarse = torch.randn(2, 2, 2, 1, 2, 4, dtype=torch.complex64)
        output = model(
            coarse, torch.randn(2, 3, 3), torch.ones(2, 3, dtype=torch.bool)
        )
        labels = {
            "delta_h": torch.zeros(2, 1, 2),
            "delta_v": torch.zeros(2, 1, 2),
            "delta_delay": torch.zeros(2, 1, 2),
            "log_amplitude": torch.zeros(2, 1, 2),
            "phase_real": torch.ones(2, 1, 2),
            "phase_imag": torch.zeros(2, 1, 2),
            "existence": torch.ones(2, 1, 2),
            "reliability": torch.zeros(2, 1, 2),
        }
        loss, terms = _parameter_loss(output, labels)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("phase", terms)
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is None
                or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_direct_score_loss_is_finite_and_backpropagates(self):
        from solution.tests.test_latent_beam_delay import LatentBeamDelayTests

        adapter, channel = LatentBeamDelayTests()._adapter()
        config = GaussianPathTransportConfig(
            p_count=1, n_count=2, map_feature_dim=3, d_model=16, heads=2
        )
        model = GaussianPathTransport(config)
        coarse = adapter.beam_delay_torch(
            torch.tensor(adapter.encode_numpy(channel[:2]))
        )
        output = model(
            coarse, torch.randn(2, 3, 3), torch.ones(2, 3, dtype=torch.bool)
        )
        labels = {
            "delta_h": torch.zeros(2, 1, 2),
            "delta_v": torch.zeros(2, 1, 2),
            "delta_delay": torch.zeros(2, 1, 2),
            "log_amplitude": torch.zeros(2, 1, 2),
            "phase_real": torch.ones(2, 1, 2),
            "phase_imag": torch.zeros(2, 1, 2),
            "existence": torch.ones(2, 1, 2),
            "reliability": torch.zeros(2, 1, 2),
        }
        loss, terms = _direct_score_loss(
            output,
            torch.tensor(channel[:2]),
            adapter,
            labels,
            scale=0.75,
            label_weight=0.05,
            trust_weight=0.001,
            nmse_objective="log",
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(terms["score"]))
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is None
                or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_checkpoint_rejects_missing_hash_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            torch.save({}, path)
            with self.assertRaisesRegex(ValueError, "hash"):
                _load_checkpoint(
                    path,
                    SimpleNamespace(fingerprint="coarse"),
                    SimpleNamespace(fingerprint="tokens"),
                    torch.device("cpu"),
                )


if __name__ == "__main__":
    unittest.main()
