from __future__ import annotations

import unittest

import numpy as np
import torch

from solution.radio_map.learning.torch_metrics import (
    row_power_cosine,
    torch_competition_metrics,
)
from solution.radio_map.metrics import competition_metrics
from solution.radio_map.transforms import AntennaLayout
from solution.tests.test_transforms import make_config


class TorchCompetitionMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = make_config()
        self.layout = AntennaLayout(self.config, ("V", "P", "H"))
        rng = np.random.default_rng(29)
        self.target_np = (
            rng.standard_normal((3, 8, 2, 4))
            + 1j * rng.standard_normal((3, 8, 2, 4))
        ).astype(np.complex128)
        self.pred_np = self.target_np * (0.8 + 0.15j)

    def _metrics(self, prediction: np.ndarray):
        return torch_competition_metrics(
            torch.from_numpy(prediction),
            torch.from_numpy(self.target_np),
            self.config,
            self.layout.order,
        )

    def test_identity_prediction_scores_one(self) -> None:
        actual = self._metrics(self.target_np)

        self.assertAlmostEqual(actual.pas.item(), 1.0, places=7)
        self.assertAlmostEqual(actual.pdp.item(), 1.0, places=7)
        self.assertAlmostEqual(actual.nmse.item(), 0.0, places=7)
        self.assertAlmostEqual(actual.score.item(), 1.0, places=7)

    def test_phase_inversion_preserves_power_metrics(self) -> None:
        actual = self._metrics(-self.target_np)

        self.assertAlmostEqual(actual.pas.item(), 1.0, places=7)
        self.assertAlmostEqual(actual.pdp.item(), 1.0, places=7)
        self.assertAlmostEqual(actual.nmse.item(), 4.0, places=7)
        self.assertAlmostEqual(actual.score.item(), 0.84, places=7)

    def test_torch_metrics_match_numpy(self) -> None:
        expected = competition_metrics(self.pred_np, self.target_np, self.layout)
        actual = self._metrics(self.pred_np)

        self.assertAlmostEqual(actual.pas.item(), expected.pas, places=5)
        self.assertAlmostEqual(actual.pdp.item(), expected.pdp, places=5)
        self.assertAlmostEqual(actual.nmse.item(), expected.nmse, places=5)
        self.assertAlmostEqual(actual.score.item(), expected.score, places=5)

    def test_torch_metric_loss_has_finite_gradient(self) -> None:
        prediction = torch.from_numpy(self.pred_np).requires_grad_(True)
        metrics = torch_competition_metrics(
            prediction,
            torch.from_numpy(self.target_np),
            self.config,
            self.layout.order,
        )
        loss = (
            0.4 * (1 - metrics.pas)
            + 0.4 * (1 - metrics.pdp)
            + 0.2 * torch.log1p(metrics.nmse)
        )
        loss.backward()

        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_row_power_cosine_handles_zero_vectors(self) -> None:
        first = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
        second = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float64)

        actual = row_power_cosine(first, second, vector_axes=(-1,))

        self.assertAlmostEqual(actual.item(), 0.5, places=7)

    def test_row_power_cosine_zero_rows_have_finite_gradients(self) -> None:
        first = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64, requires_grad=True
        )
        second = torch.tensor(
            [[0.0, 0.0], [0.0, 0.0]], dtype=torch.float64, requires_grad=True
        )

        row_power_cosine(first, second, vector_axes=(-1,)).backward()

        self.assertIsNotNone(first.grad)
        self.assertIsNotNone(second.grad)
        self.assertTrue(torch.isfinite(first.grad).all())
        self.assertTrue(torch.isfinite(second.grad).all())

    def test_mixed_zero_target_channel_has_finite_gradients(self) -> None:
        from solution.radio_map.learning.torch_metrics import torch_competition_metrics

        target = torch.from_numpy(self.target_np[:2]).clone()
        target[0] = 0
        prediction = (
            torch.from_numpy(self.target_np[:2]) * (0.8 + 0.1j)
        ).clone().requires_grad_(True)
        metrics = torch_competition_metrics(
            prediction, target, self.config, self.layout.order
        )
        loss = 0.4 * (1 - metrics.pas) + 0.4 * (1 - metrics.pdp)
        loss = loss + 0.2 * metrics.nmse / (1 + metrics.nmse)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())


if __name__ == "__main__":
    unittest.main()
