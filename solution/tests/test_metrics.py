from __future__ import annotations

import unittest

import numpy as np

from solution.radio_map.transforms import AntennaLayout
from solution.tests.test_transforms import make_config


class CompetitionMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = make_config()
        self.layout = AntennaLayout(self.config, ("H", "V", "P"))
        rng = np.random.default_rng(19)
        self.channel = (
            rng.standard_normal((5, 8, 2, 4))
            + 1j * rng.standard_normal((5, 8, 2, 4))
        ).astype(np.complex64)

    def test_identity_prediction_scores_one(self) -> None:
        from solution.radio_map.metrics import competition_metrics

        result = competition_metrics(
            self.channel, self.channel, self.layout, self.config.weights
        )

        self.assertAlmostEqual(result.pas, 1.0, places=6)
        self.assertAlmostEqual(result.pdp, 1.0, places=6)
        self.assertAlmostEqual(result.nmse, 0.0, places=7)
        self.assertAlmostEqual(result.score, 1.0, places=6)

    def test_scaled_prediction_keeps_cosines_and_has_expected_nmse(self) -> None:
        from solution.radio_map.metrics import competition_metrics

        result = competition_metrics(
            self.channel * np.float32(0.9),
            self.channel,
            self.layout,
            self.config.weights,
        )

        self.assertAlmostEqual(result.pas, 1.0, places=6)
        self.assertAlmostEqual(result.pdp, 1.0, places=6)
        self.assertAlmostEqual(result.nmse, 0.01, places=6)
        self.assertAlmostEqual(result.score, 0.8 + 0.2 / 1.01, places=6)

    def test_streaming_matches_single_batch(self) -> None:
        from solution.radio_map.metrics import MetricAccumulator, competition_metrics

        prediction = self.channel * np.complex64(0.8 + 0.1j)
        expected = competition_metrics(
            prediction, self.channel, self.layout, self.config.weights
        )
        accumulator = MetricAccumulator(self.layout, self.config.weights)
        accumulator.update(prediction[:2], self.channel[:2])
        accumulator.update(prediction[2:], self.channel[2:])
        actual = accumulator.compute()

        self.assertAlmostEqual(actual.pas, expected.pas, places=7)
        self.assertAlmostEqual(actual.pdp, expected.pdp, places=7)
        self.assertAlmostEqual(actual.nmse, expected.nmse, places=7)
        self.assertAlmostEqual(actual.score, expected.score, places=7)

    def test_zero_prediction_is_finite(self) -> None:
        from solution.radio_map.metrics import competition_metrics

        result = competition_metrics(
            np.zeros_like(self.channel),
            self.channel,
            self.layout,
            self.config.weights,
        )

        self.assertTrue(np.isfinite(result.score))
        self.assertAlmostEqual(result.pas, 0.0, places=7)
        self.assertAlmostEqual(result.pdp, 0.0, places=7)
        self.assertAlmostEqual(result.nmse, 1.0, places=7)
        self.assertAlmostEqual(result.score, 0.1, places=7)

    def test_phase_inversion_preserves_power_metrics(self) -> None:
        from solution.radio_map.metrics import competition_metrics

        result = competition_metrics(
            -self.channel, self.channel, self.layout, self.config.weights
        )

        self.assertAlmostEqual(result.pas, 1.0, places=6)
        self.assertAlmostEqual(result.pdp, 1.0, places=6)
        self.assertAlmostEqual(result.nmse, 4.0, places=6)
        self.assertAlmostEqual(result.score, 0.84, places=6)

    def test_metric_rejects_shape_mismatch(self) -> None:
        from solution.radio_map.metrics import competition_metrics

        with self.assertRaisesRegex(ValueError, "same shape"):
            competition_metrics(
                self.channel[:3], self.channel, self.layout, self.config.weights
            )


if __name__ == "__main__":
    unittest.main()
