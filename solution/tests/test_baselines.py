from __future__ import annotations

import unittest

import numpy as np


class LocalBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.positions = np.array(
            [[0.0, 0.0, 1.5], [2.0, 0.0, 1.5], [5.0, 0.0, 1.5]],
            dtype=np.float64,
        )
        values = np.array([1.0 + 1.0j, 3.0 + 3.0j, 9.0 + 9.0j])
        self.channels = np.broadcast_to(values[:, None, None, None], (3, 2, 1, 2))
        self.channels = self.channels.astype(np.complex64).copy()

    def test_nearest_anchor_copies_exact_channel(self) -> None:
        from solution.radio_map.baselines import NearestAnchorRegressor

        model = NearestAnchorRegressor().fit(self.positions, self.channels)
        prediction = model.predict(
            np.array([[0.2, 0.0, 1.5], [4.7, 0.0, 1.5]])
        )

        np.testing.assert_array_equal(prediction[0], self.channels[0])
        np.testing.assert_array_equal(prediction[1], self.channels[2])
        self.assertEqual(prediction.dtype, np.complex64)

    def test_channel_indices_address_global_source(self) -> None:
        from solution.radio_map.baselines import NearestAnchorRegressor

        model = NearestAnchorRegressor().fit(
            self.positions[[0, 2]], self.channels, channel_indices=np.array([0, 2])
        )
        prediction = model.predict(np.array([[4.8, 0.0, 1.5]]))

        np.testing.assert_array_equal(prediction[0], self.channels[2])

    def test_inverse_distance_is_exact_at_anchor_and_averages_midpoint(self) -> None:
        from solution.radio_map.baselines import InverseDistanceRegressor

        model = InverseDistanceRegressor(k=2, power=2.0).fit(
            self.positions[:2], self.channels[:2]
        )
        prediction = model.predict(
            np.array([[0.0, 0.0, 1.5], [1.0, 0.0, 1.5]])
        )

        np.testing.assert_array_equal(prediction[0], self.channels[0])
        np.testing.assert_allclose(prediction[1], 2.0 + 2.0j, rtol=1e-6)

    def test_predict_batches_preserves_order(self) -> None:
        from solution.radio_map.baselines import NearestAnchorRegressor

        queries = np.array(
            [[4.9, 0.0, 1.5], [0.1, 0.0, 1.5], [2.1, 0.0, 1.5]]
        )
        model = NearestAnchorRegressor().fit(self.positions, self.channels)

        prediction = np.concatenate(list(model.predict_batches(queries, batch_size=2)))

        np.testing.assert_array_equal(prediction[:, 0, 0, 0], [9 + 9j, 1 + 1j, 3 + 3j])


if __name__ == "__main__":
    unittest.main()
