from __future__ import annotations

import unittest

import numpy as np

from solution.radio_map.config import RoundConfig


def make_config() -> RoundConfig:
    return RoundConfig(
        p_train_declared=8,
        p_test=2,
        m=8,
        m_h=2,
        m_v=2,
        m_p=2,
        n=2,
        n_h=1,
        n_v=1,
        n_p=2,
        s=4,
        q=2,
        bs_position=(0.0, 0.0, 5.0),
        weights=(0.4, 0.4, 0.2),
    )


class BeamDelayTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = make_config()
        rng = np.random.default_rng(11)
        self.channels = (
            rng.standard_normal((3, 8, 2, 4))
            + 1j * rng.standard_normal((3, 8, 2, 4))
        ).astype(np.complex64)

    def test_candidate_orders_are_all_six_permutations(self) -> None:
        from solution.radio_map.transforms import candidate_orders

        orders = candidate_orders()

        self.assertEqual(len(orders), 6)
        self.assertEqual(len(set(orders)), 6)
        self.assertTrue(all(set(order) == {"H", "V", "P"} for order in orders))

    def test_layout_roundtrip_preserves_values_for_every_order(self) -> None:
        from solution.radio_map.transforms import AntennaLayout, candidate_orders

        for order in candidate_orders():
            with self.subTest(order=order):
                layout = AntennaLayout(self.config, order)
                reconstructed = layout.from_structured(
                    layout.to_structured(self.channels)
                )
                np.testing.assert_array_equal(reconstructed, self.channels)

    def test_beam_delay_roundtrip_for_every_order(self) -> None:
        from solution.radio_map.transforms import (
            AntennaLayout,
            beam_delay,
            candidate_orders,
            inverse_beam_delay,
        )

        for order in candidate_orders():
            with self.subTest(order=order):
                layout = AntennaLayout(self.config, order)
                reconstructed = inverse_beam_delay(
                    beam_delay(self.channels, layout), layout
                )
                np.testing.assert_allclose(
                    reconstructed, self.channels, rtol=1e-5, atol=1e-6
                )

    def test_beam_delay_preserves_energy(self) -> None:
        from solution.radio_map.transforms import AntennaLayout, beam_delay

        layout = AntennaLayout(self.config, ("H", "V", "P"))
        transformed = beam_delay(self.channels, layout)

        np.testing.assert_allclose(
            np.sum(np.abs(transformed) ** 2),
            np.sum(np.abs(self.channels) ** 2),
            rtol=1e-5,
        )

    def test_layout_rejects_wrong_channel_tail(self) -> None:
        from solution.radio_map.transforms import AntennaLayout

        layout = AntennaLayout(self.config, ("H", "V", "P"))
        wrong = np.zeros((3, 7, 2, 4), dtype=np.complex64)

        with self.assertRaisesRegex(ValueError, "expected channel tail"):
            layout.to_structured(wrong)


if __name__ == "__main__":
    unittest.main()
