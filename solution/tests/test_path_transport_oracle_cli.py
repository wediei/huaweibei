from __future__ import annotations

import unittest

import numpy as np

from solution.radio_map.config import RoundConfig
from solution.radio_map.learning.path_transport import (
    TransportParameters,
    apply_transport_numpy,
)
from solution.radio_map.learning.path_transport_oracle_cli import (
    audit_transport_ceiling,
    fit_group_transport_parameters,
)
from solution.radio_map.transforms import AntennaLayout, inverse_beam_delay


class PathTransportOracleTests(unittest.TestCase):
    def test_known_shift_and_complex_gain_are_recovered(self) -> None:
        config = RoundConfig(
            p_train_declared=2,
            p_test=1,
            m=16,
            m_h=4,
            m_v=4,
            m_p=1,
            n=1,
            n_h=1,
            n_v=1,
            n_p=1,
            s=8,
            q=1,
            bs_position=(0.0, 0.0, 1.0),
            weights=(0.4, 0.4, 0.2),
        )
        layout = AntennaLayout(config)
        reference_bd = np.zeros((2, 4, 4, 1, 1, 8), dtype=np.complex64)
        reference_bd[0, 0, 1, 0, 0, 2] = 1 + 0.5j
        reference_bd[0, 2, 3, 0, 0, 5] = 0.3 - 0.2j
        reference_bd[1, 1, 0, 0, 0, 1] = 2 - 0.5j
        reference_bd[1, 3, 2, 0, 0, 4] = -0.4 + 0.7j
        target_bd = apply_transport_numpy(
            reference_bd,
            TransportParameters(
                delta_h=1.0,
                delta_v=-1.0,
                delta_delay=2.0,
                log_amplitude=np.log(1.7),
                phase_real=0.0,
                phase_imag=1.0,
            ),
        )
        report = audit_transport_ceiling(
            inverse_beam_delay(reference_bd, layout),
            inverse_beam_delay(target_bd, layout),
            layout,
            max_h_shift=1,
            max_v_shift=1,
            max_delay_shift=2,
            batch_size=1,
        )
        self.assertEqual(report["kind"], "path_transport_o2_target_visible_ceiling")
        stages = report["stages"]
        self.assertLess(stages["identity"]["score"], stages["beam_delay_shift"]["score"])
        self.assertAlmostEqual(stages["amplitude"]["score"], 1.0, places=5)
        self.assertGreaterEqual(
            stages["reliability"]["score"], stages["amplitude"]["score"] - 1e-6
        )
        self.assertEqual(report["sample_count"], 2)
        self.assertTrue(report["target_visible"])
        labels = fit_group_transport_parameters(
            reference_bd, target_bd, 1, 1, 2
        )
        np.testing.assert_array_equal(labels["delta_h"], 1.0)
        np.testing.assert_array_equal(labels["delta_v"], -1.0)
        np.testing.assert_array_equal(labels["delta_delay"], 2.0)
        np.testing.assert_allclose(
            np.exp(labels["log_amplitude"]), 1.7, rtol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
