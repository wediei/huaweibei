from __future__ import annotations

import unittest

import numpy as np
import torch

from solution.radio_map.learning.path_transport import (
    TransportParameters,
    apply_transport_numpy,
    apply_transport_torch,
    blend_transport,
    fractional_circular_shift_numpy,
    fractional_circular_shift_torch,
)


class PathTransportTests(unittest.TestCase):
    def test_numpy_identity_and_integer_shift(self) -> None:
        rng = np.random.default_rng(3)
        values = (
            rng.standard_normal((2, 5, 4))
            + 1j * rng.standard_normal((2, 5, 4))
        ).astype(np.complex64)
        identity = fractional_circular_shift_numpy(values, 0.0, axis=1)
        np.testing.assert_array_equal(identity, values)
        shifted = fractional_circular_shift_numpy(values, 2.0, axis=1)
        np.testing.assert_allclose(
            shifted, np.roll(values, 2, axis=1), rtol=2e-6, atol=2e-6
        )

    def test_fractional_shift_is_reversible_and_energy_preserving(self) -> None:
        rng = np.random.default_rng(4)
        values = (
            rng.standard_normal((2, 7, 5))
            + 1j * rng.standard_normal((2, 7, 5))
        ).astype(np.complex128)
        shifted = fractional_circular_shift_numpy(values, 0.375, axis=1)
        restored = fractional_circular_shift_numpy(shifted, -0.375, axis=1)
        np.testing.assert_allclose(restored, values, rtol=1e-11, atol=1e-11)
        self.assertAlmostEqual(
            float(np.sum(np.abs(shifted) ** 2)),
            float(np.sum(np.abs(values) ** 2)),
            places=10,
        )

    def test_torch_shift_has_finite_gradient_and_matches_numpy(self) -> None:
        rng = np.random.default_rng(5)
        source = (
            rng.standard_normal((2, 6))
            + 1j * rng.standard_normal((2, 6))
        ).astype(np.complex64)
        values = torch.tensor(source, requires_grad=True)
        shift = torch.tensor(0.4, requires_grad=True)
        result = fractional_circular_shift_torch(values, shift, axis=1)
        expected = fractional_circular_shift_numpy(source, 0.4, axis=1)
        np.testing.assert_allclose(
            result.detach().numpy(), expected, rtol=2e-5, atol=2e-5
        )
        result.abs().square().sum().backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertTrue(torch.isfinite(shift.grad))

    def test_dense_transport_does_not_mix_p_or_n(self) -> None:
        values = np.zeros((1, 3, 4, 2, 2, 5), dtype=np.complex64)
        values[0, 0, 0, 1, 1, 0] = 1
        parameters = TransportParameters(
            delta_h=1.0,
            delta_v=-1.0,
            delta_delay=2.0,
        )
        output = apply_transport_numpy(values, parameters)
        self.assertAlmostEqual(float(np.abs(output[0, 1, 3, 1, 1, 2])), 1.0, places=5)
        self.assertEqual(np.count_nonzero(np.abs(output[:, :, :, 0, :, :]) > 1e-5), 0)
        torch_output = apply_transport_torch(
            torch.from_numpy(values), parameters
        ).numpy()
        np.testing.assert_allclose(torch_output, output, rtol=2e-5, atol=2e-5)

    def test_amplitude_phase_and_reliability_are_bounded(self) -> None:
        values = np.ones((1, 2, 2, 1, 1, 2), dtype=np.complex64)
        parameters = TransportParameters(
            log_amplitude=np.log(2.0),
            phase_real=0.0,
            phase_imag=3.0,
            existence=1.0,
            reliability=0.5,
        )
        output = apply_transport_numpy(values, parameters)
        np.testing.assert_allclose(output, 0.5 + 1.0j, atol=1e-6)

    def test_scale_zero_does_not_call_transport(self) -> None:
        coarse = np.asarray([1 + 2j], dtype=np.complex64)

        def forbidden():
            raise AssertionError("transport must not run")

        actual = blend_transport(coarse, 0.0, forbidden)
        np.testing.assert_array_equal(actual, coarse)


if __name__ == "__main__":
    unittest.main()
