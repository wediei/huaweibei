import unittest

import torch

from solution.radio_map.learning.ac_cgmrf import (
    ACCGMRF,
    ACCGMRFConfig,
    MultipathAtoms,
    continuous_complex_splat,
    continuous_delay_phase_ramp,
    support_novelty_ratio,
)


class ACCGMRFTests(unittest.TestCase):
    @staticmethod
    def _inputs():
        torch.manual_seed(7)
        base = torch.randn(2, 5, 4, 2, 2, 8, dtype=torch.complex64)
        anchors = torch.randn(
            2, 4, 5, 4, 2, 2, 8, dtype=torch.complex64
        )
        anchor_positions = torch.randn(2, 4, 3)
        target_positions = torch.randn(2, 3)
        target_tokens = torch.randn(2, 3, 6)
        anchor_tokens = torch.randn(2, 4, 3, 6)
        direct_tokens = torch.randn(2, 4, 3, 6)
        path_mask = torch.ones(2, 4, 3, dtype=torch.bool)
        anchor_mask = torch.ones(2, 4, dtype=torch.bool)
        return (
            base,
            anchors,
            anchor_positions,
            target_positions,
            target_tokens,
            anchor_tokens,
            direct_tokens,
            path_mask,
            anchor_mask,
        )

    @staticmethod
    def _model():
        torch.manual_seed(11)
        return ACCGMRF(
            ACCGMRFConfig(
                p_count=2,
                n_count=2,
                path_feature_dim=6,
                atom_count=4,
                d_model=24,
                map_hidden_dim=20,
                support_h=1,
                support_v=1,
                support_delay=1,
            )
        )

    def test_zero_gate_zero_mode_and_disabled_are_exact_o41(self):
        model = self._model()
        values = self._inputs()
        real = model(*values, mode="real")
        zero = model(*values, mode="zero")
        disabled = model(*values, mode="real", enabled=False)
        self.assertTrue(torch.equal(real.prediction, values[0]))
        self.assertTrue(torch.equal(zero.prediction, values[0]))
        self.assertTrue(torch.equal(disabled.prediction, values[0]))
        self.assertEqual(real.prediction.dtype, torch.complex64)
        self.assertTrue(torch.isfinite(real.raw_residual).all())
        self.assertLessEqual(
            float((disabled.prediction - values[0]).abs().max()), 1e-6
        )

    def test_shape_dtype_finite_gradient_and_low_rank_coupling(self):
        model = self._model()
        values = self._inputs()
        output = model(*values, mode="real")
        self.assertEqual(output.prediction.shape, values[0].shape)
        self.assertEqual(output.residual.shape, values[0].shape)
        self.assertTrue(torch.isfinite(output.prediction).all())
        self.assertEqual(output.atoms.centers.shape, (2, 4, 3))
        coupling = output.atoms.coupling
        determinant = (
            coupling[..., 0, 0] * coupling[..., 1, 1]
            - coupling[..., 0, 1] * coupling[..., 1, 0]
        )
        torch.testing.assert_close(
            determinant, torch.zeros_like(determinant), atol=2e-6, rtol=2e-6
        )
        output.prediction.real.sum().backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
        self.assertTrue(
            any(float(value.abs().sum()) > 0 for value in gradients)
        )

    def test_deterministic_and_shuffle_matches_explicit_map_permutation(self):
        model = self._model()
        with torch.no_grad():
            model.trust_head[-1].bias.fill_(0.5)
        values = self._inputs()
        first = model(*values, mode="real")
        second = model(*values, mode="real")
        torch.testing.assert_close(first.prediction, second.prediction)
        permutation = torch.tensor([1, 0], dtype=torch.long)
        shuffled = model(
            *values, mode="shuffle", map_permutation=permutation
        )
        explicit_values = list(values)
        for index in (4, 5, 6, 7):
            explicit_values[index] = explicit_values[index][permutation]
        explicit = model(*explicit_values, mode="real")
        torch.testing.assert_close(shuffled.prediction, explicit.prediction)


class ComplexSplatTests(unittest.TestCase):
    @staticmethod
    def _atoms(
        gains: torch.Tensor, centers: torch.Tensor | None = None
    ) -> MultipathAtoms:
        batch, count = gains.shape
        if centers is None:
            centers = torch.tensor(
                [[[1.25, 1.5, 2.25]]], dtype=torch.float32
            ).expand(batch, count, -1).clone()
        sigmas = torch.ones(batch, count, 3)
        coupling = torch.ones(batch, count, 1, 1, dtype=torch.complex64)
        reference = torch.ones_like(coupling)
        existence = torch.ones(batch, count)
        reliability = torch.ones(batch, count)
        return MultipathAtoms(
            centers=centers,
            sigmas=sigmas,
            complex_gain=gains,
            coupling=coupling,
            phase_reference=reference,
            existence=existence,
            reliability=reliability,
            delay_phase_ramp=continuous_delay_phase_ramp(
                centers[..., 2], 6
            ),
        )

    def test_coherent_addition_and_cancellation(self):
        one = self._atoms(torch.ones(1, 1, dtype=torch.complex64))
        twice = self._atoms(torch.ones(1, 2, dtype=torch.complex64))
        single = continuous_complex_splat(one, (4, 4, 1, 1, 6), (1, 1, 1))
        doubled = continuous_complex_splat(
            twice, (4, 4, 1, 1, 6), (1, 1, 1)
        )
        torch.testing.assert_close(doubled, 2.0 * single)
        cancelling = self._atoms(
            torch.tensor([[1.0 + 0j, -1.0 + 0j]], dtype=torch.complex64)
        )
        cancelled = continuous_complex_splat(
            cancelling, (4, 4, 1, 1, 6), (1, 1, 1)
        )
        torch.testing.assert_close(
            cancelled, torch.zeros_like(cancelled), atol=1e-6, rtol=1e-6
        )

    def test_splat_is_differentiable_in_location_and_complex_gain(self):
        centers = torch.tensor(
            [[[1.25, 1.5, 2.25]]], requires_grad=True
        )
        gain_real = torch.tensor([[0.8]], requires_grad=True)
        gain_imag = torch.tensor([[0.2]], requires_grad=True)
        atoms = self._atoms(torch.complex(gain_real, gain_imag), centers)
        result = continuous_complex_splat(
            atoms, (4, 4, 1, 1, 6), (1, 1, 1)
        )
        result.abs().square().sum().backward()
        for gradient in (centers.grad, gain_real.grad, gain_imag.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_continuous_delay_uses_exact_dft_phase_ramp(self):
        centers = torch.tensor([[1.5]])
        ramp = continuous_delay_phase_ramp(centers, 8)
        ratio = ramp[..., 1:] * ramp[..., :-1].conj()
        expected = torch.polar(
            torch.ones_like(centers),
            -2.0 * torch.pi * centers / 8.0,
        )[..., None]
        torch.testing.assert_close(ratio, expected.expand_as(ratio))

    def test_support_novelty_detects_new_beam_delay_cells(self):
        anchors = torch.zeros(1, 2, 4, 4, 1, 1, 6, dtype=torch.complex64)
        anchors[:, :, 0, 0, 0, 0, 0] = 1.0
        residual = torch.zeros(1, 4, 4, 1, 1, 6, dtype=torch.complex64)
        residual[:, 3, 3, 0, 0, 5] = 2.0
        novelty = support_novelty_ratio(
            residual,
            anchors,
            torch.ones(1, 2, dtype=torch.bool),
            threshold=1e-3,
        )
        torch.testing.assert_close(novelty, torch.ones_like(novelty))


if __name__ == "__main__":
    unittest.main()
