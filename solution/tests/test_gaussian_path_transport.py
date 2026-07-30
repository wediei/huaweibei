from __future__ import annotations

import unittest

import torch

from solution.radio_map.learning.gaussian_path_transport import (
    GaussianPathTransport,
    GaussianPathTransportConfig,
    apply_optional_gaussian_transport,
)


class GaussianPathTransportTests(unittest.TestCase):
    def test_zero_initialized_head_preserves_coarse_and_masks_padding(self) -> None:
        torch.manual_seed(3)
        config = GaussianPathTransportConfig(
            p_count=2,
            n_count=2,
            map_feature_dim=27,
            d_model=32,
            heads=4,
            layers=2,
            max_h_shift=1.5,
            max_v_shift=1.5,
            max_delay_shift=4.0,
        )
        model = GaussianPathTransport(config)
        coarse = torch.randn(2, 4, 4, 2, 2, 8, dtype=torch.complex64)
        tokens = torch.randn(2, 6, 27)
        mask = torch.tensor(
            [[True, True, False, False, False, False], [False] * 6]
        )
        output = model(coarse, tokens, mask)
        self.assertEqual(output.transported.shape, coarse.shape)
        self.assertTrue(torch.isfinite(output.transported).all())
        torch.testing.assert_close(output.transported, coarse, rtol=2e-5, atol=2e-5)
        self.assertEqual(output.attention.shape, (2, 4, 7))
        self.assertTrue(torch.isfinite(output.attention).all())
        self.assertLessEqual(
            float(output.parameters.delta_h.detach().abs().max()), 1.5
        )
        self.assertLessEqual(
            float(output.parameters.delta_delay.detach().abs().max()), 4.0
        )

    def test_causal_mode_zero_map_is_exact_identity_after_training(self) -> None:
        config = GaussianPathTransportConfig(
            p_count=1,
            n_count=2,
            map_feature_dim=3,
            d_model=16,
            heads=2,
            causal_map_residual=True,
        )
        model = GaussianPathTransport(config)
        with torch.no_grad():
            model.parameter_head.weight.normal_()
            model.parameter_head.bias.normal_()
        coarse = torch.randn(
            2, 2, 2, 1, 2, 4, dtype=torch.complex64
        )
        output = model(
            coarse,
            torch.zeros(2, 3, 3),
            torch.ones(2, 3, dtype=torch.bool),
        )
        self.assertTrue(torch.equal(output.coarse, output.transported))

    def test_scale_zero_hard_bypass_does_not_call_model(self) -> None:
        coarse = torch.randn(1, 2, 2, 1, 1, 3, dtype=torch.complex64)

        class Forbidden:
            def __call__(self, *_):
                raise AssertionError("model must not run")

        output = apply_optional_gaussian_transport(
            Forbidden(), coarse, None, None, scale=0.0
        )
        self.assertTrue(torch.equal(output, coarse))

    def test_invalid_mask_and_dimensions_are_rejected(self) -> None:
        model = GaussianPathTransport(
            GaussianPathTransportConfig(1, 1, 5, 16, 4, 1, 1, 1, 1)
        )
        coarse = torch.zeros(1, 2, 2, 1, 1, 3, dtype=torch.complex64)
        with self.assertRaisesRegex(ValueError, "map token"):
            model(coarse, torch.zeros(1, 3, 4), torch.ones(1, 3, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "mask"):
            model(coarse, torch.zeros(1, 3, 5), torch.ones(1, 2, dtype=torch.bool))

    def test_learnable_radio_gaussians_splat_and_zero_map_bypasses(self) -> None:
        config = GaussianPathTransportConfig(
            p_count=1,
            n_count=1,
            map_feature_dim=27,
            d_model=16,
            heads=2,
            layers=1,
            causal_map_residual=True,
            gaussian_count=6,
            radio_feature_dim=8,
        )
        model = GaussianPathTransport(
            config, torch.zeros(27), torch.ones(27)
        )
        with torch.no_grad():
            model.parameter_head.weight.normal_(std=0.1)
            model.radio_codes.weight[1:5].normal_(std=0.05)
        coarse = torch.randn(2, 2, 2, 1, 1, 4, dtype=torch.complex64)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        indices = torch.tensor([[0, 2, -1], [1, 3, 5]])
        tokens = torch.zeros(2, 3, 27)
        tokens[..., 15] = 0.4
        tokens[..., 16] = 0.8
        tokens[..., 26] = 1.0
        output = model(coarse, tokens, mask, indices)
        loss = (output.transported - coarse).abs().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(model.radio_codes.weight.grad).all())
        self.assertGreater(
            float(model.radio_codes.weight.grad[1:].abs().sum()), 0.0
        )
        self.assertGreater(
            float(model.radio_log_opacity.weight.grad[1:].abs().sum()), 0.0
        )
        zero = model(coarse, torch.zeros_like(tokens), mask, indices)
        self.assertTrue(torch.equal(zero.transported, coarse))
        with self.assertRaisesRegex(ValueError, "requires Gaussian indices"):
            model(coarse, tokens, mask)
        restored = GaussianPathTransport(config)
        restored.load_state_dict(model.state_dict(), strict=True)
        restored_output = restored(coarse, tokens, mask, indices)
        torch.testing.assert_close(
            restored_output.transported, output.transported
        )

    def test_multiscale_radio_grid_shares_codes_between_gaussians(self) -> None:
        config = GaussianPathTransportConfig(
            p_count=1,
            n_count=1,
            map_feature_dim=27,
            d_model=16,
            heads=2,
            layers=1,
            causal_map_residual=True,
            gaussian_count=4,
            radio_feature_dim=6,
            radio_grid_sizes=(4, 3),
        )
        # Padding row followed by four physical Gaussians. Gaussians 0/1
        # intentionally share both multiscale cells.
        mapping = torch.tensor(
            [[0, 0], [1, 1], [1, 1], [2, 1], [3, 2]]
        )
        model = GaussianPathTransport(
            config,
            torch.zeros(27),
            torch.ones(27),
            mapping,
        )
        with torch.no_grad():
            model.parameter_head.weight.normal_(std=0.1)
            for table in model.radio_codes:
                table.weight[1:].normal_(std=0.05)
        coarse = torch.randn(1, 2, 2, 1, 1, 4, dtype=torch.complex64)
        tokens = torch.zeros(1, 3, 27)
        tokens[..., 15] = 0.5
        tokens[..., 16] = 0.8
        tokens[..., 26] = 1.0
        mask = torch.tensor([[True, True, True]])
        indices = torch.tensor([[0, 1, 3]])
        output = model(coarse, tokens, mask, indices)
        (output.transported - coarse).abs().square().mean().backward()
        for table in model.radio_codes:
            self.assertGreater(float(table.weight.grad.abs().sum()), 0.0)
        zero = model(coarse, torch.zeros_like(tokens), mask, indices)
        self.assertTrue(torch.equal(zero.transported, coarse))
        restored = GaussianPathTransport(config)
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(torch.equal(restored.gaussian_grid_ids, mapping))


if __name__ == "__main__":
    unittest.main()
