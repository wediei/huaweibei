import unittest

import torch

from solution.radio_map.learning.gaussian_anchor_transport import (
    GaussianAnchorTransport,
    GaussianAnchorTransportConfig,
)


class GaussianAnchorTransportTests(unittest.TestCase):
    @staticmethod
    def _inputs():
        coarse = torch.randn(2, 3, 2, 1, 2, 4, dtype=torch.complex64)
        anchors = torch.randn(2, 3, 3, 2, 1, 2, 4, dtype=torch.complex64)
        tokens = torch.randn(2, 3, 5, 9)
        token_mask = torch.ones(2, 3, 5, dtype=torch.bool)
        distances = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.5, 4.0]])
        anchor_mask = torch.ones(2, 3, dtype=torch.bool)
        return coarse, anchors, tokens, token_mask, distances, anchor_mask

    def test_zero_map_is_exact_frozen_coarse_after_training(self):
        model = GaussianAnchorTransport(
            GaussianAnchorTransportConfig(
                p_count=1,
                n_count=2,
                map_feature_dim=9,
                d_model=16,
                heads=2,
            )
        )
        coarse, anchors, tokens, token_mask, distances, anchor_mask = (
            self._inputs()
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        output = model(
            coarse, anchors, tokens, token_mask, distances, anchor_mask
        )
        loss = output.transported.abs().square().mean()
        loss.backward()
        optimizer.step()
        zero = model(
            coarse,
            anchors,
            torch.zeros_like(tokens),
            token_mask,
            distances,
            anchor_mask,
        )
        self.assertTrue(torch.equal(zero.transported, coarse))

    def test_real_map_backpropagates_and_weights_are_distance_normalized(self):
        model = GaussianAnchorTransport(
            GaussianAnchorTransportConfig(
                p_count=1,
                n_count=2,
                map_feature_dim=9,
                d_model=16,
                heads=2,
            )
        )
        coarse, anchors, tokens, token_mask, distances, anchor_mask = (
            self._inputs()
        )
        output = model(
            coarse, anchors, tokens, token_mask, distances, anchor_mask
        )
        self.assertEqual(output.transported.shape, coarse.shape)
        self.assertEqual(output.parameters.delta_h.shape, (2, 3, 1, 2))
        torch.testing.assert_close(
            output.anchor_weights.sum(dim=1), torch.ones(2)
        )
        self.assertGreater(
            float(output.anchor_weights[0, 0]),
            float(output.anchor_weights[0, 1]),
        )
        output.transported.abs().square().mean().backward()
        self.assertTrue(
            all(
                parameter.grad is None
                or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_rejects_invalid_pair_shape(self):
        model = GaussianAnchorTransport(
            GaussianAnchorTransportConfig(
                p_count=1, n_count=2, map_feature_dim=9, d_model=16, heads=2
            )
        )
        values = list(self._inputs())
        values[2] = torch.randn(2, 3, 5, 8)
        with self.assertRaisesRegex(ValueError, "context"):
            model(*values)

    def test_groupwise_learned_fusion_compares_anchors_and_zero_bypasses(self):
        model = GaussianAnchorTransport(
            GaussianAnchorTransportConfig(
                p_count=1,
                n_count=2,
                map_feature_dim=9,
                d_model=16,
                heads=2,
                learned_anchor_fusion=True,
            )
        )
        values = self._inputs()
        output = model(*values)
        self.assertEqual(output.anchor_weights.shape, (2, 3, 1, 2))
        torch.testing.assert_close(
            output.anchor_weights.sum(dim=1), torch.ones(2, 1, 2)
        )
        output.transported.abs().square().mean().backward()
        self.assertTrue(
            all(
                parameter.grad is None
                or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )
        zero_values = list(values)
        zero_values[2] = torch.zeros_like(zero_values[2])
        zero = model(*zero_values)
        self.assertTrue(torch.equal(zero.transported, values[0]))


if __name__ == "__main__":
    unittest.main()
