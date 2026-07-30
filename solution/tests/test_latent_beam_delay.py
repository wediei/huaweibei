import unittest

import numpy as np
import torch

from solution.radio_map.config import RoundConfig
from solution.radio_map.learning.latent_adapter import FixedSupportLatentAdapter
from solution.radio_map.transforms import (
    AntennaLayout,
    beam_delay,
    inverse_beam_delay,
)


class LatentBeamDelayTests(unittest.TestCase):
    def _adapter(self):
        config = RoundConfig(
            p_train_declared=5,
            p_test=1,
            m=4,
            m_h=2,
            m_v=2,
            m_p=1,
            n=2,
            n_h=2,
            n_v=1,
            n_p=1,
            s=4,
            q=1,
            bs_position=(0.0, 0.0, 1.0),
            weights=(0.4, 0.4, 0.2),
        )
        layout = AntennaLayout(config)
        rng = np.random.default_rng(7)
        channel = (
            rng.normal(size=(5, 4, 2, 4))
            + 1j * rng.normal(size=(5, 4, 2, 4))
        ).astype(np.complex64)
        adapter = FixedSupportLatentAdapter(
            layout, support_fraction=0.5, delay_block=1
        ).fit(channel, np.arange(4), batch_size=2)
        return adapter, channel

    def test_numpy_direct_expansion_matches_decode_then_transform(self):
        adapter, channel = self._adapter()
        latent = adapter.encode_numpy(channel)
        expected = beam_delay(adapter.decode_numpy(latent), adapter.layout)
        np.testing.assert_allclose(
            adapter.beam_delay_numpy(latent), expected, atol=2e-6, rtol=2e-6
        )

    def test_torch_matches_numpy_and_preserves_gradient(self):
        adapter, channel = self._adapter()
        latent = torch.tensor(
            adapter.encode_numpy(channel[:2]), requires_grad=True
        )
        actual = adapter.beam_delay_torch(latent)
        np.testing.assert_allclose(
            actual.detach().numpy(),
            adapter.beam_delay_numpy(latent.detach().numpy()),
            atol=1e-6,
            rtol=1e-6,
        )
        actual.abs().square().sum().backward()
        self.assertIsNotNone(latent.grad)
        self.assertTrue(torch.isfinite(latent.grad).all())

    def test_torch_inverse_beam_delay_matches_numpy(self):
        adapter, channel = self._adapter()
        transformed = beam_delay(channel[:2], adapter.layout)
        values = torch.tensor(transformed, requires_grad=True)
        actual = adapter.channel_from_beam_delay_torch(values)
        np.testing.assert_allclose(
            actual.detach().numpy(),
            inverse_beam_delay(transformed, adapter.layout),
            atol=2e-6,
            rtol=2e-6,
        )
        actual.abs().square().mean().backward()
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_validation(self):
        adapter, _ = self._adapter()
        with self.assertRaisesRegex(TypeError, "complex"):
            adapter.beam_delay_numpy(
                np.zeros((1, adapter.coefficient_count), np.float32)
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            adapter.beam_delay_torch(torch.zeros(1, 2, dtype=torch.complex64))


if __name__ == "__main__":
    unittest.main()
