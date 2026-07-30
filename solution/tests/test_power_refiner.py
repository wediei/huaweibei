from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from solution.radio_map.learning.anchor_mixer import MixerOutput
from solution.radio_map.learning.power_refiner_cli import (
    PowerAwareRefiner,
    _load_refiner,
    _sha256,
)


class _Adapter:
    def __init__(self) -> None:
        self.group_ids = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        self.mean = np.asarray([0.1 + 0.05j] * 6, dtype=np.complex64)
        self.rms = np.linspace(0.5, 1.0, 6, dtype=np.float32)
        self.support_indices = np.arange(6, dtype=np.int64)
        self.fitted_indices = np.arange(4, dtype=np.int64)


class _Base(nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> MixerOutput:
        anchors = batch["anchor_latents"]
        batch_size, anchor_count, _ = anchors.shape
        latent = anchors[:, 0]
        return MixerOutput(
            latent=latent,
            nearest_latent=latent,
            anchor_weights=torch.full(
                (batch_size, anchor_count, 3),
                1.0 / anchor_count,
                dtype=anchors.real.dtype,
                device=anchors.device,
            ),
            alpha=torch.full(
                (batch_size, 3),
                0.5,
                dtype=anchors.real.dtype,
                device=anchors.device,
            ),
            complex_gain=torch.ones_like(latent),
            low_rank_residual=torch.zeros_like(latent),
        )


class PowerRefinerV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = _Adapter()

    def test_feature_v1_forward_is_finite(self) -> None:
        refiner = PowerAwareRefiner(
            _Base(), self.adapter, hidden_dim=8, feature_version=1
        )
        output = refiner(
            {
                "anchor_latents": torch.randn(
                    2, 3, 6, dtype=torch.complex64
                ),
                "anchor_distances": torch.tensor(
                    [[1.0, 2.0, float("inf")], [0.5, 1.5, 3.0]]
                ),
                "anchor_mask": torch.tensor(
                    [[True, True, False], [True, True, True]]
                ),
            }
        )
        self.assertEqual(output.latent.shape, (2, 6))
        self.assertTrue(torch.isfinite(output.latent).all())

    def test_legacy_v1_checkpoint_loads_and_rejects_v2_model(self) -> None:
        legacy = PowerAwareRefiner(
            _Base(), self.adapter, hidden_dim=8, feature_version=1
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.pt"
            torch.save(
                {
                    "format_version": 1,
                    "base_checkpoint_sha256": "a" * 64,
                    "hidden_dim": 8,
                    "state": legacy.head_state_dict(),
                    "metrics": {"score": 0.5},
                },
                path,
            )
            path.with_name(path.name + ".sha256").write_text(
                _sha256(path) + "\n", encoding="ascii"
            )
            _load_refiner(path, legacy, "a" * 64)
            incompatible = PowerAwareRefiner(
                _Base(), self.adapter, hidden_dim=8, feature_version=2
            )
            with self.assertRaisesRegex(ValueError, "legacy"):
                _load_refiner(path, incompatible, "a" * 64)


if __name__ == "__main__":
    unittest.main()
