from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from solution.radio_map.learning.anchor_mixer import MixerOutput
from solution.radio_map.learning.coarse_provider import (
    CoarseSpec,
    FrozenCoarseProvider,
    apply_optional_correction,
)
from solution.radio_map.learning.power_refiner_cli import PowerAwareRefiner


class _Adapter:
    group_ids = np.asarray([0, 0, 1, 1], dtype=np.int64)
    mean = np.zeros(4, dtype=np.complex64)
    rms = np.ones(4, dtype=np.float32)
    support_indices = np.arange(4, dtype=np.int64)
    fitted_indices = np.asarray([0, 1], dtype=np.int64)


class _Base(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, batch):
        anchors = batch["anchor_latents"]
        latent = anchors[:, 0] + self.dummy.to(anchors.dtype)
        batch_size, width = latent.shape
        return MixerOutput(
            latent=latent,
            nearest_latent=anchors[:, 0],
            anchor_weights=torch.full(
                (batch_size, anchors.shape[1], 2),
                1.0 / anchors.shape[1],
                dtype=anchors.real.dtype,
            ),
            alpha=torch.full((batch_size, 2), 0.5),
            complex_gain=torch.ones(batch_size, 2, dtype=anchors.dtype),
            low_rank_residual=torch.zeros(
                batch_size, width, dtype=anchors.dtype
            ),
        )


def _batch() -> dict[str, torch.Tensor]:
    return {
        "anchor_latents": torch.tensor(
            [
                [
                    [1 + 1j, 2 + 0j, 3 - 1j, 4 + 2j],
                    [2 + 0j, 1 + 1j, 2 + 2j, 1 - 1j],
                ]
            ],
            dtype=torch.complex64,
        ),
        "anchor_distances": torch.tensor([[1.0, 2.0]]),
        "anchor_mask": torch.ones(1, 2, dtype=torch.bool),
    }


class CoarseProviderTests(unittest.TestCase):
    def test_spec_fingerprint_is_canonical_and_binds_identity(self) -> None:
        base = Path(__file__)
        refiner = Path(__file__).with_name("test_power_refiner.py")
        first = CoarseSpec(
            base_checkpoint=base,
            refiner_checkpoint=refiner,
            power_scale=1.25,
            anchor_source="all_official_train",
        )
        initial = first.fingerprint("adapter", "fold")
        self.assertEqual(initial, dataclasses.replace(first).fingerprint("adapter", "fold"))
        self.assertNotEqual(
            initial,
            dataclasses.replace(first, power_scale=1.0).fingerprint("adapter", "fold"),
        )
        self.assertNotEqual(
            initial,
            dataclasses.replace(first, anchor_source="fold").fingerprint("adapter", "fold"),
        )
        self.assertNotEqual(
            initial,
            dataclasses.replace(first, refiner_checkpoint=base).fingerprint(
                "adapter", "fold"
            ),
        )

    def test_missing_checkpoint_is_rejected(self) -> None:
        missing = Path(__file__).with_name("definitely_missing_base.pt")
        spec = CoarseSpec(base_checkpoint=missing)
        with self.assertRaisesRegex(FileNotFoundError, "base checkpoint"):
            spec.identity("adapter", "fold")

    def test_provider_matches_existing_refiner_scaling_elementwise(self) -> None:
        torch.manual_seed(7)
        adapter = _Adapter()
        refiner = PowerAwareRefiner(_Base(), adapter, hidden_dim=8)
        batch = _batch()
        direct = refiner(batch)
        base_latent = direct.latent - direct.low_rank_residual
        expected = dataclasses.replace(
            direct,
            latent=base_latent + 1.25 * direct.low_rank_residual,
        )
        provider = FrozenCoarseProvider.from_components(
            base=refiner.base,
            adapter=adapter,
            spec=CoarseSpec(power_scale=1.25),
            manifest_fingerprint="fold",
            refiner=refiner,
        )
        actual = provider(batch)
        self.assertTrue(torch.equal(actual.latent, expected.latent))
        self.assertTrue(
            torch.equal(actual.low_rank_residual, expected.latent - base_latent)
        )

    def test_validation_forces_fold_even_when_official_test_uses_all(self) -> None:
        provider = FrozenCoarseProvider.from_components(
            base=_Base(),
            adapter=_Adapter(),
            spec=CoarseSpec(anchor_source="all_official_train"),
            manifest_fingerprint="fold",
        )
        self.assertEqual(provider.validate_anchor_source("validation"), "fold")
        self.assertEqual(
            provider.validate_anchor_source("official_test"),
            "all_official_train",
        )

    def test_disabled_optional_correction_does_not_call_operation(self) -> None:
        coarse = torch.tensor([[1 + 2j]], dtype=torch.complex64)

        def forbidden(_):
            raise AssertionError("operation must not run")

        actual = apply_optional_correction(coarse, 0.0, forbidden)
        self.assertTrue(torch.equal(actual, coarse))


if __name__ == "__main__":
    unittest.main()
