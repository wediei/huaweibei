from __future__ import annotations

import unittest

import torch


class ModelComponentTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(37)
        self.group_ids = torch.arange(192, dtype=torch.long)
        self.latents = torch.complex(
            torch.randn(2, 32, 192), torch.randn(2, 32, 192)
        )

    def test_component_shapes_and_float_fourier_features(self) -> None:
        from solution.radio_map.learning.model_components import (
            CorridorEncoder,
            FourierFeatures,
            PatchEncoder,
        )

        fourier = FourierFeatures(input_dim=3, num_frequencies=4)
        features = fourier(torch.randn(2, 3), torch.randn(2, 3))
        self.assertEqual(features.shape, (2, fourier.output_dim))
        self.assertTrue(features.dtype.is_floating_point)
        self.assertTrue(torch.isfinite(features).all())

        patch_encoder = PatchEncoder(in_channels=13, output_dim=64)
        corridor_encoder = CorridorEncoder(input_dim=15, output_dim=64)
        self.assertEqual(patch_encoder(torch.randn(2, 13, 33, 33)).shape, (2, 64))
        self.assertEqual(corridor_encoder(torch.randn(2, 32, 15)).shape, (2, 64))

    def test_group_summary_is_phase_sensitive_and_finite(self) -> None:
        from solution.radio_map.learning.model_components import GroupSummaryEncoder

        encoder = GroupSummaryEncoder(group_ids=self.group_ids, group_count=192)
        summary = encoder(self.latents)
        self.assertEqual(summary.shape, (2, 32, 192, 6))
        self.assertTrue(torch.isfinite(summary).all())

        phase_ids = torch.tensor([0, 0], dtype=torch.long)
        phase_encoder = GroupSummaryEncoder(group_ids=phase_ids, group_count=1)
        coherent = torch.tensor([[[1 + 0j, 1 + 0j]]], dtype=torch.complex64)
        cancelling = torch.tensor([[[1 + 0j, -1 + 0j]]], dtype=torch.complex64)
        self.assertGreater(phase_encoder(coherent)[0, 0, 0, 5].item(), 0.99)
        self.assertLess(phase_encoder(cancelling)[0, 0, 0, 5].item(), 1e-6)

    def test_corridor_mask_ignores_padding_and_all_empty_is_finite(self) -> None:
        from solution.radio_map.learning.model_components import CorridorEncoder

        encoder = CorridorEncoder(input_dim=15, output_dim=64).eval()
        valid = torch.randn(1, 3, 15)
        first = torch.cat((valid, torch.randn(1, 3, 15)), dim=1)
        second = torch.cat((valid, torch.randn(1, 3, 15) * 100), dim=1)
        mask = torch.tensor([[True, True, True, False, False, False]])
        torch.testing.assert_close(encoder(first, mask), encoder(second, mask))

        empty = encoder(torch.randn(2, 6, 15), torch.zeros(2, 6, dtype=torch.bool))
        self.assertTrue(torch.isfinite(empty).all())
        torch.testing.assert_close(empty, torch.zeros_like(empty))

    def test_group_summary_empty_groups_are_zero_and_exposes_mask(self) -> None:
        from solution.radio_map.learning.model_components import GroupSummaryEncoder

        encoder = GroupSummaryEncoder(torch.tensor([0, 2]), group_count=4)
        summary = encoder(torch.tensor([[[3 + 4j, 1 - 1j]]], dtype=torch.complex64))
        self.assertTrue(torch.equal(encoder.group_mask, torch.tensor([True, False, True, False])))
        torch.testing.assert_close(summary[:, :, ~encoder.group_mask], torch.zeros(1, 1, 2, 6))

    def test_components_have_finite_gradients(self) -> None:
        from solution.radio_map.learning.model_components import (
            CorridorEncoder,
            FourierFeatures,
            GroupSummaryEncoder,
            PatchEncoder,
        )

        patch = PatchEncoder(13, 16)
        corridor = CorridorEncoder(15, 16)
        fourier = FourierFeatures(3, num_frequencies=3)
        real = torch.randn(2, 4, 192, requires_grad=True)
        imaginary = torch.randn(2, 4, 192, requires_grad=True)
        latent = torch.complex(real, imaginary)
        summary = GroupSummaryEncoder(self.group_ids, 192)(latent)
        loss = (
            patch(torch.randn(2, 13, 33, 33)).square().mean()
            + corridor(torch.randn(2, 8, 15), torch.ones(2, 8, dtype=torch.bool)).square().mean()
            + fourier(torch.randn(2, 3), torch.randn(2, 3)).square().mean()
            + summary.square().mean()
        )
        loss.backward()
        self.assertTrue(torch.isfinite(real.grad).all())
        self.assertTrue(torch.isfinite(imaginary.grad).all())
        for module in (patch, corridor):
            for parameter in module.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_component_input_validation(self) -> None:
        from solution.radio_map.learning.model_components import (
            CorridorEncoder,
            FourierFeatures,
            GroupSummaryEncoder,
            PatchEncoder,
        )

        with self.assertRaises(ValueError):
            GroupSummaryEncoder(torch.tensor([0, 2]), group_count=2)
        with self.assertRaises(ValueError):
            GroupSummaryEncoder(torch.tensor([0, 1]), group_count=2)(
                torch.ones(1, 1, 3, dtype=torch.complex64)
            )
        with self.assertRaises(TypeError):
            GroupSummaryEncoder(torch.tensor([0]), group_count=1)(torch.ones(1, 1, 1))
        with self.assertRaises(ValueError):
            PatchEncoder(13, 8)(torch.randn(1, 12, 33, 33))
        with self.assertRaises(ValueError):
            CorridorEncoder(15, 8)(torch.randn(1, 3, 14))
        with self.assertRaises(TypeError):
            FourierFeatures(3)(torch.ones(1, 3, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()


class SupportAwareAnchorMixerTests(unittest.TestCase):
    """Contract tests for the Task 4 support-aware complex latent mixer."""

    def setUp(self) -> None:
        from solution.radio_map.learning.anchor_mixer import (
            SupportAwareAnchorMixer,
            SupportAwareAnchorMixerConfig,
        )

        torch.manual_seed(71)
        self.group_ids = torch.tensor([0, 0, 2, 2, 3, 3], dtype=torch.long)
        self.config = SupportAwareAnchorMixerConfig(
            latent_size=6,
            group_count=5,
            d_model=24,
            group_dim=12,
            geometry_dim=8,
            low_rank=3,
            num_frequencies=2,
        )
        self.model = SupportAwareAnchorMixer(self.config, self.group_ids)
        b, k, l = 2, 4, 6
        self.anchor_real = torch.randn(b, k, l, requires_grad=True)
        self.anchor_imag = torch.randn(b, k, l, requires_grad=True)
        self.batch = {
            "anchor_latents": torch.complex(self.anchor_real, self.anchor_imag),
            "anchor_distances": torch.tensor([[3.0, 1.0, 2.0, 4.0], [4.0, 2.0, 1.0, 3.0]]),
            "anchor_mask": torch.tensor([[True, True, True, False], [True, True, True, True]]),
            "anchor_indices": torch.tensor([[100, 30, 50, -1], [40, 20, 10, 60]]),
            "pair_features": torch.randn(b, k, 14),
            "target_positions": torch.randn(b, 3),
            "target_positions_standardized": torch.randn(b, 3),
            "target_bs_relative": torch.randn(b, 3),
            "target_bs_relative_standardized": torch.randn(b, 3),
            "target_point_features": torch.randn(b, 13),
            "target_patch": torch.randn(b, 13, 9, 9),
            "bs_patch": torch.randn(b, 13, 9, 9),
            "bs_target_corridor": torch.randn(b, 5, 15),
            "bs_target_corridor_mask": torch.tensor([[True, True, True, False, False], [True, True, True, True, True]]),
            "anchor_corridors": torch.randn(b, k, 3, 15),
            "anchor_corridor_mask": torch.tensor(
                [[[True, True, False]] * k, [[True, True, True]] * k]
            ),
        }

    @staticmethod
    def _permute_anchor_axis(batch: dict[str, torch.Tensor], permutation: torch.Tensor) -> dict[str, torch.Tensor]:
        result = dict(batch)
        for name in (
            "anchor_latents", "anchor_distances", "anchor_mask", "anchor_indices", "pair_features",
            "anchor_corridors", "anchor_corridor_mask",
        ):
            result[name] = batch[name].index_select(1, permutation)
        return result

    def test_zero_initialization_equals_masked_nearest_not_slot_zero(self) -> None:
        output = self.model(self.batch)
        expected_indices = torch.tensor([1, 2])
        expected = self.batch["anchor_latents"][torch.arange(2), expected_indices]
        torch.testing.assert_close(output.nearest_latent, expected, rtol=0, atol=1e-6)
        torch.testing.assert_close(output.latent, expected, rtol=0, atol=1e-6)

    def test_anchor_permutation_does_not_change_output(self) -> None:
        original = self.model(self.batch)
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = self._permute_anchor_axis(self.batch, permutation)
        actual = self.model(permuted)
        torch.testing.assert_close(actual.latent, original.latent, rtol=1e-5, atol=1e-6)
        inverse = torch.argsort(permutation)
        torch.testing.assert_close(actual.anchor_weights[:, inverse], original.anchor_weights)

    def test_equal_distance_nearest_uses_source_id_and_is_permutation_invariant(self) -> None:
        tied = dict(self.batch)
        tied["anchor_distances"] = self.batch["anchor_distances"].clone()
        tied["anchor_distances"][0, :2] = 1.0
        original = self.model(tied)
        expected = tied["anchor_latents"][0, 1]  # IDs 100 and 30 tie; 30 wins.
        torch.testing.assert_close(original.nearest_latent[0], expected, rtol=0, atol=1e-6)
        permutation = torch.tensor([1, 3, 0, 2])
        actual = self.model(self._permute_anchor_axis(tied, permutation))
        torch.testing.assert_close(actual.nearest_latent, original.nearest_latent, rtol=0, atol=1e-6)
        torch.testing.assert_close(actual.latent, original.latent, rtol=1e-5, atol=1e-6)

    def test_mask_shapes_weights_and_empty_group(self) -> None:
        output = self.model(self.batch)
        self.assertEqual(output.latent.shape, (2, 6))
        self.assertEqual(output.nearest_latent.shape, (2, 6))
        self.assertEqual(output.anchor_weights.shape, (2, 4, 5))
        self.assertEqual(output.alpha.shape, (2, 5))
        self.assertEqual(output.complex_gain.shape, (2, 5))
        self.assertEqual(output.low_rank_residual.shape, (2, 6))
        self.assertTrue(torch.equal(output.anchor_weights[0, 3], torch.zeros(5)))
        self.assertTrue(torch.equal(output.anchor_weights[:, :, 1], torch.zeros(2, 4)))
        group_sum = output.anchor_weights.sum(dim=1)
        torch.testing.assert_close(group_sum[:, [0, 2, 3]], torch.ones(2, 3))
        self.assertTrue(torch.equal(group_sum[:, [1, 4]], torch.zeros(2, 2)))

    def test_geometry_off_ignores_geometry_and_geometry_on_runs(self) -> None:
        off_config = type(self.config)(
            latent_size=6, group_count=5, d_model=24, group_dim=12, geometry_dim=8,
            low_rank=3, num_frequencies=2, use_geometry=False,
        )
        off_model = type(self.model)(off_config, self.group_ids)
        geometry_free = {key: value for key, value in self.batch.items() if "patch" not in key and "corridor" not in key and key != "target_point_features"}
        output = off_model(geometry_free)
        self.assertEqual(output.latent.shape, (2, 6))
        on_config = type(self.config)(**{**self.config.__dict__, "use_geometry": True})
        on_model = type(self.model)(on_config, self.group_ids)
        self.assertEqual(on_model(self.batch).latent.shape, (2, 6))

    def test_geometry_encoders_are_shared_and_bs_center_is_consumed(self) -> None:
        on_config = type(self.config)(**{**self.config.__dict__, "use_geometry": True})
        model = type(self.model)(on_config, self.group_ids)
        self.assertTrue(hasattr(model, "patch_encoder"))
        self.assertTrue(hasattr(model, "corridor_encoder"))
        self.assertTrue(hasattr(model, "point_encoder"))
        self.assertFalse(hasattr(model, "target_patch_encoder"))
        self.assertFalse(hasattr(model, "bs_patch_encoder"))
        self.assertFalse(hasattr(model, "bs_corridor_encoder"))
        self.assertFalse(hasattr(model, "anchor_corridor_encoder"))
        point_inputs: list[torch.Tensor] = []
        handle = model.point_encoder.register_forward_pre_hook(
            lambda _module, inputs: point_inputs.append(inputs[0].detach().clone())
        )
        try:
            model._geometry_features(self.batch, 2, 4, torch.device("cpu"), torch.float32)
        finally:
            handle.remove()
        self.assertEqual(len(point_inputs), 2)
        center = self.batch["bs_patch"][:, :, 4, 4]
        torch.testing.assert_close(point_inputs[1], center)
        changed = dict(self.batch)
        changed["bs_patch"] = self.batch["bs_patch"].clone()
        changed["bs_patch"][:, :, 4, 4] += 3.0
        changed_inputs: list[torch.Tensor] = []
        handle = model.point_encoder.register_forward_pre_hook(
            lambda _module, inputs: changed_inputs.append(inputs[0].detach().clone())
        )
        try:
            model._geometry_features(changed, 2, 4, torch.device("cpu"), torch.float32)
        finally:
            handle.remove()
        self.assertFalse(torch.equal(changed_inputs[1], point_inputs[1]))

    def test_mixed_real_input_dtypes_are_cast_without_autocast(self) -> None:
        on_config = type(self.config)(**{**self.config.__dict__, "use_geometry": True})
        model = type(self.model)(on_config, self.group_ids)
        mixed = dict(self.batch)
        for name in ("anchor_distances", "pair_features", "target_positions", "target_positions_standardized", "target_bs_relative", "target_bs_relative_standardized"):
            mixed[name] = mixed[name].double()
        for name in ("target_point_features", "target_patch", "bs_patch", "bs_target_corridor", "anchor_corridors"):
            mixed[name] = mixed[name].half()
        output = model(mixed)
        self.assertEqual(output.latent.dtype, torch.complex64)
        self.assertTrue(torch.isfinite(output.latent).all())
        loss = output.latent.abs().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(self.anchor_real.grad).all())
        self.assertTrue(torch.isfinite(self.anchor_imag.grad).all())

    def test_gradients_are_finite_and_no_dense_two_latent_head(self) -> None:
        output = self.model(self.batch)
        loss = output.latent.abs().square().mean() + output.anchor_weights.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(self.anchor_real.grad).all())
        self.assertTrue(torch.isfinite(self.anchor_imag.grad).all())
        for parameter in self.model.parameters():
            if parameter.requires_grad and parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        output_heads = ("alpha_head", "gain_real_head", "gain_imag_head", "residual_real_head", "residual_imag_head")
        self.assertTrue(all(
            getattr(self.model, name).out_features != 2 * self.config.latent_size
            for name in output_heads
        ))

    def test_bfloat16_autocast_forward_backward_is_finite(self) -> None:
        """AMP must promote real components before constructing complex tensors."""
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = self.model(self.batch)
            loss = (
                output.latent.abs().square().mean()
                + output.complex_gain.abs().square().mean()
                + output.low_rank_residual.abs().square().mean()
            )
        self.assertTrue(torch.isfinite(output.latent).all())
        self.assertTrue(torch.isfinite(output.complex_gain).all())
        self.assertTrue(torch.isfinite(output.low_rank_residual).all())
        loss.backward()
        self.assertTrue(torch.isfinite(self.anchor_real.grad).all())
        self.assertTrue(torch.isfinite(self.anchor_imag.grad).all())
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_bad_inputs_are_rejected(self) -> None:
        invalid = dict(self.batch)
        invalid["anchor_mask"] = torch.zeros_like(self.batch["anchor_mask"])
        with self.assertRaises(ValueError):
            self.model(invalid)
        invalid = dict(self.batch)
        invalid["anchor_distances"] = self.batch["anchor_distances"].clone()
        invalid["anchor_distances"][0, 0] = -1
        with self.assertRaises(ValueError):
            self.model(invalid)
        invalid = dict(self.batch)
        invalid["pair_features"] = torch.randn(2, 4, 13)
        with self.assertRaises(ValueError):
            self.model(invalid)
        invalid = dict(self.batch)
        invalid["anchor_indices"] = self.batch["anchor_indices"].clone()
        invalid["anchor_indices"][0, 1] = 100
        with self.assertRaises(ValueError):
            self.model(invalid)
