from __future__ import annotations

import dataclasses
import unittest

import numpy as np
import torch

from solution.radio_map.config import RoundConfig
from solution.radio_map.learning.e2e_cgpf import (
    ComplexMIMOOFDMRenderer,
    E2ECGPF,
    E2ECGPFConfig,
    GaussianFieldConfig,
    PathModes,
    PathNetworkConfig,
    RendererConfig,
    TrainableGaussianField,
)
from solution.radio_map.learning.e2e_cgpf_losses import (
    E2ECGPFLossConfig,
    e2e_cgpf_loss,
)


def round_config(
    *,
    m_h: int = 2,
    m_v: int = 1,
    m_p: int = 1,
    n_h: int = 1,
    n_v: int = 1,
    n_p: int = 1,
    s: int = 8,
) -> RoundConfig:
    return RoundConfig(
        p_train_declared=2,
        p_test=1,
        m=m_h * m_v * m_p,
        m_h=m_h,
        m_v=m_v,
        m_p=m_p,
        n=n_h * n_v * n_p,
        n_h=n_h,
        n_v=n_v,
        n_p=n_p,
        s=s,
        q=2,
        bs_position=(0.0, 0.0, 0.0),
        weights=(0.4, 0.4, 0.2),
    )


def paths(
    config: RoundConfig,
    gains: list[complex],
    *,
    delays: list[float] | None = None,
    aod: tuple[float, float] = (0.0, 0.0),
    aoa: tuple[float, float] = (0.0, 0.0),
    polarization: torch.Tensor | None = None,
) -> PathModes:
    count = len(gains)
    delays = delays or [0.0] * count
    if polarization is None:
        polarization = torch.ones(1, count, config.m_p, config.n_p, dtype=torch.complex64)
    return PathModes(
        delay_bins=torch.tensor([delays], dtype=torch.float32),
        aod=torch.tensor([[aod] * count], dtype=torch.float32),
        aoa=torch.tensor([[aoa] * count], dtype=torch.float32),
        complex_gain=torch.tensor([gains], dtype=torch.complex64),
        polarization=polarization,
        existence=torch.ones(1, count),
        reliability=torch.ones(1, count),
        width=torch.zeros(1, count),
        path_type=torch.zeros(1, count, 2),
        gate=torch.ones(1, count),
        gaussian_indices=torch.arange(count).reshape(1, count),
        selection_weight=torch.ones(1, count),
    )


class ComplexRendererTests(unittest.TestCase):
    def test_in_phase_enhancement_and_opposite_phase_cancellation(self) -> None:
        config = round_config(m_h=1)
        renderer = ComplexMIMOOFDMRenderer(config, RendererConfig(path_chunk_size=1))
        single = renderer(paths(config, [1.0 + 0.0j]))
        enhanced = renderer(paths(config, [1.0 + 0.0j, 1.0 + 0.0j]))
        cancelled = renderer(paths(config, [1.0 + 0.0j, -1.0 + 0.0j]))
        torch.testing.assert_close(enhanced, 2.0 * single)
        torch.testing.assert_close(cancelled, torch.zeros_like(cancelled), atol=1e-6, rtol=0)

    def test_delay_phase_ramp_is_explicit_and_continuous(self) -> None:
        config = round_config(m_h=1, s=8)
        renderer = ComplexMIMOOFDMRenderer(config)
        delay = 1.5
        rendered = renderer(paths(config, [1.0 + 0.0j], delays=[delay]))
        carrier = torch.arange(config.s, dtype=torch.float32)
        expected = torch.polar(
            torch.ones(config.s),
            -2.0 * torch.pi * carrier * delay / config.s,
        )
        torch.testing.assert_close(rendered[0, 0, 0], expected)

    def test_array_steering_uses_predicted_direction(self) -> None:
        config = round_config(m_h=2, m_v=1, m_p=1)
        renderer = ComplexMIMOOFDMRenderer(config)
        rendered = renderer(paths(config, [1.0 + 0.0j], aod=(0.5, 0.0)))
        ratio = rendered[0, 1, 0, 0] / rendered[0, 0, 0, 0]
        expected = torch.polar(torch.tensor(1.0), torch.tensor(torch.pi / 2))
        torch.testing.assert_close(ratio, expected)

    def test_polarization_coupling_maps_both_polarization_axes(self) -> None:
        config = round_config(m_h=1, m_p=2, n_p=2, s=1)
        renderer = ComplexMIMOOFDMRenderer(config)
        coupling = torch.tensor(
            [[[[1.0 + 0.0j, 2.0j], [-1.0j, 0.5 + 0.0j]]]],
            dtype=torch.complex64,
        )
        rendered = renderer(paths(config, [1.0 + 0.0j], polarization=coupling))
        torch.testing.assert_close(rendered[0, :, :, 0], coupling[0, 0])


class GaussianFieldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GaussianFieldConfig(
            initial_count=4,
            max_count=6,
            min_count=2,
            material_dim=4,
            densify_count=1,
            prune_count=1,
            densify_threshold=0.5,
            prune_threshold=0.1,
        )
        centers = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        )
        normals = torch.tensor([[0.0, 0.0, 1.0]] * 4)
        self.field = TrainableGaussianField(centers, normals, self.config, seed=7)

    def test_bounded_parameters_and_map_controls_have_equal_capacity(self) -> None:
        real = self.field.state("real")
        zero = self.field.state("zero")
        shuffle = self.field.state("shuffle")
        zero_field = TrainableGaussianField(
            self.field.center_prior[: self.config.initial_count],
            self.field.normal_prior[: self.config.initial_count],
            self.config,
            map_mode="zero",
            seed=7,
        )
        shuffle_field = TrainableGaussianField(
            self.field.center_prior[: self.config.initial_count],
            self.field.normal_prior[: self.config.initial_count],
            self.config,
            map_mode="shuffle",
            seed=7,
        )
        self.assertTrue(torch.all(real.scales >= self.config.min_scale))
        self.assertTrue(torch.all(real.scales <= self.config.max_scale))
        self.assertEqual(real.centers.shape, zero.centers.shape)
        self.assertEqual(real.centers.shape, shuffle.centers.shape)
        capacities = {
            sum(parameter.numel() for parameter in field.parameters())
            for field in (self.field, zero_field, shuffle_field)
        }
        self.assertEqual(len(capacities), 1)
        self.assertFalse(torch.equal(real.centers, zero.centers))
        self.assertFalse(torch.equal(real.centers, shuffle.centers))

    def test_densify_prune_respects_cap_and_snapshot_rolls_back(self) -> None:
        snapshot = self.field.structure_snapshot()
        self.field.ema_gradient[0] = 1.0
        self.field.ema_activation[1] = 0.0
        self.field.age[1] = 3
        edit = self.field.densify_prune(step=11)
        self.assertLessEqual(edit.active_after, self.config.max_count)
        self.assertGreaterEqual(edit.active_after, self.config.min_count)
        self.assertEqual(len(edit.densified), 1)
        self.assertEqual(len(edit.pruned), 1)
        self.field.restore_structure(snapshot)
        for name, expected in snapshot.items():
            torch.testing.assert_close(getattr(self.field, name).cpu(), expected)


class JointModelTests(unittest.TestCase):
    def _model(self, production: bool = False) -> E2ECGPF:
        config = (
            round_config(
                m_h=16, m_v=8, m_p=2, n_h=1, n_v=2, n_p=2, s=192
            )
            if production
            else round_config(m_h=2, m_v=2, m_p=2, n_h=1, n_v=2, n_p=2, s=8)
        )
        field_config = GaussianFieldConfig(
            initial_count=4,
            max_count=8,
            min_count=2,
            material_dim=4,
            densify_count=1,
            prune_count=1,
        )
        model_config = E2ECGPFConfig(
            field=field_config,
            paths=PathNetworkConfig(
                hidden_dim=32,
                query_dim=16,
                modes_per_gaussian=2,
                selected_gaussians=2,
                polarization_rank=1,
                path_type_dim=3,
                fourier_bands=2,
                max_delay_bins=float(config.s - 1),
            ),
            renderer=RendererConfig(path_chunk_size=2),
            seed=3,
        )
        centers = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        )
        normals = torch.tensor([[0.0, 0.0, 1.0]] * 4)
        field = TrainableGaussianField(centers, normals, field_config, seed=3)
        return E2ECGPF(
            field,
            config,
            model_config,
            torch.tensor([0.5, 0.5, 0.0]),
            torch.tensor(2.0),
        )

    def test_production_shape_dtype_device_finite_and_joint_gradients(self) -> None:
        model = self._model(production=True)
        target_position = torch.tensor([[2.0, 1.0, 0.5]])
        prediction, paths_value = model(target_position)
        self.assertEqual(tuple(prediction.shape), (1, 256, 4, 192))
        self.assertEqual(prediction.dtype, torch.complex64)
        self.assertEqual(prediction.device, target_position.device)
        self.assertTrue(torch.isfinite(prediction.real).all())
        self.assertTrue(torch.isfinite(prediction.imag).all())
        prediction.abs().square().mean().backward()
        for parameter in (
            model.field.center_delta,
            model.field.log_scale,
            model.field.orientation,
            model.field.opacity_logit,
            model.field.material_code,
            model.path_network.head.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)
        self.assertGreaterEqual(paths_value.path_count, 4)

    def test_joint_loss_fits_complex_pas_pdp_nmse_and_has_finite_gradient(self) -> None:
        model = self._model()
        positions = torch.tensor([[2.0, 1.0, 0.5], [1.5, -0.5, 0.2]])
        prediction, paths_value = model(positions)
        path_tensors = {
            "delay": paths_value.delay_bins,
            "aod": paths_value.aod,
            "aoa": paths_value.aoa,
            "gain": paths_value.complex_gain,
            "polarization": paths_value.polarization,
            "existence": paths_value.existence,
            "reliability": paths_value.reliability,
            "width": paths_value.width,
            "path_type": paths_value.path_type,
            "gate": paths_value.gate,
        }
        for tensor in path_tensors.values():
            tensor.retain_grad()
        target = (prediction.detach() * (0.8 + 0.2j)).to(torch.complex64)
        zero, _ = model(positions, map_view="zero")
        shuffle, _ = model(positions, map_view="shuffle")
        values, metrics = e2e_cgpf_loss(
            prediction,
            target,
            paths_value,
            model.field,
            model.round_config,
            model.config.antenna_order,
            E2ECGPFLossConfig(causal_weight=0.1),
            zero_prediction=zero,
            shuffle_prediction=shuffle,
        )
        self.assertTrue(torch.isfinite(values.total))
        self.assertTrue(torch.isfinite(metrics.score))
        values.total.backward()
        self.assertTrue(torch.isfinite(model.path_network.head.weight.grad).all())
        for name, tensor in path_tensors.items():
            with self.subTest(path_parameter=name):
                self.assertIsNotNone(tensor.grad)
                self.assertTrue(torch.isfinite(tensor.grad).all())
                self.assertGreater(float(tensor.grad.abs().sum()), 0.0)

    def test_model_manifest_closes_prohibited_forward_dependencies(self) -> None:
        manifest = self._model().model_manifest()
        self.assertFalse(manifest["o41_forward_dependency"])
        self.assertFalse(manifest["task020_dependency"])


if __name__ == "__main__":
    unittest.main()
