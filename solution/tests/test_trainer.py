from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from solution.radio_map.data import RoundDataset
from solution.radio_map.learning.anchor_mixer import (
    MixerOutput,
    SupportAwareAnchorMixer,
    SupportAwareAnchorMixerConfig,
)
from solution.radio_map.learning.latent_adapter import FixedSupportLatentAdapter
from solution.radio_map.transforms import AntennaLayout
from solution.tests.fixtures import create_round_dir


def _adapter() -> FixedSupportLatentAdapter:
    root = Path(tempfile.mkdtemp())
    dataset = RoundDataset.open(create_round_dir(root))
    adapter = FixedSupportLatentAdapter(AntennaLayout(dataset.config, ("P", "H", "V")), 1.0, 2)
    adapter.fit(dataset, np.arange(6), batch_size=3)
    return adapter


def _batch(adapter: FixedSupportLatentAdapter, count: int = 3) -> dict[str, torch.Tensor]:
    torch.manual_seed(9)
    latent_size = adapter.coefficient_count
    target = torch.complex(torch.randn(count, latent_size), torch.randn(count, latent_size))
    nearest = torch.zeros_like(target)
    return {
        "anchor_latents": nearest[:, None], "anchor_distances": torch.ones(count, 1),
        "anchor_mask": torch.ones(count, 1, dtype=torch.bool), "anchor_indices": torch.arange(count)[:, None],
        "pair_features": torch.zeros(count, 1, 14), "target_positions": torch.randn(count, 3),
        "target_positions_standardized": torch.randn(count, 3), "target_bs_relative": torch.randn(count, 3),
        "target_bs_relative_standardized": torch.randn(count, 3), "target_latent": target,
        "target_channel": adapter.decode_torch(target),
    }


class TrainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = _adapter()
        self.batch = _batch(self.adapter)

    def test_loss_identity_and_hand_formula_have_finite_gradients(self) -> None:
        from solution.radio_map.learning.losses import AnchorLossConfig, anchor_completion_loss, complex_mse

        target = self.batch["target_latent"]
        output = MixerOutput(target, target, torch.empty(0), torch.empty(0), torch.empty(0, dtype=torch.complex64), torch.zeros_like(target))
        values = anchor_completion_loss(output, target, self.batch["target_channel"], self.adapter, AnchorLossConfig())
        self.assertAlmostEqual(values.latent_mse.item(), 0.0, places=7)
        self.assertAlmostEqual(values.residual_energy.item(), 0.0, places=7)
        self.assertAlmostEqual(values.total.item(), 0.0, places=6)
        predicted = (target + (0.1 + 0.2j)).detach().requires_grad_()
        output = MixerOutput(predicted, target, torch.empty(0), torch.empty(0), torch.empty(0, dtype=torch.complex64), torch.zeros_like(target))
        values = anchor_completion_loss(output, target, self.batch["target_channel"], self.adapter, AnchorLossConfig())
        config = AnchorLossConfig()
        expected = (0.4 * (1 - values.pas) + 0.4 * (1 - values.pdp) + 0.2 * values.nmse / (1 + values.nmse) + config.latent_weight * complex_mse(predicted, target) + config.nearest_weight * complex_mse(predicted, target))
        torch.testing.assert_close(values.total, expected)
        log_values = anchor_completion_loss(
            output, target, self.batch["target_channel"], self.adapter,
            AnchorLossConfig(nmse_objective="log"),
        )
        self.assertGreater(log_values.nmse_loss, values.nmse_loss)
        values.total.backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())

    def test_bfloat16_boundary_and_cpu_trainer_does_not_autocast(self) -> None:
        from solution.radio_map.learning.losses import AnchorLossConfig, anchor_completion_loss
        from solution.radio_map.learning.trainer import Trainer, TrainerConfig

        config = SupportAwareAnchorMixerConfig(self.adapter.coefficient_count, d_model=12, group_dim=8, geometry_dim=8, low_rank=4, num_frequencies=2)
        model = SupportAwareAnchorMixer(config, torch.as_tensor(self.adapter.group_ids))
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = model(self.batch)
            values = anchor_completion_loss(output, self.batch["target_latent"], self.batch["target_channel"], self.adapter, AnchorLossConfig())
        self.assertTrue(torch.isfinite(values.total))
        trainer = Trainer(model, self.adapter, TrainerConfig(epochs=1), model_config=config.__dict__)
        with trainer._autocast():
            try:
                cpu_autocast_enabled = torch.is_autocast_enabled("cpu")
            except TypeError:
                # PyTorch <= 2.3 exposes the device-specific CPU query
                # separately; newer releases accept the device argument.
                cpu_autocast_enabled = torch.is_autocast_cpu_enabled()
            self.assertFalse(cpu_autocast_enabled)

    def test_streaming_validation_is_exact_across_batch_sizes(self) -> None:
        from solution.radio_map.learning.trainer import Trainer, TrainerConfig
        from torch.utils.data import DataLoader

        config = SupportAwareAnchorMixerConfig(self.adapter.coefficient_count, d_model=12, group_dim=8, geometry_dim=8, low_rank=4, num_frequencies=2)
        model = SupportAwareAnchorMixer(config, torch.as_tensor(self.adapter.group_ids))
        samples = [{key: value[index] for key, value in _batch(self.adapter, 5).items()} for index in range(5)]
        first = Trainer(model, self.adapter, TrainerConfig(epochs=1), model_config=config.__dict__).validate(DataLoader(samples, batch_size=1))
        second = Trainer(model, self.adapter, TrainerConfig(epochs=1), model_config=config.__dict__).validate(DataLoader(samples, batch_size=3))
        for key in ("pas", "pdp", "nmse", "score", "nearest_score", "gain"):
            self.assertAlmostEqual(first[key], second[key], places=10)

    def test_zero_correction_scale_is_exact_nearest_baseline(self) -> None:
        from solution.radio_map.learning.trainer import Trainer, TrainerConfig
        from torch.utils.data import DataLoader

        config = SupportAwareAnchorMixerConfig(self.adapter.coefficient_count, d_model=12, group_dim=8, geometry_dim=8, low_rank=4, num_frequencies=2)
        trainer = Trainer(SupportAwareAnchorMixer(config, torch.as_tensor(self.adapter.group_ids)), self.adapter, TrainerConfig(epochs=1), model_config=config.__dict__)
        data = _batch(self.adapter, 3)
        samples = [{key: value[index] for key, value in data.items()} for index in range(3)]
        result = trainer.validate(DataLoader(samples, batch_size=2), correction_scale=0.0)
        self.assertAlmostEqual(result["score"], result["nearest_score"], places=10)
        with self.assertRaises(ValueError):
            trainer.validate([], correction_scale=-1.0)

    def test_streaming_validation_uses_layout_score_weights(self) -> None:
        from solution.radio_map.learning.trainer import Trainer, TrainerConfig
        from solution.radio_map.metrics import competition_metrics
        from solution.radio_map.transforms import AntennaLayout
        from torch.utils.data import DataLoader

        weighted_adapter = _adapter()
        weighted_adapter.layout = AntennaLayout(
            replace(weighted_adapter.layout.config, weights=(0.1, 0.2, 0.7)),
            weighted_adapter.layout.order,
        )
        config = SupportAwareAnchorMixerConfig(weighted_adapter.coefficient_count, d_model=12, group_dim=8, geometry_dim=8, low_rank=4, num_frequencies=2)
        model = SupportAwareAnchorMixer(config, torch.as_tensor(weighted_adapter.group_ids))
        data = _batch(weighted_adapter, 2)
        samples = [{key: value[index] for key, value in data.items()} for index in range(2)]
        result = Trainer(model, weighted_adapter, TrainerConfig(epochs=1), model_config=config.__dict__).validate(DataLoader(samples, batch_size=2))
        baseline = weighted_adapter.decode_torch(torch.zeros_like(data["target_latent"])).numpy()
        expected = competition_metrics(baseline, data["target_channel"].numpy(), weighted_adapter.layout, weights=weighted_adapter.layout.config.weights)
        self.assertAlmostEqual(result["score"], expected.score, places=10)

    def test_optimizer_step_averages_full_and_tail_accumulation_windows(self) -> None:
        from solution.radio_map.learning.trainer import Trainer, TrainerConfig

        def make() -> Trainer:
            model = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                model.weight.zero_()
            trainer = Trainer(model, None, TrainerConfig(epochs=1, learning_rate=1.0, weight_decay=0.0, accumulation_steps=4, gradient_clip_norm=100.0))
            trainer.optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
            return trainer

        single, full, tail = make(), make(), make()
        single.model.weight.grad = torch.ones_like(single.model.weight)
        full.model.weight.grad = torch.full_like(full.model.weight, 4.0)
        tail.model.weight.grad = torch.ones_like(tail.model.weight)
        single._optimizer_step(1)
        full._optimizer_step(4)
        tail._optimizer_step(1)
        torch.testing.assert_close(single.model.weight, full.model.weight)
        torch.testing.assert_close(single.model.weight, tail.model.weight)

    def test_fit_one_batch_uses_a_one_batch_tail_window(self) -> None:
        from solution.radio_map.learning.trainer import Trainer, TrainerConfig

        class ScalarModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.value = torch.nn.Parameter(torch.zeros(()))

        trainer = Trainer(ScalarModel(), None, TrainerConfig(epochs=1, learning_rate=1.0, weight_decay=0.0, accumulation_steps=4, gradient_clip_norm=100.0))
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1.0)
        trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _epoch: 1.0)
        trainer._loss_for_batch = lambda _batch: (None, SimpleNamespace(total=trainer.model.value))  # type: ignore[method-assign]
        trainer.validate = lambda _loader: {"score": 0.0}  # type: ignore[method-assign]
        trainer.fit([{}], [{}])
        self.assertAlmostEqual(trainer.model.value.item(), -1.0, places=7)

    def test_five_epoch_warmup_then_cosine_boundaries(self) -> None:
        from solution.radio_map.learning.trainer import Trainer, TrainerConfig

        trainer = Trainer(torch.nn.Linear(1, 1), None, TrainerConfig(epochs=15, warmup_epochs=5))
        self.assertAlmostEqual(trainer._lr_multiplier(0), 0.2)
        self.assertAlmostEqual(trainer._lr_multiplier(4), 1.0)
        self.assertAlmostEqual(trainer._lr_multiplier(5), 1.0)
        self.assertAlmostEqual(trainer._lr_multiplier(10), 0.5)
        self.assertAlmostEqual(trainer._lr_multiplier(15), 0.0)

    def test_trainer_config_rejects_bool_non_integer_and_nonfinite_values(self) -> None:
        from solution.radio_map.learning.trainer import TrainerConfig

        for name in ("epochs", "accumulation_steps", "warmup_epochs", "patience"):
            with self.subTest(name=name, value=True), self.assertRaises(ValueError):
                TrainerConfig(**{name: True})
            with self.subTest(name=name, value=1.5), self.assertRaises(ValueError):
                TrainerConfig(**{name: 1.5})
        for name, value in (("learning_rate", True), ("learning_rate", float("nan")), ("weight_decay", float("inf")), ("gradient_clip_norm", float("nan"))):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                TrainerConfig(**{name: value})
        with self.assertRaises(ValueError):
            TrainerConfig(log_path=Path("not-a-string"))  # type: ignore[arg-type]

    def test_patience_fifteen_stops_and_saves_only_first_best(self) -> None:
        from solution.radio_map.learning.trainer import Trainer, TrainerConfig

        class ScalarModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.value = torch.nn.Parameter(torch.zeros(()))

        trainer = Trainer(ScalarModel(), None, TrainerConfig(epochs=30, patience=15))
        trainer._loss_for_batch = lambda _batch: (None, SimpleNamespace(total=trainer.model.value.square() + 1.0))  # type: ignore[method-assign]
        scores = iter([1.0] + [0.0] * 15 + [2.0])
        trainer.validate = lambda _loader: {"score": next(scores)}  # type: ignore[method-assign]
        trainer.save_checkpoint = Mock(return_value=Path("unused.pt"))  # type: ignore[method-assign]
        records = trainer.fit([{}], [{}], checkpoint_path=Path("unused.pt"))
        self.assertEqual(len(records), 16)
        self.assertEqual(trainer._bad_epochs, 15)
        trainer.save_checkpoint.assert_called_once_with(Path("unused.pt"))

    def test_checkpoint_roundtrip_and_hash_mismatch(self) -> None:
        from solution.radio_map.learning.trainer import Trainer, TrainerConfig
        from solution.radio_map.learning.losses import anchor_completion_loss

        config = SupportAwareAnchorMixerConfig(self.adapter.coefficient_count, d_model=12, group_dim=8, geometry_dim=8, low_rank=4, num_frequencies=2)
        trainer = Trainer(SupportAwareAnchorMixer(config, torch.as_tensor(self.adapter.group_ids)), self.adapter, TrainerConfig(epochs=2), model_config=config.__dict__, manifest_hash="a" * 64)
        trainer.optimizer.zero_grad()
        anchor_completion_loss(trainer.model(self.batch), self.batch["target_latent"], self.batch["target_channel"], self.adapter).total.backward()
        trainer.optimizer.step()
        expected = trainer.model(self.batch).latent.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            trainer.save_checkpoint(path)
            self.assertEqual(len(path.with_suffix(".pt.sha256").read_text().strip()), 64)
            def factory(model_config: dict, group_ids: torch.Tensor) -> SupportAwareAnchorMixer:
                return SupportAwareAnchorMixer(SupportAwareAnchorMixerConfig(**model_config), group_ids)
            restored = Trainer.load_checkpoint(path, factory, adapter=self.adapter, expected_manifest_hash="a" * 64)
            torch.testing.assert_close(restored.model(self.batch).latent, expected)
            self.assertTrue(restored.optimizer.state_dict()["state"])
            with self.assertRaises(ValueError):
                Trainer.load_checkpoint(path, factory, adapter=self.adapter, expected_manifest_hash="wrong")
            with self.assertRaises(ValueError):
                Trainer.load_checkpoint(path, factory, adapter=self.adapter, expected_adapter_hash="wrong")
            mismatched_adapter = _adapter()
            mismatched_adapter.mean[0] += 1.0
            with self.assertRaises(ValueError):
                Trainer.load_checkpoint(path, factory, adapter=mismatched_adapter, expected_adapter_hash=trainer.adapter_hash)
            from solution.radio_map.learning.trainer import _load_checkpoint_payload
            with patch("solution.radio_map.learning.trainer.torch.load", wraps=torch.load) as load_mock:
                _load_checkpoint_payload(path)
            self.assertEqual(load_mock.call_args.kwargs["map_location"], "cpu")
            payload = torch.load(path, weights_only=False, map_location="cpu")
            payload["format_version"] = True
            torch.save(payload, path)
            path.with_name(path.name + ".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n", encoding="ascii")
            with self.assertRaises(ValueError):
                Trainer.load_checkpoint(path, factory, adapter=self.adapter)

    def test_last_checkpoint_resume_matches_uninterrupted_training(self) -> None:
        from solution.radio_map.learning.trainer import Trainer, TrainerConfig

        config = SupportAwareAnchorMixerConfig(self.adapter.coefficient_count, d_model=12, group_dim=8, geometry_dim=8, low_rank=4, num_frequencies=2)
        torch.manual_seed(123)
        initial = SupportAwareAnchorMixer(config, torch.as_tensor(self.adapter.group_ids))
        whole = Trainer(copy.deepcopy(initial), self.adapter, TrainerConfig(epochs=2), model_config=config.__dict__, manifest_hash="c" * 64)
        interrupted = Trainer(copy.deepcopy(initial), self.adapter, TrainerConfig(epochs=2), model_config=config.__dict__, manifest_hash="c" * 64)
        whole.fit([self.batch], [self.batch])
        with tempfile.TemporaryDirectory() as directory:
            last = Path(directory) / "last.pt"
            original_save = interrupted.save_checkpoint

            def stop_after_last(path: str | Path) -> Path:
                saved = original_save(path)
                if Path(path) == last:
                    raise RuntimeError("simulated interruption")
                return saved

            interrupted.save_checkpoint = stop_after_last  # type: ignore[method-assign]
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                interrupted.fit([self.batch], [self.batch], last_checkpoint_path=last)
            self.assertTrue(last.is_file())
            self.assertTrue(last.with_name("last.pt.sha256").is_file())
            restored = Trainer.load_checkpoint(last, lambda values, ids: SupportAwareAnchorMixer(SupportAwareAnchorMixerConfig(**values), ids), adapter=self.adapter, expected_manifest_hash="c" * 64)
            restored.fit([self.batch], [self.batch])
        for expected, actual in zip(whole.model.parameters(), restored.model.parameters()):
            torch.testing.assert_close(expected, actual)

    def test_checkpoint_schema_rejects_before_factory(self) -> None:
        from solution.radio_map.learning.trainer import Trainer, TrainerConfig

        config = SupportAwareAnchorMixerConfig(self.adapter.coefficient_count, d_model=12, group_dim=8, geometry_dim=8, low_rank=4, num_frequencies=2)
        trainer = Trainer(SupportAwareAnchorMixer(config, torch.as_tensor(self.adapter.group_ids)), self.adapter, TrainerConfig(epochs=2), model_config=config.__dict__, manifest_hash="b" * 64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schema.pt"
            trainer.save_checkpoint(path)
            original = torch.load(path, weights_only=False, map_location="cpu")
            factory = Mock()

            def reject(mutator) -> None:
                payload = copy.deepcopy(original)
                mutator(payload)
                torch.save(payload, path)
                path.with_name(path.name + ".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n", encoding="ascii")
                factory.reset_mock()
                with self.assertRaises(ValueError):
                    Trainer.load_checkpoint(path, factory, adapter=self.adapter)
                factory.assert_not_called()

            cases = (
                lambda value: value.update({"unknown": 1}),
                lambda value: value.pop("scheduler"),
                lambda value: value.update({"format_version": True}),
                lambda value: value["rng"].update({"python": (1, 2)}),
                lambda value: value["rng"].update({"numpy": ("bad",)}),
                lambda value: value["rng"].update({"torch": torch.zeros(8)}),
                lambda value: value["rng"].update({"cuda": [torch.zeros(8, dtype=torch.float32)]}),
                lambda value: value.update({"group_ids": torch.zeros(2, 2, dtype=torch.long)}),
                lambda value: value.update({"group_ids": torch.tensor([0.0, 1.0])}),
                lambda value: value.update({"group_ids": torch.tensor([0, -1])}),
                lambda value: value.update({"adapter_hash": "xyz"}),
                lambda value: value.update({"manifest_hash": "g" * 64}),
                lambda value: value.update({"epoch": -2}),
                lambda value: value.update({"bad_epochs": -1}),
                lambda value: value.update({"best_score": float("nan")}),
                lambda value: value.update({"best_score": float("inf")}),
                lambda value: value.update({"best_score": float("-inf"), "epoch": 0}),
                lambda value: value.update({"best_metrics": {"score": float("inf")}}),
                lambda value: value.update({"trainer_config": {"epochs": 0}}),
                lambda value: value.update({"trainer_config": {**value["trainer_config"], "epochs": True}}),
                lambda value: value.update({"trainer_config": {**value["trainer_config"], "learning_rate": float("nan")}}),
                lambda value: value.update({"loss_config": {"pas_weight": 9.0}}),
                lambda value: value.update({"model_config": {**value["model_config"], "unknown": 1}}),
                lambda value: value.update({"model_config": {**value["model_config"], "use_geometry": 0}}),
                lambda value: value.update({"model_config": {**value["model_config"], "latent_size": value["model_config"]["latent_size"] + 1}}),
                lambda value: value.update({"model_config": {**value["model_config"], "group_count": 1}}),
            )
            for mutator in cases:
                with self.subTest(mutator=mutator):
                    reject(mutator)
            path.with_name(path.name + ".sha256").write_text("0" * 64 + "\n", encoding="ascii")
            with self.assertRaises(ValueError):
                Trainer.load_checkpoint(path, factory, adapter=self.adapter)

    def test_eight_sample_overfit_reduces_loss(self) -> None:
        from solution.radio_map.learning.losses import AnchorLossConfig, anchor_completion_loss

        torch.manual_seed(7)
        adapter = self.adapter
        batch = _batch(adapter, 8)
        config = SupportAwareAnchorMixerConfig(adapter.coefficient_count, d_model=16, group_dim=12, geometry_dim=8, low_rank=adapter.coefficient_count, num_frequencies=2)
        model = SupportAwareAnchorMixer(config, torch.as_tensor(adapter.group_ids))
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-2, weight_decay=0.0)
        def value() -> torch.Tensor:
            return anchor_completion_loss(model(batch), batch["target_latent"], batch["target_channel"], adapter, AnchorLossConfig(nearest_weight=0.05)).total
        initial = value().item()
        for _ in range(80):
            optimizer.zero_grad(); loss = value(); loss.backward(); optimizer.step()
        final = value().item()
        self.assertLess(final, initial * 0.35)


if __name__ == "__main__":
    unittest.main()
