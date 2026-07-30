from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from solution.radio_map.config import RoundConfig
from solution.radio_map.learning.e2e_cgpf import (
    E2ECGPF,
    E2ECGPFConfig,
    GaussianFieldConfig,
    PathNetworkConfig,
    RendererConfig,
    TrainableGaussianField,
)
from solution.radio_map.learning.e2e_cgpf_cli import (
    _require_passed_gate,
    build_parser,
)
from solution.radio_map.learning.e2e_cgpf_losses import E2ECGPFLossConfig
from solution.radio_map.learning.e2e_cgpf_runtime import (
    E2ECGPFTrainer,
    StageSpec,
    TrainingConfig,
    load_model_checkpoint,
)


def make_model() -> E2ECGPF:
    round_value = RoundConfig(
        p_train_declared=2,
        p_test=1,
        m=4,
        m_h=2,
        m_v=1,
        m_p=2,
        n=2,
        n_h=1,
        n_v=1,
        n_p=2,
        s=8,
        q=2,
        bs_position=(0.0, 0.0, 0.0),
        weights=(0.4, 0.4, 0.2),
    )
    field_config = GaussianFieldConfig(
        initial_count=3,
        max_count=5,
        min_count=2,
        material_dim=4,
        densify_count=1,
        prune_count=1,
    )
    config = E2ECGPFConfig(
        field=field_config,
        paths=PathNetworkConfig(
            hidden_dim=24,
            query_dim=12,
            modes_per_gaussian=2,
            selected_gaussians=2,
            polarization_rank=1,
            path_type_dim=3,
            fourier_bands=2,
            max_delay_bins=7.0,
        ),
        renderer=RendererConfig(path_chunk_size=2),
        seed=9,
    )
    centers = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [-0.5, 1.0, 0.2]]
    )
    normals = torch.tensor([[0.0, 0.0, 1.0]] * 3)
    field = TrainableGaussianField(centers, normals, field_config, seed=9)
    return E2ECGPF(
        field,
        round_value,
        config,
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor(2.0),
    )


class RuntimeTests(unittest.TestCase):
    def test_fit_writes_batch_epoch_logs_checkpoints_hashes_and_report(self) -> None:
        model = make_model()
        positions = torch.tensor([[1.0, 0.5, 0.2], [1.5, -0.5, 0.3]])
        with torch.no_grad():
            channels, _ = model(positions)
        loader = [
            {
                "source_index": torch.tensor([index]),
                "position": positions[index : index + 1],
                "channel": channels[index : index + 1].clone(),
            }
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer = E2ECGPFTrainer(
                model,
                TrainingConfig(
                    batch_size=1,
                    learning_rate=1e-3,
                    field_learning_rate=1e-4,
                    device="cpu",
                    densify_start_epoch=10,
                ),
                E2ECGPFLossConfig(causal_weight=0.0),
                root,
                train_indices=[0, 1],
                validation_indices=[0, 1],
            )
            result = trainer.fit(
                loader,
                loader,
                [StageSpec("A_test", "full", 1, False)],
            )
            self.assertTrue((root / "metrics.jsonl").is_file())
            records = [
                json.loads(line)
                for line in (root / "metrics.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertIn("batch", {record["kind"] for record in records})
            self.assertIn("epoch", {record["kind"] for record in records})
            for name in ("best.pt", "last.pt"):
                self.assertTrue((root / name).is_file())
                self.assertTrue((root / f"{name}.sha256").is_file())
            self.assertTrue((root / "config.json").is_file())
            self.assertTrue((root / "A_test_report.json").is_file())
            self.assertGreaterEqual(result["best_score"], 0.0)

            loaded, payload = load_model_checkpoint(root / "last.pt", "cpu")
            self.assertEqual(payload["format"], "e2e-cgpf-v1")
            with torch.no_grad():
                expected, _ = model(positions)
                actual, _ = loaded(positions)
            torch.testing.assert_close(actual, expected)

    def test_checkpoint_hash_mismatch_is_rejected(self) -> None:
        model = make_model()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer = E2ECGPFTrainer(
                model,
                TrainingConfig(device="cpu"),
                E2ECGPFLossConfig(),
                root,
                train_indices=[0],
                validation_indices=[0],
            )
            checkpoint = trainer.save_checkpoint(
                "last.pt", StageSpec("test", "full", 1)
            )
            checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_model_checkpoint(checkpoint, "cpu")

    def test_official_inference_gate_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            failed = Path(temporary) / "cross_validation_report.json"
            failed.write_text(
                json.dumps(
                    {
                        "kind": "full_spatial_cross_validation",
                        "passed": False,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "locked"):
                _require_passed_gate(failed)

    def test_cli_exposes_every_protocol_stage(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()
        for command in (
            "smoke",
            "capacity",
            "single-fold",
            "cross-validate",
            "full-train",
            "infer",
        ):
            self.assertIn(command, help_text)


if __name__ == "__main__":
    unittest.main()
