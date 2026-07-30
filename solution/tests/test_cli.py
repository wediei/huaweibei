from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from solution.tests.fixtures import create_round_dir


class CommandLineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_dir = create_round_dir(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_audit_cli_writes_json(self) -> None:
        from solution.radio_map.cli import main

        output = self.root / "reports" / "audit.json"
        exit_code = main(
            ["audit", "--data-dir", str(self.data_dir), "--output", str(output)]
        )

        self.assertEqual(exit_code, 0)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["actual_train_count"], 6)
        self.assertEqual(report["channel_dtype"], "complex64")

    def test_analyze_cli_reports_all_six_layouts(self) -> None:
        from solution.radio_map.cli import main

        output = self.root / "analysis.json"
        exit_code = main(
            [
                "analyze",
                "--data-dir",
                str(self.data_dir),
                "--sample-count",
                "3",
                "--output",
                str(output),
            ]
        )

        self.assertEqual(exit_code, 0)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(len(report["layout"]["candidates"]), 6)
        self.assertEqual(report["coverage"]["test_count"], 2)

    def test_baseline_cli_records_split_and_both_models(self) -> None:
        from solution.radio_map.cli import main

        output = self.root / "baselines.json"
        exit_code = main(
            [
                "baseline",
                "--data-dir",
                str(self.data_dir),
                "--validation-fraction",
                "0.34",
                "--grid-size",
                "2.0",
                "--batch-size",
                "1",
                "--output",
                str(output),
            ]
        )

        self.assertEqual(exit_code, 0)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(len(report["split"]["validation"]), 2)
        self.assertEqual(set(report["models"]), {"nearest", "inverse_distance"})
        self.assertTrue(np.isfinite(report["models"]["nearest"]["score"]))

    def test_validate_submission_rejects_wrong_shape(self) -> None:
        from solution.radio_map.cli import validate_submission
        from solution.radio_map.data import RoundDataset

        path = self.root / "Round1_Test_Channel.npy"
        np.save(path, np.zeros((2, 8, 2, 3), dtype=np.complex64))

        with self.assertRaisesRegex(ValueError, "shape"):
            validate_submission(path, RoundDataset.open(self.data_dir))

    def test_validate_submission_accepts_exact_complex64_file(self) -> None:
        from solution.radio_map.cli import validate_submission
        from solution.radio_map.data import RoundDataset

        path = self.root / "Round1_Test_Channel.npy"
        np.save(path, np.zeros((2, 8, 2, 4), dtype=np.complex64))

        report = validate_submission(path, RoundDataset.open(self.data_dir))

        self.assertEqual(report["shape"], [2, 8, 2, 4])
        self.assertEqual(report["dtype"], "complex64")
        self.assertTrue(report["finite"])
        self.assertEqual(len(report["sha256"]), 64)
        self.assertEqual(report["zero_sample_count"], 2)
        self.assertEqual(report["duplicate_sample_count"], 1)
        self.assertEqual(report["unique_sample_count"], 1)
        self.assertEqual(report["global_mean_power"], 0.0)
        self.assertEqual(report["sample_mean_power_quantiles"]["max"], 0.0)
        self.assertTrue(report["warnings"])

    def test_validate_submission_audits_distinct_sample_power(self) -> None:
        from solution.radio_map.cli import validate_submission
        from solution.radio_map.data import RoundDataset

        path = self.root / "Round1_Test_Channel.npy"
        values = np.empty((2, 8, 2, 4), dtype=np.complex64)
        values[0] = 1.0 + 0.0j
        values[1] = 0.0 + 2.0j
        np.save(path, values)

        report = validate_submission(path, RoundDataset.open(self.data_dir), batch_size=1)

        self.assertEqual(report["zero_sample_count"], 0)
        self.assertEqual(report["duplicate_sample_count"], 0)
        self.assertEqual(report["unique_sample_count"], 2)
        self.assertAlmostEqual(report["global_mean_power"], 2.5)
        self.assertAlmostEqual(report["peak_magnitude"], 2.0)
        self.assertAlmostEqual(report["sample_mean_power_quantiles"]["min"], 1.0)
        self.assertAlmostEqual(report["sample_mean_power_quantiles"]["max"], 4.0)
        self.assertEqual(report["warnings"], [])

    def test_codec_oracle_cli_is_leakage_free_and_finite(self) -> None:
        from solution.radio_map.cli import main

        output = self.root / "codec_oracle.json"
        exit_code = main(
            [
                "codec-oracle",
                "--data-dir",
                str(self.data_dir),
                "--output",
                str(output),
                "--validation-fraction",
                "0.34",
                "--grid-size",
                "2.0",
                "--layout-order",
                "PHV",
                "--tucker-ranks",
                "1,2,1,2,2",
                "--support-fractions",
                "0.25,0.5",
                "--fit-samples",
                "3",
                "--validation-samples",
                "2",
                "--batch-size",
                "1",
            ]
        )

        self.assertEqual(exit_code, 0)
        report = json.loads(output.read_text(encoding="utf-8"))
        fit_indices = set(report["split"]["fit_indices"])
        validation_indices = set(report["split"]["validation_indices"])
        self.assertTrue(fit_indices.isdisjoint(validation_indices))
        self.assertEqual(report["split"]["overlap_count"], 0)
        self.assertEqual(report["tucker"]["ranks"], [1, 2, 1, 2, 2])
        self.assertEqual(len(report["sparse_candidates"]), 2)
        self.assertTrue(np.isfinite(report["tucker"]["metrics"]["score"]))
        for candidate in report["sparse_candidates"]:
            self.assertTrue(
                np.isfinite(candidate["global_support_metrics"]["score"])
            )
            self.assertTrue(
                np.isfinite(candidate["per_sample_topk_oracle_metrics"]["score"])
            )

    def test_build_geometry_cli_writes_cache_and_report(self) -> None:
        from solution.radio_map.cli import main
        from solution.radio_map.geometry import GeometryPrior

        cache = self.root / "geometry" / "prior.npz"
        report_path = self.root / "geometry" / "report.json"
        exit_code = main(
            [
                "build-geometry",
                "--data-dir",
                str(self.data_dir),
                "--cache",
                str(cache),
                "--output",
                str(report_path),
                "--resolution",
                "1.0",
                "--height-layers",
                "4",
                "--batch-size",
                "1",
            ]
        )

        self.assertEqual(exit_code, 0)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        prior = GeometryPrior.load(cache)
        self.assertEqual(report["feature_count"], 13)
        self.assertEqual(report["metadata"]["source_vertex_count"], 2)
        self.assertEqual(prior.features.shape[0], 13)


class LearningCommandLineTests(unittest.TestCase):
    """Small CPU release gate for the separate learning command surface."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_dir = create_round_dir(self.root)
        from solution.radio_map.geometry import PlyPointCloud, build_geometry_prior

        self.geometry = self.root / "geometry.npz"
        build_geometry_prior(
            PlyPointCloud.open(self.data_dir / "Round1_Map.ply"),
            height_layers=4,
            batch_size=1,
        ).save(self.geometry)
        self.cache = self.root / "fold"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _prepare(self) -> None:
        from solution.radio_map.learning.cli import main

        self.assertEqual(main([
            "prepare-cache", "--data-dir", str(self.data_dir),
            "--geometry-cache", str(self.geometry), "--output-dir", str(self.cache),
            "--validation-fraction", "0.34", "--grid-size", "2.0", "--k-max", "3",
            "--anchor-count", "2", "--patch", "3", "--corridor", "2",
            "--anchor-corridor", "2", "--min-anchors", "1", "--channel-batch-size", "1",
        ]), 0)

    def test_prepare_persists_and_authenticates_exact_split(self) -> None:
        from solution.radio_map.learning.cache import FoldCacheManifest, validate_cache

        self._prepare()
        report = json.loads((self.cache / "prepare_report.json").read_text())
        self.assertEqual(report["overlap_count"], 0)
        self.assertTrue((self.cache / "train_indices.npy").is_file())
        self.assertTrue((self.cache / "validation_indices.npy").is_file())
        values = np.load(self.cache / "train_indices.npy")
        np.save(self.cache / "train_indices.npy", values[::-1])
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_cache(FoldCacheManifest.load(self.cache / "manifest.json"), self.cache)

    def test_prepare_supports_leakage_free_spatial_block_protocol(self) -> None:
        from solution.radio_map.learning.cli import main

        block_cache = self.root / "block_fold"
        self.assertEqual(main([
            "prepare-cache", "--data-dir", str(self.data_dir),
            "--geometry-cache", str(self.geometry), "--output-dir", str(block_cache),
            "--split-protocol", "block", "--block-axis", "0", "--block-side", "high",
            "--validation-fraction", "0.34", "--k-max", "3", "--anchor-count", "2",
            "--patch", "3", "--corridor", "2", "--anchor-corridor", "2",
            "--min-anchors", "1", "--channel-batch-size", "1",
        ]), 0)
        report = json.loads((block_cache / "prepare_report.json").read_text())
        self.assertEqual(report["protocol"], "block_split")
        train = np.load(block_cache / "train_indices.npy")
        validation = np.load(block_cache / "validation_indices.npy")
        positions = np.load(self.data_dir / "Round1_Train_Pos.npy")
        self.assertLessEqual(
            float(positions[train, 0].max()),
            float(positions[validation, 0].min()),
        )

    def test_cpu_train_evaluate_and_test_coordinate_infer(self) -> None:
        from solution.radio_map.learning.cli import main
        from solution.radio_map.learning.dataset import build_coordinate_batch

        self._prepare()
        run = self.root / "run"
        common = ["--d-model", "8", "--group-dim", "4", "--geometry-dim", "4", "--low-rank", "2", "--num-frequencies", "2"]
        self.assertEqual(main([
            "train", "--data-dir", str(self.data_dir), "--cache-dir", str(self.cache), "--run-dir", str(run),
            "--device", "cpu", "--epochs", "1", "--batch-size", "2", "--warmup-epochs", "1", "--patience", "1", "--no-geometry", *common,
        ]), 0)
        checkpoint = run / "best.pt"
        self.assertTrue(checkpoint.is_file())
        self.assertTrue(checkpoint.with_name("best.pt.sha256").is_file())
        self.assertEqual(main([
            "evaluate", "--data-dir", str(self.data_dir), "--cache-dir", str(self.cache), "--checkpoint", str(checkpoint), "--batch-size", "2",
        ]), 0)
        evaluation = json.loads((run / "evaluate_report.json").read_text())
        self.assertTrue(np.isfinite(evaluation["metrics"]["score"]))
        submission = self.root / "Round1_Test_Channel.npy"
        self.assertEqual(main([
            "infer", "--data-dir", str(self.data_dir), "--cache-dir", str(self.cache), "--checkpoint", str(checkpoint), "--output", str(submission), "--batch-size", "2",
        ]), 0)
        prediction = np.load(submission)
        self.assertEqual(prediction.shape, (2, 8, 2, 4))
        self.assertEqual(prediction.dtype, np.complex64)
        self.assertTrue(np.isfinite(prediction).all())
        report = json.loads((self.root / "infer_report.json").read_text())
        self.assertFalse(report["train_dataset_rows_used_for_test_positions"])
        test_batch = build_coordinate_batch(
            self.cache, np.load(self.data_dir / "Round1_Test_Pos.npy"),
            use_geometry=False,
        )
        np.testing.assert_allclose(
            test_batch["target_positions"].numpy(),
            np.load(self.data_dir / "Round1_Test_Pos.npy").astype(np.float32),
        )
        self.assertTrue(set(test_batch["anchor_indices"].numpy().ravel()).issubset(
            set(np.load(self.cache / "train_indices.npy").tolist())
        ))

    def test_coordinate_context_matches_cached_geometry_quantization(self) -> None:
        from solution.radio_map.geometry import GeometryPrior
        from solution.radio_map.learning.dataset import CoordinateBatchContext

        self._prepare()
        train = np.load(self.cache / "train_indices.npy")
        source = int(train[0])
        context = CoordinateBatchContext.open(
            self.cache, use_geometry=True, geometry=GeometryPrior.load(self.geometry),
        )
        batch = context.build(np.asarray([np.load(self.data_dir / "Round1_Train_Pos.npy")[source]]))
        np.testing.assert_array_equal(batch["target_point_features"].numpy()[0], np.load(self.cache / "target_point_features.npy")[source])
        np.testing.assert_array_equal(batch["target_patch"].numpy()[0], np.load(self.cache / "target_patches.npy")[source].astype(np.float32))
        np.testing.assert_array_equal(batch["bs_patch"].numpy()[0], np.load(self.cache / "bs_patch.npy").astype(np.float32))
        np.testing.assert_array_equal(batch["bs_target_corridor"].numpy()[0], np.load(self.cache / "bs_target_corridors.npy")[source].astype(np.float32))

    def test_epoch_seeded_sampler_recreates_resume_epoch_order(self) -> None:
        from solution.radio_map.learning.cli import _EpochSeededSampler
        from solution.radio_map.learning.dataset import CachedAnchorDataset

        self._prepare()
        train = np.load(self.cache / "train_indices.npy")
        first = _EpochSeededSampler(CachedAnchorDataset(self.cache, train, True, False), 42)
        resumed = _EpochSeededSampler(CachedAnchorDataset(self.cache, train, True, False), 42)
        first.set_epoch(3); resumed.set_epoch(3)
        self.assertEqual(list(first), list(resumed))

    def test_cli_negative_paths_and_infer_reuses_context(self) -> None:
        from unittest.mock import patch
        from solution.radio_map.anchors import AnchorMemory
        from solution.radio_map.learning import cli as learning_cli
        from solution.radio_map.learning.cli import main

        self._prepare()
        run = self.root / "run"
        common = ["--d-model", "8", "--group-dim", "4", "--geometry-dim", "4", "--low-rank", "2", "--num-frequencies", "2"]
        train = ["train", "--data-dir", str(self.data_dir), "--cache-dir", str(self.cache), "--run-dir", str(run), "--epochs", "1", "--batch-size", "2", "--no-geometry", *common]
        self.assertEqual(main(train), 0)
        checkpoint = run / "best.pt"
        with self.assertRaisesRegex(ValueError, "positive"):
            main([*train, "--run-dir", str(self.root / "bad"), "--limit-train-samples", "0"])
        # This negative-path assertion must be independent of the machine
        # running the tests: the training CLI should reject CUDA only when it
        # is unavailable, while a 4090 server is expected to accept it.
        with patch.object(learning_cli.torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA"):
                main([*train, "--run-dir", str(self.root / "cuda"), "--device", "cuda"])
        with self.assertRaisesRegex(ValueError, "zero epochs"):
            main([*train, "--resume"])
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            main([*train, "--resume", "--learning-rate", "0.001"])
        with self.assertRaisesRegex(ValueError, "positive"):
            main(["evaluate", "--data-dir", str(self.data_dir), "--cache-dir", str(self.cache), "--checkpoint", str(checkpoint), "--limit-samples", "0"])
        output = self.root / "existing.npy"
        np.save(output, np.zeros((2, 8, 2, 4), np.complex64))
        original = output.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            main(["infer", "--data-dir", str(self.data_dir), "--cache-dir", str(self.cache), "--checkpoint", str(checkpoint), "--output", str(output)])
        self.assertEqual(output.read_bytes(), original)
        fresh = self.root / "fresh.npy"
        with patch.object(learning_cli, "validate_cache", wraps=learning_cli.validate_cache) as validate_mock, patch("solution.radio_map.learning.dataset.AnchorMemory", wraps=AnchorMemory) as memory_mock:
            self.assertEqual(main(["infer", "--data-dir", str(self.data_dir), "--cache-dir", str(self.cache), "--checkpoint", str(checkpoint), "--output", str(fresh), "--batch-size", "1"]), 0)
        self.assertEqual(validate_mock.call_count, 1)
        self.assertEqual(memory_mock.call_count, 1)
        self.assertFalse(fresh.with_name(fresh.name + ".tmp").exists())
        other = self.root / "other-fold"
        self.assertEqual(main([
            "prepare-cache", "--data-dir", str(self.data_dir), "--geometry-cache", str(self.geometry), "--output-dir", str(other),
            "--split-seed", "43", "--validation-fraction", "0.34", "--grid-size", "2.0", "--k-max", "3", "--anchor-count", "2",
            "--patch", "3", "--corridor", "2", "--anchor-corridor", "2", "--min-anchors", "1", "--channel-batch-size", "1",
        ]), 0)
        with self.assertRaisesRegex(ValueError, "adapter|manifest"):
            main(["evaluate", "--data-dir", str(self.data_dir), "--cache-dir", str(other), "--checkpoint", str(checkpoint)])
        with self.assertRaisesRegex(ValueError, "adapter|manifest"):
            main(["infer", "--data-dir", str(self.data_dir), "--cache-dir", str(other), "--checkpoint", str(checkpoint), "--output", str(self.root / "mismatch.npy")])
        self.assertFalse((self.root / "mismatch.npy").exists())


if __name__ == "__main__":
    unittest.main()
