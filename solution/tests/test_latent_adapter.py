from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from solution.radio_map.transforms import AntennaLayout, beam_delay
from solution.tests.test_transforms import make_config


class FixedSupportLatentAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = make_config()
        self.layout = AntennaLayout(self.config, ("P", "H", "V"))
        rng = np.random.default_rng(53)
        self.channels = (
            rng.standard_normal((5, 8, 2, 4))
            + 1j * rng.standard_normal((5, 8, 2, 4))
        ).astype(np.complex64)

    def test_fixed_support_groups_follow_p_n_delay_blocks(self) -> None:
        from solution.radio_map.learning.latent_adapter import FixedSupportLatentAdapter

        adapter = FixedSupportLatentAdapter(
            self.layout, support_fraction=0.10, delay_block=2
        ).fit(self.channels, np.arange(4), batch_size=2)

        coordinates = adapter.latent_coordinates
        expected = (
            (coordinates[:, 2] * self.config.n + coordinates[:, 3]) * 2
            + coordinates[:, 4] // 2
        )
        np.testing.assert_array_equal(adapter.group_ids, expected)

    def test_statistics_exclude_validation_and_normalize_fit_latents(self) -> None:
        from solution.radio_map.learning.latent_adapter import FixedSupportLatentAdapter

        adapter = FixedSupportLatentAdapter(self.layout, support_fraction=0.25).fit(
            self.channels, np.array([0, 1]), batch_size=1
        )
        normalized = adapter.encode_numpy(self.channels[[0, 1]])

        self.assertEqual(set(adapter.fitted_indices), {0, 1})
        self.assertNotIn(2, adapter.fitted_indices)
        np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=2e-6)
        np.testing.assert_allclose(
            np.mean(np.abs(normalized) ** 2, axis=0), 1.0, rtol=2e-5, atol=2e-5
        )

    def test_decode_matches_selected_beam_delay_coefficients(self) -> None:
        from solution.radio_map.learning.latent_adapter import FixedSupportLatentAdapter

        adapter = FixedSupportLatentAdapter(self.layout, support_fraction=0.25).fit(
            self.channels, np.arange(4), batch_size=2
        )
        decoded = adapter.decode_numpy(adapter.encode_numpy(self.channels[4:]))
        decoded_beam = beam_delay(decoded, self.layout).reshape(1, -1)
        source_beam = beam_delay(self.channels[4:], self.layout).reshape(1, -1)

        np.testing.assert_allclose(
            decoded_beam[:, adapter.support_indices],
            source_beam[:, adapter.support_indices],
            rtol=2e-5,
            atol=2e-6,
        )

    def test_decode_torch_is_differentiable_and_stays_on_input_device(self) -> None:
        from solution.radio_map.learning.latent_adapter import FixedSupportLatentAdapter

        adapter = FixedSupportLatentAdapter(self.layout, support_fraction=0.25).fit(
            self.channels, np.arange(4), batch_size=2
        )
        latent = torch.tensor(adapter.encode_numpy(self.channels[4:]), requires_grad=True)
        decoded = adapter.decode_torch(latent)
        decoded.abs().square().mean().backward()

        self.assertEqual(decoded.device, latent.device)
        self.assertEqual(decoded.shape, (1,) + self.config.channel_shape)
        self.assertIsNotNone(latent.grad)
        self.assertTrue(torch.isfinite(latent.grad).all())
        np.testing.assert_allclose(
            decoded.detach().numpy(),
            adapter.decode_numpy(latent.detach().numpy()),
            rtol=2e-5,
            atol=2e-6,
        )

    def test_save_load_preserves_decode_and_metadata_without_pickle(self) -> None:
        from solution.radio_map.learning.latent_adapter import FixedSupportLatentAdapter

        adapter = FixedSupportLatentAdapter(
            self.layout, support_fraction=0.25, delay_block=2
        ).fit(self.channels, np.array([0, 1, 3]), batch_size=2)
        latent = adapter.encode_numpy(self.channels[2:3])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.npz"
            adapter.save(path)
            with np.load(path, allow_pickle=False) as stored:
                self.assertIn("metadata_json", stored.files)
                self.assertIn("support_indices", stored.files)
            loaded = FixedSupportLatentAdapter.load(path, self.layout)

        np.testing.assert_allclose(loaded.decode_numpy(latent), adapter.decode_numpy(latent))
        np.testing.assert_array_equal(loaded.support_indices, adapter.support_indices)
        np.testing.assert_array_equal(loaded.group_ids, adapter.group_ids)
        self.assertEqual(loaded.fit_indices_sha256, adapter.fit_indices_sha256)
        self.assertEqual(loaded.source_shape, self.channels.shape)

    def test_load_rejects_invalid_archive_invariants(self) -> None:
        from solution.radio_map.learning.latent_adapter import FixedSupportLatentAdapter

        adapter = FixedSupportLatentAdapter(
            self.layout, support_fraction=0.25, delay_block=2
        ).fit(self.channels, np.array([0, 1, 3]), batch_size=2)

        def mutate_metadata(key: str, value: object):
            def mutate(data: dict[str, np.ndarray], metadata: dict[str, object]) -> None:
                del data
                metadata[key] = value

            return mutate

        def mutate_array(key: str, value: np.ndarray):
            def mutate(data: dict[str, np.ndarray], metadata: dict[str, object]) -> None:
                del metadata
                data[key] = value

            return mutate

        valid_fit_hash = adapter.fit_indices_sha256
        support = adapter.support_indices.copy()
        coordinates = adapter.latent_coordinates.copy()
        groups = adapter.group_ids.copy()
        mean = adapter.mean.copy()
        rms = adapter.rms.copy()
        invalid_archives = {
            "unknown format version": mutate_metadata("format_version", 99),
            "invalid delay block": mutate_metadata("delay_block", None),
            "metadata coefficient count": mutate_metadata(
                "coefficient_count", adapter.coefficient_count + 1
            ),
            "source shape rank": mutate_metadata("source_shape", [5, 8, 2]),
            "source shape tail": mutate_metadata("source_shape", [5, 8, 2, 5]),
            "support shape": mutate_array("support_indices", support[None, :]),
            "support duplicate": mutate_array(
                "support_indices", np.concatenate((support[:1], support[:1], support[2:]))
            ),
            "support negative": mutate_array(
                "support_indices", np.concatenate((np.array([-1]), support[1:]))
            ),
            "support out of range": mutate_array(
                "support_indices",
                np.concatenate((np.array([adapter.total_coefficients]), support[1:])),
            ),
            "mean shape": mutate_array("mean", mean[:, None]),
            "mean finite": mutate_array(
                "mean", np.concatenate((np.array([np.inf + 0j]), mean[1:]))
            ),
            "rms shape": mutate_array("rms", rms[:, None]),
            "rms finite": mutate_array(
                "rms", np.concatenate((np.array([np.inf]), rms[1:]))
            ),
            "rms positive": mutate_array(
                "rms", np.concatenate((np.array([0.0]), rms[1:]))
            ),
            "coordinate shape": mutate_array("latent_coordinates", coordinates[:, :4]),
            "coordinate values": mutate_array(
                "latent_coordinates",
                np.vstack(
                    (
                        (coordinates[0] + np.array([1, 0, 0, 0, 0]))
                        % np.array(self.layout.structured_tail),
                        coordinates[1:],
                    )
                ),
            ),
            "group shape": mutate_array("group_ids", groups[:, None]),
            "group values": mutate_array("group_ids", groups + 1),
            "fit index shape": mutate_array("fitted_indices", adapter.fitted_indices[:, None]),
            "fit index empty": mutate_array(
                "fitted_indices", np.array([], dtype=np.int64)
            ),
            "fit index duplicate": mutate_array(
                "fitted_indices", np.array([0, 0, 3], dtype=np.int64)
            ),
            "fit index range": mutate_array(
                "fitted_indices", np.array([0, 1, len(self.channels)], dtype=np.int64)
            ),
            "fit index sha": mutate_metadata("fit_indices_sha256", "0" * 64),
        }

        with tempfile.TemporaryDirectory() as directory:
            base_path = Path(directory) / "base.npz"
            adapter.save(base_path)
            with np.load(base_path, allow_pickle=False) as stored:
                base_data = {name: stored[name].copy() for name in stored.files}
            for name, mutate in invalid_archives.items():
                with self.subTest(name=name):
                    data = {key: value.copy() for key, value in base_data.items()}
                    metadata = json.loads(str(data["metadata_json"].item()))
                    mutate(data, metadata)
                    if name != "fit index sha":
                        fit_indices = np.asarray(data["fitted_indices"], dtype=np.int64)
                        metadata["fit_indices_sha256"] = FixedSupportLatentAdapter._indices_sha256(
                            fit_indices
                        )
                    self.assertNotEqual(
                        metadata["fit_indices_sha256"], "", msg=valid_fit_hash
                    )
                    data["metadata_json"] = np.asarray(
                        json.dumps(metadata, sort_keys=True)
                    )
                    path = Path(directory) / f"{name}.npz"
                    np.savez_compressed(path, **data)
                    with self.assertRaises(ValueError):
                        FixedSupportLatentAdapter.load(path, self.layout)

    def test_load_rejects_json_configuration_type_coercion(self) -> None:
        from solution.radio_map.learning.latent_adapter import FixedSupportLatentAdapter

        adapter = FixedSupportLatentAdapter(
            self.layout, support_fraction=0.25, delay_block=2
        ).fit(self.channels, np.array([0, 1, 3]), batch_size=2)
        invalid_metadata = {
            "format bool": ("format_version", True, "format_version must be a non-bool integer"),
            "format float": ("format_version", 1.0, "format_version must be a non-bool integer"),
            "layout string": ("layout_order", "PHV", "layout_order must be a list of three strings"),
            "layout non-string": (
                "layout_order",
                ["P", 1, "V"],
                "layout_order must be a list of three strings",
            ),
            "delay float": ("delay_block", 2.9, "delay_block must be a non-bool integer"),
            "delay bool": ("delay_block", True, "delay_block must be a non-bool integer"),
            "fraction string": (
                "support_fraction",
                "0.25",
                "support_fraction must be a finite number",
            ),
            "fraction bool": (
                "support_fraction",
                True,
                "support_fraction must be a finite number",
            ),
            "fraction nan": (
                "support_fraction",
                float("nan"),
                "support_fraction must be a finite number",
            ),
            "fraction inf": (
                "support_fraction",
                float("inf"),
                "support_fraction must be a finite number",
            ),
            "count float": (
                "coefficient_count",
                float(adapter.coefficient_count),
                "coefficient_count must be a non-bool integer",
            ),
            "count bool": (
                "coefficient_count",
                True,
                "coefficient_count must be a non-bool integer",
            ),
        }

        with tempfile.TemporaryDirectory() as directory:
            base_path = Path(directory) / "base.npz"
            adapter.save(base_path)
            with np.load(base_path, allow_pickle=False) as stored:
                base_data = {name: stored[name].copy() for name in stored.files}
            for name, (key, value, message) in invalid_metadata.items():
                with self.subTest(name=name):
                    data = {array_name: array.copy() for array_name, array in base_data.items()}
                    metadata = json.loads(str(data["metadata_json"].item()))
                    metadata[key] = value
                    data["metadata_json"] = np.asarray(
                        json.dumps(metadata, sort_keys=True)
                    )
                    path = Path(directory) / f"{name}.npz"
                    np.savez_compressed(path, **data)
                    with self.assertRaisesRegex(ValueError, message):
                        FixedSupportLatentAdapter.load(path, self.layout)


if __name__ == "__main__":
    unittest.main()
