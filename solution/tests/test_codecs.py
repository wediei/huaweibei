from __future__ import annotations

import unittest

import numpy as np

from solution.radio_map.transforms import AntennaLayout
from solution.tests.test_transforms import make_config


class SharedTuckerCodecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = make_config()
        self.layout = AntennaLayout(self.config, ("P", "H", "V"))
        rng = np.random.default_rng(31)
        self.channels = (
            rng.standard_normal((7, 8, 2, 4))
            + 1j * rng.standard_normal((7, 8, 2, 4))
        ).astype(np.complex64)

    def test_full_rank_codec_roundtrip(self) -> None:
        from solution.radio_map.codecs import SharedTuckerCodec

        codec = SharedTuckerCodec(self.layout, ranks=(2, 2, 2, 2, 4))
        codec.fit(self.channels, train_indices=np.arange(5), batch_size=2)
        reconstructed = codec.reconstruct(self.channels[5:])

        np.testing.assert_allclose(
            reconstructed, self.channels[5:], rtol=2e-5, atol=2e-6
        )
        self.assertEqual(reconstructed.dtype, np.complex64)

    def test_reduced_rank_codec_has_expected_core_and_orthonormal_factors(self) -> None:
        from solution.radio_map.codecs import SharedTuckerCodec

        ranks = (1, 2, 1, 2, 2)
        codec = SharedTuckerCodec(self.layout, ranks=ranks)
        codec.fit(self.channels, train_indices=np.array([0, 2, 4, 6]), batch_size=3)
        core = codec.encode(self.channels[:2])

        self.assertEqual(core.shape, (2,) + ranks)
        self.assertEqual(core.dtype, np.complex64)
        for factor in codec.factors:
            np.testing.assert_allclose(
                factor.conj().T @ factor,
                np.eye(factor.shape[1]),
                rtol=1e-5,
                atol=1e-6,
            )

    def test_fit_records_exact_training_indices(self) -> None:
        from solution.radio_map.codecs import SharedTuckerCodec

        indices = np.array([6, 1, 4], dtype=np.int64)
        codec = SharedTuckerCodec(self.layout, ranks=(1, 1, 1, 1, 1))
        codec.fit(self.channels, train_indices=indices, batch_size=2)

        np.testing.assert_array_equal(codec.fitted_indices, indices)
        self.assertEqual(codec.fit_sample_count, 3)

    def test_invalid_rank_is_rejected(self) -> None:
        from solution.radio_map.codecs import SharedTuckerCodec

        with self.assertRaisesRegex(ValueError, "rank"):
            SharedTuckerCodec(self.layout, ranks=(3, 2, 2, 2, 4))

    def test_encode_before_fit_is_rejected(self) -> None:
        from solution.radio_map.codecs import SharedTuckerCodec

        codec = SharedTuckerCodec(self.layout, ranks=(1, 1, 1, 1, 1))

        with self.assertRaisesRegex(RuntimeError, "fitted"):
            codec.encode(self.channels[:1])


class SparseCodecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = make_config()
        self.layout = AntennaLayout(self.config, ("P", "H", "V"))
        rng = np.random.default_rng(37)
        self.channels = (
            rng.standard_normal((4, 8, 2, 4))
            + 1j * rng.standard_normal((4, 8, 2, 4))
        ).astype(np.complex64)

    def test_full_global_support_roundtrip(self) -> None:
        from solution.radio_map.codecs import GlobalSupportCodec

        codec = GlobalSupportCodec(self.layout, coefficient_count=64)
        codec.fit(self.channels, train_indices=np.array([0, 1, 2]), batch_size=2)
        reconstructed = codec.reconstruct(self.channels[3:])

        np.testing.assert_allclose(
            reconstructed, self.channels[3:], rtol=2e-5, atol=2e-6
        )

    def test_global_support_uses_only_selected_training_samples(self) -> None:
        from solution.radio_map.codecs import GlobalSupportCodec
        from solution.radio_map.transforms import inverse_beam_delay

        transformed = np.zeros((3, 2, 2, 2, 2, 4), dtype=np.complex64)
        transformed[:2].reshape(2, -1)[:, 3] = 2.0 + 0.0j
        transformed[2].reshape(1, -1)[:, 10] = 100.0 + 0.0j
        channels = inverse_beam_delay(transformed, self.layout)
        codec = GlobalSupportCodec(self.layout, coefficient_count=1)
        codec.fit(channels, train_indices=np.array([0, 1]), batch_size=1)

        np.testing.assert_array_equal(codec.support_indices, [3])
        np.testing.assert_array_equal(codec.fitted_indices, [0, 1])

    def test_oracle_topk_has_bounded_support_and_monotonic_error(self) -> None:
        from solution.radio_map.codecs import oracle_topk_reconstruction
        from solution.radio_map.transforms import beam_delay

        reconstructed_2 = oracle_topk_reconstruction(
            self.channels, self.layout, coefficient_count=2
        )
        reconstructed_5 = oracle_topk_reconstruction(
            self.channels, self.layout, coefficient_count=5
        )
        recovered_beam = beam_delay(reconstructed_2, self.layout).reshape(
            len(self.channels), -1
        )
        nonzero = np.count_nonzero(np.abs(recovered_beam) > 1e-6, axis=1)
        error_2 = np.sum(np.abs(reconstructed_2 - self.channels) ** 2)
        error_5 = np.sum(np.abs(reconstructed_5 - self.channels) ** 2)

        np.testing.assert_array_less(nonzero, np.full_like(nonzero, 3))
        self.assertLessEqual(float(error_5), float(error_2))

    def test_invalid_coefficient_count_is_rejected(self) -> None:
        from solution.radio_map.codecs import GlobalSupportCodec

        with self.assertRaisesRegex(ValueError, "coefficient_count"):
            GlobalSupportCodec(self.layout, coefficient_count=65)


if __name__ == "__main__":
    unittest.main()
