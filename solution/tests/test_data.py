from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from solution.tests.fixtures import create_round_dir


class RoundDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.round_dir = create_round_dir(Path(self.temp_dir.name))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_round_dataset_opens_channel_as_memmap(self) -> None:
        from solution.radio_map.data import RoundDataset

        dataset = RoundDataset.open(self.round_dir)

        self.assertIsInstance(dataset.train_channel, np.memmap)
        self.assertEqual(dataset.train_channel.shape, (6, 8, 2, 4))
        self.assertEqual(dataset.train_channel.dtype, np.dtype(np.complex64))

    def test_audit_records_declared_actual_count_mismatch(self) -> None:
        from solution.radio_map.data import RoundDataset

        audit = RoundDataset.open(self.round_dir).audit()

        self.assertEqual(audit.declared_train_count, 8)
        self.assertEqual(audit.actual_train_count, 6)
        self.assertEqual(audit.actual_test_count, 2)
        self.assertTrue(any("P_Train" in warning for warning in audit.warnings))

    def test_channel_batch_preserves_requested_order(self) -> None:
        from solution.radio_map.data import RoundDataset

        dataset = RoundDataset.open(self.round_dir)
        batch = dataset.channel_batch([4, 1])

        self.assertEqual(batch.shape, (2, 8, 2, 4))
        np.testing.assert_array_equal(batch[0], dataset.train_channel[4])
        np.testing.assert_array_equal(batch[1], dataset.train_channel[1])

    def test_open_rejects_channel_shape_mismatch(self) -> None:
        from solution.radio_map.data import RoundDataset

        bad = np.zeros((6, 7, 2, 4), dtype=np.complex64)
        np.save(self.round_dir / "Round1_Train_Channel.npy", bad)

        with self.assertRaisesRegex(ValueError, "channel shape"):
            RoundDataset.open(self.round_dir)


if __name__ == "__main__":
    unittest.main()
