import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from solution.radio_map.learning.ac_cgmrf_validation import (
    aggregate_cross_validation,
    audit_anchor_exclusion,
    pilot_gate,
    require_promotion_report,
    spatial_block_folds,
    spatial_fold_manifest,
)


def _metric(score, pas=0.7, pdp=0.7, nmse=1.0):
    return {"score": score, "pas": pas, "pdp": pdp, "nmse": nmse}


class SpatialValidationTests(unittest.TestCase):
    def test_spatial_folds_are_deterministic_disjoint_and_block_exclusive(self):
        positions = np.array(
            [
                [0.1, 0.1],
                [0.2, 0.2],
                [1.1, 0.1],
                [1.2, 0.2],
                [2.1, 0.1],
                [2.2, 0.2],
            ],
            dtype=np.float64,
        )
        first = spatial_block_folds(positions, 3, 1.0, seed=9)
        second = spatial_block_folds(positions, 3, 1.0, seed=9)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left.train, right.train)
            np.testing.assert_array_equal(left.validation, right.validation)
            self.assertEqual(np.intersect1d(left.train, left.validation).size, 0)
        covered = np.concatenate([fold.validation for fold in first])
        np.testing.assert_array_equal(np.sort(covered), np.arange(6))
        cells = np.floor(positions).astype(int)
        for cell in np.unique(cells, axis=0):
            members = set(np.flatnonzero(np.all(cells == cell, axis=1)))
            held_out = [
                index
                for index, fold in enumerate(first)
                if members.intersection(set(fold.validation))
            ]
            self.assertEqual(len(held_out), 1)
        manifest = spatial_fold_manifest(positions, first, 1.0, 9)
        self.assertEqual(manifest["sample_count"], 6)
        self.assertEqual(manifest["fold_count"], 3)

    def test_anchor_audit_rejects_self_and_validation_anchors(self):
        train = np.array([0, 1, 2])
        validation = np.array([3, 4])
        clean = np.array(
            [[1, 2], [0, 2], [0, 1], [0, 1], [1, 2]], dtype=np.int64
        )
        self.assertTrue(
            audit_anchor_exclusion(train, validation, clean).passed
        )
        self_leak = clean.copy()
        self_leak[3, 0] = 3
        audit = audit_anchor_exclusion(train, validation, self_leak)
        self.assertFalse(audit.passed)
        self.assertEqual(audit.target_self_leaks, (3,))
        validation_leak = clean.copy()
        validation_leak[3, 0] = 4
        audit = audit_anchor_exclusion(train, validation, validation_leak)
        self.assertFalse(audit.passed)
        self.assertEqual(audit.validation_anchor_leaks, (3,))


class ResearchGateTests(unittest.TestCase):
    @staticmethod
    def _fold(real_score=0.55):
        baseline = _metric(0.50, pas=0.70, pdp=0.71, nmse=1.00)
        real = _metric(real_score, pas=0.72, pdp=0.73, nmse=0.90)
        zero = _metric(0.52, pas=0.70, pdp=0.71, nmse=1.00)
        shuffle = _metric(0.53, pas=0.71, pdp=0.71, nmse=0.98)
        return {
            "metrics": {
                "real": {"0.0": baseline, "1.0": real},
                "zero": {"0.0": baseline, "1.0": zero},
                "shuffle": {"0.0": baseline, "1.0": shuffle},
            },
            "baseline": baseline,
            "best": {
                "real": {"scale": 1.0, **real},
                "zero": {"scale": 1.0, **zero},
                "shuffle": {"scale": 1.0, **shuffle},
            },
            "support_novelty": 0.40,
        }

    def test_pilot_gate_passes_and_stops_below_gain(self):
        passed = pilot_gate(self._fold())
        self.assertTrue(passed["passed"])
        stopped = pilot_gate(self._fold(real_score=0.505))
        self.assertFalse(stopped["passed"])
        self.assertEqual(stopped["decision"], "STOP")

    def test_cross_validation_requires_every_full_gate(self):
        result = aggregate_cross_validation(
            [self._fold(0.55), self._fold(0.56), self._fold(0.54)]
        )
        self.assertTrue(result["promoted"])
        self.assertEqual(result["best_stable_scale"], 1.0)
        failed = aggregate_cross_validation(
            [self._fold(0.55), self._fold(0.49), self._fold(0.54)]
        )
        self.assertFalse(failed["promoted"])
        self.assertFalse(failed["checks"]["all_folds_positive"])

    def test_inference_promotion_guard_rejects_nonpromoted_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "promotion.json"
            path.write_text(
                json.dumps(
                    {
                        "kind": "ac_cgmrf_cross_validation",
                        "promoted": False,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not passed"):
                require_promotion_report(path)
            path.write_text(
                json.dumps(
                    {
                        "kind": "ac_cgmrf_cross_validation",
                        "promoted": True,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(require_promotion_report(path)["promoted"])


if __name__ == "__main__":
    unittest.main()
