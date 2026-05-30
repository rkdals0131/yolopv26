from __future__ import annotations

import unittest

import numpy as np

from tools.probe_pv26_stopline_trainset_patch_verifier import (
    SCORE_KEY,
    _attach_candidate_scores,
    _attach_record_candidate_scores,
)


class StopLineTrainsetPatchVerifierTests(unittest.TestCase):
    def test_attach_candidate_scores_writes_score_key(self) -> None:
        candidates = [{"proposal_rank": 1}, {"proposal_rank": 2}]
        _attach_candidate_scores(candidates, np.asarray([0.25, 0.75], dtype=np.float32))
        self.assertAlmostEqual(candidates[0][SCORE_KEY], 0.25, places=6)
        self.assertAlmostEqual(candidates[1][SCORE_KEY], 0.75, places=6)

    def test_attach_candidate_scores_rejects_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            _attach_candidate_scores([{"proposal_rank": 1}], np.asarray([0.1, 0.2], dtype=np.float32))

    def test_attach_record_candidate_scores_respects_top_k(self) -> None:
        row_a: dict[str, float] = {}
        row_b: dict[str, float] = {}
        records = [
            {
                "candidates": [{"proposal_rank": 1}, {"proposal_rank": 51}],
                "candidate_feature_rows": [row_a, row_b],
            }
        ]
        _attach_record_candidate_scores(records, np.asarray([0.4], dtype=np.float32), top_k=50)
        self.assertAlmostEqual(records[0]["candidates"][0][SCORE_KEY], 0.4, places=6)
        self.assertAlmostEqual(row_a[SCORE_KEY], 0.4, places=6)
        self.assertNotIn(SCORE_KEY, records[0]["candidates"][1])
        self.assertNotIn(SCORE_KEY, row_b)


if __name__ == "__main__":
    unittest.main()
