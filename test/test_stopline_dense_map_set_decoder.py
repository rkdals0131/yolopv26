from __future__ import annotations

import argparse
import unittest

import numpy as np
import torch

from tools.probe_pv26_stopline_dense_map_set_decoder import (
    _apply_grouped_candidate_lines,
    _apply_union_selector,
    _assign_candidate_labels,
    _baseline_candidate_rows,
    _best_assignment,
    _canonical_segment,
    _decoder_candidate_feature,
    _haf_candidate_feature,
    _haf_candidate_rows,
    _set_decoder_loss,
    _slot_examples_for_sample,
    _slot_feature,
    _slot_refiner_loss,
    _union_candidate_rows,
)


def _identity_meta() -> dict[str, object]:
    return {
        "raw_hw": (101, 101),
        "network_hw": (101, 101),
        "transform": {
            "scale": 1.0,
            "pad_left": 0,
            "pad_top": 0,
            "pad_right": 0,
            "pad_bottom": 0,
            "resized_hw": (101, 101),
        },
    }


class StoplineDenseMapSetDecoderTests(unittest.TestCase):
    def test_canonical_segment_orders_endpoints(self) -> None:
        segment = np.asarray([[0.8, 0.4], [0.2, 0.3]], dtype=np.float32)

        canonical = _canonical_segment(segment)

        np.testing.assert_allclose(canonical[0], np.asarray([0.2, 0.3], dtype=np.float32))
        np.testing.assert_allclose(canonical[1], np.asarray([0.8, 0.4], dtype=np.float32))

    def test_best_assignment_minimizes_total_cost(self) -> None:
        cost = torch.tensor(
            [
                [5.0, 1.0],
                [1.0, 5.0],
                [2.0, 2.0],
            ],
            dtype=torch.float32,
        )

        assignment = _best_assignment(cost, 2)

        self.assertEqual(assignment, [(1, 0), (0, 1)])

    def test_set_decoder_loss_is_finite_with_empty_and_positive_targets(self) -> None:
        logits = torch.zeros((2, 2), dtype=torch.float32, requires_grad=True)
        points = torch.full((2, 2, 2, 2), 0.5, dtype=torch.float32, requires_grad=True)
        targets = [
            torch.empty((0, 2, 2), dtype=torch.float32),
            torch.tensor([[[0.2, 0.3], [0.8, 0.3]]], dtype=torch.float32),
        ]

        loss = _set_decoder_loss(logits, points, targets, pos_weight=2.0)

        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(points.grad)

    def test_metric_quality_objectness_penalizes_misaligned_segment_confidence(self) -> None:
        target = [torch.tensor([[[0.1, 0.2], [0.9, 0.2]]], dtype=torch.float32)]
        good_points = torch.tensor([[[[0.1, 0.2], [0.9, 0.2]]]], dtype=torch.float32)
        bad_points = torch.tensor([[[[0.1, 0.8], [0.9, 0.8]]]], dtype=torch.float32)
        confident_logits = torch.full((1, 1), 3.0, dtype=torch.float32)

        good_loss = _set_decoder_loss(
            confident_logits,
            good_points,
            target,
            pos_weight=1.0,
            metric_quality_objectness=True,
            metric_quality_tau=0.05,
        )
        bad_loss = _set_decoder_loss(
            confident_logits,
            bad_points,
            target,
            pos_weight=1.0,
            metric_quality_objectness=True,
            metric_quality_tau=0.05,
        )

        self.assertGreater(float(bad_loss), float(good_loss))

    def test_decoder_candidate_feature_appends_candidate_geometry(self) -> None:
        sample_features = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        segment = np.asarray([[0.1, 0.2], [0.9, 0.2]], dtype=np.float32)

        feature = _decoder_candidate_feature(sample_features, segment=segment, probability=0.75)

        self.assertEqual(feature.shape, (15,))
        self.assertTrue(np.isfinite(feature).all())
        self.assertAlmostEqual(float(feature[3]), 0.75)

    def test_assign_candidate_labels_is_one_to_one(self) -> None:
        gt_lines = [{"points_xy": [[10.0, 20.0], [90.0, 20.0]]}]
        candidates = [
            {"line": {"points_xy": [[10.0, 20.0], [90.0, 20.0]]}},
            {"line": {"points_xy": [[12.0, 20.0], [92.0, 20.0]]}},
            {"line": {"points_xy": [[10.0, 80.0], [90.0, 80.0]]}},
        ]

        labels = _assign_candidate_labels(candidates, gt_lines)

        self.assertEqual(labels, [1, 0, 0])

    def test_haf_candidate_feature_appends_vote_quality(self) -> None:
        sample_features = np.asarray([1.0, 2.0], dtype=np.float32)
        segment = np.asarray([[0.1, 0.2], [0.9, 0.2]], dtype=np.float32)
        line = {
            "score": 0.7,
            "center_score": 0.8,
            "orientation_score": 0.9,
            "length": 25.0,
            "haf_vote_count": 5,
            "haf_endpoint_covariance": 4.0,
        }

        feature = _haf_candidate_feature(sample_features, segment=segment, line=line)

        self.assertEqual(feature.shape, (21,))
        self.assertTrue(np.isfinite(feature).all())
        self.assertAlmostEqual(float(feature[-7]), 0.8)
        self.assertAlmostEqual(float(feature[-3]), np.log1p(5.0))
        self.assertAlmostEqual(float(feature[-1]), 0.2)

    def test_haf_candidate_rows_convert_raw_lines_to_features(self) -> None:
        examples = [
            {
                "sample_index": 3,
                "features": np.asarray([0.5, 0.25], dtype=np.float32),
                "meta": _identity_meta(),
                "haf_candidates": [
                    {
                        "points_xy": [[10.0, 20.0], [90.0, 20.0]],
                        "score": 0.75,
                        "haf_vote_count": 6,
                        "haf_endpoint_covariance": 3.0,
                    }
                ],
            }
        ]

        grouped = _haf_candidate_rows(examples=examples)

        self.assertEqual(len(grouped), 1)
        self.assertEqual(len(grouped[0]), 1)
        self.assertEqual(grouped[0][0]["sample_index"], 3)
        self.assertEqual(grouped[0][0]["query_index"], 0)
        self.assertEqual(grouped[0][0]["line"]["proposal_source"], "haf_consensus_candidate")
        self.assertTrue(np.isfinite(grouped[0][0]["features"]).all())

    def test_union_candidate_rows_add_source_flags_for_baseline_and_decoder(self) -> None:
        examples = [
            {
                "sample_index": 3,
                "features": np.asarray([0.5, 0.25], dtype=np.float32),
                "meta": _identity_meta(),
                "baseline_lines": [
                    {"points_xy": [[10.0, 20.0], [90.0, 20.0]], "score": 0.8}
                ],
            }
        ]
        decoder_candidates = [
            [
                {
                    "sample_index": 3,
                    "query_index": 0,
                    "probability": 0.7,
                    "segment": np.asarray([[0.1, 0.4], [0.9, 0.4]], dtype=np.float32),
                    "features": np.zeros(14, dtype=np.float32),
                    "line": {
                        "points_xy": [[10.0, 40.0], [90.0, 40.0]],
                        "score": 0.7,
                        "proposal_source": "dense_map_set_decoder",
                    },
                }
            ]
        ]

        baseline_grouped = _baseline_candidate_rows(examples=examples)
        union_grouped = _union_candidate_rows(examples=examples, decoder_candidates=decoder_candidates)

        self.assertEqual(len(baseline_grouped[0]), 1)
        self.assertEqual(len(union_grouped[0]), 2)
        self.assertEqual(union_grouped[0][0]["line"]["proposal_source"], "retained_projection_comp")
        self.assertEqual(union_grouped[0][1]["line"]["proposal_source"], "dense_map_set_decoder")
        self.assertEqual(union_grouped[0][0]["features"].shape, union_grouped[0][1]["features"].shape)
        np.testing.assert_allclose(union_grouped[0][0]["features"][-5:-2], np.asarray([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(union_grouped[0][1]["features"][-5:-2], np.asarray([0.0, 1.0, 0.0]))
        self.assertTrue(np.isfinite(union_grouped[0][0]["features"]).all())

    def test_apply_union_selector_preserves_detector_prediction_contract(self) -> None:
        class ConstantVerifier(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("feature_mean", torch.zeros(2), persistent=True)
                self.register_buffer("feature_std", torch.ones(2), persistent=True)

            def forward(self, features: torch.Tensor) -> torch.Tensor:
                return torch.full((features.shape[0],), 10.0, dtype=features.dtype, device=features.device)

        baseline_predictions = [
            {
                "detections": [{"class_id": 2, "score": 0.1}],
                "stop_lines": [{"points_xy": [[0.0, 0.0], [10.0, 0.0]], "score": 0.4}],
            }
        ]
        examples = [{"gt_stop_lines": []}]
        grouped = [
            [
                {
                    "query_index": 0,
                    "probability": 0.8,
                    "features": np.asarray([0.3, 0.7], dtype=np.float32),
                    "line": {
                        "points_xy": [[10.0, 10.0], [20.0, 10.0]],
                        "score": 0.8,
                        "proposal_source": "dense_map_set_decoder",
                    },
                }
            ]
        ]
        args = argparse.Namespace(candidate_verifier_threshold=0.5, max_output_segments=2)

        predictions, rows = _apply_union_selector(
            examples=examples,
            baseline_predictions=baseline_predictions,
            grouped_candidates=grouped,
            verifier=ConstantVerifier(),
            args=args,
            device="cpu",
        )

        self.assertEqual(predictions[0]["detections"], baseline_predictions[0]["detections"])
        self.assertEqual(len(predictions[0]["stop_lines"]), 1)
        self.assertEqual(predictions[0]["stop_lines"][0]["proposal_source"], "dense_map_set_decoder")
        self.assertEqual(rows[0]["selected"], 1)

    def test_apply_grouped_candidate_lines_preserves_baseline_and_caps_candidates(self) -> None:
        baseline_predictions = [
            {"stop_lines": [{"points_xy": [[0.0, 0.0], [10.0, 0.0]], "score": 0.4}]}
        ]
        examples = [{"targets": np.zeros((1, 2, 2), dtype=np.float32)}]
        grouped = [
            [
                {
                    "query_index": 0,
                    "probability": 0.3,
                    "line": {"points_xy": [[10.0, 10.0], [20.0, 10.0]], "score": 0.3},
                },
                {
                    "query_index": 1,
                    "probability": 0.9,
                    "line": {"points_xy": [[10.0, 20.0], [20.0, 20.0]], "score": 0.9},
                },
            ]
        ]

        candidate_only, baseline_plus, rows = _apply_grouped_candidate_lines(
            examples=examples,
            baseline_predictions=baseline_predictions,
            grouped_candidates=grouped,
            args=argparse.Namespace(max_output_segments=1),
        )

        self.assertEqual(len(candidate_only[0]["stop_lines"]), 1)
        self.assertEqual(candidate_only[0]["stop_lines"][0]["score"], 0.9)
        self.assertEqual(len(baseline_plus[0]["stop_lines"]), 1)
        self.assertEqual(baseline_plus[0]["stop_lines"][0]["score"], 0.9)
        self.assertEqual([row["selected"] for row in rows], [0, 1])

    def test_slot_feature_appends_kind_flags(self) -> None:
        feature = _slot_feature(
            np.asarray([1.0, 2.0], dtype=np.float32),
            anchor=np.asarray([[0.1, 0.2], [0.9, 0.2]], dtype=np.float32),
            slot_kind="fallback",
            slot_rank=1,
        )

        self.assertEqual(feature.shape, (17,))
        self.assertAlmostEqual(float(feature[-3]), 0.0)
        self.assertAlmostEqual(float(feature[-2]), 1.0)
        self.assertAlmostEqual(float(feature[-1]), 0.5)

    def test_slot_examples_assign_baseline_and_fallback_targets(self) -> None:
        sample_features = np.asarray([0.5, 0.25], dtype=np.float32)
        baseline_lines = [{"points_xy": [[10.0, 20.0], [90.0, 20.0]]}]
        gt_lines = [
            {"points_xy": [[12.0, 20.0], [92.0, 20.0]]},
            {"points_xy": [[20.0, 80.0], [100.0, 80.0]]},
        ]
        gt_segments = np.asarray(
            [
                [[0.1, 0.2], [0.9, 0.2]],
                [[0.2, 0.8], [1.0, 0.8]],
            ],
            dtype=np.float32,
        )
        baseline_anchors = [np.asarray([[0.1, 0.2], [0.9, 0.2]], dtype=np.float32)]
        fallback_anchors = [np.asarray([[0.2, 0.7], [1.0, 0.7]], dtype=np.float32)]

        slots = _slot_examples_for_sample(
            sample_features=sample_features,
            baseline_lines=baseline_lines,
            gt_lines=gt_lines,
            gt_segments=gt_segments,
            baseline_anchors=baseline_anchors,
            fallback_anchors=fallback_anchors,
            loose_positive_distance=40.0,
            sample_meta=_identity_meta(),
        )

        self.assertEqual([slot["slot_kind"] for slot in slots], ["baseline", "fallback"])
        self.assertEqual([slot["positive"] for slot in slots], [1, 1])
        np.testing.assert_allclose(slots[1]["target"], gt_segments[1])

    def test_slot_examples_do_not_assign_far_fallback_target(self) -> None:
        slots = _slot_examples_for_sample(
            sample_features=np.asarray([0.5, 0.25], dtype=np.float32),
            baseline_lines=[],
            gt_lines=[{"points_xy": [[20.0, 80.0], [100.0, 80.0]]}],
            gt_segments=np.asarray([[[0.2, 0.8], [1.0, 0.8]]], dtype=np.float32),
            baseline_anchors=[],
            fallback_anchors=[np.asarray([[0.1, 0.2], [0.9, 0.2]], dtype=np.float32)],
            loose_positive_distance=20.0,
            sample_meta=_identity_meta(),
        )

        self.assertEqual(len(slots), 1)
        self.assertEqual(slots[0]["positive"], 0)
        self.assertEqual(slots[0]["assigned_gt_index"], -1)

    def test_slot_refiner_loss_is_finite_and_backpropagates(self) -> None:
        logits = torch.zeros((2,), dtype=torch.float32, requires_grad=True)
        points = torch.full((2, 2, 2), 0.5, dtype=torch.float32, requires_grad=True)
        targets = torch.tensor(
            [
                [[0.1, 0.2], [0.9, 0.2]],
                [[0.5, 0.5], [0.6, 0.5]],
            ],
            dtype=torch.float32,
        )
        anchors = torch.tensor(
            [
                [[0.1, 0.2], [0.9, 0.2]],
                [[0.5, 0.5], [0.6, 0.5]],
            ],
            dtype=torch.float32,
        )
        positive = torch.tensor([1.0, 0.0], dtype=torch.float32)

        loss = _slot_refiner_loss(logits, points, targets, anchors, positive, pos_weight=2.0)

        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(points.grad)


if __name__ == "__main__":
    unittest.main()
