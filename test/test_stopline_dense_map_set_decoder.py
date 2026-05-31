from __future__ import annotations

import unittest

import numpy as np
import torch

from tools.probe_pv26_stopline_dense_map_set_decoder import (
    _assign_candidate_labels,
    _best_assignment,
    _canonical_segment,
    _decoder_candidate_feature,
    _set_decoder_loss,
    _slot_examples_for_sample,
    _slot_feature,
    _slot_refiner_loss,
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
