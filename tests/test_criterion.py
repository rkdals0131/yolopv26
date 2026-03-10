import unittest
from unittest import mock

import torch

from pv26.loss.criterion import PV26Criterion
from pv26.loss.det_ultralytics_e2e import UltralyticsE2EDetLossAdapter
from pv26.model.outputs import PV26MultiHeadOutput
from pv26.dataset.loading.sample_types import Pv26Sample
from pv26.training.prepared_batch import PV26PreparedBatch


class TestPV26CriterionMasking(unittest.TestCase):
    def setUp(self):
        self.criterion = PV26Criterion(num_det_classes=2)

    def _base_batch(self):
        return {
            "det_yolo": [torch.zeros((0, 5), dtype=torch.float32)],
            "da_mask": torch.full((1, 2, 2), 255, dtype=torch.uint8),
            "rm_mask": torch.full((1, 3, 2, 2), 255, dtype=torch.uint8),
            "rm_lane_subclass_mask": torch.full((1, 2, 2), 255, dtype=torch.uint8),
            "has_det": torch.tensor([0], dtype=torch.long),
            "has_da": torch.tensor([0], dtype=torch.long),
            "has_rm_lane_marker": torch.tensor([0], dtype=torch.long),
            "has_rm_road_marker_non_lane": torch.tensor([0], dtype=torch.long),
            "has_rm_stop_line": torch.tensor([0], dtype=torch.long),
            "has_rm_lane_subclass": torch.tensor([0], dtype=torch.long),
            "det_label_scope": ["none"],
        }

    def _preds(self):
        return PV26MultiHeadOutput(
            det=torch.zeros(1, 7, 1, 1),
            da=torch.zeros(1, 1, 2, 2),
            rm=torch.zeros(1, 3, 2, 2),
            rm_lane_subclass=torch.zeros(1, 5, 2, 2),
        )

    def test_da_loss_ignores_255_pixels(self):
        preds_a = self._preds()
        preds_b = self._preds()

        # Change only ignored pixel logit.
        preds_b.da[0, 0, 0, 0] = 100.0

        batch = self._base_batch()
        batch["has_da"] = torch.tensor([1], dtype=torch.long)
        batch["da_mask"] = torch.tensor([[[255, 1], [0, 1]]], dtype=torch.uint8)

        da_a = self.criterion(preds_a, batch)["da"]
        da_b = self.criterion(preds_b, batch)["da"]
        self.assertAlmostEqual(float(da_a), float(da_b), places=6)

    def test_da_loss_is_fully_masked_when_has_da_is_zero(self):
        preds = self._preds()
        preds.da.fill_(10.0)

        batch = self._base_batch()
        # Even with non-ignore labels, has_da=0 should hard-mask to zero.
        batch["da_mask"] = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.uint8)
        batch["has_da"] = torch.tensor([0], dtype=torch.long)

        da_loss = self.criterion(preds, batch)["da"]
        self.assertEqual(float(da_loss), 0.0)

    def test_rm_stop_line_channel_is_zeroed_when_has_rm_stop_line_is_zero(self):
        preds_a = self._preds()
        preds_b = self._preds()

        # Perturb only stop_line channel logits.
        preds_b.rm[:, 2, :, :] = 20.0

        batch = self._base_batch()
        batch["has_rm_lane_marker"] = torch.tensor([1], dtype=torch.long)
        batch["has_rm_road_marker_non_lane"] = torch.tensor([1], dtype=torch.long)
        batch["has_rm_stop_line"] = torch.tensor([0], dtype=torch.long)
        batch["rm_mask"] = torch.zeros((1, 3, 2, 2), dtype=torch.uint8)

        rm_a = self.criterion(preds_a, batch)["rm"]
        rm_b = self.criterion(preds_b, batch)["rm"]
        self.assertAlmostEqual(float(rm_a), float(rm_b), places=6)

    def test_od_loss_is_zero_when_det_label_scope_is_none(self):
        preds = self._preds()
        preds.det[:, 4, :, :] = 3.0  # non-zero objectness logits

        batch = self._base_batch()
        batch["has_det"] = torch.tensor([1], dtype=torch.long)
        batch["det_label_scope"] = ["none"]

        od_loss = self.criterion(preds, batch)["od"]
        self.assertEqual(float(od_loss), 0.0)

    def test_od_empty_gt_full_is_positive(self):
        preds = self._preds()
        preds.det[:, 4, :, :] = 3.0  # non-zero objectness logits

        batch = self._base_batch()
        batch["has_det"] = torch.tensor([1], dtype=torch.long)
        batch["det_label_scope"] = ["full"]
        od_full = self.criterion(preds, batch)["od"]

        self.assertGreater(float(od_full), 0.0)

    def test_rm_lane_subclass_is_masked_when_flag_is_zero(self):
        preds_a = self._preds()
        preds_b = self._preds()
        preds_b.rm_lane_subclass[:, 1, :, :] = 10.0

        batch = self._base_batch()
        batch["has_rm_lane_subclass"] = torch.tensor([0], dtype=torch.long)
        batch["rm_lane_subclass_mask"] = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.uint8)

        loss_a = self.criterion(preds_a, batch)["rm_lane_subclass"]
        loss_b = self.criterion(preds_b, batch)["rm_lane_subclass"]
        self.assertAlmostEqual(float(loss_a), float(loss_b), places=6)

    def test_rm_lane_subclass_ignores_pixels_outside_gt_lane_marker(self):
        preds_a = self._preds()
        preds_b = self._preds()
        preds_b.rm_lane_subclass[:, 1:, :, 1] = 25.0

        batch = self._base_batch()
        batch["has_rm_lane_marker"] = torch.tensor([1], dtype=torch.long)
        batch["has_rm_lane_subclass"] = torch.tensor([1], dtype=torch.long)
        batch["rm_mask"] = torch.tensor(
            [[[[1, 0], [1, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]],
            dtype=torch.uint8,
        )
        batch["rm_lane_subclass_mask"] = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.uint8)

        loss_a = self.criterion(preds_a, batch)["rm_lane_subclass"]
        loss_b = self.criterion(preds_b, batch)["rm_lane_subclass"]
        self.assertAlmostEqual(float(loss_a), float(loss_b), places=6)

    def test_rm_lane_subclass_sparse_matches_dense_masked_semantics(self):
        preds = PV26MultiHeadOutput(
            det=torch.zeros(2, 7, 1, 1),
            da=torch.zeros(2, 1, 2, 3),
            rm=torch.zeros(2, 3, 2, 3),
            rm_lane_subclass=torch.tensor(
                [
                    [
                        [[2.0, 0.1, -1.0], [0.3, 0.2, -0.1]],
                        [[0.1, 1.8, 0.2], [1.2, 0.0, 0.1]],
                        [[-0.4, 0.3, 2.1], [0.1, 1.5, 0.2]],
                        [[-1.0, -0.7, 0.4], [0.3, -0.2, 2.4]],
                        [[-0.2, -0.5, 0.0], [0.8, 0.7, 0.6]],
                    ],
                    [
                        [[1.0, 0.5, 0.2], [0.3, 0.9, 0.1]],
                        [[0.1, 1.2, 0.4], [0.2, 0.1, 0.0]],
                        [[0.2, 0.1, 1.8], [0.1, 0.2, 0.3]],
                        [[0.0, 0.2, 0.1], [1.7, 0.5, 0.4]],
                        [[0.3, 0.4, 0.5], [0.6, 0.7, 0.8]],
                    ],
                ],
                dtype=torch.float32,
            ),
        )
        batch = {
            "det_yolo": [torch.zeros((0, 5), dtype=torch.float32), torch.zeros((0, 5), dtype=torch.float32)],
            "da_mask": torch.full((2, 2, 3), 255, dtype=torch.uint8),
            "rm_mask": torch.full((2, 3, 2, 3), 255, dtype=torch.uint8),
            "rm_lane_subclass_mask": torch.tensor(
                [
                    [[0, 1, 255], [2, 4, 0]],
                    [[3, 0, 2], [255, 1, 0]],
                ],
                dtype=torch.uint8,
            ),
            "has_det": torch.tensor([0, 0], dtype=torch.long),
            "has_da": torch.tensor([0, 0], dtype=torch.long),
            "has_rm_lane_marker": torch.tensor([0, 0], dtype=torch.long),
            "has_rm_road_marker_non_lane": torch.tensor([0, 0], dtype=torch.long),
            "has_rm_stop_line": torch.tensor([0, 0], dtype=torch.long),
            "has_rm_lane_subclass": torch.tensor([1, 1], dtype=torch.long),
            "det_label_scope": ["none", "none"],
        }

        dense = PV26Criterion(num_det_classes=2, rm_lane_subclass_loss_impl="dense_masked")
        sparse = PV26Criterion(num_det_classes=2, rm_lane_subclass_loss_impl="sparse_pos")

        dense_loss = dense(preds, batch)["rm_lane_subclass"]
        sparse_loss = sparse(preds, batch)["rm_lane_subclass"]
        self.assertAlmostEqual(float(dense_loss), float(sparse_loss), places=6)

    def test_invalid_rm_lane_subclass_loss_impl_raises(self):
        with self.assertRaises(ValueError):
            PV26Criterion(num_det_classes=2, rm_lane_subclass_loss_impl="bogus")

    def test_prepared_batch_forward_loss_backward_smoke(self):
        criterion = PV26Criterion(num_det_classes=2)
        preds = PV26MultiHeadOutput(
            det=torch.randn(2, 7, 2, 2, requires_grad=True),
            da=torch.randn(2, 1, 4, 4, requires_grad=True),
            rm=torch.randn(2, 3, 4, 4, requires_grad=True),
            rm_lane_subclass=torch.randn(2, 5, 4, 4, requires_grad=True),
        )
        prepared = PV26PreparedBatch.from_mapping(
            {
                "det_yolo": [
                    torch.tensor([[1.0, 0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
                    torch.zeros((0, 5), dtype=torch.float32),
                ],
                "det_label_scope": ["full", "none"],
                "has_det": torch.tensor([1, 0], dtype=torch.long),
                "has_da": torch.tensor([1, 1], dtype=torch.long),
                "has_rm_lane_marker": torch.tensor([1, 1], dtype=torch.long),
                "has_rm_road_marker_non_lane": torch.tensor([1, 1], dtype=torch.long),
                "has_rm_stop_line": torch.tensor([1, 0], dtype=torch.long),
                "has_rm_lane_subclass": torch.tensor([1, 1], dtype=torch.long),
                "da_mask": torch.tensor(
                    [
                        [[0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]],
                        [[1, 1, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1]],
                    ],
                    dtype=torch.uint8,
                ),
                "rm_mask": torch.tensor(
                    [
                        [
                            [[1, 0, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
                            [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]],
                            [[0, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 0]],
                        ],
                        [
                            [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]],
                            [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
                            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                        ],
                    ],
                    dtype=torch.uint8,
                ),
                "rm_lane_subclass_mask": torch.tensor(
                    [
                        [[0, 1, 2, 0], [3, 0, 4, 0], [0, 1, 0, 2], [4, 0, 3, 0]],
                        [[1, 0, 0, 2], [0, 3, 0, 4], [2, 0, 1, 0], [0, 4, 0, 3]],
                    ],
                    dtype=torch.uint8,
                ),
            },
            device=torch.device("cpu"),
        )

        losses = criterion(preds, prepared)
        losses["total"].backward()

        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertIsNotNone(preds.det.grad)
        self.assertIsNotNone(preds.da.grad)
        self.assertIsNotNone(preds.rm.grad)
        self.assertIsNotNone(preds.rm_lane_subclass.grad)

    def test_raw_pv26sample_batches_are_rejected(self):
        criterion = PV26Criterion(num_det_classes=2)
        preds = self._preds()
        sample = Pv26Sample(
            sample_id="s0",
            split="train",
            image=torch.zeros((3, 4, 4), dtype=torch.uint8),
            det_yolo=torch.zeros((0, 5), dtype=torch.float32),
            da_mask=torch.zeros((4, 4), dtype=torch.uint8),
            rm_mask=torch.zeros((3, 4, 4), dtype=torch.uint8),
            rm_lane_subclass_mask=torch.zeros((4, 4), dtype=torch.uint8),
            has_det=0,
            has_da=0,
            has_rm_lane_marker=0,
            has_rm_road_marker_non_lane=0,
            has_rm_stop_line=0,
            has_rm_lane_subclass=0,
            det_label_scope="none",
            det_annotated_class_ids="",
        )

        with self.assertRaisesRegex(TypeError, "PV26PreparedBatch.from_samples"):
            criterion(preds, [sample])

    def test_seg_loss_block_matches_helper_numerics(self):
        criterion = PV26Criterion(num_det_classes=2)
        da_logits = torch.randn(2, 1, 4, 4, dtype=torch.float32)
        da_mask = torch.tensor(
            [
                [[0, 1, 255, 1], [1, 0, 1, 0], [0, 1, 0, 1], [255, 0, 1, 0]],
                [[1, 0, 1, 0], [0, 255, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1]],
            ],
            dtype=torch.uint8,
        )
        has_da = torch.tensor([1, 1], dtype=torch.long)
        rm_logits = torch.randn(2, 3, 4, 4, dtype=torch.float32)
        rm_mask = torch.tensor(
            [
                [
                    [[1, 0, 255, 1], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
                    [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]],
                    [[0, 0, 1, 0], [0, 1, 0, 1], [255, 0, 1, 0], [0, 1, 0, 0]],
                ],
                [
                    [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]],
                    [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
                    [[0, 0, 0, 0], [0, 0, 255, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                ],
            ],
            dtype=torch.uint8,
        )
        has_rm = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)

        da_ref = criterion._da_loss(da_logits=da_logits, da_mask=da_mask, has_da=has_da)
        rm_ref = criterion._rm_loss(rm_logits=rm_logits, rm_mask=rm_mask, has_rm=has_rm)
        da_block, rm_block = criterion._seg_loss_block(
            da_logits,
            da_mask,
            has_da,
            rm_logits,
            rm_mask,
            has_rm,
        )

        self.assertAlmostEqual(float(da_ref), float(da_block), places=6)
        self.assertAlmostEqual(float(rm_ref), float(rm_block), places=6)

    def test_prepared_batch_downsamples_seg_targets_for_stride2(self):
        prepared = PV26PreparedBatch.from_mapping(
            {
                "det_yolo": [torch.zeros((0, 5), dtype=torch.float32)],
                "det_label_scope": ["none"],
                "has_det": torch.tensor([0], dtype=torch.long),
                "has_da": torch.tensor([1], dtype=torch.long),
                "has_rm_lane_marker": torch.tensor([1], dtype=torch.long),
                "has_rm_road_marker_non_lane": torch.tensor([1], dtype=torch.long),
                "has_rm_stop_line": torch.tensor([1], dtype=torch.long),
                "has_rm_lane_subclass": torch.tensor([1], dtype=torch.long),
                "da_mask": torch.tensor(
                    [[[1, 0, 255, 255], [0, 0, 255, 255], [0, 1, 0, 0], [0, 0, 0, 0]]],
                    dtype=torch.uint8,
                ),
                "rm_mask": torch.tensor(
                    [
                        [
                            [[1, 0, 255, 255], [0, 0, 255, 255], [0, 1, 0, 0], [0, 0, 0, 0]],
                            [[0, 0, 0, 0], [0, 0, 0, 0], [255, 255, 255, 255], [255, 255, 255, 255]],
                            [[0, 0, 1, 0], [0, 0, 0, 0], [255, 255, 0, 0], [255, 255, 0, 0]],
                        ]
                    ],
                    dtype=torch.uint8,
                ),
                "rm_lane_subclass_mask": torch.tensor(
                    [[[1, 1, 2, 0], [0, 255, 2, 255], [0, 0, 255, 255], [255, 255, 255, 255]]],
                    dtype=torch.uint8,
                ),
            },
            device=torch.device("cpu"),
            seg_output_stride=2,
        )

        self.assertTrue(
            torch.equal(
                prepared.da_mask,
                torch.tensor([[[1, 255], [1, 0]]], dtype=torch.uint8),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared.rm_mask,
                torch.tensor(
                    [
                        [
                            [[1, 255], [1, 0]],
                            [[0, 0], [255, 255]],
                            [[0, 1], [255, 0]],
                        ]
                    ],
                    dtype=torch.uint8,
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared.rm_lane_subclass_mask,
                torch.tensor([[[1, 2], [0, 255]]], dtype=torch.uint8),
            )
        )

    def test_prepared_batch_can_preserve_fullres_masks_for_eval(self):
        sample = mock.Mock()
        sample.sample_id = "s0"
        sample.det_yolo = torch.zeros((0, 5), dtype=torch.float32)
        sample.det_label_scope = "none"
        sample.has_det = 0
        sample.has_da = 1
        sample.has_rm_lane_marker = 1
        sample.has_rm_road_marker_non_lane = 1
        sample.has_rm_stop_line = 1
        sample.has_rm_lane_subclass = 1
        sample.da_mask = torch.zeros((4, 4), dtype=torch.uint8)
        sample.rm_mask = torch.zeros((3, 4, 4), dtype=torch.uint8)
        sample.rm_lane_subclass_mask = torch.zeros((4, 4), dtype=torch.uint8)

        prepared = PV26PreparedBatch.from_samples(
            [sample],
            include_sample_id=True,
            include_fullres_masks=True,
            seg_output_stride=2,
        )

        self.assertEqual(prepared.sample_id, ("s0",))
        self.assertEqual(tuple(prepared.da_mask.shape), (1, 2, 2))
        self.assertEqual(tuple(prepared.rm_mask.shape), (1, 3, 2, 2))
        self.assertEqual(tuple(prepared.rm_lane_subclass_mask.shape), (1, 2, 2))
        self.assertEqual(tuple(prepared.da_mask_fullres.shape), (1, 4, 4))
        self.assertEqual(tuple(prepared.rm_mask_fullres.shape), (1, 3, 4, 4))
        self.assertEqual(tuple(prepared.rm_lane_subclass_mask_fullres.shape), (1, 4, 4))

    def test_enable_compile_seg_loss_uses_torch_compile(self):
        criterion = PV26Criterion(num_det_classes=2)
        compiled_sentinel = torch.nn.Identity()

        with mock.patch("pv26.loss.criterion.torch.compile", return_value=compiled_sentinel) as compile_mock:
            criterion.enable_compile_seg_loss(compile_mode="reduce-overhead", compile_fullgraph=True)

        compile_mock.assert_called_once_with(
            criterion._seg_loss_block,
            mode="reduce-overhead",
            fullgraph=True,
        )
        self.assertIs(criterion._seg_loss_block_impl, compiled_sentinel)
        self.assertTrue(criterion.seg_loss_compile_enabled)

        criterion.disable_compile_seg_loss()
        self.assertIs(criterion._seg_loss_block_impl, criterion._seg_loss_block)
        self.assertFalse(criterion.seg_loss_compile_enabled)

    def test_enable_compile_seg_loss_propagates_compile_failure(self):
        criterion = PV26Criterion(num_det_classes=2)

        with mock.patch("pv26.loss.criterion.torch.compile", side_effect=RuntimeError("compile boom")):
            with self.assertRaisesRegex(RuntimeError, "compile boom"):
                criterion.enable_compile_seg_loss(compile_mode="default", compile_fullgraph=False)

        criterion.disable_compile_seg_loss()
        self.assertIs(criterion._seg_loss_block_impl, criterion._seg_loss_block)
        self.assertFalse(criterion.seg_loss_compile_enabled)


class _FakeUltraDetLoss:
    def __init__(self):
        self.calls = []

    def __call__(self, preds_sel, det_batch):
        self.calls.append((preds_sel, det_batch))
        return torch.tensor(5.0), torch.zeros(3, dtype=torch.float32)


class _FakeUltraDetReduceLoss:
    def __call__(self, preds_sel, det_batch):
        total = preds_sel["one2many"]["boxes"].sum()
        total = total + preds_sel["one2many"]["scores"].sum()
        total = total + preds_sel["one2one"]["boxes"].sum()
        total = total + preds_sel["one2one"]["scores"].sum()
        total = total + sum(f.sum() for f in preds_sel["one2many"]["feats"])
        total = total + sum(f.sum() for f in preds_sel["one2one"]["feats"])
        total = total + det_batch["batch_idx"].sum() + det_batch["cls"].sum() + det_batch["bboxes"].sum()
        return total.to(dtype=torch.float32), torch.zeros(3, dtype=torch.float32)


def _make_fake_ultralytics_det_loss_adapter(fake_loss):
    adapter = UltralyticsE2EDetLossAdapter.__new__(UltralyticsE2EDetLossAdapter)
    adapter._ultra_det_loss = fake_loss
    return adapter


class TestPV26CriterionUltralyticsSubsetHandling(unittest.TestCase):
    def _make_det_out(self):
        return {
            "one2many": {
                "boxes": torch.arange(24, dtype=torch.float32).view(3, 2, 4),
                "scores": torch.arange(12, dtype=torch.float32).view(3, 2, 2),
                "feats": [torch.arange(96, dtype=torch.float32).view(3, 8, 2, 2)],
            },
            "one2one": {
                "boxes": (100 + torch.arange(24, dtype=torch.float32)).view(3, 2, 4),
                "scores": (200 + torch.arange(12, dtype=torch.float32)).view(3, 2, 2),
                "feats": [(300 + torch.arange(96, dtype=torch.float32)).view(3, 8, 2, 2)],
            },
        }

    def _old_slow_path_scalar(self, det_out, idx, det_tgt_batch_idx, det_tgt_cls, det_tgt_bboxes):
        device = det_out["one2many"]["boxes"].device
        idx = idx.to(device=device, dtype=torch.long)

        def _index_head(h):
            return {
                "boxes": h["boxes"].index_select(0, idx),
                "scores": h["scores"].index_select(0, idx),
                "feats": [f.index_select(0, idx) for f in h["feats"]],
            }

        preds_sel = {"one2many": _index_head(det_out["one2many"]), "one2one": _index_head(det_out["one2one"])}
        bsz = int(det_out["one2many"]["boxes"].shape[0])
        old_to_new = torch.full((bsz,), -1, device=device, dtype=torch.long)
        old_to_new[idx] = torch.arange(idx.shape[0], device=device, dtype=torch.long)

        src_old = det_tgt_batch_idx.to(device=device, dtype=torch.long)
        new_idx = old_to_new[src_old]
        m = new_idx.ge(0)
        det_batch = {
            "batch_idx": new_idx.masked_select(m).to(dtype=torch.float32),
            "cls": det_tgt_cls.to(device=device, dtype=torch.float32).masked_select(m),
            "bboxes": det_tgt_bboxes.to(device=device, dtype=torch.float32)[m],
        }
        loss_total, _ = _FakeUltraDetReduceLoss()(preds_sel, det_batch)
        return loss_total / float(int(idx.shape[0]))

    def test_ultralytics_od_excludes_none_samples_instead_of_raising(self):
        criterion = PV26Criterion(num_det_classes=2)
        fake_loss = _FakeUltraDetLoss()
        criterion.od_loss_impl = "ultralytics_e2e"
        criterion.det_loss_adapter = _make_fake_ultralytics_det_loss_adapter(fake_loss)

        det_out = {
            "one2many": {
                "boxes": torch.zeros(3, 2, 4),
                "scores": torch.zeros(3, 2, 2),
                "feats": [torch.zeros(3, 8, 4, 4)],
            },
            "one2one": {
                "boxes": torch.zeros(3, 2, 4),
                "scores": torch.zeros(3, 2, 2),
                "feats": [torch.zeros(3, 8, 4, 4)],
            },
        }

        od = criterion._od_loss_ultralytics(
            det_out=det_out,
            det_yolo=(),
            has_det=torch.tensor([1, 1, 1], dtype=torch.long),
            det_label_scope=("full", "none", "none"),
            det_scope_code=torch.tensor([0, 1, 1], dtype=torch.long),
            det_tgt_batch_idx=torch.tensor([0, 1], dtype=torch.long),
            det_tgt_cls=torch.tensor([1.0, 0.0], dtype=torch.float32),
            det_tgt_bboxes=torch.tensor(
                [[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.1, 0.1]],
                dtype=torch.float32,
            ),
        )

        self.assertEqual(float(od), 5.0)
        self.assertEqual(len(fake_loss.calls), 1)
        preds_sel, det_batch = fake_loss.calls[0]
        self.assertEqual(int(preds_sel["one2many"]["boxes"].shape[0]), 1)
        self.assertEqual(int(preds_sel["one2one"]["boxes"].shape[0]), 1)
        self.assertTrue(torch.equal(det_batch["batch_idx"], torch.tensor([0.0], dtype=torch.float32)))
        self.assertTrue(torch.equal(det_batch["cls"], torch.tensor([1.0], dtype=torch.float32)))
        self.assertTrue(
            torch.equal(
                det_batch["bboxes"],
                torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
            )
        )

    def test_ultralytics_od_subset_scope_now_raises(self):
        criterion = PV26Criterion(num_det_classes=2)
        criterion.od_loss_impl = "ultralytics_e2e"
        criterion.det_loss_adapter = _make_fake_ultralytics_det_loss_adapter(_FakeUltraDetLoss())

        det_out = self._make_det_out()
        with self.assertRaises(ValueError):
            criterion._od_loss_ultralytics(
                det_out=det_out,
                det_yolo=(),
                has_det=torch.tensor([1], dtype=torch.long),
                det_label_scope=("subset",),
                det_scope_code=torch.tensor([1], dtype=torch.long),
                det_tgt_batch_idx=torch.tensor([0], dtype=torch.long),
                det_tgt_cls=torch.tensor([1.0], dtype=torch.float32),
                det_tgt_bboxes=torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
            )

    def test_ultralytics_od_full_batch_fast_path_matches_old_slow_path_scalar(self):
        criterion = PV26Criterion(num_det_classes=2)
        criterion.od_loss_impl = "ultralytics_e2e"
        criterion.det_loss_adapter = _make_fake_ultralytics_det_loss_adapter(_FakeUltraDetReduceLoss())

        det_out = self._make_det_out()
        det_tgt_batch_idx = torch.tensor([0, 0, 1, 2], dtype=torch.long)
        det_tgt_cls = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
        det_tgt_bboxes = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.3, 0.4, 0.5],
                [0.3, 0.4, 0.5, 0.6],
                [0.4, 0.5, 0.6, 0.7],
            ],
            dtype=torch.float32,
        )

        od_fast = criterion._od_loss_ultralytics(
            det_out=det_out,
            det_yolo=(),
            has_det=torch.tensor([1, 1, 1], dtype=torch.long),
            det_label_scope=("full", "full", "full"),
            det_scope_code=torch.tensor([0, 0, 0], dtype=torch.long),
            det_tgt_batch_idx=det_tgt_batch_idx,
            det_tgt_cls=det_tgt_cls,
            det_tgt_bboxes=det_tgt_bboxes,
        )
        od_slow = self._old_slow_path_scalar(
            det_out=det_out,
            idx=torch.tensor([0, 1, 2], dtype=torch.long),
            det_tgt_batch_idx=det_tgt_batch_idx,
            det_tgt_cls=det_tgt_cls,
            det_tgt_bboxes=det_tgt_bboxes,
        )

        self.assertAlmostEqual(float(od_fast), float(od_slow), places=6)

    def test_ultralytics_od_full_batch_fast_path_reuses_prediction_tensors(self):
        criterion = PV26Criterion(num_det_classes=2)
        fake_loss = _FakeUltraDetLoss()
        criterion.od_loss_impl = "ultralytics_e2e"
        criterion.det_loss_adapter = _make_fake_ultralytics_det_loss_adapter(fake_loss)

        det_out = self._make_det_out()
        od = criterion._od_loss_ultralytics(
            det_out=det_out,
            det_yolo=(),
            has_det=torch.tensor([1, 1, 1], dtype=torch.long),
            det_label_scope=("full", "full", "full"),
            det_scope_code=torch.tensor([0, 0, 0], dtype=torch.long),
            det_tgt_batch_idx=torch.tensor([0, 1], dtype=torch.long),
            det_tgt_cls=torch.tensor([1.0, 0.0], dtype=torch.float32),
            det_tgt_bboxes=torch.tensor(
                [[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.1, 0.1]],
                dtype=torch.float32,
            ),
        )

        self.assertAlmostEqual(float(od), 5.0 / 3.0, places=6)
        preds_sel, det_batch = fake_loss.calls[0]
        self.assertIs(preds_sel, det_out)
        self.assertIs(preds_sel["one2many"]["boxes"], det_out["one2many"]["boxes"])
        self.assertIs(preds_sel["one2one"]["boxes"], det_out["one2one"]["boxes"])
        self.assertIs(preds_sel["one2many"]["feats"][0], det_out["one2many"]["feats"][0])
        self.assertTrue(torch.equal(det_batch["batch_idx"], torch.tensor([0.0, 1.0], dtype=torch.float32)))


if __name__ == "__main__":
    unittest.main()
