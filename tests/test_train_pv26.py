import argparse
import unittest

import torch
from torch import nn

from pv26.training.optimizer_factory import build_optimizer
from pv26.training.prepared_batch import PV26PreparedBatch
from pv26.training.runner import _apply_train_multiscale, _pick_train_multiscale_size
from pv26.training.train_config import SCRIPT_DEFAULTS, build_argparser


class _FakeDetContainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.det_model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
        )


class _FakeTrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.det_backend = _FakeDetContainer()
        self.det_model = self.det_backend.det_model
        self.seg_head = nn.Conv2d(8, 4, kernel_size=1)

    def forward(self, x):
        x = self.det_backend.det_model(x)
        return self.seg_head(x)


class TestTrainPv26OptimizerGrouping(unittest.TestCase):
    def test_build_optimizer_treats_det_backend_det_model_as_trunk(self):
        model = _FakeTrainModel()
        args = argparse.Namespace(
            det_pretrained=None,
            weight_decay=1e-4,
            optimizer="adamw",
            momentum=0.937,
        )

        opt = build_optimizer(model=model, args=args, base_lr=1e-3)
        group_names = [str(g.get("name")) for g in opt.param_groups]

        self.assertIn("trunk_decay", group_names)
        self.assertIn("trunk_no_decay", group_names)
        self.assertIn("head_decay", group_names)
        self.assertIn("head_no_decay", group_names)


class TestTrainPv26ScriptDefaults(unittest.TestCase):
    def test_argparser_uses_script_default_block(self):
        args = build_argparser().parse_args([])

        self.assertEqual(SCRIPT_DEFAULTS.seg_output_stride, 1)
        self.assertEqual(args.dataset_root, SCRIPT_DEFAULTS.dataset_root)
        self.assertEqual(args.epochs, SCRIPT_DEFAULTS.epochs)
        self.assertEqual(args.batch_size, SCRIPT_DEFAULTS.batch_size)
        self.assertEqual(args.seg_output_stride, SCRIPT_DEFAULTS.seg_output_stride)
        self.assertEqual(args.compile, SCRIPT_DEFAULTS.compile)
        self.assertEqual(args.compile_seg_loss, SCRIPT_DEFAULTS.compile_seg_loss)
        self.assertEqual(args.progress, SCRIPT_DEFAULTS.progress)
        self.assertEqual(args.tensorboard, SCRIPT_DEFAULTS.tensorboard)
        self.assertEqual(args.aug_scale_min, SCRIPT_DEFAULTS.aug_scale_min)
        self.assertEqual(args.aug_scale_max, SCRIPT_DEFAULTS.aug_scale_max)
        self.assertEqual(args.aug_translate, SCRIPT_DEFAULTS.aug_translate)
        self.assertEqual(args.aug_multiscale_min, SCRIPT_DEFAULTS.aug_multiscale_min)
        self.assertEqual(args.aug_multiscale_max, SCRIPT_DEFAULTS.aug_multiscale_max)

    def test_cli_arguments_override_script_default_block(self):
        args = build_argparser().parse_args(
            [
                "--epochs",
                "9",
                "--batch-size",
                "4",
                "--seg-output-stride",
                "1",
                "--compile",
                "--no-compile-seg-loss",
                "--no-progress",
                "--no-tensorboard",
                "--aug-scale-min",
                "0.8",
                "--aug-scale-max",
                "1.2",
                "--aug-translate",
                "0.15",
                "--aug-multiscale-min",
                "0.85",
                "--aug-multiscale-max",
                "1.15",
            ]
        )

        self.assertEqual(args.epochs, 9)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.seg_output_stride, 1)
        self.assertTrue(args.compile)
        self.assertFalse(args.compile_seg_loss)
        self.assertFalse(args.progress)
        self.assertFalse(args.tensorboard)
        self.assertAlmostEqual(args.aug_scale_min, 0.8)
        self.assertAlmostEqual(args.aug_scale_max, 1.2)
        self.assertAlmostEqual(args.aug_translate, 0.15)
        self.assertAlmostEqual(args.aug_multiscale_min, 0.85)
        self.assertAlmostEqual(args.aug_multiscale_max, 1.15)


class TestTrainPv26Multiscale(unittest.TestCase):
    def test_pick_train_multiscale_size_rounds_to_32(self):
        self.assertEqual(
            _pick_train_multiscale_size(base_h=544, base_w=960, scale_min=0.9, scale_max=0.9),
            (480, 864),
        )

    def test_apply_train_multiscale_resizes_images_and_masks(self):
        images = torch.zeros((1, 3, 544, 960), dtype=torch.float32)
        batch = PV26PreparedBatch(
            det_yolo=(torch.tensor([[0.0, 0.5, 0.5, 0.25, 0.25]], dtype=torch.float32),),
            det_label_scope=("full",),
            has_det=torch.tensor([1], dtype=torch.long),
            has_da=torch.tensor([1], dtype=torch.long),
            has_rm=torch.tensor([[1, 1, 0]], dtype=torch.long),
            has_rm_lane_subclass=torch.tensor([1], dtype=torch.long),
            da_mask=torch.zeros((1, 544, 960), dtype=torch.uint8),
            rm_mask=torch.zeros((1, 3, 544, 960), dtype=torch.uint8),
            rm_lane_subclass_mask=torch.zeros((1, 544, 960), dtype=torch.uint8),
        )

        resized_images, resized_batch = _apply_train_multiscale(
            images=images,
            target_batch=batch,
            seg_output_stride=1,
            scale_min=0.9,
            scale_max=0.9,
        )

        self.assertEqual(tuple(resized_images.shape), (1, 3, 480, 864))
        self.assertEqual(tuple(resized_batch.da_mask.shape), (1, 480, 864))
        self.assertEqual(tuple(resized_batch.rm_mask.shape), (1, 3, 480, 864))
        self.assertEqual(tuple(resized_batch.rm_lane_subclass_mask.shape), (1, 480, 864))
        self.assertTrue(torch.equal(resized_batch.det_yolo[0], batch.det_yolo[0]))

if __name__ == "__main__":
    unittest.main()
