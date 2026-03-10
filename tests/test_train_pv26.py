import argparse
import unittest

import torch
from torch import nn

from pv26.training.optimizer_factory import build_optimizer
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
            ]
        )

        self.assertEqual(args.epochs, 9)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.seg_output_stride, 1)
        self.assertTrue(args.compile)
        self.assertFalse(args.compile_seg_loss)
        self.assertFalse(args.progress)
        self.assertFalse(args.tensorboard)


if __name__ == "__main__":
    unittest.main()
