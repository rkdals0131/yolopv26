import argparse
import unittest

import torch
from torch import nn

from tools.train.train_pv26 import _build_optimizer


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

        opt = _build_optimizer(model=model, args=args, base_lr=1e-3)
        group_names = [str(g.get("name")) for g in opt.param_groups]

        self.assertIn("trunk_decay", group_names)
        self.assertIn("trunk_no_decay", group_names)
        self.assertIn("head_decay", group_names)
        self.assertIn("head_no_decay", group_names)


if __name__ == "__main__":
    unittest.main()
