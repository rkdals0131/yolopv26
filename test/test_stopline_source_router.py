from __future__ import annotations

import unittest

import numpy as np
import torch

from model.data.transform import compute_letterbox_transform
from tools.probe_pv26_stopline_source_router import (
    ROUTER_MODES,
    _draw_stopline_raster,
    _source_raster_for_sample,
    _train_raster_router,
)


def _meta() -> dict[str, object]:
    transform = compute_letterbox_transform((608, 800), (608, 800))
    return {
        "raw_hw": (608, 800),
        "network_hw": (608, 800),
        "transform": transform.as_meta(),
    }


class StoplineSourceRouterTests(unittest.TestCase):
    def test_draw_stopline_raster_marks_horizontal_line(self) -> None:
        raster = _draw_stopline_raster(
            [{"points_xy": [[100.0, 300.0], [700.0, 300.0]]}],
            _meta(),
            size=(32, 48),
        )

        self.assertEqual(raster.shape, (32, 48))
        self.assertGreater(float(raster.sum()), 8.0)
        self.assertGreater(float(raster.max()), 0.9)

    def test_source_raster_for_sample_stacks_source_and_dense_channels(self) -> None:
        outputs = {
            "stop_line_mask_logits": torch.zeros((1, 1, 8, 12), dtype=torch.float32),
            "stop_line_center_logits": torch.ones((1, 1, 8, 12), dtype=torch.float32),
            "stop_line_selector_map_logits": torch.full((1, 1, 8, 12), -1.0, dtype=torch.float32),
        }
        image = torch.ones((1, 3, 64, 96), dtype=torch.float32)
        primary = {"stop_lines": [{"points_xy": [[100.0, 300.0], [700.0, 300.0]]}]}
        specialist = {"stop_lines": [{"points_xy": [[120.0, 320.0], [680.0, 320.0]]}]}

        raster = _source_raster_for_sample(
            primary,
            specialist,
            primary_outputs=outputs,
            specialist_outputs=outputs,
            image=image,
            sample_index=0,
            meta=_meta(),
            size=(16, 24),
        )

        self.assertEqual(raster.shape, (10, 16, 24))
        self.assertTrue(np.isfinite(raster).all())
        self.assertGreater(float(raster[1].sum()), 0.0)
        self.assertGreater(float(raster[2].sum()), 0.0)
        self.assertGreater(float(raster[6].mean()), float(raster[4].mean()))

    def test_train_raster_router_can_fit_tiny_contract(self) -> None:
        rasters: list[np.ndarray] = []
        labels: list[int] = []
        for index in range(len(ROUTER_MODES)):
            raster = np.zeros((3, 8, 8), dtype=np.float32)
            raster[:, index : index + 1, :] = 1.0
            rasters.append(raster)
            labels.append(index)

        model, mean, std, diagnostics = _train_raster_router(
            rasters,
            labels,
            base_channels=4,
            epochs=5,
            lr=1.0e-3,
            weight_decay=0.0,
            seed=123,
            device="cpu",
        )

        self.assertEqual(int(mean.shape[1]), 3)
        self.assertEqual(int(std.shape[1]), 3)
        self.assertEqual(diagnostics["feature_count"], len(ROUTER_MODES))
        self.assertEqual(next(model.parameters()).device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
