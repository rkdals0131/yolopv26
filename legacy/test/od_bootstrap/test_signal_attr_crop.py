from __future__ import annotations

import math
import unittest

from PIL import Image

from tools.od_bootstrap.signal_attr import (
    DEFAULT_SIGNAL_ATTR_CROP_CONFIG,
    SIGNAL_ATTR_CROP_REASONS,
    SignalAttrCropConfig,
    count_crop_reasons,
    crop_signal_attr_roi,
    signal_attr_crop_config_to_dict,
    signal_attr_crop_window,
)


class SignalAttrCropTests(unittest.TestCase):
    def test_signal_attr_crop_rejects_empty_or_nonfinite_roi(self) -> None:
        invalid_boxes = [
            [10.0, 10.0, 10.0, 20.0],
            [0.0, 0.0, math.nan, 10.0],
            [0.0, 0.0, math.inf, 10.0],
            [0.0, 0.0, 3.0, 20.0],
            [100.0, 100.0, 120.0, 120.0],
        ]

        for bbox in invalid_boxes:
            with self.subTest(bbox=bbox):
                result = signal_attr_crop_window(bbox, (64, 48))

                self.assertFalse(result.valid)
                self.assertIsNone(result.crop_box)
                self.assertIsNone(result.clipped_box)
                self.assertEqual(result.reason, "signal_attr_teacher_invalid_roi")

    def test_signal_attr_crop_clips_pads_and_resizes_to_config_shape(self) -> None:
        image = Image.new("RGB", (40, 30), "black")
        config = SignalAttrCropConfig(input_size=16, padding_ratio=0.25, min_crop_side_px=4)

        result = crop_signal_attr_roi(image, [-2.0, 4.0, 12.0, 20.0], config=config)

        self.assertTrue(result.valid)
        self.assertEqual(result.reason, "valid")
        self.assertEqual(result.clipped_box, (0.0, 4.0, 12.0, 20.0))
        self.assertEqual(result.crop_box, (0, 0, 15, 24))
        self.assertIsNotNone(result.crop_image)
        self.assertEqual(result.crop_image.size, (16, 16))
        self.assertEqual(result.crop_image.mode, "RGB")

    def test_signal_attr_crop_reason_counts_and_config_shape_are_closed(self) -> None:
        valid = signal_attr_crop_window([1.0, 2.0, 12.0, 18.0], (64, 48))
        invalid = signal_attr_crop_window([1.0, 2.0, 1.0, 18.0], (64, 48))

        self.assertEqual(
            count_crop_reasons([valid, invalid, invalid]),
            {"valid": 1, "signal_attr_teacher_invalid_roi": 2},
        )
        self.assertEqual(tuple(count_crop_reasons([valid]).keys()), SIGNAL_ATTR_CROP_REASONS)
        self.assertEqual(
            signal_attr_crop_config_to_dict(DEFAULT_SIGNAL_ATTR_CROP_CONFIG),
            {
                "input_size": 128,
                "padding_ratio": 0.15,
                "min_crop_side_px": 4,
                "clip_to_image": True,
                "color_space": "rgb",
                "normalization": "imagenet",
                "interpolation": "bilinear",
            },
        )
