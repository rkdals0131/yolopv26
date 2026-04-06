from __future__ import annotations

import random
import unittest

import torch

from model.data.transform import (
    TrainAugmentationConfig,
    apply_train_augmentations,
    clip_box_xyxy,
    clip_points,
    compute_letterbox_transform,
    inverse_transform_box_xyxy,
    inverse_transform_points,
    transform_box_xyxy,
    transform_from_meta,
    transform_points,
)


class PV26TransformRoundtripTests(unittest.TestCase):
    def test_compute_letterbox_transform_preserves_resized_plus_padding_contract(self) -> None:
        network_hw = (608, 800)
        for raw_hw in ((720, 1280), (1080, 1920), (1536, 512), (512, 1536)):
            with self.subTest(raw_hw=raw_hw):
                transform = compute_letterbox_transform(raw_hw, network_hw=network_hw)
                self.assertGreater(transform.scale, 0.0)
                self.assertEqual(transform.resized_hw[0] + transform.pad_top + transform.pad_bottom, network_hw[0])
                self.assertEqual(transform.resized_hw[1] + transform.pad_left + transform.pad_right, network_hw[1])
                self.assertLessEqual(transform.resized_hw[0], network_hw[0])
                self.assertLessEqual(transform.resized_hw[1], network_hw[1])

    def test_transform_from_meta_roundtrip_preserves_letterbox_fields(self) -> None:
        transform = compute_letterbox_transform((720, 1280), network_hw=(608, 800))
        restored = transform_from_meta(
            {
                "raw_hw": transform.raw_hw,
                "network_hw": transform.network_hw,
                "transform": transform.as_meta(),
            }
        )

        self.assertEqual(restored, transform)

    def test_box_roundtrip_restores_original_coordinates(self) -> None:
        transform = compute_letterbox_transform((720, 1280), network_hw=(608, 800))
        original = [120.0, 180.0, 640.0, 520.0]

        transformed = transform_box_xyxy(original, transform)
        restored = inverse_transform_box_xyxy(transformed, transform)

        self.assertIsNotNone(restored)
        for expected, actual in zip(original, restored or []):
            self.assertAlmostEqual(actual, expected, places=4)

    def test_points_roundtrip_restores_original_coordinates(self) -> None:
        transform = compute_letterbox_transform((1080, 1920), network_hw=(608, 800))
        original = [[240.0, 900.0], [720.0, 600.0], [1280.0, 120.0]]

        transformed = transform_points(original, transform)
        restored = inverse_transform_points(transformed, transform)

        for expected, actual in zip(original, restored):
            self.assertAlmostEqual(actual[0], expected[0], places=4)
            self.assertAlmostEqual(actual[1], expected[1], places=4)

    def test_clip_geometry_enforces_bounds_and_rejects_degenerate_boxes(self) -> None:
        self.assertIsNone(clip_box_xyxy([-10.0, -10.0, -2.0, -1.0], network_hw=(608, 800)))
        self.assertEqual(
            clip_box_xyxy([-10.0, 12.0, 900.0, 700.0], network_hw=(608, 800)),
            [0.0, 12.0, 799.0, 607.0],
        )
        self.assertEqual(
            clip_points([[-5.0, -1.0], [900.0, 700.0]], network_hw=(608, 800)),
            [[0.0, 0.0], [799.0, 607.0]],
        )

    def test_double_horizontal_flip_restores_boxes_geometry_and_image(self) -> None:
        image = torch.arange(3 * 8 * 10, dtype=torch.float32).reshape(3, 8, 10) / 255.0
        det_boxes = [[1.0, 1.0, 4.0, 5.0], [5.0, 2.0, 8.0, 7.0]]
        lanes = [{"points_xy": torch.tensor([[1.0, 6.0], [3.0, 4.0]], dtype=torch.float32), "color": 0, "lane_type": 1}]
        stop_lines = [{"points_xy": torch.tensor([[1.0, 5.0], [5.0, 5.0]], dtype=torch.float32)}]
        crosswalks = [
            {"points_xy": torch.tensor([[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0]], dtype=torch.float32)}
        ]
        config = TrainAugmentationConfig(
            horizontal_flip_prob=1.0,
            brightness_delta=0.0,
            contrast_range=(1.0, 1.0),
            gamma_range=(1.0, 1.0),
        )

        flipped = apply_train_augmentations(
            image,
            det_boxes=det_boxes,
            lanes=lanes,
            stop_lines=stop_lines,
            crosswalks=crosswalks,
            network_hw=(8, 10),
            config=config,
            rng=random.Random(1),
        )
        restored = apply_train_augmentations(
            flipped[0],
            det_boxes=flipped[1],
            lanes=flipped[2],
            stop_lines=flipped[3],
            crosswalks=flipped[4],
            network_hw=(8, 10),
            config=config,
            rng=random.Random(2),
        )

        self.assertTrue(torch.allclose(restored[0], image))
        self.assertEqual(restored[1], det_boxes)
        self.assertTrue(torch.equal(restored[2][0]["points_xy"], lanes[0]["points_xy"]))
        self.assertTrue(torch.equal(restored[3][0]["points_xy"], stop_lines[0]["points_xy"]))
        self.assertTrue(torch.equal(restored[4][0]["points_xy"], crosswalks[0]["points_xy"]))

    def test_photometric_jitter_stays_bounded_and_reports_sampled_parameters(self) -> None:
        image = torch.full((3, 8, 10), 0.5, dtype=torch.float32)
        config = TrainAugmentationConfig(
            horizontal_flip_prob=0.0,
            brightness_delta=0.2,
            contrast_range=(0.8, 1.2),
            gamma_range=(0.9, 1.1),
        )

        augmented = apply_train_augmentations(
            image,
            det_boxes=[],
            lanes=[],
            stop_lines=[],
            crosswalks=[],
            network_hw=(8, 10),
            config=config,
            rng=random.Random(7),
        )

        self.assertFalse(bool(augmented[5]["horizontal_flip"]))
        self.assertGreaterEqual(float(augmented[0].min().item()), 0.0)
        self.assertLessEqual(float(augmented[0].max().item()), 1.0)
        self.assertGreaterEqual(float(augmented[5]["brightness"]), 0.8)
        self.assertLessEqual(float(augmented[5]["brightness"]), 1.2)
        self.assertGreaterEqual(float(augmented[5]["contrast"]), 0.8)
        self.assertLessEqual(float(augmented[5]["contrast"]), 1.2)
        self.assertGreaterEqual(float(augmented[5]["gamma"]), 0.9)
        self.assertLessEqual(float(augmented[5]["gamma"]), 1.1)


if __name__ == "__main__":
    unittest.main()
