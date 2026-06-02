from __future__ import annotations

import json
import os
import random
import subprocess
import sys
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
    def test_network_hw_can_be_overridden_before_import(self) -> None:
        env = dict(os.environ)
        env["PV26_NETWORK_HW"] = "672x896"
        output = subprocess.check_output(
            [
                sys.executable,
                "-c",
                (
                    "import json;"
                    "from model.data.transform import NETWORK_HW;"
                    "from model.data.roadmark_v2_targets import ROADMARK_DENSE_OUTPUT_HW;"
                    "print(json.dumps({'network_hw': NETWORK_HW, 'dense_hw': ROADMARK_DENSE_OUTPUT_HW}))"
                ),
            ],
            cwd=os.getcwd(),
            env=env,
            text=True,
        )

        payload = json.loads(output)
        self.assertEqual(payload["network_hw"], [672, 896])
        self.assertEqual(payload["dense_hw"], [168, 224])

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

    def test_compute_letterbox_transform_rejects_nonpositive_dimensions(self) -> None:
        invalid_pairs = (
            ((0, 1280), (608, 800)),
            ((720, 0), (608, 800)),
            ((-1, 1280), (608, 800)),
            ((720, 1280), (0, 800)),
            ((720, 1280), (608, -1)),
        )
        for raw_hw, network_hw in invalid_pairs:
            with self.subTest(raw_hw=raw_hw, network_hw=network_hw):
                with self.assertRaisesRegex(ValueError, "letterbox dimensions must be positive"):
                    compute_letterbox_transform(raw_hw, network_hw=network_hw)

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

    def test_transform_from_meta_rejects_malformed_letterbox_handoff(self) -> None:
        transform = compute_letterbox_transform((720, 1280), network_hw=(608, 800))
        valid_meta = {
            "raw_hw": transform.raw_hw,
            "network_hw": transform.network_hw,
            "transform": transform.as_meta(),
        }
        cases = (
            ("zero_scale", {"scale": 0.0}, "letterbox transform scale must be positive and finite"),
            ("nan_scale", {"scale": float("nan")}, "letterbox transform scale must be positive and finite"),
            ("negative_pad", {"pad_top": -1}, "letterbox transform padding must be non-negative"),
            ("bad_canvas_sum", {"pad_bottom": 80}, "letterbox resized/padding must match network_hw"),
            ("bad_resized_scale", {"scale": 0.5}, "letterbox resized_hw must match raw_hw and scale"),
        )

        for name, overrides, message in cases:
            with self.subTest(name=name):
                meta = {
                    "raw_hw": valid_meta["raw_hw"],
                    "network_hw": valid_meta["network_hw"],
                    "transform": {**valid_meta["transform"], **overrides},
                }
                with self.assertRaisesRegex(ValueError, message):
                    transform_from_meta(meta)

    def test_transform_from_meta_rejects_noncanonical_letterbox_handoff(self) -> None:
        meta = {
            "raw_hw": (720, 1280),
            "network_hw": (608, 800),
            "transform": {
                "scale": 0.5,
                "pad_left": 80,
                "pad_top": 124,
                "pad_right": 80,
                "pad_bottom": 124,
                "resized_hw": (360, 640),
            },
        }

        with self.assertRaisesRegex(ValueError, "letterbox transform must match raw_hw/network_hw"):
            transform_from_meta(meta)

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

    def test_stopline_focus_crop_zooms_geometry_without_changing_network_shape(self) -> None:
        image = torch.arange(3 * 8 * 10, dtype=torch.float32).reshape(3, 8, 10) / 255.0
        lanes = [{"points_xy": torch.tensor([[3.0, 7.0], [5.0, 3.0]], dtype=torch.float32), "color": 0}]
        stop_lines = [{"points_xy": torch.tensor([[4.0, 4.0], [6.0, 4.0]], dtype=torch.float32)}]
        crosswalks = [
            {"points_xy": torch.tensor([[3.0, 3.0], [6.0, 3.0], [6.0, 6.0], [3.0, 6.0]], dtype=torch.float32)}
        ]
        config = TrainAugmentationConfig(
            horizontal_flip_prob=0.0,
            brightness_delta=0.0,
            contrast_range=(1.0, 1.0),
            gamma_range=(1.0, 1.0),
            stopline_focus_crop_prob=1.0,
            stopline_focus_crop_scale_range=(2.0, 2.0),
            stopline_focus_crop_jitter=0.0,
        )

        augmented = apply_train_augmentations(
            image,
            det_boxes=[],
            lanes=lanes,
            stop_lines=stop_lines,
            crosswalks=crosswalks,
            network_hw=(8, 10),
            config=config,
            rng=random.Random(3),
        )

        self.assertEqual(tuple(augmented[0].shape), (3, 8, 10))
        crop_meta = augmented[5]["stopline_focus_crop"]
        self.assertEqual(
            crop_meta,
            {
                "applied": True,
                "zoom": 2.0,
                "crop_left": 2,
                "crop_top": 2,
                "crop_right": 7,
                "crop_bottom": 6,
                "focus_center": [5.0, 4.0],
            },
        )
        self.assertTrue(torch.allclose(augmented[3][0]["points_xy"], torch.tensor([[4.0, 4.0], [8.0, 4.0]])))
        self.assertGreaterEqual(float(augmented[2][0]["points_xy"].min().item()), 0.0)
        self.assertLessEqual(float(augmented[2][0]["points_xy"][..., 0].max().item()), 9.0)
        self.assertLessEqual(float(augmented[4][0]["points_xy"][..., 1].max().item()), 7.0)

    def test_stopline_focus_crop_skips_detector_supervised_samples(self) -> None:
        image = torch.zeros((3, 8, 10), dtype=torch.float32)
        det_boxes = [[1.0, 1.0, 4.0, 5.0]]
        stop_lines = [{"points_xy": torch.tensor([[4.0, 4.0], [6.0, 4.0]], dtype=torch.float32)}]
        config = TrainAugmentationConfig(
            horizontal_flip_prob=0.0,
            brightness_delta=0.0,
            contrast_range=(1.0, 1.0),
            gamma_range=(1.0, 1.0),
            stopline_focus_crop_prob=1.0,
            stopline_focus_crop_scale_range=(2.0, 2.0),
            stopline_focus_crop_jitter=0.0,
        )

        augmented = apply_train_augmentations(
            image,
            det_boxes=det_boxes,
            lanes=[],
            stop_lines=stop_lines,
            crosswalks=[],
            network_hw=(8, 10),
            config=config,
            rng=random.Random(3),
        )

        self.assertEqual(augmented[1], det_boxes)
        self.assertIsNone(augmented[5]["stopline_focus_crop"])
        self.assertTrue(torch.equal(augmented[3][0]["points_xy"], stop_lines[0]["points_xy"]))

    def test_shared_affine_translates_all_geometry_and_keeps_shape(self) -> None:
        image = torch.zeros((3, 8, 10), dtype=torch.float32)
        det_boxes = [[1.0, 1.0, 4.0, 5.0]]
        lanes = [{"points_xy": torch.tensor([[1.0, 6.0], [3.0, 4.0]], dtype=torch.float32), "color": 0}]
        stop_lines = [{"points_xy": torch.tensor([[1.0, 5.0], [5.0, 5.0]], dtype=torch.float32)}]
        crosswalks = [
            {"points_xy": torch.tensor([[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0]], dtype=torch.float32)}
        ]
        config = TrainAugmentationConfig(
            horizontal_flip_prob=0.0,
            brightness_delta=0.0,
            contrast_range=(1.0, 1.0),
            gamma_range=(1.0, 1.0),
            affine_prob=1.0,
            affine_degrees=0.0,
            affine_translate_frac=0.10,
            affine_scale_range=(1.0, 1.0),
            affine_shear_degrees=0.0,
        )

        augmented = apply_train_augmentations(
            image,
            det_boxes=det_boxes,
            lanes=lanes,
            stop_lines=stop_lines,
            crosswalks=crosswalks,
            network_hw=(8, 10),
            config=config,
            rng=random.Random(5),
        )

        self.assertEqual(tuple(augmented[0].shape), (3, 8, 10))
        affine_meta = augmented[5]["shared_affine"]
        self.assertIsNotNone(affine_meta)
        tx, ty = [float(value) for value in affine_meta["translate"]]
        expected_lane = lanes[0]["points_xy"].clone()
        expected_lane[:, 0] = (expected_lane[:, 0] + tx).clamp(0.0, 9.0)
        expected_lane[:, 1] = (expected_lane[:, 1] + ty).clamp(0.0, 7.0)
        self.assertTrue(torch.allclose(augmented[2][0]["points_xy"], expected_lane, atol=1.0e-5))
        self.assertIsNotNone(augmented[1][0])
        self.assertGreaterEqual(float(augmented[3][0]["points_xy"].min().item()), 0.0)
        self.assertLessEqual(float(augmented[4][0]["points_xy"][..., 0].max().item()), 9.0)

    def test_synthetic_stopline_injection_adds_label_and_pixels_from_lanes(self) -> None:
        image = torch.zeros((3, 8, 10), dtype=torch.float32)
        lanes = [
            {"points_xy": torch.tensor([[2.0, 1.0], [2.0, 7.0]], dtype=torch.float32), "color": 0},
            {"points_xy": torch.tensor([[7.0, 1.0], [7.0, 7.0]], dtype=torch.float32), "color": 0},
        ]
        config = TrainAugmentationConfig(
            horizontal_flip_prob=0.0,
            brightness_delta=0.0,
            contrast_range=(1.0, 1.0),
            gamma_range=(1.0, 1.0),
            synthetic_stopline_prob=1.0,
            synthetic_stopline_thickness_px=2.0,
        )

        augmented = apply_train_augmentations(
            image,
            det_boxes=[],
            lanes=lanes,
            stop_lines=[],
            crosswalks=[],
            network_hw=(8, 10),
            config=config,
            rng=random.Random(9),
        )

        self.assertEqual(len(augmented[3]), 1)
        self.assertTrue(bool(augmented[3][0]["synthetic"]))
        points = augmented[3][0]["points_xy"]
        self.assertEqual(tuple(points.shape), (2, 2))
        self.assertGreater(float(points[:, 0].max().item() - points[:, 0].min().item()), 4.0)
        self.assertGreater(float(augmented[0].max().item()), 0.5)
        self.assertIsNotNone(augmented[5]["synthetic_stopline"])

    def test_stopline_copy_paste_adds_real_patch_label_and_pixels_from_donor(self) -> None:
        image = torch.zeros((3, 8, 10), dtype=torch.float32)
        donor_image = torch.zeros((3, 8, 10), dtype=torch.float32)
        donor_image[:, 3:5, 1:9] = 0.9
        lanes = [
            {"points_xy": torch.tensor([[2.0, 1.0], [2.0, 7.0]], dtype=torch.float32), "color": 0},
            {"points_xy": torch.tensor([[7.0, 1.0], [7.0, 7.0]], dtype=torch.float32), "color": 0},
        ]
        donor = {
            "image": donor_image,
            "stop_lines": [{"points_xy": torch.tensor([[1.0, 4.0], [8.0, 4.0]], dtype=torch.float32)}],
            "sample_id": "donor_stopline",
        }
        config = TrainAugmentationConfig(
            horizontal_flip_prob=0.0,
            brightness_delta=0.0,
            contrast_range=(1.0, 1.0),
            gamma_range=(1.0, 1.0),
            stopline_copy_paste_prob=1.0,
            stopline_copy_paste_margin_px=1.5,
            stopline_copy_paste_alpha=1.0,
        )

        augmented = apply_train_augmentations(
            image,
            det_boxes=[],
            lanes=lanes,
            stop_lines=[],
            crosswalks=[],
            network_hw=(8, 10),
            config=config,
            rng=random.Random(9),
            stopline_copy_paste_donor=donor,
        )

        self.assertEqual(len(augmented[3]), 1)
        self.assertTrue(bool(augmented[3][0]["copy_paste"]))
        self.assertGreater(float(augmented[0].max().item()), 0.5)
        self.assertIsNotNone(augmented[5]["stopline_copy_paste"])
        self.assertEqual(augmented[5]["stopline_copy_paste"]["donor_sample_id"], "donor_stopline")


if __name__ == "__main__":
    unittest.main()
