from __future__ import annotations

import unittest

from common.pv26_schema import TL_BITS
from tools.od_bootstrap.signal_attr import (
    collapse_aihub_traffic_light_attr,
    combo_name,
    extract_product_signal_attr_target,
)
from tools.od_bootstrap.source.aihub.traffic_worker import _tl_bits_from_annotation


class SignalAttrAIHubPolicyTests(unittest.TestCase):
    def test_product_target_keeps_left_distinct_and_accepts_pedestrian_colors(self) -> None:
        car = extract_product_signal_attr_target(
            {
                "type": "car",
                "attribute": [{"red": "on", "yellow": "off", "green": "off", "left_arrow": "on", "others_arrow": "on"}],
            },
            all_off_is_valid=False,
        )
        other_only = extract_product_signal_attr_target(
            {
                "type": "car",
                "attribute": [{"red": "off", "yellow": "off", "green": "on", "left_arrow": "off", "others_arrow": "on"}],
            },
            all_off_is_valid=False,
        )
        other_lamp_only = extract_product_signal_attr_target(
            {
                "type": "car",
                "attribute": [{"red": "off", "yellow": "off", "green": "off", "left_arrow": "off", "others_arrow": "on"}],
            },
            all_off_is_valid=False,
        )
        pedestrian = extract_product_signal_attr_target(
            {"type": "pedestrian", "attribute": [{"red": "on", "green": "off"}]},
            all_off_is_valid=False,
        )
        all_off = extract_product_signal_attr_target(
            {"type": "pedestrian", "attribute": [{"red": "off", "green": "off"}]},
            all_off_is_valid=False,
        )
        self.assertEqual((car.base_color, car.left_arrow, car.state_valid), ("red", 1, True))
        self.assertEqual((other_only.base_color, other_only.left_arrow, other_only.state_valid), ("green", 0, True))
        self.assertEqual((other_lamp_only.base_color, other_lamp_only.left_arrow, other_lamp_only.state_valid),
                         ("off", 0, True))
        self.assertEqual((pedestrian.base_color, pedestrian.state_valid), ("red", True))
        self.assertEqual((all_off.state_valid, all_off.reason), (False, "all_off_unverified"))

    def test_signal_attr_label_extractor_matches_traffic_worker_policy(self) -> None:
        cases = [
            (
                {"type": "car", "attribute": {"red": "on", "yellow": "off", "green": "off", "left_arrow": "on"}},
                {"red": 1, "yellow": 0, "green": 0, "arrow": 1},
                1,
                "valid",
            ),
            (
                {"type": "car", "attribute": {"red": "off", "yellow": "off", "green": "off", "others_arrow": "on"}},
                {"red": 0, "yellow": 0, "green": 0, "arrow": 1},
                1,
                "valid",
            ),
            (
                {"type": "car", "attribute": [{"red": "off", "yellow": "off", "green": "off", "left_arrow": "off"}]},
                {"red": 0, "yellow": 0, "green": 0, "arrow": 0},
                1,
                "valid",
            ),
            (
                {
                    "type": "pedestrian",
                    "attribute": {"red": "on", "yellow": "off", "green": "off", "left_arrow": "off"},
                },
                {"red": 0, "yellow": 0, "green": 0, "arrow": 0},
                0,
                "non_car_traffic_light",
            ),
            (
                {"type": "car", "attribute": "red"},
                {"red": 0, "yellow": 0, "green": 0, "arrow": 0},
                0,
                "missing_attribute_map",
            ),
            (
                {"type": "car", "attribute": {"red": "on", "yellow": "off", "green": "off", "x_light": "on"}},
                {"red": 1, "yellow": 0, "green": 0, "arrow": 0},
                0,
                "x_light_active",
            ),
            (
                {"type": "car", "attribute": {"red": "on", "yellow": "on", "green": "off", "left_arrow": "off"}},
                {"red": 1, "yellow": 1, "green": 0, "arrow": 0},
                0,
                "multi_color_active",
            ),
            (
                {
                    "type": "car",
                    "attribute": [
                        {"red": "off", "yellow": "on", "green": "off", "left_arrow": "off"},
                        {"red": "on", "yellow": "off", "green": "off", "left_arrow": "off"},
                    ],
                },
                {"red": 0, "yellow": 1, "green": 0, "arrow": 0},
                1,
                "valid",
            ),
        ]

        for annotation, expected_bits, expected_valid, expected_reason in cases:
            with self.subTest(annotation=annotation):
                label = collapse_aihub_traffic_light_attr(annotation)

                expected = (expected_bits, expected_valid, expected_reason)
                self.assertEqual(label.as_traffic_worker_tuple(), expected)
                self.assertEqual(_tl_bits_from_annotation(annotation), expected)
                self.assertEqual(list(label.tl_bits), list(TL_BITS))

    def test_signal_attr_label_extractor_outputs_canonical_combo_names(self) -> None:
        label = collapse_aihub_traffic_light_attr(
            {
                "type": "car",
                "attribute": {"red": "on", "yellow": "off", "green": "off", "others_arrow": "on"},
            }
        )

        self.assertEqual(label.base_color, "red")
        self.assertEqual(label.arrow, 1)
        self.assertEqual(combo_name(label.tl_bits), "red+arrow")
