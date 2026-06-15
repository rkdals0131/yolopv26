from __future__ import annotations

import unittest

from common.pv26_schema import TL_BITS
from tools.od_bootstrap.signal_attr import collapse_aihub_traffic_light_attr, combo_name
from tools.od_bootstrap.source.aihub.traffic_worker import _tl_bits_from_annotation


class SignalAttrAIHubPolicyTests(unittest.TestCase):
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
