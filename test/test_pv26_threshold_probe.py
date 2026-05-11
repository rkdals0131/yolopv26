import unittest

from model.engine.postprocess import PV26PostprocessConfig
from tools.probe_pv26_lane60_postprocess_thresholds import _filter_variants, _threshold_variants


class PV26ThresholdProbeTest(unittest.TestCase):
    def test_filter_variants_preserves_requested_order_and_config(self) -> None:
        variants = _threshold_variants(PV26PostprocessConfig())

        selected = _filter_variants(
            variants,
            "baseline,lane_obj_0.45__cross_mask_0.40__cross_area_32",
        )

        self.assertEqual(
            [name for name, _ in selected],
            ["baseline", "lane_obj_0.45__cross_mask_0.40__cross_area_32"],
        )
        _, candidate = selected[1]
        self.assertEqual(candidate.lane_obj_threshold, 0.45)
        self.assertEqual(candidate.crosswalk_mask_binary_threshold, 0.40)
        self.assertEqual(candidate.crosswalk_min_component_pixels, 32)

    def test_filter_variants_rejects_unknown_name(self) -> None:
        variants = _threshold_variants(PV26PostprocessConfig())

        with self.assertRaisesRegex(ValueError, "unknown threshold variant"):
            _filter_variants(variants, "baseline,missing_variant")


if __name__ == "__main__":
    unittest.main()
