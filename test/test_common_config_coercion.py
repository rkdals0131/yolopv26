from __future__ import annotations

import unittest

from common.config_coercion import (
    coerce_bool,
    coerce_float_tuple,
    coerce_int,
    coerce_mapping,
    coerce_str,
    coerce_str_tuple,
)


class ConfigCoercionTests(unittest.TestCase):
    def test_coerce_mapping_accepts_none_and_dict(self) -> None:
        self.assertEqual(coerce_mapping(None, field_name="x"), {})
        self.assertEqual(coerce_mapping({"a": 1}, field_name="x"), {"a": 1})

    def test_coerce_bool_accepts_common_string_forms(self) -> None:
        self.assertTrue(coerce_bool("yes", field_name="flag"))
        self.assertTrue(coerce_bool("1", field_name="flag"))
        self.assertFalse(coerce_bool("off", field_name="flag"))
        self.assertFalse(coerce_bool("0", field_name="flag"))

    def test_coerce_int_rejects_boolean(self) -> None:
        with self.assertRaises(TypeError):
            coerce_int(True, field_name="count")

    def test_coerce_str_rejects_blank_values(self) -> None:
        with self.assertRaises(ValueError):
            coerce_str("   ", field_name="name")

    def test_tuple_helpers_normalize_values(self) -> None:
        self.assertEqual(coerce_str_tuple("abc", field_name="names"), ("abc",))
        self.assertEqual(coerce_str_tuple(["a", "b"], field_name="names"), ("a", "b"))
        self.assertEqual(coerce_float_tuple([1, "2.5"], field_name="weights"), (1.0, 2.5))
