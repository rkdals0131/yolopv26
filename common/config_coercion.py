from __future__ import annotations

from typing import Any


def coerce_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a mapping")
    return dict(value)


def coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise TypeError(f"{field_name} must be a boolean")


def coerce_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    return int(value)


def coerce_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a float")
    return float(value)


def coerce_str(value: Any, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


def coerce_str_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (coerce_str(value, field_name=field_name),)
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list")
    return tuple(coerce_str(item, field_name=f"{field_name}[]") for item in value)


def coerce_float_tuple(value: Any, *, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list")
    return tuple(coerce_float(item, field_name=f"{field_name}[]") for item in value)


def coerce_int_tuple(value: Any, *, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list")
    return tuple(coerce_int(item, field_name=f"{field_name}[]") for item in value)
