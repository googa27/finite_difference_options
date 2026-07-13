"""Type-sensitive identity guard for executable public-synthetic fixtures."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any, cast


def matches_exact_public_fixture(payload: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Return true only for identical trees of exact built-in JSON types."""

    if not _is_exact_json_value(payload) or not _is_exact_json_value(expected):
        return False
    try:
        supplied_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        expected_json = json.dumps(expected, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return supplied_json == expected_json


def _is_exact_json_value(value: object) -> bool:
    value_type = type(value)
    if value is None or value_type in {bool, int, str}:
        return True
    if value_type is float:
        return math.isfinite(cast(float, value))
    if value_type is list:
        return all(_is_exact_json_value(item) for item in cast("list[object]", value))
    if value_type is dict:
        return all(
            type(key) is str and _is_exact_json_value(item) for key, item in cast("dict[object, object]", value).items()
        )
    return False


__all__ = ["matches_exact_public_fixture"]
