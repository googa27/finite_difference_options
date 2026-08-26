"""Private option-type validation helpers for pricing instruments."""

from __future__ import annotations

from typing import Literal, TypeAlias, cast

from finite_difference_options.exceptions import ValidationError

VanillaOptionType: TypeAlias = Literal["call", "put"]


def _validate_vanilla_option_type(option_type: object) -> VanillaOptionType:
    """Return a supported vanilla option type or fail closed."""
    if not isinstance(option_type, str) or option_type not in ("call", "put"):
        raise ValidationError(f"option_type must be 'call' or 'put', got {option_type}")
    return cast(VanillaOptionType, option_type)
