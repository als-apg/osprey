"""Channel value coercion and zero values per ``value_type``.

One function set shared by the connectors package and the facility build, so
both accept and refuse exactly the same values. The module imports only the
standard library: no framework enters through it.

Stored representations:

* ``float`` -> ``float``; ``int`` -> ``int``;
* ``bool`` / ``enum`` -> the label (``str``) from ``options``;
* ``string`` -> ``str``;
* ``waveform`` -> a flat, row-major ``list[float]`` whose length is the
  product of ``shape``.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from typing import Any

VALUE_TYPES = ("float", "int", "bool", "enum", "string", "waveform")
DEFAULT_VALUE_TYPE = "float"
DEFAULT_BOOL_OPTIONS = ("FALSE", "TRUE")

__all__ = [
    "DEFAULT_BOOL_OPTIONS",
    "DEFAULT_VALUE_TYPE",
    "VALUE_TYPES",
    "coerce",
    "zero",
]


def _refuse(value: Any, value_type: str, reason: str) -> ValueError:
    return ValueError(f"value {value!r} is not a valid {value_type}: {reason}")


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _is_integer(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _labels(value_type: str, options: Sequence[str] | None) -> list[str]:
    if options is None:
        if value_type == "bool":
            return list(DEFAULT_BOOL_OPTIONS)
        raise ValueError(f"value_type {value_type} needs options (its labels)")
    return [str(label) for label in options]


def _size(shape: Sequence[int] | None) -> int:
    if not shape:
        raise ValueError("value_type waveform needs shape (a list of positive ints)")
    return math.prod(shape)


def _flatten(value: Any, out: list[Any]) -> None:
    if isinstance(value, (list, tuple)):
        for item in value:
            _flatten(item, out)
    else:
        out.append(value)


def _coerce_label(value: Any, value_type: str, labels: list[str]) -> str:
    if isinstance(value, str):
        if value in labels:
            return value
        raise _refuse(value, value_type, f"label not in {labels}")
    if isinstance(value, bool) or _is_integer(value):
        index = int(value)
    else:
        raise _refuse(value, value_type, "expected an index, a bool or a label")
    if 0 <= index < len(labels):
        return labels[index]
    raise _refuse(value, value_type, f"index outside 0..{len(labels) - 1}")


def coerce(
    value: Any,
    value_type: str | None,
    options: Sequence[str] | None = None,
    shape: Sequence[int] | None = None,
) -> Any:
    """Coerce ``value`` to the stored representation of ``value_type``.

    Args:
        value: The value to coerce.
        value_type: One of :data:`VALUE_TYPES`; ``None`` means ``float``.
        options: The labels of a ``bool`` or ``enum`` channel; a ``bool``
            without options uses :data:`DEFAULT_BOOL_OPTIONS`.
        shape: The dimensions of a ``waveform`` channel.

    Returns:
        The coerced value in its stored representation.

    Raises:
        ValueError: The value is refused for the type; the message names the
            type and the value.
    """
    value_type = value_type or DEFAULT_VALUE_TYPE
    if value_type == "float":
        if _is_number(value) and math.isfinite(value):
            return float(value)
        raise _refuse(value, value_type, "expected a finite int or float")
    if value_type == "int":
        if _is_integer(value):
            return int(value)
        if _is_number(value) and math.isfinite(value) and float(value).is_integer():
            return int(value)
        raise _refuse(value, value_type, "expected an int or an integral float")
    if value_type in ("bool", "enum"):
        return _coerce_label(value, value_type, _labels(value_type, options))
    if value_type == "string":
        if isinstance(value, str):
            return value
        raise _refuse(value, value_type, "expected a str")
    if value_type == "waveform":
        size = _size(shape)
        if not isinstance(value, (list, tuple)):
            raise _refuse(value, value_type, "expected a numeric list")
        flat: list[Any] = []
        _flatten(value, flat)
        if not all(_is_number(item) and math.isfinite(item) for item in flat):
            raise _refuse(value, value_type, "expected finite numbers only")
        if len(flat) != size:
            raise _refuse(
                value, value_type, f"expected {size} values for shape {list(shape or ())}"
            )
        return [float(item) for item in flat]
    raise ValueError(f"unknown value_type {value_type!r} (expected one of {list(VALUE_TYPES)})")


def zero(
    value_type: str | None,
    options: Sequence[str] | None,
    shape: Sequence[int] | None,
) -> Any:
    """Return the zero value of ``value_type`` in its stored representation.

    Args:
        value_type: One of :data:`VALUE_TYPES`; ``None`` means ``float``.
        options: The labels of a ``bool`` or ``enum`` channel.
        shape: The dimensions of a ``waveform`` channel.

    Returns:
        ``0.0``, ``0``, ``options[0]``, ``""`` or ``product(shape)`` zeros.

    Raises:
        ValueError: ``value_type`` is unknown, or its options or shape are missing.
    """
    value_type = value_type or DEFAULT_VALUE_TYPE
    if value_type == "float":
        return 0.0
    if value_type == "int":
        return 0
    if value_type in ("bool", "enum"):
        labels = _labels(value_type, options)
        if not labels:
            raise ValueError(f"value_type {value_type} needs at least one label")
        return labels[0]
    if value_type == "string":
        return ""
    if value_type == "waveform":
        return [0.0] * _size(shape)
    raise ValueError(f"unknown value_type {value_type!r} (expected one of {list(VALUE_TYPES)})")
