"""The family-body normaliser shared by every MML loader.

A family body reaches OSPREY in two spellings: decoded from a ``.mat`` file by
scipy, or parsed from the MATLAB exporter's JSON. The two disagree on empty
chars, logicals, non-finite numbers and function handles. ``normalize_family``
rewrites a raw body into the one canonical spelling so both lanes produce the
same ``ao.json``, the same raw slot counts and the same empty-list census.

The rules are key-scoped. Only channel keys and the family arrays turn bare
strings into lists and blank slots into ``None``; every other string keeps its
value, because scalars such as ``Units`` and ``DataType`` are bound into TEXT
columns downstream. An integral number under an index or status array is
written as an ``int``, so a MATLAB double and a JSON integer spell one fact one
way. The functions here are pure and perform no I/O.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
from scipy.io.matlab import MatlabFunction, mat_struct

from osprey.services.channel_finder.databases.middle_layer import CHANNEL_KEYS
from osprey.services.mml.family import FAMILY_ARRAYS

__all__ = ["normalize_family"]

#: Family arrays counted or indexed downstream. MATLAB stores every number as
#: a double, so an index reaching Python as ``2.0`` is written ``2`` here and
#: the ``.mat`` and JSON lanes spell the same array the same way.
_INTEGRAL_KEYS: frozenset[str] = frozenset({"DeviceList", "ElementList", "Status"})

#: Keys dropped wherever they sit (MATLAB graphics handles carry no meaning).
_DROPPED_KEYS: frozenset[str] = frozenset({"Handles"})

#: With the channel keys, the per-device family arrays are the only keys whose
#: bare strings become one-slot lists and whose blank slots become ``None``.
_SLOT_KEYS: frozenset[str] = frozenset((*CHANNEL_KEYS, *FAMILY_ARRAYS))

_NON_FINITE = re.compile(r"^[+-]?(inf|infinity|nan)$", re.IGNORECASE)

#: MATLAB ``deblank`` strips trailing whitespace and NUL padding.
_DEBLANK_CHARS = " \t\n\r\f\v\x00"


def normalize_family(body: dict) -> dict:
    """Rewrite one raw family body into the canonical spelling.

    Args:
        body: A family body as a loader decoded it. It is not modified.

    Returns:
        A new dict with every normalisation rule applied. Unknown keys keep
        their values, and list shapes (1-row or N-row) are preserved.
    """
    return _normalize_dict(body)


def _normalize_dict(mapping: dict) -> dict:
    out: dict = {}
    for key, value in mapping.items():
        if key in _DROPPED_KEYS:
            continue
        out[key] = _normalize_value(value, key if isinstance(key, str) else "")
    return out


def _normalize_value(value: Any, key: str) -> Any:
    """Normalise ``value`` found under the dict key ``key``."""
    handle = _as_handle(value, key)
    if handle is not None:
        return handle
    if isinstance(value, dict):
        return _normalize_dict(value)
    if isinstance(value, str):
        if key in _SLOT_KEYS:
            row = _slot(value)
            return [] if row is None else [row]
        return _canonical_string(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_element(item, key) for item in value]
    if isinstance(value, np.ndarray):
        return _normalize_value(value.tolist(), key)
    return _normalize_scalar(value, key)


def _normalize_element(item: Any, key: str) -> Any:
    """Normalise one element of a list found under ``key``."""
    if isinstance(item, str):
        if key in _SLOT_KEYS:
            return _slot(item)
        return _canonical_string(item.rstrip(_DEBLANK_CHARS))
    if isinstance(item, (list, tuple)):
        return [_normalize_element(sub, key) for sub in item]
    return _normalize_value(item, key)


def _slot(text: str) -> str | None:
    """Deblank a slot row; a blank row is an absent slot."""
    row = text.rstrip(_DEBLANK_CHARS)
    if not row.strip():
        return None
    return _canonical_string(row)


def _canonical_string(text: str) -> str:
    """Spell a non-finite string as ``Inf``/``-Inf``/``NaN``; keep any other string."""
    match = _NON_FINITE.match(text)
    if match is None:
        return text
    if match.group(1).lower() == "nan":
        return "NaN"
    return "-Inf" if text.startswith("-") else "Inf"


def _normalize_scalar(value: Any, key: str) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            if math.isnan(value):
                return "NaN"
            return "Inf" if value > 0 else "-Inf"
        if key in _INTEGRAL_KEYS and value.is_integer():
            return int(value)
    return value


def _as_handle(value: Any, key: str) -> dict | None:
    """Return the ``{"$fn", "file"}`` form when ``value`` is a function handle."""
    if isinstance(value, MatlabFunction):
        decoded = _decode_matlab(np.asarray(value))
        if isinstance(decoded, dict):
            return _as_handle(decoded, key) or {"$fn": None, "file": None}
        if isinstance(decoded, str):
            return {"$fn": decoded, "file": None}
        return {"$fn": None, "file": None}
    if isinstance(value, dict):
        if "$fn" in value:
            return {
                "$fn": _optional_str(value.get("$fn")),
                "file": _optional_str(value.get("file")),
            }
        if "function_handle" in value:
            inner = value["function_handle"]
            if isinstance(inner, dict):
                return {
                    "$fn": _optional_str(inner.get("function")),
                    "file": _optional_str(inner.get("file")),
                }
            return {"$fn": _optional_str(inner), "file": None}
        return None
    if key.endswith("Fcn"):
        if isinstance(value, str):
            return {"$fn": value, "file": None}
        if (
            not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, np.integer))
            and value == 1
        ):
            return {"$fn": None, "file": None}
    return None


def _optional_str(value: Any) -> str | None:
    """A handle field as a string, with empty values read as absent."""
    if isinstance(value, np.ndarray):
        value = _decode_matlab(value)
    if isinstance(value, str):
        return value if value else None
    return None


def _decode_matlab(value: Any) -> Any:
    """Unwrap scipy's MATLAB containers into plain dicts, lists and strings."""
    if isinstance(value, mat_struct):
        return {name: _decode_matlab(getattr(value, name)) for name in value._fieldnames}
    if isinstance(value, np.void) and value.dtype.names:
        return {name: _decode_matlab(value[name]) for name in value.dtype.names}
    if isinstance(value, np.ndarray):
        if value.dtype.names:
            records = [_decode_matlab(item) for item in value.flat]
            return records[0] if len(records) == 1 else records
        if value.dtype.kind == "U":
            rows = [str(item) for item in value.flat]
            return rows[0] if len(rows) == 1 else rows
        if value.dtype.kind == "O":
            items = [_decode_matlab(item) for item in value.flat]
            return items[0] if len(items) == 1 else items
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
