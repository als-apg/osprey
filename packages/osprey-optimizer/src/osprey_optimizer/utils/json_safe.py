"""Scrubbers keeping payloads JSON-serializable.

Starlette renders responses with ``json.dumps(allow_nan=False)``, so a NaN or Inf
anywhere in a body raises and the endpoint returns 500 instead of its real status.
Failed PV reads surface as NaN, so payloads are scrubbed on the way out. The leaf
policy differs per caller — drop, null out, or stringify — but the walk does not.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any


def map_floats(obj: Any, leaf: Callable[[float], Any]) -> Any:
    """Rebuild *obj*, applying *leaf* to every float nested inside it.

    Pure: returns new containers, primitives pass through. ``bool`` is not a
    ``float`` subclass, so flags are left alone.
    """
    if isinstance(obj, float):
        return leaf(obj)
    if isinstance(obj, dict):
        return {k: map_floats(v, leaf) for k, v in obj.items()}
    if isinstance(obj, list):
        return [map_floats(v, leaf) for v in obj]
    if isinstance(obj, tuple):
        return tuple(map_floats(v, leaf) for v in obj)
    return obj


def drop_non_finite(values: dict[str, float]) -> dict[str, float]:
    """Drop non-finite (and non-numeric) entries from a flat mapping."""
    return {k: v for k, v in values.items() if isinstance(v, (int, float)) and math.isfinite(v)}
