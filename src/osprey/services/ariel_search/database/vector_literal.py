"""Text form of a pgvector value, for binding as ``%(v)s::vector``.

No pgvector adapter is registered on ARIEL's connections, so a vector travels
as its bracketed text literal and the statement casts it.
"""

import math
from collections.abc import Iterable

#: The largest finite single-precision float; pgvector stores float32 and
#: rejects a component beyond it.
FLOAT32_MAX = 3.4028234663852886e38


def vector_literal(vec: Iterable[float]) -> str:
    """Return the pgvector text literal of *vec*, e.g. ``[0.5,-1.0,2.0]``.

    Every component is written as a Python ``float``, whose ``str`` form
    round-trips exactly, so the text is exact; the column stores it at single
    precision. Bind the result as a parameter and cast it in SQL:
    ``%(v)s::vector``.

    Args:
        vec: The vector's components; any real numbers (ints, floats, numpy
            scalars).

    Returns:
        The bracketed, comma-separated literal.

    Raises:
        ValueError: *vec* is empty, or a component is NaN, infinite or beyond
            single-precision range; pgvector accepts none of these.
    """
    parts: list[str] = []
    for x in vec:
        value = float(x)
        if not math.isfinite(value):
            raise ValueError(f"vector component {x!r} is not finite")
        if abs(value) > FLOAT32_MAX:
            raise ValueError(f"vector component {x!r} is beyond single-precision range")
        parts.append(str(value))
    if not parts:
        raise ValueError("vector must have at least one component")
    return "[" + ",".join(parts) + "]"
