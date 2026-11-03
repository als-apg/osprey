"""Which of a write pass's values the runner posts.

A write pass reads back every channel it serves, but a client already holds
the value most of them carry. :func:`changed_outputs` keeps the values a client
does not hold yet: the written channels, which a put always processes, the
channels whose value changed since they were last posted, and those whose
undefined condition flipped. :func:`same_value` is the equality that decides
it: two values are the same only when they would reach a client as the same
bytes.

Nothing here imports the serving runtime, so the rule is decided and tested
in process.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from typing import Any

import numpy as np

__all__ = ["changed_outputs", "same_value"]


def same_value(a: Any, b: Any) -> bool:
    """Whether ``a`` and ``b`` would be posted as the same bytes.

    The two must be of one type, so ``True`` is not ``1`` and ``1`` is not
    ``1.0``. Floats are the same when equal with the same sign bit, so ``-0.0``
    is not ``0.0``, or when both are NaN. Arrays are the same when their dtype
    and shape match and every element is, NaN included. Anything else
    compares with ``==``.
    """
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
    if isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        equal_nan = a.dtype.kind in "fc"
        return bool(np.array_equal(a, b, equal_nan=equal_nan))
    return bool(a == b)


def changed_outputs(
    out_values: Mapping[str, Any],
    served: Mapping[str, Any],
    written: Collection[str],
    udf_now: Collection[str],
    udf_served: Collection[str],
) -> dict[str, Any]:
    """The entries of ``out_values`` a write pass posts, in their order.

    Args:
        out_values: The pass's values by name.
        served: The value each name carried when it was last posted.
        written: The names the pass's put wrote.
        udf_now: The names undefined this pass.
        udf_served: The names undefined when they were last posted.

    Returns:
        Every entry whose name was written, was never posted, carries a value
        not the same as its last post's, or whose undefined condition flipped.
        A ``None`` value is always kept.
    """
    return {
        name: value
        for name, value in out_values.items()
        if value is None
        or name in written
        or name not in served
        or not same_value(value, served[name])
        or (name in udf_now) != (name in udf_served)
    }
