"""Single point where a native PVReading value is reduced to a float.

The optimizer (xopt) requires floats for the objective and for VOCS current
values; everything upstream now carries native types via ``PVReading``. This
helper is the only place that reduction happens. Callers decide what ``None``
means in their context (raise EvaluationError vs OptimizerConnectionError).
"""

import math
from collections.abc import Iterable, Mapping

from osprey.services.optimizer.models.protocols import PVReading


def reading_to_float(reading: PVReading) -> float | None:
    """The reading's value as a float, or ``None`` if it isn't a usable number.

    Returns ``None`` for a disconnected read, a missing/NaN value, a ``bool``
    (digital channel, not an optimization scalar), and non-numeric values
    (strings, waveforms). Callers decide what ``None`` means in their context.
    """
    v = reading.value
    if not reading.connected:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v):
        return None
    # ``math.inf`` passes through intentionally as a real float (not ``None``): it is
    # xopt's hard-fail sentinel, and the runner's drop_nonfinite_rows /
    # best_finite_row exclude non-finite objective rows downstream. Only
    # NaN / None / non-numeric / disconnected reduce to ``None`` here.
    return float(v)


def coerce_readings(
    readings: Mapping[str, PVReading],
    names: Iterable[str],
) -> tuple[dict[str, float | None], list[str]]:
    """Coerce each requested name's reading to a float, and report the unreadable ones.

    A name absent from *readings* counts as unreadable. Callers raise their own
    error type over the returned name list — the shape of "which variables could
    not be read" is what is shared, not the failure policy.
    """
    coerced = {n: reading_to_float(readings[n]) if n in readings else None for n in names}
    return coerced, [n for n, v in coerced.items() if v is None]
