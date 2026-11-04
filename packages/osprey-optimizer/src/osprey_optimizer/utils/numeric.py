"""Numeric safety predicates shared across hardware-facing boundaries."""

import math
import numbers
from typing import TypeGuard


def is_usable_setpoint_number(value: object) -> TypeGuard[numbers.Real]:
    """True for a real, finite, non-bool number — the only values allowed to become hardware setpoints.

    The last-line guard before a generator suggestion or a restored snapshot
    value is written to the machine.
    """
    return not isinstance(value, bool) and isinstance(value, numbers.Real) and math.isfinite(value)
