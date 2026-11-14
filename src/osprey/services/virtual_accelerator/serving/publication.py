"""Which of a pass's values the runner posts, and which it forces.

A write pass reads back every channel it serves, but a client already holds
the value most of them carry. :func:`changed_outputs` keeps the values a client
does not hold yet: the written channels, which a put always processes, the
channels whose value changed since they were last posted, and those whose
undefined condition flipped. :func:`same_value` is the equality that decides
it: two values are the same only when they would reach a client as the same
bytes.

A view's ``fast`` and ``event`` channels publish on timed passes as well.
:func:`timed_pass_names` names what a timed pass reads: every ``fast``
channel, and the ``event`` channels the composite emitted since the last pass.
:func:`forced_names` names the posted channels Channel Access raises a monitor
event for even at a repeated value: what was emitted, and on a timed pass the
``fast`` channels too, less every channel under a ``frozen`` or
``disconnected`` read fault. :func:`untimed_outputs` is the posting of any other
pass -- start-up, a write pass, a global tick: there ``fast`` and ``event``
channels are posted only when changed, unless the pass drained them as
emitted, so a step is published once, by whichever pass advanced over it.
A global tick reads every channel and posts, like a write pass, only those a
client does not hold yet.
:func:`next_wake` is the instant the next timed pass is due.

Nothing here imports the serving runtime, so the rules are decided and tested
in process.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from typing import Any

import numpy as np

__all__ = [
    "changed_outputs",
    "forced_names",
    "next_wake",
    "same_value",
    "timed_pass_names",
    "untimed_outputs",
]


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


def timed_pass_names(fast: Collection[str], emitted: Collection[str]) -> frozenset[str]:
    """The names a timed pass reads: every ``fast`` name and every emitted one.

    Args:
        fast: The view's ``fast`` addresses.
        emitted: The names the composite emitted since the last pass, drained.

    Returns:
        The union of the two.
    """
    return frozenset(fast) | frozenset(emitted)


def forced_names(
    fast: Collection[str], emitted: Collection[str], read_faulted: Collection[str]
) -> frozenset[str]:
    """The names a posting forces a Channel Access event for, at a repeated value too.

    A timed pass passes the view's ``fast`` names; any other pass passes none,
    so it forces only what it drained as emitted.

    Args:
        fast: The ``fast`` names the pass publishes at the fast period, or
            none on a pass that is not timed.
        emitted: The names the composite emitted, drained by this pass.
        read_faulted: The names under a ``frozen`` or ``disconnected`` read
            fault; a frozen reading raises no event, and a disconnected one
            raises its severity change, not a repeat.

    Returns:
        ``fast`` and ``emitted`` together, less ``read_faulted``.
    """
    return (frozenset(fast) | frozenset(emitted)) - frozenset(read_faulted)


def untimed_outputs(
    out_values: Mapping[str, Any],
    served: Mapping[str, Any],
    timed: Collection[str],
    emitted: Collection[str],
    udf_now: Collection[str],
    udf_served: Collection[str],
    written: Collection[str] | None = None,
    *,
    tick: bool = False,
) -> dict[str, Any]:
    """The entries of ``out_values`` a pass that is not timed posts, in their order.

    A ``fast`` or ``event`` name is posted only as :func:`changed_outputs`
    would post it, unless the pass drained it as emitted: an emitted name is
    always posted, since this pass is the one that publishes its step. A
    write pass and a global tick narrow every name with
    :func:`changed_outputs`; any other pass posts every other name.

    Args:
        out_values: The pass's values by name.
        served: The value each name carried when it was last posted.
        timed: The view's ``fast`` and ``event`` names.
        emitted: The names the composite emitted, drained by this pass.
        udf_now: The names undefined this pass.
        udf_served: The names undefined when they were last posted.
        written: The names a write pass's put wrote, or ``None`` on a pass
            that writes nothing.
        tick: Whether the pass is a global tick and nothing else.

    Returns:
        The posting.
    """
    changed = changed_outputs(
        out_values,
        served,
        frozenset() if written is None else written,
        udf_now,
        udf_served,
    )
    narrowed = tick or written is not None
    return {
        name: value
        for name, value in out_values.items()
        if name in emitted or name in changed or not (narrowed or name in timed)
    }


def next_wake(now: float, fast_period: float, deadline: float | None = None) -> float:
    """The instant the next timed pass is due: a fast period on, or the deadline if sooner.

    Args:
        now: The current instant, in epoch seconds.
        fast_period: The fast period in seconds; greater than zero.
        deadline: The composite's next due step in epoch seconds, or ``None``.
            One already past is due at once and returned as it is.

    Returns:
        ``now + fast_period``, or ``deadline`` when it is earlier.

    Raises:
        ValueError: ``fast_period`` is not greater than zero.
    """
    if not fast_period > 0:
        raise ValueError(f"the fast period must be greater than zero, got {fast_period!r}")
    wake = now + fast_period
    return wake if deadline is None else min(wake, deadline)
