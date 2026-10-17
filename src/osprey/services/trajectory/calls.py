"""Every connector read and write of the trajectory service, run so its outcome is never lost.

A write the service sends cannot be taken back once it is in flight: cancelling
the coroutine that awaits it does not stop the connector, and the value may
land on the machine after the caller has gone. :func:`guarded_call` therefore
runs the call as its own task and never lets the caller leave before that task
has finished. When the caller is cancelled (a stop, a deadline, a shutdown),
the helper keeps awaiting the call, hands its outcome to ``record``, and only
then re-raises the cancellation. Everything the caller holds around the call,
the run lock first of all, is released only after the write has returned and
its outcome is in the caller's ``moved`` list.

``record`` is called exactly once per call, with the call's value when it
returned and with the exception when it raised (a cancelled call records its
:class:`asyncio.CancelledError`). The call task copies the caller's context,
so the owner bound for the action applies to it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

__all__ = ["guarded_call", "read_value"]

T = TypeVar("T")


async def guarded_call(
    fn: Callable[[], Awaitable[T]], *, record: Callable[[T | BaseException], Any]
) -> T:
    """Run ``fn()`` as a task, record its outcome, and outlast any cancellation of the caller.

    Args:
        fn: A zero-argument callable returning the read or write to await.
        record: Called once with the call's outcome: its value, or the
            exception it raised. Called before this helper returns or raises,
            also when the caller was cancelled while the call was in flight.

    Returns:
        The call's value.

    Raises:
        asyncio.CancelledError: The caller was cancelled while the call was in
            flight; raised only after the call finished and was recorded.
        BaseException: Whatever the call raised, after it was recorded.
    """
    task = asyncio.ensure_future(fn())
    cancelled: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            # The caller was cancelled, or the call itself was; either way the
            # call is awaited to its end before anything is given back.
            if _caller_cancelled(task):
                cancelled = exc
        except BaseException:  # the task holds the outcome; it is read below
            pass
    outcome: Any
    if task.cancelled():
        try:
            task.result()
        except asyncio.CancelledError as exc:
            outcome = exc
    else:
        outcome = task.exception()
        if outcome is None:
            outcome = task.result()
    try:
        record(outcome)
    finally:
        # A failing ``record`` never swallows the caller's cancellation.
        if cancelled is not None:
            raise cancelled
    return task.result()


async def read_value(connector: Any, address: str) -> Any:
    """Read one channel through :func:`guarded_call`, its value as a plain Python scalar.

    A numpy scalar (anything with a callable ``item``) is unwrapped; every
    other value is returned as the connector read it. A read moves nothing,
    so its outcome is not recorded.
    """
    reading = await guarded_call(lambda: connector.read_channel(address), record=lambda _: None)
    value = reading.value
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _caller_cancelled(task: asyncio.Future[Any]) -> bool:
    """Whether a ``CancelledError`` out of the shield was the caller's, not the call's own."""
    if not task.done() or not task.cancelled():
        # The call is still running or ended on its own: the shield raised
        # because the caller was cancelled.
        return True
    current = asyncio.current_task()
    return current is not None and current.cancelling() > 0
