"""The write door: marks a put as made by a connector, not by a raw client.

A connector opens the door around its own call into a control-system client
library; the raw-client guard asks :func:`door_is_open` and lets a put through
only when the answer is yes. A put made from user code that reached for the
client directly finds the door closed.

**Contract.** A connector makes its client put either in the context that
opened the door or inside ``asyncio.to_thread`` started from that context.
The flag is a context variable, so it reaches exactly the code that shares
that context: ``asyncio.to_thread`` copies the context into its worker thread
and carries the open door with it, as does every task created while the door
is open. ``loop.run_in_executor``, a raw executor ``submit`` and a bare
``threading.Thread`` start from a fresh context and see the door closed, so a
connector that puts behind one of those hops is refused.

**Cooperative bypasses.** The door records attribution, not permission, and
some paths open it for code that is not a connector's own put. These are
accepted, because attribution is not a boundary:

- pyepics' preemptive callback threads are started by libca, not from the
  caller's context, so they never carry the door. With
  ``PREEMPTIVE_CALLBACK=False`` pyepics runs callbacks only while a thread
  polls, including the poll inside a connector's own waiting put, so a put
  made from such a callback runs inside the open door.
- ``ControlSystemConnector`` wraps every subclass's ``write_channel`` and
  ``write_multiple_channels``, including those of a user-defined connector
  subclass, so a user-defined connector puts inside an open door.

This module is stdlib-only: the guard that reads it runs in executor sandboxes
and notebook kernels where ``osprey`` is not importable.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager

__all__ = ["door_is_open", "open_door"]

#: Whether the current context is inside a connector's write.
#:
#: Set and reset by token in :func:`open_door`, so a nested exit restores the
#: enclosing value rather than closing the outer door, and an exception leaves
#: the prior value in place.
_DOOR: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "osprey_connector_write_door", default=False
)


@contextmanager
def open_door() -> Iterator[None]:
    """Open the write door for the body of the ``with`` block."""
    token = _DOOR.set(True)
    try:
        yield
    finally:
        _DOOR.reset(token)


def door_is_open() -> bool:
    """Whether the calling context is inside :func:`open_door`."""
    return _DOOR.get()
