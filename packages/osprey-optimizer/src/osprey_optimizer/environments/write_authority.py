"""Under which write authorization this process's writes go out.

The optimizer writes to a machine only under a program's bounded write
authorization (:mod:`osprey_connectors.write_authorization`): the connector
child that carries a write names the authorization, and the child checks every
write against it. :class:`~osprey_optimizer.environments.interface.ConnectorInterface`
asks :func:`current_write_authority` on every write, and opens its connector
under the owner and the authorization it names.

A job worker installs one authority for its whole process
(:func:`install_process_authority`), which names the authorization once the run
holds one. An API route that writes once sets one for its own call
(:func:`use_write_authority`), which wins over the process's. An authority that
is *required* refuses a write while it names no authorization, before the
connector is asked, so nothing the service does writes under a person's switch
alone. An authority that has *ended* refuses every write from then on.

With no authority installed at all, writes go out as the connector's own
verdict decides; that is the interface used outside the service (its own
tests, a notebook).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

__all__ = [
    "NO_AUTHORIZATION",
    "WriteAuthority",
    "current_write_authority",
    "install_process_authority",
    "use_write_authority",
]

#: The reason a write is refused when its authority requires an authorization
#: and names none: the run has not been granted one, or the route asked for none.
NO_AUTHORIZATION = "no_authorization"


@dataclass
class WriteAuthority:
    """The owner and the authorization this process's writes go out under.

    Attributes:
        owner: The login the writes are for, stamped on the connector child;
            ``None`` leaves it to the owner ladder the child inherits (a job
            worker stamps its run's owner in its own environment).
        authorization: The id of the write authorization the child names, or
            ``None`` while there is none.
        required: Refuse a write while :attr:`authorization` is ``None``.
        ended: The reason the authorization ended, or ``None``. Once set it is
            never cleared, and every write after it is refused.
        on_end: Told ``(reason, detail)`` by the call that ends it.
    """

    owner: str | None = None
    authorization: str | None = None
    required: bool = True
    ended: str | None = None
    on_end: Callable[[str, str | None], None] | None = field(
        default=None, repr=False, compare=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def end(self, reason: str, detail: str | None = None) -> bool:
        """End the authority for *reason*; whether this call ended it (the first reason stands).

        The call that ends it tells :attr:`on_end`, so the job a run belongs to
        shows why its writes stopped, whoever noticed first.
        """
        with self._lock:
            if self.ended is not None:
                return False
            self.ended = reason
        if self.on_end is not None:
            self.on_end(reason, detail)
        return True


_CALL_AUTHORITY: ContextVar[WriteAuthority | None] = ContextVar(
    "optimizer_write_authority", default=None
)
_process_authority: WriteAuthority | None = None


def current_write_authority() -> WriteAuthority | None:
    """The authority a write made now goes out under: the call's, else the process's."""
    return _CALL_AUTHORITY.get() or _process_authority


def install_process_authority(authority: WriteAuthority | None) -> None:
    """Make *authority* the one every write of this process goes out under; ``None`` removes it."""
    global _process_authority
    _process_authority = authority


@contextmanager
def use_write_authority(authority: WriteAuthority) -> Iterator[WriteAuthority]:
    """Make *authority* the one the writes made inside the block go out under."""
    token = _CALL_AUTHORITY.set(authority)
    try:
        yield authority
    finally:
        _CALL_AUTHORITY.reset(token)
