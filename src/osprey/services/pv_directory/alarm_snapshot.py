"""One process's task around a live alarm-state backend.

:class:`AlarmSnapshot` is handed a constructed backend and knows nothing of config:
who builds the backend, and who replaces the snapshot when the target moves, is the
directory service. The snapshot owns only the task lifetime. The first read inside
a running loop starts the backend's ``start()``; :meth:`AlarmSnapshot.aclose` cancels
it and closes the backend. A read waits for the start up to the caller's deadline,
then reads the backend once with what is left of it; the backend does the waiting
for catch-up. The answer is exactly the backend's: a feed still loading, a missing
client or an unreachable broker stays its ``Unavailable``, never a clear state, and
``stale`` is the backend's flag, passed on unchanged. :meth:`AlarmSnapshot.probe`
waits the same way, with one read, and then asks the backend's probe, which
answers at once.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeVar

__all__ = ["AlarmSnapshot"]

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

#: The PV a probe's waiting read asks for; any name waits the same.
_ANY_PV = ""


class _AlarmStateBackend(Protocol):
    """The part of the connectors' ``AlarmState`` a snapshot drives."""

    async def start(self) -> None: ...

    async def aclose(self) -> None: ...

    async def state_for(self, pv: str, deadline: float) -> Any: ...

    async def config_for(self, pv: str, deadline: float) -> Any: ...

    async def probe(self, deadline: float) -> Any: ...


class AlarmSnapshot:
    """The alarm state of one target, read from one backend this process starts once.

    Args:
        backend: a constructed alarm-state backend, not yet started.
        target: the control target the backend answers for.
    """

    def __init__(self, backend: _AlarmStateBackend, target: str) -> None:
        self.backend = backend
        self.target = target
        self._task: asyncio.Task[None] | None = None

    async def state_for(self, pv: str, deadline: float) -> Any:
        """The backend's alarm state of *pv*, read once within *deadline* seconds."""
        return await self._read(self.backend.state_for, pv, deadline)

    async def config_for(self, pv: str, deadline: float) -> Any:
        """The backend's live alarm configuration of *pv*, read as :meth:`state_for` reads."""
        return await self._read(self.backend.config_for, pv, deadline)

    async def probe(self, deadline: float) -> Any:
        """The backend's probe, once the feed has had up to *deadline* seconds to catch up.

        The wait is one read, whose answer is dropped: the backend waits for
        catch-up inside its reads, and its probe says what it has right now.
        """
        loop = asyncio.get_running_loop()
        until = loop.time() + max(deadline, 0.0)
        await self._read(self.backend.state_for, _ANY_PV, deadline)
        return await self.backend.probe(max(until - loop.time(), 0.0))

    async def aclose(self) -> None:
        """Cancel the start task and close the backend; idempotent."""
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await self.backend.aclose()

    async def _read(
        self, read: Callable[[str, float], Awaitable[_T]], pv: str, deadline: float
    ) -> _T:
        loop = asyncio.get_running_loop()
        until = loop.time() + max(deadline, 0.0)
        task = self._started()
        if not task.done():
            # wait, not wait_for: a deadline that runs out leaves the start running.
            await asyncio.wait({task}, timeout=max(until - loop.time(), 0.0))
        return await read(pv, max(until - loop.time(), 0.0))

    def _started(self) -> asyncio.Task[None]:
        if self._task is None:
            self._task = asyncio.create_task(self._start(), name=f"alarm-state-{self.target}")
        return self._task

    async def _start(self) -> None:
        try:
            await self.backend.start()
        except asyncio.CancelledError:
            raise
        except Exception:
            # The backend's reads say why it has no answer; the start only logs.
            logger.exception("alarm-state backend for %s did not start", self.target)
