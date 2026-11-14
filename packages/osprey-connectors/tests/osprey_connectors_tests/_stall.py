"""A transport call that stalls until the test lets it go.

DOOCS and TANGO bound each blocking transport call with ``asyncio.wait_for``
over a worker thread. A test of that bound used to time the call and compare
the elapsed seconds with a ceiling, which a busy runner can break. A
:class:`Stall` asks a question no load can change: did the connector answer
before the transport call returned? Only a bound that fired can do that;
without one, the connector answers only after the transport does.
"""

import threading
from typing import Any

#: Real seconds a stall stays parked before it gives up on its own. A passing
#: test releases it first; only a missing or too-long bound lets it run out, so
#: a failing test ends instead of hanging.
STALL_CEILING_S = 10.0


class Stall:
    """A stand-in transport call: it parks until released, then answers ``answer``.

    Used as a context manager, it is released on the way out, so the worker
    thread it holds ends before the test's event loop closes.
    """

    def __init__(self, answer: Any = None):
        self.answer = answer
        self.entered = threading.Event()
        self._returned = threading.Event()
        self._release = threading.Event()

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        self.entered.set()
        self._release.wait(STALL_CEILING_S)
        self._returned.set()
        return self.answer

    def __enter__(self) -> "Stall":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

    def release(self) -> None:
        self._release.set()

    @property
    def returned(self) -> bool:
        """Whether the transport call has come back."""
        return self._returned.is_set()
