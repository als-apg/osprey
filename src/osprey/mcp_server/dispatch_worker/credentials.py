"""The dispatch worker's bearer, held in memory rather than in ``os.environ``.

The worker authenticates the dispatcher's calls to its ``/dispatch`` API with
``DISPATCH_WORKER_TOKEN``, and it starts agent runs in its own process tree. A
credential has to be out of this process's ``os.environ`` before the first run
starts, and the memory it moves into has to be closed to the processes this one
spawns — the same two rules the web server applies to its operator secret in
:mod:`osprey.interfaces.web_auth`.

So the entry point populates this holder before the server binds: the bearer
is popped out of ``os.environ`` into module memory, ``EVENT_DISPATCHER_TOKEN``
is popped alongside it (a deployment hands it to the worker's container too,
and the worker presents it to nothing), and
:func:`osprey.utils.sensitive_env.guard_process_memory` closes the process.

**Population happens once per process and is idempotent.** The request-time
check reads the same holder the entry point filled. A value that appears in
``os.environ`` afterwards is not a credential.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

from osprey.utils.sensitive_env import guard_process_memory

__all__ = [
    "DISPATCHER_TOKEN_ENV",
    "WORKER_TOKEN_ENV",
    "WorkerCredentials",
    "reset_worker_credentials",
    "worker_credentials",
]

#: The bearer the dispatcher presents to this worker's ``/dispatch`` API.
WORKER_TOKEN_ENV = "DISPATCH_WORKER_TOKEN"

#: The dispatcher's own bearer. Taken out of the environment with the worker's,
#: and kept only so run output can still be masked against it.
DISPATCHER_TOKEN_ENV = "EVENT_DISPATCHER_TOKEN"


@dataclass(frozen=True)
class WorkerCredentials:
    """What one worker process authenticates with and masks in its output."""

    #: The bearer ``/dispatch`` callers must present; ``""`` when the deployment
    #: supplied none, in which case every call is refused.
    worker_token: str

    #: Every non-blank value the holder took out of the environment, longest
    #: first, for scrubbing run output. Includes the worker token itself.
    scrub_values: tuple[str, ...]


_LOCK = threading.Lock()
_CREDENTIALS: WorkerCredentials | None = None


def worker_credentials() -> WorkerCredentials:
    """Return this process's credentials, populating them on the first call.

    The first caller pops the carriers out of ``os.environ``, then guards the
    process memory — in that order, so an error from the guard never leaves a
    carrier behind for a child to inherit. An empty or whitespace-only value
    counts as absent. Every later caller gets the same object back.

    Raises:
        RuntimeError: from the guard, when the kernel refuses to close the
            process — see :func:`osprey.utils.sensitive_env.guard_process_memory`.
    """
    global _CREDENTIALS
    with _LOCK:
        if _CREDENTIALS is None:
            worker_token = os.environ.pop(WORKER_TOKEN_ENV, "").strip()
            dispatcher_token = os.environ.pop(DISPATCHER_TOKEN_ENV, "").strip()
            guard_process_memory()
            held = tuple(
                sorted({v for v in (worker_token, dispatcher_token) if v}, key=len, reverse=True)
            )
            _CREDENTIALS = WorkerCredentials(worker_token=worker_token, scrub_values=held)
        return _CREDENTIALS


def reset_worker_credentials() -> None:
    """Forget the process credentials so the next call re-populates them.

    For tests: population is process-wide and one-shot, so without this the
    first test in a worker would decide the bearer for every test after it. It
    does not restore the environment variables population popped.
    """
    global _CREDENTIALS
    with _LOCK:
        _CREDENTIALS = None
