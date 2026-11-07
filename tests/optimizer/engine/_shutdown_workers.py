"""Spawn targets for the service-shutdown tests (see test_job_tracker.py).

Module-level so a ``spawn`` child can import them, and kept light: the child
imports only what it runs. Each stands in for ``run_optimization_process`` with
one behaviour under a cancel: none writes a setpoint. Each sets ``ready`` once
it is in the state a test means to interrupt.
"""

from __future__ import annotations

import signal
import time
from multiprocessing.synchronize import Event
from unittest.mock import MagicMock

from osprey_optimizer.api.job_tracker import JobLifecycle
from osprey_optimizer.models.exceptions import CancellationError
from osprey_optimizer.utils.cancellation import CancellationToken, get_cancellation_token
from osprey_optimizer.utils.process import install_worker_signal_handlers

_POLL_SECONDS = 0.02


def cooperative_job(job_id: str, ready: Event) -> None:
    """A run that polls its token between steps, as the engine does."""
    with JobLifecycle(job_id):
        token = get_cancellation_token(job_id)
        ready.set()
        while True:
            token.check_cancellation()
            time.sleep(_POLL_SECONDS)


def stubborn_job(job_id: str, ready: Event) -> None:
    """A run stuck in one step: it never polls its token and ignores SIGTERM."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    with JobLifecycle(job_id):
        ready.set()
        while True:
            time.sleep(_POLL_SECONDS)


def signalled_token_loop(ready: Event) -> None:
    """Poll a token whose Redis state never says cancelled; exit 0 on a cancel.

    With the worker handlers installed, only a signal can cancel it.
    """
    install_worker_signal_handlers()
    redis = MagicMock()
    redis.hgetall.return_value = {}
    token = CancellationToken(redis_client=redis, redis_key="job:signalled:cancel")
    ready.set()
    try:
        while True:
            token.check_cancellation()
            time.sleep(_POLL_SECONDS)
    except CancellationError:
        return


def signalled_but_stuck(ready: Event) -> None:
    """Install the worker handlers, then never poll: only the deadline ends it."""
    install_worker_signal_handlers()
    ready.set()
    while True:
        time.sleep(_POLL_SECONDS)
