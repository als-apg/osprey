"""The served runner's stopped-IOC swap, without a server.

A stop's Channel Access record swap runs on the CA server's thread, and the
pass that asked for it waits: a swap that raises or does not run in time fails
the pass, never runs late, and leaves the stopped set for the next pass to
change again. The live half, held clients across a stop, is
``tests/va/test_stopped_servers_live.py``; it does not import the runner, whose CA
server library must stay out of a process holding CA clients.
"""

from __future__ import annotations

import queue
import threading
import types
from typing import Any

import pytest

from tests.va.test_runner_emitted_post import serving  # noqa: F401 - the fixture


def _runner_stub(**fields: Any) -> types.SimpleNamespace:
    """What ``ModelRunner``'s stopped-IOC methods read of their runner, and nothing more."""
    return types.SimpleNamespace(
        _ca_jobs=queue.SimpleQueue(), _stopped=frozenset(), config={}, pvs={}, **fields
    )


def test_a_swap_the_ca_thread_never_runs_fails_and_never_runs_late(
    serving: tuple[Any, Any],  # noqa: F811 - the imported fixture
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, runner = serving
    monkeypatch.setattr(runner, "_CA_JOB_TIMEOUT_S", 0.05)
    stub = _runner_stub()
    ran: list[bool] = []
    with pytest.raises(RuntimeError, match="did not run a record swap in time"):
        runner.ModelRunner._on_ca_thread(stub, lambda: ran.append(True))
    stub._ca_jobs.get_nowait()()  # the CA thread gets to it late
    assert ran == []


def test_a_swap_that_raises_fails_the_caller(
    serving: tuple[Any, Any],  # noqa: F811 - the imported fixture
) -> None:
    _, runner = serving
    stub = _runner_stub()

    def swap() -> None:
        raise KeyError("dropped")

    def ca_thread() -> None:
        stub._ca_jobs.get()()

    worker = threading.Thread(target=ca_thread)
    worker.start()
    with pytest.raises(KeyError):
        runner.ModelRunner._on_ca_thread(stub, swap)
    worker.join()


def test_a_failed_swap_leaves_the_stopped_set_for_the_next_pass(
    serving: tuple[Any, Any],  # noqa: F811 - the imported fixture
) -> None:
    _, runner = serving

    def late(_job: Any) -> None:
        raise RuntimeError("The Channel Access server did not run a record swap in time")

    stub = _runner_stub(
        ca_server=object(),
        _read_stopped=lambda: frozenset({"ZZ:M:RB"}),
        _swap_ca_records=lambda *_: None,
        _on_ca_thread=late,
    )
    with pytest.raises(RuntimeError):
        runner.ModelRunner._follow_stopped(stub)
    assert stub._stopped == frozenset()
