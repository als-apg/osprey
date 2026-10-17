"""The process's Channel Access loop: what it points the CA context at.

Each test makes a ``CaLoop`` of its own rather than the process's, and puts a
stand-in for aioca's context in ``sys.modules``, so nothing here talks Channel
Access or touches the shared loop.
"""

import asyncio
import os
import sys
import types

import pytest

from osprey_connectors.control_system.ca_loop import CA_ENV_KEYS, CaLoop

GATEWAY_A = {
    "EPICS_CA_ADDR_LIST": "gw-a",
    "EPICS_CA_SERVER_PORT": "5064",
    "EPICS_CA_NAME_SERVERS": None,
    "EPICS_CA_AUTO_ADDR_LIST": "NO",
}
GATEWAY_B = dict(GATEWAY_A, EPICS_CA_ADDR_LIST="gw-b")


async def _nothing() -> None:
    return None


class _FakeContext:
    """aioca's ``_Context``, recording what the loop does to it."""

    def __init__(self) -> None:
        self._ca_context = 0xC0FFEE
        self._should_destroy = True
        self.purged = 0
        self.detached = 0

    def purge_channel_caches(self) -> None:
        self.purged += 1


@pytest.fixture
def context(monkeypatch):
    fake = _FakeContext()

    def detach() -> None:
        fake.detached += 1

    catools = types.SimpleNamespace(
        _Context=fake, cadef=types.SimpleNamespace(ca_detach_context=detach)
    )
    monkeypatch.setitem(sys.modules, "aioca._catools", catools)
    return fake


@pytest.fixture
def ca_loop(monkeypatch):
    for key in CA_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    loop = CaLoop()
    yield loop
    loop.stop()


def _environment() -> dict:
    return {key: os.environ.get(key) for key in CA_ENV_KEYS}


def test_pointing_before_any_call_only_sets_the_environment(ca_loop, context):
    asyncio.run(ca_loop.point_at(GATEWAY_A))
    asyncio.run(ca_loop.point_at(GATEWAY_B))

    assert _environment() == GATEWAY_B
    assert context.purged == 0


def test_pointing_elsewhere_after_a_call_drops_the_context(ca_loop, context):
    asyncio.run(ca_loop.point_at(GATEWAY_A))
    ca_loop.run_sync(_nothing(), timeout=5)

    asyncio.run(ca_loop.point_at(GATEWAY_B))

    assert _environment() == GATEWAY_B
    assert (context.purged, context.detached) == (1, 1)
    assert context._ca_context is None
    assert context._should_destroy is False


def test_pointing_at_the_same_settings_keeps_the_context(ca_loop, context):
    asyncio.run(ca_loop.point_at(GATEWAY_A))
    ca_loop.run_sync(_nothing(), timeout=5)

    asyncio.run(ca_loop.point_at(dict(GATEWAY_A)))

    assert context.purged == 0


def test_a_dropped_context_is_dropped_again_only_after_the_next_call(ca_loop, context):
    asyncio.run(ca_loop.point_at(GATEWAY_A))
    ca_loop.run_sync(_nothing(), timeout=5)
    asyncio.run(ca_loop.point_at(GATEWAY_B))
    asyncio.run(ca_loop.point_at(GATEWAY_A))
    assert context.purged == 1

    ca_loop.run_sync(_nothing(), timeout=5)
    asyncio.run(ca_loop.point_at(GATEWAY_B))
    assert context.purged == 2


def test_a_first_call_with_nothing_pointed_pins_the_environment_it_ran_with(
    ca_loop, context, monkeypatch
):
    """aioca used directly calls CA without pointing; a connector after it still moves it."""
    for key, setting in GATEWAY_A.items():
        if setting is not None:
            monkeypatch.setenv(key, setting)
    ca_loop.run_sync(_nothing(), timeout=5)

    asyncio.run(ca_loop.point_at(GATEWAY_A))
    assert context.purged == 0
    asyncio.run(ca_loop.point_at(GATEWAY_B))
    assert context.purged == 1


def test_pointing_from_the_loop_thread_runs_in_place(ca_loop, context):
    async def nested() -> None:
        await ca_loop.run(_nothing())
        await ca_loop.point_at(GATEWAY_B)

    asyncio.run(ca_loop.point_at(GATEWAY_A))
    asyncio.run_coroutine_threadsafe(nested(), ca_loop.loop).result(5)

    assert _environment() == GATEWAY_B
    assert context.purged == 1


def _register(ca_loop, reopen) -> None:
    """Register a re-opener the way the connector does: on the loop thread.

    The registering call is itself a CA call, so a test points the loop first
    and that call pins the settings it was pointed at.
    """

    async def on_loop() -> None:
        ca_loop.add_reopener(reopen)

    ca_loop.run_sync(on_loop(), timeout=5)


def test_a_repoint_reopens_each_registered_monitor_on_the_loop_thread(ca_loop, context):
    asyncio.run(ca_loop.point_at(GATEWAY_A))
    ran: list[tuple[str, bool, int]] = []
    _register(ca_loop, lambda: ran.append(("a", ca_loop.on_loop_thread(), context.purged)))
    _register(ca_loop, lambda: ran.append(("b", ca_loop.on_loop_thread(), context.purged)))

    asyncio.run(ca_loop.point_at(GATEWAY_B))

    # After the old context was dropped, in the order they were registered.
    assert ran == [("a", True, 1), ("b", True, 1)]


def test_a_reopened_monitor_counts_as_a_call_on_the_new_context(ca_loop, context):
    """Pointing away again must drop the context the re-opened monitors live on."""
    asyncio.run(ca_loop.point_at(GATEWAY_A))
    _register(ca_loop, lambda: None)
    asyncio.run(ca_loop.point_at(GATEWAY_B))

    asyncio.run(ca_loop.point_at(GATEWAY_A))

    assert context.purged == 2


@pytest.mark.usefixtures("context")
def test_nothing_is_reopened_when_the_context_stays(ca_loop):
    asyncio.run(ca_loop.point_at(GATEWAY_A))
    ran: list[str] = []
    _register(ca_loop, lambda: ran.append("reopened"))

    asyncio.run(ca_loop.point_at(dict(GATEWAY_A)))

    assert ran == []


@pytest.mark.usefixtures("context")
def test_a_discarded_reopener_is_not_run(ca_loop):
    asyncio.run(ca_loop.point_at(GATEWAY_A))
    ran: list[str] = []

    def reopen() -> None:
        ran.append("reopened")

    _register(ca_loop, reopen)

    async def discard() -> None:
        ca_loop.discard_reopener(reopen)
        ca_loop.discard_reopener(reopen)  # a no-op the second time

    ca_loop.run_sync(discard(), timeout=5)

    asyncio.run(ca_loop.point_at(GATEWAY_B))

    assert ran == []


@pytest.mark.usefixtures("context")
def test_one_monitor_that_fails_to_reopen_leaves_the_rest_reopened(ca_loop, caplog):
    asyncio.run(ca_loop.point_at(GATEWAY_A))
    ran: list[str] = []

    def broken() -> None:
        raise RuntimeError("no such channel")

    _register(ca_loop, broken)
    _register(ca_loop, lambda: ran.append("reopened"))

    asyncio.run(ca_loop.point_at(GATEWAY_B))

    assert ran == ["reopened"]
    assert "not re-opened" in caplog.text


@pytest.mark.skipif(sys.platform == "win32", reason="fork() is POSIX")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_a_forked_child_is_refused_rather_than_left_to_hang(ca_loop):
    """The child inherits the loop object but not its thread."""
    ca_loop.run_sync(_nothing(), timeout=5)
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child reports through the pipe
        try:
            ca_loop.run_sync(_nothing(), timeout=5)
            verdict = b"ran"
        except RuntimeError as exc:
            verdict = b"refused" if b"forked" in str(exc).encode() else b"other"
        except Exception:
            verdict = b"other"
        os.write(write, verdict)
        os._exit(0)
    os.close(write)
    os.waitpid(pid, 0)
    with os.fdopen(read, "rb") as child:
        assert child.read() == b"refused"
