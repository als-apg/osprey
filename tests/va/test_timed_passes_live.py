"""Timed passes, seen from the far side of both wires.

:class:`~osprey.services.virtual_accelerator.serving.runner.ModelRunner`
serving an evolving :class:`~osprey_connectors.simulation.composite.Composite`
over the injection fixture (``tests/facility/fixtures/injection``) queues a
timed pass every fast period: the fast stored current is read and posted with
a forced Channel Access event, so a monitor sees it at the fast rate even at a
constant value. A real ``aioca`` client and a real ``p4p`` client in the
pytest process watch a real ``pcaspy`` and ``p4p`` server.

**One server per process**, spawned as in ``test_lume_pva_seam.py``: the
spawned process builds the composite with ``evolve=True`` and the runner with
``fast_interval_s`` itself, and answers a few commands over a pipe -- the
cycles its run loop ran and the threads it holds. One module-scoped server
serves the fixture; a second, for one test, serves the same render with every
``fast`` and ``event`` channel rewritten to ``pass``.

**Timing.** The bounds scale by ``OSPREY_LIVE_CA_TIMING_SLACK`` (default 1).

**The venue is linux.** ``lume-pva-apg`` installs only there, so the module
skips whole anywhere else; ``scripts/va/live_ca/gate.py --pva`` runs it in a
process of its own and fails on any skip.
"""

from __future__ import annotations

import importlib.util
import json
import math
import multiprocessing
import os
import threading
import time
from collections.abc import Callable, Coroutine, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import pytest

from osprey_connectors_tests._harness._ports import hold_ports, reserved_ports

pytest.importorskip(
    "lume_pva_apg",
    reason="lume-pva-apg installs on linux only; run this module through the live-CA gate",
)
pytest.importorskip("p4p", reason="the PVAccess client ships with lume-pva-apg's pva extra")

# Found, never imported at module level: aioca's libca beside pcaspy's own, in
# a spawned server re-importing this module, drops the client on every write.
if importlib.util.find_spec("aioca") is None:
    pytest.skip("the Channel Access client is aioca", allow_module_level=True)

#: The served views, each in a process of its own.
SERVERS = ("injection", "quiet")

#: Whether this is the pytest process rather than a server it spawns.
_IN_PYTEST = multiprocessing.parent_process() is None

CA_PORTS: dict[str, str] = (
    {server: str(port) for server, port in zip(SERVERS, hold_ports(len(SERVERS)), strict=True)}
    if _IN_PYTEST
    else dict.fromkeys(SERVERS, "")
)
# Handed to CaLoop.point_at before a server starts, never written at import:
# other modules collected into this process write their own.
CA_CLIENT_ENV: dict[str, str | None] = {
    "EPICS_CA_ADDR_LIST": " ".join(f"127.0.0.1:{CA_PORTS[s]}" for s in SERVERS),
    "EPICS_CA_AUTO_ADDR_LIST": "NO",
    "EPICS_CA_SERVER_PORT": None,
    "EPICS_CA_NAME_SERVERS": None,
}
# import-time required because libca reads the repeater port when the first CA
# context in the process is created, which may be another module's, before any
# of this module's fixtures run; point_at does not carry it. The port is held
# for the process unless the environment pins one.
if _IN_PYTEST and "EPICS_CA_REPEATER_PORT" not in os.environ:
    os.environ["EPICS_CA_REPEATER_PORT"] = str(hold_ports(1)[0])

if importlib.util.find_spec("pcaspy") is None:
    pytest.skip(
        "the Channel Access server ships with lume-pva-apg's ca extra",
        allow_module_level=True,
    )

from p4p.client.thread import Context  # noqa: E402

from osprey.services.virtual_accelerator.serving.runner import SCHEDULER_THREAD  # noqa: E402
from tests.facility._injection_fixture import CURRENT_FAST, injection_view  # noqa: E402
from tests.va._collection import assert_module_collected  # noqa: E402

#: Floor for this module's own test count.
MIN_COLLECTED_TESTS = 5

#: The factor on every timing bound.
SLACK = float(os.environ.get("OSPREY_LIVE_CA_TIMING_SLACK", "1.0"))

#: The served runner's global tick and fast period.
TICK_S = 1.0
FAST_S = 0.1

#: Events the fast channel delivers in every 1 s window, on each wire.
RATE = (math.floor(9 / SLACK), math.ceil(11 * SLACK))
#: How long the rate is measured, in 1 s windows.
WINDOWS = 10

#: The scenario that freezes the fast channel.
FROZEN_SCENARIO = "frozen-fast"
#: How long a frozen fast channel is watched, and how soon it resumes once cleared.
FROZEN_WATCH_S = 2.0
RESUME_S = 1.0 * SLACK
#: How long an activated fault is given to take before it is watched.
FAULT_SETTLE_S = 0.5

CA_TIMEOUT_S = 30.0
STARTUP_TIMEOUT_S = 300.0
COMMAND_TIMEOUT_S = 60.0


# =============================================================================
# The views
# =============================================================================


def _with_frozen_scenario(view_dir: Path) -> Path:
    """Add a scenario freezing the fast channel to the render's scenarios."""
    path = view_dir / "scenarios.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["scenarios"].append(
        {"name": FROZEN_SCENARIO, "channel_faults": {CURRENT_FAST: "frozen"}}
    )
    path.write_text(json.dumps(document), encoding="utf-8")
    return view_dir


def _without_timed_classes(view_dir: Path) -> Path:
    """Rewrite every ``fast`` and ``event`` binding of the render to ``pass``."""
    path = view_dir / "variables.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    for model in document["models"]:
        for record in model["wiring"]:
            if record.get("refresh") in ("fast", "event"):
                record["refresh"] = "pass"
    path.write_text(json.dumps(document), encoding="utf-8")
    return view_dir


def _activate(state_dir: Path, *names: str) -> None:
    """Write the active set, moving the file's modification time forward."""
    path = state_dir / "active_scenarios"
    before = path.stat().st_mtime_ns if path.exists() else 0
    path.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")
    stamp = max(before + 1_000_000, path.stat().st_mtime_ns)
    os.utime(path, ns=(stamp, stamp))


# =============================================================================
# The server process
# =============================================================================


def _serve(spec: dict[str, Any], conn: Any) -> None:
    """Server process: serve one view, then answer the parent's commands."""
    os.environ.update(
        {
            "EPICS_CA_SERVER_PORT": spec["ca_port"],
            "EPICS_CAS_SERVER_PORT": spec["ca_port"],
            "EPICS_CAS_AUTO_BEACON_ADDR_LIST": "NO",
            "EPICS_CAS_BEACON_ADDR_LIST": "127.0.0.1",
            "EPICS_PVAS_SERVER_PORT": spec["pva_port"],
            "EPICS_PVAS_BROADCAST_PORT": spec["pva_broadcast"],
            "EPICS_PVAS_AUTO_BEACON_ADDR_LIST": "NO",
            "EPICS_PVAS_BEACON_ADDR_LIST": "127.0.0.1",
        }
    )
    lock = threading.Lock()
    cycles = {"total": 0, "jobs": 0, "tick": 0}
    try:
        from osprey.services.virtual_accelerator.serving.runner import ModelRunner
        from osprey_connectors.simulation.composite import Composite
        from osprey_connectors.simulation.view import SimulatorView

        view = SimulatorView.open(Path(spec["view"]))
        composite = Composite(view, state_dir=spec["state"], model_log=False, evolve=True)
        runner = ModelRunner(
            composite,
            view,
            model_write_token=None,
            tick_interval_s=TICK_S,
            fast_interval_s=FAST_S,
        )
        run_cycle = runner._run_cycle

        def counted(item: dict[str, Any]) -> None:
            with lock:
                cycles["total"] += 1
                cycles["jobs"] += bool(item.get("jobs"))
                cycles["tick"] += bool(item.get("tick"))
            run_cycle(item)

        runner._run_cycle = counted  # type: ignore[method-assign]
        threading.Thread(target=runner._run, daemon=True, name="model-runner-loop").start()
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while runner._passes_run < 1:
            if time.monotonic() > deadline:
                raise TimeoutError("the runner's first pass never finished")
            time.sleep(0.05)
    except BaseException as exc:
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
        os._exit(1)
    conn.send(("ready", None))

    def counts() -> dict[str, int]:
        with lock:
            return dict(cycles)

    commands: dict[str, Callable[..., Any]] = {
        "cycles": counts,
        "threads": lambda: sorted(thread.name for thread in threading.enumerate()),
    }
    while True:
        try:
            name, *args = conn.recv()
        except EOFError:
            os._exit(0)
        if name == "stop":
            conn.send(("ok", None))
            os._exit(0)
        try:
            conn.send(("ok", commands[name](*args)))
        except Exception as exc:
            conn.send(("error", f"{type(exc).__name__}: {exc}"))


class _Server:
    """A spawned server process and the PVAccess client that reaches it."""

    def __init__(self, spec: dict[str, Any]) -> None:
        context = multiprocessing.get_context("spawn")
        self._conn, child = context.Pipe()
        self._proc = context.Process(target=_serve, args=(spec, child), daemon=True)
        self._proc.start()
        child.close()
        # Searched over TCP, straight at this server.
        self.pva = Context(
            "pva",
            conf={
                "EPICS_PVA_NAME_SERVERS": f"127.0.0.1:{spec['pva_port']}",
                "EPICS_PVA_ADDR_LIST": "",
                "EPICS_PVA_AUTO_ADDR_LIST": "NO",
            },
            useenv=False,
            nt=False,
        )
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while not self._conn.poll(0.5):
            if not self._proc.is_alive():
                raise AssertionError(f"the server process exited ({self._proc.exitcode})")
            if time.monotonic() > deadline:
                raise AssertionError("the server process never served")
        kind, payload = self._conn.recv()
        if kind != "ready":
            raise AssertionError(f"the server process could not serve: {payload}")

    def ask(self, name: str, *args: Any) -> Any:
        """Send one command to the server process and return its answer."""
        self._conn.send((name, *args))
        if not self._conn.poll(COMMAND_TIMEOUT_S):
            raise AssertionError(f"the server process did not answer {name!r}")
        kind, payload = self._conn.recv()
        if kind != "ok":
            raise AssertionError(f"the server process failed {name!r}: {payload}")
        return payload

    def stop(self) -> None:
        self.pva.close()
        with suppress(AssertionError, OSError):
            self.ask("stop")
        self._proc.join(timeout=10)
        if self._proc.is_alive():
            self._proc.kill()
            self._proc.join(timeout=10)


@contextmanager
def _served(view_dir: Path, server: str, state_dir: Path) -> Iterator[_Server]:
    from osprey_connectors.control_system._ca_loop import get_ca_loop

    with reserved_ports(2) as (pva_port, pva_broadcast):
        spec = {
            "view": str(view_dir),
            "state": str(state_dir),
            "ca_port": CA_PORTS[server],
            "pva_port": str(pva_port),
            "pva_broadcast": str(pva_broadcast),
        }
        _on_ca_loop(get_ca_loop().point_at(CA_CLIENT_ENV))
        served = _Server(spec)
        try:
            yield served
        finally:
            served.stop()


# =============================================================================
# Client helpers
# =============================================================================


def _on_ca_loop(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run one aioca coroutine on the process's CA loop; block for its result."""
    from osprey_connectors.control_system._ca_loop import get_ca_loop

    return get_ca_loop().run_sync(coro, timeout=CA_TIMEOUT_S + 5.0)


def _wait_until(predicate: Callable[[], Any], timeout: float) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.05)
    return predicate()


class _Events:
    """When each monitor event of one channel arrived, on each wire."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._arrivals: dict[str, list[float]] = {"pva": [], "ca": []}

    def arrived(self, wire: str) -> None:
        with self._lock:
            self._arrivals[wire].append(time.monotonic())

    def arrivals(self, wire: str) -> list[float]:
        with self._lock:
            return list(self._arrivals[wire])

    def between(self, wire: str, start: float, end: float) -> int:
        return sum(start <= t < end for t in self.arrivals(wire))

    def first_after(self, wire: str, start: float) -> float | None:
        return next((t for t in self.arrivals(wire) if t >= start), None)


@contextmanager
def _monitored(server: _Server, address: str) -> Iterator[_Events]:
    """Monitor ``address`` on both wires; yields once each has delivered its connect event."""
    import aioca

    events = _Events()

    def on_pva(value: Any) -> None:
        if not isinstance(value, Exception):
            events.arrived("pva")

    async def subscribe() -> Any:
        return aioca.camonitor(address, lambda _: events.arrived("ca"), format=aioca.FORMAT_TIME)

    async def unsubscribe() -> None:
        ca_monitor.close()

    pva_monitor = server.pva.monitor(address, on_pva, notify_disconnect=False)
    ca_monitor = _on_ca_loop(subscribe())
    try:
        connected = _wait_until(
            lambda: events.arrivals("pva") and events.arrivals("ca"), timeout=CA_TIMEOUT_S
        )
        assert connected, f"{address}: a monitor never delivered its connect event"
        yield events
    finally:
        pva_monitor.close()
        _on_ca_loop(unsubscribe())


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def injection(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[_Server, Path]]:
    """The injection fixture, served with timed passes; and its state directory."""
    view_dir = _with_frozen_scenario(injection_view(tmp_path_factory.mktemp("injection")))
    state_dir = view_dir.parent / "state"
    state_dir.mkdir()
    with _served(view_dir, "injection", state_dir) as served:
        yield served, state_dir


# =============================================================================
# Tests
# =============================================================================


def test_the_fast_channel_delivers_the_fast_rate_on_both_wires_at_a_constant_value(
    injection: tuple[_Server, Path],
) -> None:
    server, _ = injection

    with _monitored(server, CURRENT_FAST) as events:
        start = time.monotonic() + FAST_S
        time.sleep(WINDOWS + 2 * FAST_S)
        counts = {
            wire: [events.between(wire, start + i, start + i + 1) for i in range(WINDOWS)]
            for wire in ("ca", "pva")
        }

    low, high = RATE
    for wire, per_window in counts.items():
        assert all(low <= count <= high for count in per_window), (wire, per_window)


def test_a_frozen_fast_channel_goes_quiet_and_resumes_once_cleared(
    injection: tuple[_Server, Path],
) -> None:
    server, state_dir = injection

    with _monitored(server, CURRENT_FAST) as events:
        _activate(state_dir, FROZEN_SCENARIO)
        try:
            time.sleep(FAULT_SETTLE_S)
            watch = time.monotonic()
            time.sleep(FROZEN_WATCH_S)
            frozen = {
                wire: events.between(wire, watch, watch + FROZEN_WATCH_S) for wire in ("ca", "pva")
            }
        finally:
            cleared = time.monotonic()
            _activate(state_dir)
        _wait_until(lambda: events.first_after("ca", cleared), timeout=CA_TIMEOUT_S)
        resumed = {wire: events.first_after(wire, cleared) for wire in ("ca", "pva")}

    assert frozen == {"ca": 0, "pva": 0}
    for wire, at in resumed.items():
        assert at is not None, wire
        assert at - cleared <= RESUME_S, (wire, at - cleared)


def test_a_view_without_timed_classes_runs_no_pass_beyond_the_global_tick(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    view_dir = _without_timed_classes(injection_view(tmp_path_factory.mktemp("quiet")))
    state_dir = view_dir.parent / "state"
    state_dir.mkdir()

    with _served(view_dir, "quiet", state_dir) as server:
        threads = server.ask("threads")
        before = server.ask("cycles")
        time.sleep(3 * TICK_S + 0.5)
        after = server.ask("cycles")

    assert SCHEDULER_THREAD not in threads
    assert after["jobs"] == before["jobs"] == 0
    assert after["total"] - before["total"] == after["tick"] - before["tick"]
    assert after["tick"] - before["tick"] <= 4


def test_the_served_injection_view_runs_its_scheduler_thread(
    injection: tuple[_Server, Path],
) -> None:
    server, _ = injection

    assert SCHEDULER_THREAD in server.ask("threads")
    assert server.ask("cycles")["jobs"] > 0


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    assert_module_collected(request, __file__, MIN_COLLECTED_TESTS)
