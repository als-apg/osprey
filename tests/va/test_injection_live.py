"""The injection chain, seen from the far side of both wires.

The virtual accelerator's own entrypoint serves the injection fixture
(``tests/facility/fixtures/injection``) in a spawned process: the evolving
composite, the model runner and both servers. A real ``aioca`` client and a
real ``p4p`` client in the pytest process write the request channel and watch
what each shot publishes: one monitor event per shot on every per-shot channel,
each inside its timing bound; the stored current, the counters and the
efficiencies a shot leaves; a request during a shot ignored without an alarm;
the free-running busy cycle; a ``disconnected`` per-shot channel, and a
scenario activated mid-shot.

**One server per process.** One module-scoped server serves the fixture with
two scenarios added to its render. A scenario change rebuilds the chain at its
start state, so the tests that change the active set run last and every other
test measures from the values it reads first.

**Timing.** The bounds scale by ``OSPREY_LIVE_CA_TIMING_SLACK`` (default 1).

**The venue is linux.** ``lume-pva-apg`` installs only there, so the module
skips whole anywhere else; ``scripts/va/live_ca/gate.py --pva`` runs it in a
process of its own and fails on any skip.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import multiprocessing
import os
import threading
import time
from collections.abc import Callable, Coroutine, Iterator, Sequence
from contextlib import ExitStack, contextmanager
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

#: Whether this is the pytest process rather than the server it spawns.
_IN_PYTEST = multiprocessing.parent_process() is None

# The server and CAS ports equal: a search reply carries the server's own port.
CA_PORT = str(hold_ports(1)[0]) if _IN_PYTEST else ""
# Handed to CaLoop.point_at before the server starts, never written at import:
# other modules collected into this process write their own.
CA_CLIENT_ENV: dict[str, str | None] = {
    "EPICS_CA_ADDR_LIST": f"127.0.0.1:{CA_PORT}",
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

from tests.facility._injection_fixture import (  # noqa: E402
    BUSY,
    CHARGES,
    COUNTERS,
    CURRENT,
    CURRENT_FAST,
    EFFICIENCIES,
    OFFSETS,
    PARAMETERS,
    READINGS,
    REQUEST,
    SETPOINTS,
    injection_view,
)
from tests.va._collection import assert_module_collected  # noqa: E402

#: Floor for this module's own test count.
MIN_COLLECTED_TESTS = 8

#: The factor on every timing bound.
SLACK = float(os.environ.get("OSPREY_LIVE_CA_TIMING_SLACK", "1.0"))

#: The served runner's global tick and fast period.
TICK_S = 1.0
FAST_S = 0.1

#: How late past its offset a step's event may reach the client.
BOUND_S = 0.1 * SLACK

#: The fixture's charge per shot, stage efficiencies and current per charge.
SOURCE_CHARGE = "INJ:GUN:CHARGE:SP"
STAGE_EFFICIENCIES = ("INJ:TRANSFER:EFF:SP", "INJ:RING:EFF:SP")
CURRENT_PER_CHARGE = 2.0  # models.yaml, settings.injection.current_per_charge

#: The fixture's busy cycle.
PERIOD_S = 1.0
WIDTH_S = 0.5

#: The channels a shot posts an event on, with the offset of each.
PER_SHOT = {address: offset for address, offset in OFFSETS.items() if address != CURRENT}
#: A shot's last step.
LAST_STEP_S = max(OFFSETS.values())

#: The ten-shot run: how many requests, and how far apart they start.
SHOTS = 10
SPACING_S = 0.25
#: The two requests a shot ignores the second of.
SECOND_REQUEST_AFTER_S = 0.05
#: Periods of the busy cycle watched.
PERIODS = 10

#: The scenarios added to the render.
DISCONNECTED_SCENARIO = "disconnected-charge"
DISCONNECTED = CHARGES[0]
MID_SHOT_SCENARIO = "mid-shot"
#: The shot's first step, whose PVAccess event says the shot started; the
#: per-shot channels whose steps come after a mid-shot activation; and the
#: latest after that event the activation may land for the test to say so.
#: The composite reads the active set before it applies a step.
FIRST = COUNTERS[0]
LATE = (CHARGES[1], EFFICIENCIES[1])
ACTIVATE_BY_S = min(OFFSETS[address] for address in LATE) - OFFSETS[FIRST] - 0.02

INSTANCE = "virtual_accelerator"
NO_ALARM = 0
INVALID = 3

CA_TIMEOUT_S = 30.0
STARTUP_TIMEOUT_S = 300.0

#: Every request writes a value of its own.
_REQUESTS = itertools.count(1)


def _step() -> float:
    """The stored current one shot adds at the fixture's start parameters."""
    charge = PARAMETERS[SOURCE_CHARGE]
    for name in STAGE_EFFICIENCIES:
        charge *= PARAMETERS[name]
    return charge * CURRENT_PER_CHARGE


def _charge_after(stage: int, efficiencies: Sequence[float]) -> float:
    charge = PARAMETERS[SOURCE_CHARGE]
    for efficiency in efficiencies[: stage + 1]:
        charge *= efficiency
    return charge


# =============================================================================
# The view
# =============================================================================


def _with_scenarios(view_dir: Path) -> Path:
    """Add a ``disconnected`` per-shot channel and an empty scenario to the render."""
    path = view_dir / "scenarios.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["scenarios"].extend(
        [
            {"name": DISCONNECTED_SCENARIO, "channel_faults": {DISCONNECTED: "disconnected"}},
            {"name": MID_SHOT_SCENARIO, "channel_faults": {}},
        ]
    )
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


def _serve(env: dict[str, str]) -> None:
    """Server process: boot the entrypoint over the injection view."""
    os.environ.update(env)
    from osprey.services.virtual_accelerator import entrypoint

    entrypoint.main()


class _Served:
    """The spawned entrypoint and the PVAccess client that reaches it."""

    def __init__(self, env: dict[str, str]) -> None:
        context = multiprocessing.get_context("spawn")
        self._proc = context.Process(target=_serve, args=(env,), daemon=True)
        self._proc.start()
        # Searched over TCP, straight at this server.
        self.pva = Context(
            "pva",
            conf={
                "EPICS_PVA_NAME_SERVERS": f"127.0.0.1:{env['EPICS_PVAS_SERVER_PORT']}",
                "EPICS_PVA_ADDR_LIST": "",
                "EPICS_PVA_AUTO_ADDR_LIST": "NO",
            },
            useenv=False,
            nt=False,
        )

    def alive(self) -> bool:
        return self._proc.is_alive()

    def stop(self) -> None:
        self.pva.close()
        self._proc.kill()
        self._proc.join(timeout=10)


# =============================================================================
# Client helpers
# =============================================================================


def _on_ca_loop(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run one aioca coroutine on the process's CA loop; block for its result."""
    from osprey_connectors.control_system._ca_loop import get_ca_loop

    return get_ca_loop().run_sync(coro, timeout=CA_TIMEOUT_S + 5.0)


def _wait_until(
    predicate: Callable[[], Any], timeout: float = CA_TIMEOUT_S, poll_s: float = 0.02
) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(poll_s)
    return predicate()


def _sleep_until(moment: float) -> None:
    time.sleep(max(0.0, moment - time.monotonic()))


def _caget(address: str, timeout: float = 2.0) -> Any:
    """One fresh read over CA with its alarm, or ``None`` if none arrived."""
    import aioca

    reading = _on_ca_loop(
        aioca.caget(address, format=aioca.FORMAT_TIME, timeout=timeout, throw=False)
    )
    return reading if reading.ok else None


def _ca(address: str) -> tuple[float, int]:
    """One read over CA: value and severity."""
    reading = _caget(address, timeout=CA_TIMEOUT_S)
    assert reading is not None, f"{address} returned nothing over CA"
    return float(reading), int(reading.severity)


def _pva(server: _Served, address: str) -> tuple[float, int]:
    """One read over PVA: value and severity."""
    value = server.pva.get(address, timeout=CA_TIMEOUT_S)
    return float(value["value"]), int(value["alarm.severity"])


def _caput(address: str, value: float, *, wait: bool = True) -> bool:
    """Write over CA; with ``wait``, True once the write's pass has finished."""
    import aioca

    done = _on_ca_loop(aioca.caput(address, value, wait=wait, timeout=CA_TIMEOUT_S, throw=False))
    return bool(done.ok)


def _request(*, wait: bool = True) -> tuple[float, float]:
    """Write the request channel; return when the put was issued and when it returned."""
    issued = time.monotonic()
    assert _caput(REQUEST, float(next(_REQUESTS)), wait=wait), "the request put failed"
    return issued, time.monotonic()


def _close(value: float, expected: float) -> bool:
    return math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-9)


class _Events:
    """Each monitor event of one channel after its connection event, on each wire."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._connected = {"pva": False, "ca": False}
        self._events: dict[str, list[tuple[float, float]]] = {"pva": [], "ca": []}

    def arrived(self, wire: str, value: float) -> None:
        at = time.monotonic()
        with self._lock:
            if self._connected[wire]:
                self._events[wire].append((at, value))
            else:
                self._connected[wire] = True

    def connected(self) -> bool:
        with self._lock:
            return all(self._connected.values())

    def events(self, wire: str) -> list[tuple[float, float]]:
        with self._lock:
            return list(self._events[wire])

    def count(self, wire: str) -> int:
        return len(self.events(wire))


@contextmanager
def _monitored(server: _Served, *addresses: str) -> Iterator[dict[str, _Events]]:
    """Monitor each address on both wires; yields once every monitor has connected."""
    import aioca

    events = {address: _Events() for address in addresses}

    async def subscribe(address: str) -> Any:
        return aioca.camonitor(
            address,
            lambda value: events[address].arrived("ca", float(value)),
            format=aioca.FORMAT_TIME,
        )

    def on_pva(address: str) -> Callable[[Any], None]:
        def arrived(value: Any) -> None:
            if not isinstance(value, Exception):
                events[address].arrived("pva", float(value["value"]))

        return arrived

    with ExitStack() as stack:
        for address in addresses:
            pva = server.pva.monitor(address, on_pva(address), notify_disconnect=False)
            stack.callback(pva.close)
            ca = _on_ca_loop(subscribe(address))

            async def unsubscribe(monitor: Any = ca) -> None:
                monitor.close()

            stack.callback(lambda close=unsubscribe: _on_ca_loop(close()))
        connected = _wait_until(lambda: all(e.connected() for e in events.values()))
        assert connected, "a monitor never delivered its connect event"
        yield events


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def served(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[_Served, Path]]:
    """The entrypoint serving the injection fixture; and its state directory."""
    root = tmp_path_factory.mktemp("injection-live")
    view_dir = _with_scenarios(injection_view(root))
    state_dir = root / "state"
    state_dir.mkdir()
    from osprey_connectors.control_system._ca_loop import get_ca_loop

    with reserved_ports(2) as (pvas_port, pvas_broadcast_port):
        env = {
            "VA_DATA_DIR": str(view_dir.parent),
            "VA_STATE_DIR": str(state_dir),
            "VA_INSTANCE": INSTANCE,
            "VA_POLL_INTERVAL_S": str(TICK_S),
            "VA_FAST_INTERVAL_S": str(FAST_S),
            "EPICS_PVAS_SERVER_PORT": str(pvas_port),
            "EPICS_PVAS_BROADCAST_PORT": str(pvas_broadcast_port),
            "EPICS_PVAS_AUTO_BEACON_ADDR_LIST": "NO",
            "EPICS_PVAS_BEACON_ADDR_LIST": "127.0.0.1",
            "EPICS_CA_SERVER_PORT": CA_PORT,
            "EPICS_CAS_SERVER_PORT": CA_PORT,
            "EPICS_CAS_AUTO_BEACON_ADDR_LIST": "NO",
            "EPICS_CAS_BEACON_ADDR_LIST": "127.0.0.1",
        }
        _on_ca_loop(get_ca_loop().point_at(CA_CLIENT_ENV))
        server = _Served(env)
        try:
            deadline = time.monotonic() + STARTUP_TIMEOUT_S
            while _caget(REQUEST) is None:
                if not server.alive():
                    pytest.fail("the entrypoint exited before it served")
                if time.monotonic() > deadline:
                    pytest.fail("the entrypoint never served")
            yield server, state_dir
        finally:
            server.stop()


# =============================================================================
# Tests -- the scenario tests last: a change of the active set rebuilds the chain
# =============================================================================


def test_every_channel_reads_on_both_wires_and_every_parameter_at_its_start_value(
    served: tuple[_Served, Path],
) -> None:
    server, _ = served

    for address in (*SETPOINTS, *READINGS):
        ca = _ca(address)
        pva = _pva(server, address)
        if address in PARAMETERS:
            assert _close(ca[0], PARAMETERS[address]), (address, ca)
            assert _close(pva[0], PARAMETERS[address]), (address, pva)


def test_ten_shots_post_ten_events_per_channel_in_bound_and_leave_ten_steps(
    served: tuple[_Served, Path],
) -> None:
    server, _ = served
    before = {address: _ca(address)[0] for address in (*COUNTERS, CURRENT, CURRENT_FAST)}

    with _monitored(server, *PER_SHOT) as events:
        puts: list[tuple[float, float]] = []
        first = time.monotonic()
        for shot in range(SHOTS):
            if puts:
                _sleep_until(max(first + shot * SPACING_S, puts[-1][1] + LAST_STEP_S + BOUND_S))
            puts.append(_request())
            landed = _wait_until(
                lambda n=shot + 1: all(e.count("ca") >= n for e in events.values())
            )
            assert landed, f"shot {shot + 1} never posted on every channel"
        # A global tick alone posts nothing on a per-shot channel.
        time.sleep(TICK_S + 2 * BOUND_S)
        seen = {address: {w: e.events(w) for w in ("ca", "pva")} for address, e in events.items()}

    assert puts[-1][0] - puts[0][0] >= 2 * TICK_S
    for address, wires in seen.items():
        assert {wire: len(got) for wire, got in wires.items()} == {"ca": SHOTS, "pva": SHOTS}, (
            address
        )
        offset = PER_SHOT[address]
        for (issued, done), (at, _) in zip(puts, wires["ca"], strict=True):
            assert issued + offset <= at <= done + offset + BOUND_S, (
                address,
                at - issued,
                done - issued,
            )

    step = _step()
    for address in (CURRENT, CURRENT_FAST):
        expected = before[address] + SHOTS * step
        assert _wait_until(lambda a=address, x=expected: _close(_ca(a)[0], x)), address
        assert _close(_pva(server, address)[0], expected), address
    for address in COUNTERS:
        assert _ca(address)[0] == before[address] + SHOTS, address
    for address, parameter in zip(EFFICIENCIES, STAGE_EFFICIENCIES, strict=True):
        assert _close(_ca(address)[0], PARAMETERS[parameter]), address


def test_an_efficiency_a_client_writes_shows_in_the_next_shot(
    served: tuple[_Served, Path],
) -> None:
    server, _ = served
    parameter, reading = STAGE_EFFICIENCIES[1], EFFICIENCIES[1]
    written = 0.5

    with _monitored(server, reading, CHARGES[1]) as events:
        assert _caput(parameter, written)
        try:
            _request()
            landed = _wait_until(
                lambda: all(e.count(w) >= 1 for e in events.values() for w in ("ca", "pva"))
            )
        finally:
            assert _caput(parameter, PARAMETERS[parameter])
        time.sleep(LAST_STEP_S + BOUND_S)

    assert landed, "the shot never posted"
    efficiencies = [PARAMETERS[STAGE_EFFICIENCIES[0]], written]
    for wire in ("ca", "pva"):
        assert [value for _, value in events[reading].events(wire)] == [written], wire
        charge = [value for _, value in events[CHARGES[1]].events(wire)]
        assert len(charge) == 1, wire
        assert _close(charge[0], _charge_after(1, efficiencies)), wire


def test_a_request_50_ms_into_a_shot_is_ignored_and_completes_without_an_alarm(
    served: tuple[_Served, Path],
) -> None:
    server, _ = served
    before = _ca(COUNTERS[0])[0]

    with _monitored(server, *PER_SHOT) as events:
        issued, _ = _request(wait=False)
        _sleep_until(issued + SECOND_REQUEST_AFTER_S)
        second, completed = _request()
        time.sleep(LAST_STEP_S + TICK_S)
        counts = {address: {w: e.count(w) for w in ("ca", "pva")} for address, e in events.items()}

    assert second - issued < LAST_STEP_S, "the second request missed the first shot"
    assert completed >= second
    assert counts == {address: {"ca": 1, "pva": 1} for address in PER_SHOT}
    assert _ca(COUNTERS[0])[0] == before + 1
    assert _ca(REQUEST)[1] == NO_ALARM


def test_the_busy_cycle_holds_its_period_and_width_over_ten_periods(
    served: tuple[_Served, Path],
) -> None:
    server, _ = served

    with _monitored(server, BUSY) as events:
        time.sleep((PERIODS + 1) * PERIOD_S + WIDTH_S + BOUND_S)
        edges = events[BUSY].events("ca")

    # From the first rise on; each event is an edge, so the levels alternate.
    while edges and edges[0][1] != 1.0:
        edges.pop(0)
    levels = [value for _, value in edges]
    assert levels == [1.0 - (i % 2) for i in range(len(levels))], levels
    rises = [at for at, value in edges if value == 1.0]
    falls = [at for at, value in edges if value == 0.0]
    assert len(rises) >= PERIODS + 1, len(rises)
    periods = [later - earlier for earlier, later in itertools.pairwise(rises[: PERIODS + 1])]
    widths = [fall - rise for rise, fall in zip(rises[:PERIODS], falls, strict=False)]
    assert all(abs(period - PERIOD_S) <= BOUND_S for period in periods), periods
    assert len(widths) == PERIODS
    assert all(abs(width - WIDTH_S) <= BOUND_S for width in widths), widths


def test_a_disconnected_per_shot_channel_reads_invalid_on_ca_and_recovers(
    served: tuple[_Served, Path],
) -> None:
    _, state_dir = served

    _activate(state_dir, DISCONNECTED_SCENARIO)
    try:
        faulted = _wait_until(lambda: _ca(DISCONNECTED)[1] == INVALID)
        severity = _ca(DISCONNECTED)[1]
    finally:
        _activate(state_dir)

    assert faulted, severity
    assert _wait_until(lambda: _ca(DISCONNECTED)[1] == NO_ALARM)


def test_a_scenario_activated_mid_shot_ends_that_shots_events(
    served: tuple[_Served, Path],
) -> None:
    server, state_dir = served
    # What each late step would post, at the fixture's start parameters.
    efficiencies = [PARAMETERS[name] for name in STAGE_EFFICIENCIES]
    step_values = {CHARGES[1]: _charge_after(1, efficiencies), EFFICIENCIES[1]: efficiencies[1]}

    with _monitored(server, FIRST, *LATE) as events:
        # Not waiting on put completion: a Channel Access put completes once
        # the shot's steps are on the wire, PVAccess delivers the first step
        # as it happens.
        _request(wait=False)
        _wait_until(lambda: events[FIRST].count("pva"), timeout=1.0, poll_s=0.001)
        _activate(state_dir, MID_SHOT_SCENARIO)
        activated = time.monotonic()
        try:
            time.sleep(LAST_STEP_S + TICK_S)
        finally:
            _activate(state_dir)
        seen = {address: {w: events[address].events(w) for w in ("ca", "pva")} for address in LATE}

    started = events[FIRST].events("pva")
    assert started, "the shot never started"
    assert activated - started[0][0] < ACTIVATE_BY_S, "the scenario landed after the late steps"
    for address, wires in seen.items():
        for wire, got in wires.items():
            assert not [v for _, v in got if _close(v, step_values[address])], (address, wire, got)


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    assert_module_collected(request, __file__, MIN_COLLECTED_TESTS)
