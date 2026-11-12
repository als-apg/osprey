"""One armed turn-by-turn shot, seen from the far side of both wires.

The virtual accelerator's own entrypoint serves the turn-by-turn fixture
(``tests/facility/fixtures/turn_by_turn``) in a spawned process: the injection
chain feeding the ring on pyAT, the model runner and both servers. A real
``aioca`` client and a real ``p4p`` client in the pytest process arm every
monitor over the wire, request a shot, wait for armed 1 then 0, and read the
x, y and sum arrays over Channel Access and over PVAccess.

**At default settings** the armed drop lands within 2 s of the shot, every
array's update reaches each wire before any monitor's armed-0 update, and
both wires read the same arrays.

**With the particle count raised** a shot is tracked for a few seconds. While
it is, a read returns the last value at once; a write, a model RPC call and a
response-matrix request are delayed by at most the tracking time and succeed;
and the injection chain's records post again afterwards.

**Two servers, one after the other.** Each class serves its own render on its
own Channel Access port; repointing the client at the second port gives it a
fresh context, so no channel searches for a server that has gone.

**Timing.** The bounds scale by ``OSPREY_LIVE_CA_TIMING_SLACK`` (default 1).

**The venue is linux.** ``lume-pva-apg`` installs only there, so the module
skips whole anywhere else.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Coroutine, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
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

#: Whether this is the pytest process rather than a server it spawns.
_IN_PYTEST = multiprocessing.parent_process() is None

# One CA port per server; the server and CAS ports equal, since a search reply
# carries the server's own port.
_CA_PORTS = [str(port) for port in hold_ports(2)] if _IN_PYTEST else ["", ""]


def _ca_client_env(port: str) -> dict[str, str | None]:
    """The client settings that reach the server on ``port`` and nothing else."""
    return {
        "EPICS_CA_ADDR_LIST": f"127.0.0.1:{port}",
        "EPICS_CA_AUTO_ADDR_LIST": "NO",
        "EPICS_CA_SERVER_PORT": None,
        "EPICS_CA_NAME_SERVERS": None,
    }


# import-time required because libca reads the repeater port when the first CA
# context in the process is created, which may be another module's.
if _IN_PYTEST and "EPICS_CA_REPEATER_PORT" not in os.environ:
    os.environ["EPICS_CA_REPEATER_PORT"] = str(hold_ports(1)[0])

if importlib.util.find_spec("pcaspy") is None:
    pytest.skip(
        "the Channel Access server ships with lume-pva-apg's ca extra",
        allow_module_level=True,
    )

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from p4p.client.thread import Context  # noqa: E402
from p4p.util import ThreadedWorkQueue  # noqa: E402

from osprey.services.virtual_accelerator.serving.model_rpc import (  # noqa: E402
    RPC_PV,
    build_request,
    parse_reply,
)
from tests.facility._turn_by_turn_fixture import (  # noqa: E402
    ARMED,
    ARMS,
    CURRENT,
    FIXTURE,
    MONITORS,
    REQUEST,
    RING,
    SAMPLES,
    SHOT,
    STORED,
    TURN_SUM,
    TURN_X,
    TURN_Y,
    TURNS,
    TURNS_TRACKED,
    turn_by_turn_view,
)
from tests.va._collection import assert_module_collected  # noqa: E402

#: Floor for this module's own test count.
MIN_COLLECTED_TESTS = 9

#: The factor on every timing bound.
SLACK = float(os.environ.get("OSPREY_LIVE_CA_TIMING_SLACK", "1.0"))

#: The served runner's global tick and fast period.
TICK_S = 1.0
FAST_S = 0.1

#: The ring shot counter's offset into a shot (models.yaml): the moment the
#: shot lands on the ring and its tracking starts.
LAND_S = 0.06

#: How long after the shot lands armed may drop at default settings.
ARMED_DROP_S = 2.0 * SLACK

#: The fixture's trigger sample: the first sample of the first turn.
TRIGGER_SAMPLE = SAMPLES - TURNS

#: The particles of the slow server's shot: a few seconds of tracking.
SLOW_PARTICLES = 96
#: The shortest tracking that makes the slow test say anything.
MIN_TRACKING_S = 1.0
#: How far into the tracking the calls during it are made.
INTO_TRACKING_S = 0.3
#: How long a read during tracking may take.
READ_S = 0.5 * SLACK
#: How late past the tracking a delayed call may still complete.
LATE_BOUND_S = 0.5 * SLACK

#: The injection chain's busy cycle and its counter of every shot.
BUSY = "INJ:SHOT:BUSY"
PERIOD_S = 1.0
WIDTH_S = 0.5

#: What the slow test writes during tracking.
WRITTEN = 42.0

#: Two correctors the response-matrix request asks about, and the step the
#: render's measurement record states (the fixture has no measurement file).
CORRECTORS = ("SR:MAG:HCM:01:CURRENT:SP", "SR:MAG:VCM:01:CURRENT:SP")
CORRECTOR_DELTA = 1.0e-5

#: Every array a shot publishes.
ARRAYS = (*TURN_X.values(), *TURN_Y.values(), *TURN_SUM.values())

INSTANCE = "virtual_accelerator"

CA_TIMEOUT_S = 30.0
STARTUP_TIMEOUT_S = 300.0
SHOT_TIMEOUT_S = 60.0 * SLACK

#: Every request writes a value of its own.
_REQUESTS = itertools.count(1)


# =============================================================================
# The renders
# =============================================================================


def _slow_view(root: Path) -> Path:
    """The fixture rendered with :data:`SLOW_PARTICLES` and a corrector step.

    The same three calls as :func:`turn_by_turn_view`, over a copy whose ring
    tracks more particles; the render's facility file gains the measurement
    step a response-matrix request is solved at.
    """
    from osprey.facility import FACILITY_FILE
    from osprey.facility.build import build_facility
    from osprey.facility.render import render_facility_outputs
    from tests.facility.served_tree import IN_PROCESS_RENDER_CONFIG

    facility_dir = root / "data" / "facility"
    shutil.copytree(FIXTURE, facility_dir)
    models_path = facility_dir / "models.yaml"
    models = yaml.safe_load(models_path.read_text(encoding="utf-8"))
    (ring,) = [model for model in models if model["name"] == RING]
    ring["settings"]["pyat"]["turn_by_turn"]["particles"] = SLOW_PARTICLES
    models_path.write_text(yaml.safe_dump(models, sort_keys=False), encoding="utf-8")

    render_dir = root / "build"
    render_dir.mkdir(parents=True)
    document = build_facility(facility_dir, project_name="turn-by-turn-slow")
    render_facility_outputs(
        render_dir, document, IN_PROCESS_RENDER_CONFIG, facility_dir, omitted_reported=set()
    )

    facility_path = render_dir / FACILITY_FILE
    facility = json.loads(facility_path.read_text(encoding="utf-8"))
    (record,) = [model for model in facility["models"] if model.get("name") == RING]
    record["measurement"] = {"corrector_delta": CORRECTOR_DELTA}
    facility_path.write_text(json.dumps(facility), encoding="utf-8")
    return render_dir / "data" / "simulator"


# =============================================================================
# The server process
# =============================================================================


def _serve(env: dict[str, str]) -> None:
    """Server process: boot the entrypoint over the turn-by-turn view."""
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


@contextmanager
def _serving(root: Path, view_dir: Path, ca_port: str) -> Iterator[_Served]:
    """The entrypoint serving ``view_dir`` on ``ca_port``, once it answers."""
    from osprey_connectors.control_system._ca_loop import get_ca_loop

    state_dir = root / "state"
    state_dir.mkdir()
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
            "EPICS_CA_SERVER_PORT": ca_port,
            "EPICS_CAS_SERVER_PORT": ca_port,
            "EPICS_CAS_AUTO_BEACON_ADDR_LIST": "NO",
            "EPICS_CAS_BEACON_ADDR_LIST": "127.0.0.1",
        }
        # A port of its own: the client gets a fresh context, not channels
        # still searching for the previous server.
        _on_ca_loop(get_ca_loop().point_at(_ca_client_env(ca_port)))
        server = _Served(env)
        try:
            deadline = time.monotonic() + STARTUP_TIMEOUT_S
            while _caget(REQUEST) is None:
                if not server.alive():
                    pytest.fail("the entrypoint exited before it served")
                if time.monotonic() > deadline:
                    pytest.fail("the entrypoint never served")
            yield server
        finally:
            server.stop()


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
    """One fresh read over CA, or ``None`` if none arrived."""
    import aioca

    reading = _on_ca_loop(
        aioca.caget(address, format=aioca.FORMAT_TIME, timeout=timeout, throw=False)
    )
    return reading if reading.ok else None


def _ca(address: str) -> Any:
    """One read over CA: a float, or a copied array."""
    reading = _caget(address, timeout=CA_TIMEOUT_S)
    assert reading is not None, f"{address} returned nothing over CA"
    return _plain(reading)


def _pva(server: _Served, address: str) -> Any:
    """One read over PVA: a float, or a copied array."""
    return _plain(server.pva.get(address, timeout=CA_TIMEOUT_S)["value"])


def _plain(value: Any) -> Any:
    array = np.array(value, dtype=float)
    return float(array) if array.ndim == 0 else array


def _caput(addresses: str | list[str], value: float, *, wait: bool = True) -> bool:
    """Write over CA, every address at once; with ``wait``, True once every write's
    pass has finished."""
    import asyncio

    import aioca

    names = [addresses] if isinstance(addresses, str) else addresses

    async def put_all() -> list[Any]:
        return await asyncio.gather(
            *(
                aioca.caput(name, value, wait=wait, timeout=CA_TIMEOUT_S, throw=False)
                for name in names
            )
        )

    return all(result.ok for result in _on_ca_loop(put_all()))


def _request() -> float:
    """Write the request channel without waiting on the shot; when it was issued."""
    issued = time.monotonic()
    assert _caput(REQUEST, float(next(_REQUESTS)), wait=False), "the request put failed"
    return issued


@dataclass(frozen=True)
class _Event:
    """One monitor update: its order on its wire, when it arrived, and its value."""

    seq: int
    at: float
    value: Any


@dataclass
class _Wire:
    """Every update of one wire, numbered in arrival order across its channels."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    count: Iterator[int] = field(default_factory=itertools.count)


class _Events:
    """Each monitor update of one channel after its connection update, on each wire."""

    def __init__(self, wires: dict[str, _Wire]) -> None:
        self._wires = wires
        self._lock = threading.Lock()
        self._connected = {"pva": False, "ca": False}
        self._events: dict[str, list[_Event]] = {"pva": [], "ca": []}

    def arrived(self, wire: str, value: Any) -> None:
        with self._wires[wire].lock:
            event = _Event(next(self._wires[wire].count), time.monotonic(), _plain(value))
        with self._lock:
            if self._connected[wire]:
                self._events[wire].append(event)
            else:
                self._connected[wire] = True

    def connected(self) -> bool:
        with self._lock:
            return all(self._connected.values())

    def events(self, wire: str, after: float = 0.0) -> list[_Event]:
        with self._lock:
            return [event for event in self._events[wire] if event.at >= after]


def _unsubscribe(monitor: Any) -> None:
    """Close one aioca monitor on the CA loop."""

    async def close() -> None:
        monitor.close()

    _on_ca_loop(close())


@contextmanager
def _monitored(server: _Served, *addresses: str) -> Iterator[dict[str, _Events]]:
    """Monitor each address on both wires; yields once every monitor has connected.

    Every PVAccess callback runs on one worker, and every Channel Access
    callback on the one CA loop, so each wire's sequence numbers are the order
    its updates arrived in, across channels.
    """
    import aioca

    wires = {"ca": _Wire(), "pva": _Wire()}
    events = {address: _Events(wires) for address in addresses}
    queue = ThreadedWorkQueue(name="turn-by-turn-live", daemon=True, maxsize=0).start()

    async def subscribe(address: str) -> Any:
        return aioca.camonitor(
            address,
            lambda value: events[address].arrived("ca", value),
            format=aioca.FORMAT_TIME,
            all_updates=True,
        )

    def on_pva(address: str) -> Callable[[Any], None]:
        def arrived(value: Any) -> None:
            if not isinstance(value, Exception):
                events[address].arrived("pva", value["value"])

        return arrived

    with ExitStack() as stack:
        stack.callback(queue.stop)
        for address in addresses:
            pva = server.pva.monitor(address, on_pva(address), notify_disconnect=False, queue=queue)
            stack.callback(pva.close)
            stack.callback(_unsubscribe, _on_ca_loop(subscribe(address)))
        connected = _wait_until(lambda: all(e.connected() for e in events.values()))
        assert connected, "a monitor never delivered its connect event"
        yield events


def _armed(events: dict[str, _Events], level: float, wires: tuple[str, ...], after: float) -> bool:
    """Whether every monitor's armed reading has posted ``level`` on each wire since ``after``."""
    return all(
        any(event.value == level for event in events[ARMED[monitor]].events(wire, after))
        for monitor in MONITORS
        for wire in wires
    )


def _arm_every_monitor(events: dict[str, _Events], wires: tuple[str, ...]) -> None:
    """Arm every monitor over CA and wait for each armed reading to post 1."""
    since = time.monotonic()
    assert _caput(list(ARMS.values()), 1.0), "an arm put failed"
    assert _wait_until(lambda: _armed(events, 1.0, wires, since)), "a monitor never read armed"


# =============================================================================
# The shot at default settings
# =============================================================================


@dataclass(frozen=True)
class _Shot:
    """What one armed shot at default settings posted, read after it."""

    issued: float
    events: dict[str, dict[str, list[_Event]]]
    ca: dict[str, Any]
    pva: dict[str, Any]


class TestOneArmedShotAtDefaultSettings:
    """Arm every monitor, request a shot, watch armed 1 → 0, read the arrays."""

    @pytest.fixture(scope="class")
    def shot(self, tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Shot]:
        root = tmp_path_factory.mktemp("turn-by-turn-live")
        view_dir = turn_by_turn_view(root)
        with (
            _serving(root, view_dir, _CA_PORTS[0]) as server,
            _monitored(server, *ARMED.values(), *ARRAYS) as events,
        ):
            _arm_every_monitor(events, ("ca", "pva"))
            issued = _request()
            dropped = _wait_until(
                lambda: _armed(events, 0.0, ("ca", "pva"), issued), timeout=SHOT_TIMEOUT_S
            )
            assert dropped, "armed never returned to 0 on every monitor"
            # Every update the shot posts is in; nothing more is expected.
            time.sleep(TICK_S)
            seen = {
                address: {wire: e.events(wire, issued) for wire in ("ca", "pva")}
                for address, e in events.items()
            }
            read = (*ARRAYS, TURNS_TRACKED)
            ca = {address: _ca(address) for address in read}
            pva = {address: _pva(server, address) for address in read}
        yield _Shot(issued=issued, events=seen, ca=ca, pva=pva)

    def test_armed_reads_1_then_0_on_every_monitor_over_both_wires(self, shot: _Shot) -> None:
        for monitor in MONITORS:
            for wire in ("ca", "pva"):
                levels = [event.value for event in shot.events[ARMED[monitor]][wire]]
                assert levels == [0.0], (monitor, wire, levels)

    def test_armed_drops_within_2_s_of_the_shot_landing(self, shot: _Shot) -> None:
        landed = shot.issued + LAND_S
        for monitor in MONITORS:
            for wire in ("ca", "pva"):
                (drop,) = shot.events[ARMED[monitor]][wire]
                assert drop.at - landed <= ARMED_DROP_S, (monitor, wire, drop.at - landed)

    def test_every_array_update_reaches_each_wire_before_any_armed_0_update(
        self, shot: _Shot
    ) -> None:
        for wire in ("ca", "pva"):
            first_drop = min(shot.events[ARMED[monitor]][wire][0].seq for monitor in MONITORS)
            for address in ARRAYS:
                updates = shot.events[address][wire]
                assert updates, (address, wire, "no update")
                assert updates[-1].seq < first_drop, (address, wire)

    def test_the_arrays_read_the_same_over_ca_and_pva_and_carry_the_shot(self, shot: _Shot) -> None:
        assert shot.ca[TURNS_TRACKED] == shot.pva[TURNS_TRACKED] == float(TURNS)
        for address in ARRAYS:
            ca, pva = shot.ca[address], shot.pva[address]
            assert ca.shape == pva.shape == (SAMPLES,), address
            assert np.array_equal(ca, pva), address
            for wire in ("ca", "pva"):
                assert np.array_equal(shot.events[address][wire][-1].value, ca), (address, wire)
        for monitor in MONITORS:
            # The shot's charge (0.72 per shot) reaches every monitor's sum.
            assert np.max(shot.ca[TURN_SUM[monitor]][TRIGGER_SAMPLE:]) > 0.5, monitor


# =============================================================================
# A slow shot: what the served loop does while it tracks
# =============================================================================


@dataclass
class _Call:
    """One call made during tracking: when it started and ended, and its outcome."""

    started: float = 0.0
    done: float = 0.0
    result: Any = None
    error: BaseException | None = None

    def run(self, call: Callable[[], Any]) -> threading.Thread:
        def target() -> None:
            self.started = time.monotonic()
            try:
                self.result = call()
            except BaseException as exc:  # reported by the test
                self.error = exc
            finally:
                self.done = time.monotonic()

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        return thread


@dataclass(frozen=True)
class _SlowShot:
    """What happened around one shot tracked for a few seconds."""

    landed: float
    tracked: float
    read_started: float
    read_done: float
    armed_read: float
    sum_before: Any
    sum_read: Any
    current_before: float
    current_read: float
    write: _Call
    stored_after: float
    rpc: _Call
    matrix: _Call
    busy: list[_Event]
    shots_before: float
    shots_after: float


def _response_matrix(render_root: Path) -> dict[str, Any] | None:
    """A trajectory response request, solved as the tool solves it: in a child."""
    from osprey.simulation.model_request import child_script, resolve_model_request, result_line

    request = resolve_model_request(
        RING, list(CORRECTORS), render_root=render_root, kind="trajectory"
    )
    done = subprocess.run(
        [sys.executable, "-c", child_script(request)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return result_line(done.stdout)


class TestASlowShot:
    """A shot tracked for seconds delays the served loop; it never refuses it."""

    @pytest.fixture(scope="class")
    def slow(self, tmp_path_factory: pytest.TempPathFactory) -> Iterator[_SlowShot]:
        root = tmp_path_factory.mktemp("turn-by-turn-slow")
        view_dir = _slow_view(root)
        render_root = view_dir.parent.parent
        first_sum = TURN_SUM[MONITORS[0]]
        with (
            _serving(root, view_dir, _CA_PORTS[1]) as server,
            _monitored(server, *ARMED.values(), first_sum, BUSY, SHOT) as events,
        ):
            _arm_every_monitor(events, ("ca",))
            sum_before = _ca(first_sum)
            current_before = _ca(CURRENT)
            shots_before = _ca(SHOT)

            issued = _request()
            landed = issued + LAND_S
            _sleep_until(landed + INTO_TRACKING_S)
            read_started = time.monotonic()
            armed_read = _ca(ARMED[MONITORS[0]])
            sum_read = _ca(first_sum)
            current_read = _ca(CURRENT)
            read_done = time.monotonic()

            write, rpc, matrix = _Call(), _Call(), _Call()
            threads = [
                write.run(lambda: _caput(STORED[0], WRITTEN)),
                rpc.run(
                    lambda: parse_reply(
                        server.pva.rpc(RPC_PV, build_request("status"), timeout=CA_TIMEOUT_S)
                    )
                ),
                matrix.run(lambda: _response_matrix(render_root)),
            ]

            arrived = _wait_until(lambda: events[first_sum].events("ca", issued), SHOT_TIMEOUT_S)
            assert arrived, "the slow shot's arrays never arrived"
            tracked = arrived[0].at
            for thread in threads:
                thread.join(timeout=600)
            dropped = _wait_until(lambda: _armed(events, 0.0, ("ca",), issued))
            assert dropped, "armed never returned to 0 after the slow shot"
            stored_after = _ca(STORED[0])

            # The chain carries on: its busy cycle, and a shot of an unarmed ring.
            _sleep_until(tracked + 2 * PERIOD_S + WIDTH_S)
            again = _request()
            assert _wait_until(lambda: events[SHOT].events("ca", again)), "no second shot"
            shots_after = _ca(SHOT)
            busy = events[BUSY].events("ca", issued)

        yield _SlowShot(
            landed=landed,
            tracked=tracked,
            read_started=read_started,
            read_done=read_done,
            armed_read=armed_read,
            sum_before=sum_before,
            sum_read=sum_read,
            current_before=current_before,
            current_read=current_read,
            write=write,
            stored_after=stored_after,
            rpc=rpc,
            matrix=matrix,
            busy=busy,
            shots_before=shots_before,
            shots_after=shots_after,
        )

    def test_the_shot_tracks_long_enough_to_say_anything(self, slow: _SlowShot) -> None:
        assert slow.tracked - slow.landed >= MIN_TRACKING_S, slow.tracked - slow.landed
        assert slow.read_done < slow.tracked, "the reads were not made during tracking"

    def test_a_read_during_tracking_returns_the_last_value_at_once(self, slow: _SlowShot) -> None:
        assert slow.read_done - slow.read_started <= READ_S, slow.read_done - slow.read_started
        assert slow.armed_read == 1.0
        assert np.array_equal(slow.sum_read, slow.sum_before)
        assert slow.current_read == slow.current_before

    def test_a_write_a_model_call_and_a_response_request_succeed_late(
        self, slow: _SlowShot
    ) -> None:
        tracking = slow.tracked - slow.landed
        for name, call in (("write", slow.write), ("rpc", slow.rpc), ("matrix", slow.matrix)):
            assert call.error is None, (name, call.error)
            assert call.done > 0.0, (name, "never finished")
        for name, call in (("write", slow.write), ("rpc", slow.rpc)):
            assert call.done - call.started <= tracking + LATE_BOUND_S, (
                name,
                call.done - call.started,
                tracking,
            )
        assert slow.write.result is True
        assert slow.stored_after == WRITTEN
        assert isinstance(slow.rpc.result, dict)
        matrix = slow.matrix.result
        assert matrix is not None and "refused" not in matrix, matrix
        assert matrix["columns"] == list(CORRECTORS)
        assert np.asarray(matrix["matrix"]).shape == (len(matrix["rows"]), len(CORRECTORS))

    def test_the_injection_chain_posts_again_after_tracking(self, slow: _SlowShot) -> None:
        tracking = slow.tracked - slow.landed
        times = [event.at for event in slow.busy]
        after = [at for at in times if at > slow.tracked]
        assert len(after) >= 4, after
        assert after[0] - slow.tracked <= WIDTH_S + LATE_BOUND_S, after[0] - slow.tracked
        gaps = [later - earlier for earlier, later in itertools.pairwise(times)]
        assert max(gaps) <= tracking + WIDTH_S + LATE_BOUND_S, (gaps, tracking)
        assert slow.shots_after == slow.shots_before + 2


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    assert_module_collected(request, __file__, MIN_COLLECTED_TESTS)
