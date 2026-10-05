"""The stuck-setpoint scenario fault, as the composite holds it and as a client sees it.

A scenario's ``faults`` block marks a setpoint ``stuck`` with the literal
string as the fault value. While that scenario is active the composite accepts
a write to the setpoint and does not forward it, so the device behind it never
moves: its paired readback holds whatever the device last delivered. The point
of the fault is that it is a property of the served machine rather than of one
client's view -- every client reading the device sees the same frozen
readback.

Two halves:

* :class:`TestStuckSetpointInTheComposite` builds the composite over a stub
  view in process and needs no server, so it runs everywhere;
* the live classes boot the virtual accelerator's own entrypoint over the same
  view in a spawned process -- the composite, the model runner and both
  servers -- and drive it with a real ``pyepics`` client and a real ``p4p``
  client from the pytest process. The serving stack installs on linux/x86_64
  only, so they skip on a developer's Mac and run in the live venue
  (``scripts/va/live_ca/gate.py``) and on CI.

The stub engine is the one physics child of the view: its readback reads the
input its paired setpoint was last given, so a readback that moves is a write
that reached the model, and one that holds is a write that did not.

The address set here is disjoint from every other module's, so two servers
alive in one pytest session can never answer for each other's names.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import socket
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _free_port() -> str:
    """An unused loopback TCP port, as a string ready for the environment."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return str(probe.getsockname()[1])


# import-time required because libca latches the EPICS_CA_* environment when
# the C library initialises, which happens on the first `import epics` anywhere
# in the process. Loopback only, on an ephemeral port unless the environment
# pins one, with the server and CAS ports equal -- a search reply carries the
# server's own port, so a server listening anywhere else hands clients a dead
# address. The spawned server inherits the same values.
os.environ.setdefault("EPICS_CA_ADDR_LIST", "127.0.0.1")
os.environ.setdefault("EPICS_CA_AUTO_ADDR_LIST", "NO")
os.environ.setdefault("EPICS_CA_SERVER_PORT", _free_port())
os.environ.setdefault("EPICS_CAS_SERVER_PORT", os.environ["EPICS_CA_SERVER_PORT"])
os.environ.setdefault("EPICS_CA_REPEATER_PORT", _free_port())

# Floor for this module's own test count -- a guard against a refactor that
# leaves the file importable but empty, which would otherwise pass silently.
MIN_COLLECTED_TESTS = 15

CA_TIMEOUT_S = 30.0
SETTLE_TIMEOUT_S = 60.0
#: How long the spawned entrypoint may take to serve.
STARTUP_TIMEOUT_S = 300.0
#: The period of the served runner's own passes.
TICK_S = 0.2

CODE = "ZZAF"
MODEL = "M"
STUB_ENGINE = "zz-apply-fault-echo"
STUCK_SCENARIO = "zz-af-stuck"
INSTANCE = "virtual_accelerator"

# A faulted pair and an unfaulted one, so the fault can be shown to be per
# channel rather than a model-wide switch.
STUCK_SP = f"{CODE}:RF:CAV:01:VOLTAGE:SP"
STUCK_RB = f"{CODE}:RF:CAV:01:VOLTAGE:RB"
LIVE_SP = f"{CODE}:RF:CAV:02:VOLTAGE:SP"
LIVE_RB = f"{CODE}:RF:CAV:02:VOLTAGE:RB"
PAIRS = {STUCK_RB: STUCK_SP, LIVE_RB: LIVE_SP}

# The setpoints boot at a nonzero value on purpose: a frozen readback holding
# zero is indistinguishable from an unseeded one, so "never moved" would be a
# claim the test could not actually make.
BOOT = 2.5
BAND = [-100.0, 100.0]


# =============================================================================
# The stub engine: one echo child, loaded by name
# =============================================================================


def _echo_model_class() -> type:
    """The stub's LUME model class, defined where LUME is imported."""
    from lume.model import LUMEModel
    from lume.variables import ScalarVariable

    class EchoModel(LUMEModel):
        """Each readback reads the input its paired setpoint was last given."""

        def __init__(self, wiring: list[Mapping[str, Any]], active: Mapping[str, Any]) -> None:
            self._variables: dict[str, Any] = {}
            self._defaults: dict[str, float] = {}
            for entry in wiring:
                address = str(entry["address"])
                writes = entry.get("direction") == "write"
                self._variables[address] = ScalarVariable(name=address, read_only=not writes)
                if writes:
                    self._defaults[address] = float(entry.get("default", 0.0))
            self._defaults.update(
                {name: float(value) for name, value in active.items() if name in self._defaults}
            )
            self.inputs: dict[str, float] = {}
            self.written: list[str] = []
            self.reset()

        @property
        def supported_variables(self) -> dict[str, Any]:
            return self._variables

        def reset(self) -> None:
            self.inputs = dict(self._defaults)

        def _set(self, values: dict[str, Any]) -> None:
            self.written.extend(values)
            self.inputs.update({name: float(value) for name, value in values.items()})

        def _get(self, names: list[str]) -> dict[str, Any]:
            return {name: self.inputs[PAIRS.get(name, name)] for name in names}

    return EchoModel


#: Every echo model built in this process, newest last.
BUILT: list[Any] = []


def _echo_build(model: str, wiring: Any, deck: Any, settings: Any, active: Any = None) -> Any:
    del model, deck, settings
    built = _echo_model_class()(list(wiring), dict(active or {}))
    BUILT.append(built)
    return built


ECHO = SimpleNamespace(build=_echo_build, error_text=lambda exc: str(exc))


def _with_echo_engine() -> None:
    """Make the composite load :data:`ECHO` for :data:`STUB_ENGINE`."""
    from osprey_connectors.simulation.composite import Composite

    real = Composite._engine

    def engine(name: str) -> Any:
        return ECHO if name == STUB_ENGINE else real(name)

    Composite._engine = staticmethod(engine)  # type: ignore[method-assign]


# =============================================================================
# The view
# =============================================================================


def _channel(address: str, role: str) -> dict[str, Any]:
    return {
        "address": address,
        "role": role,
        "pair": None,
        "value_type": "float",
        "unit": None,
        "description": None,
        "writable": role == "setpoint",
        "value_range": BAND if role == "setpoint" else None,
        "owner": MODEL,
    }


def _write_view(data_dir: Path) -> Path:
    """The stub view under ``data_dir/simulator``; returns the view directory."""
    channels = [
        _channel(address, "setpoint" if address in PAIRS.values() else "readback")
        for address in sorted([*PAIRS, *PAIRS.values()])
    ]
    wiring = [
        {"id": str(index), "address": address, "direction": "write", "default": BOOT}
        for index, address in enumerate(sorted(PAIRS.values()))
    ] + [
        {"id": str(index + len(PAIRS)), "address": address, "direction": "read"}
        for index, address in enumerate(sorted(PAIRS))
    ]
    documents = {
        "served_models.json": {"models": [MODEL, "texture"]},
        "addresses.json": {
            "channels": [channel["address"] for channel in channels],
            "status": [f"{CODE}:SIM:{MODEL}:STATUS"],
        },
        "variables.json": {
            "code": CODE,
            "models": [
                {
                    "name": MODEL,
                    "engine": STUB_ENGINE,
                    "served": True,
                    "deck": None,
                    "settings": {},
                    "wiring": wiring,
                },
                {
                    "name": "texture",
                    "engine": "texture",
                    "served": True,
                    "deck": None,
                    "settings": {},
                    "wiring": [],
                },
            ],
            "channels": channels,
        },
        "seeds.json": {"seeds": {}},
        "scenarios.json": {
            "scenarios": [
                {"name": "nominal"},
                {"name": STUCK_SCENARIO, "faults": {MODEL: {"writes": {STUCK_SP: "stuck"}}}},
            ]
        },
    }
    view = data_dir / "simulator"
    view.mkdir(parents=True)
    for name, document in documents.items():
        (view / name).write_text(json.dumps(document), encoding="utf-8")
    return view


def _activate(state_dir: Path, *names: str) -> None:
    """Write the active set, moving the file's modification time forward."""
    path = state_dir / "active_scenarios"
    before = path.stat().st_mtime_ns if path.exists() else 0
    path.write_text("".join(f"{name}\n" for name in ("nominal", *names)), encoding="utf-8")
    stamp = max(before + 1_000_000, path.stat().st_mtime_ns)
    os.utime(path, ns=(stamp, stamp))


# =============================================================================
# In process: the composite holds the fault
# =============================================================================


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    """The state directory, with the stuck scenario active."""
    state = tmp_path / "state"
    state.mkdir()
    _activate(state, STUCK_SCENARIO)
    return state


@pytest.fixture
def composite(tmp_path: Path, state_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The composite over the stub view, reading ``state_dir``."""
    from osprey_connectors.simulation.composite import Composite

    monkeypatch.setattr(Composite, "_engine", staticmethod(lambda name: ECHO))
    BUILT.clear()
    return Composite(_write_view(tmp_path / "data"), state_dir=state_dir, model_log=False)


class TestStuckSetpointInTheComposite:
    """The fault, without a server."""

    def test_a_write_to_the_stuck_setpoint_is_accepted(self, composite: Any) -> None:
        composite.set({STUCK_SP: 7.25})

    def test_the_stuck_write_never_reaches_the_model(self, composite: Any) -> None:
        composite.set({STUCK_SP: 7.25})

        assert STUCK_SP not in BUILT[-1].written

    def test_the_stuck_readback_holds_its_boot_value(self, composite: Any) -> None:
        for value in (1.0, -3.0, 11.5):
            composite.set({STUCK_SP: value})

        assert composite.held([STUCK_RB]) == {STUCK_RB: BOOT}

    def test_the_unfaulted_sibling_reaches_the_model(self, composite: Any) -> None:
        composite.set({LIVE_SP: 4.5})

        assert composite.held([LIVE_RB]) == {LIVE_RB: 4.5}

    def test_one_batch_forwards_the_sibling_and_holds_the_stuck_one(self, composite: Any) -> None:
        composite.set({STUCK_SP: 8.0, LIVE_SP: 3.0})

        assert composite.held([STUCK_RB, LIVE_RB]) == {STUCK_RB: BOOT, LIVE_RB: 3.0}

    def test_the_fault_is_off_without_the_scenario(self, composite: Any, state_dir: Path) -> None:
        """No setpoint is stuck unless an active scenario says so."""
        _activate(state_dir)

        composite.set({STUCK_SP: 6.0})

        assert composite.held([STUCK_RB]) == {STUCK_RB: 6.0}

    def test_the_active_set_names_the_scenario(self, composite: Any) -> None:
        assert composite.active == ["nominal", STUCK_SCENARIO]


# =============================================================================
# Over the wire: the entrypoint serves the fault
# =============================================================================


def _serve(env: dict[str, str]) -> None:
    """Server process: boot the entrypoint over the stub view."""
    os.environ.update(env)
    _with_echo_engine()
    from osprey.services.virtual_accelerator import entrypoint

    entrypoint.main()


class _Served:
    """The spawned entrypoint and the PVAccess client that reaches it."""

    def __init__(self, env: dict[str, str]) -> None:
        from p4p.client.thread import Context

        context = multiprocessing.get_context("spawn")
        self._proc = context.Process(target=_serve, args=(env,), daemon=True)
        self._proc.start()
        # Searched over TCP, straight at this server: a UDP search needs a
        # broadcast path the venue's network namespace may not offer.
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

    def rpc(self, verb: str) -> Any:
        from osprey.services.virtual_accelerator.serving.model_rpc import (
            RPC_PV,
            build_request,
            parse_reply,
        )

        return parse_reply(self.pva.rpc(RPC_PV, build_request(verb), timeout=CA_TIMEOUT_S))

    def stop(self) -> None:
        self.pva.close()
        self._proc.kill()
        self._proc.join(timeout=10)


@pytest.fixture(scope="module")
def live(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Served]:
    """The entrypoint serving the stub view with the stuck scenario active."""
    import epics

    # The client binds pyepics' own libca before the server extension is
    # imported: pcaspy exports the ca_* client symbols too, and whichever
    # stack binds first wins. The same extra that carries pcaspy carries
    # lume-pva-apg and p4p, so one skip covers the whole serving stack.
    epics.ca.initialize_libca()
    pytest.importorskip(
        "pcaspy",
        reason=(
            "the live Channel Access venue needs pcaspy, which has no loadable "
            "macOS arm64 wheel; run this suite in a linux container or on CI"
        ),
    )

    root = tmp_path_factory.mktemp("apply-fault")
    _write_view(root / "data")
    state = root / "state"
    state.mkdir()
    _activate(state, STUCK_SCENARIO)
    env = {
        "VA_DATA_DIR": str(root / "data"),
        "VA_STATE_DIR": str(state),
        "VA_INSTANCE": INSTANCE,
        "VA_POLL_INTERVAL_S": str(TICK_S),
        "EPICS_PVAS_SERVER_PORT": _free_port(),
        "EPICS_PVAS_BROADCAST_PORT": _free_port(),
        "EPICS_PVAS_AUTO_BEACON_ADDR_LIST": "NO",
        "EPICS_PVAS_BEACON_ADDR_LIST": "127.0.0.1",
        "EPICS_CAS_AUTO_BEACON_ADDR_LIST": "NO",
        "EPICS_CAS_BEACON_ADDR_LIST": "127.0.0.1",
    }
    served = _Served(env)
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while _caget(LIVE_RB) is None:
            if not served.alive():
                pytest.fail("the entrypoint exited before it served")
            if time.monotonic() > deadline:
                pytest.fail("the entrypoint never served")
        yield served
    finally:
        served.stop()


def _wait_until(predicate: Any, *, timeout: float = SETTLE_TIMEOUT_S) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.05)
    return predicate()


def _caget(address: str) -> Any:
    """Read one value over the wire, never from pyepics' monitor cache.

    ``use_monitor=False`` is load-bearing, not stylistic: pyepics' ``caget``
    otherwise returns whatever its monitor subscription last cached, which
    after a write is whatever arrived BEFORE the write did. This suite asserts
    that a refused forward moved NOTHING, and a stale cache never moves, so
    that assertion would pass for free.
    """
    import epics

    return epics.caget(address, timeout=2.0, connection_timeout=2.0, use_monitor=False)


def _caput(address: str, value: float) -> Any:
    """Write with put-completion: returns once the write's pass has finished."""
    import epics

    return epics.caput(address, value, wait=True, timeout=CA_TIMEOUT_S)


def _settle(address: str, expected: float) -> Any:
    last: list[Any] = [None]

    def reached() -> bool:
        last[0] = _caget(address)
        return last[0] is not None and abs(last[0] - expected) < 1e-9

    _wait_until(reached)
    return last[0]


class TestLiveStuckSetpoint:
    """The stuck pair, over the wire."""

    @pytest.mark.usefixtures("live")
    def test_the_write_completes(self) -> None:
        """A write that never completed would postpone every later write to
        its channel, and the client would hang rather than be told a lie it
        could detect."""
        assert _caput(STUCK_SP, 1.0) == 1
        assert _caput(STUCK_SP, 2.0) == 1

    @pytest.mark.usefixtures("live")
    def test_the_readback_stays_at_its_boot_value(self) -> None:
        assert _caput(STUCK_SP, 4.0) == 1
        time.sleep(3 * TICK_S)

        assert _caget(STUCK_RB) == pytest.approx(BOOT)

    @pytest.mark.usefixtures("live")
    def test_repeated_writes_never_move_it(self) -> None:
        """Frozen means frozen, not merely lagging by one write."""
        for value in (1.0, -3.0, 11.5):
            assert _caput(STUCK_SP, value) == 1
        time.sleep(3 * TICK_S)

        assert _caget(STUCK_RB) == pytest.approx(BOOT)

    @pytest.mark.usefixtures("live")
    def test_a_monitoring_client_sees_no_other_value(self) -> None:
        """The freeze is in the served value, so a subscriber sees a device
        that never moves, whatever the passes after the write publish."""
        import epics

        seen: list[float] = []
        readback = epics.PV(STUCK_RB, auto_monitor=True)
        try:
            assert readback.wait_for_connection(timeout=CA_TIMEOUT_S)
            assert readback.get(use_monitor=False) == pytest.approx(BOOT)
            readback.add_callback(lambda value=None, **_: seen.append(value))

            assert _caput(STUCK_SP, 6.0) == 1
            time.sleep(3 * TICK_S)
        finally:
            # Explicit, in a finally: a PV finalised by the garbage collector
            # tears libca down from the wrong thread.
            readback.disconnect()

        assert [value for value in seen if abs(value - BOOT) > 1e-9] == []


class TestLiveUnfaultedSibling:
    """The pair the scenario does not name, over the wire."""

    @pytest.mark.usefixtures("live")
    def test_the_sibling_reaches_the_model(self) -> None:
        assert _caput(LIVE_SP, 4.5) == 1

        assert _settle(LIVE_RB, 4.5) == pytest.approx(4.5)

    @pytest.mark.usefixtures("live")
    def test_faulting_one_channel_leaves_its_sibling_alone(self) -> None:
        assert _caput(LIVE_SP, 3.0) == 1
        _settle(LIVE_RB, 3.0)

        assert _caput(STUCK_SP, 8.0) == 1
        time.sleep(3 * TICK_S)

        assert _caget(LIVE_RB) == pytest.approx(3.0)
        assert _caget(STUCK_RB) == pytest.approx(BOOT)


class TestLiveInstance:
    """The instance the entrypoint was started as, as the model RPC reports it."""

    def test_the_status_reply_names_the_instance(self, live: _Served) -> None:
        assert live.rpc("status")["instance"] == INSTANCE


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_apply_fault.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
