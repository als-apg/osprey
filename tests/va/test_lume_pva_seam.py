"""The model runner's serving seam, seen from the far side of both wires.

:class:`~osprey.services.virtual_accelerator.serving.runner.ModelRunner`
serves a :class:`~osprey_connectors.simulation.composite.Composite` on
Channel Access and PVAccess from one configuration, and answers the model RPC
over the composite's simulator view. Everything this file asserts is what a
client reads: a real ``pyepics`` client and a real ``p4p`` client in the
pytest process, against a real ``pcaspy`` and ``p4p`` server.

**One server per process.** Each served view runs in a spawned process of its
own: the composite, the runner, its run loop and both servers, in-process with
one another. A Channel Access server cannot be stopped once it serves, and two
in one process would share the process-global libca, so a process is the unit
a server is started and ended in. The child answers a few commands over a pipe
-- the applied configuration, a periodic pass on demand, the solve counts and
queue depth it observed, a one-shot failure of the next pass -- so what the
wire cannot show is still measured where it happens.

Three views are served:

* a hand-written tree of names in two separator conventions, built through
  ``osprey build`` with no physics model -- the names are served verbatim and
  boot with no alarm;
* the control-assistant demo, whose SR model is a pyAT child -- the
  write safety, the chromaticity left for the next tick, the model RPC's
  fault writes, and a child that fails to solve;
* a hand-written view of stub children -- the alarm each kind of output
  carries, an integer channel's wire type, a float waveform served as the
  array it declares, and the run loop under a slow model and a pass that
  raises.

**The venue is linux/x86_64.** ``lume-pva-apg`` installs only there, so the
module skips whole anywhere else and carries no test that skips on the venue.
``scripts/va/live_ca/gate.py --pva`` runs it in a process of its own and fails
on any skip.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import shutil
import socket
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

pytest.importorskip(
    "lume_pva_apg",
    reason="lume-pva-apg installs on linux/x86_64 only; run this module through the live-CA gate",
)
pytest.importorskip("p4p", reason="the PVAccess client ships with lume-pva-apg's pva extra")
epics = pytest.importorskip("epics", reason="the Channel Access client is pyepics")


def _free_port() -> str:
    """An unused loopback TCP port, as a string ready for the environment."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return str(probe.getsockname()[1])


#: The three served views, each in a process of its own.
SERVERS = ("names", "demo", "stub")

# import-time required because libca latches the EPICS_CA_* environment when
# the C library initialises, which happens below, before pcaspy is imported.
# Each server listens on a Channel Access port of its own, so the client
# searches all three; the pairing of server and port is fixed here, once.
CA_PORTS: dict[str, str] = {server: _free_port() for server in SERVERS}
# import-time required because libca reads the search list once, at the
# initialisation below.
os.environ["EPICS_CA_ADDR_LIST"] = " ".join(f"127.0.0.1:{CA_PORTS[s]}" for s in SERVERS)
os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
os.environ.setdefault("EPICS_CA_REPEATER_PORT", _free_port())

# The client binds pyepics' own libca before the server extension is imported:
# pcaspy exports the ca_* client symbols too, and whichever stack binds first
# wins.
epics.ca.initialize_libca()
pytest.importorskip("pcaspy", reason="the Channel Access server ships with lume-pva-apg's ca extra")

from p4p.client.thread import Context  # noqa: E402

from osprey.services.virtual_accelerator.serving.model_rpc import (  # noqa: E402
    RPC_PV,
    ModelRpcError,
    build_request,
    parse_reply,
)
from tests._builds import BuiltProject, init_project, run_build  # noqa: E402
from tests.facility._synthetic_trees import write_tree  # noqa: E402

#: The model write token every served runner is started with.
TOKEN = "seam-token"

#: How long one client operation may take. Generous: an emulated container is
#: a venue, and a timeout here has to mean "never", not "not yet".
CA_TIMEOUT_S = 30.0
SETTLE_TIMEOUT_S = 60.0

#: How long a spawned server may take to build its composite and serve.
STARTUP_TIMEOUT_S = 900.0

#: How long the server process may take to answer one command.
COMMAND_TIMEOUT_S = 120.0

#: The write-path keys of every applied configuration, spelled out so a
#: change to any of them is a change to this file too.
SAFETY = {
    "update_rate": 0.0,
    "echo_unconfirmed_writes": False,
    "alarm_on_refused_write": True,
    "clamp_writes": True,
    "control_pvs": False,
}

#: (severity, status) of an undefined output on each wire.
CA_UDF = (3, 17)
PVA_UDF = (3, 6)
#: (severity, status) of a float outside its range on PVAccess.
PVA_RANGE = (2, 2)
NO_ALARM = (0, 0)

# -- the names tree: two separator conventions, no physics model ---------------

SR_SETPOINT, SR_READBACK = "SR04U___GDS1PS_AC00", "SR04U___GDS1PS_AM00"
BTS_SETPOINT, BTS_READBACK = "BTS:HCM1:AC", "BTS:HCM1:AM"
COLLAPSED = "SR04U_GDS1PS_AC00"
NAMES_TREE: dict[str, Any] = {
    "records/channels.yaml": [
        {
            "id": SR_SETPOINT,
            "role": "setpoint",
            "pair": SR_READBACK,
            "simulation": {"nominal": 12.5},
        },
        {"id": SR_READBACK},
        {
            "id": BTS_SETPOINT,
            "role": "setpoint",
            "pair": BTS_READBACK,
            "simulation": {"nominal": 0.25},
        },
        {"id": BTS_READBACK},
    ],
    "limits.yaml": {
        "records": [
            {"address": SR_SETPOINT, "min_value": 0.0, "max_value": 20.0},
            {"address": BTS_SETPOINT, "min_value": -1.0, "max_value": 1.0},
        ]
    },
}
NAMES = (SR_SETPOINT, SR_READBACK, BTS_SETPOINT, BTS_READBACK)

# -- the demo ------------------------------------------------------------------

CORRECTOR = "SR:MAG:HCM:01:CURRENT:SP"
BPM_X = "SR:DIAG:BPM:03:POSITION:X"
TUNES = ("SR:DIAG:TUNE:X", "SR:DIAG:TUNE:Y")
CHROMATICITY = ("SR:DIAG:CHROM:X", "SR:DIAG:CHROM:Y")
SR_STATUS = "ca:SIM:SR:STATUS"

#: BPM03's horizontal offset fault, as the model RPC names it.
BPM_OFFSET = f"SR/{BPM_X}/offset"
#: The offset written, in the reading's own unit (metres). A reading is the
#: position less its offset, so the reading falls by exactly this much.
OFFSET_M = 1e-3
#: Far wider than the reading's own motion between two passes.
READING_TOLERANCE_M = 2e-5

#: A scenario the demo view gains for these tests: twice the demo's QF01
#: current, where the one-turn map has no stable orbit, so the pyAT child
#: fails to build.
UNSTABLE = "zz-seam-unstable-quadrupole"
UNSTABLE_QUADRUPOLE = {"SR:MAG:QF:01:CURRENT:SP": 712.2}

# -- the stub view -------------------------------------------------------------

STUB_ENGINE = "zz-seam-stub"
STUB_CODE = "ZZSEAM"
STUB_RANGE = "ZZSEAM:M:RB"  # a float readback that reads outside its range
STUB_SETPOINT = "ZZSEAM:M:SP"
STUB_FLAG = "ZZSEAM:F:FLAG"  # bool output of a child that fails to build
STUB_MODE = "ZZSEAM:F:MODE"  # enum output of the same child
STUB_COUNT = "ZZSEAM:T:COUNT"  # an int channel of the texture
STUB_WAVE = "ZZSEAM:T:WAVE"  # a float waveform readback of the texture
STUB_DELAY = "M/delay"  # the stub's own variable: seconds each read takes
STUB_PRECISION = 3
STUB_READING = 4.0
STUB_SETPOINT_START = 1.0
STUB_COUNT_NOMINAL = 7
STUB_WAVE_NOMINAL = [0.13, 0.22, 0.0086]
#: The periodic pass of the stub view's runner.
STUB_TICK_S = 0.05
#: How long each read of the slow stub takes.
SLOW_PASS_S = 0.25


# =============================================================================
# The stub engine: loaded by the server process, by name
# =============================================================================


def _stub_model_class() -> type:
    """The stub's LUME model class, defined where LUME is imported."""
    from lume.model import LUMEModel
    from lume.variables import ScalarVariable

    class StubModel(LUMEModel):
        """Readbacks read :data:`STUB_READING`; ``delay`` slows every read."""

        def __init__(self, wiring: list[Mapping[str, Any]]) -> None:
            self._variables: dict[str, Any] = {}
            self._defaults: dict[str, float] = {"delay": 0.0}
            for entry in wiring:
                address = str(entry["address"])
                writes = entry.get("direction") == "write"
                self._variables[address] = ScalarVariable(name=address, read_only=not writes)
                if writes:
                    self._defaults[address] = float(entry.get("default", 0.0))
            self._variables["delay"] = ScalarVariable(name="delay", read_only=False)
            self.inputs: dict[str, float] = {}
            self.reset()

        @property
        def supported_variables(self) -> dict[str, Any]:
            return self._variables

        def reset(self) -> None:
            self.inputs = dict(self._defaults)

        def _set(self, values: dict[str, Any]) -> None:
            self.inputs.update({name: float(value) for name, value in values.items()})

        def _get(self, names: list[str]) -> dict[str, Any]:
            time.sleep(self.inputs["delay"])
            return {name: self.inputs.get(name, STUB_READING) for name in names}

    return StubModel


def _stub_build(model: str, wiring: Any, deck: Any, settings: Any, active: Any = None) -> Any:
    del model, deck, active
    if settings and settings.get("fail"):
        raise RuntimeError(settings["fail"])
    return _stub_model_class()(list(wiring))


STUB = SimpleNamespace(build=_stub_build, error_text=lambda exc: str(exc))


def _stub_channel(address: str, owner: str, **fields: Any) -> dict[str, Any]:
    role = fields.pop("role", "readback")
    return {
        "address": address,
        "role": role,
        "pair": None,
        "value_type": fields.pop("value_type", "float"),
        "unit": fields.pop("unit", None),
        "description": None,
        "writable": role == "setpoint",
        "value_range": fields.pop("value_range", None),
        "owner": owner,
        **fields,
    }


def _write_stub_view(root: Path) -> Path:
    """A view of two stub children and the texture; returns its directory."""
    channels = [
        _stub_channel(STUB_RANGE, "M", value_range=[0.0, 1.0], precision=STUB_PRECISION),
        _stub_channel(STUB_SETPOINT, "M", role="setpoint", value_range=[-10.0, 10.0]),
        _stub_channel(STUB_FLAG, "F", value_type="bool"),
        _stub_channel(STUB_MODE, "F", value_type="enum", options=["IDLE", "RUN", "FAULT"]),
        _stub_channel(STUB_COUNT, "texture", value_type="int"),
        _stub_channel(STUB_WAVE, "texture", value_type="waveform", shape=[3]),
    ]
    stub_models: dict[str, dict[str, Any]] = {
        "M": {
            "settings": {},
            "wiring": [
                {
                    "id": "1",
                    "address": STUB_SETPOINT,
                    "direction": "write",
                    "default": STUB_SETPOINT_START,
                },
                {"id": "2", "address": STUB_RANGE, "direction": "read"},
            ],
        },
        "F": {
            "settings": {"fail": "the stub has no deck"},
            "wiring": [
                {"id": "3", "address": STUB_FLAG, "direction": "read"},
                {"id": "4", "address": STUB_MODE, "direction": "read"},
            ],
        },
    }
    models: list[dict[str, Any]] = [
        {"name": name, "engine": STUB_ENGINE, "served": True, "deck": None, **model}
        for name, model in sorted(stub_models.items())
    ]
    models.append(
        {
            "name": "texture",
            "engine": "texture",
            "served": True,
            "settings": {},
            "deck": None,
            "wiring": [],
        }
    )
    documents = {
        "served_models.json": {"models": ["F", "M", "texture"]},
        "addresses.json": {
            "channels": sorted(channel["address"] for channel in channels),
            "status": [f"{STUB_CODE}:SIM:F:STATUS", f"{STUB_CODE}:SIM:M:STATUS"],
        },
        "variables.json": {
            "code": STUB_CODE,
            "models": models,
            "channels": sorted(channels, key=lambda channel: channel["address"]),
        },
        "seeds.json": {
            "seeds": {
                STUB_COUNT: {"nominal": STUB_COUNT_NOMINAL},
                STUB_WAVE: {"nominal": STUB_WAVE_NOMINAL},
            }
        },
        "scenarios.json": {"scenarios": [{"name": "nominal"}]},
    }
    view = root / "data" / "simulator"
    view.mkdir(parents=True)
    for name, document in documents.items():
        (view / name).write_text(json.dumps(document), encoding="utf-8")
    return view


# =============================================================================
# The server process
# =============================================================================


class _Observed:
    """What the server process counts while it serves."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.passes_started = 0
        self.passes_done = 0
        self.solves = 0
        self.chromatic_solves = 0
        self.put_lags: list[int] = []
        self.max_depth = 0
        self.sampling = False


class _PassFault:
    """A one-shot failure of the next model read that follows a ``model.set``."""

    def __init__(self) -> None:
        self.text: str | None = None
        self.values_only = False
        self.after_set = False


def _count_physics(observed: _Observed) -> None:
    """Count every closed-orbit solve and every chromatic solve in this process."""
    import at
    import lume_pyat.simulator

    solve_orbit = lume_pyat.simulator.solve_orbit
    get_optics = at.get_optics

    def counted_solve(*args: Any, **kwargs: Any) -> Any:
        with observed.lock:
            observed.solves += 1
        return solve_orbit(*args, **kwargs)

    def counted_optics(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("get_chrom"):
            with observed.lock:
                observed.chromatic_solves += 1
        return get_optics(*args, **kwargs)

    lume_pyat.simulator.solve_orbit = counted_solve
    at.get_optics = counted_optics


def _inject_pass_fault(composite: Any, fault: _PassFault) -> None:
    """Make the composite's next read after a write raise ``fault.text``, once."""
    real_get, real_set = composite.get, composite.set

    def set_(values: dict[str, Any]) -> None:
        real_set(values)
        if fault.text is not None and (values or not fault.values_only):
            fault.after_set = True

    def get_(names: Any) -> Any:
        if fault.after_set and fault.text is not None:
            text, fault.text, fault.after_set = fault.text, None, False
            raise RuntimeError(text)
        return real_get(names)

    composite.set = set_
    composite.get = get_


def _instrument(runner: Any, observed: _Observed) -> None:
    """Count the run loop's passes and how many each value-carrying item waited."""
    run_cycle = runner._run_cycle
    put = runner.queue.put

    def counted_cycle(item: dict[str, Any]) -> None:
        with observed.lock:
            observed.passes_started += 1
            if item.get("values") and "_enqueued_at" in item:
                observed.put_lags.append(observed.passes_started - item["_enqueued_at"])
        try:
            run_cycle(item)
        finally:
            with observed.lock:
                observed.passes_done += 1

    def marked_put(item: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        if item.get("values"):
            with observed.lock:
                item["_enqueued_at"] = observed.passes_started
        put(item, *args, **kwargs)

    def sample() -> None:
        while True:
            depth = runner.queue.qsize()
            with observed.lock:
                if observed.sampling:
                    observed.max_depth = max(observed.max_depth, depth)
            time.sleep(0.005)

    runner._run_cycle = counted_cycle
    runner.queue.put = marked_put
    threading.Thread(target=sample, daemon=True, name="queue-depth-sampler").start()


def _on_loop(runner: Any, job: Callable[[], Any]) -> Any:
    """Run ``job`` on the run loop's thread and return what it returned."""
    done = threading.Event()
    result: list[Any] = []

    def wrapped() -> None:
        try:
            result.append(job())
        finally:
            done.set()

    runner._enqueue({}, jobs=[wrapped])
    if not done.wait(COMMAND_TIMEOUT_S):
        raise TimeoutError("the run loop never reached the job")
    return result[0]


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
    try:
        from osprey_connectors.simulation.composite import Composite

        observed = _Observed()
        if spec["physics"]:
            _count_physics(observed)
        if spec["stub"]:
            real_engine = Composite._engine

            def engine(name: str) -> Any:
                return STUB if name == STUB_ENGINE else real_engine(name)

            Composite._engine = staticmethod(engine)  # type: ignore[method-assign]

        view_dir = Path(spec["view"])
        composite = Composite(view_dir, state_dir=spec["state"], model_log=False)
        fault = _PassFault()
        _inject_pass_fault(composite, fault)

        from osprey.services.virtual_accelerator.serving.runner import ModelRunner

        runner = ModelRunner(
            composite,
            json.loads((view_dir / "variables.json").read_text(encoding="utf-8")),
            json.loads((view_dir / "addresses.json").read_text(encoding="utf-8")),
            model_write_token=TOKEN,
            tick_interval_s=spec["tick"],
        )
        _instrument(runner, observed)
        threading.Thread(target=runner._run, daemon=True, name="model-runner-loop").start()
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while observed.passes_done < 1:
            if time.monotonic() > deadline:
                raise TimeoutError("the runner's first pass never finished")
            time.sleep(0.05)
    except BaseException as exc:
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
        os._exit(1)
    conn.send(("ready", None))

    def counts() -> dict[str, int]:
        with observed.lock:
            return {
                "passes": observed.passes_done,
                "solves": observed.solves,
                "chromatic_solves": observed.chromatic_solves,
            }

    def start_sampling() -> None:
        with observed.lock:
            observed.max_depth = 0
            observed.put_lags.clear()
            observed.sampling = True

    def stop_sampling() -> dict[str, Any]:
        with observed.lock:
            observed.sampling = False
            return {"max_depth": observed.max_depth, "put_lags": list(observed.put_lags)}

    def fail_next_pass(text: str, values_only: bool) -> None:
        fault.values_only = values_only
        fault.after_set = False
        fault.text = text

    commands: dict[str, Callable[..., Any]] = {
        "config": lambda: json.loads(json.dumps(runner.config)),
        "tick": runner._tick,
        "counts": counts,
        "held": lambda names: _on_loop(runner, lambda: composite.held(names)),
        "start_sampling": start_sampling,
        "stop_sampling": stop_sampling,
        "fail_next_pass": fail_next_pass,
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
    """A spawned server process and the clients that reach it."""

    def __init__(self, spec: dict[str, Any], view: dict[str, Any], state_dir: Path | None) -> None:
        self.view = view
        self.state_dir = state_dir
        self.channels = {str(c["address"]): c for c in view["channels"]}
        context = multiprocessing.get_context("spawn")
        self._conn, child = context.Pipe()
        self._proc = context.Process(target=_serve, args=(spec, child), daemon=True)
        self._proc.start()
        child.close()
        # Searched over TCP, straight at this server: a UDP search needs a
        # broadcast path the venue's network namespace may not offer.
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
        self._await_ready()

    def _await_ready(self) -> None:
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

    def rpc(self, verb: str, **fields: Any) -> Any:
        """One model RPC call; the result, or :class:`ModelRpcError` on a refusal."""
        reply = self.pva.rpc(RPC_PV, build_request(verb, **fields), timeout=CA_TIMEOUT_S)
        return parse_reply(reply)

    def pva_get(self, address: str) -> Any:
        """The whole PVAccess value of ``address``, alarm and display included."""
        return self.pva.get(address, timeout=CA_TIMEOUT_S)

    def stop(self) -> None:
        self.pva.close()
        try:
            self.ask("stop")
        except (AssertionError, OSError):
            pass
        self._proc.join(timeout=10)
        if self._proc.is_alive():
            self._proc.kill()
            self._proc.join(timeout=10)


@contextmanager
def _served(
    view_dir: Path,
    server: str,
    *,
    state_dir: Path | None = None,
    tick_interval_s: float | None = None,
    physics: bool = False,
    stub: bool = False,
) -> Iterator[_Server]:
    spec = {
        "view": str(view_dir),
        "state": None if state_dir is None else str(state_dir),
        "tick": tick_interval_s,
        "ca_port": CA_PORTS[server],
        "pva_port": _free_port(),
        "pva_broadcast": _free_port(),
        "physics": physics,
        "stub": stub,
    }
    view = json.loads((view_dir / "variables.json").read_text(encoding="utf-8"))
    served = _Server(spec, view, state_dir)
    try:
        yield served
    finally:
        served.stop()


# =============================================================================
# Client helpers
# =============================================================================


def _wait_until(predicate: Callable[[], Any], timeout: float = SETTLE_TIMEOUT_S) -> Any:
    """Poll ``predicate`` until it is truthy, then return what it returned."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.1)
    return predicate()


def _ca_pv(address: str) -> Any:
    pv = epics.get_pv(address, form="time", auto_monitor=False)
    assert pv.wait_for_connection(timeout=CA_TIMEOUT_S), f"{address} never connected over CA"
    return pv


def _ca(address: str) -> dict[str, Any]:
    """One read over CA, never from a monitor cache: value, alarm and time stamp."""
    data = _ca_pv(address).get_with_metadata(
        use_monitor=False, form="time", timeout=CA_TIMEOUT_S, as_numpy=False
    )
    assert data is not None, f"{address} returned nothing over CA"
    reading: dict[str, Any] = data
    return reading


def _ca_text(address: str) -> str:
    """A string channel's text, read once over CA."""
    text: str = _ca_pv(address).get(use_monitor=False, as_string=True, timeout=CA_TIMEOUT_S)
    return text


def _ca_alarm(address: str) -> tuple[int, int]:
    data = _ca(address)
    return (int(data["severity"]), int(data["status"]))


def _caput(address: str, value: float) -> int:
    """Write with put-completion: returns once the write's pass has finished."""
    status: int = epics.caput(address, value, wait=True, timeout=CA_TIMEOUT_S)
    return status


def _pva_alarm(value: Any) -> tuple[int, int]:
    return (int(value["alarm.severity"]), int(value["alarm.status"]))


def _assert_safety_config(
    config: Mapping[str, Any],
    view: Mapping[str, Any],
    limits: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    """The applied configuration: the safety keys, each mode, band and precision.

    A setpoint's band is the limits record's ``[min_value, max_value]`` when the
    record states both bounds, and nothing otherwise; with ``limits`` ``None``
    it is the view's own.
    """
    assert {key: config[key] for key in SAFETY} == SAFETY
    channels = {str(c["address"]): c for c in view["channels"]}
    variables = config["variables"]
    assert set(channels) <= set(variables)
    for address, channel in channels.items():
        entry = variables[address]
        setpoint = channel["role"] == "setpoint"
        assert entry["mode"] == ("rw" if setpoint and channel["writable"] is True else "ro"), (
            address
        )
        if setpoint:
            if limits is None:
                band = channel["value_range"]
            else:
                record = limits.get(address, {})
                bounds = (record.get("min_value"), record.get("max_value"))
                band = None if None in bounds else list(bounds)
            assert entry.get("value_range") == band, address
        else:
            assert "value_range" not in entry, address
        if channel.get("value_type", "float") == "float" and channel.get("precision") is not None:
            assert entry["precision"] == channel["precision"], address
        else:
            assert "precision" not in entry, address


def _limits_records(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """The limits view's records by address, its schema version left out."""
    return {address: record for address, record in document.items() if isinstance(record, dict)}


# =============================================================================
# Names in two separator conventions, served verbatim
# =============================================================================


@pytest.fixture(scope="class")
def names_build(tmp_path_factory: pytest.TempPathFactory) -> BuiltProject:
    """The names tree built through ``osprey build`` under a mock control system."""
    repo = init_project(tmp_path_factory.mktemp("names"), "hello-world", "names")
    facility_dir = repo / "data" / "facility"
    shutil.rmtree(facility_dir)
    write_tree(facility_dir, NAMES_TREE)
    profile_file = repo / "profile.yml"
    profile = yaml.safe_load(profile_file.read_text(encoding="utf-8"))
    profile["config"] = {**(profile.get("config") or {}), "control_system.type": "mock"}
    profile_file.write_text(yaml.safe_dump(profile, sort_keys=True), encoding="utf-8")
    result = run_build(repo)
    assert result.exit_code == 0, result.output
    return BuiltProject(repo)


@pytest.fixture(scope="class")
def names_server(names_build: BuiltProject) -> Iterator[_Server]:
    with _served(names_build.build_dir / "data" / "simulator", "names") as served:
        yield served


class TestNamesServedVerbatim:
    """A tree with no physics model: its names, its bands, no alarm at boot."""

    def test_als_shaped_names_served_verbatim_no_alarm(self, names_server: _Server) -> None:
        for address in NAMES:
            assert _ca_pv(address).connected
        assert epics.PV(COLLAPSED).wait_for_connection(timeout=5.0) is False

        assert _ca(SR_READBACK)["value"] == 12.5
        assert _ca(BTS_READBACK)["value"] == 0.25
        for address in NAMES:
            assert _ca_alarm(address) == NO_ALARM, address
            assert int(names_server.pva_get(address)["alarm.severity"]) == 0, address

        assert _caput(SR_SETPOINT, 13.0) == 1
        echo = _ca(SR_READBACK)
        assert (echo["value"], (echo["severity"], echo["status"])) == (13.0, NO_ALARM)

    def test_the_applied_config_holds_the_safety_keys_and_the_limits_bands(
        self, names_server: _Server, names_build: BuiltProject
    ) -> None:
        limits = json.loads(
            (names_build.build_dir / "data" / "channel_limits.json").read_text(encoding="utf-8")
        )

        config = names_server.ask("config")

        _assert_safety_config(config, names_server.view, _limits_records(limits))
        assert config["variables"][SR_SETPOINT]["value_range"] == [0.0, 20.0]
        assert config["variables"][BTS_SETPOINT]["value_range"] == [-1.0, 1.0]


# =============================================================================
# The demo: a pyAT child beside the texture
# =============================================================================


@pytest.fixture(scope="class")
def demo_files(built_control_assistant: BuiltProject) -> dict[str, bytes]:
    files: dict[str, bytes] = built_control_assistant.outputs[0].files
    return files


@pytest.fixture(scope="class")
def demo_server(
    demo_files: dict[str, bytes], tmp_path_factory: pytest.TempPathFactory
) -> Iterator[_Server]:
    root = tmp_path_factory.mktemp("demo")
    prefix = "data/simulator/"
    view_dir = root / "data" / "simulator"
    for name, data in demo_files.items():
        if name.startswith(prefix):
            target = view_dir / name[len(prefix) :]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    scenarios_file = view_dir / "scenarios.json"
    scenarios = json.loads(scenarios_file.read_text(encoding="utf-8"))
    scenarios["scenarios"].append({"name": UNSTABLE, "overrides": UNSTABLE_QUADRUPOLE})
    scenarios_file.write_text(json.dumps(scenarios), encoding="utf-8")
    state_dir = root / "state"
    state_dir.mkdir()
    with _served(view_dir, "demo", state_dir=state_dir, physics=True) as served:
        yield served


@contextmanager
def _sr_failed(server: _Server) -> Iterator[None]:
    """Serve the demo with its pyAT child failed, then serve it healthy again."""
    assert server.state_dir is not None
    active = server.state_dir / "active_scenarios"
    active.write_text(f"{UNSTABLE}\n", encoding="utf-8")
    try:
        _tick_and_wait(server)
        yield
    finally:
        active.unlink()
        _tick_and_wait(server)


def _tick_and_wait(server: _Server) -> None:
    """Run one periodic pass and return once it has published every channel.

    A CA read is answered from the server's own copy of each value, which a
    pass sets one channel at a time, so a read made during a pass can see one
    channel moved and the next not yet.
    """
    passes = server.ask("counts")["passes"]
    server.ask("tick")
    assert _wait_until(lambda: server.ask("counts")["passes"] > passes), "the pass never finished"


class TestDemoComposite:
    """The demo's composite served whole: write safety, physics and the model RPC."""

    def test_the_applied_config_holds_the_safety_keys_and_every_setpoint_band(
        self, demo_server: _Server, demo_files: dict[str, bytes]
    ) -> None:
        limits = _limits_records(json.loads(demo_files["data/channel_limits.json"]))

        config = demo_server.ask("config")

        _assert_safety_config(config, demo_server.view, limits)
        assert config["variables"][CORRECTOR]["value_range"] == [-12.0, 12.0]

    def test_a_corrector_put_publishes_orbit_and_tunes_and_chromaticity_next_tick(
        self, demo_server: _Server
    ) -> None:
        before = demo_server.ask("counts")
        orbit_before = _ca(BPM_X)["value"]
        put_at = time.time()

        assert _caput(CORRECTOR, 1.5) == 1

        after_put = demo_server.ask("counts")
        for address in (BPM_X, *TUNES):
            assert _ca(address)["timestamp"] >= put_at, address
        for address in CHROMATICITY:
            assert _ca(address)["timestamp"] < put_at, address
        assert _ca(BPM_X)["value"] != orbit_before
        assert after_put["solves"] > before["solves"]
        assert after_put["chromatic_solves"] == before["chromatic_solves"]

        tick_at = time.time()
        demo_server.ask("tick")

        def published_since_tick(address: str) -> bool:
            stamp: float = _ca(address)["timestamp"]
            return stamp >= tick_at

        for address in CHROMATICITY:
            assert _wait_until(partial(published_since_tick, address)), address
        after_tick = demo_server.ask("counts")
        assert after_tick["chromatic_solves"] - after_put["chromatic_solves"] == 1

    def test_an_rpc_offset_fault_moves_the_ca_reading_before_the_reply(
        self, demo_server: _Server
    ) -> None:
        before = _ca(BPM_X)["value"]
        try:
            written = demo_server.rpc("set", values={BPM_OFFSET: OFFSET_M}, token=TOKEN)
            after = _ca(BPM_X)["value"]
        finally:
            demo_server.rpc("set", values={BPM_OFFSET: 0.0}, token=TOKEN)

        assert written == [BPM_OFFSET]
        assert after - before == pytest.approx(-OFFSET_M, abs=READING_TOLERANCE_M)

    def test_a_pass_raising_after_a_fault_set_gives_an_error_reply(
        self, demo_server: _Server
    ) -> None:
        demo_server.ask("fail_next_pass", "the pass failed on purpose", False)
        try:
            with pytest.raises(ModelRpcError, match="the pass failed on purpose"):
                demo_server.rpc("set", values={BPM_OFFSET: OFFSET_M}, token=TOKEN)
            status = demo_server.rpc("status")
        finally:
            demo_server.rpc("set", values={BPM_OFFSET: 0.0}, token=TOKEN)

        assert "the pass failed on purpose" in status["last_refused_write"]

    def test_a_failed_pyat_childs_in_range_float_reports_udf_on_both_wires(
        self, demo_server: _Server
    ) -> None:
        with _sr_failed(demo_server):
            reading = _ca(CORRECTOR)
            pva = demo_server.pva_get(CORRECTOR)
            status = _ca_text(SR_STATUS)

        assert -12.0 <= reading["value"] <= 12.0
        assert (reading["severity"], reading["status"]) == CA_UDF
        assert -12.0 <= pva["value"] <= 12.0
        assert _pva_alarm(pva) == PVA_UDF
        assert status != "ok"

    def test_a_rejected_write_on_a_failed_child_costs_no_solve(self, demo_server: _Server) -> None:
        with _sr_failed(demo_server):
            before = demo_server.ask("counts")
            with pytest.raises(ModelRpcError, match="has failed"):
                demo_server.rpc("set", values={BPM_OFFSET: OFFSET_M}, token=TOKEN)
            after = demo_server.ask("counts")

        assert after["passes"] > before["passes"]
        assert after["solves"] == before["solves"]


# =============================================================================
# Stub children: each kind of output, and the run loop under load
# =============================================================================


@pytest.fixture(scope="class")
def stub_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Server]:
    view_dir = _write_stub_view(tmp_path_factory.mktemp("stub"))
    with _served(view_dir, "stub", tick_interval_s=STUB_TICK_S, stub=True) as served:
        yield served


class TestStubComposite:
    """Stub children: their outputs' alarms and types, a slow pass, a failed pass."""

    def test_the_applied_config_holds_the_safety_keys_and_the_precision_served(
        self, stub_server: _Server
    ) -> None:
        config = stub_server.ask("config")

        _assert_safety_config(config, stub_server.view, None)
        assert config["variables"][STUB_RANGE]["precision"] == STUB_PRECISION
        assert _ca_pv(STUB_RANGE).get_ctrlvars(timeout=CA_TIMEOUT_S)["precision"] == (
            STUB_PRECISION
        )
        assert int(stub_server.pva_get(STUB_RANGE)["display.precision"]) == STUB_PRECISION

    def test_a_failed_childs_bool_and_enum_outputs_are_invalid(self, stub_server: _Server) -> None:
        for address in (STUB_FLAG, STUB_MODE):
            assert _ca_alarm(address) == CA_UDF, address
            assert _pva_alarm(stub_server.pva_get(address)) == PVA_UDF, address

    def test_a_healthy_childs_out_of_range_float_alarms_on_pva_only(
        self, stub_server: _Server
    ) -> None:
        reading = _ca(STUB_RANGE)
        pva = stub_server.pva_get(STUB_RANGE)

        assert (reading["value"], (reading["severity"], reading["status"])) == (
            STUB_READING,
            NO_ALARM,
        )
        assert (pva["value"], _pva_alarm(pva)) == (STUB_READING, PVA_RANGE)

    def test_an_int_channel_is_an_int_on_both_wires(self, stub_server: _Server) -> None:
        pva = stub_server.pva_get(STUB_COUNT)
        pv = _ca_pv(STUB_COUNT)

        assert stub_server.channels[STUB_COUNT]["value_type"] == "int"
        assert (pva.type()["value"], pva["value"]) == ("i", STUB_COUNT_NOMINAL)
        assert (pv.ftype, _ca(STUB_COUNT)["value"]) == (epics.dbr.TIME_LONG, STUB_COUNT_NOMINAL)

    def test_a_float_waveform_is_served_on_both_wires_and_a_put_still_lands(
        self, stub_server: _Server
    ) -> None:
        ca = _ca(STUB_WAVE)["value"]
        pva = stub_server.pva_get(STUB_WAVE)["value"]
        target = _ca(STUB_SETPOINT)["value"] + 0.5

        assert list(ca) == pytest.approx(STUB_WAVE_NOMINAL)
        assert list(pva) == pytest.approx(STUB_WAVE_NOMINAL)
        assert _caput(STUB_SETPOINT, target) == 1
        assert _ca(STUB_SETPOINT)["value"] == pytest.approx(target)

    def test_ticks_coalesce_under_a_slow_stub_and_a_put_lands_within_two_passes(
        self, stub_server: _Server
    ) -> None:
        stub_server.rpc("set", values={STUB_DELAY: SLOW_PASS_S}, token=TOKEN)
        try:
            stub_server.ask("start_sampling")
            time.sleep(20 * STUB_TICK_S)
            assert _caput(STUB_SETPOINT, 2.5) == 1
            time.sleep(4 * SLOW_PASS_S)
            observed = stub_server.ask("stop_sampling")
        finally:
            stub_server.rpc("set", values={STUB_DELAY: 0.0}, token=TOKEN)

        assert observed["max_depth"] <= 2
        assert observed["put_lags"] and max(observed["put_lags"]) <= 2
        assert _wait_until(lambda: _ca(STUB_SETPOINT)["value"] == 2.5)

    def test_a_pass_raising_after_an_accepted_put_leaves_the_pre_put_input(
        self, stub_server: _Server
    ) -> None:
        held = stub_server.ask("held", [STUB_SETPOINT])[STUB_SETPOINT]
        stub_server.ask("fail_next_pass", "the read after the put failed", True)

        assert _caput(STUB_SETPOINT, held + 3.0) == 1

        assert stub_server.ask("held", [STUB_SETPOINT]) == {STUB_SETPOINT: held}
        assert _ca(STUB_SETPOINT)["value"] == held
        assert not math.isclose(_ca(STUB_SETPOINT)["value"], held + 3.0)
