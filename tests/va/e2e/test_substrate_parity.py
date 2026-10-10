"""The virtual accelerator serves the same values in both venues, channel for channel.

Both venues serve one simulator view through one composite: in process, and
over Channel Access from the container.
This module holds the container to the in-process composite over the same
rendered view, on every served channel:

* **at a written operating point**, every channel without declared motion --
  the held values and the model-solved ones alike -- reads the same on the
  wire as in process, to 1e-9;
* **over the noisy channels**, N reads per substrate agree in mean and in
  spread, drift-detrended, within the bounds a z chosen for K channels at a
  family-wise false-alarm rate of 1e-4 allows;
* **across a scenario switch**, every paired texture setpoint and its readback
  return to their seed in both;
* **under a scenario the physics model cannot solve**, every channel reads
  with the alarm severity the composite gives it;
* **at nominal**, the served values are the captured
  ``tests/facility/golden/nominal_va.json``, every difference declared below
  by cause;
* **on every string channel**, the synthetic one the directory conftest adds
  among them, :func:`decode_char_waveform` reads the wire value as the text
  the composite holds.

**Two containers.** The held, nominal, scenario-switch, failed-scenario and
string checks use
this directory's session container (``conftest.va_container``), which serves
its monitors without their declared motion so a monitor reading is the solved
orbit. The noise statistics need the declared motion, so they run against a
module-scoped container of their own, booted from an unstilled render of the
same facility tree with a fast runner tick; that is this module's only extra
boot.

**Every Channel Access operation runs in a subprocess**, for the reasons the
directory conftest gives: libca latches the ``EPICS_CA_*`` environment when it
initialises, so one process cannot be a client of two containers, and a
main-thread pyepics call deadlocks the connector-based suites sharing this
process. The worker below is dispatched before this module imports anything
heavy and speaks JSON on stdout.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

# The CA-client worker. Dispatched before this module imports pytest or any
# osprey package, so the client process stays a bare pyepics client.


def _worker(request: dict) -> dict:
    """Run one scripted Channel Access operation and return its result.

    ``read``
        Connect each address and report its value, read off the wire, with
        its alarm severity. An enum reads as its index and a char waveform as
        its character codes.
    ``put``
        Write each value with put-completion, in order; report which writes
        the server accepted.
    ``watch``
        Subscribe to each address and collect ``[timestamp, value]`` monitor
        events until every address has ``count`` of them or ``seconds``
        elapse. The connection's own first update is not collected.
    """
    import epics

    op = request["op"]
    timeout = request.get("timeout", 30.0)

    def plain(value: Any) -> Any:
        return value.tolist() if hasattr(value, "tolist") else value

    def connect(addresses: list[str], *, monitor: bool = False) -> dict[str, Any]:
        pvs = {address: epics.PV(address, auto_monitor=monitor) for address in addresses}
        missing = [address for address, pv in pvs.items() if not pv.wait_for_connection(timeout)]
        if missing:
            raise RuntimeError(f"never connected to {missing[:10]} ({len(missing)} in all)")
        return pvs

    if op == "read":
        pvs = connect(request["addresses"])
        try:
            channels = {}
            for address, pv in pvs.items():
                data = pv.get_with_metadata(use_monitor=False, form="time", timeout=timeout)
                if data is None:
                    raise RuntimeError(f"no value for {address}")
                channels[address] = {
                    "value": plain(data["value"]),
                    "severity": data.get("severity"),
                }
            return {"channels": channels}
        finally:
            for pv in pvs.values():
                pv.disconnect()

    if op == "put":
        accepted = {}
        for address, value in request["writes"]:
            accepted[address] = bool(epics.caput(address, value, wait=True, timeout=timeout))
        return {"accepted": accepted}

    if op == "watch":
        pvs = connect(request["addresses"], monitor=True)
        events: dict[str, list] = {address: [] for address in pvs}
        for pv in pvs.values():
            pv.get()
        for address, pv in pvs.items():
            pv.add_callback(
                lambda value=None, timestamp=None, _sink=events[address], **_kw: _sink.append(
                    [timestamp, float(value)]
                )
            )
        try:
            deadline = time.monotonic() + request["seconds"]
            while time.monotonic() < deadline:
                if all(len(sink) >= request["count"] for sink in events.values()):
                    break
                time.sleep(0.2)
            return {"events": {address: list(sink) for address, sink in events.items()}}
        finally:
            # Explicit, and in a finally: a pyepics PV whose subscription is
            # still live segfaults libca when the garbage collector finalises
            # it at some arbitrary later point.
            for pv in pvs.values():
                pv.disconnect()

    raise ValueError(f"unknown worker op {op!r}")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "--worker":
    print(json.dumps(_worker(json.loads(sys.argv[2])), default=str), flush=True)
    # Leave through ``os._exit`` for the reason every other bare-pyepics child
    # in this directory does: a normal exit runs pyepics' ``finalize_libca``
    # atexit hook, which can wedge in ``ca_context_destroy`` after the channels
    # above were used. The answer is already on stdout; a wedged exit would
    # only hold the parent's ``subprocess.run`` until its timeout, so one wedge
    # reads as a container that never served.
    sys.stdout.flush()
    os._exit(0)

import math  # noqa: E402
import subprocess  # noqa: E402
from collections.abc import Iterator, Mapping  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from statistics import NormalDist  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from osprey_connectors.control_system.va_in_process_connector import UDF_SEVERITY  # noqa: E402
from osprey_connectors.simulation import decode_char_waveform  # noqa: E402
from osprey_connectors.simulation.envelope import declares_motion, noise_sigma  # noqa: E402
from osprey_connectors.simulation.view import (  # noqa: E402
    TEXTURE,
    VIEW_RELPATH,
    Channel,
    SimulatorView,
)
from tests.va.e2e import conftest as e2e_conftest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Floor for this module's own test count -- a guard against a refactor that
#: leaves the file importable but empty, which would otherwise pass silently.
MIN_COLLECTED_TESTS = 8

#: The captured nominal reads of the demo, taken with every reading's noise
#: zeroed.
NOMINAL_GOLDEN = REPO_ROOT / "tests" / "facility" / "golden" / "nominal_va.json"

#: Agreement between the wire and the in-process composite on a channel
#: without motion, relative to the value's magnitude and never below this.
EXACT_TOL = 1e-9
#: The chromaticity is a finite difference of two tune solves, so a solve's
#: last-bit rounding reaches it divided by the momentum step: the container's
#: platform and this process's agree on it to this, relative (a few parts in a
#: million apart on the demo), and on every other channel to :data:`EXACT_TOL`.
CHROMATICITY_TOL = 1e-5
CHROMATICITY = frozenset({"SR:DIAG:CHROM:X", "SR:DIAG:CHROM:Y"})

#: The noisy served channels of the demo: every served address whose seed
#: declares noise or drift.
EXPECTED_NOISY_CHANNELS = 626
#: Reads per substrate per noisy channel.
SAMPLES = 200
#: The family-wise false-alarm rate the noise bounds are set for.
FAMILY_ALPHA = 1e-4

#: The noise container's runner tick, so :data:`SAMPLES` reads take a minute.
NOISE_TICK_S = 0.25
#: The session container's runner tick, ``DEFAULT_TICK_S``: it is booted
#: without ``VA_POLL_INTERVAL_S``.
SESSION_TICK_S = 1.0
#: How long an apply may take to reach the session container: the composite
#: sees the new state on the runner pass after the file changes, and a desktop
#: container runtime's file sharing can report the change seconds late.
SWITCH_BOUND_S = 30.0
#: The burst gauge reads above this only while the synthetic burst scenario is
#: active: far above its seed noise, far below the override.
BURST_THRESHOLD = 1e-6

#: The operating point the held comparison is taken at: a horizontal and a
#: vertical corrector the other suites in this directory never write, so the
#: closed orbit is nonzero at every monitor, and a texture setpoint whose
#: write echoes into its readback.
OPERATING_POINT: dict[str, float] = {
    "SR:MAG:HCM:11:CURRENT:SP": 1.5,
    "SR:MAG:VCM:11:CURRENT:SP": -0.75,
    "SR:RF:CAVITY:01:VOLTAGE:SP": 2.6,
}

#: Every address whose served nominal is not the captured golden's, by cause.
#: An address here is held to the in-process composite instead.
NOMINAL_REBASELINES: dict[str, frozenset[str]] = {
    # The model owns the cavity frequency, and the deck states it.
    "lattice-owned: the value follows the deck": frozenset(
        {"SR:RF:CAVITY:01:FREQUENCY:RB", "SR:RF:CAVITY:01:FREQUENCY:SP"}
    ),
    # The facility's seeds state the ion-pump voltages the golden served as zero.
    "seeded: the value follows the seed's nominal": frozenset(
        f"SR:VAC:ION-PUMP:{index:02d}:VOLTAGE:{suffix}"
        for index in range(1, 7)
        for suffix in ("RB", "SP")
    ),
}
#: Served addresses the golden predates, by cause.
NOMINAL_ADDITIONS: dict[str, frozenset[str]] = {
    "lattice-owned: the model's optics outputs": frozenset(
        {"SR:DIAG:CHROM:X", "SR:DIAG:CHROM:Y", "SR:DIAG:TUNE:X", "SR:DIAG:TUNE:Y"}
    ),
    "the physics model's status channel": frozenset({"ca:SIM:SR:STATUS"}),
    "the suite's synthetic string channel": frozenset({e2e_conftest.STRING_CHANNEL}),
}
#: The physics model's status channel: ``ok`` while it solves, its error text
#: once it fails.
SR_STATUS = "ca:SIM:SR:STATUS"

#: Container-name prefix of the noise container; the pid follows, as for the
#: session container (see the directory conftest for why).
NOISE_CONTAINER = f"osprey-va-e2e-noise-{os.getpid()}"


# ---------------------------------------------------------------------------
# The view and the in-process composite
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class View:
    """A rendered simulator view, opened once through its reader."""

    reader: SimulatorView

    @classmethod
    def read(cls, path: Path) -> View:
        return cls(reader=SimulatorView.open(path))

    @property
    def path(self) -> Path:
        return self.reader.path

    @property
    def served(self) -> list[str]:
        """Every channel address the view lists."""
        return list(self.reader.channels())

    @property
    def status(self) -> list[str]:
        """Every served physics model's status address."""
        return list(self.reader.status_addresses().values())

    @property
    def addresses(self) -> list[str]:
        """Every address the view serves: its channels and its status addresses."""
        return [*self.served, *self.status]

    def channel(self, address: str) -> Channel:
        return self.reader.channel(address)

    def seed(self, address: str) -> Mapping[str, Any]:
        return self.reader.seed(address) or {}

    def moving(self, address: str) -> bool:
        return declares_motion(self.seed(address))

    def motion_band(self, address: str, z: float) -> float:
        """How far a served read may sit from the held value: z noise sigmas plus the drift."""
        seed = self.seed(address)
        drift = seed.get("drift") or {}
        sigma = noise_sigma(seed.get("noise"), seed.get("nominal"))
        return z * sigma + abs(float(drift.get("amplitude") or 0.0))

    def labels(self, address: str) -> list[str]:
        return list(self.channel(address).options or ("FALSE", "TRUE"))

    def value_type(self, address: str) -> str:
        if address in self.status:
            return "string"
        return str(self.channel(address).value_type or "float")


def _composite(view: View, state_dir: Path | None) -> Any:
    from osprey_connectors.simulation.composite import Composite

    return Composite(view.path, state_dir=state_dir, model_log=False)


def _wire_value(view: View, address: str, raw: Any) -> Any:
    """A wire value in the composite's stored representation."""
    value_type = view.value_type(address)
    if value_type in ("bool", "enum"):
        return view.labels(address)[int(raw)]
    if value_type == "string":
        return decode_char_waveform(raw)
    return raw


def _tolerance(address: str) -> float:
    return CHROMATICITY_TOL if address in CHROMATICITY else EXACT_TOL


def _same(address: str, expected: Any, served: Any) -> bool:
    if isinstance(expected, float) or isinstance(served, float):
        return abs(float(served) - float(expected)) <= _tolerance(address) * max(
            1.0, abs(float(expected))
        )
    return served == expected


def _family_z(count: int) -> float:
    """z = Φ⁻¹(1 − α/(2K)) for K channels at family-wise rate α."""
    return NormalDist().inv_cdf(1.0 - FAMILY_ALPHA / (2 * count))


# ---------------------------------------------------------------------------
# Channel Access, out of process
# ---------------------------------------------------------------------------


def _ca_call(port: int, request: dict, *, timeout: float = 600.0) -> dict:
    """Run one worker operation against the container published on ``port``."""
    environment = {
        **os.environ,
        "EPICS_CA_NAME_SERVERS": f"localhost:{port}",
        "EPICS_CA_AUTO_ADDR_LIST": "NO",
    }
    for stale in ("EPICS_CA_ADDR_LIST", "EPICS_CA_SERVER_PORT", "EPICS_CAS_SERVER_PORT"):
        environment.pop(stale, None)
    result = subprocess.run(
        [sys.executable, __file__, "--worker", json.dumps(request)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"CA worker failed ({request['op']}):\n{result.stdout}\n{result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def _read(port: int, addresses: list[str]) -> dict[str, dict[str, Any]]:
    return _ca_call(port, {"op": "read", "addresses": addresses})["channels"]


def _put(port: int, writes: Mapping[str, Any]) -> dict[str, bool]:
    return _ca_call(port, {"op": "put", "writes": list(writes.items())})["accepted"]


def _switch(project: e2e_conftest.VaProject, scenario: str) -> None:
    """``osprey sim apply`` a scenario and wait until the session container serves it.

    Only the synthetic burst scenario and ``nominal`` are switched between, so
    the burst gauge tells which of the two the container serves.
    """
    applied = project.sim_apply(scenario)
    assert applied.returncode == 0, applied.stdout + applied.stderr
    burst = scenario == e2e_conftest.BURST_SCENARIO_NAME
    deadline = time.monotonic() + SWITCH_BOUND_S
    while True:
        gauge = _read(e2e_conftest.CA_PORT, [e2e_conftest.BURST_CHANNEL])
        if (gauge[e2e_conftest.BURST_CHANNEL]["value"] > BURST_THRESHOLD) == burst:
            return
        assert time.monotonic() < deadline, f"the container never served {scenario!r}"
        time.sleep(0.2)


def _to_nominal(project: e2e_conftest.VaProject) -> None:
    """Serve ``nominal`` from a fresh rebuild, every earlier session write dropped.

    Through the burst scenario, so the composite sees a changed state file
    twice and the return to ``nominal`` is observed rather than assumed.
    """
    _switch(project, e2e_conftest.BURST_SCENARIO_NAME)
    _switch(project, "nominal")


# ---------------------------------------------------------------------------
# The session container
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def session_view(va_container: e2e_conftest.VaProject) -> View:
    """The view the session container serves, monitors stilled."""
    return View.read(va_container.data_dir / VIEW_RELPATH.name)


@pytest.fixture(scope="module")
def nominal_reads(va_container: e2e_conftest.VaProject, session_view: View) -> dict[str, Any]:
    """Every served address as the session container serves it at nominal."""
    _to_nominal(va_container)
    return _read(e2e_conftest.CA_PORT, session_view.addresses)


@pytest.fixture(scope="module")
def operating_point(
    va_container: e2e_conftest.VaProject,
    session_view: View,
    nominal_reads: dict[str, Any],  # noqa: ARG001 - the nominal reads are taken first
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The wire and the composite at :data:`OPERATING_POINT`.

    Returns:
        The wire reads, the composite's reads and its held values, by address.
    """
    _to_nominal(va_container)
    accepted = _put(e2e_conftest.CA_PORT, OPERATING_POINT)
    assert all(accepted.values()), accepted
    mock = _composite(session_view, va_container.state_dir)
    mock.set(dict(OPERATING_POINT))
    # A write's own pass leaves the chromaticity to the next periodic pass.
    time.sleep(2 * SESSION_TICK_S)
    wire = _read(e2e_conftest.CA_PORT, session_view.addresses)
    return wire, mock.get(session_view.addresses), mock.held(session_view.addresses)


def test_every_channel_without_motion_agrees_at_the_operating_point(
    session_view: View, operating_point: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
) -> None:
    wire, mock, _held = operating_point
    still = [address for address in session_view.addresses if not session_view.moving(address)]

    differing = {
        address: (mock[address], _wire_value(session_view, address, wire[address]["value"]))
        for address in still
        if not _same(
            address, mock[address], _wire_value(session_view, address, wire[address]["value"])
        )
    }

    assert len(still) > len(session_view.addresses) // 2
    assert differing == {}


def test_every_moving_channel_reads_its_held_value_within_its_motion(
    session_view: View, operating_point: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
) -> None:
    wire, _mock, held = operating_point
    z = _family_z(len(session_view.addresses))
    moving = [address for address in session_view.addresses if session_view.moving(address)]

    outside = {
        address: (held[address], wire[address]["value"])
        for address in moving
        if abs(float(wire[address]["value"]) - float(held[address]))
        > session_view.motion_band(address, z) + EXACT_TOL * max(1.0, abs(float(held[address])))
    }

    assert moving
    assert outside == {}


def test_the_operating_point_moves_the_orbit_at_every_monitor(
    session_view: View, operating_point: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
) -> None:
    """Anti-vacuous guard: at rest the closed orbit is zero at every monitor,
    and the agreement above would be a comparison of zeros."""
    wire, _mock, _held = operating_point
    monitors = [
        address
        for address in session_view.served
        if address.startswith("SR:DIAG:BPM:") and ":POSITION:" in address
    ]

    assert monitors
    assert [address for address in monitors if wire[address]["value"] == 0.0] == []


def test_the_served_nominals_are_the_captured_golden(
    session_view: View, nominal_reads: dict[str, Any], va_container: e2e_conftest.VaProject
) -> None:
    golden = json.loads(NOMINAL_GOLDEN.read_text(encoding="utf-8"))["channels"]
    rebaselined = frozenset().union(*NOMINAL_REBASELINES.values())
    added = frozenset().union(*NOMINAL_ADDITIONS.values())
    z = _family_z(len(session_view.addresses))
    mock = _composite(session_view, va_container.state_dir).held(session_view.addresses)

    differing: dict[str, tuple[Any, Any]] = {}
    for address in session_view.addresses:
        raw = nominal_reads[address]["value"]
        if address in rebaselined or address in added:
            expected = mock[address]
            served = _wire_value(session_view, address, raw)
        else:
            expected = golden[address]["nominal"]
            served = raw
        band = session_view.motion_band(address, z) if session_view.moving(address) else 0.0
        if isinstance(expected, str):
            ok = served == expected
        else:
            ok = abs(float(served) - float(expected)) <= band + _tolerance(address) * max(
                1.0, abs(float(expected))
            )
        if not ok:
            differing[address] = (expected, served)

    assert set(golden) - set(session_view.addresses) == set()
    assert set(session_view.addresses) - set(golden) == added
    assert differing == {}


def test_every_declared_rebaseline_really_differs_from_the_golden(
    session_view: View, nominal_reads: dict[str, Any]
) -> None:
    """The declared set holds only real differences: an address that came back
    to its golden value leaves it."""
    golden = json.loads(NOMINAL_GOLDEN.read_text(encoding="utf-8"))["channels"]
    z = _family_z(len(session_view.addresses))

    unchanged = sorted(
        address
        for addresses in NOMINAL_REBASELINES.values()
        for address in addresses
        if abs(float(nominal_reads[address]["value"]) - float(golden[address]["nominal"]))
        <= session_view.motion_band(address, z)
    )

    assert unchanged == []


def test_a_scenario_switch_returns_every_paired_texture_setpoint_to_its_seed(
    va_container: e2e_conftest.VaProject,
    session_view: View,
) -> None:
    view = session_view
    pairs = {
        address: str(channel.pair)
        for address in view.served
        if (channel := view.channel(address)).owner == TEXTURE
        and channel.role == "setpoint"
        and channel.pair not in (None, address)
    }
    seeds = {address: float(view.seed(address)["nominal"]) for address in pairs}
    written = {
        address: seed + 0.1 * abs(seed) + 1.0
        for address, seed in seeds.items()
        if view.channel(address).writable
    }
    z = _family_z(len(view.addresses))

    _to_nominal(va_container)
    mock = _composite(view, va_container.state_dir)
    accepted = _put(e2e_conftest.CA_PORT, written)
    mock.set(dict(written))
    before = _read(e2e_conftest.CA_PORT, list(written))
    _switch(va_container, e2e_conftest.BURST_SCENARIO_NAME)
    try:
        after = _read(e2e_conftest.CA_PORT, [*pairs, *pairs.values()])
        held = mock.held([*pairs, *pairs.values()])
    finally:
        _switch(va_container, "nominal")

    assert pairs and all(accepted.values()), accepted
    assert {address: before[address]["value"] for address in written} == written
    assert {address: after[address]["value"] for address in pairs} == seeds
    assert {address: held[address] for address in pairs} == seeds
    assert {readback: held[readback] for readback in pairs.values()} == {
        readback: seeds[setpoint] for setpoint, readback in pairs.items()
    }
    assert {
        readback: after[readback]["value"]
        for setpoint, readback in pairs.items()
        if abs(after[readback]["value"] - seeds[setpoint]) > view.motion_band(readback, z)
    } == {}


def test_every_string_channel_decodes_to_the_text_the_composite_holds(
    session_view: View, nominal_reads: dict[str, Any], va_container: e2e_conftest.VaProject
) -> None:
    strings = [
        address
        for address in session_view.addresses
        if session_view.value_type(address) == "string"
    ]
    mock = _composite(session_view, va_container.state_dir).get(strings)

    decoded = {
        address: decode_char_waveform(nominal_reads[address]["value"]) for address in strings
    }

    assert e2e_conftest.STRING_CHANNEL in strings
    assert decoded == mock
    assert decoded[e2e_conftest.STRING_CHANNEL] == e2e_conftest.STRING_NOMINAL


def _sr_status() -> str:
    return decode_char_waveform(_read(e2e_conftest.CA_PORT, [SR_STATUS])[SR_STATUS]["value"])


def test_a_failed_scenario_reads_with_the_composites_severity(
    va_container: e2e_conftest.VaProject, session_view: View
) -> None:
    """A channel of the failed model reads ``udf``'s severity on the wire and
    in process alike; every other channel reads none."""
    _to_nominal(va_container)
    applied = va_container.sim_apply(e2e_conftest.UNSTABLE_SCENARIO_NAME)
    assert applied.returncode == 0, applied.stdout + applied.stderr
    try:
        deadline = time.monotonic() + SWITCH_BOUND_S
        while _sr_status() == "ok":
            assert time.monotonic() < deadline, "the container never failed the physics model"
            time.sleep(0.2)
        wire = _read(e2e_conftest.CA_PORT, session_view.addresses)
        mock = _composite(session_view, va_container.state_dir)
        failed = mock.output_severity(session_view.addresses)
        mock_status = mock.status("SR")
    finally:
        _to_nominal(va_container)

    expected = {
        address: UDF_SEVERITY if address in failed else 0 for address in session_view.addresses
    }
    differing = {
        address: (expected[address], wire[address]["severity"])
        for address in session_view.addresses
        if wire[address]["severity"] != expected[address]
    }

    assert failed and mock_status != "ok"
    assert differing == {}


# ---------------------------------------------------------------------------
# The noise container
# ---------------------------------------------------------------------------


def _docker(*args: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def _served(port: int) -> bool:
    """Whether the container published on ``port`` serves the readiness channel.

    Asked out of process, leaving through ``os._exit`` for the reason the
    directory conftest's probe gives.
    """
    code = (
        "import sys, epics\n"
        f"v = epics.caget({e2e_conftest.READINESS_ADDRESS!r}, timeout=1.0, "
        "connection_timeout=1.0)\n"
        "sys.stdout.write('SERVED' if v is not None else 'NONE')\n"
        "sys.stdout.flush()\n"
        "import os; os._exit(0)\n"
    )
    environment = {
        **os.environ,
        "EPICS_CA_NAME_SERVERS": f"localhost:{port}",
        "EPICS_CA_AUTO_ADDR_LIST": "NO",
    }
    for stale in ("EPICS_CA_ADDR_LIST", "EPICS_CA_SERVER_PORT"):
        environment.pop(stale, None)
    try:
        probe = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
            env=environment,
        )
    except subprocess.TimeoutExpired:
        return False
    return probe.stdout.strip() == "SERVED"


@dataclass(frozen=True)
class NoisyVa:
    """The noise container: its published port and the view it serves."""

    port: int
    view: View


@pytest.fixture(scope="module")
def noisy_va(tmp_path_factory: pytest.TempPathFactory) -> Iterator[NoisyVa]:
    """A container serving an unstilled render of the session container's facility.

    Booted and torn down the way the session container is: one name, removed
    on the way in as stale-cleanup and on the way out whatever happens.
    """
    root = e2e_conftest.stage_demo_data_dir(
        tmp_path_factory.mktemp("va_noise_data"), still_monitors=False
    )

    def container(port: int) -> tuple[str, list[str]]:
        return NOISE_CONTAINER, [
            "run",
            "-d",
            "--name",
            NOISE_CONTAINER,
            "-p",
            f"127.0.0.1:{port}:{e2e_conftest.CONTAINER_CA_PORT}/tcp",
            *e2e_conftest.data_root_run_args(root),
            "-e",
            f"VA_POLL_INTERVAL_S={NOISE_TICK_S}",
            *e2e_conftest.DEMO_NAMESPACE_RUN_ARGS,
            e2e_conftest.IMAGE,
        ]

    port, _ = e2e_conftest.run_on_free_port(container)
    try:
        deadline = time.monotonic() + e2e_conftest.CONTAINER_BOOT_TIMEOUT_S
        while not _served(port):
            if time.monotonic() > deadline:
                logs = _docker("logs", "--tail", "40", NOISE_CONTAINER)
                raise RuntimeError(
                    f"{NOISE_CONTAINER} never served {e2e_conftest.READINESS_ADDRESS} within "
                    f"{e2e_conftest.CONTAINER_BOOT_TIMEOUT_S}s.\n"
                    f"{e2e_conftest.boot_report(NOISE_CONTAINER, port)}\n"
                    f"{logs.stdout}\n{logs.stderr}"
                )
            time.sleep(0.5)
        yield NoisyVa(port=port, view=View.read(root / VIEW_RELPATH.name))
    finally:
        _docker("rm", "-f", NOISE_CONTAINER)


def test_the_unstilled_render_serves_the_demos_noisy_channels(noisy_va: NoisyVa) -> None:
    noisy = [address for address in noisy_va.view.served if noisy_va.view.moving(address)]

    assert len(noisy) == EXPECTED_NOISY_CHANNELS


@pytest.fixture(scope="module")
def noise_samples(noisy_va: NoisyVa) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """:data:`SAMPLES` reads of every noisy channel from each substrate.

    The wire reads are monitor events, each stamped by the server; the
    composite is read at those same instants, so the drift both carry is the
    same function of the same times.

    Returns:
        ``{address: (times, wire, composite)}``.
    """
    view = noisy_va.view
    noisy = [address for address in view.served if view.moving(address)]
    seconds = 20 * SAMPLES * NOISE_TICK_S
    events = _ca_call(
        noisy_va.port,
        {"op": "watch", "addresses": noisy, "count": SAMPLES, "seconds": seconds},
        timeout=seconds + 300.0,
    )["events"]
    short = {address: len(events[address]) for address in noisy if len(events[address]) < SAMPLES}
    assert short == {}, f"channels with fewer than {SAMPLES} reads: {short}"

    mock = _composite(view, None)
    groups = {address: mock.readout_group(address) for address in noisy}
    held = mock.held(sorted({name for group in groups.values() for name in group}))
    samples: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for address in noisy:
        stamped = np.asarray(events[address][:SAMPLES], dtype=np.float64)
        times, wire = stamped[:, 0], stamped[:, 1]
        levels = {name: float(held[name]) for name in groups[address]}
        samples[address] = (times, wire, mock.readings(levels, times)[address])
    return samples


def _detrended_sigma(times: np.ndarray, values: np.ndarray) -> float:
    """The spread about the least-squares line through the samples."""
    centred = times - times.mean()
    slope, intercept = np.polyfit(centred, values, 1)
    residuals = values - (slope * centred + intercept)
    return float(np.sqrt(np.sum(residuals**2) / (len(values) - 2)))


def test_the_noisy_channels_agree_in_mean_and_spread(
    noise_samples: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    """Criterion bounds over K channels, z = Φ⁻¹(1 − 1e-4/(2K)):
    ``|mean_mock − mean_va| ≤ z·σ·sqrt(2/N)`` and
    ``|ln(σ_mock/σ_va)| ≤ z·sqrt(1/(N−1))``, σ drift-detrended."""
    count = len(noise_samples)
    z = _family_z(count)
    mean_bound = math.sqrt(2.0 / SAMPLES)
    spread_bound = z * math.sqrt(1.0 / (SAMPLES - 1))

    failures: dict[str, str] = {}
    for address, (times, wire, composite) in noise_samples.items():
        sigma_wire = _detrended_sigma(times, wire)
        sigma_mock = _detrended_sigma(times, composite)
        if sigma_wire <= 0.0 or sigma_mock <= 0.0:
            failures[address] = f"no spread (wire {sigma_wire}, composite {sigma_mock})"
            continue
        sigma = math.sqrt((sigma_wire**2 + sigma_mock**2) / 2.0)
        mean_gap = abs(float(composite.mean()) - float(wire.mean()))
        spread_gap = abs(math.log(sigma_mock / sigma_wire))
        if mean_gap > z * sigma * mean_bound:
            failures[address] = f"means differ by {mean_gap}, bound {z * sigma * mean_bound}"
        elif spread_gap > spread_bound:
            failures[address] = f"|ln(σ ratio)| {spread_gap}, bound {spread_bound}"

    assert count == EXPECTED_NOISY_CHANNELS
    assert failures == {}


# ---------------------------------------------------------------------------


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_substrate_parity.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
