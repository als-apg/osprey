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
  ``tests/facility/golden/nominal_va.json``, written by
  ``uv run python scripts/va/capture_nominals.py --out
  tests/facility/golden/nominal_va.json`` and checked by its ``--check``; the
  served set is the golden's but for the two kinds the capture leaves out, the
  suite's synthetic string channel and every waveform, both held to the
  in-process composite instead;
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
initialises, so one process cannot be a client of two containers, and this
process's connectors run Channel Access on one CA loop that claims the
environment on first use and refuses to be repointed, so a client of these
containers made in-process would either be refused or pin the session to the
wrong port. The worker below is dispatched before this module imports anything
heavy and speaks JSON on stdout.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any

# The CA-client worker. Dispatched before this module imports pytest or any
# osprey package, so the client process stays a bare aioca client.


async def _worker(request: dict) -> dict:
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

    Every read is a fresh one off the wire: aioca keeps no monitor cache a
    read could be answered from.
    """
    import aioca

    op = request["op"]
    timeout = request.get("timeout", 30.0)

    def plain(value: Any) -> Any:
        """An aioca value as the JSON scalar or list it augments."""
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, float):
            return float(value)
        if isinstance(value, int):
            return int(value)
        return str(value)

    async def connect(addresses: list[str]) -> None:
        results = await aioca.connect(addresses, timeout=timeout, throw=False)
        missing = [
            address for address, result in zip(addresses, results, strict=True) if not result.ok
        ]
        if missing:
            raise RuntimeError(f"never connected to {missing[:10]} ({len(missing)} in all)")

    if op == "read":
        addresses = list(request["addresses"])
        await connect(addresses)
        channels = {}
        for address in addresses:
            data = await aioca.caget(
                address, format=aioca.FORMAT_TIME, timeout=timeout, throw=False
            )
            if not data.ok:
                raise RuntimeError(f"no value for {address}")
            channels[address] = {"value": plain(data), "severity": int(data.severity)}
        return {"channels": channels}

    if op == "put":
        accepted = {}
        for address, value in request["writes"]:
            put = await aioca.caput(address, value, wait=True, timeout=timeout, throw=False)
            accepted[address] = bool(put.ok)
        return {"accepted": accepted}

    if op == "watch":
        addresses = list(request["addresses"])
        await connect(addresses)
        events: dict[str, list] = {address: [] for address in addresses}
        primed = {address: asyncio.Event() for address in addresses}

        def on_update(value: Any, index: int) -> None:
            # The first update is the value the subscription connected to, so
            # it is consumed before anything is recorded.
            address = addresses[index]
            if primed[address].is_set():
                events[address].append([float(value.timestamp), float(value)])
            else:
                primed[address].set()

        subscriptions = aioca.camonitor(
            addresses, on_update, format=aioca.FORMAT_TIME, all_updates=True
        )
        try:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(event.wait() for event in primed.values())), timeout
                )
            except TimeoutError:
                silent = [address for address, event in primed.items() if not event.is_set()]
                raise RuntimeError(
                    f"never connected to {silent[:10]} ({len(silent)} in all)"
                ) from None
            deadline = time.monotonic() + request["seconds"]
            while time.monotonic() < deadline:
                if all(len(sink) >= request["count"] for sink in events.values()):
                    break
                await asyncio.sleep(0.2)
            return {"events": {address: list(sink) for address, sink in events.items()}}
        finally:
            for subscription in subscriptions:
                subscription.close()

    raise ValueError(f"unknown worker op {op!r}")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "--worker":
    # Leave through ``os._exit`` whichever way the operation ends. A normal
    # exit runs aioca's teardown hook, whose ``ca_context_destroy`` blocks for
    # as long as a name-server circuit is still retrying a port nothing serves
    # yet. The answer, or the traceback, is already written; a wedged exit
    # would only hold the parent's ``subprocess.run`` until its timeout, so one
    # wedge reads as a container that never served. That includes the failing
    # path: a read of an address not served yet has used Channel Access just
    # the same.
    try:
        print(json.dumps(asyncio.run(_worker(json.loads(sys.argv[2]))), default=str), flush=True)
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)

import math  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
from collections.abc import Iterator, Mapping  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from statistics import NormalDist  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

from osprey_connectors.control_system.va_in_process_connector import UDF_SEVERITY  # noqa: E402
from osprey_connectors.simulation import decode_char_waveform, pulse  # noqa: E402
from osprey_connectors.simulation.envelope import declares_motion, noise_sigma  # noqa: E402
from osprey_connectors.simulation.values import is_waveform  # noqa: E402
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

#: The captured nominal reads of the demo, every reading still: the capture of
#: ``scripts/va/capture_nominals.py``, which leaves out the suite's synthetic
#: string channel and every waveform.
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
#: declares noise or drift. The transfer line adds none: it carries no seeds.
EXPECTED_NOISY_CHANNELS = 751
#: The example facility's seeds, keyed by address.
DEMO_SEEDS = REPO_ROOT / "src" / "osprey" / "templates" / "facilities" / "example" / "seeds.yaml"
#: The address prefix of the demo's transfer line, the physics model solved single-pass.
LINE_PREFIX = "LINE:"
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

#: The served address the capture leaves out besides the waveforms: the suite's
#: synthetic string channel, which the demo does not carry.
SUITE_ADDITIONS = frozenset({e2e_conftest.STRING_CHANNEL})
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


def _is_trace(value: Any) -> bool:
    """Whether ``value`` is a waveform's trace: a sequence or array, never text."""
    return not isinstance(value, str) and (
        isinstance(value, (list, tuple)) or getattr(value, "ndim", 0) > 0
    )


#: How far the peaks of two captures of one pulse may differ, as a fraction of the
#: peak: the amplitude drift spans twice its bound, and the sample noise on the
#: flat top adds a few of its standard deviations.
TRACE_PEAK_TOL = 2 * pulse.AMPLITUDE_DRIFT_FRACTION + 6 * pulse.NOISE_FRACTION


def _same(address: str, expected: Any, served: Any, band: float = 0.0) -> bool:
    """Whether ``served`` reads as ``expected``.

    A trace is a capture keyed by the instant it is read, so two reads of one
    pulse never agree sample for sample: they agree in length and in peak, which
    follows the supply's setpoint and on-state.
    """
    if _is_trace(expected) or _is_trace(served):
        if not (_is_trace(expected) and _is_trace(served)):
            return False
        wanted = np.asarray(expected, dtype=np.float64)
        got = np.asarray(served, dtype=np.float64)
        if wanted.shape != got.shape:
            return False
        peak = float(np.max(np.abs(wanted)))
        return abs(float(np.max(np.abs(got))) - peak) <= band + TRACE_PEAK_TOL * max(1.0, peak)
    if isinstance(expected, str) or isinstance(served, str):
        return served == expected
    if band or isinstance(expected, float) or isinstance(served, float):
        return abs(float(served) - float(expected)) <= band + _tolerance(address) * max(
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

    line_monitors = [
        address
        for address in session_view.served
        if address.startswith(f"{LINE_PREFIX}DIAG:BPM:") and ":POSITION:" in address
    ]

    assert len(still) > len(session_view.addresses) // 2
    assert line_monitors
    assert set(line_monitors) <= set(still)
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
    # A trace is computed when it is read, not held: the composite's read is its value.
    traces = frozenset(a for a in session_view.addresses if is_waveform(session_view.value_type(a)))
    added = SUITE_ADDITIONS | traces
    z = _family_z(len(session_view.addresses))
    composite = _composite(session_view, va_container.state_dir)
    mock = composite.held(sorted(added))
    mock.update(composite.get(sorted(traces)))

    differing: dict[str, tuple[Any, Any]] = {}
    for address in session_view.addresses:
        served = _wire_value(session_view, address, nominal_reads[address]["value"])
        expected = mock[address] if address in added else golden[address]["nominal"]
        band = session_view.motion_band(address, z) if session_view.moving(address) else 0.0
        if not _same(address, expected, served, band):
            differing[address] = (expected, served)

    assert traces
    assert set(golden) - set(session_view.addresses) == set()
    assert set(session_view.addresses) - set(golden) == added
    assert differing == {}


def test_a_scenario_switch_returns_every_paired_texture_setpoint_to_its_seed(
    va_container: e2e_conftest.VaProject,
    session_view: View,
) -> None:
    view = session_view
    # The write moves each setpoint off its seed by a step, which only a float
    # setpoint takes; a bool or enum setpoint has labels, not a magnitude.
    pairs = {
        address: str(channel.pair)
        for address in view.served
        if (channel := view.channel(address)).owner == TEXTURE
        and channel.role == "setpoint"
        and channel.value_type == "float"
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
        "import asyncio, os, sys, aioca\n"
        f"v = asyncio.run(aioca.caget({e2e_conftest.READINESS_ADDRESS!r}, timeout=2.0, "
        "throw=False))\n"
        "sys.stdout.write('SERVED' if v.ok else 'NONE')\n"
        "sys.stdout.flush()\n"
        "os._exit(0)\n"
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


def test_the_transfer_line_carries_no_seeds() -> None:
    """The noisy count holds with the transfer line served: no seed names it."""
    seeds = yaml.safe_load(DEMO_SEEDS.read_text(encoding="utf-8"))

    assert seeds
    assert sorted(address for address in seeds if address.startswith(LINE_PREFIX)) == []


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


def test_a_worker_that_fails_leaves_as_promptly_as_one_that_answers() -> None:
    """A read of an address nothing serves is the worker's ordinary answer
    while a container is still booting, so that failure must return to the
    caller at once and carry its own error. An exit that ran libca's teardown
    instead would hold the caller for longer than the boot deadline.

    No container: the name server is a port nothing listens on.
    """
    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        port = unused.getsockname()[1]
    environment = {
        **os.environ,
        "EPICS_CA_NAME_SERVERS": f"localhost:{port}",
        "EPICS_CA_AUTO_ADDR_LIST": "NO",
    }
    request = {"op": "read", "addresses": ["PARITY:NOT:SERVED"], "timeout": 1.0}

    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, __file__, "--worker", json.dumps(request)],
        capture_output=True,
        text=True,
        timeout=120.0,
        env=environment,
    )
    elapsed = time.monotonic() - started

    assert result.returncode != 0
    assert "never connected to ['PARITY:NOT:SERVED']" in result.stderr
    # The connect timeout is 1 s; libca's teardown alone is tens of seconds.
    assert elapsed < 15.0, f"the failed worker took {elapsed:.1f}s to leave"


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_substrate_parity.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
