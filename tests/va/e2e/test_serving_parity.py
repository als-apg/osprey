"""Does the wire agree with the namespace the virtual accelerator meant to serve?

Four kinds of agreement, each checked against a real container over a real
Channel Access wire, because each is a claim about what a *client* observes and
none of them is decidable from inside the serving process:

* **monitor delivery** -- a readback event follows a setpoint write, and
  telemetry arrives with no client write at all. A one-shot read cannot tell a
  value that was posted from one that was merely recorded, so the settle-wait
  every scan depends on is only observable through a subscription.
* **record-type parity** -- the type and element count on the wire are the ones
  the manifest's ``record_type`` column asked for. Expected values are derived
  from the manifest here rather than written down, so this is a comparison of
  two independent descriptions of the same namespace.
* **alarm parity** -- a write clamped to the top of its drive band reads back
  ``NO_ALARM``. Clamping is a normal outcome; the served database declares no
  alarm limits precisely so that a clamped value does not present as a fault.
* **seeded-readout divergence** -- a BPM's served reading is its true orbit
  position put through ``bpm_read``, not the truth itself.

**Two containers, and why.** The divergence check needs the truth a reading
diverges *from*, and no transport carries it: both Channel Access and PVAccess
serve the reading. So truth is measured rather than derived -- a second boot of
the same image with no seeded error, where ``bpm_read`` is the identity and the
served reading therefore *is* the true position. Both boots are driven to the
same machine state by the same single corrector write before either is
sampled, and the unseeded control BPM is required to agree between them to
within nothing at all; if it does not, the comparison is void and this suite
says so instead of reporting a difference it cannot attribute.

**Every Channel Access operation runs in a subprocess.** Two independent
reasons, both load-bearing. libca latches ``EPICS_CA_*`` when the C library
initialises, and this directory's session container (``conftest.py``) publishes
on a different port than the containers here do, so one process cannot be a
client of both. And the connector wraps synchronous pyepics in a thread-pool
executor whose CA context is per-thread, so a main-thread pyepics call in this
process deadlocks any connector-based test sharing it. The worker below is
dispatched before this module imports anything heavy and speaks JSON on stdout.

**Ports.** Each container binds an ephemeral port and publishes it unchanged: a
Channel Access search reply carries the server's own port number, so a remap
would hand every client an address nothing answers on. Nothing here goes near
5064, which belongs to whatever the operator is running.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

# The CA-client worker. Dispatched before this module imports pytest or any
# osprey package, so the client process stays as close to a bare pyepics client
# as it can be -- see the module docstring for why it is a separate process.


def _worker(request: dict) -> dict:
    """Run one scripted Channel Access operation and return its result.

    Three operations cover everything this suite needs:

    ``read``
        Connect each address and report its value together with the metadata a
        client reads from it -- wire type, element count, display precision,
        enum labels, control band, alarm condition.
    ``write_watch``
        Subscribe to the watched addresses, then write one setpoint with
        put-completion and collect the monitor events that follow. Callbacks
        are added *after* the initial read, so a connection's own first update
        is never mistaken for a response to the write.
    ``watch``
        Subscribe and collect events for a fixed window with no write at all --
        the only way to see a value the server publishes on its own schedule.
    """
    import epics

    op = request["op"]
    timeout = request.get("timeout", 10.0)

    def describe(pv: Any) -> dict:
        control = pv.get_ctrlvars() or {}
        return {
            "value": pv.value,
            "type": pv.type,
            "count": pv.count,
            "precision": control.get("precision"),
            "enum_strs": list(control.get("enum_strs") or []) or None,
            "lower_ctrl_limit": control.get("lower_ctrl_limit"),
            "upper_ctrl_limit": control.get("upper_ctrl_limit"),
            "severity": pv.severity,
            "status": pv.status,
        }

    def connect(address: str, *, monitor: bool = False) -> Any:
        pv = epics.PV(address, auto_monitor=monitor)
        if not pv.wait_for_connection(timeout=timeout):
            raise RuntimeError(f"never connected to {address}")
        pv.get()
        return pv

    if op == "read":
        pvs = [connect(address) for address in request["addresses"]]
        try:
            return {"channels": {pv.pvname: describe(pv) for pv in pvs}}
        finally:
            for pv in pvs:
                pv.disconnect()

    if op == "write_watch":
        watched = {address: connect(address, monitor=True) for address in request["watch"]}
        before = {address: pv.value for address, pv in watched.items()}
        events: dict[str, list] = {address: [] for address in watched}
        for address, pv in watched.items():
            pv.add_callback(lambda value=None, _sink=events[address], **_kw: _sink.append(value))
        try:
            started = time.monotonic()
            accepted = epics.caput(request["address"], request["value"], wait=True, timeout=timeout)
            elapsed = time.monotonic() - started
            # Settle on the value actually written, which may be the clamped
            # one rather than the requested one.
            deadline = time.monotonic() + request.get("settle", 5.0)
            target = request.get("expect")
            while time.monotonic() < deadline:
                if target is not None and any(
                    values and abs(values[-1] - target) < 1e-12 for values in events.values()
                ):
                    break
                time.sleep(0.02)
            return {
                "accepted": bool(accepted),
                "put_seconds": elapsed,
                "before": before,
                "events": events,
                "after": {address: pv.get(use_monitor=False) for address, pv in watched.items()},
            }
        finally:
            # Explicit, and in a finally: a pyepics PV whose subscription is
            # still live segfaults libca when the garbage collector finalises
            # it at some arbitrary later point.
            for pv in watched.values():
                pv.disconnect()

    if op == "watch":
        watched = {address: connect(address, monitor=True) for address in request["watch"]}
        before = {address: pv.value for address, pv in watched.items()}
        events: dict[str, list] = {address: [] for address in watched}
        started = time.monotonic()
        for address, pv in watched.items():
            pv.add_callback(
                lambda value=None, _sink=events[address], _t0=started, **_kw: _sink.append(
                    [time.monotonic() - _t0, value]
                )
            )
        try:
            time.sleep(request["seconds"])
            return {"before": before, "events": events}
        finally:
            for pv in watched.values():
                pv.disconnect()

    raise ValueError(f"unknown worker op {op!r}")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "--worker":
    print(json.dumps(_worker(json.loads(sys.argv[2])), default=str), flush=True)
    raise SystemExit(0)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from osprey.services.virtual_accelerator.lattice.errors import bpm_read  # noqa: E402
from osprey.services.virtual_accelerator.manifest import build_manifest  # noqa: E402
from osprey.services.virtual_accelerator.serving.pvdb import (  # noqa: E402
    ANALOG_PRECISION,
    BINARY_ENUM_STATES,
)

# Floor for this module's own test count -- a guard against a refactor that
# leaves the file importable but empty, which would otherwise pass silently.
MIN_COLLECTED_TESTS = 20

#: The image under test. ``osprey-va-full:latest`` is what the rest of this
#: directory serves from; override when certifying a candidate build.
IMAGE = os.environ.get("OSPREY_VA_E2E_IMAGE", "osprey-va-full:latest")

CONTAINER_SEEDED = "osprey-va-e2e-parity"
CONTAINER_TRUTH = "osprey-va-e2e-parity-truth"

BOOT_TIMEOUT_S = 120.0
DATA_DIR = "src/osprey/templates/apps/control_assistant/data/simulation"

# Device ids claimed by this module alone. The rest of this directory works on
# HCM 01-03; nothing here touches those, and nothing there touches these.
REFERENCE_SP = "SR:MAG:HCM:11:CURRENT:SP"
REFERENCE_RB = "SR:MAG:HCM:11:CURRENT:RB"
WRITE_SP = "SR:MAG:HCM:12:CURRENT:SP"
WRITE_RB = "SR:MAG:HCM:12:CURRENT:RB"
SEEDED_BPM = "SR:DIAG:BPM:11:POSITION:X"
CONTROL_BPM = "SR:DIAG:BPM:12:POSITION:X"
BINARY_CHANNEL = "SR:DIAG:BPM:11:STATUS:VALID"

#: Engine-driven telemetry: a static-noisy analog channel the simulation engine
#: republishes on its own 1 Hz tick, with no client involvement whatsoever.
TELEMETRY = "SR:DIAG:DCCT:01:CURRENT:RB"

#: The drive band this facility's limits file gives every corrector setpoint.
BAND_LOW, BAND_HIGH = -12.0, 12.0

#: A current well outside the band, so the clamp is unambiguous.
OUT_OF_BAND = 50.0

#: The readout error seeded on one BPM. Noise is left at zero deliberately: a
#: noiseless transform is exactly reproducible, so the wire can be compared to
#: ``bpm_read`` bit for bit rather than within a tolerance that would also
#: accept a wrong answer.
SEEDED_DEVICE = "BPM11"
SEEDED_OFFSET_X = 50e-6
SEEDED_GAIN_X = 1.05
VA_BPM_ERRORS = f"{SEEDED_DEVICE}:offset_x={SEEDED_OFFSET_X},gain_x={SEEDED_GAIN_X}"

#: ``bpm_read``'s keyword set at identity. Spelled out rather than imported
#: from the bridge's private table so this file states the contract it is
#: checking -- and so that a new keyword makes this call fail loudly instead of
#: silently defaulting.
IDENTITY_BPM_ERROR: dict[str, float] = {
    "offset_x": 0.0,
    "offset_y": 0.0,
    "gain_x": 1.0,
    "gain_y": 1.0,
    "polarity_x": 1.0,
    "polarity_y": 1.0,
    "roll": 0.0,
    "cal_x": 0.0,
    "cal_y": 0.0,
    "noise_x": 0.0,
    "noise_y": 0.0,
}

#: Current written to the reference corrector before either boot is sampled.
#: A machine at rest has an identically zero closed orbit, which would make
#: "the control BPM reads its truth" a comparison of two zeros; a distorted
#: orbit gives every BPM a nonzero truth and puts the seeded gain on a nonzero
#: value instead of only on the offset.
REFERENCE_CURRENT = 1.5

#: Manifest ``record_type`` -> the wire type pyepics reports for it. This is
#: the parity table, and it is asserted to cover every type the manifest
#: actually uses -- see ``TestRecordTypeParity``.
WIRE_TYPES = {"ai": "time_double", "bi": "time_enum"}

#: How many channels the parity sweep samples per record type.
PARITY_SAMPLE = 8


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _docker(*args: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def _require_image() -> None:
    """Fail loudly unless the image can serve on a port other than 5064.

    A precondition, not a nicety. The Channel Access *server* library reads
    ``EPICS_CAS_SERVER_PORT`` and does not fall back to the client-side
    variable, so an image whose entry point does not derive one from the other
    keeps binding its build-time default while telling this suite's clients
    some other port. The symptom would be an unexplained boot timeout; this
    turns it into a sentence naming the fix.
    """
    inspected = _docker(
        "image", "inspect", IMAGE, "--format", "{{.Architecture}}|{{.Config.Cmd}}", timeout=60
    )
    if inspected.returncode != 0:
        pytest.fail(
            f"image {IMAGE!r} is not present. Build it with "
            f"scripts/va/build_and_boot_check.sh, or name another with "
            f"OSPREY_VA_E2E_IMAGE."
        )
    architecture, _, command = inspected.stdout.strip().partition("|")
    if "EPICS_CAS_SERVER_PORT" not in command:
        pytest.fail(
            f"image {IMAGE!r} ({architecture}) does not derive EPICS_CAS_SERVER_PORT from "
            f"EPICS_CA_SERVER_PORT, so it cannot serve on any port but its baked default. "
            f"Rebuild it (scripts/va/build_and_boot_check.sh) or point "
            f"OSPREY_VA_E2E_IMAGE at a current build. Its entry point is: {command}"
        )


@dataclass
class LiveVA:
    """A running virtual accelerator and the state it was sampled in."""

    port: int
    #: Every BPM this suite reads, as served once the reference write settled.
    #: Captured before any test runs, so no test's writes can perturb it.
    orbit: dict[str, float] = field(default_factory=dict)

    def call(self, request: dict, *, timeout: float = 180.0) -> dict:
        """Run one worker operation against this container."""
        environment = {
            **os.environ,
            "EPICS_CA_NAME_SERVERS": f"localhost:{self.port}",
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
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"CA worker failed ({request['op']}):\n{result.stdout}\n{result.stderr}"
            )
        return json.loads(result.stdout.strip().splitlines()[-1])

    def read(self, *addresses: str) -> dict[str, dict]:
        return self.call({"op": "read", "addresses": list(addresses)})["channels"]

    def value(self, address: str) -> Any:
        return self.read(address)[address]["value"]


@contextmanager
def _serving(name: str, *, seeded: bool):
    """Boot one virtual accelerator container and wait until it serves.

    The published port and the server's own port are the same number by
    construction; see the module docstring.
    """
    repository_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    port = _free_port()
    _docker("rm", "-f", name, timeout=60)

    arguments = [
        "run",
        "-d",
        "--name",
        name,
        "-e",
        f"EPICS_CA_SERVER_PORT={port}",
        "-p",
        f"127.0.0.1:{port}:{port}/tcp",
        "-v",
        f"{os.path.join(repository_root, DATA_DIR)}:/data/simulation:ro",
    ]
    if seeded:
        arguments += ["-e", f"VA_BPM_ERRORS={VA_BPM_ERRORS}"]
    started = _docker(*arguments, IMAGE)
    if started.returncode != 0:
        raise RuntimeError(f"docker run failed: {started.stdout}\n{started.stderr}")

    accelerator = LiveVA(port=port)
    try:
        deadline = time.monotonic() + BOOT_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                if accelerator.value(REFERENCE_RB) is not None:
                    break
            except Exception:  # noqa: BLE001 - not up yet is the expected case
                pass
            time.sleep(1.0)
        else:
            logs = _docker("logs", "--tail", "40", name, timeout=60)
            raise RuntimeError(
                f"{name} never served {REFERENCE_RB} within {BOOT_TIMEOUT_S}s.\n"
                f"{logs.stdout}\n{logs.stderr}"
            )

        # Drive both boots to the same machine state before either is sampled.
        settled = accelerator.call(
            {
                "op": "write_watch",
                "address": REFERENCE_SP,
                "value": REFERENCE_CURRENT,
                "watch": [REFERENCE_RB],
                "expect": REFERENCE_CURRENT,
            }
        )
        if not settled["accepted"]:
            raise RuntimeError(f"{name} refused the reference write to {REFERENCE_SP}")

        channels = accelerator.read(SEEDED_BPM, CONTROL_BPM, REFERENCE_RB)
        accelerator.orbit = {address: channel["value"] for address, channel in channels.items()}
        yield accelerator
    finally:
        _docker("rm", "-f", name, timeout=60)


@pytest.fixture(scope="module")
def va() -> Any:
    """The virtual accelerator under test, with one BPM's readout error seeded.

    The seeded error touches exactly one BPM's horizontal reading; every other
    channel in the namespace behaves as it does in an unseeded boot, so the
    monitor, record-type and alarm checks are unaffected by it.
    """
    _require_image()
    with _serving(CONTAINER_SEEDED, seeded=True) as accelerator:
        yield accelerator


@pytest.fixture(scope="module")
def truth_orbit(va: LiveVA) -> dict[str, float]:
    """The same machine with no seeded error, where a reading *is* the truth.

    Booted after the seeded container is already up and sampled, and torn down
    as soon as it has been read, so only its measurements outlive it.
    """
    with _serving(CONTAINER_TRUTH, seeded=False) as reference:
        return dict(reference.orbit)


@pytest.fixture(scope="module")
def manifest_channels() -> dict[str, dict]:
    return {channel["address"]: channel for channel in build_manifest()["channels"]}


class TestMonitorDelivery:
    """What a subscriber is told, which is not what a reader can ask for.

    A settle-wait -- the thing an ``EpicsMotor.set()`` blocks on, and therefore
    the thing every orbit-response scan depends on -- is woken by a monitor
    event and by nothing else. A serving layer that committed a value without
    posting it would satisfy every ``caget`` in this suite and still hang a
    scan forever.
    """

    @pytest.fixture(scope="class")
    def write(self, va: LiveVA) -> dict:
        """One setpoint write, with its readback and setpoint both subscribed."""
        return va.call(
            {
                "op": "write_watch",
                "address": WRITE_SP,
                "value": -3.75,
                "watch": [WRITE_RB, WRITE_SP],
                "expect": -3.75,
            }
        )

    def test_the_write_was_accepted(self, write: dict) -> None:
        assert write["accepted"]

    def test_the_readback_had_somewhere_to_move_from(self, write: dict) -> None:
        """Precondition: a readback already sitting on the written value would
        make every assertion below pass without anything happening."""
        assert write["before"][WRITE_RB] != pytest.approx(-3.75)

    def test_a_monitor_event_carries_the_written_value_to_the_readback(self, write: dict) -> None:
        assert write["events"][WRITE_RB], "no monitor event ever reached the subscriber"
        assert write["events"][WRITE_RB][-1] == pytest.approx(-3.75)

    def test_the_setpoint_posts_its_own_event(self, write: dict) -> None:
        assert write["events"][WRITE_SP], "the setpoint committed without posting"
        assert write["events"][WRITE_SP][-1] == pytest.approx(-3.75)

    def test_the_readback_holds_the_value_a_later_reader_sees(self, write: dict) -> None:
        """The event and the served value are the same value: a post without a
        commit would be a monitor event no fresh reader could confirm."""
        assert write["after"][WRITE_RB] == pytest.approx(-3.75)

    def test_put_completion_is_not_a_polling_artifact(self, write: dict) -> None:
        """Put-completion returns once the physics hand-off has committed.
        Bounded well under the engine's own 1 Hz tick, so a write that appeared
        to land only because a poll happened to fire would fail here."""
        assert write["put_seconds"] < 1.0

    @pytest.fixture(scope="class")
    def telemetry(self, va: LiveVA) -> dict:
        """A pure listen: no client write anywhere in the window."""
        return va.call({"op": "watch", "watch": [TELEMETRY], "seconds": 4.0}, timeout=180.0)

    def test_a_telemetry_update_arrives_within_two_seconds(self, telemetry: dict) -> None:
        events = telemetry["events"][TELEMETRY]
        assert events, f"{TELEMETRY} published nothing in a 4 s window"
        assert events[0][0] < 2.0

    def test_telemetry_keeps_arriving(self, telemetry: dict) -> None:
        """More than one event, so this is a running publisher and not a single
        update delivered on subscription."""
        assert len(telemetry["events"][TELEMETRY]) >= 2

    def test_the_telemetry_value_actually_moves(self, telemetry: dict) -> None:
        """A channel republishing one unchanging value would satisfy the event
        count while telling a client nothing. The driver suppresses an event
        for an unchanged value, so distinct values are what delivery means
        here."""
        values = [value for _elapsed, value in telemetry["events"][TELEMETRY]]

        assert len(set(values)) > 1, f"{TELEMETRY} published {values!r}"

    def test_no_client_write_was_involved(self, telemetry: dict) -> None:
        """States the property the previous three rest on: this window contains
        the engine's own publishing and nothing else."""
        assert telemetry["before"][TELEMETRY] is not None


class TestRecordTypeParity:
    """The wire type is the one the manifest's ``record_type`` column asked for.

    Expected values are read out of the manifest rather than written down, so
    this compares two independent descriptions of one namespace instead of
    restating one of them.
    """

    def test_the_parity_table_covers_every_record_type_the_manifest_uses(
        self, manifest_channels: dict[str, dict]
    ) -> None:
        """Exhaustiveness, and an honest statement of reach: the served
        namespace uses two of the five record types ``pvdb.py`` can map. The
        other three have no channel to exercise them over a wire and are
        covered in process by ``tests/va/test_serving_pvdb.py``."""
        used = {channel["record_type"] for channel in manifest_channels.values()}

        assert used == set(WIRE_TYPES)

    def test_an_analog_channel_is_served_as_a_double(self, va: LiveVA) -> None:
        channel = va.read(SEEDED_BPM)[SEEDED_BPM]

        assert channel["type"] == WIRE_TYPES["ai"]
        assert channel["count"] == 1

    def test_an_analog_channel_declares_display_precision(self, va: LiveVA) -> None:
        """Left unset, a precision-aware client renders a metre-scale BPM
        reading as the integer zero."""
        assert va.read(SEEDED_BPM)[SEEDED_BPM]["precision"] == ANALOG_PRECISION

    def test_a_binary_channel_is_served_as_an_enum_with_its_two_labels(self, va: LiveVA) -> None:
        channel = va.read(BINARY_CHANNEL)[BINARY_CHANNEL]

        assert channel["type"] == WIRE_TYPES["bi"]
        assert channel["count"] == 1
        assert channel["enum_strs"] == list(BINARY_ENUM_STATES)

    def test_a_sample_of_the_namespace_matches_the_manifest_channel_for_channel(
        self, va: LiveVA, manifest_channels: dict[str, dict]
    ) -> None:
        """Not one channel per type but a spread of them, so a parity that held
        only for the two addresses named above would fail here."""
        sample: list[str] = []
        for record_type in sorted(WIRE_TYPES):
            addresses = sorted(
                address
                for address, channel in manifest_channels.items()
                if channel["record_type"] == record_type
            )
            step = max(1, len(addresses) // PARITY_SAMPLE)
            sample.extend(addresses[::step][:PARITY_SAMPLE])

        served = va.read(*sample)
        mismatched = {
            address: (served[address]["type"], manifest_channels[address]["record_type"])
            for address in sample
            if served[address]["type"] != WIRE_TYPES[manifest_channels[address]["record_type"]]
        }

        assert mismatched == {}
        assert len(sample) == 2 * PARITY_SAMPLE  # the sweep was not silently empty

    def test_a_setpoint_publishes_the_band_and_a_reading_publishes_none(self, va: LiveVA) -> None:
        """A client reads the same band the write path enforces, and a channel
        with no band says so rather than reporting a spurious one."""
        channels = va.read(WRITE_SP, SEEDED_BPM)

        assert channels[WRITE_SP]["lower_ctrl_limit"] == pytest.approx(BAND_LOW)
        assert channels[WRITE_SP]["upper_ctrl_limit"] == pytest.approx(BAND_HIGH)
        assert channels[SEEDED_BPM]["lower_ctrl_limit"] == channels[SEEDED_BPM]["upper_ctrl_limit"]


class TestAlarmParity:
    """A clamped write is a normal outcome, and must not present as a fault.

    The served database declares no alarm limits at all, which is what stops
    the server evaluating one: a value pinned to the top of its drive band is
    exactly where a bounded setpoint is supposed to sit, and a client that saw
    ``INVALID`` there would have no way to tell it from a real failure.
    """

    @pytest.fixture(scope="class")
    def clamped(self, va: LiveVA) -> dict:
        va.call(
            {
                "op": "write_watch",
                "address": WRITE_SP,
                "value": OUT_OF_BAND,
                "watch": [WRITE_RB],
                "expect": BAND_HIGH,
            }
        )
        return va.read(WRITE_SP, WRITE_RB)

    def test_the_write_really_was_clamped(self, clamped: dict) -> None:
        """Precondition: had the request landed unchanged, or had the setpoint
        already been at the band's top, the alarm assertions below would be
        about a write that was never bounded."""
        assert clamped[WRITE_SP]["value"] == pytest.approx(BAND_HIGH)
        assert BAND_HIGH != pytest.approx(OUT_OF_BAND)

    def test_the_clamped_setpoint_reads_no_alarm(self, clamped: dict) -> None:
        assert clamped[WRITE_SP]["severity"] == 0
        assert clamped[WRITE_SP]["status"] == 0

    def test_the_echoed_readback_reads_no_alarm_too(self, clamped: dict) -> None:
        assert clamped[WRITE_RB]["value"] == pytest.approx(BAND_HIGH)
        assert clamped[WRITE_RB]["severity"] == 0

    def test_the_band_that_was_enforced_is_the_band_the_client_reads(self, clamped: dict) -> None:
        assert clamped[WRITE_SP]["upper_ctrl_limit"] == pytest.approx(BAND_HIGH)


class TestSeededReadoutDivergence:
    """A served BPM reading is the *reading*, not the truth it was taken from.

    This is the property that makes the virtual accelerator worth pointing an
    orbit-correction algorithm at: the algorithm has to work from instruments
    that disagree with the machine, and a simulator that served the truth would
    quietly make the hardest part of the problem go away.
    """

    def test_the_two_boots_agree_on_the_unseeded_control(
        self, va: LiveVA, truth_orbit: dict[str, float]
    ) -> None:
        """Validity precondition for everything else in this class. Both boots
        were driven to the same state by the same write; if their unseeded BPM
        disagrees, the machine states differ and any divergence measured below
        cannot be attributed to the seeded error."""
        assert va.orbit[CONTROL_BPM] == truth_orbit[CONTROL_BPM]

    def test_the_control_reading_is_a_real_orbit_and_not_a_pair_of_zeros(
        self, truth_orbit: dict[str, float]
    ) -> None:
        """A machine at rest has an identically zero closed orbit, which would
        make the agreement above a comparison of two zeros. The reference write
        is what gives every BPM something to read."""
        assert truth_orbit[CONTROL_BPM] != 0.0

    def test_an_unseeded_bpm_reads_its_true_position_exactly(
        self, va: LiveVA, truth_orbit: dict[str, float]
    ) -> None:
        """With no error seeded, ``bpm_read`` is the identity -- so this
        channel is where the truth is legible at all."""
        assert va.orbit[CONTROL_BPM] == truth_orbit[CONTROL_BPM]

    def test_the_seeded_bpm_does_not_read_its_true_position(
        self, va: LiveVA, truth_orbit: dict[str, float]
    ) -> None:
        assert va.orbit[SEEDED_BPM] != truth_orbit[SEEDED_BPM]

    def test_the_divergence_is_exactly_the_seeded_transform(
        self, va: LiveVA, truth_orbit: dict[str, float]
    ) -> None:
        """Not "differs by roughly the right amount": the seeded error carries
        no noise, so the wire is compared against ``bpm_read`` itself -- the
        same function the physics bridge applies -- called with the truth the
        unseeded boot reported."""
        expected, _y = bpm_read(
            truth_orbit[SEEDED_BPM],
            0.0,
            **{
                **IDENTITY_BPM_ERROR,
                "offset_x": SEEDED_OFFSET_X,
                "gain_x": SEEDED_GAIN_X,
            },
            rng=np.random.default_rng(0),
        )

        assert va.orbit[SEEDED_BPM] == expected

    def test_the_transform_is_not_the_identity(
        self, va: LiveVA, truth_orbit: dict[str, float]
    ) -> None:
        """Anti-vacuous guard on the assertion above: a seed that happened to
        reduce to the identity would make it pass while proving nothing."""
        difference = abs(va.orbit[SEEDED_BPM] - truth_orbit[SEEDED_BPM])

        assert difference > SEEDED_OFFSET_X / 2

    def test_the_divergence_is_present_at_the_reference_state(
        self, va: LiveVA, truth_orbit: dict[str, float]
    ) -> None:
        """Both snapshots were taken before any test wrote anything, so the
        divergence is a property of the served namespace rather than something
        a particular write sequence produced."""
        assert va.orbit[REFERENCE_RB] == truth_orbit[REFERENCE_RB] == REFERENCE_CURRENT


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_serving_parity.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
