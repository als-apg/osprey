"""What a client sees when it writes a co-hosted setpoint over real Channel Access.

Everything here runs in **one process**: a real ``pcaspy`` server hosting the
real serving database, the real write path deciding what each write means, and
a real ``pyepics`` client driving it from the pytest thread. No subprocess
split, no fake driver -- the assertions are made from the far side of the wire,
which is the only place the properties this file exists to pin are observable
at all. ``tests/va/test_serving_pvdb.py`` and ``tests/va/test_serving_runner.py``
already prove the same contracts in process against a duck-typed driver; this
file is the check that the wire agrees with them.

Two things about the venue are worth stating plainly, because a green run means
nothing without them.

**pcaspy is the venue's hard requirement.** It publishes no macOS arm64 wheel
that this host can load, so on a developer machine the live classes *skip* --
loudly, via ``importorskip``, and a skipped live suite proves nothing. The
venue that proves them is a linux container (and CI's linux lanes); everything
in this file that does not need a server -- the drive-limit derivation, the
collection floor -- runs everywhere and is not gated.

**The client and the server share a process.** ``pcaspy``'s compiled extension
links EPICS statically and globally exports the ``ca_*`` client symbols, so
whichever EPICS stack binds first wins. ``epics.ca.initialize_libca()`` is
therefore called *before* pcaspy is imported, so the client's calls bind to
pyepics' own libca and not to the copy buried in the server extension. That
ordering is the whole reason these can be plain in-process tests.

The run loop is stood in for by :class:`_RunLoop`, which reproduces the one
property of the real loop the write path depends on: the model is touched from
one thread that is never the server's, and the write is completed from there.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from queue import Queue
from typing import Any

import pytest


def _free_port() -> str:
    """An unused loopback TCP port, as a string ready for the environment."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return str(probe.getsockname()[1])


# Set at import time because libca latches the EPICS_CA_* environment when the
# C library initialises, which happens on the first `import epics` anywhere in
# the process -- earlier than any fixture can run.
#
# Loopback only, and on an ephemeral port unless the environment already pins
# one: this worktree is shared by concurrent test runs and by the operator's
# own containers, so a hardcoded port would race with them. The server port and
# the CAS port must be the same number -- a search reply carries the server's
# own port, so a server listening anywhere else hands clients an address
# nothing answers on.
os.environ.setdefault("EPICS_CA_ADDR_LIST", "127.0.0.1")
os.environ.setdefault("EPICS_CA_AUTO_ADDR_LIST", "NO")
os.environ.setdefault("EPICS_CA_SERVER_PORT", _free_port())
os.environ.setdefault("EPICS_CAS_SERVER_PORT", os.environ["EPICS_CA_SERVER_PORT"])
os.environ.setdefault("EPICS_CA_REPEATER_PORT", _free_port())

from osprey.services.virtual_accelerator.entrypoint import (  # noqa: E402
    _channel_limits_path,
    _load_drive_limits,
)
from osprey.services.virtual_accelerator.manifest import (  # noqa: E402
    PARTITION_PYAT_COUPLED,
    PARTITION_SP_ECHO,
    PARTITION_STATIC_NOISY,
    RECORD_TYPE_ANALOG,
)
from osprey.services.virtual_accelerator.serving.pvdb import build_serving_pvdb  # noqa: E402
from osprey.services.virtual_accelerator.serving.write_path import (  # noqa: E402
    CohostWritePath,
    physics_setpoint_addresses,
)

# Floor for this module's own test count -- a guard against a refactor that
# leaves the file importable but empty, which would otherwise pass silently.
MIN_COLLECTED_TESTS = 24

# How long a client waits for a value to arrive over the wire. Generous: an
# emulated container is the venue, and a timeout here has to mean "never", not
# "not yet".
CA_TIMEOUT_S = 10.0
SETTLE_TIMEOUT_S = 10.0

# The whole served namespace of this module, kept disjoint from every other
# test module's so two servers alive in one pytest session can never answer for
# each other's names.
RING = "ZZRF"

# A pyat-coupled magnet with a drive band: the clamp subject.
BAND_SP = f"{RING}:MAG:HCM:01:CURRENT:SP"
BAND_RB = f"{RING}:MAG:HCM:01:CURRENT:RB"
BAND_LOW, BAND_HIGH = -12.0, 12.0

# A second pyat-coupled magnet, seeded to a nonzero boot value: the boot-value
# subject, and the bystander every isolation assertion is made against.
SEEDED_SP = f"{RING}:MAG:HCM:02:CURRENT:SP"
SEEDED_RB = f"{RING}:MAG:HCM:02:CURRENT:RB"
SEEDED_BOOT = 3.25

# A third, frozen by a build-time apply fault.
STUCK_SP = f"{RING}:MAG:HCM:03:CURRENT:SP"
STUCK_RB = f"{RING}:MAG:HCM:03:CURRENT:RB"

# The BPM reading the physics hook pushes after an accepted write. Its value is
# deliberately unrelated to the current that was written, so a reading that
# appears here can only have come from the hook's own push.
BPM = f"{RING}:DIAG:BPM:01:POSITION:X"
BPM_READING = -0.000123

# A plain setpoint/readback echo pair: no physics behind it, and a band of its
# own so the clamp can be shown to apply to a write that never reaches a model.
ECHO_SP = f"{RING}:RF:CAV:01:VOLTAGE:SP"
ECHO_RB = f"{RING}:RF:CAV:01:VOLTAGE:RB"
ECHO_LOW, ECHO_HIGH = 0.0, 5.0

# Telemetry: driven by the engine source in production, never writable.
TELEMETRY = f"{RING}:DIAG:TEMP:01:VALUE:RB"

# The value the physics hook refuses. A refusal has to be a property of the
# value rather than of the address, so the same setpoint can be shown to accept
# a later write.
REFUSED_VALUE = 9.75
REFUSAL_REASON = "lattice rejected the commanded current"

DRIVE_LIMITS = {BAND_SP: (BAND_LOW, BAND_HIGH), ECHO_SP: (ECHO_LOW, ECHO_HIGH)}
BOOT_VALUES = {SEEDED_SP: SEEDED_BOOT, SEEDED_RB: SEEDED_BOOT}

# Every served address, with the value it must hold before any client writes.
# Written out rather than derived so a change to the served set has to be made
# deliberately here as well.
BOOT_STATE = {
    BAND_SP: 0.0,
    BAND_RB: 0.0,
    SEEDED_SP: SEEDED_BOOT,
    SEEDED_RB: SEEDED_BOOT,
    STUCK_SP: 0.0,
    STUCK_RB: 0.0,
    BPM: 0.0,
    ECHO_SP: 0.0,
    ECHO_RB: 0.0,
    TELEMETRY: 0.0,
}


def _channel(
    address: str,
    *,
    partition: str,
    subfield: str,
    family: str,
    device: str,
    field: str,
    system: str = "MAG",
    record_type: str = RECORD_TYPE_ANALOG,
    noise: bool = False,
) -> dict:
    """One synthetic manifest entry, in the shape ``build_serving_pvdb`` reads."""
    return {
        "address": address,
        "ring": RING,
        "system": system,
        "family": family,
        "device": device,
        "field": field,
        "subfield": subfield,
        "partition": partition,
        "record_type": record_type,
        "noise": noise,
    }


def _magnet(device: str, partition: str = PARTITION_PYAT_COUPLED) -> list[dict]:
    """A magnet's setpoint and its own current readback."""
    return [
        _channel(
            f"{RING}:MAG:HCM:{device}:CURRENT:SP",
            partition=partition,
            subfield="SP",
            family="HCM",
            device=device,
            field="CURRENT",
        ),
        _channel(
            f"{RING}:MAG:HCM:{device}:CURRENT:RB",
            partition=partition,
            subfield="RB",
            family="HCM",
            device=device,
            field="CURRENT",
        ),
    ]


SERVED_CHANNELS = [
    *_magnet("01"),
    *_magnet("02"),
    *_magnet("03"),
    _channel(
        BPM,
        partition=PARTITION_PYAT_COUPLED,
        subfield="X",
        system="DIAG",
        family="BPM",
        device="01",
        field="POSITION",
    ),
    _channel(
        ECHO_SP,
        partition=PARTITION_SP_ECHO,
        subfield="SP",
        system="RF",
        family="CAV",
        device="01",
        field="VOLTAGE",
    ),
    _channel(
        ECHO_RB,
        partition=PARTITION_SP_ECHO,
        subfield="RB",
        system="RF",
        family="CAV",
        device="01",
        field="VOLTAGE",
    ),
    _channel(
        TELEMETRY,
        partition=PARTITION_STATIC_NOISY,
        subfield="RB",
        system="DIAG",
        family="TEMP",
        device="01",
        field="VALUE",
    ),
]


class _RunLoop:
    """The one property of the real run loop the write path depends on.

    A queued write is applied and completed from a thread that is never the
    Channel Access server's, because the model underneath is not thread-safe.
    Everything else the real loop does -- batching, the output pass, the
    rollback -- is either disabled by the runner's configuration policy or
    irrelevant to what a client observes, and is proven in process elsewhere.
    """

    def __init__(self, on_setpoint) -> None:  # noqa: ANN001 - test-local callable
        self._on_setpoint = on_setpoint
        self._queue: Queue = Queue()
        self.thread = threading.Thread(target=self._run, daemon=True, name="va-test-run-loop")
        self.thread.start()

    def enqueue(self, values: dict, *, done) -> None:  # noqa: ANN001 - test-local callable
        self._queue.put((values, done))

    def _run(self) -> None:
        while True:
            values, done = self._queue.get()
            error = None
            try:
                for address, item in values.items():
                    self._on_setpoint(address, item["value"])
            except Exception as exc:  # noqa: BLE001 - the loop reports, never raises
                error = str(exc)
            done(error)


class _PhysicsHook:
    """Stands in for the physics bridge: applies a write, then pushes a reading.

    Faithful on the two points a client can see. It runs on the loop's thread,
    and the reading it pushes onto the BPM is its own -- never a value read back
    out of whatever took the write -- so a reading on the wire can only have
    come from here.
    """

    def __init__(self, records) -> None:  # noqa: ANN001 - ServingRecords
        self._records = records
        self.calls: list[tuple[str, float]] = []
        self.threads: set[str] = set()

    def __call__(self, address: str, value: float) -> None:
        self.threads.add(threading.current_thread().name)
        if value == REFUSED_VALUE:
            # The model refuses by raising; the loop turns that into the error
            # string the write path completes on.
            raise ValueError(REFUSAL_REASON)
        self.calls.append((address, value))
        self._records.all[BPM].set(BPM_READING)


class LiveNamespace:
    """A served namespace, its write path, and the loop behind it."""

    def __init__(self, records, path, hook, loop, driver, refusal_alarm) -> None:  # noqa: ANN001
        self.records = records
        self.path = path
        self.hook = hook
        self.loop = loop
        self.driver = driver
        # The (alarm, severity) the write path was built with, so the alarm
        # assertions compare against the condition this server actually
        # declares rather than against a number copied out of EPICS.
        self.refusal_alarm = refusal_alarm


def _build_namespace(pcaspy: Any) -> LiveNamespace:
    """Assemble the serving layer exactly as the entrypoint assembles it.

    Database first, with every boot value already in it, because the Channel
    Access server copies each spec when it creates the PV; the driver, and so
    the records' attachment to it, only after the server exists.
    """
    records = build_serving_pvdb(
        SERVED_CHANNELS,
        drive_limits=DRIVE_LIMITS,
        boot_values=BOOT_VALUES,
        async_setpoints=True,
    )
    hook = _PhysicsHook(records)
    loop = _RunLoop(hook)
    refusal_alarm = (pcaspy.Alarm.WRITE_ALARM, pcaspy.Severity.INVALID_ALARM)
    path = CohostWritePath(
        records,
        enqueue=loop.enqueue,
        physics_setpoints=physics_setpoint_addresses(records),
        stuck_setpoints=frozenset({STUCK_SP}),
        drive_limits=DRIVE_LIMITS,
        refusal_alarm=refusal_alarm,
    )

    class LiveDriver(pcaspy.Driver):
        """The production driver's whole body: delegate to the write path."""

        def write(self, reason: str, value: Any) -> bool:  # noqa: D102 - pcaspy contract
            accepted: bool = path.write(self, reason, value)
            return accepted

    server = pcaspy.SimpleServer()
    server.createPV("", records.pvdb)
    driver = LiveDriver()
    records.attach_driver(driver)

    stop = threading.Event()

    def serve() -> None:
        while not stop.is_set():
            server.process(0.05)

    thread = threading.Thread(target=serve, daemon=True, name="va-test-cas")
    thread.start()

    namespace = LiveNamespace(records, path, hook, loop, driver, refusal_alarm)
    # Kept alive for the fixture's lifetime: a server or its serving thread
    # collected out from under the client would fail every test below for a
    # reason that has nothing to do with the write path.
    namespace._server = server  # noqa: SLF001 - this object is the test's own
    namespace._stop = stop  # noqa: SLF001
    namespace._thread = thread  # noqa: SLF001
    return namespace


@pytest.fixture(scope="module")
def live() -> Any:
    """A live Channel Access server over the real serving layer.

    ``initialize_libca()`` runs before pcaspy is imported, so the client's
    ``ca_*`` calls bind to pyepics' libca rather than to the second, unset-up
    copy the pcaspy extension exports. Import order is the mitigation; see the
    module docstring.
    """
    import epics

    epics.ca.initialize_libca()
    pcaspy = pytest.importorskip(
        "pcaspy",
        reason=(
            "the live Channel Access venue needs pcaspy, which has no loadable "
            "macOS arm64 wheel; run this suite in a linux container or on CI"
        ),
    )

    namespace = _build_namespace(pcaspy)
    if not _wait_until(lambda: _caget(TELEMETRY) is not None):
        pytest.fail("the Channel Access server never became reachable")
    yield namespace
    namespace._stop.set()  # noqa: SLF001
    namespace._thread.join(timeout=5)  # noqa: SLF001


def _wait_until(predicate, *, timeout: float = SETTLE_TIMEOUT_S) -> Any:  # noqa: ANN001
    """Poll ``predicate`` until it is truthy, then return what it returned."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.05)
    return predicate()


def _caget(address: str) -> Any:
    import epics

    return epics.caget(address, timeout=CA_TIMEOUT_S, connection_timeout=CA_TIMEOUT_S)


def _caput(address: str, value: float) -> Any:
    """Write with put-completion: the client stays blocked until the write ends."""
    import epics

    return epics.caput(address, value, wait=True, timeout=CA_TIMEOUT_S)


def _settle(address: str, expected: float) -> Any:
    """Wait for ``address`` to carry ``expected``, then return what it carries.

    Returns the last value read either way, so a failure reports the value that
    was actually served rather than a bare timeout.
    """
    last: list[Any] = [None]

    def reached() -> bool:
        last[0] = _caget(address)
        return last[0] is not None and abs(last[0] - expected) < 1e-9

    _wait_until(reached)
    return last[0]


def _snapshot(*addresses: str) -> dict[str, Any]:
    return {address: _caget(address) for address in addresses}


class TestBootValues:
    """The namespace comes up holding machine state, not zeros.

    Read over the wire on purpose: a boot value has to survive the server's own
    copy of each spec at PV-creation time, which is the step that makes the
    order in the entrypoint a contract rather than a convenience.
    """

    def test_every_served_address_boots_at_its_seeded_value(self, live: Any) -> None:
        assert _snapshot(*BOOT_STATE) == pytest.approx(BOOT_STATE)

    def test_a_seeded_setpoint_boots_at_the_scenario_value(self, live: Any) -> None:
        assert _caget(SEEDED_SP) == pytest.approx(SEEDED_BOOT)

    def test_its_readback_boots_there_too(self, live: Any) -> None:
        """A machine whose readbacks disagree with its setpoints at boot is a
        machine every client would immediately try to correct."""
        assert _caget(SEEDED_RB) == pytest.approx(SEEDED_BOOT)

    def test_an_unseeded_setpoint_boots_at_the_type_default(self, live: Any) -> None:
        assert _caget(BAND_SP) == pytest.approx(0.0)


class TestPhysicsRoundTrip:
    """The scan-hang regression (FR10), from the far side of the wire.

    A corrector's own ``:CURRENT:RB`` must follow its ``:CURRENT:SP``. The
    physics bridge never touches a magnet's own readback -- it pushes BPM
    readings and nothing else -- so if the serving layer does not echo the
    accepted value onto the paired readback, an ``EpicsMotor.set()`` settle-wait
    never returns and every orbit-response scan hangs at its first step.
    """

    def test_the_readback_follows_the_setpoint(self, live: Any) -> None:
        assert _caput(BAND_SP, 2.0) == 1
        assert _settle(BAND_RB, 2.0) == pytest.approx(2.0)

    def test_the_setpoint_carries_the_written_value_too(self, live: Any) -> None:
        assert _caput(BAND_SP, 1.5) == 1
        assert _settle(BAND_SP, 1.5) == pytest.approx(1.5)

    def test_a_settle_wait_on_the_readback_is_woken_by_a_monitor_event(self, live: Any) -> None:
        """What a settle-wait actually subscribes to.

        A one-shot read can be satisfied by a value the server recorded without
        posting. A settle-wait cannot: it blocks on a monitor event, so a write
        path that committed the echo and never posted it would hang the scan
        exactly as a missing echo would.
        """
        import epics

        seen: list[float] = []
        readback = epics.PV(BAND_RB, auto_monitor=True)
        try:
            assert readback.wait_for_connection(timeout=CA_TIMEOUT_S)
            readback.add_callback(lambda value=None, **_: seen.append(value))

            assert _caput(BAND_SP, -4.25) == 1
            _wait_until(lambda: any(abs(value + 4.25) < 1e-9 for value in seen))
        finally:
            # Explicit, in a finally: a PV finalised by the garbage collector
            # tears libca down from the wrong thread.
            readback.disconnect()

        assert seen, "no monitor event ever reached the subscriber"
        assert seen[-1] == pytest.approx(-4.25)

    def test_the_hook_is_handed_the_written_value(self, live: Any) -> None:
        live.hook.calls.clear()
        assert _caput(BAND_SP, 3.0) == 1
        _wait_until(lambda: live.hook.calls)

        assert live.hook.calls == [(BAND_SP, pytest.approx(3.0))]

    def test_the_reading_on_the_wire_is_the_one_the_hook_pushed(self, live: Any) -> None:
        """Never a value read back out of the model: a BPM reading carries its
        own seeded readout error, and republishing the model's own view would
        overwrite the reading with the truth it exists to differ from."""
        assert _caput(BAND_SP, 0.5) == 1

        assert _settle(BPM, BPM_READING) == pytest.approx(BPM_READING)

    def test_the_model_is_never_touched_from_the_server_thread(self, live: Any) -> None:
        live.hook.threads.clear()
        assert _caput(BAND_SP, 0.75) == 1
        _wait_until(lambda: live.hook.threads)

        assert live.hook.threads == {live.loop.thread.name}

    def test_put_completion_does_not_return_before_the_value_is_committed(self, live: Any) -> None:
        """A client that unblocks and reads immediately must see its write.

        No sleep and no settle poll between the two calls: ``callbackPV`` fires
        only after the write path has committed the setpoint and its echo, so
        the read below is answered from a store that already holds them.
        """
        assert _caput(BAND_SP, -6.5) == 1

        assert _caget(BAND_SP) == pytest.approx(-6.5)
        assert _caget(BAND_RB) == pytest.approx(-6.5)


class TestDriveLimitClamp:
    """DRVL/DRVH, enforced where it is the only enforcement there is.

    Neither server rejects an out-of-band write despite publishing the band as
    display limits, and the model does not enforce it either, so the write path
    is the whole of it. What a client observes is the clamped value on the
    setpoint *and* on its readback -- a real bound below the scan plan's own
    schema, applied to any writer.
    """

    def test_a_write_above_the_band_lands_at_the_top_of_it(self, live: Any) -> None:
        assert _caput(BAND_SP, 50.0) == 1

        assert _settle(BAND_SP, BAND_HIGH) == pytest.approx(BAND_HIGH)
        assert _settle(BAND_RB, BAND_HIGH) == pytest.approx(BAND_HIGH)

    def test_a_write_below_the_band_lands_at_the_bottom_of_it(self, live: Any) -> None:
        assert _caput(BAND_SP, -50.0) == 1

        assert _settle(BAND_SP, BAND_LOW) == pytest.approx(BAND_LOW)
        assert _settle(BAND_RB, BAND_LOW) == pytest.approx(BAND_LOW)

    def test_the_model_is_handed_the_clamped_value_not_the_requested_one(self, live: Any) -> None:
        """Otherwise the lattice and the wire would disagree about what the
        machine was set to."""
        live.hook.calls.clear()
        assert _caput(BAND_SP, 99.0) == 1
        _wait_until(lambda: live.hook.calls)

        assert live.hook.calls == [(BAND_SP, pytest.approx(BAND_HIGH))]

    def test_an_in_band_write_is_served_untouched(self, live: Any) -> None:
        assert _caput(BAND_SP, -11.875) == 1

        assert _settle(BAND_SP, -11.875) == pytest.approx(-11.875)

    def test_the_band_is_published_as_the_control_limits(self, live: Any) -> None:
        """A client reads the same band the write path enforces, so it can
        refuse the write itself rather than discover the clamp afterwards."""
        import epics

        setpoint = epics.PV(BAND_SP)
        try:
            assert setpoint.wait_for_connection(timeout=CA_TIMEOUT_S)
            control = setpoint.get_ctrlvars()
        finally:
            setpoint.disconnect()

        assert control["lower_ctrl_limit"] == pytest.approx(BAND_LOW)
        assert control["upper_ctrl_limit"] == pytest.approx(BAND_HIGH)

    def test_an_echo_setpoint_with_no_model_behind_it_is_clamped_too(self, live: Any) -> None:
        """The clamp is a property of the write path, not of the physics
        hand-off, so a write that never reaches a model is bounded as well."""
        assert _caput(ECHO_SP, 40.0) == 1

        assert _settle(ECHO_SP, ECHO_HIGH) == pytest.approx(ECHO_HIGH)
        assert _settle(ECHO_RB, ECHO_HIGH) == pytest.approx(ECHO_HIGH)


class TestEchoWithoutPhysics:
    """A setpoint with no model behind it: the readback is a plain value copy."""

    def test_the_readback_follows_immediately(self, live: Any) -> None:
        assert _caput(ECHO_SP, 2.5) == 1

        assert _settle(ECHO_RB, 2.5) == pytest.approx(2.5)

    def test_no_write_reaches_the_hook(self, live: Any) -> None:
        live.hook.calls.clear()
        assert _caput(ECHO_SP, 1.0) == 1
        _settle(ECHO_RB, 1.0)

        assert live.hook.calls == []


class TestStuckSetpointFreeze:
    """An apply fault, honestly, for every reader.

    The setpoint still records what was written to it -- a client that reads
    back what it wrote is told the truth about the command it issued -- while
    the readback never moves again. Nothing about that is one client's view: it
    is the served value, so every reader sees the same frozen device.
    """

    def test_the_setpoint_latches_the_written_value(self, live: Any) -> None:
        assert _caput(STUCK_SP, 7.5) == 1

        assert _settle(STUCK_SP, 7.5) == pytest.approx(7.5)

    def test_the_readback_never_moves(self, live: Any) -> None:
        before = _caget(STUCK_RB)
        assert _caput(STUCK_SP, 4.0) == 1
        _settle(STUCK_SP, 4.0)

        assert _caget(STUCK_RB) == pytest.approx(before)

    def test_the_machine_never_moves_either(self, live: Any) -> None:
        """The fault suppresses the physics hand-off, not just the echo:
        a frozen magnet's current must not reach the lattice."""
        live.hook.calls.clear()
        assert _caput(STUCK_SP, 5.5) == 1
        _settle(STUCK_SP, 5.5)

        assert live.hook.calls == []

    def test_the_write_still_completes(self, live: Any) -> None:
        """A setpoint whose asynchronous write is never completed has every
        later write to it postponed by the server library, which would turn a
        frozen readback into a frozen setpoint as well."""
        assert _caput(STUCK_SP, 1.0) == 1
        assert _caput(STUCK_SP, 2.0) == 1

        assert _settle(STUCK_SP, 2.0) == pytest.approx(2.0)

    def test_an_unfaulted_sibling_still_echoes(self, live: Any) -> None:
        """The fault is per channel, not a partition-wide switch."""
        assert _caput(SEEDED_SP, 6.25) == 1

        assert _settle(SEEDED_RB, 6.25) == pytest.approx(6.25)


class TestRefusedWrite:
    """A refusal moves nothing, and says so the only way Channel Access can.

    Put-completion has no failure status, so a refusal is expressed by
    withholding the echo. Withholding has to mean withholding the *commit* as
    well: a value recorded in the parameter store and merely not posted would
    still be served to the next one-shot reader.
    """

    def test_a_one_shot_reader_sees_no_movement(self, live: Any) -> None:
        assert _caput(BAND_SP, 1.25) == 1
        _settle(BAND_SP, 1.25)

        assert _caput(BAND_SP, REFUSED_VALUE) == 1
        time.sleep(0.5)

        assert _caget(BAND_SP) == pytest.approx(1.25)
        assert _caget(BAND_RB) == pytest.approx(1.25)

    def test_a_monitoring_reader_sees_no_movement_either(self, live: Any) -> None:
        import epics

        assert _caput(BAND_SP, 2.75) == 1
        _settle(BAND_RB, 2.75)

        seen: list[float] = []
        readback = epics.PV(BAND_RB, auto_monitor=True)
        try:
            assert readback.wait_for_connection(timeout=CA_TIMEOUT_S)
            assert readback.get(use_monitor=False) == pytest.approx(2.75)
            readback.add_callback(lambda value=None, **_: seen.append(value))

            assert _caput(BAND_SP, REFUSED_VALUE) == 1
            time.sleep(0.5)
        finally:
            readback.disconnect()

        assert [value for value in seen if abs(value - 2.75) > 1e-9] == []

    def test_the_refusal_raises_an_alarm(self, live: Any) -> None:
        """The only trace a refusal can leave for a Channel Access client."""
        import epics

        setpoint = epics.PV(BAND_SP)
        try:
            assert setpoint.wait_for_connection(timeout=CA_TIMEOUT_S)
            assert _caput(BAND_SP, REFUSED_VALUE) == 1
            _wait_until(lambda: setpoint.get(use_monitor=False) is not None and setpoint.severity)
            severity = setpoint.severity
        finally:
            setpoint.disconnect()

        assert severity == live.refusal_alarm[1]

    def test_a_later_write_still_lands(self, live: Any) -> None:
        """Completion is signalled on a refusal too, so the setpoint is not
        left with an asynchronous write in flight forever."""
        assert _caput(BAND_SP, REFUSED_VALUE) == 1
        assert _caput(BAND_SP, 8.5) == 1

        assert _settle(BAND_SP, 8.5) == pytest.approx(8.5)
        assert _settle(BAND_RB, 8.5) == pytest.approx(8.5)

    def test_the_alarm_clears_on_the_next_accepted_write(self, live: Any) -> None:
        """Self-clearing, because the served database declares no alarm limits:
        the next accepted write's ``setParam`` recomputes the condition from the
        value, and with nothing to compare against it always lands on
        NO_ALARM."""
        import epics

        setpoint = epics.PV(BAND_SP)
        try:
            assert setpoint.wait_for_connection(timeout=CA_TIMEOUT_S)
            assert _caput(BAND_SP, REFUSED_VALUE) == 1
            _wait_until(lambda: setpoint.get(use_monitor=False) is not None and setpoint.severity)
            assert _caput(BAND_SP, 0.25) == 1
            _settle(BAND_SP, 0.25)
            _wait_until(lambda: setpoint.get(use_monitor=False) is not None)
            severity = setpoint.severity
        finally:
            setpoint.disconnect()

        assert severity == 0  # NO_ALARM


class TestPerWriteIsolation:
    """One write moves one device.

    The database is one dict and the driver one parameter store, so a write
    path that swept the whole namespace -- or that shared a spec between two
    channels -- would show up here as a bystander moving. Asserted over the
    wire, because that is where "moved" is defined.
    """

    def test_writing_one_magnet_leaves_every_other_address_alone(self, live: Any) -> None:
        bystanders = (SEEDED_SP, SEEDED_RB, STUCK_SP, STUCK_RB, ECHO_SP, ECHO_RB, TELEMETRY)
        before = _snapshot(*bystanders)

        assert _caput(BAND_SP, -2.5) == 1
        _settle(BAND_RB, -2.5)

        assert _snapshot(*bystanders) == pytest.approx(before)

    def test_writing_one_magnet_leaves_the_others_readback_alone(self, live: Any) -> None:
        assert _caput(SEEDED_SP, 1.0) == 1
        _settle(SEEDED_RB, 1.0)

        assert _caput(BAND_SP, 9.0) == 1
        _settle(BAND_RB, 9.0)

        assert _caget(SEEDED_RB) == pytest.approx(1.0)


class TestNonWritableChannels:
    """Refusing a write is what keeps a served value the value its source published."""

    def test_a_telemetry_channel_refuses_the_write(self, live: Any) -> None:
        before = _caget(TELEMETRY)
        _caput(TELEMETRY, 42.0)
        time.sleep(0.5)

        assert _caget(TELEMETRY) == pytest.approx(before)

    def test_a_readback_of_a_setpoint_pair_refuses_the_write(self, live: Any) -> None:
        assert _caput(ECHO_SP, 3.5) == 1
        _settle(ECHO_RB, 3.5)

        _caput(ECHO_RB, 42.0)
        time.sleep(0.5)

        assert _caget(ECHO_RB) == pytest.approx(3.5)


class TestDerivedDriveLimitsMatchChannelLimitsFile:
    """The serving layer is file-blind: ``_load_drive_limits()`` is the only
    place that turns ``channel_limits.json`` into the map the clamp above is
    built from. This pins that derivation against the file's own contents --
    read the same path the entrypoint reads, independently re-derive the
    expected map from the raw JSON, and require full-dict equality -- so a
    future channel_limits retune that silently changes which ``:SP`` addresses
    are writable-with-bounds fails here instead of just drifting the deployed
    clamp unnoticed.

    Needs no server: this is the derivation, not its enforcement.
    """

    def test_derived_map_matches_writable_sp_entries_in_the_file(self) -> None:
        raw = json.loads(_channel_limits_path().read_text())
        defaults = raw.get("defaults", {})
        expected: dict[str, tuple[float, float]] = {}
        for address, entry in raw.items():
            if address.startswith("_") or address == "defaults" or not address.endswith(":SP"):
                continue
            merged = {**defaults, **entry}
            if not merged.get("writable", True):
                continue
            min_value = merged.get("min_value")
            max_value = merged.get("max_value")
            if min_value is None or max_value is None:
                continue
            expected[address] = (float(min_value), float(max_value))

        derived = _load_drive_limits()
        assert derived == expected
        # Sanity count: pins today's writable-:SP-with-bounds population so a
        # channel_limits.json edit that silently drops or adds entries is caught
        # here, not only downstream in clamp behavior.
        assert len(expected) == 396


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_record_factory.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
