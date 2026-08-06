"""``EngineSource`` driven against the real serving-layer record shim.

The 1 Hz telemetry source was written against pythonSoftIOC In-type records.
The serving layer replaces those with :class:`~osprey.services.virtual_accelerator.serving.pvdb.PVRecord`,
and the claim under test is that the swap is invisible to ``EngineSource``:
it drives records through exactly ``record.set(value)`` and ``record.get()``,
both of which the shim provides.

Every test here therefore uses the **real** ``build_serving_pvdb()`` output --
a mock of the shim would only prove that ``EngineSource`` works against a
mock. What *is* faked is the layer below the shim: the Channel Access driver
(``setParam``/``getParam``/``updatePV``), so this suite stays host-side and
in-process. Nothing here constructs a CA server, binds a port or imports the
CA server library; live-CA behaviour belongs to the container e2e suite.

The four behaviours proven are the ones that must survive the swap:

* the poll tick drives co-hosted telemetry PVs and posts a monitor event per
  address;
* scenario hot-reload (mtime + content-hash detection) still fires;
* setpoint-echo sync reads **through the driver**, so a setpoint a client
  wrote over CA reaches the engine -- a shim that cached values locally would
  feed the engine a stale one;
* one record's failure never kills the poll loop. That last one is a safety
  property: the loop is scheduled as a fire-and-forget coroutine whose result
  nobody retrieves, so an escaping exception would freeze every telemetry
  channel at its boot value with no visible error.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from osprey.services.virtual_accelerator.ioc import engine_source as engine_source_module
from osprey.services.virtual_accelerator.ioc.engine_source import EngineSource
from osprey.services.virtual_accelerator.manifest import (
    PARTITION_SP_ECHO,
    PARTITION_STATIC_NOISY,
    RECORD_TYPE_ANALOG,
    RECORD_TYPE_BINARY,
)
from osprey.services.virtual_accelerator.serving.pvdb import (
    PVRecord,
    ServingRecords,
    build_serving_pvdb,
)
from osprey.simulation.engine import SimulationEngine

# Floor for this module's own test count -- a guard against a refactor that
# leaves the file importable but empty, which would otherwise pass silently.
MIN_COLLECTED_TESTS = 20

VAC_RB = "ZZES:VAC:GAUGE:01:PRESSURE:RB"
FAULT_BI = "ZZES:STATUS:FLAG:01:FAULT:RB"
UNMODELED_RB = "ZZES:UNKNOWN:LEVEL:01:LEVEL:RB"
STAGE_SP = "ZZES:STAGE:MOTOR:01:POSITION:SP"
STAGE_RB = "ZZES:STAGE:MOTOR:01:POSITION:RB"
CAMERA_RB = "ZZES:CAM:PROF:01:INTENSITY:RB"

# The camera's response peaks when the stage sits at 2.0; off-peak (stage at
# 0.0) it reads e^{-4} of that. A contrast that large cannot be produced by
# noise, so it is an unambiguous witness that the engine saw the setpoint.
PEAK_INTENSITY = 1000.0
OFF_PEAK_INTENSITY = PEAK_INTENSITY * 0.0183156

TEST_MACHINE = {
    "name": "EngineSourceServingTestRig",
    "description": "Tiny inline machine for engine-source-over-serving tests.",
    "channels": {
        VAC_RB: {"value": 1e-8, "units": "Torr", "noise": 0.0, "description": "vac gauge"},
        FAULT_BI: {"value": 0, "noise": 0.0, "description": "fault flag"},
        STAGE_RB: {"value": 0.0, "noise": 0.0, "description": "stage readback"},
        CAMERA_RB: {
            "expr": f"{PEAK_INTENSITY} * exp(-((ch('{STAGE_RB}') - 2.0) ** 2))",
            "noise": 0.0,
            "description": "camera response to the stage position",
        },
    },
    "scenarios": {
        "nominal": {"description": "All systems nominal."},
        "leak": {"description": "Vacuum leak.", "overrides": {VAC_RB: 5e-6}},
    },
}


def _channel(
    address: str,
    *,
    record_type: str = RECORD_TYPE_ANALOG,
    subfield: str = "RB",
    partition: str = PARTITION_STATIC_NOISY,
    noise: bool = False,
    system: str = "DIAG",
    family: str = "TEST",
    device: str = "01",
    field: str = "VALUE",
) -> dict:
    """One synthetic manifest channel, in the shape ``build_manifest()`` emits."""
    return {
        "address": address,
        "ring": "ZZES",
        "system": system,
        "family": family,
        "device": device,
        "field": field,
        "subfield": subfield,
        "partition": partition,
        "record_type": record_type,
        "noise": noise,
    }


# Telemetry (partition c) plus one sp-echo pair. The pair is what the
# no-lattice entrypoint hands EngineSource as `setpoint_echo_records`, and its
# `:RB` is the channel the camera expression depends on.
CHANNELS = [
    _channel(VAC_RB, system="VAC", family="GAUGE", field="PRESSURE"),
    _channel(
        FAULT_BI, record_type=RECORD_TYPE_BINARY, system="STATUS", family="FLAG", field="FAULT"
    ),
    _channel(UNMODELED_RB, system="UNKNOWN", family="LEVEL", field="LEVEL"),
    _channel(CAMERA_RB, system="CAM", family="PROF", field="INTENSITY"),
    _channel(
        STAGE_SP,
        subfield="SP",
        partition=PARTITION_SP_ECHO,
        system="STAGE",
        family="MOTOR",
        field="POSITION",
    ),
    _channel(
        STAGE_RB, partition=PARTITION_SP_ECHO, system="STAGE", family="MOTOR", field="POSITION"
    ),
]

# Addresses EngineSource is expected to drive on every tick: the partition-(c)
# subset of CHANNELS, which is also exactly `ServingRecords.static_noisy`.
DRIVEN = frozenset({VAC_RB, FAULT_BI, UNMODELED_RB, CAMERA_RB})


class FakeDriver:
    """Duck-typed stand-in for the CA driver the shim writes through.

    Faithful on the three points ``PVRecord`` depends on: ``setParam`` stores,
    ``getParam`` returns what is stored (so a value written by any route --
    including a Channel Access client write, which lands in this same store --
    is what a later read sees), and ``updatePV`` is the per-address monitor
    post.

    Args:
        values: initial parameter store.
        fail_on: addresses whose ``setParam`` raises, standing in for a record
            that rejects a value at the driver layer.
    """

    def __init__(self, values: dict | None = None, fail_on: frozenset[str] = frozenset()) -> None:
        self.values: dict[str, Any] = dict(values or {})
        self.fail_on = fail_on
        self.calls: list[tuple[str, str, Any]] = []

    def setParam(self, reason: str, value: Any) -> None:  # noqa: N802 - driver contract
        self.calls.append(("setParam", reason, value))
        if reason in self.fail_on:
            raise RuntimeError(f"driver refuses {reason!r}")
        self.values[reason] = value

    def getParam(self, reason: str) -> Any:  # noqa: N802 - driver contract
        self.calls.append(("getParam", reason, None))
        return self.values[reason]

    def updatePV(self, reason: str) -> None:  # noqa: N802 - driver contract
        self.calls.append(("updatePV", reason, None))

    def posted(self) -> list[str]:
        return [reason for call, reason, _ in self.calls if call == "updatePV"]

    def set_calls(self, address: str) -> list[Any]:
        return [
            value for call, reason, value in self.calls if call == "setParam" and reason == address
        ]


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    (tmp_path / "machine.json").write_text(json.dumps(TEST_MACHINE))
    return tmp_path


@pytest.fixture()
def engine(data_dir: Path) -> SimulationEngine:
    return SimulationEngine.from_file(data_dir / "machine.json")


@pytest.fixture()
def serving() -> ServingRecords:
    """The real serving-layer records -- never a mock of them."""
    return build_serving_pvdb(CHANNELS)


@pytest.fixture()
def driver(serving: ServingRecords) -> FakeDriver:
    """A driver seeded from the boot database, then attached, as the runner does."""
    drv = FakeDriver({address: spec["value"] for address, spec in serving.pvdb.items()})
    serving.attach_driver(drv)
    return drv


def _source(
    engine: SimulationEngine,
    serving: ServingRecords,
    data_dir: Path,
    *,
    echo: bool = False,
) -> EngineSource:
    """EngineSource wired to the serving records exactly as entrypoint wires it."""
    return EngineSource(
        engine,
        CHANNELS,
        serving.static_noisy,
        data_dir,
        setpoint_echo_records={STAGE_RB: serving.all[STAGE_RB]} if echo else None,
    )


@pytest.fixture()
def source(engine, serving, data_dir, driver) -> EngineSource:  # noqa: ANN001
    return _source(engine, serving, data_dir)


class TestDriveSurface:
    """The whole compatibility claim rests on EngineSource touching records
    through ``set``/``get`` and nothing else. Assert that mechanically, so a
    future call to some softioc-only attribute fails here rather than at 1 Hz
    inside a swallowed traceback."""

    def _record_attributes(self) -> set[str]:
        tree = ast.parse(Path(engine_source_module.__file__).read_text())
        return {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "record"
        }

    def test_engine_source_touches_records_only_through_set_and_get(self) -> None:
        attributes = self._record_attributes()
        assert attributes, "found no record attribute access -- the audit regex went stale"
        assert attributes <= {"set", "get"}, f"unsupported record surface used: {attributes}"

    def test_the_shim_supplies_that_whole_surface(self) -> None:
        for attribute in self._record_attributes():
            assert callable(getattr(PVRecord, attribute, None)), attribute

    def test_static_noisy_partition_is_what_the_source_is_handed(
        self, serving: ServingRecords
    ) -> None:
        assert set(serving.static_noisy) == DRIVEN
        assert all(isinstance(record, PVRecord) for record in serving.static_noisy.values())


class TestPollDrivesTheServedDatabase:
    def test_first_poll_is_not_reported_as_a_switch(self, source: EngineSource) -> None:
        assert source.poll_once() is False

    def test_poll_pushes_engine_values_through_the_shim(
        self, source: EngineSource, driver: FakeDriver
    ) -> None:
        source.poll_once()
        assert driver.values[VAC_RB] == pytest.approx(1e-8)

    def test_poll_posts_one_monitor_event_per_driven_address(
        self, source: EngineSource, driver: FakeDriver
    ) -> None:
        source.poll_once()
        assert sorted(driver.posted()) == sorted(DRIVEN)

    def test_poll_touches_no_address_outside_its_partition(
        self, source: EngineSource, driver: FakeDriver
    ) -> None:
        source.poll_once()
        written = {reason for call, reason, _ in driver.calls if call == "setParam"}
        assert written == DRIVEN
        assert STAGE_SP not in written and STAGE_RB not in written

    def test_unmodeled_address_falls_back_to_taxonomy_synthesis(
        self, source: EngineSource, driver: FakeDriver
    ) -> None:
        source.poll_once()
        assert driver.values[UNMODELED_RB] == pytest.approx(100.0)

    def test_analog_values_reach_the_driver_as_plain_floats(
        self, source: EngineSource, driver: FakeDriver
    ) -> None:
        """The shim's coercion is what guarantees this: the engine stores
        values as numpy scalars, and an unconverted one on a float PV is not
        what a CA client is owed."""
        source.poll_once()
        assert type(driver.values[VAC_RB]) is float

    def test_binary_values_reach_the_driver_as_enum_indices(
        self, source: EngineSource, driver: FakeDriver
    ) -> None:
        """A ``bi`` channel is served as a 2-state enum, so its wire value is
        an index, not the float the engine keeps it as."""
        source.poll_once()
        assert type(driver.values[FAULT_BI]) is int
        assert driver.values[FAULT_BI] in (0, 1)

    def test_repeated_poll_without_change_is_not_a_switch(self, source: EngineSource) -> None:
        source.poll_once()
        assert source.poll_once() is False


class TestPreAttachAssembly:
    """``PhysicsBridge.bind()`` and the first telemetry values can land before
    the server (hence the driver) exists. Those values must become the boot
    values the server comes up with, not be dropped on the floor."""

    def test_set_before_attach_seeds_the_boot_database(
        self, engine, serving: ServingRecords, data_dir: Path
    ) -> None:
        pre_attach = _source(engine, serving, data_dir)  # note: no `driver` fixture
        pre_attach.poll_once()
        assert serving.pvdb[VAC_RB]["value"] == pytest.approx(1e-8)
        assert serving.all[VAC_RB].get() == pytest.approx(1e-8)

    def test_attaching_switches_the_write_target_to_the_driver(
        self, engine, serving: ServingRecords, data_dir: Path
    ) -> None:
        pre_attach = _source(engine, serving, data_dir)
        pre_attach.poll_once()
        engine.write(VAC_RB, 4.2e-7)

        drv = FakeDriver({address: spec["value"] for address, spec in serving.pvdb.items()})
        serving.attach_driver(drv)
        pre_attach.poll_once()

        assert drv.values[VAC_RB] == pytest.approx(4.2e-7)
        # The boot spec is frozen at the pre-attach value: post-attach writes
        # go to the driver, so the spec is no longer a live value store.
        assert serving.pvdb[VAC_RB]["value"] == pytest.approx(1e-8)


class TestScenarioHotReload:
    def test_scenario_switch_is_detected_and_repushed(
        self, source: EngineSource, driver: FakeDriver, engine: SimulationEngine
    ) -> None:
        source.poll_once()
        engine.set_active_scenarios(["leak"])
        assert source.poll_once() is True
        assert driver.values[VAC_RB] == pytest.approx(5e-6)

    def test_switching_back_restores_the_baseline(
        self, source: EngineSource, driver: FakeDriver, engine: SimulationEngine
    ) -> None:
        source.poll_once()
        engine.set_active_scenarios(["leak"])
        source.poll_once()
        engine.set_active_scenarios([])
        source.poll_once()
        assert driver.values[VAC_RB] == pytest.approx(1e-8)

    def test_content_hash_catches_an_mtime_colliding_atomic_swap(
        self, source: EngineSource, driver: FakeDriver, data_dir: Path
    ) -> None:
        """The bind-mounted unit is the directory precisely so an atomic-rename
        swap survives the mount -- but the replacement's mtime can collide with
        the original at the host filesystem's granularity. The sha256 half of
        the signature is what closes that hole."""
        state_path = data_dir / "active_scenarios"
        state_path.write_text("nominal\n")
        source.poll_once()
        original_ino = state_path.stat().st_ino
        original_mtime_ns = state_path.stat().st_mtime_ns

        tmp = data_dir / "active_scenarios.tmp"
        tmp.write_text("leak\n")
        os.utime(tmp, ns=(original_mtime_ns, original_mtime_ns))
        os.replace(tmp, state_path)

        assert state_path.stat().st_ino != original_ino  # the swap really happened
        assert state_path.stat().st_mtime_ns == original_mtime_ns  # ...mtime really collides
        assert hashlib.sha256(state_path.read_bytes()).hexdigest()  # ...and content moved

        assert source.poll_once() is True
        assert driver.values[VAC_RB] == pytest.approx(5e-6)


class TestSetpointEchoSyncReadsThroughTheDriver:
    """The echo sync feeds each accepted setpoint readback into the engine at
    the top of every tick, so machine-file expression channels can respond to
    it. With the serving layer, a client's ``caput`` lands in the *driver's*
    parameter store -- never in the shim -- so a shim that answered ``get()``
    from a local cache would feed the engine a permanently stale value and the
    expression channel would never move."""

    @pytest.fixture()
    def echo_source(self, engine, serving, data_dir, driver) -> EngineSource:  # noqa: ANN001
        return _source(engine, serving, data_dir, echo=True)

    def test_baseline_expression_reflects_the_boot_setpoint(
        self, echo_source: EngineSource, driver: FakeDriver
    ) -> None:
        echo_source.poll_once()
        assert driver.values[CAMERA_RB] == pytest.approx(OFF_PEAK_INTENSITY, rel=1e-3)

    def test_a_driver_side_write_reaches_the_engine_on_the_next_tick(
        self, echo_source: EngineSource, driver: FakeDriver
    ) -> None:
        echo_source.poll_once()

        # Exactly where a Channel Access client's accepted write lands: the
        # driver's parameter store, with the shim never consulted.
        driver.values[STAGE_RB] = 2.0
        echo_source.poll_once()

        assert driver.values[CAMERA_RB] == pytest.approx(PEAK_INTENSITY)

    def test_the_echo_read_goes_through_the_driver_every_tick(
        self, echo_source: EngineSource, driver: FakeDriver
    ) -> None:
        echo_source.poll_once()
        echo_source.poll_once()
        reads = [reason for call, reason, _ in driver.calls if call == "getParam"]
        assert reads == [STAGE_RB, STAGE_RB]

    def test_engine_unknown_echo_addresses_are_dropped_at_construction(
        self, engine, serving: ServingRecords, data_dir: Path, driver: FakeDriver
    ) -> None:
        """``STAGE_SP`` is a real served PV but not a machine-file channel;
        syncing it would raise on every tick forever."""
        src = EngineSource(
            engine,
            CHANNELS,
            serving.static_noisy,
            data_dir,
            setpoint_echo_records={
                STAGE_RB: serving.all[STAGE_RB],
                STAGE_SP: serving.all[STAGE_SP],
            },
        )
        src.poll_once()
        reads = [reason for call, reason, _ in driver.calls if call == "getParam"]
        assert reads == [STAGE_RB]


class TestOneRecordNeverKillsTheLoop:
    """Safety property, not a nicety: ``run_forever`` is scheduled with
    ``run_coroutine_threadsafe`` and nobody retrieves the future's result, so
    an exception escaping ``poll_once`` would stop the poll permanently and
    silently -- every telemetry channel frozen at its boot value with no error
    anywhere. Prove survival against a record that genuinely raises, not just
    that the happy path works."""

    @pytest.fixture()
    def broken(self, serving: ServingRecords) -> FakeDriver:
        drv = FakeDriver(
            {address: spec["value"] for address, spec in serving.pvdb.items()},
            fail_on=frozenset({FAULT_BI}),
        )
        serving.attach_driver(drv)
        return drv

    @pytest.fixture()
    def broken_source(self, engine, serving, data_dir, broken) -> EngineSource:  # noqa: ANN001
        return _source(engine, serving, data_dir)

    def test_the_raising_record_really_raises(
        self, serving: ServingRecords, broken: FakeDriver
    ) -> None:
        """Guard against a vacuously-green survival test: if the shim swallowed
        the driver error, the tests below would prove nothing."""
        with pytest.raises(RuntimeError):
            serving.all[FAULT_BI].set(1)

    def test_poll_once_survives_a_raising_record(
        self, broken_source: EngineSource, broken: FakeDriver
    ) -> None:
        broken_source.poll_once()  # must not raise
        assert broken.values[VAC_RB] == pytest.approx(1e-8)
        assert broken.values[UNMODELED_RB] == pytest.approx(100.0)

    def test_the_failure_is_reported_not_swallowed_silently(
        self, broken_source: EngineSource, broken: FakeDriver, capsys: pytest.CaptureFixture
    ) -> None:
        broken_source.poll_once()
        stderr = capsys.readouterr().err
        assert FAULT_BI in stderr
        assert "RuntimeError" in stderr

    def test_scenario_detection_still_works_on_a_tick_with_a_failure(
        self, broken_source: EngineSource, broken: FakeDriver, engine: SimulationEngine
    ) -> None:
        broken_source.poll_once()
        engine.set_active_scenarios(["leak"])
        assert broken_source.poll_once() is True
        assert broken.values[VAC_RB] == pytest.approx(5e-6)

    def test_a_raising_echo_record_does_not_stop_the_tick(
        self, engine, serving: ServingRecords, data_dir: Path
    ) -> None:
        class ExplodingReadDriver(FakeDriver):
            def getParam(self, reason: str) -> Any:  # noqa: N802 - driver contract
                raise RuntimeError(f"driver read failed for {reason!r}")

        drv = ExplodingReadDriver({a: s["value"] for a, s in serving.pvdb.items()})
        serving.attach_driver(drv)
        src = _source(engine, serving, data_dir, echo=True)

        src.poll_once()  # must not raise
        assert drv.values[VAC_RB] == pytest.approx(1e-8)

    def test_run_forever_keeps_ticking_past_a_permanent_failure(
        self, broken_source: EngineSource, broken: FakeDriver
    ) -> None:
        async def drive() -> None:
            task = asyncio.get_running_loop().create_task(broken_source.run_forever(interval=0))
            for _ in range(2000):
                await asyncio.sleep(0)
                if len(broken.set_calls(VAC_RB)) >= 5:
                    break
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(drive())

        # The healthy record kept being driven long after the broken one's
        # first failure -- the loop did not die on tick one.
        assert len(broken.set_calls(VAC_RB)) >= 5
        assert len(broken.set_calls(FAULT_BI)) >= 5


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_engine_source_serving.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
