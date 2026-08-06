"""Tests for the co-hosted PV database and its record shims.

Host-side and in-process only: nothing here constructs a Channel Access
server, binds a port or imports the CA server library. The database is plain
data and the shim talks to a duck-typed driver, so both are fully checkable
without a server -- and must stay that way, because this suite runs
concurrently with container work that owns the real CA ports. Live-CA
behaviour (monitor delivery, alarm severity over the wire) is proven in the
container e2e suite instead.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from osprey.services.virtual_accelerator.manifest import (
    PARTITION_PYAT_COUPLED,
    PARTITION_SP_ECHO,
    PARTITION_STATIC_NOISY,
    RECORD_TYPE_ANALOG,
    RECORD_TYPE_BINARY,
    RECORD_TYPE_LONG_STRING,
    RECORD_TYPE_MBB,
    RECORD_TYPE_STRING,
    build_manifest,
)
from osprey.services.virtual_accelerator.serving.pvdb import (
    ALARM_LIMIT_KEYS,
    ANALOG_PRECISION,
    BINARY_ENUM_STATES,
    LONG_STRING_LENGTH,
    MBB_STATE_COUNT,
    ManifestContractError,
    ServingRecords,
    build_serving_pvdb,
)

# Pinned namespace counts. These are a client-visible contract, not an
# implementation detail: the channel-finder databases, the safety limits file
# and every CA client agree on exactly this channel set.
EXPECTED_TOTAL = 2908
EXPECTED_PYAT_COUPLED = 840
EXPECTED_STATIC_NOISY = 1922
EXPECTED_MAGNET_SETPOINTS = 348
EXPECTED_MAGNET_READBACKS = 348
EXPECTED_BPM_READINGS = 144
EXPECTED_SETPOINTS = 396

# Floor for this module's own test count -- a guard against a refactor that
# leaves the file importable but empty, which would otherwise pass silently.
MIN_COLLECTED_TESTS = 30


def _channel(
    address: str,
    *,
    record_type: str = RECORD_TYPE_ANALOG,
    subfield: str = "RB",
    partition: str = PARTITION_STATIC_NOISY,
    noise: bool = False,
    ring: str = "SR",
    system: str = "DIAG",
    family: str = "TEST",
    device: str = "01",
    field: str = "VALUE",
) -> dict:
    """One synthetic manifest channel."""
    return {
        "address": address,
        "ring": ring,
        "system": system,
        "family": family,
        "device": device,
        "field": field,
        "subfield": subfield,
        "partition": partition,
        "record_type": record_type,
        "noise": noise,
    }


def _sp_rb_pair(prefix: str, *, partition: str = PARTITION_SP_ECHO) -> list[dict]:
    """A setpoint and its paired readback, sharing device identity."""
    return [
        _channel(
            f"{prefix}:SP",
            subfield="SP",
            partition=partition,
            system="MAG",
            family="HCM",
            field="CURRENT",
        ),
        _channel(
            f"{prefix}:RB",
            subfield="RB",
            partition=partition,
            system="MAG",
            family="HCM",
            field="CURRENT",
        ),
    ]


class FakeDriver:
    """Duck-typed stand-in for the CA driver: records every call."""

    def __init__(self, values: dict | None = None) -> None:
        self.values = dict(values or {})
        self.calls: list[tuple[str, str, object]] = []

    def setParam(self, reason: str, value: object) -> None:  # noqa: N802 - driver contract
        self.values[reason] = value
        self.calls.append(("setParam", reason, value))

    def getParam(self, reason: str) -> object:  # noqa: N802 - driver contract
        self.calls.append(("getParam", reason, None))
        return self.values[reason]

    def updatePV(self, reason: str) -> None:  # noqa: N802 - driver contract
        self.calls.append(("updatePV", reason, None))

    def posted(self) -> list[str]:
        return [reason for call, reason, _ in self.calls if call == "updatePV"]


# (manifest record type, served type, element count, enum state count)
MAPPING_CASES = [
    (RECORD_TYPE_ANALOG, "float", 1, 0),
    (RECORD_TYPE_BINARY, "enum", 1, 2),
    (RECORD_TYPE_MBB, "enum", 1, MBB_STATE_COUNT),
    (RECORD_TYPE_STRING, "string", 1, 0),
    (RECORD_TYPE_LONG_STRING, "char", LONG_STRING_LENGTH, 0),
]


class TestRecordTypeMapping:
    """One case per record type: served type, element count, enum states."""

    @pytest.mark.parametrize(("record_type", "served_type", "count", "enum_count"), MAPPING_CASES)
    def test_record_type_maps_to_served_type_and_count(
        self, record_type: str, served_type: str, count: int, enum_count: int
    ) -> None:
        records = build_serving_pvdb([_channel("X:1", record_type=record_type)])
        spec = records.pvdb["X:1"]

        assert spec["type"] == served_type
        assert spec["count"] == count
        assert len(spec.get("enums", [])) == enum_count

    def test_mapping_table_covers_every_manifest_record_type(self) -> None:
        """Exhaustiveness: a new manifest record type must land here, not in
        an unhandled-type failure at boot."""
        assert {case[0] for case in MAPPING_CASES} == {
            RECORD_TYPE_ANALOG,
            RECORD_TYPE_BINARY,
            RECORD_TYPE_MBB,
            RECORD_TYPE_STRING,
            RECORD_TYPE_LONG_STRING,
        }

    def test_binary_states_are_the_two_boolean_labels(self) -> None:
        records = build_serving_pvdb([_channel("X:1", record_type=RECORD_TYPE_BINARY)])

        assert records.pvdb["X:1"]["enums"] == list(BINARY_ENUM_STATES)

    def test_mbb_states_are_numeric_labels(self) -> None:
        """The manifest carries no state labels, so an mbb is a numeric enum."""
        records = build_serving_pvdb([_channel("X:1", record_type=RECORD_TYPE_MBB)])

        assert records.pvdb["X:1"]["enums"] == [str(state) for state in range(MBB_STATE_COUNT)]

    def test_analog_declares_display_precision(self) -> None:
        """Left unset, the server renders every analog channel as an integer."""
        records = build_serving_pvdb([_channel("X:1", record_type=RECORD_TYPE_ANALOG)])

        assert records.pvdb["X:1"]["prec"] == ANALOG_PRECISION

    def test_enum_state_lists_are_not_shared_between_records(self) -> None:
        records = build_serving_pvdb(
            [
                _channel("X:1", record_type=RECORD_TYPE_BINARY),
                _channel("X:2", record_type=RECORD_TYPE_BINARY),
            ]
        )
        records.pvdb["X:1"]["enums"].append("EXTRA")

        assert records.pvdb["X:2"]["enums"] == list(BINARY_ENUM_STATES)


class TestDriveLimits:
    """DRVL/DRVH wiring -- and its separation from alarm limits."""

    def test_drive_limits_become_the_control_band(self) -> None:
        channels = _sp_rb_pair("SR:MAG:HCM:01:CURRENT", partition=PARTITION_PYAT_COUPLED)
        records = build_serving_pvdb(
            channels, drive_limits={"SR:MAG:HCM:01:CURRENT:SP": (-12.5, 12.5)}
        )
        spec = records.pvdb["SR:MAG:HCM:01:CURRENT:SP"]

        assert spec["lolim"] == -12.5
        assert spec["hilim"] == 12.5

    def test_limits_are_coerced_to_float(self) -> None:
        records = build_serving_pvdb(
            [_channel("X:1", subfield="SP")], drive_limits={"X:1": (0, 10)}
        )
        spec = records.pvdb["X:1"]

        assert isinstance(spec["lolim"], float)
        assert isinstance(spec["hilim"], float)

    def test_channel_without_limits_declares_none(self) -> None:
        records = build_serving_pvdb(
            [_channel("X:1", subfield="SP"), _channel("X:2", subfield="SP")],
            drive_limits={"X:1": (0.0, 1.0)},
        )

        assert "lolim" not in records.pvdb["X:2"]
        assert "hilim" not in records.pvdb["X:2"]

    def test_no_drive_limits_map_leaves_every_channel_unbounded(self) -> None:
        records = build_serving_pvdb([_channel("X:1", subfield="SP")])

        assert "lolim" not in records.pvdb["X:1"]

    def test_drive_limits_are_never_alarm_limits(self) -> None:
        """A value clamped to the top of its band is a normal outcome, not a
        limit violation: emitting alarm limits here would make the clamped
        readback alarm MAJOR for every client."""
        records = build_serving_pvdb(
            [_channel("X:1", subfield="SP")], drive_limits={"X:1": (-1.0, 1.0)}
        )
        spec = records.pvdb["X:1"]

        assert [key for key in ALARM_LIMIT_KEYS if key in spec] == []


class TestBootValues:
    """Boot-value seeding: PVs come up holding machine state, not zeros."""

    def test_setpoint_and_readback_are_seeded(self) -> None:
        channels = _sp_rb_pair("SR:MAG:HCM:01:CURRENT")
        records = build_serving_pvdb(
            channels,
            boot_values={
                "SR:MAG:HCM:01:CURRENT:SP": 3.25,
                "SR:MAG:HCM:01:CURRENT:RB": 3.24,
            },
        )

        assert records.pvdb["SR:MAG:HCM:01:CURRENT:SP"]["value"] == 3.25
        assert records.pvdb["SR:MAG:HCM:01:CURRENT:RB"]["value"] == 3.24

    def test_seed_is_ignored_for_other_subfields(self) -> None:
        """Telemetry gets its value from the engine source's first poll tick,
        not from a stale scenario seed."""
        records = build_serving_pvdb(
            [_channel("SR:DIAG:BPM:01:POSITION:X", subfield="X")],
            boot_values={"SR:DIAG:BPM:01:POSITION:X": 7.0},
        )

        assert records.pvdb["SR:DIAG:BPM:01:POSITION:X"]["value"] == 0.0

    def test_unseeded_setpoint_boots_at_the_type_default(self) -> None:
        records = build_serving_pvdb([_channel("X:1", subfield="SP")], boot_values={"OTHER": 5.0})

        assert records.pvdb["X:1"]["value"] == 0.0

    @pytest.mark.parametrize(
        ("record_type", "default"),
        [
            (RECORD_TYPE_ANALOG, 0.0),
            (RECORD_TYPE_BINARY, 0),
            (RECORD_TYPE_MBB, 0),
            (RECORD_TYPE_STRING, ""),
            (RECORD_TYPE_LONG_STRING, ""),
        ],
    )
    def test_type_default_when_unseeded(self, record_type: str, default: object) -> None:
        records = build_serving_pvdb([_channel("X:1", record_type=record_type)])

        assert records.pvdb["X:1"]["value"] == default

    def test_seed_is_coerced_to_the_record_type(self) -> None:
        """Scenario data stores every channel as a float, including the ones
        served as discrete states."""
        records = build_serving_pvdb(
            [_channel("X:1", record_type=RECORD_TYPE_BINARY, subfield="RB")],
            boot_values={"X:1": 1.0},
        )
        value = records.pvdb["X:1"]["value"]

        assert value == 1
        assert isinstance(value, int)


class TestRecordShim:
    """The ``.set()``/``.get()`` surface the existing value sources drive."""

    def _one_record(self, **kwargs):
        records = build_serving_pvdb([_channel("X:1", **kwargs)])
        return records, records.all["X:1"]

    def test_set_commits_and_posts_a_monitor_event(self) -> None:
        records, record = self._one_record()
        driver = FakeDriver({"X:1": 0.0})
        records.attach_driver(driver)

        record.set(1.5)

        assert driver.values["X:1"] == 1.5
        assert driver.posted() == ["X:1"]

    def test_set_posts_only_its_own_address(self) -> None:
        """Per-address posting, not a database-wide sweep: the telemetry
        source sets ~1,900 records per tick over a 2,908-PV database."""
        records = build_serving_pvdb([_channel("X:1"), _channel("X:2")])
        driver = FakeDriver({"X:1": 0.0, "X:2": 0.0})
        records.attach_driver(driver)

        records.all["X:1"].set(2.0)

        assert driver.posted() == ["X:1"]

    def test_get_reads_through_the_driver(self) -> None:
        """Not from a local cache: a setpoint written over CA lands in the
        driver, and ``.get()`` must see it."""
        records, record = self._one_record(subfield="SP")
        driver = FakeDriver({"X:1": 0.0})
        records.attach_driver(driver)

        driver.setParam("X:1", 9.0)

        assert record.get() == 9.0

    def test_set_coerces_to_the_record_type(self) -> None:
        records, record = self._one_record(record_type=RECORD_TYPE_BINARY)
        driver = FakeDriver({"X:1": 0})
        records.attach_driver(driver)

        record.set(1.0)

        assert driver.values["X:1"] == 1
        assert isinstance(driver.values["X:1"], int)

    def test_set_before_a_driver_exists_becomes_the_boot_value(self) -> None:
        """The database is built before the server and driver exist, so
        values pushed during assembly (the first BPM push) have to land as
        the values the server comes up with."""
        records, record = self._one_record()

        record.set(4.5)

        assert records.pvdb["X:1"]["value"] == 4.5
        assert record.get() == 4.5

    def test_attach_driver_wires_every_record(self) -> None:
        records = build_serving_pvdb([_channel("X:1"), _channel("X:2")])
        driver = FakeDriver({"X:1": 0.0, "X:2": 0.0})
        records.attach_driver(driver)

        for record in records.all.values():
            record.set(1.0)

        assert sorted(driver.posted()) == ["X:1", "X:2"]
        assert records.pvdb["X:1"]["value"] == 0.0  # committed to the driver, not the spec

    def test_shim_offers_exactly_what_the_value_sources_call(self) -> None:
        """``EngineSource`` and ``PhysicsBridge`` drive their records with
        ``set(value)`` and ``get()`` and nothing else."""
        _records, record = self._one_record()

        assert callable(record.set)
        assert callable(record.get)


class TestPartitionsAndPairing:
    """The dicts each downstream consumer is handed."""

    def test_pyat_coupled_records_are_grouped(self) -> None:
        channels = _sp_rb_pair("SR:MAG:HCM:01:CURRENT", partition=PARTITION_PYAT_COUPLED)
        records = build_serving_pvdb(channels)

        assert sorted(records.pyat_coupled) == [
            "SR:MAG:HCM:01:CURRENT:RB",
            "SR:MAG:HCM:01:CURRENT:SP",
        ]
        assert records.static_noisy == {}

    def test_static_noisy_records_are_grouped(self) -> None:
        records = build_serving_pvdb([_channel("X:1", partition=PARTITION_STATIC_NOISY)])

        assert list(records.static_noisy) == ["X:1"]
        assert records.pyat_coupled == {}

    def test_setpoint_is_paired_with_its_readback(self) -> None:
        records = build_serving_pvdb(_sp_rb_pair("SR:MAG:HCM:01:CURRENT"))

        assert records.setpoint_readbacks == {
            "SR:MAG:HCM:01:CURRENT:SP": "SR:MAG:HCM:01:CURRENT:RB"
        }

    def test_pyat_setpoint_may_stand_alone(self) -> None:
        """Its physics effect is observed on the BPM readbacks."""
        channels = [
            _channel(
                "SR:MAG:HCM:01:CURRENT:SP",
                subfield="SP",
                partition=PARTITION_PYAT_COUPLED,
                system="MAG",
                family="HCM",
                field="CURRENT",
            )
        ]
        records = build_serving_pvdb(channels)

        assert records.setpoint_readbacks == {}


class TestAsyncSetpoints:
    """Asynchronous-write declaration is opt-in."""

    def test_setpoints_are_synchronous_by_default(self) -> None:
        records = build_serving_pvdb(_sp_rb_pair("SR:MAG:HCM:01:CURRENT"))

        assert all("asyn" not in spec for spec in records.pvdb.values())

    def test_opt_in_marks_setpoints_only(self) -> None:
        records = build_serving_pvdb(_sp_rb_pair("SR:MAG:HCM:01:CURRENT"), async_setpoints=True)

        assert records.pvdb["SR:MAG:HCM:01:CURRENT:SP"]["asyn"] is True
        assert "asyn" not in records.pvdb["SR:MAG:HCM:01:CURRENT:RB"]


class TestContractViolations:
    def test_unknown_record_type_is_rejected(self) -> None:
        with pytest.raises(ManifestContractError, match="unknown record_type"):
            build_serving_pvdb([_channel("X:1", record_type="waveform")])

    @pytest.mark.parametrize(
        "record_type", [RECORD_TYPE_BINARY, RECORD_TYPE_MBB, RECORD_TYPE_LONG_STRING]
    )
    def test_noisy_discrete_channel_is_rejected(self, record_type: str) -> None:
        with pytest.raises(ManifestContractError, match="reject noise"):
            build_serving_pvdb([_channel("X:1", record_type=record_type, noise=True)])

    def test_duplicate_address_is_rejected(self) -> None:
        """Silently collapsing two channels into one PV would break the
        pinned channel count without any client noticing."""
        with pytest.raises(ManifestContractError, match="duplicate"):
            build_serving_pvdb([_channel("X:1"), _channel("X:1")])

    def test_sp_echo_setpoint_without_a_readback_is_rejected(self) -> None:
        channels = [
            _channel(
                "SR:RF:CAV:01:VOLTAGE:SP",
                subfield="SP",
                partition=PARTITION_SP_ECHO,
                system="RF",
                family="CAV",
                field="VOLTAGE",
            )
        ]
        with pytest.raises(ManifestContractError, match="no matching RB"):
            build_serving_pvdb(channels)


class TestFullManifest:
    """The real namespace: pinned counts and database-wide invariants."""

    @pytest.fixture(scope="class")
    def records(self) -> ServingRecords:
        from osprey.services.virtual_accelerator.entrypoint import (
            _load_boot_values,
            _load_drive_limits,
        )

        return build_serving_pvdb(
            build_manifest()["channels"],
            drive_limits=_load_drive_limits(),
            boot_values=_load_boot_values(),
        )

    def test_every_manifest_channel_is_served_once(self, records: ServingRecords) -> None:
        assert len(records.pvdb) == EXPECTED_TOTAL
        assert len(records.all) == EXPECTED_TOTAL

    def test_partition_counts(self, records: ServingRecords) -> None:
        assert len(records.pyat_coupled) == EXPECTED_PYAT_COUPLED
        assert len(records.static_noisy) == EXPECTED_STATIC_NOISY

    def test_magnet_and_bpm_counts(self, records: ServingRecords) -> None:
        setpoints = [a for a in records.pyat_coupled if a.endswith(":CURRENT:SP")]
        readbacks = [a for a in records.pyat_coupled if a.endswith(":CURRENT:RB")]
        bpms = [a for a in records.pyat_coupled if a.endswith((":POSITION:X", ":POSITION:Y"))]

        assert len(setpoints) == EXPECTED_MAGNET_SETPOINTS
        assert len(readbacks) == EXPECTED_MAGNET_READBACKS
        assert len(bpms) == EXPECTED_BPM_READINGS

    def test_every_setpoint_is_paired_and_limited(self, records: ServingRecords) -> None:
        setpoints = [a for a in records.pvdb if a.endswith(":SP")]
        limited = [a for a, spec in records.pvdb.items() if "hilim" in spec]

        assert len(setpoints) == EXPECTED_SETPOINTS
        assert sorted(records.setpoint_readbacks) == sorted(setpoints)
        assert sorted(limited) == sorted(setpoints)

    def test_no_channel_is_server_scanned(self, records: ServingRecords) -> None:
        """A scanning PV is skipped by the driver's monitor post, so its
        driver-pushed values would silently never reach a subscriber."""
        assert [a for a, spec in records.pvdb.items() if "scan" in spec] == []

    def test_no_channel_declares_alarm_limits(self, records: ServingRecords) -> None:
        offenders = [
            address
            for address, spec in records.pvdb.items()
            if any(key in spec for key in ALARM_LIMIT_KEYS)
        ]

        assert offenders == []

    def test_boot_values_reach_the_setpoints(self, records: ServingRecords) -> None:
        """Not a zeroed machine: real setpoints come up at scenario state."""
        seeded = [
            address
            for address in records.setpoint_readbacks
            if records.pvdb[address]["value"] != 0.0
        ]

        assert len(seeded) > 0

    def test_every_served_value_matches_its_declared_type(self, records: ServingRecords) -> None:
        for address, spec in records.pvdb.items():
            value = spec["value"]
            if spec["type"] == "float":
                assert isinstance(value, float), address
            elif spec["type"] == "enum":
                assert isinstance(value, int), address
            else:
                assert isinstance(value, str), address


class TestPackageImportStaysLight:
    """The serving package sits on the no-lattice boot path."""

    def test_importing_the_package_pulls_in_no_serving_or_physics_stack(self) -> None:
        script = (
            "import sys;"
            "import osprey.services.virtual_accelerator.serving as s;"
            "heavy=[m for m in ('lume_pva_apg','pcaspy','p4p','at','lume') if m in sys.modules];"
            "print(heavy);"
            "assert not heavy, heavy;"
            "assert callable(s.build_serving_pvdb)"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
        )

        assert result.returncode == 0, result.stderr

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        import osprey.services.virtual_accelerator.serving as serving

        with pytest.raises(AttributeError):
            serving.no_such_name


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_serving_pvdb.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
