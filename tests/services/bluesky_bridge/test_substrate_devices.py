"""Unit tests for the canonical EPICS-substrate plan-device derivation.

Covers ``osprey.services.bluesky_bridge.substrate_devices`` -- the single
host-side producer shared by the build's device-file staging
(``compose_generator._stage_bluesky_devices``) and ``tests/e2e/_orm_stack.py``:
the device document derived from the channel roster's records, and the atomic
write of that document to a file the queueserver worker mounts.

Two properties are load-bearing here and nowhere else.

The first is that the derivation is *filter-free*. Its input already says which
channels exist and which way they point; a ring/family/grammar opinion at this
seam is what made a build report 144 devices for a 2908-channel machine. So the
tests feed it addresses no address grammar would recognise and require every one
of them through, and pin that the module's own text carries no partition
vocabulary.

The second is the readback rule: a settable names a readback only where the
roster paired one. An unpaired setpoint has to reach the worker with no
``readback`` key, which is how the loader is told the device reads its setpoint
back -- not with the setpoint restated as its own readback, which would claim a
pairing the roster did not make.
"""

from __future__ import annotations

from collections.abc import Iterator
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import pytest
import yaml

import osprey.channel_roster as channel_roster
from osprey.channel_roster import (
    ChannelRecord,
    RosterSource,
    RosterSourceKind,
    registered_channels,
)
from osprey.services.bluesky_bridge import substrate_devices
from osprey.services.bluesky_bridge.substrate_devices import (
    devices_document,
    write_devices_file,
)

#: What the shipped demo corpus holds, pinned alongside
#: ``tests/channel_roster/test_facade.py`` and
#: ``tests/services/facility_knowledge/test_demo_ttl_consistency.py``.
DEMO_WRITES = 396
DEMO_READS = 2512

_SOURCE = RosterSource(kind=RosterSourceKind.GRAPH, path=Path("/data/demo_machine.ttl"))


def _write(address: str, readback: str | None = None) -> ChannelRecord:
    """A write-direction record, optionally carrying the readback pairing gave it."""
    return ChannelRecord(address=address, source=_SOURCE, direction="write", readback=readback)


def _read(address: str) -> ChannelRecord:
    """A read-direction record."""
    return ChannelRecord(address=address, source=_SOURCE, direction="read")


#: A roster slice covering every shape the document builder has to distinguish:
#: a paired settable, an unpaired settable, plain readables, and a record whose
#: direction the source could not say.
_RECORDS = (
    _write("SR:MAG:COIL:01:CURRENT:SP", readback="SR:MAG:COIL:01:CURRENT:RB"),
    _write("SR:RF:CAV:01:VOLTAGE:SP"),
    _read("SR:MAG:COIL:01:CURRENT:RB"),
    _read("SR:DIAG:MON:01:POSITION:X"),
    ChannelRecord(address="SR:DIAG:MON:01:POSITION:Y", source=_SOURCE),
)


class TestDevicesDocument:
    def test_both_top_level_keys_are_always_present(self) -> None:
        """Even a facility that yields nothing gets both keys, so a caller can
        see WHICH half came up empty instead of inferring it from a missing key."""
        assert devices_document(()) == {"settables": [], "readables": []}

    def test_every_write_record_is_a_settable_in_roster_order(self) -> None:
        document = devices_document(_RECORDS)

        assert document["settables"] == [
            {
                "name": "SR:MAG:COIL:01:CURRENT:SP",
                "setpoint": "SR:MAG:COIL:01:CURRENT:SP",
                "readback": "SR:MAG:COIL:01:CURRENT:RB",
            },
            {
                "name": "SR:RF:CAV:01:VOLTAGE:SP",
                "setpoint": "SR:RF:CAV:01:VOLTAGE:SP",
            },
        ]

    def test_every_read_record_is_its_own_readable(self) -> None:
        """One device per readable channel, name and pv both the address --
        no pairing, no grouping, no field filter."""
        document = devices_document(_RECORDS)

        assert document["readables"] == [
            {"name": "SR:MAG:COIL:01:CURRENT:RB", "pv": "SR:MAG:COIL:01:CURRENT:RB"},
            {"name": "SR:DIAG:MON:01:POSITION:X", "pv": "SR:DIAG:MON:01:POSITION:X"},
        ]

    def test_unpaired_settable_carries_no_readback_key(self) -> None:
        """Absent (not null, not the setpoint restated): that is how the loader
        is told the device reads its setpoint back."""
        document = devices_document((_write("A:B:C:SP"),))

        assert document["settables"] == [{"name": "A:B:C:SP", "setpoint": "A:B:C:SP"}]

    def test_never_emits_a_null_readback(self) -> None:
        document = devices_document(_RECORDS)

        assert all("readback" not in entry or entry["readback"] for entry in document["settables"])

    def test_a_readback_equal_to_the_setpoint_is_not_restated(self) -> None:
        document = devices_document((_write("A:B:C:SP", readback="A:B:C:SP"),))

        assert document["settables"] == [{"name": "A:B:C:SP", "setpoint": "A:B:C:SP"}]

    def test_a_record_with_no_direction_becomes_no_device(self) -> None:
        """An unknown direction is not silently demoted to a readable; the
        build refuses to stage a derived file rather than guess."""
        document = devices_document((ChannelRecord(address="A:B:C", source=_SOURCE),))

        assert document == {"settables": [], "readables": []}

    def test_the_build_validator_accepts_what_the_producer_emits(self) -> None:
        """The producer and the build's refusal gate must agree: anything this
        writes has to pass ``validate_device_document`` unchanged."""
        from osprey.services.bluesky_bridge.devices._specs_from_file import (
            validate_device_document,
        )

        assert validate_device_document(devices_document(_RECORDS)) == []


class TestTheDerivationIsFilterFree:
    """No address grammar lives here -- the roster already answered."""

    #: Addresses nothing in this module could classify: no six colon-separated
    #: parts, no ``:SP``/``:RB`` suffix, no facility vocabulary at all. A
    #: derivation that filtered would drop every one of them.
    _ALIEN = (
        _write("plain_name", readback="plain_name_rb"),
        _write("dev/motor/1/position"),
        _write("SR:MAG:COIL:01,02:CURRENT:SETPOINT", readback="SR:MAG:COIL:01,02:CURRENT:MONITOR"),
        _read("temperature"),
        _read("BR:PSU:03:STATUS"),
        _read("beamline.det.7.counts"),
    )

    def test_every_record_survives_whatever_its_address_looks_like(self) -> None:
        document = devices_document(self._ALIEN)

        assert [entry["name"] for entry in document["settables"]] == [
            "plain_name",
            "dev/motor/1/position",
            "SR:MAG:COIL:01,02:CURRENT:SETPOINT",
        ]
        assert [entry["name"] for entry in document["readables"]] == [
            "temperature",
            "BR:PSU:03:STATUS",
            "beamline.det.7.counts",
        ]

    def test_counts_are_exactly_the_roster_split(self) -> None:
        document = devices_document(self._ALIEN)

        assert len(document["settables"]) == 3
        assert len(document["readables"]) == 3

    @pytest.mark.parametrize(
        "token",
        [
            "HCM",
            "VCM",
            "ALS_U_AR",
            "classify_partition",
            "virtual_accelerator",
            "channel_limits",
        ],
    )
    def test_the_module_names_no_partition_vocabulary(self, token: str) -> None:
        """The gate this task is measured by, kept honest from inside the suite:
        no corrector/BPM family, no partition classifier, and no channel-limits
        projection anywhere in the module -- prose included."""
        text = Path(substrate_devices.__file__).read_text(encoding="utf-8")

        assert token not in text


class TestTheShippedDemoRoster:
    """The whole machine through the real facade, not a hand-built slice."""

    @pytest.fixture(autouse=True)
    def cold_cache(self) -> Iterator[None]:
        """The roster cache is process-wide by design; start and leave it empty."""
        channel_roster._roster_cache.clear()
        yield
        channel_roster._roster_cache.clear()

    @pytest.fixture
    def demo_config(self) -> Iterator[dict[str, Any]]:
        resource = (
            files("osprey.templates")
            .joinpath("apps")
            .joinpath("control_assistant")
            .joinpath("data")
            .joinpath("demo_machine.ttl")
        )
        with as_file(resource) as path:
            yield {
                "config_dir": str(path.parent),
                "channel_finder": {"pipeline_mode": "graph"},
                "services": {"graphdb": {"ttl_path": path.name}},
            }

    def test_the_demo_machine_yields_a_device_per_channel(self, demo_config) -> None:
        """396 settables / 2512 readables -- the numbers the feature exists for.
        The build that read the write-limits projection reported 144/144."""
        result = registered_channels(demo_config)

        document = devices_document(result.records)

        assert len(document["settables"]) == DEMO_WRITES
        assert len(document["readables"]) == DEMO_READS

    def test_every_demo_settable_names_the_readback_the_roster_paired(self, demo_config) -> None:
        result = registered_channels(demo_config)

        document = devices_document(result.records)

        assert all("readback" in entry for entry in document["settables"])
        assert all(
            entry["readback"] == entry["setpoint"][: -len("SP")] + "RB"
            for entry in document["settables"]
        )


class TestWriteDevicesFile:
    def test_written_yaml_parses_back_to_the_returned_document(self, tmp_path) -> None:
        path = tmp_path / "bluesky_devices.yml"

        document = write_devices_file(path, _RECORDS, source=_SOURCE)

        assert yaml.safe_load(path.read_text(encoding="utf-8")) == document
        assert document == devices_document(_RECORDS)

    def test_header_marks_the_file_generated_and_names_the_roster_source(self, tmp_path) -> None:
        """The header has to answer both questions a reader of a staged file
        asks: may I edit this (no), and which artifact is the device set a
        projection of."""
        path = tmp_path / "bluesky_devices.yml"

        write_devices_file(path, _RECORDS, source=_SOURCE)

        text = "\n".join(
            line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("#")
        )
        assert "Generated by OSPREY" in text
        assert _SOURCE.describe() in text
        assert "bluesky.devices_file" in text

    def test_header_names_a_database_source_as_the_database(self, tmp_path) -> None:
        """Provenance is the source's own phrasing, so the staged file and the
        build's fact line call the same file the same thing."""
        path = tmp_path / "bluesky_devices.yml"
        source = RosterSource(kind=RosterSourceKind.DATABASE, path=tmp_path / "channels.json")

        write_devices_file(path, _RECORDS, source=source)

        assert source.describe() in path.read_text(encoding="utf-8")

    def test_an_empty_roster_still_names_its_provenance(self, tmp_path) -> None:
        """Which is why the source is passed rather than read off a record."""
        path = tmp_path / "bluesky_devices.yml"

        document = write_devices_file(path, (), source=_SOURCE)

        assert document == {"settables": [], "readables": []}
        assert _SOURCE.describe() in path.read_text(encoding="utf-8")

    def test_leaves_no_temp_file_behind(self, tmp_path) -> None:
        path = tmp_path / "bluesky_devices.yml"

        write_devices_file(path, _RECORDS, source=_SOURCE)

        assert [entry.name for entry in tmp_path.iterdir()] == ["bluesky_devices.yml"]

    def test_rewrite_replaces_rather_than_appends(self, tmp_path) -> None:
        """Every render rewrites the staged file; a second write must not leave
        two concatenated documents behind."""
        path = tmp_path / "bluesky_devices.yml"

        write_devices_file(path, _RECORDS, source=_SOURCE)
        document = write_devices_file(path, _RECORDS, source=_SOURCE)

        assert yaml.safe_load(path.read_text(encoding="utf-8")) == document

    def test_failed_write_leaves_the_previous_document_intact(self, tmp_path, monkeypatch) -> None:
        """Atomicity is the point of the temp file: a deploy may be mounting
        this path, so a failure must not truncate what is already there."""
        import os

        path = tmp_path / "bluesky_devices.yml"
        write_devices_file(path, _RECORDS, source=_SOURCE)
        before = path.read_text(encoding="utf-8")

        def _boom(*args, **kwargs):
            raise OSError("rename failed")

        monkeypatch.setattr(os, "replace", _boom)
        with pytest.raises(OSError):
            write_devices_file(path, (), source=_SOURCE)

        assert path.read_text(encoding="utf-8") == before
        assert [entry.name for entry in tmp_path.iterdir()] == ["bluesky_devices.yml"]

    def test_written_file_is_readable_by_a_container_user(self, tmp_path) -> None:
        """The file is bind-mounted ``:ro`` into the worker; ``mkstemp``'s 0600
        would make it unreadable to any uid but the one that rendered it."""
        path = tmp_path / "bluesky_devices.yml"

        write_devices_file(path, _RECORDS, source=_SOURCE)

        assert path.stat().st_mode & 0o044 == 0o044


class TestDeviceFileRoundTrip:
    """The written file, read back by the worker's own parser."""

    # Addresses the env-var channel could not have carried: every name holds
    # colons, and one settable's device component holds a comma (16 of ALS's
    # BTS quadrupoles really do).
    _AWKWARD = (
        _write("SR:MAG:COIL:01,02:CURRENT:SP", readback="SR:MAG:COIL:01,02:CURRENT:RB"),
        _write("SR:RF:CAV:03:VOLTAGE:SP"),
        _read("SR:DIAG:MON:01:POSITION:X"),
    )

    def test_colon_and_comma_named_devices_survive_the_worker_parser(self, tmp_path) -> None:
        """Nothing on either side splits a value on any character, so an
        address-named device reaches the worker with its name and PVs intact --
        the whole reason the device file replaced the comma-separated env var.
        """
        from osprey.services.bluesky_bridge.devices._specs_from_file import specs_from_file

        path = tmp_path / "bluesky_devices.yml"
        write_devices_file(path, self._AWKWARD, source=_SOURCE)

        settables, readables = specs_from_file(path)

        assert [(spec.name, spec.setpoint_pv, spec.readback_pv) for spec in settables] == [
            (
                "SR:MAG:COIL:01,02:CURRENT:SP",
                "SR:MAG:COIL:01,02:CURRENT:SP",
                "SR:MAG:COIL:01,02:CURRENT:RB",
            ),
            ("SR:RF:CAV:03:VOLTAGE:SP", "SR:RF:CAV:03:VOLTAGE:SP", None),
        ]
        assert [(spec.name, spec.read_pv) for spec in readables] == [
            ("SR:DIAG:MON:01:POSITION:X", "SR:DIAG:MON:01:POSITION:X")
        ]

    def test_full_derivation_round_trips_with_nothing_dropped(self, tmp_path) -> None:
        """No two records name the same device, so ``_drop_duplicate_names``
        keeps every spec: what went in comes back out."""
        from osprey.services.bluesky_bridge.devices._specs_from_file import specs_from_file

        path = tmp_path / "bluesky_devices.yml"
        document = write_devices_file(path, _RECORDS, source=_SOURCE)

        settables, readables = specs_from_file(path)

        assert [spec.name for spec in settables] == [
            entry["name"] for entry in document["settables"]
        ]
        assert [spec.name for spec in readables] == [
            entry["name"] for entry in document["readables"]
        ]
        assert all(spec.name == spec.setpoint_pv for spec in settables)
        assert all(spec.name == spec.read_pv for spec in readables)
