"""Unit tests for the knowledge-graph roster reader.

Covers ``osprey.channel_roster.graph`` -- the reader that turns the Turtle
corpus a graph-mode build stages into channel records. The load-bearing
assertions are against the corpus OSPREY actually ships
(``templates/apps/control_assistant/data/demo_machine.ttl``): 2908 channels,
396 of them settable. Those are the numbers the feature exists for -- the build
that reported ``144 settable / 144 readable`` was reading the write-limits
projection, and a reader that quietly enumerated a subset of the corpus would
reproduce that bug with a new source name on it.

The rest is failure behaviour: a corpus that cannot be read, and an rdflib that
will not import, both have to come back as data an operator can be shown rather
than as an exception mid-build.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path

import pytest

from osprey.channel_roster import (
    RosterAbsenceReason,
    RosterSource,
    RosterSourceKind,
)
from osprey.channel_roster.graph import read_graph_roster

#: What the shipped demo corpus holds, pinned alongside
#: ``tests/services/facility_knowledge/test_demo_ttl_consistency.py``.
DEMO_CHANNELS = 2908
DEMO_WRITES = 396
DEMO_READS = 2512

_PREAMBLE = """\
@prefix narad_p: <https://narad.example.org/property/> .
@prefix narad_sem: <https://narad.example.org/schema/shared_semantics/> .
"""


@contextmanager
def _demo_corpus() -> Iterator[Path]:
    """Yield the shipped demo corpus as a filesystem path."""
    resource = (
        files("osprey.templates")
        .joinpath("apps")
        .joinpath("control_assistant")
        .joinpath("data")
        .joinpath("demo_machine.ttl")
    )
    with as_file(resource) as path:
        yield path


@pytest.fixture
def demo_source() -> Iterator[RosterSource]:
    """The shipped demo corpus, as a resolved roster source."""
    with _demo_corpus() as path:
        yield RosterSource(kind=RosterSourceKind.GRAPH, path=path)


def _corpus(tmp_path: Path, body: str, name: str = "corpus.ttl") -> RosterSource:
    """Write ``body`` under the NARAD prefixes and return it as a graph source."""
    path = tmp_path / name
    path.write_text(_PREAMBLE + body, encoding="utf-8")
    return RosterSource(kind=RosterSourceKind.GRAPH, path=path)


def _binding(name: str, address: str, predicate: str | None, binding_id: str | None = None) -> str:
    """Render one ``ChannelBinding`` with the given direction predicate.

    ``binding_id`` is the corpus's ``narad_p:bindingId`` literal, whose field
    token is what the device grouping pairs setpoints with readbacks on.
    """
    direction = f" ;\n    narad_p:{predicate} narad_sem:{name}_signal" if predicate else ""
    identity = f' ;\n    narad_p:bindingId "{binding_id}"' if binding_id else ""
    return (
        f'<https://narad.example.org/binding/{name}> narad_p:fullPv "{address}"'
        f"{direction}{identity} .\n"
    )


def _device(name: str, *bindings: str) -> str:
    """Render one device grouping the named bindings under ``narad_p:hasBinding``."""
    objects = ",\n        ".join(f"<https://narad.example.org/binding/{b}>" for b in bindings)
    return f"<https://narad.example.org/device/{name}> narad_p:hasBinding {objects} .\n"


def _magnet(
    device: str = "bend",
    *,
    setpoint: str = "SR01C___B______AC00",
    monitor: str = "SR01C___B______AM00",
    setpoint_predicate: str | None = "writesSignal",
    monitor_predicate: str | None = "readsSignal",
    stem: str = "",
    grouped: bool = True,
) -> str:
    """An ALS-shaped device: a ``<stem>Setpoint`` and a ``<stem>Monitor`` binding.

    The addresses are the facility's own (no ``:SP``/``:RB`` grammar to fall
    back on), so a readback here can only have come from the grouping.
    """
    sp, mon = f"{device}_sp", f"{device}_mon"
    body = _binding(
        sp, setpoint, setpoint_predicate, f"narad:binding:als:SR:BEND:0:{stem}Setpoint:val"
    ) + _binding(mon, monitor, monitor_predicate, f"narad:binding:als:SR:BEND:0:{stem}Monitor:val")
    if grouped:
        body += _device(device, sp, mon)
    return body


def _readbacks(source: RosterSource) -> dict[str, str | None]:
    """``address -> readback`` for every record the source yields."""
    return {record.address: record.readback for record in read_graph_roster(source).records}


class TestTheShippedDemoCorpus:
    def test_enumerates_every_channel_the_corpus_declares(self, demo_source) -> None:
        result = read_graph_roster(demo_source)

        assert result.absence is None
        assert len(result.records) == DEMO_CHANNELS

    def test_directions_come_from_the_corpus_not_from_address_grammar(self, demo_source) -> None:
        result = read_graph_roster(demo_source)

        assert len(result.write_records) == DEMO_WRITES
        assert len(result.read_records) == DEMO_READS
        assert DEMO_WRITES + DEMO_READS == DEMO_CHANNELS

    def test_names_the_corpus_it_read_on_the_result_and_on_every_record(self, demo_source) -> None:
        result = read_graph_roster(demo_source)

        assert result.source is demo_source
        assert {record.source for record in result.records} == {demo_source}

    def test_addresses_are_unique_and_sorted(self, demo_source) -> None:
        addresses = read_graph_roster(demo_source).addresses

        assert list(addresses) == sorted(addresses)
        assert len(set(addresses)) == len(addresses)

    def test_a_corpus_stating_no_pairs_yields_unpaired_records(self, demo_source) -> None:
        """The demo corpus groups bindings under devices, but its field names
        (``GOLDEN_X``, ``POSITION_X``, ...) are not the ``Setpoint``/``Monitor``
        vocabulary, so it states no pair and every readback is left to the
        address-grammar pass."""
        result = read_graph_roster(demo_source)

        assert all(record.readback is None for record in result.records)


class TestOneRecordPerAddress:
    """The roster is a namespace: bindings sharing a ``fullPv`` are one channel."""

    def test_two_bindings_on_one_address_are_one_record(self, tmp_path) -> None:
        # A delay generator's channel, bound once per device it serves.
        body = _binding("evr_a", "B0215:EVR1-DlyGen:3:Delay-SP", "writesSignal") + _binding(
            "evr_b", "B0215:EVR1-DlyGen:3:Delay-SP", "writesSignal"
        )
        source = _corpus(tmp_path, body)

        result = read_graph_roster(source)

        assert result.addresses == ("B0215:EVR1-DlyGen:3:Delay-SP",)
        assert result.records[0].direction == "write"

    def test_bindings_disagreeing_on_direction_leave_an_honest_unknown(self, tmp_path) -> None:
        body = _binding("sp", "SR04C___BSC_P__AC01", "writesSignal") + _binding(
            "rb", "SR04C___BSC_P__AC01", "readsSignal"
        )
        source = _corpus(tmp_path, body)

        (record,) = read_graph_roster(source).records

        assert record.direction is None
        assert record.readback is None

    def test_a_directionless_binding_abstains(self, tmp_path) -> None:
        body = _binding("sp", "SR01C___B______AC00", "writesSignal") + _binding(
            "bare", "SR01C___B______AC00", None
        )
        source = _corpus(tmp_path, body)

        (record,) = read_graph_roster(source).records

        assert record.direction == "write"

    def test_a_stated_readback_survives_the_collapse(self, tmp_path) -> None:
        # The same setpoint address bound a second time, off any device.
        body = _magnet() + _binding("again", "SR01C___B______AC00", "writesSignal")
        source = _corpus(tmp_path, body)

        assert _readbacks(source) == {
            "SR01C___B______AC00": "SR01C___B______AM00",
            "SR01C___B______AM00": None,
        }

    def test_the_demo_corpus_is_already_a_namespace(self, demo_source) -> None:
        addresses = read_graph_roster(demo_source).addresses

        assert len(addresses) == DEMO_CHANNELS == len(set(addresses))


class TestCorpusStatedReadbacks:
    """A device carrying ``<stem>Setpoint`` and ``<stem>Monitor`` states a pair."""

    def test_a_setpoint_and_monitor_on_one_device_pair_on_facility_addresses(
        self, tmp_path
    ) -> None:
        source = _corpus(tmp_path, _magnet())

        readbacks = _readbacks(source)

        assert readbacks == {
            "SR01C___B______AC00": "SR01C___B______AM00",
            "SR01C___B______AM00": None,
        }

    def test_the_readback_rides_the_write_record_only(self, tmp_path) -> None:
        source = _corpus(tmp_path, _magnet())

        result = read_graph_roster(source)

        (setpoint,) = result.write_records
        assert setpoint.readback == "SR01C___B______AM00"
        assert all(record.readback is None for record in result.read_records)

    def test_a_stemmed_field_pair_is_recognised(self, tmp_path) -> None:
        """``GapSetpoint``/``GapMonitor`` is an insertion device's pair."""
        source = _corpus(
            tmp_path,
            _magnet(
                "id", setpoint="SR04U___GDS1PS_AC00", monitor="SR04U___GDS1PS_AM00", stem="Gap"
            ),
        )

        assert _readbacks(source)["SR04U___GDS1PS_AC00"] == "SR04U___GDS1PS_AM00"

    def test_a_setpoint_without_a_monitor_on_its_device_stays_unpaired(self, tmp_path) -> None:
        body = (
            _binding(
                "sp",
                "SR01C___B______AC00",
                "writesSignal",
                "narad:binding:als:SR:BEND:0:Setpoint:val",
            )
            + _binding(
                "golden",
                "SR01C:BEND:Setpoint:Golden",
                "writesSignal",
                "narad:binding:als:SR:BEND:0:SetpointGolden:val",
            )
            + _device("bend", "sp", "golden")
        )
        source = _corpus(tmp_path, body)

        assert _readbacks(source)["SR01C___B______AC00"] is None

    def test_a_monitor_the_corpus_calls_settable_is_not_a_readback(self, tmp_path) -> None:
        # A corpus asserting the Monitor is driven has drifted; reading it
        # back would report a demand, not the machine.
        source = _corpus(tmp_path, _magnet(monitor_predicate="writesSignal"))

        assert _readbacks(source)["SR01C___B______AC00"] is None

    def test_a_directionless_monitor_is_not_a_readback(self, tmp_path) -> None:
        source = _corpus(tmp_path, _magnet(monitor_predicate=None))

        assert _readbacks(source)["SR01C___B______AC00"] is None

    def test_a_setpoint_claiming_both_directions_states_no_pair(self, tmp_path) -> None:
        body = (
            (
                "<https://narad.example.org/binding/both> "
                'narad_p:fullPv "SR01C___B______AC00" ;\n'
                "    narad_p:writesSignal narad_sem:bend_sp ;\n"
                "    narad_p:readsSignal narad_sem:bend_sp_rb ;\n"
                '    narad_p:bindingId "narad:binding:als:SR:BEND:0:Setpoint:val" .\n'
            )
            + _binding(
                "mon",
                "SR01C___B______AM00",
                "readsSignal",
                "narad:binding:als:SR:BEND:0:Monitor:val",
            )
            + _device("bend", "both", "mon")
        )
        source = _corpus(tmp_path, body)

        assert _readbacks(source) == {"SR01C___B______AC00": None, "SR01C___B______AM00": None}

    def test_bindings_no_device_groups_state_no_pair(self, tmp_path) -> None:
        """Matching field names alone are not a pair: the device is the grouping."""
        source = _corpus(tmp_path, _magnet(grouped=False))

        assert _readbacks(source)["SR01C___B______AC00"] is None

    def test_bindings_on_different_devices_do_not_pair(self, tmp_path) -> None:
        body = (
            _binding(
                "sp",
                "SR01C___B______AC00",
                "writesSignal",
                "narad:binding:als:SR:BEND:0:Setpoint:val",
            )
            + _binding(
                "mon",
                "SR02C___B______AM00",
                "readsSignal",
                "narad:binding:als:SR:BEND:1:Monitor:val",
            )
            + _device("bend0", "sp")
            + _device("bend1", "mon")
        )
        source = _corpus(tmp_path, body)

        assert _readbacks(source)["SR01C___B______AC00"] is None

    def test_a_binding_without_an_id_states_no_field(self, tmp_path) -> None:
        body = (
            _binding("sp", "SR01C___B______AC00", "writesSignal")
            + _binding("mon", "SR01C___B______AM00", "readsSignal")
            + _device("bend", "sp", "mon")
        )
        source = _corpus(tmp_path, body)

        assert _readbacks(source)["SR01C___B______AC00"] is None

    def test_a_binding_id_without_the_value_slot_still_names_its_field(self, tmp_path) -> None:
        """The demo corpus spells ids without the trailing ``val`` token."""
        body = (
            _binding(
                "sp", "SR01C___B______AC00", "writesSignal", "narad:binding:demo:SR:B:Setpoint"
            )
            + _binding(
                "mon", "SR01C___B______AM00", "readsSignal", "narad:binding:demo:SR:B:Monitor"
            )
            + _device("bend", "sp", "mon")
        )
        source = _corpus(tmp_path, body)

        assert _readbacks(source)["SR01C___B______AC00"] == "SR01C___B______AM00"

    def test_a_setpoint_and_monitor_sharing_one_address_state_no_pair(self, tmp_path) -> None:
        source = _corpus(tmp_path, _magnet(monitor="SR01C___B______AC00"))

        assert _readbacks(source) == {"SR01C___B______AC00": None}

    def test_two_devices_stating_different_readbacks_resolve_in_device_order(
        self, tmp_path
    ) -> None:
        """A corpus that says two things picks one deterministically, not by parse order."""
        body = (
            _binding(
                "sp",
                "SR01C___B______AC00",
                "writesSignal",
                "narad:binding:als:SR:BEND:0:Setpoint:val",
            )
            + _binding(
                "mon_z",
                "SR01C___B______AM99",
                "readsSignal",
                "narad:binding:als:SR:BEND:9:Monitor:val",
            )
            + _binding(
                "mon_a",
                "SR01C___B______AM00",
                "readsSignal",
                "narad:binding:als:SR:BEND:0:Monitor:val",
            )
            + _device("z_bend", "sp", "mon_z")
            + _device("a_bend", "sp", "mon_a")
        )
        source = _corpus(tmp_path, body)

        assert _readbacks(source)["SR01C___B______AC00"] == "SR01C___B______AM00"


class TestDirection:
    def test_a_writes_signal_binding_is_settable(self, tmp_path) -> None:
        source = _corpus(tmp_path, _binding("sp", "SR:MAG:HCM:01:CURRENT:SP", "writesSignal"))

        (record,) = read_graph_roster(source).records

        assert record.direction == "write"

    def test_a_reads_signal_binding_is_readable(self, tmp_path) -> None:
        source = _corpus(tmp_path, _binding("rb", "SR:MAG:HCM:01:CURRENT:RB", "readsSignal"))

        (record,) = read_graph_roster(source).records

        assert record.direction == "read"

    def test_a_binding_claiming_neither_direction_is_an_honest_unknown(self, tmp_path) -> None:
        source = _corpus(tmp_path, _binding("bare", "SR:DIAG:BPM:01:POSITION:X", None))

        (record,) = read_graph_roster(source).records

        assert record.address == "SR:DIAG:BPM:01:POSITION:X"
        assert record.direction is None

    def test_a_binding_claiming_both_directions_is_not_called_settable(self, tmp_path) -> None:
        body = (
            "<https://narad.example.org/binding/both> "
            'narad_p:fullPv "SR:MAG:QF:01:CURRENT:SP" ;\n'
            "    narad_p:writesSignal narad_sem:qf_current_sp ;\n"
            "    narad_p:readsSignal narad_sem:qf_current_rb .\n"
        )
        source = _corpus(tmp_path, body)

        result = read_graph_roster(source)

        assert result.records[0].direction is None
        assert result.write_records == ()


class TestMembership:
    def test_a_corpus_with_no_bindings_is_an_absence_not_an_empty_facility(self, tmp_path) -> None:
        """An unseeded corpus is a staging gap, and every consumer must hear so.

        Served as an empty roster it would tell an operator the facility has no
        channels, and would mark every real channel invalid on the way.
        """
        source = _corpus(tmp_path, "")

        result = read_graph_roster(source)

        assert result.records == ()
        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.EMPTY_SOURCE
        assert result.absence.path == source.path

    def test_a_blank_address_is_not_a_channel(self, tmp_path) -> None:
        body = _binding("blank", "", "readsSignal") + _binding(
            "real", "SR:DIAG:BPM:01:POSITION:X", "readsSignal"
        )
        source = _corpus(tmp_path, body)

        assert read_graph_roster(source).addresses == ("SR:DIAG:BPM:01:POSITION:X",)

    def test_an_iri_object_is_not_an_address_but_a_tagged_literal_is(self, tmp_path) -> None:
        """``fullPv`` is a literal by schema. A binding that points it at an
        IRI instead names no channel -- stringifying the IRI would put a URL in
        the roster -- while a literal carrying a language tag is still an
        address, and contributes its lexical form without the tag."""
        body = (
            "<https://narad.example.org/binding/iri> narad_p:fullPv <urn:not-a-literal> .\n"
            '<https://narad.example.org/binding/tagged> narad_p:fullPv "SR:BPM:01:X"@en .\n'
        )
        source = _corpus(tmp_path, body)

        assert read_graph_roster(source).addresses == ("SR:BPM:01:X",)

    def test_the_turtle_format_is_forced_rather_than_guessed_from_the_extension(
        self, tmp_path
    ) -> None:
        source = _corpus(
            tmp_path,
            _binding("sp", "SR:MAG:HCM:01:CURRENT:SP", "writesSignal"),
            name="corpus.rdf",
        )

        result = read_graph_roster(source)

        assert result.addresses == ("SR:MAG:HCM:01:CURRENT:SP",)


class TestAnUnreadableCorpus:
    def test_unparseable_turtle_is_reported_as_a_corrupt_source(self, tmp_path, caplog) -> None:
        path = tmp_path / "broken.ttl"
        path.write_text("this is not turtle at all <<<", encoding="utf-8")
        source = RosterSource(kind=RosterSourceKind.GRAPH, path=path)

        with caplog.at_level("WARNING"):
            result = read_graph_roster(source)

        assert result.records == ()
        assert result.source is None
        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.CORRUPT_SOURCE
        assert result.absence.path == path
        assert str(path) in result.absence.message()
        assert str(path) in caplog.text

    def test_a_ttl_path_naming_a_directory_is_corrupt_rather_than_missing(
        self, tmp_path, caplog
    ) -> None:
        """A directory where a corpus was configured IS there -- it just cannot
        be parsed. Calling it missing would tell the build to stay browse-only
        and wait for a file that has already arrived, wearing the wrong shape.
        """
        directory = tmp_path / "corpus.ttl"
        directory.mkdir()
        source = RosterSource(kind=RosterSourceKind.GRAPH, path=directory)

        with caplog.at_level("WARNING"):
            result = read_graph_roster(source)

        assert result.records == ()
        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.CORRUPT_SOURCE
        assert str(directory) in caplog.text

    def test_a_missing_corpus_is_absent_rather_than_corrupt(self, tmp_path, caplog) -> None:
        """A corpus that is not there is a different state from one that is
        there and unreadable, and the two get opposite treatment downstream:
        the build stays browse-only on this one and refuses on the other. The
        reader says which it is, so no consumer has to re-``stat`` the file to
        find out.
        """
        missing = tmp_path / "absent.ttl"
        source = RosterSource(kind=RosterSourceKind.GRAPH, path=missing)

        with caplog.at_level("WARNING"):
            result = read_graph_roster(source)

        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.MISSING_SOURCE
        assert str(missing) in result.absence.message()
        assert str(missing) in caplog.text, "the build log is where an operator meets this"
        assert "services.graphdb.ttl_path" in result.absence.message(), (
            "the remedy is a config edit, so the key that declares the corpus is named"
        )

    def test_a_missing_corpus_names_the_configured_spelling_when_it_has_one(self, tmp_path) -> None:
        """An operator is handed the path they wrote, not the one a build
        resolved it to inside its own staging tree."""
        source = RosterSource(
            kind=RosterSourceKind.GRAPH,
            path=tmp_path / "build" / ".tmp" / "data" / "corpus.ttl",
            spelled="./data/corpus.ttl",
        )

        message = read_graph_roster(source).absence.message()

        assert "./data/corpus.ttl" in message
        assert ".tmp" not in message


class TestAnUnimportableRdflib:
    def test_degrades_to_an_absence_naming_the_reinstall(
        self, tmp_path, monkeypatch, caplog
    ) -> None:
        source = _corpus(tmp_path, _binding("sp", "SR:MAG:HCM:01:CURRENT:SP", "writesSignal"))
        monkeypatch.setitem(sys.modules, "rdflib", None)

        with caplog.at_level("WARNING"):
            result = read_graph_roster(source)

        assert result.records == ()
        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.CORRUPT_SOURCE
        assert result.absence.path == source.path
        assert "rdflib" in result.absence.message()
        assert "osprey-framework" in result.absence.message()

    def test_warns_so_the_incomplete_environment_is_visible_in_the_build_log(
        self, tmp_path, monkeypatch, caplog
    ) -> None:
        source = _corpus(tmp_path, _binding("sp", "SR:MAG:HCM:01:CURRENT:SP", "writesSignal"))
        monkeypatch.setitem(sys.modules, "rdflib", None)

        with caplog.at_level("WARNING"):
            read_graph_roster(source)

        warnings = [record.message for record in caplog.records if record.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "rdflib" in warnings[0]
        assert str(source.path) in warnings[0]
