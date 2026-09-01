"""Unit tests for the channel-roster record and result types.

Covers ``osprey.channel_roster.records`` -- the declarative types every roster
reader produces and every roster consumer reads: the per-channel record, the
source provenance, and the absence reasons that carry the build's honesty as
data rather than as per-consumer prose.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import osprey.channel_roster.records as records_module
from osprey.channel_roster import (
    ABSENCE_TEMPLATES,
    ChannelRecord,
    RosterAbsence,
    RosterAbsenceReason,
    RosterResult,
    RosterSource,
    RosterSourceKind,
)
from osprey.channel_roster.records import _template_fields

_GRAPH = RosterSource(kind=RosterSourceKind.GRAPH, path=Path("/data/demo_machine.ttl"))
_DB = RosterSource(kind=RosterSourceKind.DATABASE, path=Path("/data/hierarchical.json"))


def _record(address: str, direction: str | None = None, **kwargs: object) -> ChannelRecord:
    """Build a record on the graph source, for tests that do not care which."""
    return ChannelRecord(address=address, source=_GRAPH, direction=direction, **kwargs)


class TestChannelRecord:
    def test_carries_address_direction_readback_and_provenance(self) -> None:
        record = ChannelRecord(
            address="SR:MAG:HCM:01:CURRENT:SP",
            source=_GRAPH,
            direction="write",
            readback="SR:MAG:HCM:01:CURRENT:RB",
        )
        assert record.address == "SR:MAG:HCM:01:CURRENT:SP"
        assert record.direction == "write"
        assert record.readback == "SR:MAG:HCM:01:CURRENT:RB"
        assert record.source is _GRAPH

    def test_direction_and_readback_default_to_unknown_and_unpaired(self) -> None:
        record = _record("SR:DIAG:BPM:01:POSITION:X")
        assert record.direction is None
        assert record.readback is None

    def test_is_frozen(self) -> None:
        record = _record("SR:MAG:QF:01:CURRENT:SP", "write")
        with pytest.raises(FrozenInstanceError):
            record.direction = "read"  # type: ignore[misc]

    def test_records_are_hashable_and_value_compared(self) -> None:
        # Consumers set-compare roster membership; a record has to behave as a value.
        assert _record("SR:MAG:QF:01:CURRENT:RB", "read") == _record(
            "SR:MAG:QF:01:CURRENT:RB", "read"
        )
        assert len({_record("A", "read"), _record("A", "read")}) == 1

    def test_with_readback_returns_a_paired_copy(self) -> None:
        setpoint = _record("SR:MAG:HCM:01:CURRENT:SP", "write")
        paired = setpoint.with_readback("SR:MAG:HCM:01:CURRENT:RB")
        assert paired.readback == "SR:MAG:HCM:01:CURRENT:RB"
        assert paired.address == setpoint.address
        assert paired.source is setpoint.source
        assert setpoint.readback is None

    def test_empty_address_is_refused(self) -> None:
        with pytest.raises(ValueError, match="needs an address"):
            _record("")

    def test_unknown_direction_is_refused(self) -> None:
        # A typo here would silently unsettle every channel a consumer compares
        # against "write", which is the class of defect this feature removes.
        with pytest.raises(ValueError, match="Unknown channel direction"):
            _record("SR:MAG:HCM:01:CURRENT:SP", "settable")

    @pytest.mark.parametrize("direction", [None, "read"])
    def test_readback_without_a_write_direction_is_refused(self, direction: str | None) -> None:
        with pytest.raises(ValueError, match="readback pairs a setpoint"):
            _record("SR:MAG:HCM:01:CURRENT:SP", direction, readback="SR:MAG:HCM:01:CURRENT:RB")


class TestRosterSource:
    def test_kinds_are_the_two_authoritative_sources(self) -> None:
        assert {kind.value for kind in RosterSourceKind} == {"graph", "database"}

    def test_describe_names_the_kind_and_the_resolved_path(self) -> None:
        assert _GRAPH.describe() == "the facility knowledge graph (/data/demo_machine.ttl)"
        assert _DB.describe() == "the channel finder database (/data/hierarchical.json)"

    def test_describe_prefers_the_spelling_an_operator_configured(self) -> None:
        """The resolved path is where the bytes are; the configured spelling is
        what an operator can retype and edit.

        A build resolves a relative corpus into its own staging tree, so a fact
        naming the resolved path hands the reader a ``build/.tmp/...`` file that
        exists only for the duration of the render. Display follows the
        spelling; I/O and the memo key keep following ``path``.
        """
        source = RosterSource(
            kind=RosterSourceKind.GRAPH,
            path=Path("/repo/build/.tmp/proj/data/demo_machine.ttl"),
            spelled="./data/demo_machine.ttl",
        )

        assert source.describe() == "the facility knowledge graph (./data/demo_machine.ttl)"
        assert source.for_display() == "./data/demo_machine.ttl"
        assert source.path == Path("/repo/build/.tmp/proj/data/demo_machine.ttl")

    def test_every_kind_has_a_label(self) -> None:
        for kind in RosterSourceKind:
            assert RosterSource(kind=kind, path=Path("/x")).describe()


class TestRosterAbsence:
    def test_a_missing_source_is_named_by_the_spelling_that_declared_it(self) -> None:
        """Same display rule as the source it is about: the message names the
        configured path and the key that carries it, because both are things an
        operator can act on."""
        absence = RosterAbsence(
            reason=RosterAbsenceReason.MISSING_SOURCE,
            path=Path("/repo/build/.tmp/proj/data/demo_machine.ttl"),
            spelled="./data/demo_machine.ttl",
            config_keys=("services.graphdb.ttl_path",),
        )

        message = absence.message()

        assert "./data/demo_machine.ttl" in message
        assert "services.graphdb.ttl_path" in message
        assert ".tmp" not in message, "the resolved staging path is not a thing to retype"
        assert absence.path == Path("/repo/build/.tmp/proj/data/demo_machine.ttl")

    def test_missing_and_corrupt_are_distinct_reasons(self) -> None:
        """The pair consumers branch on: absent is fail-soft, unreadable is
        fail-closed. One reason for both would force every consumer to re-probe
        the file to decide which rule applies."""
        assert RosterAbsenceReason.MISSING_SOURCE is not RosterAbsenceReason.CORRUPT_SOURCE
        assert {RosterAbsenceReason.MISSING_SOURCE, RosterAbsenceReason.CORRUPT_SOURCE} <= set(
            ABSENCE_TEMPLATES
        )

    def test_every_reason_has_phrasing(self) -> None:
        # The table is what keeps build facts and 503 bodies saying the same
        # thing; a reason added without phrasing must fail here, not render blank.
        assert set(ABSENCE_TEMPLATES) == set(RosterAbsenceReason)

    def test_no_source_needs_no_subject(self) -> None:
        absence = RosterAbsence(reason=RosterAbsenceReason.NO_SOURCE)
        assert absence.message() == (
            "No channel roster source is configured, so the set of channels this "
            "facility has is unknown."
        )

    def test_graph_no_ttl_names_both_config_keys(self) -> None:
        absence = RosterAbsence(
            reason=RosterAbsenceReason.GRAPH_NO_TTL,
            config_keys=("services.graphdb.ttl_path", "services.graphdb.uri"),
        )
        message = absence.message()
        assert "services.graphdb.ttl_path and services.graphdb.uri" in message
        assert "unknown" in message

    def test_direction_underivable_names_the_database_path(self) -> None:
        absence = RosterAbsence(
            reason=RosterAbsenceReason.DIRECTION_UNDERIVABLE, path=Path("/data/flat.json")
        )
        message = absence.message()
        assert "/data/flat.json" in message
        assert "settable" in message

    def test_corrupt_source_names_the_path_and_the_failure(self) -> None:
        absence = RosterAbsence(
            reason=RosterAbsenceReason.CORRUPT_SOURCE,
            path=Path("/data/demo_machine.ttl"),
            detail="bad syntax at line 12",
        )
        assert absence.message() == (
            "The channel roster source at /data/demo_machine.ttl could not be read: "
            "bad syntax at line 12."
        )

    def test_config_keys_are_normalised_to_a_tuple(self) -> None:
        absence = RosterAbsence(
            reason=RosterAbsenceReason.GRAPH_NO_TTL,
            config_keys=["services.graphdb.ttl_path"],
        )
        assert absence.config_keys == ("services.graphdb.ttl_path",)
        assert hash(absence)

    def test_single_config_key_renders_without_a_conjunction(self) -> None:
        absence = RosterAbsence(
            reason=RosterAbsenceReason.GRAPH_NO_TTL,
            config_keys=("services.graphdb.ttl_path",),
        )
        assert "declared by services.graphdb.ttl_path." in absence.message()

    @pytest.mark.parametrize(
        ("reason", "kwargs", "missing"),
        [
            (RosterAbsenceReason.GRAPH_NO_TTL, {}, "config_keys"),
            (RosterAbsenceReason.DIRECTION_UNDERIVABLE, {}, "path"),
            (RosterAbsenceReason.CORRUPT_SOURCE, {"detail": "boom"}, "path"),
            (RosterAbsenceReason.CORRUPT_SOURCE, {"path": Path("/x")}, "detail"),
        ],
    )
    def test_an_absence_missing_its_subject_is_refused(
        self, reason: RosterAbsenceReason, kwargs: dict[str, object], missing: str
    ) -> None:
        # Rejected at construction rather than rendered as "at None" downstream.
        with pytest.raises(ValueError, match=missing):
            RosterAbsence(reason=reason, **kwargs)

    def test_no_consumer_needs_a_switch_to_render_a_reason(self) -> None:
        # Every reason renders through the same call, with only the subjects
        # its own phrasing names supplied.
        subjects: dict[str, object] = {
            "path": Path("/data/source"),
            "config_keys": ("services.graphdb.ttl_path", "services.graphdb.uri"),
            "detail": "unreadable",
        }
        for reason in RosterAbsenceReason:
            needed = _template_fields(ABSENCE_TEMPLATES[reason])
            absence = RosterAbsence(reason=reason, **{k: subjects[k] for k in needed})
            message = absence.message()
            assert message.endswith(".")
            assert "{" not in message
            assert "None" not in message


class TestRosterResult:
    def test_splits_records_by_direction(self) -> None:
        result = RosterResult(
            records=(
                _record("SR:MAG:HCM:01:CURRENT:SP", "write"),
                _record("SR:MAG:HCM:01:CURRENT:RB", "read"),
                _record("SR:DIAG:BPM:01:POSITION:X", "read"),
            ),
            source=_GRAPH,
        )
        assert result.addresses == (
            "SR:MAG:HCM:01:CURRENT:SP",
            "SR:MAG:HCM:01:CURRENT:RB",
            "SR:DIAG:BPM:01:POSITION:X",
        )
        assert [r.address for r in result.write_records] == ["SR:MAG:HCM:01:CURRENT:SP"]
        assert [r.address for r in result.read_records] == [
            "SR:MAG:HCM:01:CURRENT:RB",
            "SR:DIAG:BPM:01:POSITION:X",
        ]

    def test_direction_unknown_records_are_neither_settable_nor_readable(self) -> None:
        result = RosterResult(
            records=(_record("FLAT_CHANNEL_1"), _record("FLAT_CHANNEL_2")),
            source=_DB,
            absence=RosterAbsence(reason=RosterAbsenceReason.DIRECTION_UNDERIVABLE, path=_DB.path),
        )
        assert result.write_records == ()
        assert result.read_records == ()
        assert result.addresses == ("FLAT_CHANNEL_1", "FLAT_CHANNEL_2")
        assert result.absence is not None
        assert str(_DB.path) in result.absence.message()

    def test_records_are_normalised_to_a_tuple(self) -> None:
        result = RosterResult(records=[_record("A", "read")], source=_GRAPH)
        assert result.records == (_record("A", "read"),)

    def test_an_absent_roster_carries_its_reason_and_no_source(self) -> None:
        result = RosterResult(absence=RosterAbsence(reason=RosterAbsenceReason.NO_SOURCE))
        assert result.records == ()
        assert result.source is None
        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.NO_SOURCE

    def test_a_sourced_result_with_no_records_is_a_legal_shape(self) -> None:
        """The type permits it; no reader builds one.

        A source that enumerates nothing comes back as an
        :attr:`RosterAbsenceReason.EMPTY_SOURCE` absence instead (see the two
        readers), so this shape is legal here and unreachable in practice.
        """
        result = RosterResult(source=_GRAPH)
        assert result.records == ()
        assert result.absence is None

    def test_a_result_that_says_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must say why"):
            RosterResult()

    def test_records_without_a_source_are_refused(self) -> None:
        with pytest.raises(ValueError, match="must name the source"):
            RosterResult(
                records=(_record("A", "read"),),
                absence=RosterAbsence(reason=RosterAbsenceReason.NO_SOURCE),
            )

    def test_is_frozen(self) -> None:
        result = RosterResult(source=_GRAPH)
        with pytest.raises(FrozenInstanceError):
            result.source = _DB  # type: ignore[misc]


class TestNoIO:
    def test_module_imports_nothing_that_touches_a_source(self) -> None:
        # These types are declarative data: readers do the I/O, not this module.
        source = Path(records_module.__file__ or "").read_text(encoding="utf-8")
        for forbidden in ("open(", "read_text", "rdflib", "json.load", "requests"):
            assert forbidden not in source
