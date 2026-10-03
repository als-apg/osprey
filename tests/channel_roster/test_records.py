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

_FACILITY = RosterSource(kind=RosterSourceKind.FACILITY, path=Path("/data/facility.json"))


def _record(address: str, direction: str | None = None, **kwargs: object) -> ChannelRecord:
    """Build a record on the facility source, for tests that do not care which."""
    return ChannelRecord(address=address, source=_FACILITY, direction=direction, **kwargs)


class TestChannelRecord:
    def test_carries_address_direction_readback_and_provenance(self) -> None:
        record = ChannelRecord(
            address="SR:MAG:HCM:01:CURRENT:SP",
            source=_FACILITY,
            direction="write",
            readback="SR:MAG:HCM:01:CURRENT:RB",
        )
        assert record.address == "SR:MAG:HCM:01:CURRENT:SP"
        assert record.direction == "write"
        assert record.readback == "SR:MAG:HCM:01:CURRENT:RB"
        assert record.source is _FACILITY

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
    def test_the_one_kind_is_the_facility_file(self) -> None:
        assert {kind.value for kind in RosterSourceKind} == {"facility"}

    def test_describe_names_the_kind_and_the_resolved_path(self) -> None:
        assert _FACILITY.describe() == "this project's facility file (/data/facility.json)"

    def test_describe_prefers_the_name_the_file_is_shown_under(self) -> None:
        """The resolved path is where the bytes are; the spelled name is what
        the build fact and the web body call the file.

        A build resolves the facility file into the render it is writing, so a
        fact naming the resolved path hands the reader a ``build/.tmp/...``
        file that exists only for the duration of the render. Display follows
        the spelling; I/O and the memo key keep following ``path``.
        """
        source = RosterSource(
            kind=RosterSourceKind.FACILITY,
            path=Path("/repo/build/.tmp/proj/facility.json"),
            spelled="facility.json",
        )

        assert source.describe() == "this project's facility file (facility.json)"
        assert source.for_display() == "facility.json"
        assert source.path == Path("/repo/build/.tmp/proj/facility.json")

    def test_every_kind_has_a_label(self) -> None:
        for kind in RosterSourceKind:
            assert RosterSource(kind=kind, path=Path("/x")).describe()


class TestRosterAbsence:
    def test_an_absence_is_named_by_the_spelling_of_its_source(self) -> None:
        """Same display rule as the source it is about: the message names the
        file as the build fact calls it, not the render path it resolved to."""
        absence = RosterAbsence(
            reason=RosterAbsenceReason.FACILITY_NOT_BUILT,
            path=Path("/repo/build/.tmp/proj/facility.json"),
            spelled="facility.json",
        )

        message = absence.message()

        assert "facility.json" in message
        assert ".tmp" not in message, "the resolved staging path is not a thing to retype"
        assert absence.path == Path("/repo/build/.tmp/proj/facility.json")

    def test_every_reason_has_phrasing(self) -> None:
        # The table is what keeps build facts and 503 bodies saying the same
        # thing; a reason added without phrasing must fail here, not render blank.
        assert set(ABSENCE_TEMPLATES) == set(RosterAbsenceReason)

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

    @pytest.mark.parametrize(
        ("reason", "kwargs", "missing"),
        [
            (RosterAbsenceReason.FACILITY_NOT_BUILT, {}, "path"),
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
            source=_FACILITY,
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

    def test_records_are_normalised_to_a_tuple(self) -> None:
        result = RosterResult(records=[_record("A", "read")], source=_FACILITY)
        assert result.records == (_record("A", "read"),)

    def test_an_absent_roster_carries_its_reason_and_no_source(self) -> None:
        result = RosterResult(
            absence=RosterAbsence(
                reason=RosterAbsenceReason.FACILITY_NOT_BUILT, path=Path("/data/facility.json")
            )
        )
        assert result.records == ()
        assert result.source is None
        assert result.absence is not None
        assert result.absence.reason is RosterAbsenceReason.FACILITY_NOT_BUILT

    def test_a_sourced_result_with_no_records_is_a_legal_shape(self) -> None:
        """The type permits it; no reader builds one."""
        result = RosterResult(source=_FACILITY)
        assert result.records == ()
        assert result.absence is None

    def test_a_result_that_says_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must say why"):
            RosterResult()

    def test_records_without_a_source_are_refused(self) -> None:
        with pytest.raises(ValueError, match="must name the source"):
            RosterResult(
                records=(_record("A", "read"),),
                absence=RosterAbsence(
                    reason=RosterAbsenceReason.FACILITY_NOT_BUILT,
                    path=Path("/data/facility.json"),
                ),
            )

    def test_is_frozen(self) -> None:
        result = RosterResult(source=_FACILITY)
        with pytest.raises(FrozenInstanceError):
            result.source = RosterSource(  # type: ignore[misc]
                kind=RosterSourceKind.FACILITY, path=Path("/data/facility.json")
            )


class TestNoIO:
    def test_module_imports_nothing_that_touches_a_source(self) -> None:
        # These types are declarative data: readers do the I/O, not this module.
        source = Path(records_module.__file__ or "").read_text(encoding="utf-8")
        for forbidden in ("open(", "read_text", "rdflib", "json.load", "requests"):
            assert forbidden not in source
