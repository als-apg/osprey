"""
Unit tests for the shared write-result type and the protocol-agnostic core.

The outcome vocabulary itself is pinned in ``test_write_outcome.py`` and the
comparison rule in ``test_values_match.py``; what is left here is the shape of
:class:`ChannelWriteResult` — its defaults, its declared types and its survival
through a model round trip — plus the guard that keeps the shared core free
of any control-system client library.
"""

import ast
from datetime import UTC, datetime
from pathlib import Path
from typing import get_type_hints

import pytest

from osprey_connectors.control_system import _base as base_module
from osprey_connectors.control_system._base import (
    Alarm,
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    WriteOutcome,
)
from osprey_connectors.refusals import RefusalReason

_TS = datetime(2026, 11, 16, 8, 0, tzinfo=UTC)


class TestChannelWriteResultShape:
    """The result type every connector returns, pinned at the model."""

    def test_optional_fields_default_to_none_on_the_model(self):
        """Only address, value and outcome are required; every other field defaults to None.

        A connector states what became of the write and nothing more, so a
        ``refused`` result carries no observed reading and a ``confirmed`` one
        carries no message.
        """
        model_fields = ChannelWriteResult.model_fields
        required = {name for name, spec in model_fields.items() if spec.is_required()}
        assert required == {"address", "value_written", "outcome"}

        defaults = {name: spec.default for name, spec in model_fields.items()}

        assert defaults["refusal_reason"] is None
        assert defaults["error_message"] is None
        assert defaults["observed"] is None
        assert defaults["notes"] is None
        # The re-read's value and alarm live on the reading, not beside it.
        assert {"observed_value", "alarm_status", "alarm_severity"}.isdisjoint(model_fields)

    def test_field_annotations_are_pinned(self):
        """The declared types are the contract.

        Pinned with ``get_type_hints`` rather than ``isinstance`` on a
        constructed instance: asserting the type of a literal the test itself
        passed in only re-checks the literal, and would keep passing if a field
        were widened. ``field.type`` is no good either — it degrades to a raw
        string the moment the module adopts postponed annotations.
        """
        hints = get_type_hints(ChannelWriteResult)

        assert hints["outcome"] == WriteOutcome
        assert hints["refusal_reason"] == (RefusalReason | None)
        assert hints["error_message"] == (str | None)
        assert hints["notes"] == (str | None)
        # The confirming re-read itself: value, stamp, alarm and label.
        assert hints["observed"] == (ChannelValue | None)

    def test_healthy_severity_zero_is_distinct_from_not_reported(self):
        """Severity 0 is a reported fact; ``None`` is the absence of one."""
        reported = ChannelWriteResult.confirmed(
            "TEST:CHANNEL", 100.0, _reading(100.0, alarm=Alarm(status="NO_ALARM", severity=0))
        )
        absent = ChannelWriteResult.confirmed("TEST:CHANNEL", 100.0, _reading(100.0))

        assert reported.observed is not None and reported.observed.alarm is not None
        assert reported.observed.alarm.severity == 0
        assert absent.observed is not None
        assert absent.observed.alarm is None

    def test_full_field_set_round_trips_through_model_dump(self):
        """Every field survives a model -> dict -> model round trip."""
        original = ChannelWriteResult.mismatch(
            "TEST:CHANNEL",
            100.0,
            _reading(95.0, alarm=Alarm(status="HIHI", severity=2), enum_label=None),
        )

        restored = ChannelWriteResult(**original.model_dump())

        assert restored == original
        assert restored.model_dump() == original.model_dump()
        assert restored.outcome is WriteOutcome.MISMATCH
        assert restored.observed == original.observed

    def test_observed_number_narrows_only_what_is_a_number(self):
        """A reading with no numeric form is reported as such, never coerced."""
        numeric = ChannelWriteResult.confirmed("TEST:CHANNEL", 100.0, _reading(100.0))
        textual = ChannelWriteResult.confirmed("TEST:CHANNEL", "Open", _reading("Open"))
        unread = ChannelWriteResult.unrequested("TEST:CHANNEL", 100.0)

        assert numeric.observed_number == 100.0
        assert textual.observed_number is None
        assert unread.observed_number is None

    @pytest.mark.parametrize("outcome", [WriteOutcome.CONFIRMED, WriteOutcome.MISMATCH])
    def test_a_re_read_rides_on_the_outcomes_it_decided(self, outcome):
        result = ChannelWriteResult(
            address="TEST:CHANNEL", value_written=1.0, outcome=outcome, observed=_reading(1.0)
        )

        assert result.observed == _reading(1.0)

    @pytest.mark.parametrize(
        "outcome,extra",
        [
            (
                WriteOutcome.REFUSED,
                {"refusal_reason": RefusalReason.WRITES_DISABLED, "error_message": "off"},
            ),
            (WriteOutcome.FAILED, {"error_message": "not taken"}),
            (WriteOutcome.UNCONFIRMED, {"error_message": "no acknowledgement"}),
            (WriteOutcome.UNREQUESTED, {}),
        ],
    )
    def test_a_re_read_on_any_other_outcome_is_refused_at_construction(self, outcome, extra):
        """``observed`` is set only on ``confirmed`` or ``mismatch``, which a re-read decided."""
        with pytest.raises(ValueError, match="observed must be None"):
            ChannelWriteResult(
                address="TEST:CHANNEL",
                value_written=1.0,
                outcome=outcome,
                observed=_reading(1.0),
                **extra,
            )


def _reading(value, *, alarm=None, enum_label=None):
    return ChannelValue(
        address="TEST:CHANNEL",
        value=value,
        sampled_at=_TS,
        alarm=alarm,
        enum_label=enum_label,
    )


class TestReadingRecords:
    """The alarm and the label ride on the reading; the metadata stays static."""

    def test_alarm_fields_are_names_and_ints_reported_or_not(self):
        # get_type_hints, not ``field.type``: the latter degrades to a raw
        # string the moment the module adopts postponed annotations, and a
        # string never equals ``str | None``, so the pin would pass vacuously.
        hints = get_type_hints(Alarm)
        assert hints == {"status": str | None, "severity": int | None}
        assert Alarm() == Alarm(status=None, severity=None)

    def test_a_reading_names_its_channel_and_carries_its_alarm_and_label(self):
        hints = get_type_hints(ChannelValue)

        assert hints["address"] is str
        assert hints["alarm"] == (Alarm | None)
        assert hints["enum_label"] == (str | None)
        assert ChannelValue.model_fields["address"].is_required()

    def test_the_metadata_holds_no_per_reading_state(self):
        assert {"alarm_status", "alarm_severity", "enum_label"}.isdisjoint(
            ChannelMetadata.model_fields
        )


#: Protocol client libraries the shared core must never reach for.
_PROTOCOL_PACKAGES = {"epics", "pyepics", "p4p", "aioca", "caproto"}


class TestSharedCoreIsProtocolAgnostic:
    """The shared connector core must not import EPICS constants (FR4)."""

    def test_base_module_imports_no_epics_symbols(self):
        """Checked on the import statements, so prose may still say "EPICS"."""
        tree = ast.parse(Path(base_module.__file__).read_text())

        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Relative or absolute; a relative one carries no package
                    # prefix, which is why segments are matched individually.
                    imported.add(node.module)
                else:
                    # ``from . import epics_connector`` puts the module in names.
                    imported.update(alias.name for alias in node.names)

        # A sibling connector module is just as much of a protocol leak as the
        # client library itself, so both shapes are flagged, on any segment.
        offenders = sorted(
            name
            for name in imported
            if any(
                part in _PROTOCOL_PACKAGES or part.endswith("_connector")
                for part in name.split(".")
            )
        )
        assert not offenders, (
            f"shared control-system base must stay protocol-agnostic, but imports {offenders}; "
            "alarm-code mapping belongs in epics_connector.py"
        )
