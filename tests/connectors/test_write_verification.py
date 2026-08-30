"""
Unit tests for the shared write-result type and the protocol-agnostic core.

The outcome vocabulary itself is pinned in ``test_write_outcome.py`` and the
comparison rule in ``test_values_match.py``; what is left here is the shape of
:class:`ChannelWriteResult` — its defaults, its declared types and its survival
through a dataclass round trip — plus the guard that keeps the shared core free
of any control-system client library.
"""

import ast
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, get_type_hints

from osprey.connectors.control_system import base as base_module
from osprey.connectors.control_system.base import (
    ChannelMetadata,
    ChannelWriteResult,
    WriteOutcome,
)


class TestChannelWriteResultShape:
    """The result type every connector returns, pinned at the dataclass."""

    def test_only_address_value_and_outcome_are_required(self):
        """A connector states what became of the write and nothing more.

        Everything else is optional, so a ``refused`` result carries no observed
        value and a ``confirmed`` one carries no message.
        """
        result = ChannelWriteResult(
            channel_address="TEST:CHANNEL",
            value_written=100.0,
            outcome=WriteOutcome.CONFIRMED,
        )

        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.refusal_reason is None
        assert result.error_message is None
        assert result.observed_value is None
        assert result.alarm_status is None
        assert result.alarm_severity is None
        assert result.notes is None

    def test_optional_fields_default_to_none_on_the_dataclass(self):
        """Pinned on the dataclass, not just via one constructor call."""
        defaults = {f.name: f.default for f in fields(ChannelWriteResult)}

        assert defaults["refusal_reason"] is None
        assert defaults["error_message"] is None
        assert defaults["observed_value"] is None
        assert defaults["alarm_status"] is None
        assert defaults["alarm_severity"] is None
        assert defaults["notes"] is None

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
        assert hints["refusal_reason"] == (str | None)
        assert hints["error_message"] == (str | None)
        assert hints["alarm_status"] == (str | None)
        assert hints["alarm_severity"] == (int | None)
        assert hints["notes"] == (str | None)
        # Whatever the channel holds — a number, a string, an enum label, a
        # waveform — reaches consumers untouched.
        assert hints["observed_value"] == Any

    def test_healthy_severity_zero_is_distinct_from_not_reported(self):
        """Severity 0 is a reported fact; ``None`` is the absence of one."""
        reported = ChannelWriteResult(
            channel_address="TEST:CHANNEL",
            value_written=100.0,
            outcome=WriteOutcome.CONFIRMED,
            alarm_severity=0,
        )
        absent = ChannelWriteResult(
            channel_address="TEST:CHANNEL",
            value_written=100.0,
            outcome=WriteOutcome.CONFIRMED,
        )

        assert reported.alarm_severity == 0
        assert reported.alarm_severity is not None
        assert absent.alarm_severity is None

    def test_full_field_set_round_trips_through_asdict(self):
        """Every field survives a dataclass -> dict -> dataclass round trip."""
        original = ChannelWriteResult(
            channel_address="TEST:CHANNEL",
            value_written=100.0,
            outcome=WriteOutcome.MISMATCH,
            refusal_reason=None,
            error_message=None,
            observed_value=95.0,
            alarm_status="HIHI",
            alarm_severity=2,
            notes="observed 95.0, sent 100.0",
        )

        restored = ChannelWriteResult(**asdict(original))

        assert restored == original
        assert asdict(restored) == asdict(original)
        assert restored.outcome is WriteOutcome.MISMATCH

    def test_observed_number_narrows_only_what_is_a_number(self):
        """A reading with no numeric form is reported as such, never coerced."""
        numeric = ChannelWriteResult(
            channel_address="TEST:CHANNEL",
            value_written=100.0,
            outcome=WriteOutcome.CONFIRMED,
            observed_value=100.0,
        )
        textual = ChannelWriteResult(
            channel_address="TEST:CHANNEL",
            value_written="Open",
            outcome=WriteOutcome.CONFIRMED,
            observed_value="Open",
        )

        assert numeric.observed_number == 100.0
        assert textual.observed_number is None


class TestChannelMetadataAlarmStatusNotWidened:
    """``ChannelMetadata.alarm_status`` stays ``str | None`` — no type widening.

    Alarm *names* are produced at the protocol boundary (the EPICS connector
    maps raw codes), so the shared metadata type does not need to admit ints.
    """

    def test_alarm_status_annotation_is_str_or_none(self):
        # get_type_hints, not ``field.type``: the latter degrades to a raw
        # string the moment the module adopts postponed annotations, and a
        # string never equals ``str | None``, so the pin would pass vacuously.
        assert get_type_hints(ChannelMetadata)["alarm_status"] == (str | None)

    def test_alarm_status_defaults_to_none(self):
        assert ChannelMetadata().alarm_status is None

    def test_alarm_status_holds_a_name(self):
        assert ChannelMetadata(alarm_status="NO_ALARM").alarm_status == "NO_ALARM"


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
