"""
Unit tests for channel write verification system.

Tests the three-tier verification approach:
- none: Fast write, no verification
- callback: Control system confirms processing
- readback: Full verification with readback comparison
"""

import ast
from dataclasses import asdict, fields
from pathlib import Path
from typing import get_type_hints
from unittest.mock import patch

import pytest

from osprey.connectors.control_system import base as base_module
from osprey.connectors.control_system.base import (
    ChannelMetadata,
    ChannelWriteResult,
    WriteVerification,
)
from osprey.connectors.control_system.mock_connector import MockConnector


def _config_with_writes_enabled(key, default=None):
    if key == "control_system.writes_enabled":
        return True
    return default


class TestMockConnectorVerification:
    """Test verification levels in MockConnector."""

    @pytest.fixture
    async def connector(self):
        """Create and connect MockConnector."""
        conn = MockConnector()
        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_config_with_writes_enabled,
        ):
            await conn.connect(
                {
                    "response_delay_ms": 1,  # Fast for testing
                    "noise_level": 0.001,  # Low noise
                }
            )
            yield conn
            await conn.disconnect()

    @pytest.mark.asyncio
    async def test_verification_none(self, connector):
        """Test 'none' verification level - no verification performed."""
        result = await connector.write_channel("TEST:CHANNEL", 100.0, verification_level="none")

        assert isinstance(result, ChannelWriteResult)
        assert result.success is True
        assert result.channel_address == "TEST:CHANNEL"
        assert result.value_written == 100.0
        assert result.verification is not None
        assert result.verification.level == "none"
        assert result.verification.verified is False  # No verification performed
        assert result.verification.readback_value is None

    @pytest.mark.asyncio
    async def test_verification_callback(self, connector):
        """Test 'callback' verification level - simulated callback confirmation."""
        result = await connector.write_channel("TEST:CHANNEL", 100.0, verification_level="callback")

        assert isinstance(result, ChannelWriteResult)
        assert result.success is True
        assert result.verification is not None
        assert result.verification.level == "callback"
        assert result.verification.verified is True  # Mock always succeeds
        assert "callback" in result.verification.notes.lower()

    @pytest.mark.asyncio
    async def test_verification_readback_success(self, connector):
        """Test 'readback' verification level - successful verification."""
        result = await connector.write_channel(
            "TEST:CHANNEL",
            100.0,
            verification_level="readback",
            tolerance=1.0,  # Larger tolerance to account for mock noise (0.1% noise on 100 = ~0.1, but can be larger)
        )

        assert isinstance(result, ChannelWriteResult)
        assert result.success is True
        assert result.verification is not None
        assert result.verification.level == "readback"
        assert result.verification.verified is True  # Within tolerance
        assert result.verification.readback_value is not None
        assert result.verification.tolerance_used == 1.0

        # Check readback value is close to written value
        diff = abs(result.verification.readback_value - 100.0)
        assert diff <= 1.0  # Within tolerance

    @pytest.mark.asyncio
    async def test_verification_readback_with_percentage_tolerance(self, connector):
        """Test readback verification with percentage-based tolerance."""
        # Write a value
        written_value = 1000.0
        # noise_level=0.001 → sigma = 1000 * 0.001 = 1.0
        # Use 1% tolerance (= 10 sigma) to avoid flaky failures
        tolerance_percent = 1.0
        absolute_tolerance = written_value * tolerance_percent / 100.0  # = 10.0

        result = await connector.write_channel(
            "TEST:CHANNEL",
            written_value,
            verification_level="readback",
            tolerance=absolute_tolerance,
        )

        assert result.success is True
        assert result.verification.verified is True
        # Mock adds small noise, should be within 10.0
        diff = abs(result.verification.readback_value - written_value)
        assert diff <= absolute_tolerance

    @pytest.mark.asyncio
    async def test_writes_disabled_returns_failure(self):
        """Test that writes fail when writes_enabled=false (via base class wrapper)."""
        conn = MockConnector()
        with patch("osprey.utils.config.get_config_value", return_value=False):
            await conn.connect({"response_delay_ms": 1})
            result = await conn.write_channel("TEST:CHANNEL", 100.0, verification_level="callback")
            await conn.disconnect()

        assert result.success is False
        assert result.error_message is not None
        assert "disabled" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_invalid_verification_level_raises_error(self, connector):
        """Test that invalid verification level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid verification_level"):
            await connector.write_channel("TEST:CHANNEL", 100.0, verification_level="invalid_level")


class TestVerificationDataModels:
    """Test verification data model structures."""

    def test_write_verification_model(self):
        """Test WriteVerification dataclass structure."""
        verif = WriteVerification(
            level="readback",
            verified=True,
            readback_value=100.5,
            tolerance_used=0.1,
            notes="Readback confirmed",
        )

        assert verif.level == "readback"
        assert verif.verified is True
        assert verif.readback_value == 100.5
        assert verif.tolerance_used == 0.1
        assert verif.notes == "Readback confirmed"

    def test_channel_write_result_model(self):
        """Test ChannelWriteResult dataclass structure."""
        result = ChannelWriteResult(
            channel_address="TEST:CHANNEL",
            value_written=100.0,
            success=True,
            verification=WriteVerification(
                level="callback", verified=True, notes="IOC callback confirmed"
            ),
        )

        assert result.channel_address == "TEST:CHANNEL"
        assert result.value_written == 100.0
        assert result.success is True
        assert result.verification is not None
        assert result.verification.level == "callback"
        assert result.verification.verified is True

    def test_channel_write_result_without_verification(self):
        """Test ChannelWriteResult can be created without verification."""
        result = ChannelWriteResult(
            channel_address="TEST:CHANNEL", value_written=100.0, success=True, verification=None
        )

        assert result.success is True
        assert result.verification is None


class TestVerificationTolerance:
    """Test tolerance calculation strategies."""

    def test_absolute_tolerance(self):
        """Test absolute tolerance comparison."""
        written_value = 100.0
        readback_value = 100.05
        tolerance = 0.1  # Absolute

        diff = abs(readback_value - written_value)
        verified = diff <= tolerance

        assert verified is True
        assert diff == pytest.approx(0.05, rel=1e-9)

    def test_percentage_tolerance(self):
        """Test percentage-based tolerance calculation."""
        written_value = 1000.0
        tolerance_percent = 0.1  # 0.1%
        absolute_tolerance = written_value * tolerance_percent / 100.0

        assert absolute_tolerance == 1.0

        # Test cases
        assert abs(1000.5 - written_value) <= absolute_tolerance  # Pass
        assert abs(1001.0 - written_value) <= absolute_tolerance  # Pass
        assert abs(1001.5 - written_value) > absolute_tolerance  # Fail

    def test_scale_adaptive_tolerance(self):
        """Test that percentage tolerance adapts to value scale."""
        tolerance_percent = 0.1  # 0.1% (one per mil)

        # Large value (MHz scale)
        large_value = 500.5  # MHz
        large_tolerance = large_value * tolerance_percent / 100.0
        assert large_tolerance == pytest.approx(0.5005, rel=1e-6)

        # Small value (mA scale)
        small_value = 5.0  # mA
        small_tolerance = small_value * tolerance_percent / 100.0
        assert small_tolerance == pytest.approx(0.005, rel=1e-6)

        # Both are 0.1% of their respective values
        assert (large_tolerance / large_value) * 100 == pytest.approx(0.1, rel=1e-6)
        assert (small_tolerance / small_value) * 100 == pytest.approx(0.1, rel=1e-6)


class TestWriteVerificationAlarmAndFailureFields:
    """Round-trip the optional alarm / failure-kind fields on WriteVerification.

    These are the machine-readable half of the write contract (issue #465):
    consumers classify a write from them instead of parsing ``notes``. They are
    optional, so every pre-existing construction — Mock, DOOCS, EPICS, virtual
    accelerator, and out-of-tree custom connectors — must stay valid untouched.
    """

    def test_new_fields_default_to_none(self):
        """A construction that predates the new fields still works, all null.

        ``None`` means "not reported" and is deliberately distinct from a
        reported-healthy severity of 0.
        """
        verif = WriteVerification(level="callback", verified=True)

        assert verif.readback_alarm_status is None
        assert verif.readback_alarm_severity is None
        assert verif.failure_kind is None

    def test_legacy_positional_construction_still_valid(self):
        """The new fields are appended, so positional callers are unaffected."""
        verif = WriteVerification("readback", True, 100.5, 0.1, "Readback confirmed")

        assert verif.level == "readback"
        assert verif.verified is True
        assert verif.readback_value == 100.5
        assert verif.tolerance_used == 0.1
        assert verif.notes == "Readback confirmed"
        assert verif.readback_alarm_status is None
        assert verif.readback_alarm_severity is None
        assert verif.failure_kind is None

    def test_alarm_status_round_trips_as_a_name_string(self):
        """Alarm status carries the protocol's alarm *name*, not a raw code."""
        verif = WriteVerification(
            level="readback",
            verified=True,
            readback_value=100.5,
            readback_alarm_status="NO_ALARM",
            readback_alarm_severity=0,
        )

        assert verif.readback_alarm_status == "NO_ALARM"

    def test_alarm_severity_round_trips_as_an_int(self):
        """Severity is an integer: 0 healthy, 1 MINOR, 2 MAJOR, 3 INVALID."""
        verif = WriteVerification(
            level="readback",
            verified=True,
            readback_value=100.5,
            readback_alarm_status="HIHI",
            readback_alarm_severity=2,
        )

        assert verif.readback_alarm_severity == 2

    def test_healthy_severity_zero_is_distinct_from_not_reported(self):
        """Severity 0 is a reported fact; ``None`` is the absence of one."""
        reported = WriteVerification(level="readback", verified=True, readback_alarm_severity=0)
        absent = WriteVerification(level="readback", verified=True)

        assert reported.readback_alarm_severity == 0
        assert reported.readback_alarm_severity is not None
        assert absent.readback_alarm_severity is None

    def test_failure_kind_readback_failed(self):
        """A readback that raised is classified structurally, not via notes."""
        verif = WriteVerification(
            level="readback",
            verified=False,
            failure_kind="readback_failed",
            notes="Readback read raised TimeoutError",
        )

        assert verif.failure_kind == "readback_failed"
        assert verif.verified is False

    def test_plain_mismatch_leaves_failure_kind_none(self):
        """An ordinary value mismatch is not a readback failure."""
        verif = WriteVerification(
            level="readback",
            verified=False,
            readback_value=95.0,
            tolerance_used=0.1,
            notes="Readback 95.0 outside tolerance of 100.0",
        )

        assert verif.failure_kind is None

    def test_full_field_set_round_trips_through_asdict(self):
        """Every field survives a dataclass -> dict -> dataclass round trip."""
        original = WriteVerification(
            level="readback",
            verified=False,
            readback_value=95.0,
            tolerance_used=0.1,
            notes="mismatch with active alarm",
            readback_alarm_status="HIHI",
            readback_alarm_severity=2,
            failure_kind=None,
        )

        restored = WriteVerification(**asdict(original))

        assert restored == original
        assert asdict(restored) == asdict(original)

    def test_new_fields_are_optional_on_the_dataclass(self):
        """Pin the defaults at the dataclass level, not just via a constructor."""
        defaults = {f.name: f.default for f in fields(WriteVerification)}

        assert defaults["readback_alarm_status"] is None
        assert defaults["readback_alarm_severity"] is None
        assert defaults["failure_kind"] is None

    def test_new_field_annotations_are_pinned(self):
        """The declared types are the contract.

        Pinned on the dataclass, not with ``isinstance`` on a constructed
        instance: asserting the type of a literal the test itself passed in only
        re-checks the literal, and would keep passing if a field were widened.
        """
        hints = get_type_hints(WriteVerification)

        assert hints["readback_alarm_status"] == (str | None)
        assert hints["readback_alarm_severity"] == (int | None)
        assert hints["failure_kind"] == (str | None)

    def test_write_verification_nests_in_channel_write_result(self):
        """The new fields reach consumers through the shipped result type."""
        result = ChannelWriteResult(
            channel_address="TEST:CHANNEL",
            value_written=100.0,
            success=True,
            verification=WriteVerification(
                level="readback",
                verified=True,
                readback_value=100.0,
                readback_alarm_status="MINOR",
                readback_alarm_severity=1,
            ),
        )

        assert result.verification.readback_alarm_status == "MINOR"
        assert result.verification.readback_alarm_severity == 1
        assert result.verification.failure_kind is None


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
