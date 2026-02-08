"""Tests for native ChannelWriteCapability.

Tests class attributes, error classification, context classes, and guides.
"""

from osprey.base.errors import ErrorSeverity
from osprey.capabilities.channel_write import (
    AmbiguousWriteOperationError,
    ChannelNotWritableError,
    ChannelWriteAccessError,
    ChannelWriteCapability,
    ChannelWriteDependencyError,
    ChannelWriteResult,
    ChannelWriteResultsContext,
    ChannelWriteTimeoutError,
    WriteOperation,
    WriteOperationsOutput,
    WriteParsingError,
    WriteVerificationInfo,
)


class TestChannelWriteCapabilityAttributes:
    """Test capability class attributes."""

    def test_name(self):
        assert ChannelWriteCapability.name == "channel_write"

    def test_provides(self):
        assert "CHANNEL_WRITE_RESULTS" in ChannelWriteCapability.provides

    def test_requires(self):
        assert "CHANNEL_ADDRESSES" in ChannelWriteCapability.requires

    def test_has_description(self):
        assert ChannelWriteCapability.description
        assert len(ChannelWriteCapability.description) > 0


class TestChannelWriteErrorClassification:
    """Test classify_error returns correct severity for each error type."""

    def test_ambiguous_write_is_replanning(self):
        exc = AmbiguousWriteOperationError("Cannot determine target channel")
        result = ChannelWriteCapability.classify_error(exc, {})
        assert result.severity == ErrorSeverity.REPLANNING

    def test_write_parsing_is_retriable(self):
        exc = WriteParsingError("Failed to parse write operations")
        result = ChannelWriteCapability.classify_error(exc, {})
        assert result.severity == ErrorSeverity.RETRIABLE

    def test_timeout_is_retriable(self):
        exc = ChannelWriteTimeoutError("Write timed out")
        result = ChannelWriteCapability.classify_error(exc, {})
        assert result.severity == ErrorSeverity.RETRIABLE

    def test_access_error_is_critical(self):
        exc = ChannelWriteAccessError("Write access denied")
        result = ChannelWriteCapability.classify_error(exc, {})
        assert result.severity == ErrorSeverity.CRITICAL

    def test_not_writable_is_critical(self):
        exc = ChannelNotWritableError("Channel is read-only")
        result = ChannelWriteCapability.classify_error(exc, {})
        assert result.severity == ErrorSeverity.CRITICAL

    def test_dependency_error_is_replanning(self):
        exc = ChannelWriteDependencyError("Missing CHANNEL_ADDRESSES")
        result = ChannelWriteCapability.classify_error(exc, {})
        assert result.severity == ErrorSeverity.REPLANNING

    def test_unknown_error_is_critical(self):
        exc = RuntimeError("Unexpected error")
        result = ChannelWriteCapability.classify_error(exc, {})
        assert result.severity == ErrorSeverity.CRITICAL


class TestChannelWriteContextClasses:
    """Test context and internal model classes."""

    def test_write_verification_info(self):
        info = WriteVerificationInfo(
            level="readback",
            verified=True,
            readback_value=50.0,
            tolerance_used=0.1,
            notes="Readback within tolerance",
        )
        assert info.verified is True
        assert info.level == "readback"

    def test_channel_write_result(self):
        result = ChannelWriteResult(
            channel_address="MAG:HCM01:CURRENT:SP",
            value_written=5.0,
            success=True,
        )
        assert result.success is True
        assert result.channel_address == "MAG:HCM01:CURRENT:SP"

    def test_channel_write_result_with_verification(self):
        result = ChannelWriteResult(
            channel_address="MAG:HCM01:CURRENT:SP",
            value_written=5.0,
            success=True,
            verification=WriteVerificationInfo(
                level="callback",
                verified=True,
            ),
        )
        assert result.verification.verified is True

    def test_write_results_context_type(self):
        assert ChannelWriteResultsContext.CONTEXT_TYPE == "CHANNEL_WRITE_RESULTS"
        assert ChannelWriteResultsContext.CONTEXT_CATEGORY == "COMPUTATIONAL_DATA"

    def test_write_results_context(self):
        ctx = ChannelWriteResultsContext(
            results=[
                ChannelWriteResult(
                    channel_address="MAG:HCM01:CURRENT:SP",
                    value_written=5.0,
                    success=True,
                ),
                ChannelWriteResult(
                    channel_address="MAG:HCM02:CURRENT:SP",
                    value_written=10.0,
                    success=False,
                    error_message="Write failed",
                ),
            ],
            total_writes=2,
            successful_count=1,
            failed_count=1,
        )
        summary = ctx.get_summary()
        assert summary["total_writes"] == 2
        assert summary["successful"] == 1
        assert summary["failed"] == 1

    def test_write_operation_model(self):
        op = WriteOperation(
            channel_address="MAG:HCM01:CURRENT:SP",
            value=5.0,
            units="A",
            notes="Test write",
        )
        assert op.value == 5.0
        assert op.units == "A"

    def test_write_operations_output_model(self):
        output = WriteOperationsOutput(
            write_operations=[
                WriteOperation(channel_address="MAG:HCM01:CURRENT:SP", value=5.0),
            ],
            found=True,
        )
        assert output.found is True
        assert len(output.write_operations) == 1


class TestChannelWriteGuides:
    """Test orchestrator and classifier guides."""

    def test_orchestrator_guide_exists(self):
        cap = ChannelWriteCapability()
        guide = cap._create_orchestrator_guide()
        assert guide is not None
        assert guide.instructions
        assert len(guide.examples) > 0

    def test_classifier_guide_exists(self):
        cap = ChannelWriteCapability()
        guide = cap._create_classifier_guide()
        assert guide is not None
        assert guide.instructions
        assert len(guide.examples) > 0
