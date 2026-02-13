"""Tests for native ChannelReadCapability.

Tests class attributes, error classification, context class, and guides.
"""

from datetime import datetime

from osprey.base.errors import ErrorSeverity
from osprey.capabilities.channel_read import (
    ChannelAccessError,
    ChannelDependencyError,
    ChannelNotFoundError,
    ChannelReadCapability,
    ChannelTimeoutError,
    ChannelValue,
    ChannelValuesContext,
)


class TestChannelReadCapabilityAttributes:
    """Test capability class attributes."""

    def test_name(self):
        assert ChannelReadCapability.name == "channel_read"

    def test_provides(self):
        assert "CHANNEL_VALUES" in ChannelReadCapability.provides

    def test_requires(self):
        assert "CHANNEL_ADDRESSES" in ChannelReadCapability.requires

    def test_has_description(self):
        assert ChannelReadCapability.description
        assert len(ChannelReadCapability.description) > 0


class TestChannelReadErrorClassification:
    """Test classify_error returns correct severity for each error type."""

    def test_timeout_is_retriable(self):
        exc = ChannelTimeoutError("Connection timed out")
        result = ChannelReadCapability.classify_error(exc, {})
        assert result.severity == ErrorSeverity.RETRIABLE

    def test_access_error_is_retriable(self):
        exc = ChannelAccessError("Permission denied")
        result = ChannelReadCapability.classify_error(exc, {})
        assert result.severity == ErrorSeverity.RETRIABLE

    def test_not_found_is_replanning(self):
        exc = ChannelNotFoundError("Channel does not exist")
        result = ChannelReadCapability.classify_error(exc, {})
        assert result.severity == ErrorSeverity.REPLANNING

    def test_dependency_error_is_replanning(self):
        exc = ChannelDependencyError("Missing CHANNEL_ADDRESSES")
        result = ChannelReadCapability.classify_error(exc, {})
        assert result.severity == ErrorSeverity.REPLANNING

    def test_unknown_error_is_retriable(self):
        """channel_read treats unknown errors as retriable (defensive, non-critical)."""
        exc = RuntimeError("Unexpected error")
        result = ChannelReadCapability.classify_error(exc, {})
        assert result.severity == ErrorSeverity.RETRIABLE


class TestChannelValuesContext:
    """Test ChannelValuesContext data class."""

    def test_context_type(self):
        assert ChannelValuesContext.CONTEXT_TYPE == "CHANNEL_VALUES"

    def test_context_category(self):
        assert ChannelValuesContext.CONTEXT_CATEGORY == "COMPUTATIONAL_DATA"

    def test_create_context(self):
        ctx = ChannelValuesContext(
            channel_values={
                "SR:CURRENT:RB": ChannelValue(
                    value="400.5",
                    timestamp=datetime(2024, 1, 1, 12, 0, 0),
                    units="mA",
                ),
            }
        )
        assert ctx.channel_count == 1

    def test_get_summary(self):
        ctx = ChannelValuesContext(
            channel_values={
                "SR:CURRENT:RB": ChannelValue(
                    value="400.5",
                    timestamp=datetime(2024, 1, 1, 12, 0, 0),
                    units="mA",
                ),
            }
        )
        summary = ctx.get_summary()
        assert summary["type"] == "Channel Values"
        assert "SR:CURRENT:RB" in summary["channel_data"]

    def test_get_access_details(self):
        ctx = ChannelValuesContext(
            channel_values={
                "SR:CURRENT:RB": ChannelValue(
                    value="400.5",
                    timestamp=datetime(2024, 1, 1, 12, 0, 0),
                    units="mA",
                ),
            }
        )
        details = ctx.get_access_details("test_key")
        assert details["channel_count"] == 1


class TestChannelReadGuides:
    """Test orchestrator and classifier guides."""

    def test_orchestrator_guide_exists(self):
        cap = ChannelReadCapability()
        guide = cap._create_orchestrator_guide()
        assert guide is not None
        assert guide.instructions
        assert len(guide.examples) > 0

    def test_classifier_guide_exists(self):
        cap = ChannelReadCapability()
        guide = cap._create_classifier_guide()
        assert guide is not None
        assert guide.instructions
        assert len(guide.examples) > 0
