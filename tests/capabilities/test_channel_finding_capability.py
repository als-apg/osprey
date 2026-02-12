"""Tests for native ChannelFindingCapability.

Tests class attributes, error classification, context class, and guides.
"""

from osprey.base.errors import ErrorSeverity
from osprey.capabilities.channel_finding import (
    ChannelAddressesContext,
    ChannelFinderServiceError,
    ChannelFindingCapability,
    ChannelNotFoundError,
)


class TestChannelFindingCapabilityAttributes:
    """Test capability class attributes."""

    def test_name(self):
        assert ChannelFindingCapability.name == "channel_finding"

    def test_provides(self):
        assert "CHANNEL_ADDRESSES" in ChannelFindingCapability.provides

    def test_requires(self):
        assert ChannelFindingCapability.requires == []

    def test_has_description(self):
        assert ChannelFindingCapability.description
        assert len(ChannelFindingCapability.description) > 0


class TestChannelFindingErrorClassification:
    """Test classify_error returns correct severity for each error type."""

    def test_channel_not_found_is_replanning(self):
        exc = ChannelNotFoundError("No channels found for 'beam current'")
        result = ChannelFindingCapability.classify_error(exc, {})
        assert result.severity == ErrorSeverity.REPLANNING
        assert "suggestions" in result.metadata

    def test_service_error_is_critical(self):
        exc = ChannelFinderServiceError("Service unavailable")
        result = ChannelFindingCapability.classify_error(exc, {})
        assert result.severity == ErrorSeverity.CRITICAL

    def test_unknown_error_is_critical(self):
        exc = RuntimeError("Unexpected error")
        result = ChannelFindingCapability.classify_error(exc, {})
        assert result.severity == ErrorSeverity.CRITICAL


class TestChannelAddressesContext:
    """Test ChannelAddressesContext data class."""

    def test_context_type(self):
        assert ChannelAddressesContext.CONTEXT_TYPE == "CHANNEL_ADDRESSES"

    def test_context_category(self):
        assert ChannelAddressesContext.CONTEXT_CATEGORY == "METADATA"

    def test_create_context(self):
        ctx = ChannelAddressesContext(
            channels=["SR:CURRENT:RB", "SR:BPM:X"],
            original_query="beam current and position",
        )
        assert len(ctx.channels) == 2
        assert ctx.original_query == "beam current and position"

    def test_get_access_details(self):
        ctx = ChannelAddressesContext(
            channels=["SR:CURRENT:RB"],
            original_query="beam current",
        )
        details = ctx.get_access_details("test_key")
        assert "channels" in details
        assert details["total_available"] == 1

    def test_get_summary(self):
        ctx = ChannelAddressesContext(
            channels=["SR:CURRENT:RB", "SR:BPM:X"],
            original_query="beam current and position",
        )
        summary = ctx.get_summary()
        assert summary["total_channels"] == 2
        assert summary["type"] == "Channel Addresses"


class TestChannelFindingGuides:
    """Test orchestrator and classifier guides."""

    def test_orchestrator_guide_exists(self):
        cap = ChannelFindingCapability()
        guide = cap._create_orchestrator_guide()
        assert guide is not None
        assert guide.instructions
        assert len(guide.examples) > 0

    def test_classifier_guide_exists(self):
        cap = ChannelFindingCapability()
        guide = cap._create_classifier_guide()
        assert guide is not None
        assert guide.instructions
        assert len(guide.examples) > 0
