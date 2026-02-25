"""Tests for the ChannelFindingCapability.

Covers class attributes, error classification, and context class behavior.
"""

from osprey.base.errors import ErrorSeverity
from osprey.capabilities.channel_finding import (
    ChannelAddressesContext,
    ChannelFindingCapability,
)


class TestChannelFindingCapability:
    """Tests for ChannelFindingCapability class attributes and methods."""

    def test_name(self):
        cap = ChannelFindingCapability()
        assert cap.name == "channel_finding"

    def test_description(self):
        cap = ChannelFindingCapability()
        assert cap.description is not None
        assert len(cap.description) > 0

    def test_provides(self):
        cap = ChannelFindingCapability()
        assert "CHANNEL_ADDRESSES" in cap.provides

    def test_requires_empty(self):
        cap = ChannelFindingCapability()
        assert cap.requires == []


class TestChannelAddressesContext:
    """Tests for ChannelAddressesContext."""

    def test_context_type(self):
        assert ChannelAddressesContext.CONTEXT_TYPE == "CHANNEL_ADDRESSES"

    def test_creation(self):
        ctx = ChannelAddressesContext(
            channels=["SR:BPM:01", "SR:BPM:02"],
            original_query="beam position monitors",
        )
        assert len(ctx.channels) == 2
        assert ctx.original_query == "beam position monitors"

    def test_summary(self):
        ctx = ChannelAddressesContext(
            channels=["SR:BPM:01"],
            original_query="test",
        )
        summary = ctx.get_summary()
        assert "SR:BPM:01" in summary

    def test_access_details(self):
        ctx = ChannelAddressesContext(
            channels=["SR:BPM:01"],
            original_query="test",
        )
        details = ctx.get_access_details("my_key")
        assert details["context_type"] == "CHANNEL_ADDRESSES"
        assert details["channel_count"] == 1


class TestClassifyError:
    """Tests for ChannelFindingCapability.classify_error."""

    def test_unknown_exception_is_retriable(self):
        result = ChannelFindingCapability.classify_error(RuntimeError("boom"), {})
        assert result.severity == ErrorSeverity.RETRIABLE
