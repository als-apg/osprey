"""Tests for context_tools — read_context and list_available_context.

Verifies that get_summary() is called without extra positional arguments,
matching the CapabilityContext.get_summary() signature.
"""

from osprey.capabilities.context_tools import create_context_tools
from osprey.context.base import CapabilityContext


# Minimal concrete subclass for testing
class _StubContext(CapabilityContext):
    description: str = "stub"

    def get_access_details(self, key: str) -> dict:
        return {"key": key}

    def get_summary(self) -> dict:
        return {"description": self.description, "type": "STUB"}


class TestReadContext:
    def test_read_context_calls_get_summary_without_extra_args(self):
        """get_summary() must be called with zero positional args (self only)."""
        ctx = _StubContext(description="beam current addresses")
        state = {
            "capability_context_data": {
                "PV_ADDRESSES": {
                    "beam_channels": ctx.model_dump(),
                },
            },
        }
        tools = create_context_tools(state, "test_cap")
        read_context = tools[0]  # first tool is read_context

        # Should NOT raise TypeError
        result = read_context.invoke(
            {"context_type": "PV_ADDRESSES", "context_key": "beam_channels"}
        )
        assert "beam current addresses" in result

    def test_read_context_no_context_found(self):
        state = {"capability_context_data": {}}
        tools = create_context_tools(state, "test_cap")
        read_context = tools[0]

        result = read_context.invoke({"context_type": "PV_ADDRESSES", "context_key": "missing"})
        assert "not found" in result


class TestListAvailableContext:
    def test_list_context_calls_get_summary_without_extra_args(self):
        """list_available_context must call get_summary() with no extra args."""
        ctx = _StubContext(description="test data")
        state = {
            "capability_context_data": {
                "PV_ADDRESSES": {
                    "beam_channels": ctx.model_dump(),
                },
            },
        }
        tools = create_context_tools(state, "test_cap")
        list_ctx = tools[1]  # second tool is list_available_context

        # Should NOT raise TypeError
        result = list_ctx.invoke({})
        assert "PV_ADDRESSES" in result

    def test_list_context_empty_state(self):
        state = {"capability_context_data": {}}
        tools = create_context_tools(state, "test_cap")
        list_ctx = tools[1]

        result = list_ctx.invoke({})
        assert "No context data" in result
