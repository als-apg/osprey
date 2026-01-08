"""Tests for context management tools.

This module tests the LangChain tools that allow ReAct agents to
read, save, list, remove, and summarize context data.
"""


class TestContextManagementTools:
    """Tests for context management tools in context_tools.py."""

    def test_create_context_tools_returns_six_tools(self):
        """Should create six context management tools."""
        from osprey.capabilities.context_tools import create_context_tools

        state = {
            "capability_context_data": {},
            "session_state": {},
        }

        tools = create_context_tools(state, "test_capability")

        assert len(tools) == 6

        tool_names = [t.name for t in tools]
        assert "read_context" in tool_names
        assert "list_available_context" in tool_names
        assert "save_result_to_context" in tool_names
        assert "remove_context" in tool_names
        assert "clear_context_type" in tool_names
        assert "get_context_summary" in tool_names

    def test_list_context_returns_empty_message(self):
        """list_available_context should handle empty context."""
        from osprey.capabilities.context_tools import create_context_tools

        state = {"capability_context_data": {}, "session_state": {}}

        tools = create_context_tools(state, "test_cap")
        list_tool = next(t for t in tools if t.name == "list_available_context")

        result = list_tool.invoke({})

        assert "No context data available" in result

    def test_get_context_summary_empty(self):
        """get_context_summary should handle empty context."""
        from osprey.capabilities.context_tools import create_context_tools

        state = {"capability_context_data": {}, "session_state": {}}

        tools = create_context_tools(state, "test_cap")
        summary_tool = next(t for t in tools if t.name == "get_context_summary")

        result = summary_tool.invoke({})

        assert "No context data accumulated" in result

    def test_get_context_summary_with_data(self):
        """get_context_summary should return counts by type."""
        from osprey.capabilities.context_tools import create_context_tools

        state = {
            "capability_context_data": {
                "WEATHER_RESULTS": {"sf": {}, "nyc": {}},
                "PV_ADDRESSES": {"beam_current": {}},
            },
            "session_state": {},
        }

        tools = create_context_tools(state, "test_cap")
        summary_tool = next(t for t in tools if t.name == "get_context_summary")

        result = summary_tool.invoke({})

        assert "WEATHER_RESULTS" in result
        assert "2" in result  # 2 weather items
        assert "PV_ADDRESSES" in result
        assert "Total:" in result

    def test_remove_context_nonexistent(self):
        """remove_context should handle nonexistent context gracefully."""
        from osprey.capabilities.context_tools import create_context_tools

        state = {"capability_context_data": {}, "session_state": {}}

        tools = create_context_tools(state, "test_cap")
        remove_tool = next(t for t in tools if t.name == "remove_context")

        result = remove_tool.invoke({"context_type": "NONEXISTENT", "context_key": "key"})

        assert "not found" in result.lower()

    def test_clear_context_type_nonexistent(self):
        """clear_context_type should handle nonexistent type gracefully."""
        from osprey.capabilities.context_tools import create_context_tools

        state = {"capability_context_data": {}, "session_state": {}}

        tools = create_context_tools(state, "test_cap")
        clear_tool = next(t for t in tools if t.name == "clear_context_type")

        result = clear_tool.invoke({"context_type": "NONEXISTENT"})

        assert "No context data" in result or "not found" in result.lower()
