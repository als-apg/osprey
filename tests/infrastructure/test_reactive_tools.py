"""Tests for reactive_tools.py — tool conversion and execution utilities."""

from unittest.mock import MagicMock, patch

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from osprey.infrastructure.reactive_tools import (
    build_capability_tools,
    build_lightweight_tools,
    execute_lightweight_tool,
    langchain_tool_to_openai_schema,
)


class TestLangchainToolToOpenAISchema:
    """Test conversion from LangChain @tool to OpenAI function format."""

    def test_langchain_tool_to_openai_schema(self):
        """Converts a @tool with args_schema to valid OpenAI format."""

        class ReadArgs(BaseModel):
            context_type: str = Field(description="Type of context")
            context_key: str | None = Field(default=None, description="Specific key")

        @tool(args_schema=ReadArgs)
        def read_context(context_type: str, context_key: str | None = None) -> str:
            """Read accumulated context data."""
            return ""

        schema = langchain_tool_to_openai_schema(read_context)

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "read_context"
        assert schema["function"]["description"] == "Read accumulated context data."
        assert "context_type" in schema["function"]["parameters"]["properties"]
        assert "required" in schema["function"]["parameters"]

    def test_langchain_tool_no_args(self):
        """Tool with no arguments produces empty properties."""

        @tool
        def no_args_tool() -> str:
            """A tool with no args."""
            return "result"

        schema = langchain_tool_to_openai_schema(no_args_tool)

        assert schema["function"]["parameters"]["type"] == "object"
        # Tool with no explicit args still gets the auto-generated schema
        assert isinstance(schema["function"]["parameters"]["properties"], dict)


class TestBuildLightweightTools:
    """Test lightweight tool filtering and construction."""

    @patch("osprey.capabilities.state_tools.create_state_tools")
    @patch("osprey.capabilities.context_tools.create_context_tools")
    def test_build_lightweight_tools_filters_read_only(self, mock_context_tools, mock_state_tools):
        """Only the 6 read-only tools are included."""
        # Create mock tools for each context tool
        context_tool_names = [
            "read_context",
            "list_available_context",
            "save_result_to_context",
            "remove_context",
            "clear_context_type",
            "get_context_summary",
        ]
        state_tool_names = [
            "get_session_info",
            "get_execution_status",
            "list_system_capabilities",
            "get_agent_settings",
            "clear_session",
            "modify_agent_setting",
        ]

        def make_mock_tool(name):
            t = MagicMock()
            t.name = name
            t.description = f"Description of {name}"
            t.args_schema = None
            return t

        mock_context_tools.return_value = [make_mock_tool(n) for n in context_tool_names]
        mock_state_tools.return_value = [make_mock_tool(n) for n in state_tool_names]

        state = {}
        tool_defs, tool_map = build_lightweight_tools(state)

        expected_names = {
            "read_context",
            "list_available_context",
            "get_context_summary",
            "get_session_info",
            "get_execution_status",
            "list_system_capabilities",
        }

        actual_names = {d["function"]["name"] for d in tool_defs}
        assert actual_names == expected_names

        # Excluded: save_result_to_context, remove_context, clear_context_type,
        # get_agent_settings, clear_session, modify_agent_setting
        assert "save_result_to_context" not in actual_names
        assert "clear_session" not in actual_names

    @patch("osprey.capabilities.state_tools.create_state_tools")
    @patch("osprey.capabilities.context_tools.create_context_tools")
    def test_build_lightweight_tools_returns_executors(self, mock_context_tools, mock_state_tools):
        """Returned executor map has entries for each included tool name."""

        def make_mock_tool(name):
            t = MagicMock()
            t.name = name
            t.description = f"Description of {name}"
            t.args_schema = None
            return t

        mock_context_tools.return_value = [make_mock_tool("read_context")]
        mock_state_tools.return_value = [make_mock_tool("get_session_info")]

        state = {}
        tool_defs, tool_map = build_lightweight_tools(state)

        assert "read_context" in tool_map
        assert "get_session_info" in tool_map
        assert len(tool_map) == 2


class TestBuildCapabilityTools:
    """Test capability-to-tool conversion."""

    def test_build_capability_tools_schema(self):
        """Each capability tool has task_objective, context_key as required params."""
        cap = MagicMock()
        cap.name = "channel_finding"
        cap.description = "Find EPICS channels"
        cap.requires = []
        cap.provides = ["CHANNEL_ADDRESSES"]

        tools = build_capability_tools([cap])

        assert len(tools) == 1
        fn = tools[0]["function"]
        assert fn["name"] == "channel_finding"
        assert "task_objective" in fn["parameters"]["properties"]
        assert "context_key" in fn["parameters"]["properties"]
        assert "task_objective" in fn["parameters"]["required"]
        assert "context_key" in fn["parameters"]["required"]

    def test_build_capability_tools_uses_capability_description(self):
        """Tool description enriched with provides/requires info."""
        cap = MagicMock()
        cap.name = "channel_finding"
        cap.description = "Find EPICS channels by name or pattern"
        cap.requires = []
        cap.provides = ["CHANNEL_ADDRESSES"]

        tools = build_capability_tools([cap])

        desc = tools[0]["function"]["description"]
        assert desc.startswith("Find EPICS channels by name or pattern")
        assert "Provides: CHANNEL_ADDRESSES" in desc


class TestExecuteLightweightTool:
    """Test inline tool execution."""

    def test_execute_lightweight_tool_invokes_langchain_tool(self):
        """Dispatches to .invoke() with parsed arguments and returns result."""
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "Context Summary: 3 items"

        tool_map = {"get_context_summary": mock_tool}
        result = execute_lightweight_tool("get_context_summary", "{}", tool_map)

        mock_tool.invoke.assert_called_once_with({})
        assert result == "Context Summary: 3 items"

    def test_execute_lightweight_tool_with_args(self):
        """Arguments are parsed from JSON string."""
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "data"

        tool_map = {"read_context": mock_tool}
        execute_lightweight_tool(
            "read_context",
            '{"context_type": "PV_ADDRESSES", "context_key": "beam"}',
            tool_map,
        )

        mock_tool.invoke.assert_called_once_with(
            {"context_type": "PV_ADDRESSES", "context_key": "beam"}
        )

    def test_execute_lightweight_tool_unknown_name(self):
        """Returns error dict for unknown tool name."""
        tool_map = {}
        result = execute_lightweight_tool("nonexistent", "{}", tool_map)
        assert "error" in result.lower() or "Unknown" in result

    def test_execute_lightweight_tool_exception_handling(self):
        """Tool that raises returns error string, doesn't propagate."""
        mock_tool = MagicMock()
        mock_tool.invoke.side_effect = RuntimeError("tool crashed")

        tool_map = {"get_session_info": mock_tool}
        result = execute_lightweight_tool("get_session_info", "{}", tool_map)

        assert "Error" in result
        assert "tool crashed" in result
