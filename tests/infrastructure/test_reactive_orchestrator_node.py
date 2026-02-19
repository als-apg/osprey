"""Tests for ReactiveOrchestratorNode.

Tests mock get_chat_completion to avoid LLM calls. Tests verify
the node produces correct state updates based on LLM decisions.

The reactive orchestrator uses two response formats:
- Tool call list: [{"id": ..., "function": {"name": ..., "arguments": ...}}]
- Text fallback: JSON string parseable as ExecutionPlan (backward compatibility)
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from osprey.base.errors import (
    InvalidContextKeyError,
    ReclassificationRequiredError,
)
from osprey.infrastructure.reactive_orchestrator_node import (
    ReactiveOrchestratorNode,
    _build_chat_request,
    _build_missing_requirements_response,
    _format_execution_history,
    _format_observation,
    _resolve_inputs,
)
from osprey.models.messages import ChatCompletionRequest
from tests.conftest import create_test_state


def _make_tool_call(name, arguments=None, call_id=None):
    if arguments is None:
        arguments = {}
    return {
        "id": call_id or f"call_{name}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments) if isinstance(arguments, dict) else arguments,
        },
    }


def _create_react_node_state(**overrides):
    defaults = {
        "user_message": "Find beam current",
        "task_current_task": "Find beam current channels",
        "planning_active_capabilities": ["channel_finding", "channel_read"],
        "planning_execution_plan": None,
        "planning_current_step_index": 0,
    }
    defaults.update(overrides)
    return create_test_state(**defaults)


@pytest.fixture
def mock_registry():
    registry = MagicMock()

    def make_cap(name, description, provides, requires):
        cap = MagicMock()
        cap.name = name
        cap.description = description
        cap.provides = provides
        cap.requires = requires
        cap.orchestrator_guide = None
        return cap

    capabilities = {
        "channel_finding": make_cap("channel_finding", "Find channels", ["CHANNEL_ADDRESSES"], []),
        "channel_read": make_cap(
            "channel_read", "Read channels", ["CHANNEL_VALUES"], ["CHANNEL_ADDRESSES"]
        ),
        "respond": make_cap("respond", "Generate response", ["FINAL_RESPONSE"], []),
    }

    registry.get_capability.side_effect = lambda name: capabilities.get(name)
    registry.get_node.side_effect = lambda name: MagicMock() if name in capabilities else None
    registry.get_all_capabilities.return_value = list(capabilities.values())
    registry.get_stats.return_value = {"capability_names": list(capabilities.keys())}
    return registry


@pytest.fixture
def mock_prompt_builder():
    builder = MagicMock()
    builder.get_reactive_instructions.return_value = "You are an expert.\n\nREACTIVE MODE"
    provider = MagicMock()
    provider.get_orchestrator_prompt_builder.return_value = builder
    return provider


def _common_patches(mock_registry, mock_prompt_builder):
    return {
        "registry": patch(
            "osprey.infrastructure.reactive_orchestrator_node.get_registry",
            return_value=mock_registry,
        ),
        "model_config": patch(
            "osprey.infrastructure.reactive_orchestrator_node.get_model_config",
            return_value={"provider": "test", "model_id": "test"},
        ),
        "api_context": patch("osprey.models.set_api_call_context"),
        "validate_registry": patch(
            "osprey.infrastructure.orchestration_node.get_registry", return_value=mock_registry
        ),
        "prompts": patch(
            "osprey.infrastructure.reactive_orchestrator_node.get_framework_prompts",
            return_value=mock_prompt_builder,
        ),
        "lw_ctx": patch("osprey.capabilities.context_tools.create_context_tools", return_value=[]),
        "lw_state": patch("osprey.capabilities.state_tools.create_state_tools", return_value=[]),
        "interface_ctx": patch(
            "osprey.infrastructure.reactive_orchestrator_node.get_interface_context",
            return_value="cli",
        ),
    }


def _enter(patches):
    return [p.start() for p in patches.values()]


def _exit(patches):
    for p in patches.values():
        p.stop()


class TestReactiveOrchestratorFirstStep:
    @pytest.mark.asyncio
    async def test_first_step_execute_capability(self, mock_registry, mock_prompt_builder):
        llm_response = [
            _make_tool_call(
                "channel_finding",
                {
                    "task_objective": "Find beam current channels",
                    "context_key": "beam_channels",
                    "expected_output": "CHANNEL_ADDRESSES",
                    "success_criteria": "Channels found",
                },
            )
        ]
        state = _create_react_node_state()
        patches = _common_patches(mock_registry, mock_prompt_builder)
        _enter(patches)
        try:
            with patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                return_value=llm_response,
            ):
                node = ReactiveOrchestratorNode()
                node._state = state
                result = await node.execute()
        finally:
            _exit(patches)

        plan = result["planning_execution_plan"]
        assert len(plan["steps"]) == 1
        assert plan["steps"][0]["capability"] == "channel_finding"
        assert plan["steps"][0]["context_key"] == "beam_channels"
        assert result["react_step_count"] == 1
        assert len(result["react_messages"]) == 1


class TestReactiveOrchestratorSecondStep:
    @pytest.mark.asyncio
    async def test_second_step_with_context(self, mock_registry, mock_prompt_builder):
        llm_response = [
            _make_tool_call(
                "channel_read",
                {"task_objective": "Read beam current values", "context_key": "beam_values"},
            )
        ]
        state = _create_react_node_state(
            capability_context_data={"CHANNEL_ADDRESSES": {"beam_channels": {"pvs": ["SR:C01"]}}},
            react_messages=[{"role": "assistant", "content": "Step 1: find channels"}],
            react_step_count=1,
            execution_step_results={
                "beam_channels": {
                    "step_index": 0,
                    "capability": "channel_finding",
                    "success": True,
                    "task_objective": "Find channels",
                }
            },
        )
        patches = _common_patches(mock_registry, mock_prompt_builder)
        _enter(patches)
        try:
            with patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                return_value=llm_response,
            ):
                node = ReactiveOrchestratorNode()
                node._state = state
                result = await node.execute()
        finally:
            _exit(patches)

        plan = result["planning_execution_plan"]
        assert plan["steps"][0]["capability"] == "channel_read"
        assert plan["steps"][0]["inputs"] == [{"CHANNEL_ADDRESSES": "beam_channels"}]
        assert result["react_step_count"] == 2
        assert len(result["react_messages"]) == 3
        assert result["react_messages"][1]["role"] == "observation"


class TestReactiveOrchestratorRespond:
    @pytest.mark.asyncio
    async def test_respond_action(self, mock_registry, mock_prompt_builder):
        """Respond action should generate response directly (AIMessage) instead of creating a PlannedStep."""
        orchestrator_call = [
            _make_tool_call(
                "respond", {"task_objective": "Summarize values", "context_key": "user_response"}
            )
        ]
        response_text = "The beam current reading is 42.5 mA."

        # First call: orchestrator LLM returns respond tool call
        # Second call: response LLM generates text
        mock_completion = MagicMock(side_effect=[orchestrator_call, response_text])

        # Add format_reactive_response_context mock
        orch_builder = mock_prompt_builder.get_orchestrator_prompt_builder()
        orch_builder.format_reactive_response_context.return_value = "[Decision] Respond"

        # Add response builder mock
        response_builder = MagicMock()
        response_builder.get_system_instructions.return_value = "You are an assistant."
        mock_prompt_builder.get_response_generation_prompt_builder.return_value = response_builder

        state = _create_react_node_state(
            react_step_count=2,
            planning_active_capabilities=["channel_finding", "channel_read", "respond"],
        )
        patches = _common_patches(mock_registry, mock_prompt_builder)
        _enter(patches)
        try:
            with patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                mock_completion,
            ):
                node = ReactiveOrchestratorNode()
                node._state = state
                result = await node.execute()
        finally:
            _exit(patches)

        # Should have AIMessage instead of planning_execution_plan
        assert "messages" in result
        from langchain_core.messages import AIMessage

        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == response_text
        assert result["react_response_generated"] is True
        assert result["react_step_count"] == 3
        # Should NOT have planning_execution_plan
        assert "planning_execution_plan" not in result


class TestReactiveOrchestratorMessageAccumulation:
    @pytest.mark.asyncio
    async def test_messages_accumulate(self, mock_registry, mock_prompt_builder):
        llm_response = [
            _make_tool_call(
                "channel_finding", {"task_objective": "Find more", "context_key": "more_channels"}
            )
        ]
        state = _create_react_node_state(
            react_messages=[
                {"role": "assistant", "content": "S1"},
                {"role": "observation", "content": "R1"},
                {"role": "assistant", "content": "S2"},
                {"role": "observation", "content": "R2"},
            ],
            react_step_count=2,
        )
        patches = _common_patches(mock_registry, mock_prompt_builder)
        _enter(patches)
        try:
            with patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                return_value=llm_response,
            ):
                node = ReactiveOrchestratorNode()
                node._state = state
                result = await node.execute()
        finally:
            _exit(patches)

        assert len(result["react_messages"]) == 5


class TestReactiveOrchestratorErrorScenarios:
    @pytest.mark.asyncio
    async def test_no_task_raises(self, mock_registry):
        state = _create_react_node_state(task_current_task=None)
        node = ReactiveOrchestratorNode()
        node._state = state
        with pytest.raises(ValueError, match="No current task"):
            await node.execute()

    @pytest.mark.asyncio
    async def test_unknown_capability_feeds_back_error(self, mock_registry, mock_prompt_builder):
        llm_response = [
            _make_tool_call(
                "totally_fake", {"task_objective": "Do something", "context_key": "fake"}
            )
        ]
        state = _create_react_node_state()
        patches = _common_patches(mock_registry, mock_prompt_builder)
        _enter(patches)
        try:
            with patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                return_value=llm_response,
            ):
                node = ReactiveOrchestratorNode()
                node._state = state
                with pytest.raises(ValueError, match="Exceeded maximum"):
                    await node.execute()
        finally:
            _exit(patches)


class TestReactiveOrchestratorOnlyActiveCapabilities:
    @pytest.mark.asyncio
    async def test_only_active_capabilities_in_prompt(self, mock_registry, mock_prompt_builder):
        llm_response = [
            _make_tool_call(
                "channel_finding", {"task_objective": "Find channels", "context_key": "channels"}
            )
        ]
        state = _create_react_node_state(planning_active_capabilities=["channel_finding"])
        patches = _common_patches(mock_registry, mock_prompt_builder)
        builder = mock_prompt_builder.get_orchestrator_prompt_builder()
        _enter(patches)
        try:
            with patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                return_value=llm_response,
            ):
                node = ReactiveOrchestratorNode()
                node._state = state
                await node.execute()
        finally:
            _exit(patches)

        mock_registry.get_all_capabilities.assert_not_called()
        builder.get_reactive_instructions.assert_called_once()
        caps_arg = builder.get_reactive_instructions.call_args[1]["active_capabilities"]
        assert len(caps_arg) == 1
        assert caps_arg[0].name == "channel_finding"


class TestResolveInputs:
    def test_resolves_requires(self, mock_registry):
        state = {
            "capability_context_data": {"CHANNEL_ADDRESSES": {"beam_channels": {"pvs": ["SR:C01"]}}}
        }
        with patch(
            "osprey.infrastructure.reactive_orchestrator_node.get_registry",
            return_value=mock_registry,
        ):
            assert _resolve_inputs("channel_read", state) == [
                {"CHANNEL_ADDRESSES": "beam_channels"}
            ]

    def test_no_requires(self, mock_registry):
        with patch(
            "osprey.infrastructure.reactive_orchestrator_node.get_registry",
            return_value=mock_registry,
        ):
            assert _resolve_inputs("channel_finding", {"capability_context_data": {}}) == []

    def test_unknown_capability(self, mock_registry):
        mock_registry.get_capability.return_value = None
        with patch(
            "osprey.infrastructure.reactive_orchestrator_node.get_registry",
            return_value=mock_registry,
        ):
            assert _resolve_inputs("nonexistent", {"capability_context_data": {}}) == []


class TestFormatObservation:
    def test_no_observation(self):
        assert (
            _format_observation(
                {
                    "control_error_info": None,
                    "execution_last_result": None,
                    "execution_step_results": {},
                }
            )
            is None
        )

    def test_error_observation(self):
        c = MagicMock()
        c.user_message = "Timeout"
        c.severity = MagicMock(value="RETRIABLE")
        result = _format_observation(
            {
                "control_error_info": {"classification": c, "capability_name": "ch"},
                "execution_last_result": None,
                "execution_step_results": {},
            }
        )
        assert "ERROR" in result

    def test_success_observation(self):
        lr = MagicMock(success=True, capability="channel_finding")
        result = _format_observation(
            {"control_error_info": None, "execution_last_result": lr, "execution_step_results": {}}
        )
        assert "successfully" in result


class TestBuildChatRequest:
    def test_first_step_returns_chat_request(self):
        result = _build_chat_request("sys", "Find beam current", [])
        assert isinstance(result, ChatCompletionRequest)
        assert len(result.messages) == 2

    def test_system_message_is_first(self):
        result = _build_chat_request("sys", "Find beam current", [])
        assert result.messages[0].role == "system"

    def test_user_task_is_second(self):
        result = _build_chat_request("sys", "Find beam current", [])
        assert "USER TASK:" in result.messages[1].content

    def test_with_prior_messages(self):
        messages = [
            {"role": "assistant", "content": "Did step 1"},
            {"role": "observation", "content": "Worked"},
        ]
        result = _build_chat_request("sys", "task", messages)
        assert len(result.messages) == 4
        assert result.messages[2].role == "assistant"
        assert result.messages[3].role == "user"

    def test_observation_content_prefixed(self):
        result = _build_chat_request(
            "sys", "task", [{"role": "observation", "content": "Step 1 worked"}]
        )
        assert result.messages[2].content.startswith("OBSERVATION:")


class TestObservationDeduplication:
    def test_observation_appears_once(self):
        react_messages = [{"role": "assistant", "content": "Action: channel_finding"}]
        obs = "Step completed successfully (capability: channel_finding)"
        react_messages.append({"role": "observation", "content": obs})
        result = _build_chat_request("sys", "task", react_messages)
        assert sum(1 for m in result.messages if obs in (m.content or "")) == 1


class TestLightweightToolExecution:
    @pytest.mark.asyncio
    async def test_lightweight_tool_call_executes_and_loops(
        self, mock_registry, mock_prompt_builder
    ):
        lw_call = [_make_tool_call("get_context_summary", {}, "call_lw")]
        cap_call = [
            _make_tool_call(
                "channel_finding",
                {"task_objective": "Find channels", "context_key": "beam_channels"},
                "call_cap",
            )
        ]
        mock_completion = MagicMock(side_effect=[lw_call, cap_call])
        state = _create_react_node_state()

        mock_lw_tool = MagicMock()
        mock_lw_tool.name = "get_context_summary"
        mock_lw_tool.description = "Get context summary"
        mock_lw_tool.args_schema = None
        mock_lw_tool.invoke.return_value = "Context: 3 items found"

        patches = _common_patches(mock_registry, mock_prompt_builder)
        with (
            patch(
                "osprey.capabilities.context_tools.create_context_tools",
                return_value=[mock_lw_tool],
            ),
            patch("osprey.capabilities.state_tools.create_state_tools", return_value=[]),
            patches["registry"],
            patches["model_config"],
            patches["api_context"],
            patches["validate_registry"],
            patches["prompts"],
            patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                mock_completion,
            ),
        ):
            node = ReactiveOrchestratorNode()
            node._state = state
            result = await node.execute()

        assert mock_completion.call_count == 2
        mock_lw_tool.invoke.assert_called_once()
        assert result["planning_execution_plan"]["steps"][0]["capability"] == "channel_finding"

    @pytest.mark.asyncio
    async def test_multiple_lightweight_tools_in_one_call(self, mock_registry, mock_prompt_builder):
        lw_calls = [
            _make_tool_call("get_context_summary", {}, "c1"),
            _make_tool_call("get_session_info", {}, "c2"),
            _make_tool_call("list_available_context", {}, "c3"),
        ]
        cap_call = [
            _make_tool_call(
                "channel_finding",
                {"task_objective": "Find channels", "context_key": "beam_channels"},
            )
        ]
        mock_completion = MagicMock(side_effect=[lw_calls, cap_call])
        state = _create_react_node_state()

        tools = []
        for name in ["get_context_summary", "get_session_info", "list_available_context"]:
            t = MagicMock(name=name, description=f"Desc {name}", args_schema=None)
            t.name = name  # MagicMock(name=...) sets _mock_name, need explicit
            t.invoke.return_value = f"result from {name}"
            tools.append(t)

        patches = _common_patches(mock_registry, mock_prompt_builder)
        with (
            patch(
                "osprey.capabilities.context_tools.create_context_tools",
                return_value=[tools[0], tools[2]],
            ),
            patch("osprey.capabilities.state_tools.create_state_tools", return_value=[tools[1]]),
            patches["registry"],
            patches["model_config"],
            patches["api_context"],
            patches["validate_registry"],
            patches["prompts"],
            patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                mock_completion,
            ),
        ):
            node = ReactiveOrchestratorNode()
            node._state = state
            await node.execute()

        for t in tools:
            t.invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_lightweight_tool_limit_raises(self, mock_registry, mock_prompt_builder):
        lw_call = [_make_tool_call("get_context_summary", {}, "call_lw")]
        mock_completion = MagicMock(return_value=lw_call)
        state = _create_react_node_state()

        mock_lw = MagicMock(name="get_context_summary", description="Get summary", args_schema=None)
        mock_lw.name = "get_context_summary"
        mock_lw.invoke.return_value = "summary"

        patches = _common_patches(mock_registry, mock_prompt_builder)
        with (
            patch("osprey.capabilities.context_tools.create_context_tools", return_value=[mock_lw]),
            patch("osprey.capabilities.state_tools.create_state_tools", return_value=[]),
            patches["registry"],
            patches["model_config"],
            patches["api_context"],
            patches["validate_registry"],
            patches["prompts"],
            patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                mock_completion,
            ),
        ):
            node = ReactiveOrchestratorNode()
            node._state = state
            with pytest.raises(ValueError, match="Exceeded maximum"):
                await node.execute()

    @pytest.mark.asyncio
    async def test_capability_tool_call_creates_execution_plan(
        self, mock_registry, mock_prompt_builder
    ):
        llm_response = [
            _make_tool_call(
                "channel_finding",
                {
                    "task_objective": "Find beam current channels",
                    "context_key": "beam_channels",
                    "expected_output": "CHANNEL_ADDRESSES",
                    "success_criteria": "Channels found",
                },
            )
        ]
        state = _create_react_node_state()
        patches = _common_patches(mock_registry, mock_prompt_builder)
        _enter(patches)
        try:
            with patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                return_value=llm_response,
            ):
                node = ReactiveOrchestratorNode()
                node._state = state
                result = await node.execute()
        finally:
            _exit(patches)

        step = result["planning_execution_plan"]["steps"][0]
        assert step["capability"] == "channel_finding"
        assert step["task_objective"] == "Find beam current channels"
        assert step["context_key"] == "beam_channels"

    @pytest.mark.asyncio
    async def test_capability_tool_call_validation_error_retries(
        self, mock_registry, mock_prompt_builder
    ):
        unknown_call = [
            _make_tool_call(
                "nonexistent_cap", {"task_objective": "Do something", "context_key": "fake"}
            )
        ]
        valid_call = [
            _make_tool_call(
                "channel_finding",
                {"task_objective": "Find channels", "context_key": "beam_channels"},
            )
        ]
        mock_completion = MagicMock(side_effect=[unknown_call, valid_call])
        state = _create_react_node_state()
        patches = _common_patches(mock_registry, mock_prompt_builder)
        _enter(patches)
        try:
            with patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                mock_completion,
            ):
                node = ReactiveOrchestratorNode()
                node._state = state
                result = await node.execute()
        finally:
            _exit(patches)

        assert mock_completion.call_count == 2
        assert result["planning_execution_plan"]["steps"][0]["capability"] == "channel_finding"

    @pytest.mark.asyncio
    async def test_mixed_lightweight_and_capability(self, mock_registry, mock_prompt_builder):
        mixed_calls = [
            _make_tool_call("get_context_summary", {}, "call_lw"),
            _make_tool_call(
                "channel_finding",
                {"task_objective": "Find channels", "context_key": "beam_channels"},
                "call_cap",
            ),
        ]
        state = _create_react_node_state()
        mock_lw = MagicMock(name="get_context_summary", description="Get summary", args_schema=None)
        mock_lw.name = "get_context_summary"
        mock_lw.invoke.return_value = "Context summary"

        patches = _common_patches(mock_registry, mock_prompt_builder)
        with (
            patch("osprey.capabilities.context_tools.create_context_tools", return_value=[mock_lw]),
            patch("osprey.capabilities.state_tools.create_state_tools", return_value=[]),
            patches["registry"],
            patches["model_config"],
            patches["api_context"],
            patches["validate_registry"],
            patches["prompts"],
            patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                return_value=mixed_calls,
            ),
        ):
            node = ReactiveOrchestratorNode()
            node._state = state
            result = await node.execute()

        mock_lw.invoke.assert_called_once()
        assert result["planning_execution_plan"]["steps"][0]["capability"] == "channel_finding"

    @pytest.mark.asyncio
    async def test_text_response_fallback_parses_as_execution_plan(
        self, mock_registry, mock_prompt_builder
    ):
        text_response = json.dumps(
            {
                "steps": [
                    {
                        "capability": "channel_finding",
                        "task_objective": "Find channels",
                        "context_key": "beam_channels",
                    }
                ]
            }
        )
        state = _create_react_node_state()
        patches = _common_patches(mock_registry, mock_prompt_builder)
        _enter(patches)
        try:
            with patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                return_value=text_response,
            ):
                node = ReactiveOrchestratorNode()
                node._state = state
                result = await node.execute()
        finally:
            _exit(patches)

        assert result["planning_execution_plan"]["steps"][0]["capability"] == "channel_finding"


class TestDirectResponseGeneration:
    """Test that respond intercept generates AIMessage directly."""

    @pytest.mark.asyncio
    async def test_non_respond_capability_creates_planned_step(
        self, mock_registry, mock_prompt_builder
    ):
        """Non-respond capabilities should still create PlannedStep as before."""
        llm_response = [
            _make_tool_call(
                "channel_finding", {"task_objective": "Find channels", "context_key": "channels"}
            )
        ]
        state = _create_react_node_state()
        patches = _common_patches(mock_registry, mock_prompt_builder)
        _enter(patches)
        try:
            with patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                return_value=llm_response,
            ):
                node = ReactiveOrchestratorNode()
                node._state = state
                result = await node.execute()
        finally:
            _exit(patches)

        assert "planning_execution_plan" in result
        assert result["planning_execution_plan"]["steps"][0]["capability"] == "channel_finding"
        assert result.get("react_response_generated") is not True

    @pytest.mark.asyncio
    async def test_text_fallback_respond_generates_directly(
        self, mock_registry, mock_prompt_builder
    ):
        """Text fallback with respond capability also generates response directly."""
        # First call: orchestrator returns text JSON with respond step
        orchestrator_response = json.dumps(
            {
                "steps": [
                    {"capability": "respond", "task_objective": "Summarize", "context_key": "resp"}
                ]
            }
        )
        response_text = "Here is the summary."

        mock_completion = MagicMock(side_effect=[orchestrator_response, response_text])

        orch_builder = mock_prompt_builder.get_orchestrator_prompt_builder()
        orch_builder.format_reactive_response_context.return_value = "[Decision] Respond"

        response_builder = MagicMock()
        response_builder.get_system_instructions.return_value = "You are an assistant."
        mock_prompt_builder.get_response_generation_prompt_builder.return_value = response_builder

        state = _create_react_node_state(
            planning_active_capabilities=["channel_finding", "channel_read", "respond"],
        )
        patches = _common_patches(mock_registry, mock_prompt_builder)
        _enter(patches)
        try:
            with patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                mock_completion,
            ):
                node = ReactiveOrchestratorNode()
                node._state = state
                result = await node.execute()
        finally:
            _exit(patches)

        from langchain_core.messages import AIMessage

        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == response_text
        assert result["react_response_generated"] is True


class TestRejectionLoopLimit:
    """Test that rejection counter increments and gate is skipped after limit."""

    @pytest.mark.asyncio
    async def test_rejection_increments_counter(self, mock_registry, mock_prompt_builder):
        """After rejection, react_rejection_count should increment."""
        llm_response = [
            _make_tool_call(
                "channel_finding", {"task_objective": "Find channels", "context_key": "channels_v2"}
            )
        ]
        state = _create_react_node_state(
            react_rejection_count=1,
            approval_approved=False,
            approved_payload=None,
        )
        # Simulate rejection resume: has_approval_resume=True, approved_payload=None
        patches = _common_patches(mock_registry, mock_prompt_builder)
        _enter(patches)
        try:
            with (
                patch(
                    "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                    return_value=llm_response,
                ),
                patch(
                    "osprey.infrastructure.reactive_orchestrator_node.get_approval_resume_data",
                    return_value=(True, None),
                ),
            ):
                node = ReactiveOrchestratorNode()
                node._state = state
                result = await node.execute()
        finally:
            _exit(patches)

        assert result["react_rejection_count"] == 2

    @pytest.mark.asyncio
    async def test_approval_resets_counter(self, mock_registry, mock_prompt_builder):
        """After approval, react_rejection_count should reset to 0."""
        approved_payload = {
            "execution_plan": {
                "steps": [
                    {"capability": "channel_finding", "context_key": "ch", "task_objective": "Find"}
                ]
            },
            "react_messages": [{"role": "assistant", "content": "Action: channel_finding"}],
            "react_step_count": 2,
        }
        state = _create_react_node_state(react_rejection_count=2)
        patches = _common_patches(mock_registry, mock_prompt_builder)
        _enter(patches)
        try:
            with patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_approval_resume_data",
                return_value=(True, approved_payload),
            ):
                node = ReactiveOrchestratorNode()
                node._state = state
                result = await node.execute()
        finally:
            _exit(patches)

        assert result["react_rejection_count"] == 0

    @pytest.mark.asyncio
    async def test_normal_step_resets_counter(self, mock_registry, mock_prompt_builder):
        """Non-rejection path resets counter to 0."""
        llm_response = [
            _make_tool_call(
                "channel_finding", {"task_objective": "Find channels", "context_key": "channels"}
            )
        ]
        state = _create_react_node_state()
        patches = _common_patches(mock_registry, mock_prompt_builder)
        _enter(patches)
        try:
            with patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                return_value=llm_response,
            ):
                node = ReactiveOrchestratorNode()
                node._state = state
                result = await node.execute()
        finally:
            _exit(patches)

        assert result["react_rejection_count"] == 0


# =============================================================================
# TEST GAP COVERAGE — classify_error, retry_policy, format helpers, edge cases
# =============================================================================


class TestReactiveClassifyError:
    """Test all branches in ReactiveOrchestratorNode.classify_error."""

    def test_timeout_class_name(self):
        """Custom exception with 'timeout' in class name → RETRIABLE."""

        class LLMTimeoutError(Exception):
            pass

        exc = LLMTimeoutError("timed out")
        result = ReactiveOrchestratorNode.classify_error(exc, {})
        assert result.severity.value == "retriable"

    def test_connection_error(self):
        """ConnectionError → RETRIABLE."""
        result = ReactiveOrchestratorNode.classify_error(ConnectionError("refused"), {})
        assert result.severity.value == "retriable"

    def test_timeout_error(self):
        """TimeoutError → RETRIABLE."""
        result = ReactiveOrchestratorNode.classify_error(TimeoutError("deadline"), {})
        assert result.severity.value == "retriable"

    def test_validation_value_error(self):
        """ValueError with 'validation error' → RETRIABLE."""
        exc = ValueError("validation error for ExecutionPlan")
        result = ReactiveOrchestratorNode.classify_error(exc, {})
        assert result.severity.value == "retriable"

    def test_json_parsing_error(self):
        """ValueError with 'json parsing' → RETRIABLE."""
        exc = ValueError("json parsing failed")
        result = ReactiveOrchestratorNode.classify_error(exc, {})
        assert result.severity.value == "retriable"

    def test_pydantic_error(self):
        """ValueError with 'pydantic' → RETRIABLE."""
        exc = ValueError("pydantic validation issue")
        result = ReactiveOrchestratorNode.classify_error(exc, {})
        assert result.severity.value == "retriable"

    def test_reclassification_required(self):
        """ReclassificationRequiredError → RECLASSIFICATION."""
        exc = ReclassificationRequiredError("needs reclassification")
        result = ReactiveOrchestratorNode.classify_error(exc, {})
        assert result.severity.value == "reclassification"

    def test_invalid_context_key(self):
        """InvalidContextKeyError → RETRIABLE."""
        exc = InvalidContextKeyError("bad key ref")
        result = ReactiveOrchestratorNode.classify_error(exc, {})
        assert result.severity.value == "retriable"

    def test_generic_value_error(self):
        """ValueError without indicator keywords → CRITICAL."""
        exc = ValueError("bad config")
        result = ReactiveOrchestratorNode.classify_error(exc, {})
        assert result.severity.value == "critical"

    def test_type_error(self):
        """TypeError → CRITICAL."""
        exc = TypeError("wrong type")
        result = ReactiveOrchestratorNode.classify_error(exc, {})
        assert result.severity.value == "critical"

    def test_unknown_error(self):
        """RuntimeError (catch-all) → CRITICAL."""
        exc = RuntimeError("unexpected")
        result = ReactiveOrchestratorNode.classify_error(exc, {})
        assert result.severity.value == "critical"
        assert "Unknown" in result.user_message


class TestReactiveRetryPolicy:
    """Test get_retry_policy returns expected configuration."""

    def test_get_retry_policy(self):
        policy = ReactiveOrchestratorNode.get_retry_policy()
        assert policy["max_attempts"] == 4
        assert policy["delay_seconds"] == 2.0
        assert policy["backoff_factor"] == 2.0


class TestFormatExecutionHistory:
    """Test the module-level _format_execution_history helper."""

    def test_empty_results(self):
        result = _format_execution_history({"execution_step_results": {}})
        assert result == "No steps executed yet"

    def test_success_step(self):
        state = {
            "execution_step_results": {
                "beam_channels": {
                    "step_index": 0,
                    "capability": "channel_finding",
                    "success": True,
                    "task_objective": "Find channels",
                },
            },
        }
        result = _format_execution_history(state)
        assert "SUCCESS" in result
        assert "channel_finding" in result

    def test_failed_step(self):
        state = {
            "execution_step_results": {
                "values": {
                    "step_index": 0,
                    "capability": "channel_read",
                    "success": False,
                    "task_objective": "Read values",
                },
            },
        }
        result = _format_execution_history(state)
        assert "FAILED" in result
        assert "channel_read" in result


class TestHandleTextResponseErrors:
    """Test error paths in _handle_text_response via execute()."""

    @pytest.mark.asyncio
    async def test_invalid_json_raises(self, mock_registry, mock_prompt_builder):
        """Non-JSON text response raises ValueError."""
        state = _create_react_node_state()
        patches = _common_patches(mock_registry, mock_prompt_builder)
        _enter(patches)
        try:
            with patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                return_value="not valid json {",
            ):
                node = ReactiveOrchestratorNode()
                node._state = state
                with pytest.raises(ValueError, match="not valid ExecutionPlan JSON"):
                    await node.execute()
        finally:
            _exit(patches)

    @pytest.mark.asyncio
    async def test_empty_steps_raises(self, mock_registry, mock_prompt_builder):
        """Text response with empty steps list raises ValueError."""
        state = _create_react_node_state()
        patches = _common_patches(mock_registry, mock_prompt_builder)
        _enter(patches)
        try:
            with patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                return_value='{"steps": []}',
            ):
                node = ReactiveOrchestratorNode()
                node._state = state
                with pytest.raises(ValueError, match="no steps"):
                    await node.execute()
        finally:
            _exit(patches)

    @pytest.mark.asyncio
    async def test_multi_step_uses_first(self, mock_registry, mock_prompt_builder):
        """Text response with multiple steps uses the first step only."""
        text_response = json.dumps(
            {
                "steps": [
                    {
                        "capability": "channel_finding",
                        "task_objective": "Find channels",
                        "context_key": "beam_channels",
                    },
                    {
                        "capability": "channel_read",
                        "task_objective": "Read values",
                        "context_key": "beam_values",
                    },
                ]
            }
        )
        state = _create_react_node_state()
        patches = _common_patches(mock_registry, mock_prompt_builder)
        _enter(patches)
        try:
            with patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                return_value=text_response,
            ):
                node = ReactiveOrchestratorNode()
                node._state = state
                result = await node.execute()
        finally:
            _exit(patches)

        assert result["planning_execution_plan"]["steps"][0]["capability"] == "channel_finding"


class TestFormatObservationExtended:
    """Additional _format_observation branch coverage."""

    def test_dict_last_result_success(self):
        """Dict-typed execution_last_result with success=True."""
        state = {
            "control_error_info": None,
            "execution_last_result": {"success": True, "capability": "ch_find"},
            "execution_step_results": {},
        }
        result = _format_observation(state)
        assert "successfully" in result

    def test_step_results_fallback(self):
        """Falls back to execution_step_results when last_result is None."""
        state = {
            "control_error_info": None,
            "execution_last_result": None,
            "execution_step_results": {
                "k": {"capability": "x", "success": True, "step_index": 0},
            },
        }
        result = _format_observation(state)
        assert "Previous step" in result


class TestResolveInputsExtended:
    """Additional _resolve_inputs branch coverage."""

    def test_resolves_tuple_requires(self):
        """Capability with tuple-style requires resolves correctly."""
        registry = MagicMock()
        cap = MagicMock()
        cap.requires = [("CHANNEL_ADDRESSES", "one_or_more")]
        registry.get_capability.return_value = cap

        state = {
            "capability_context_data": {
                "CHANNEL_ADDRESSES": {"beam_channels": {"pvs": ["SR:C01"]}},
            },
        }
        with patch(
            "osprey.infrastructure.reactive_orchestrator_node.get_registry",
            return_value=registry,
        ):
            result = _resolve_inputs("channel_read", state)

        assert result == [{"CHANNEL_ADDRESSES": "beam_channels"}]


class TestNoValidCapabilities:
    """Test error when planning_active_capabilities lists nonexistent names."""

    @pytest.mark.asyncio
    async def test_no_valid_capabilities_raises(self, mock_prompt_builder):
        """ValueError raised when all capability names resolve to None."""
        registry = MagicMock()
        registry.get_capability.return_value = None

        state = _create_react_node_state(
            planning_active_capabilities=["nonexistent"],
        )
        patches = _common_patches(registry, mock_prompt_builder)
        _enter(patches)
        try:
            node = ReactiveOrchestratorNode()
            node._state = state
            with pytest.raises(ValueError, match="No valid capability instances"):
                await node.execute()
        finally:
            _exit(patches)


class TestMultipleCapabilityToolCalls:
    """Test that only the first capability tool call is used."""

    @pytest.mark.asyncio
    async def test_multiple_capability_calls_uses_first(self, mock_registry, mock_prompt_builder):
        """When LLM returns two capability tool calls, only the first is dispatched."""
        llm_response = [
            _make_tool_call(
                "channel_finding", {"task_objective": "Find channels", "context_key": "ch1"}
            ),
            _make_tool_call(
                "channel_read", {"task_objective": "Read values", "context_key": "ch2"}
            ),
        ]
        state = _create_react_node_state()
        patches = _common_patches(mock_registry, mock_prompt_builder)
        _enter(patches)
        try:
            with patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                return_value=llm_response,
            ):
                node = ReactiveOrchestratorNode()
                node._state = state
                result = await node.execute()
        finally:
            _exit(patches)

        assert result["planning_execution_plan"]["steps"][0]["capability"] == "channel_finding"
        # Verify channel_read was ignored (only 1 step in plan)
        assert len(result["planning_execution_plan"]["steps"]) == 1


class TestUnknownToolFeedback:
    """Test that unknown tool names feed back an error and the LLM retries."""

    @pytest.mark.asyncio
    async def test_unknown_tool_feeds_back_error_and_retries(
        self, mock_registry, mock_prompt_builder
    ):
        """Unknown tool triggers error feedback; next LLM call succeeds with valid capability."""
        unknown_call = [_make_tool_call("totally_unknown", {"task_objective": "Do something"})]
        valid_call = [
            _make_tool_call(
                "channel_finding",
                {"task_objective": "Find channels", "context_key": "beam_channels"},
            )
        ]
        mock_completion = MagicMock(side_effect=[unknown_call, valid_call])

        state = _create_react_node_state()
        patches = _common_patches(mock_registry, mock_prompt_builder)
        _enter(patches)
        try:
            with patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                mock_completion,
            ):
                node = ReactiveOrchestratorNode()
                node._state = state
                result = await node.execute()
        finally:
            _exit(patches)

        assert mock_completion.call_count == 2
        assert result["planning_execution_plan"]["steps"][0]["capability"] == "channel_finding"


class TestAutoExpandActiveCapabilities:
    """Test _build_missing_requirements_response auto-expands active capabilities."""

    def test_auto_expansion_adds_provider(self):
        """Provider names from missing tuples are added to planning_active_capabilities."""
        missing = [("CHANNEL_ADDRESSES", "channel_finding")]
        state = {"planning_active_capabilities": ["channel_write", "respond"]}
        logger = MagicMock()

        result = _build_missing_requirements_response(
            "channel_write", missing, [], 0, logger, state
        )

        assert "planning_active_capabilities" in result
        assert "channel_finding" in result["planning_active_capabilities"]
        assert "channel_write" in result["planning_active_capabilities"]
        assert "respond" in result["planning_active_capabilities"]

    def test_no_expansion_when_provider_already_active(self):
        """No duplicate when provider is already in the active set."""
        missing = [("CHANNEL_ADDRESSES", "channel_finding")]
        state = {"planning_active_capabilities": ["channel_write", "channel_finding", "respond"]}
        logger = MagicMock()

        result = _build_missing_requirements_response(
            "channel_write", missing, [], 0, logger, state
        )

        # No expansion needed — key should be absent from result
        assert "planning_active_capabilities" not in result

    def test_no_expansion_when_provider_is_none(self):
        """When provider is None, the active set is unchanged."""
        missing = [("UNKNOWN_TYPE", None)]
        state = {"planning_active_capabilities": ["channel_write", "respond"]}
        logger = MagicMock()

        result = _build_missing_requirements_response(
            "channel_write", missing, [], 0, logger, state
        )

        assert "planning_active_capabilities" not in result
