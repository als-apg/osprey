"""Integration tests for the reactive orchestration loop.

These tests verify the end-to-end reactive routing flow by testing
the interaction between router_conditional_edge and ReactiveOrchestratorNode
without compiling a full StateGraph.

All LLM calls are mocked to keep tests deterministic.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from osprey.base.errors import ErrorClassification, ErrorSeverity
from osprey.infrastructure.reactive_orchestrator_node import (
    ReactiveOrchestratorNode,
)
from osprey.infrastructure.router_node import router_conditional_edge
from tests.conftest import create_test_state


def _create_integration_state(**overrides):
    """Create a state suitable for integration testing.

    Wraps ``create_test_state()`` from conftest with reactive-loop defaults.
    """
    defaults = {
        "user_message": "Find beam current",
        "task_current_task": "Find beam current channels and read values",
        "planning_active_capabilities": ["channel_finding", "channel_read", "respond"],
        "planning_execution_plan": None,
        "planning_current_step_index": 0,
        "execution_start_time": 1.0,
        "control_plans_created_count": 0,
    }
    defaults.update(overrides)
    return create_test_state(**defaults)


@pytest.fixture
def mock_registry():
    """Registry with channel_finding, channel_read, respond capabilities."""
    registry = MagicMock()

    def make_cap(name, desc, provides, requires):
        cap = MagicMock()
        cap.name = name
        cap.description = desc
        cap.provides = provides
        cap.requires = requires
        cap.orchestrator_guide = None
        cap.direct_chat_enabled = False
        return cap

    caps = {
        "channel_finding": make_cap("channel_finding", "Find channels", ["CHANNEL_ADDRESSES"], []),
        "channel_read": make_cap(
            "channel_read", "Read channels", ["CHANNEL_VALUES"], ["CHANNEL_ADDRESSES"]
        ),
        "respond": make_cap("respond", "Respond to user", ["FINAL_RESPONSE"], []),
    }

    known_nodes = {
        "router",
        "task_extraction",
        "classifier",
        "orchestrator",
        "reactive_orchestrator",
        "error",
        "respond",
        "channel_finding",
        "channel_read",
    }

    registry.get_node.side_effect = lambda name: MagicMock() if name in known_nodes else None
    registry.get_capability.side_effect = lambda name: caps.get(name)
    registry.get_all_capabilities.return_value = list(caps.values())
    registry.get_stats.return_value = {"capability_names": list(caps.keys())}

    return registry


@pytest.fixture
def mock_prompt_builder():
    """Mock prompt builder for reactive orchestrator."""
    builder = MagicMock()
    builder.get_reactive_instructions.return_value = (
        "You are an expert execution planner.\n\n"
        "Each step must follow the PlannedStep structure.\n\n"
        "REACTIVE MODE\n\n"
        "# CAPABILITY PLANNING GUIDELINES"
    )
    builder.format_reactive_response_context.return_value = "Reactive context summary"

    response_builder = MagicMock()
    response_builder.get_system_instructions.return_value = "You are a response generator."

    provider = MagicMock()
    provider.get_orchestrator_prompt_builder.return_value = builder
    provider.get_response_generation_prompt_builder.return_value = response_builder
    return provider


def _react_patches(mock_registry, mock_prompt_builder):
    """Common patches for reactive mode."""
    return {
        "config": patch(
            "osprey.infrastructure.router_node.get_config_value",
            side_effect=lambda path, default=None: (
                "react"
                if path == "execution_control.agent_control.orchestration_mode"
                else (100 if path == "execution_control.limits.graph_recursion_limit" else default)
            ),
        ),
        "router_registry": patch(
            "osprey.infrastructure.router_node.get_registry",
            return_value=mock_registry,
        ),
        "node_registry": patch(
            "osprey.infrastructure.reactive_orchestrator_node.get_registry",
            return_value=mock_registry,
        ),
        "validate_registry": patch(
            "osprey.infrastructure.orchestration_node.get_registry",
            return_value=mock_registry,
        ),
        "model_config": patch(
            "osprey.infrastructure.reactive_orchestrator_node.get_model_config",
            return_value={"provider": "test", "model_id": "test"},
        ),
        "api_context": patch(
            "osprey.models.set_api_call_context",
        ),
        "prompts": patch(
            "osprey.infrastructure.reactive_orchestrator_node.get_framework_prompts",
            return_value=mock_prompt_builder,
        ),
        "lw_ctx": patch(
            "osprey.capabilities.context_tools.create_context_tools",
            return_value=[],
        ),
        "lw_state": patch(
            "osprey.capabilities.state_tools.create_state_tools",
            return_value=[],
        ),
        "interface_ctx": patch(
            "osprey.infrastructure.reactive_orchestrator_node.get_interface_context",
            return_value={"interface": "cli"},
        ),
    }


class TestTwoStepChain:
    """Test a two-step reactive chain: find channels -> read values -> respond."""

    @pytest.mark.asyncio
    async def test_two_step_chain(self, mock_registry, mock_prompt_builder):
        """Simulate a full two-step reactive chain."""
        state = _create_integration_state()
        patches = _react_patches(mock_registry, mock_prompt_builder)

        # --- Step 1: Router routes to reactive_orchestrator ---
        with patches["config"], patches["router_registry"]:
            route1 = router_conditional_edge(state)
        assert route1 == "reactive_orchestrator"

        # --- Step 2: Reactive orchestrator decides channel_finding ---
        llm_response1 = [
            {
                "id": "call_channel_finding",
                "type": "function",
                "function": {
                    "name": "channel_finding",
                    "arguments": json.dumps(
                        {
                            "task_objective": "Find beam current channels",
                            "context_key": "beam_channels",
                            "expected_output": "CHANNEL_ADDRESSES",
                        }
                    ),
                },
            }
        ]

        with (
            patches["node_registry"],
            patches["model_config"],
            patches["api_context"],
            patches["validate_registry"],
            patches["prompts"],
            patches["lw_ctx"],
            patches["lw_state"],
            patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                return_value=llm_response1,
            ),
        ):
            node = ReactiveOrchestratorNode()
            node._state = state
            updates1 = await node.execute()

        state.update(updates1)

        # Verify step was created
        assert state["planning_execution_plan"]["steps"][0]["capability"] == "channel_finding"
        assert state["react_step_count"] == 1

        # --- Step 3: Router routes to channel_finding capability ---
        with patches["config"], patches["router_registry"]:
            route2 = router_conditional_edge(state)
        assert route2 == "channel_finding"

        # --- Simulate capability execution ---
        state["planning_current_step_index"] = 1  # Step completed
        state["execution_last_result"] = MagicMock(success=True, capability="channel_finding")
        state["capability_context_data"] = {
            "CHANNEL_ADDRESSES": {"beam_channels": {"pvs": ["SR:C01:BEAM"]}}
        }
        state["control_has_error"] = False

        # --- Step 4: Router sees step completed, routes back to reactive_orchestrator ---
        with patches["config"], patches["router_registry"]:
            route3 = router_conditional_edge(state)
        assert route3 == "reactive_orchestrator"

        # --- Step 5: Reactive orchestrator decides respond ---
        # _generate_direct_response makes TWO LLM calls:
        # 1. The orchestrator loop call that returns the "respond" tool call
        # 2. The response generation call inside _generate_direct_response
        llm_response2 = [
            {
                "id": "call_respond",
                "type": "function",
                "function": {
                    "name": "respond",
                    "arguments": json.dumps(
                        {
                            "task_objective": "Report the beam current channels",
                            "context_key": "user_response",
                            "expected_output": "FINAL_RESPONSE",
                        }
                    ),
                },
            }
        ]
        llm_responses_iter = iter([llm_response2, "The beam current channels are SR:C01:BEAM."])

        with (
            patches["node_registry"],
            patches["model_config"],
            patches["api_context"],
            patches["validate_registry"],
            patches["prompts"],
            patches["lw_ctx"],
            patches["lw_state"],
            patches["interface_ctx"],
            patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                side_effect=lambda *a, **kw: next(llm_responses_iter),
            ),
        ):
            node = ReactiveOrchestratorNode()
            node._state = state
            updates2 = await node.execute()

        state.update(updates2)

        # _generate_direct_response returns react_response_generated, not an execution plan
        assert state["react_response_generated"] is True
        assert state["react_step_count"] == 2


class TestErrorRecovery:
    """Test error recovery in reactive loop."""

    @pytest.mark.asyncio
    async def test_error_routes_back_to_orchestrator(self, mock_registry, mock_prompt_builder):
        """A CRITICAL error routes back to reactive_orchestrator for re-evaluation."""
        classification = ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message="Capability failed",
            metadata={},
        )

        state = _create_integration_state(
            control_has_error=True,
            control_error_info={
                "classification": classification,
                "capability_name": "channel_finding",
                "retry_policy": {},
            },
        )
        patches = _react_patches(mock_registry, mock_prompt_builder)

        with patches["config"], patches["router_registry"]:
            route = router_conditional_edge(state)
        assert route == "reactive_orchestrator"


class TestMaxIterationsGuard:
    """Test max iterations safety guard."""

    def test_max_iterations_terminates(self, mock_registry, mock_prompt_builder):
        """Reaching max iterations routes to error."""
        state = _create_integration_state(react_step_count=100)
        patches = _react_patches(mock_registry, mock_prompt_builder)

        with patches["config"], patches["router_registry"]:
            route = router_conditional_edge(state)
        assert route == "error"


class TestPlanFirstRegression:
    """Ensure plan-first mode works unchanged."""

    def test_plan_first_ignores_react_fields(self, mock_registry, mock_prompt_builder):
        """Plan-first mode doesn't use react_* fields."""
        state = _create_integration_state(
            react_step_count=100,
        )

        with (
            patch(
                "osprey.infrastructure.router_node.get_config_value",
                side_effect=lambda path, default=None: (
                    "plan_first"
                    if path == "execution_control.agent_control.orchestration_mode"
                    else default
                ),
            ),
            patch(
                "osprey.infrastructure.router_node.get_registry",
                return_value=mock_registry,
            ),
        ):
            route = router_conditional_edge(state)
            # plan-first with task + capabilities but no plan -> orchestrator
            assert route == "orchestrator"


class TestThreeStepChain:
    """Test a three-step reactive chain verifying observation persistence across all iterations."""

    @pytest.mark.asyncio
    async def test_three_step_observation_accumulation(self, mock_registry, mock_prompt_builder):
        """channel_finding -> channel_read -> respond preserves all observations in react_messages."""
        state = _create_integration_state()
        patches = _react_patches(mock_registry, mock_prompt_builder)

        # --- Step 1: Router -> reactive_orchestrator ---
        with patches["config"], patches["router_registry"]:
            assert router_conditional_edge(state) == "reactive_orchestrator"

        # --- Step 2: Orchestrator decides channel_finding ---
        llm_step1 = [
            {
                "id": "call_channel_finding",
                "type": "function",
                "function": {
                    "name": "channel_finding",
                    "arguments": json.dumps(
                        {
                            "task_objective": "Find beam current channels",
                            "context_key": "beam_channels",
                            "expected_output": "CHANNEL_ADDRESSES",
                        }
                    ),
                },
            }
        ]

        with (
            patches["node_registry"],
            patches["model_config"],
            patches["api_context"],
            patches["validate_registry"],
            patches["prompts"],
            patches["lw_ctx"],
            patches["lw_state"],
            patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                return_value=llm_step1,
            ),
        ):
            node = ReactiveOrchestratorNode()
            node._state = state
            updates1 = await node.execute()

        state.update(updates1)
        assert state["react_step_count"] == 1
        # 1 assistant message so far
        assert len(state["react_messages"]) == 1

        # --- Simulate channel_finding execution ---
        with patches["config"], patches["router_registry"]:
            assert router_conditional_edge(state) == "channel_finding"

        state["planning_current_step_index"] = 1
        state["execution_last_result"] = MagicMock(success=True, capability="channel_finding")
        state["capability_context_data"] = {
            "CHANNEL_ADDRESSES": {"beam_channels": {"pvs": ["SR:C01:BEAM"]}}
        }

        # --- Step 3: Router -> reactive_orchestrator (observation from channel_finding) ---
        with patches["config"], patches["router_registry"]:
            assert router_conditional_edge(state) == "reactive_orchestrator"

        # --- Step 4: Orchestrator decides channel_read ---
        llm_step2 = [
            {
                "id": "call_channel_read",
                "type": "function",
                "function": {
                    "name": "channel_read",
                    "arguments": json.dumps(
                        {
                            "task_objective": "Read beam current values",
                            "context_key": "beam_values",
                            "expected_output": "CHANNEL_VALUES",
                        }
                    ),
                },
            }
        ]

        with (
            patches["node_registry"],
            patches["model_config"],
            patches["api_context"],
            patches["validate_registry"],
            patches["prompts"],
            patches["lw_ctx"],
            patches["lw_state"],
            patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                return_value=llm_step2,
            ),
        ):
            node = ReactiveOrchestratorNode()
            node._state = state
            updates2 = await node.execute()

        state.update(updates2)
        assert state["react_step_count"] == 2
        # 3 messages: assistant(step1) + observation(step1 result) + assistant(step2)
        assert len(state["react_messages"]) == 3
        assert state["react_messages"][0]["role"] == "assistant"
        assert state["react_messages"][1]["role"] == "observation"
        assert state["react_messages"][2]["role"] == "assistant"
        # Observation should reference channel_finding success
        assert "channel_finding" in state["react_messages"][1]["content"]

        # --- Simulate channel_read execution ---
        with patches["config"], patches["router_registry"]:
            assert router_conditional_edge(state) == "channel_read"

        state["planning_current_step_index"] = 1
        state["execution_last_result"] = MagicMock(success=True, capability="channel_read")
        state["capability_context_data"]["CHANNEL_VALUES"] = {"beam_values": {"values": [42.0]}}
        state["execution_step_results"]["beam_channels"] = {
            "step_index": 0,
            "capability": "channel_finding",
            "success": True,
            "task_objective": "Find beam current channels",
        }

        # --- Step 5: Router -> reactive_orchestrator (observation from channel_read) ---
        with patches["config"], patches["router_registry"]:
            assert router_conditional_edge(state) == "reactive_orchestrator"

        # --- Step 6: Orchestrator decides respond ---
        # _generate_direct_response makes TWO LLM calls
        llm_step3 = [
            {
                "id": "call_respond",
                "type": "function",
                "function": {
                    "name": "respond",
                    "arguments": json.dumps(
                        {
                            "task_objective": "Report the beam current values",
                            "context_key": "user_response",
                            "expected_output": "FINAL_RESPONSE",
                        }
                    ),
                },
            }
        ]
        llm_responses_iter = iter([llm_step3, "The beam current value is 42.0 mA."])

        with (
            patches["node_registry"],
            patches["model_config"],
            patches["api_context"],
            patches["validate_registry"],
            patches["prompts"],
            patches["lw_ctx"],
            patches["lw_state"],
            patches["interface_ctx"],
            patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                side_effect=lambda *a, **kw: next(llm_responses_iter),
            ),
        ):
            node = ReactiveOrchestratorNode()
            node._state = state
            updates3 = await node.execute()

        state.update(updates3)
        assert state["react_step_count"] == 3
        assert state["react_response_generated"] is True

        # 5 messages: asst(1) + obs(1) + asst(2) + obs(2) + asst(respond)
        assert len(state["react_messages"]) == 5
        roles = [m["role"] for m in state["react_messages"]]
        assert roles == ["assistant", "observation", "assistant", "observation", "assistant"]
        # Second observation should reference channel_read
        assert "channel_read" in state["react_messages"][3]["content"]


class TestErrorThenRespondRecovery:
    """Test that after a capability error the orchestrator can respond gracefully."""

    @pytest.mark.asyncio
    async def test_error_then_respond_clears_error_state(self, mock_registry, mock_prompt_builder):
        """After a capability error, the reactive orchestrator can decide to respond
        and properly clears error state so the router routes to respond (not error handling)."""
        patches = _react_patches(mock_registry, mock_prompt_builder)

        # Start with error state after channel_read failure
        classification = ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message="Channel read timed out",
            metadata={},
        )
        state = _create_integration_state(
            react_step_count=2,
            react_messages=[
                {
                    "role": "assistant",
                    "content": "Action: channel_finding\nObjective: Find channels",
                },
                {
                    "role": "observation",
                    "content": "Step completed successfully (capability: channel_finding)",
                },
                {"role": "assistant", "content": "Action: channel_read\nObjective: Read values"},
            ],
            control_has_error=True,
            control_error_info={
                "classification": classification,
                "capability_name": "channel_read",
                "retry_policy": {},
            },
            control_last_error="Channel read timed out",
            control_retry_count=1,
            control_current_step_retry_count=1,
        )

        # Router should route back to reactive_orchestrator (CRITICAL error in react mode)
        with patches["config"], patches["router_registry"]:
            route = router_conditional_edge(state)
        assert route == "reactive_orchestrator"

        # Orchestrator decides to respond with an error explanation
        # _generate_direct_response makes TWO LLM calls
        llm_respond = [
            {
                "id": "call_respond",
                "type": "function",
                "function": {
                    "name": "respond",
                    "arguments": json.dumps(
                        {
                            "task_objective": "Inform user that channel read failed and provide partial results",
                            "context_key": "error_response",
                            "expected_output": "FINAL_RESPONSE",
                        }
                    ),
                },
            }
        ]
        llm_responses_iter = iter([llm_respond, "Channel read failed due to timeout."])

        with (
            patches["node_registry"],
            patches["model_config"],
            patches["api_context"],
            patches["validate_registry"],
            patches["prompts"],
            patches["lw_ctx"],
            patches["lw_state"],
            patches["interface_ctx"],
            patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                side_effect=lambda *a, **kw: next(llm_responses_iter),
            ),
        ):
            node = ReactiveOrchestratorNode()
            node._state = state
            result = await node.execute()

        state.update(result)

        # Error state must be cleared
        assert state["control_has_error"] is False
        assert state["control_error_info"] is None
        assert state["control_last_error"] is None
        assert state["control_retry_count"] == 0
        assert state["control_current_step_retry_count"] == 0
        # _generate_direct_response returns react_response_generated, not execution plan
        assert state["react_response_generated"] is True

        # The error observation should be persisted in react_messages
        observations = [m for m in state["react_messages"] if m["role"] == "observation"]
        error_obs = [o for o in observations if "ERROR" in o["content"]]
        assert len(error_obs) == 1
        assert "channel_read" in error_obs[0]["content"]


class TestReactiveInputResolution:
    """Test that inputs are correctly resolved in the reactive loop."""

    @pytest.mark.asyncio
    async def test_channel_read_gets_channel_addresses_input(
        self, mock_registry, mock_prompt_builder
    ):
        """channel_read step gets CHANNEL_ADDRESSES input auto-resolved."""
        llm_response = [
            {
                "id": "call_channel_read",
                "type": "function",
                "function": {
                    "name": "channel_read",
                    "arguments": json.dumps(
                        {
                            "task_objective": "Read beam current values",
                            "context_key": "beam_values",
                            "expected_output": "CHANNEL_VALUES",
                        }
                    ),
                },
            }
        ]

        state = _create_integration_state(
            capability_context_data={"CHANNEL_ADDRESSES": {"beam_channels": {"pvs": ["SR:C01"]}}},
            react_step_count=1,
        )
        patches = _react_patches(mock_registry, mock_prompt_builder)

        with (
            patches["node_registry"],
            patches["model_config"],
            patches["api_context"],
            patches["validate_registry"],
            patches["prompts"],
            patches["lw_ctx"],
            patches["lw_state"],
            patch(
                "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
                return_value=llm_response,
            ),
        ):
            node = ReactiveOrchestratorNode()
            node._state = state
            result = await node.execute()

        step = result["planning_execution_plan"]["steps"][0]
        assert step["inputs"] == [{"CHANNEL_ADDRESSES": "beam_channels"}]
