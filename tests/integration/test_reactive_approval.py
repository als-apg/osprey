"""Integration tests for reactive orchestration with human approval workflows.

These tests validate that LangGraph interrupt/resume mechanics work correctly
within reactive (ReAct) orchestration chains.  They compile a **minimal**
``StateGraph`` (router, reactive_orchestrator, a mock approval capability,
respond, error) with a ``MemorySaver`` checkpointer so that real checkpointing,
interrupts, and ``Command(update=…)`` resume flows are exercised.

All LLM calls are mocked for determinism.

Key scenarios covered:
    - Core flow: reactive step → approval interrupt → approve → continue → respond
    - Rejection: reactive step → approval interrupt → reject → graceful respond
    - Error recovery: error → orchestrator picks approval capability → approve
    - Multiple approvals: two interrupt/resume cycles in one chain
    - State preservation: react_messages / react_step_count survive interrupt/resume
    - Router signal: react_route_to is read-only in router; orchestrator manages lifecycle
"""

from __future__ import annotations

import copy
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from osprey.approval.approval_system import (
    create_approval_type,
    create_code_approval_interrupt,
    get_approval_resume_data,
)
from osprey.base.errors import ErrorClassification, ErrorSeverity
from osprey.infrastructure.router_node import router_conditional_edge
from osprey.state import AgentState
from tests.conftest import create_test_state

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_reactive_approval_state(**overrides) -> AgentState:
    """Create an ``AgentState`` pre-configured for reactive-approval tests."""
    defaults: dict[str, Any] = {
        "user_message": "Write a value to the control system",
        "task_current_task": "Write a value to the EPICS PV",
        "planning_active_capabilities": ["mock_approval_cap", "respond"],
        "planning_execution_plan": None,
        "planning_current_step_index": 0,
        "execution_start_time": 1.0,
        "control_plans_created_count": 0,
        # Reactive fields start clean
        "react_messages": [],
        "react_step_count": 0,
        # Approval fields start clean
        "approval_approved": None,
        "approved_payload": None,
        # session_state needed by router
        "session_state": {},
    }
    defaults.update(overrides)
    return create_test_state(**defaults)


# ---------------------------------------------------------------------------
# Mock capability that calls interrupt()
# ---------------------------------------------------------------------------

_MOCK_CAP_APPROVAL_TYPE = create_approval_type("python_executor")


async def _mock_approval_capability_node(state: AgentState, **kwargs) -> dict[str, Any]:
    """Mock capability node that requires approval before 'executing'.

    Follows the same resume pattern used by real capabilities
    (python, channel_write, memory):
        1. Check ``get_approval_resume_data`` for a prior decision.
        2. If no decision yet → call ``interrupt()``.
        3. If approved → return success.
        4. If rejected → raise ``ValueError`` (handled by decorator/error system).
    """
    has_resume, payload = get_approval_resume_data(state, _MOCK_CAP_APPROVAL_TYPE)

    if has_resume:
        if payload:
            # Approved – simulate successful execution
            return {
                "execution_last_result": {
                    "capability": "mock_approval_cap",
                    "success": True,
                },
                "planning_current_step_index": state.get("planning_current_step_index", 0) + 1,
                "control_has_error": False,
                "control_error_info": None,
                "control_retry_count": 0,
                # Clear approval state after consuming
                "approval_approved": None,
                "approved_payload": None,
            }
        else:
            # Rejected – return error state for router
            return {
                "execution_last_result": {
                    "capability": "mock_approval_cap",
                    "success": False,
                    "error": "Operation rejected by user",
                },
                "planning_current_step_index": state.get("planning_current_step_index", 0) + 1,
                "control_has_error": True,
                "control_error_info": {
                    "capability_name": "mock_approval_cap",
                    "classification": ErrorClassification(
                        severity=ErrorSeverity.CRITICAL,
                        user_message="User rejected the operation",
                        metadata={},
                    ),
                    "retry_policy": {},
                },
                # Clear approval state after consuming
                "approval_approved": None,
                "approved_payload": None,
            }

    # No prior decision → create interrupt for approval
    interrupt_data = create_code_approval_interrupt(
        code="caput('SR:C01:BEAM', 42.0)",
        analysis_details={"safety_level": "high", "approval_reasoning": "Control write"},
        execution_mode="write",
        safety_concerns=["EPICS write operation"],
        step_objective="Write to EPICS PV",
    )
    interrupt(interrupt_data)

    # interrupt() never returns – but if it somehow does, raise
    raise RuntimeError("interrupt() should have paused execution")


# Give the callable attributes the decorator normally sets
_mock_approval_capability_node.name = "mock_approval_cap"
_mock_approval_capability_node.capability_name = "mock_approval_cap"


async def _mock_respond_node(state: AgentState, **kwargs) -> dict[str, Any]:
    """Minimal respond stub that returns a final response."""
    return {
        "messages": state.get("messages", []),
        "execution_last_result": {
            "capability": "respond",
            "success": True,
        },
    }


_mock_respond_node.name = "respond"


async def _mock_error_node(state: AgentState, **kwargs) -> dict[str, Any]:
    """Minimal error stub."""
    return {
        "execution_last_result": {
            "capability": "error",
            "success": False,
        },
    }


_mock_error_node.name = "error"


# ---------------------------------------------------------------------------
# Mock registry for router lookups
# ---------------------------------------------------------------------------


def _make_mock_registry(extra_caps: dict[str, MagicMock] | None = None):
    """Build a mock registry that knows about our minimal node set."""
    registry = MagicMock()

    def make_cap(name, provides=None, requires=None):
        cap = MagicMock()
        cap.name = name
        cap.description = f"Mock {name}"
        cap.provides = provides or []
        cap.requires = requires or []
        cap.orchestrator_guide = None
        cap.direct_chat_enabled = False
        return cap

    caps: dict[str, MagicMock] = {
        "mock_approval_cap": make_cap("mock_approval_cap", provides=["MOCK_RESULT"]),
        "respond": make_cap("respond", provides=["FINAL_RESPONSE"]),
    }
    if extra_caps:
        caps.update(extra_caps)

    known_nodes = {
        "router",
        "reactive_orchestrator",
        "mock_approval_cap",
        "respond",
        "error",
    } | set(extra_caps.keys() if extra_caps else [])

    registry.get_node.side_effect = lambda n: MagicMock() if n in known_nodes else None
    registry.get_capability.side_effect = lambda n: caps.get(n)
    registry.get_all_capabilities.return_value = list(caps.values())
    registry.get_stats.return_value = {"capability_names": list(caps.keys())}

    return registry


# ---------------------------------------------------------------------------
# Common patches
# ---------------------------------------------------------------------------


def _common_patches(mock_registry):
    """Return dict of context-manager patches for reactive-approval tests."""
    mock_prompt_builder = MagicMock()
    mock_prompt_builder.get_reactive_instructions.return_value = (
        "You are an expert execution planner.\n\n"
        "REACTIVE MODE: decide the NEXT SINGLE ACTION.\n\n"
        "# CAPABILITY PLANNING GUIDELINES"
    )
    mock_prompt_builder.format_reactive_response_context.return_value = "Reactive context summary"

    mock_response_builder = MagicMock()
    mock_response_builder.get_system_instructions.return_value = "You are a response generator."

    mock_prompt_provider = MagicMock()
    mock_prompt_provider.get_orchestrator_prompt_builder.return_value = mock_prompt_builder
    mock_prompt_provider.get_response_generation_prompt_builder.return_value = mock_response_builder

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
            return_value=mock_prompt_provider,
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


# ---------------------------------------------------------------------------
# Graph factory – builds the minimal compiled StateGraph
# ---------------------------------------------------------------------------


def _build_test_graph(
    mock_registry,
    llm_responses: list[list[dict[str, Any]]],
    extra_nodes: dict[str, Any] | None = None,
):
    """Compile a minimal reactive-approval graph with ``MemorySaver``.

    Parameters
    ----------
    mock_registry
        Mock registry returned by ``_make_mock_registry``.
    llm_responses
        Ordered list of tool-call lists; each call to
        ``get_chat_completion`` pops the next one.
    extra_nodes
        Optional ``{name: async_callable}`` for additional capability nodes.

    Returns
    -------
    tuple[CompiledGraph, dict, list[patch]]
        ``(compiled_graph, config, active_patches)`` where *active_patches*
        are context managers that **must** be entered before invoking the
        graph (they mock the LLM and registry at the module level).
    """
    from osprey.infrastructure.reactive_orchestrator_node import (
        ReactiveOrchestratorNode,
    )

    patches = _common_patches(mock_registry)

    # Build a sequential LLM mock – each call returns the next response
    llm_iter = iter(llm_responses)

    def _next_llm_response(*args, **kwargs):
        return next(llm_iter)

    llm_patch = patch(
        "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
        side_effect=_next_llm_response,
    )

    # Minimal StateGraph
    workflow = StateGraph(AgentState)

    # Router is a plain function (conditional edge), not a node.
    # We add a thin passthrough node at the entry so we can set_entry_point.
    async def _router_passthrough(state: AgentState, **kw) -> dict[str, Any]:
        return {}

    workflow.add_node("router", _router_passthrough)
    workflow.add_node("reactive_orchestrator", ReactiveOrchestratorNode.langgraph_node)
    workflow.add_node("mock_approval_cap", _mock_approval_capability_node)
    workflow.add_node("respond", _mock_respond_node)
    workflow.add_node("error", _mock_error_node)

    if extra_nodes:
        for name, func in extra_nodes.items():
            workflow.add_node(name, func)

    # Routing
    all_node_names = ["router", "reactive_orchestrator", "mock_approval_cap", "respond", "error"]
    if extra_nodes:
        all_node_names.extend(extra_nodes.keys())

    routing_map = {n: n for n in all_node_names}
    routing_map["END"] = END

    workflow.set_entry_point("router")

    def _conditional_edge(state):
        # Enter all module-level patches so router_conditional_edge sees the
        # mocked config / registry.
        return router_conditional_edge(state)

    workflow.add_conditional_edges("router", _conditional_edge, routing_map)
    workflow.add_edge("respond", END)
    workflow.add_edge("error", END)
    for name in all_node_names:
        if name not in ("router", "respond", "error"):
            workflow.add_edge(name, "router")

    checkpointer = MemorySaver()
    compiled = workflow.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "test-reactive-approval"}}

    all_patches = list(patches.values()) + [llm_patch]
    return compiled, config, all_patches


def _enter_patches(patches):
    """Enter a list of patch context managers and return the exit list."""
    entered = []
    for p in patches:
        p.start()
        entered.append(p)
    return entered


def _exit_patches(patches):
    for p in patches:
        p.stop()


# ===================================================================
# TEST CLASSES
# ===================================================================


class TestReactiveApprovalCoreFlow:
    """Core flow: reactive_orchestrator → mock_approval_cap (interrupt) → approve → respond."""

    @pytest.mark.asyncio
    async def test_approve_and_respond(self):
        """Full reactive chain with approval succeeds and reaches respond."""
        mock_registry = _make_mock_registry()

        # LLM response sequence:
        # 1. Orchestrator picks mock_approval_cap
        # 2. After approval + success, orchestrator picks respond
        llm_responses = [
            [
                {
                    "id": "call_mock_approval_cap",
                    "type": "function",
                    "function": {
                        "name": "mock_approval_cap",
                        "arguments": json.dumps(
                            {
                                "task_objective": "Write to EPICS PV",
                                "context_key": "write_result",
                                "expected_output": "MOCK_RESULT",
                            }
                        ),
                    },
                }
            ],
            [
                {
                    "id": "call_respond",
                    "type": "function",
                    "function": {
                        "name": "respond",
                        "arguments": json.dumps(
                            {
                                "task_objective": "Confirm write succeeded",
                                "context_key": "user_response",
                                "expected_output": "FINAL_RESPONSE",
                            }
                        ),
                    },
                }
            ],
            "The write operation completed successfully.",  # response generation
        ]

        graph, config, patches = _build_test_graph(mock_registry, llm_responses)
        started = _enter_patches(patches)
        try:
            state = _create_reactive_approval_state()

            # Phase 1: Execute until interrupt
            await graph.ainvoke(state, config)

            # Should have hit the approval interrupt
            graph_state = graph.get_state(config)
            assert graph_state.interrupts, "Expected approval interrupt"

            # react_messages should already have the orchestrator's action
            result_values = graph_state.values
            assert result_values["react_step_count"] == 1
            assert len(result_values["react_messages"]) >= 1

            # Phase 2: Approve and resume
            resume_payload = graph_state.interrupts[-1].value.get("resume_payload", {})
            resume_cmd = Command(
                update={
                    "approval_approved": True,
                    "approved_payload": resume_payload,
                }
            )
            await graph.ainvoke(resume_cmd, config)

            # Should generate response directly and terminate
            final_state = graph.get_state(config)
            assert final_state.values.get("react_response_generated") is True

            # react_messages should be preserved through the cycle
            assert final_state.values["react_step_count"] == 2

        finally:
            _exit_patches(started)


class TestReactiveApprovalRejection:
    """Rejection: reactive_orchestrator → mock_approval_cap → reject → graceful respond."""

    @pytest.mark.asyncio
    async def test_reject_leads_to_respond(self):
        """User rejects approval; orchestrator recovers and responds."""
        mock_registry = _make_mock_registry()

        llm_responses = [
            # 1. Orchestrator picks mock_approval_cap
            [
                {
                    "id": "call_mock_approval_cap",
                    "type": "function",
                    "function": {
                        "name": "mock_approval_cap",
                        "arguments": json.dumps(
                            {
                                "task_objective": "Write to EPICS PV",
                                "context_key": "write_result",
                                "expected_output": "MOCK_RESULT",
                            }
                        ),
                    },
                }
            ],
            # 2. After rejection error → orchestrator decides to respond
            [
                {
                    "id": "call_respond",
                    "type": "function",
                    "function": {
                        "name": "respond",
                        "arguments": json.dumps(
                            {
                                "task_objective": "Inform user the operation was cancelled",
                                "context_key": "user_response",
                                "expected_output": "FINAL_RESPONSE",
                            }
                        ),
                    },
                }
            ],
            "The operation was cancelled as requested.",  # response generation
        ]

        graph, config, patches = _build_test_graph(mock_registry, llm_responses)
        started = _enter_patches(patches)
        try:
            state = _create_reactive_approval_state()

            # Phase 1: Execute until interrupt
            await graph.ainvoke(state, config)

            graph_state = graph.get_state(config)
            assert graph_state.interrupts, "Expected approval interrupt"

            # Phase 2: Reject
            resume_cmd = Command(
                update={
                    "approval_approved": False,
                    "approved_payload": None,
                }
            )
            await graph.ainvoke(resume_cmd, config)

            # Should generate response directly (orchestrator recovers from rejection error)
            final_state = graph.get_state(config)
            assert final_state.values.get("react_response_generated") is True

        finally:
            _exit_patches(started)


class TestReactiveApprovalErrorRecovery:
    """Error → orchestrator picks approval capability → approve → continue → respond."""

    @pytest.mark.asyncio
    async def test_error_recovery_with_approval(self):
        """After a prior error, orchestrator uses approval-requiring cap successfully."""
        mock_registry = _make_mock_registry()

        from osprey.base.errors import ErrorClassification, ErrorSeverity

        error_classification = ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message="Previous capability failed",
            metadata={},
        )

        llm_responses = [
            # 1. Orchestrator (called after error) picks mock_approval_cap
            [
                {
                    "id": "call_mock_approval_cap",
                    "type": "function",
                    "function": {
                        "name": "mock_approval_cap",
                        "arguments": json.dumps(
                            {
                                "task_objective": "Write to EPICS PV (recovery)",
                                "context_key": "recovery_write",
                                "expected_output": "MOCK_RESULT",
                            }
                        ),
                    },
                }
            ],
            # 2. After approval success → respond
            [
                {
                    "id": "call_respond",
                    "type": "function",
                    "function": {
                        "name": "respond",
                        "arguments": json.dumps(
                            {
                                "task_objective": "Report recovery succeeded",
                                "context_key": "user_response",
                                "expected_output": "FINAL_RESPONSE",
                            }
                        ),
                    },
                }
            ],
            "Recovery succeeded. The write was completed.",  # response generation
        ]

        graph, config, patches = _build_test_graph(mock_registry, llm_responses)
        started = _enter_patches(patches)
        try:
            # Start with error state (simulating a prior capability failure)
            state = _create_reactive_approval_state(
                control_has_error=True,
                control_error_info={
                    "classification": error_classification,
                    "capability_name": "some_capability",
                    "retry_policy": {},
                },
                control_last_error="Previous capability failed",
                react_step_count=1,
                react_messages=[
                    {
                        "role": "assistant",
                        "content": "Action: some_capability\nObjective: Do something",
                    },
                    {
                        "role": "observation",
                        "content": "ERROR in some_capability: Previous capability failed",
                    },
                ],
            )

            # Phase 1: Execute until interrupt
            await graph.ainvoke(state, config)

            graph_state = graph.get_state(config)
            assert graph_state.interrupts, "Expected approval interrupt"

            # Phase 2: Approve
            resume_payload = graph_state.interrupts[-1].value.get("resume_payload", {})
            resume_cmd = Command(
                update={
                    "approval_approved": True,
                    "approved_payload": resume_payload,
                }
            )
            await graph.ainvoke(resume_cmd, config)

            final_state = graph.get_state(config)
            assert final_state.values.get("react_response_generated") is True

            # Error state from the initial failure should have been cleared
            # (reactive orchestrator clears error state on each step)
            assert final_state.values.get("control_has_error") is False

        finally:
            _exit_patches(started)


class TestReactiveMultipleApprovals:
    """Two approval interrupts in a single reactive chain."""

    @pytest.mark.asyncio
    async def test_two_approvals_in_chain(self):
        """Two sequential capabilities each require approval; both approved."""

        # We need a second approval capability
        async def _mock_approval_cap2_node(state: AgentState, **kw) -> dict[str, Any]:
            """Second mock approval capability."""
            has_resume, payload = get_approval_resume_data(state, "mock_approval_cap2")
            if has_resume:
                if payload:
                    return {
                        "execution_last_result": {
                            "capability": "mock_approval_cap2",
                            "success": True,
                        },
                        "planning_current_step_index": state.get("planning_current_step_index", 0)
                        + 1,
                        "control_has_error": False,
                        "control_error_info": None,
                        "control_retry_count": 0,
                        "approval_approved": None,
                        "approved_payload": None,
                    }
                else:
                    raise ValueError("Rejected")
            interrupt_data = create_code_approval_interrupt(
                code="caput('SR:C01:BEAM2', 99.0)",
                analysis_details={
                    "safety_level": "high",
                    "approval_reasoning": "Second write",
                },
                execution_mode="write",
                safety_concerns=["Second EPICS write"],
                step_objective="Second write",
            )
            # Override approval_type so get_approval_resume_data can match it
            interrupt_data["resume_payload"]["approval_type"] = "mock_approval_cap2"
            interrupt(interrupt_data)
            raise RuntimeError("unreachable")

        _mock_approval_cap2_node.name = "mock_approval_cap2"

        def make_cap2():
            cap = MagicMock()
            cap.name = "mock_approval_cap2"
            cap.description = "Second approval cap"
            cap.provides = ["MOCK_RESULT2"]
            cap.requires = []
            cap.orchestrator_guide = None
            cap.direct_chat_enabled = False
            return cap

        mock_registry = _make_mock_registry(extra_caps={"mock_approval_cap2": make_cap2()})

        llm_responses = [
            # 1. Orchestrator → mock_approval_cap
            [
                {
                    "id": "call_mock_approval_cap",
                    "type": "function",
                    "function": {
                        "name": "mock_approval_cap",
                        "arguments": json.dumps(
                            {
                                "task_objective": "First write",
                                "context_key": "write1",
                                "expected_output": "MOCK_RESULT",
                            }
                        ),
                    },
                }
            ],
            # 2. After first approval → mock_approval_cap2
            [
                {
                    "id": "call_mock_approval_cap2",
                    "type": "function",
                    "function": {
                        "name": "mock_approval_cap2",
                        "arguments": json.dumps(
                            {
                                "task_objective": "Second write",
                                "context_key": "write2",
                                "expected_output": "MOCK_RESULT2",
                            }
                        ),
                    },
                }
            ],
            # 3. After second approval → respond
            [
                {
                    "id": "call_respond",
                    "type": "function",
                    "function": {
                        "name": "respond",
                        "arguments": json.dumps(
                            {
                                "task_objective": "Confirm both writes succeeded",
                                "context_key": "user_response",
                                "expected_output": "FINAL_RESPONSE",
                            }
                        ),
                    },
                }
            ],
            "Both writes completed successfully.",  # response generation
        ]

        graph, config, patches = _build_test_graph(
            mock_registry,
            llm_responses,
            extra_nodes={"mock_approval_cap2": _mock_approval_cap2_node},
        )
        started = _enter_patches(patches)
        try:
            state = _create_reactive_approval_state(
                planning_active_capabilities=[
                    "mock_approval_cap",
                    "mock_approval_cap2",
                    "respond",
                ],
            )

            # Phase 1: Execute until first interrupt
            await graph.ainvoke(state, config)
            gs1 = graph.get_state(config)
            assert gs1.interrupts, "Expected first approval interrupt"
            assert gs1.values["react_step_count"] == 1

            # Approve first
            payload1 = gs1.interrupts[-1].value.get("resume_payload", {})
            await graph.ainvoke(
                Command(update={"approval_approved": True, "approved_payload": payload1}),
                config,
            )

            # Phase 2: Should hit second interrupt
            gs2 = graph.get_state(config)
            assert gs2.interrupts, "Expected second approval interrupt"
            assert gs2.values["react_step_count"] == 2

            # Approve second
            payload2 = gs2.interrupts[-1].value.get("resume_payload", {})
            await graph.ainvoke(
                Command(update={"approval_approved": True, "approved_payload": payload2}),
                config,
            )

            # Should generate response directly
            gs_final = graph.get_state(config)
            assert gs_final.values.get("react_response_generated") is True
            assert gs_final.values["react_step_count"] == 3

            # react_messages should have accumulated across both cycles
            msgs = gs_final.values.get("react_messages", [])
            assert len(msgs) >= 5  # 3 actions + 2 observations minimum

        finally:
            _exit_patches(started)


class TestReactiveStatePreservation:
    """Validate that reactive state fields survive an interrupt/resume cycle."""

    @pytest.mark.asyncio
    async def test_react_messages_preserved_through_interrupt(self):
        """react_messages accumulated before interrupt are intact after resume."""
        mock_registry = _make_mock_registry()

        llm_responses = [
            [
                {
                    "id": "call_mock_approval_cap",
                    "type": "function",
                    "function": {
                        "name": "mock_approval_cap",
                        "arguments": json.dumps(
                            {
                                "task_objective": "Write to PV",
                                "context_key": "write_result",
                                "expected_output": "MOCK_RESULT",
                            }
                        ),
                    },
                }
            ],
            [
                {
                    "id": "call_respond",
                    "type": "function",
                    "function": {
                        "name": "respond",
                        "arguments": json.dumps(
                            {
                                "task_objective": "Done",
                                "context_key": "user_response",
                                "expected_output": "FINAL_RESPONSE",
                            }
                        ),
                    },
                }
            ],
            "Done. The operation completed.",  # response generation
        ]

        graph, config, patches = _build_test_graph(mock_registry, llm_responses)
        started = _enter_patches(patches)
        try:
            # Start with pre-existing react_messages (simulating a prior step)
            prior_messages = [
                {
                    "role": "assistant",
                    "content": "Action: channel_finding\nObjective: Find channels",
                },
                {
                    "role": "observation",
                    "content": "Step completed successfully (capability: channel_finding)",
                },
            ]
            state = _create_reactive_approval_state(
                react_messages=prior_messages,
                react_step_count=1,
                planning_active_capabilities=["mock_approval_cap", "respond"],
            )

            # Execute until interrupt
            await graph.ainvoke(state, config)
            gs = graph.get_state(config)
            assert gs.interrupts

            # Capture state before resume
            msgs_before = copy.deepcopy(gs.values["react_messages"])
            count_before = gs.values["react_step_count"]

            # The prior messages should still be there, plus the new action
            assert len(msgs_before) >= len(prior_messages)
            # Prior messages are a prefix (observation may have been appended)
            assert msgs_before[0]["role"] == "assistant"
            assert "channel_finding" in msgs_before[0]["content"]

            # Resume with approval
            resume_payload = gs.interrupts[-1].value.get("resume_payload", {})
            await graph.ainvoke(
                Command(update={"approval_approved": True, "approved_payload": resume_payload}),
                config,
            )

            gs_after = graph.get_state(config)

            # react_step_count should have increased
            assert gs_after.values["react_step_count"] > count_before

            # All messages from before the interrupt should still be present
            msgs_after = gs_after.values["react_messages"]
            for i, msg in enumerate(msgs_before):
                assert msgs_after[i]["role"] == msg["role"]
                assert msgs_after[i]["content"] == msg["content"]

        finally:
            _exit_patches(started)

    @pytest.mark.asyncio
    async def test_capability_context_preserved_through_interrupt(self):
        """capability_context_data from prior steps survives interrupt/resume."""
        mock_registry = _make_mock_registry()

        llm_responses = [
            [
                {
                    "id": "call_mock_approval_cap",
                    "type": "function",
                    "function": {
                        "name": "mock_approval_cap",
                        "arguments": json.dumps(
                            {
                                "task_objective": "Write to PV",
                                "context_key": "write_result",
                                "expected_output": "MOCK_RESULT",
                            }
                        ),
                    },
                }
            ],
            [
                {
                    "id": "call_respond",
                    "type": "function",
                    "function": {
                        "name": "respond",
                        "arguments": json.dumps(
                            {
                                "task_objective": "Done",
                                "context_key": "user_response",
                                "expected_output": "FINAL_RESPONSE",
                            }
                        ),
                    },
                }
            ],
            "Done. Context preserved.",  # response generation
        ]

        graph, config, patches = _build_test_graph(mock_registry, llm_responses)
        started = _enter_patches(patches)
        try:
            prior_context = {
                "CHANNEL_ADDRESSES": {"beam_channels": {"pvs": ["SR:C01:BEAM", "SR:C02:BEAM"]}}
            }
            state = _create_reactive_approval_state(
                capability_context_data=prior_context,
            )

            await graph.ainvoke(state, config)
            gs = graph.get_state(config)
            assert gs.interrupts

            # Context data should be intact before resume
            ctx_before = gs.values.get("capability_context_data", {})
            assert "CHANNEL_ADDRESSES" in ctx_before
            assert "beam_channels" in ctx_before["CHANNEL_ADDRESSES"]

            # Resume
            resume_payload = gs.interrupts[-1].value.get("resume_payload", {})
            await graph.ainvoke(
                Command(update={"approval_approved": True, "approved_payload": resume_payload}),
                config,
            )

            gs_after = graph.get_state(config)
            ctx_after = gs_after.values.get("capability_context_data", {})
            assert "CHANNEL_ADDRESSES" in ctx_after
            assert (
                ctx_after["CHANNEL_ADDRESSES"]["beam_channels"]
                == prior_context["CHANNEL_ADDRESSES"]["beam_channels"]
            )

        finally:
            _exit_patches(started)


class TestRespondViaExecutionPlanWithCheckpointing:
    """Validate that respond routing works via execution plan dispatch.

    With react_route_to removed, respond/clarify are routed like any other
    capability: the orchestrator creates a single-step execution plan, and
    the router's execution plan dispatch routes to the respond node.
    The graph terminates via respond→END.
    """

    @pytest.mark.asyncio
    async def test_respond_dispatched_via_execution_plan(self):
        """Respond generates response directly in reactive orchestrator."""
        mock_registry = _make_mock_registry()

        llm_responses = [
            [
                {
                    "id": "call_respond",
                    "type": "function",
                    "function": {
                        "name": "respond",
                        "arguments": json.dumps(
                            {
                                "task_objective": "Respond to user",
                                "context_key": "user_response",
                                "expected_output": "FINAL_RESPONSE",
                            }
                        ),
                    },
                }
            ],
            "Here is the response to the user.",  # response generation
        ]

        graph, config, patches = _build_test_graph(mock_registry, llm_responses)
        started = _enter_patches(patches)
        try:
            state = _create_reactive_approval_state()
            await graph.ainvoke(state, config)

            final_state = graph.get_state(config)
            assert final_state.values.get("react_response_generated") is True

        finally:
            _exit_patches(started)

    @pytest.mark.asyncio
    async def test_approval_then_respond_via_execution_plan(self):
        """Approval chain completes and respond is generated directly."""
        mock_registry = _make_mock_registry()

        llm_responses = [
            [
                {
                    "id": "call_mock_approval_cap",
                    "type": "function",
                    "function": {
                        "name": "mock_approval_cap",
                        "arguments": json.dumps(
                            {
                                "task_objective": "Write to PV",
                                "context_key": "write_result",
                                "expected_output": "MOCK_RESULT",
                            }
                        ),
                    },
                }
            ],
            [
                {
                    "id": "call_respond",
                    "type": "function",
                    "function": {
                        "name": "respond",
                        "arguments": json.dumps(
                            {
                                "task_objective": "Done",
                                "context_key": "user_response",
                                "expected_output": "FINAL_RESPONSE",
                            }
                        ),
                    },
                }
            ],
            "Done. Write succeeded.",  # response generation
        ]

        graph, config, patches = _build_test_graph(mock_registry, llm_responses)
        started = _enter_patches(patches)
        try:
            state = _create_reactive_approval_state()

            # Phase 1: Interrupt
            await graph.ainvoke(state, config)
            gs = graph.get_state(config)
            assert gs.interrupts

            # Phase 2: Approve and finish
            resume_payload = gs.interrupts[-1].value.get("resume_payload", {})
            await graph.ainvoke(
                Command(update={"approval_approved": True, "approved_payload": resume_payload}),
                config,
            )

            # Graph should have generated response directly
            final = graph.get_state(config)
            assert final.values.get("react_response_generated") is True

        finally:
            _exit_patches(started)


# ===================================================================
# ORCHESTRATOR-LEVEL STEP APPROVAL TESTS
# ===================================================================


async def _mock_simple_capability_node(state: AgentState, **kwargs) -> dict[str, Any]:
    """Non-interrupting capability node for orchestrator-level approval tests.

    Unlike ``_mock_approval_capability_node``, this capability does NOT call
    ``interrupt()`` itself.  This isolates the orchestrator-level approval gate
    from capability-level gates.
    """
    return {
        "execution_last_result": {
            "capability": "mock_simple_cap",
            "success": True,
        },
        "planning_current_step_index": state.get("planning_current_step_index", 0) + 1,
        "control_has_error": False,
        "control_error_info": None,
        "control_retry_count": 0,
        "approval_approved": None,
        "approved_payload": None,
    }


_mock_simple_capability_node.name = "mock_simple_cap"
_mock_simple_capability_node.capability_name = "mock_simple_cap"


def _make_simple_cap_registry():
    """Build a mock registry with mock_simple_cap + respond."""
    registry = MagicMock()

    def make_cap(name, provides=None, requires=None):
        cap = MagicMock()
        cap.name = name
        cap.description = f"Mock {name}"
        cap.provides = provides or []
        cap.requires = requires or []
        cap.orchestrator_guide = None
        cap.direct_chat_enabled = False
        return cap

    caps = {
        "mock_simple_cap": make_cap("mock_simple_cap", provides=["MOCK_RESULT"]),
        "respond": make_cap("respond", provides=["FINAL_RESPONSE"]),
    }

    known_nodes = {
        "router",
        "reactive_orchestrator",
        "mock_simple_cap",
        "respond",
        "error",
    }

    registry.get_node.side_effect = lambda n: MagicMock() if n in known_nodes else None
    registry.get_capability.side_effect = lambda n: caps.get(n)
    registry.get_all_capabilities.return_value = list(caps.values())
    registry.get_stats.return_value = {"capability_names": list(caps.keys())}

    return registry


def _build_step_approval_graph(
    mock_registry,
    llm_responses: list[list[dict[str, Any]]],
):
    """Build a graph for orchestrator-level step approval tests.

    Patches ``_is_planning_mode_enabled`` to return True (simulating
    ``agent_control.planning_mode_enabled``).
    """
    from osprey.infrastructure.reactive_orchestrator_node import ReactiveOrchestratorNode

    patches = _common_patches(mock_registry)
    patches["planning_mode"] = patch(
        "osprey.infrastructure.reactive_orchestrator_node._is_planning_mode_enabled",
        return_value=True,
    )

    llm_iter = iter(llm_responses)

    def _next_llm_response(*args, **kwargs):
        return next(llm_iter)

    llm_patch = patch(
        "osprey.infrastructure.reactive_orchestrator_node.get_chat_completion",
        side_effect=_next_llm_response,
    )

    workflow = StateGraph(AgentState)

    async def _router_passthrough(state: AgentState, **kw) -> dict[str, Any]:
        return {}

    workflow.add_node("router", _router_passthrough)
    workflow.add_node("reactive_orchestrator", ReactiveOrchestratorNode.langgraph_node)
    workflow.add_node("mock_simple_cap", _mock_simple_capability_node)
    workflow.add_node("respond", _mock_respond_node)
    workflow.add_node("error", _mock_error_node)

    all_node_names = ["router", "reactive_orchestrator", "mock_simple_cap", "respond", "error"]
    routing_map = {n: n for n in all_node_names}
    routing_map["END"] = END

    workflow.set_entry_point("router")

    def _conditional_edge(state):
        return router_conditional_edge(state)

    workflow.add_conditional_edges("router", _conditional_edge, routing_map)
    workflow.add_edge("respond", END)
    workflow.add_edge("error", END)
    for name in all_node_names:
        if name not in ("router", "respond", "error"):
            workflow.add_edge(name, "router")

    checkpointer = MemorySaver()
    compiled = workflow.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "test-step-approval"}}

    all_patches = list(patches.values()) + [llm_patch]
    return compiled, config, all_patches


class TestReactiveOrchestratorStepApproval:
    """Orchestrator-level per-step approval with interrupt/resume."""

    @pytest.mark.asyncio
    async def test_step_approval_approve_and_execute(self):
        """Orchestrator interrupts for step approval; approve → capability executes → respond."""
        mock_registry = _make_simple_cap_registry()

        llm_responses = [
            # 1. Orchestrator picks mock_simple_cap → interrupt
            [
                {
                    "id": "call_mock_simple_cap",
                    "type": "function",
                    "function": {
                        "name": "mock_simple_cap",
                        "arguments": json.dumps(
                            {
                                "task_objective": "Do the thing",
                                "context_key": "result",
                                "expected_output": "MOCK_RESULT",
                            }
                        ),
                    },
                }
            ],
            # 2. After approval + capability success → respond
            [
                {
                    "id": "call_respond",
                    "type": "function",
                    "function": {
                        "name": "respond",
                        "arguments": json.dumps(
                            {
                                "task_objective": "Done",
                                "context_key": "user_response",
                                "expected_output": "FINAL_RESPONSE",
                            }
                        ),
                    },
                }
            ],
            "Done. The operation completed successfully.",  # response generation
        ]

        graph, config, patches = _build_step_approval_graph(mock_registry, llm_responses)
        started = _enter_patches(patches)
        try:
            state = _create_reactive_approval_state(
                planning_active_capabilities=["mock_simple_cap", "respond"],
            )

            # Phase 1: Execute until orchestrator interrupt
            await graph.ainvoke(state, config)
            gs = graph.get_state(config)
            assert gs.interrupts, "Expected orchestrator step approval interrupt"

            # Verify interrupt metadata
            interrupt_payload = gs.interrupts[-1].value.get("resume_payload", {})
            assert interrupt_payload["approval_type"] == "reactive_orchestrator_step"
            # react_step_count in checkpointed state is still 0 (interrupt prevents return),
            # but the resume payload carries the updated count
            assert interrupt_payload["react_step_count"] == 1

            # Phase 2: Approve and resume
            await graph.ainvoke(
                Command(update={"approval_approved": True, "approved_payload": interrupt_payload}),
                config,
            )

            # Should generate response directly
            final = graph.get_state(config)
            assert final.values.get("react_response_generated") is True
            assert final.values["react_step_count"] == 2

        finally:
            _exit_patches(started)

    @pytest.mark.asyncio
    async def test_step_approval_reject_and_recover(self):
        """Orchestrator interrupts for step approval; reject → LLM re-evaluates → respond."""
        mock_registry = _make_simple_cap_registry()

        llm_responses = [
            # 1. Orchestrator picks mock_simple_cap → interrupt
            [
                {
                    "id": "call_mock_simple_cap",
                    "type": "function",
                    "function": {
                        "name": "mock_simple_cap",
                        "arguments": json.dumps(
                            {
                                "task_objective": "Do the thing",
                                "context_key": "result",
                                "expected_output": "MOCK_RESULT",
                            }
                        ),
                    },
                }
            ],
            # 2. After rejection → LLM decides to respond
            [
                {
                    "id": "call_respond",
                    "type": "function",
                    "function": {
                        "name": "respond",
                        "arguments": json.dumps(
                            {
                                "task_objective": "Cancelled by user",
                                "context_key": "user_response",
                                "expected_output": "FINAL_RESPONSE",
                            }
                        ),
                    },
                }
            ],
            "The operation was cancelled by the user.",  # response generation
        ]

        graph, config, patches = _build_step_approval_graph(mock_registry, llm_responses)
        started = _enter_patches(patches)
        try:
            state = _create_reactive_approval_state(
                planning_active_capabilities=["mock_simple_cap", "respond"],
            )

            # Phase 1: Execute until orchestrator interrupt
            await graph.ainvoke(state, config)
            gs = graph.get_state(config)
            assert gs.interrupts, "Expected orchestrator step approval interrupt"

            # Phase 2: Reject
            await graph.ainvoke(
                Command(update={"approval_approved": False, "approved_payload": None}),
                config,
            )

            # Should generate response directly (LLM re-evaluated after rejection)
            final = graph.get_state(config)
            assert final.values.get("react_response_generated") is True

        finally:
            _exit_patches(started)

    @pytest.mark.asyncio
    async def test_step_approval_state_preserved_through_interrupt(self):
        """react_messages from before interrupt are intact after resume."""
        mock_registry = _make_simple_cap_registry()

        llm_responses = [
            # 1. Orchestrator picks mock_simple_cap → interrupt
            [
                {
                    "id": "call_mock_simple_cap",
                    "type": "function",
                    "function": {
                        "name": "mock_simple_cap",
                        "arguments": json.dumps(
                            {
                                "task_objective": "Do the thing",
                                "context_key": "result",
                                "expected_output": "MOCK_RESULT",
                            }
                        ),
                    },
                }
            ],
            # 2. After approval → respond
            [
                {
                    "id": "call_respond",
                    "type": "function",
                    "function": {
                        "name": "respond",
                        "arguments": json.dumps(
                            {
                                "task_objective": "Done",
                                "context_key": "user_response",
                                "expected_output": "FINAL_RESPONSE",
                            }
                        ),
                    },
                }
            ],
            "Done. State preserved.",  # response generation
        ]

        graph, config, patches = _build_step_approval_graph(mock_registry, llm_responses)
        # Use unique thread_id to avoid state collision
        config = {"configurable": {"thread_id": "test-step-approval-state-preserve"}}
        started = _enter_patches(patches)
        try:
            prior_messages = [
                {
                    "role": "assistant",
                    "content": "Action: channel_finding\nObjective: Find channels",
                },
                {
                    "role": "observation",
                    "content": "Step completed successfully (capability: channel_finding)",
                },
            ]
            state = _create_reactive_approval_state(
                planning_active_capabilities=["mock_simple_cap", "respond"],
                react_messages=prior_messages,
                react_step_count=1,
                capability_context_data={
                    "CHANNEL_ADDRESSES": {"beam_channels": {"pvs": ["SR:C01"]}}
                },
            )

            # Phase 1: Execute until interrupt
            await graph.ainvoke(state, config)
            gs = graph.get_state(config)
            assert gs.interrupts

            # react_messages should include prior messages + new action
            msgs_before = gs.values["react_messages"]
            assert len(msgs_before) >= len(prior_messages)
            assert msgs_before[0]["content"] == prior_messages[0]["content"]

            # Phase 2: Approve
            payload = gs.interrupts[-1].value.get("resume_payload", {})
            await graph.ainvoke(
                Command(update={"approval_approved": True, "approved_payload": payload}),
                config,
            )

            final = graph.get_state(config)

            # react_step_count incremented
            assert final.values["react_step_count"] > 1

            # Prior messages preserved
            msgs_after = final.values["react_messages"]
            assert msgs_after[0]["content"] == prior_messages[0]["content"]
            assert msgs_after[1]["content"] == prior_messages[1]["content"]

            # capability_context_data preserved
            ctx = final.values.get("capability_context_data", {})
            assert "CHANNEL_ADDRESSES" in ctx

        finally:
            _exit_patches(started)
