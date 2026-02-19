"""Tests for _reactive_routing() in router_node.

All tests are deterministic - no LLM calls. Tests mock get_config_value
to activate react mode and verify routing decisions.
"""

from unittest.mock import MagicMock, patch

import pytest

from osprey.base.errors import ErrorClassification, ErrorSeverity
from osprey.infrastructure.router_node import router_conditional_edge
from tests.conftest import create_test_state


def _create_react_state(**overrides):
    """Create a minimal state for reactive routing tests.

    Wraps ``create_test_state()`` from conftest with reactive-routing defaults.
    """
    defaults = {
        "task_current_task": None,
        "planning_active_capabilities": [],
        "planning_execution_plan": None,
        "planning_current_step_index": 0,
        "execution_start_time": 1.0,
        "control_plans_created_count": 0,
    }
    defaults.update(overrides)
    return create_test_state(**defaults)


@pytest.fixture
def mock_registry():
    """Mock registry that recognizes standard capabilities."""
    registry = MagicMock()

    def get_node(name):
        known = {
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
        return MagicMock() if name in known else None

    def get_capability(name):
        if name in {"channel_finding", "channel_read", "respond"}:
            cap = MagicMock()
            cap.direct_chat_enabled = False
            return cap
        return None

    registry.get_node.side_effect = get_node
    registry.get_capability.side_effect = get_capability
    return registry


def _patch_react_mode(mock_registry):
    """Return a set of patches for reactive mode testing."""
    return [
        patch(
            "osprey.infrastructure.router_node.get_config_value",
            side_effect=lambda path, default=None: (
                "react"
                if path == "execution_control.agent_control.orchestration_mode"
                else (100 if path == "execution_control.limits.graph_recursion_limit" else default)
            ),
        ),
        patch(
            "osprey.infrastructure.router_node.get_registry",
            return_value=mock_registry,
        ),
    ]


class TestReactiveRoutingFreshState:
    """Test routing with fresh state (no task, no capabilities)."""

    def test_fresh_state_routes_to_task_extraction(self, mock_registry):
        """Fresh state with no task routes to task_extraction."""
        state = _create_react_state()
        patches = _patch_react_mode(mock_registry)

        with patches[0], patches[1]:
            result = router_conditional_edge(state)
            assert result == "task_extraction"

    def test_task_but_no_capabilities_routes_to_classifier(self, mock_registry):
        """State with task but no capabilities routes to classifier."""
        state = _create_react_state(
            task_current_task="Find beam current",
        )
        patches = _patch_react_mode(mock_registry)

        with patches[0], patches[1]:
            result = router_conditional_edge(state)
            assert result == "classifier"

    def test_task_and_capabilities_routes_to_reactive_orchestrator(self, mock_registry):
        """State with task and capabilities but no plan routes to reactive_orchestrator."""
        state = _create_react_state(
            task_current_task="Find beam current",
            planning_active_capabilities=["channel_finding"],
        )
        patches = _patch_react_mode(mock_registry)

        with patches[0], patches[1]:
            result = router_conditional_edge(state)
            assert result == "reactive_orchestrator"


class TestReactiveRoutingAfterCapability:
    """Test routing after a capability has executed."""

    def test_step_completed_routes_to_reactive_orchestrator(self, mock_registry):
        """After step completes (index >= plan length), routes back to reactive_orchestrator."""
        state = _create_react_state(
            task_current_task="Find beam current",
            planning_active_capabilities=["channel_finding"],
            planning_execution_plan={"steps": [{"capability": "channel_finding"}]},
            planning_current_step_index=1,  # Past the end
        )
        patches = _patch_react_mode(mock_registry)

        with patches[0], patches[1]:
            result = router_conditional_edge(state)
            assert result == "reactive_orchestrator"

    def test_step_pending_routes_to_capability(self, mock_registry):
        """Step not yet executed routes to the capability."""
        state = _create_react_state(
            task_current_task="Find beam current",
            planning_active_capabilities=["channel_finding"],
            planning_execution_plan={"steps": [{"capability": "channel_finding"}]},
            planning_current_step_index=0,
        )
        patches = _patch_react_mode(mock_registry)

        with patches[0], patches[1]:
            result = router_conditional_edge(state)
            assert result == "channel_finding"


class TestReactiveRoutingRespondViaExecutionPlan:
    """Test respond routing via execution plan dispatch (no react_route_to signal)."""

    def test_respond_step_routes_to_respond(self, mock_registry):
        """Respond step in execution plan routes to respond capability."""
        state = _create_react_state(
            task_current_task="Find beam current",
            planning_active_capabilities=["respond"],
            planning_execution_plan={"steps": [{"capability": "respond"}]},
            planning_current_step_index=0,
        )
        patches = _patch_react_mode(mock_registry)

        with patches[0], patches[1]:
            result = router_conditional_edge(state)
            assert result == "respond"

    def test_completed_respond_step_routes_back(self, mock_registry):
        """After respond step completes (index past end), routes to reactive_orchestrator."""
        state = _create_react_state(
            task_current_task="Find beam current",
            planning_active_capabilities=["respond"],
            planning_execution_plan={"steps": [{"capability": "respond"}]},
            planning_current_step_index=1,
        )
        patches = _patch_react_mode(mock_registry)

        with patches[0], patches[1]:
            result = router_conditional_edge(state)
            assert result == "reactive_orchestrator"


class TestReactiveRoutingErrorHandling:
    """Test error handling in reactive routing."""

    def test_retriable_error_retries_capability(self, mock_registry):
        """RETRIABLE error retries the capability."""
        classification = ErrorClassification(
            severity=ErrorSeverity.RETRIABLE,
            user_message="Timeout",
            metadata={},
        )
        state = _create_react_state(
            task_current_task="Find beam current",
            planning_active_capabilities=["channel_finding"],
            control_has_error=True,
            control_error_info={
                "classification": classification,
                "capability_name": "channel_finding",
                "retry_policy": {"max_attempts": 3},
            },
            control_retry_count=0,
        )
        patches = _patch_react_mode(mock_registry)

        with patches[0], patches[1]:
            result = router_conditional_edge(state)
            assert result == "channel_finding"
            assert state["control_retry_count"] == 1

    def test_retriable_error_exhausted_routes_to_reactive_orchestrator(self, mock_registry):
        """Exhausted RETRIABLE retries route to reactive_orchestrator."""
        classification = ErrorClassification(
            severity=ErrorSeverity.RETRIABLE,
            user_message="Timeout",
            metadata={},
        )
        state = _create_react_state(
            task_current_task="Find beam current",
            planning_active_capabilities=["channel_finding"],
            control_has_error=True,
            control_error_info={
                "classification": classification,
                "capability_name": "channel_finding",
                "retry_policy": {"max_attempts": 3},
            },
            control_retry_count=3,
        )
        patches = _patch_react_mode(mock_registry)

        with patches[0], patches[1]:
            result = router_conditional_edge(state)
            assert result == "reactive_orchestrator"

    def test_replanning_error_routes_to_reactive_orchestrator(self, mock_registry):
        """REPLANNING error routes to reactive_orchestrator."""
        classification = ErrorClassification(
            severity=ErrorSeverity.REPLANNING,
            user_message="Bad plan",
            metadata={},
        )
        state = _create_react_state(
            task_current_task="Find beam current",
            planning_active_capabilities=["channel_finding"],
            control_has_error=True,
            control_error_info={
                "classification": classification,
                "capability_name": "channel_finding",
                "retry_policy": {},
            },
        )
        patches = _patch_react_mode(mock_registry)

        with patches[0], patches[1]:
            result = router_conditional_edge(state)
            assert result == "reactive_orchestrator"

    def test_reclassification_error_routes_to_reactive_orchestrator(self, mock_registry):
        """RECLASSIFICATION error routes to reactive_orchestrator."""
        classification = ErrorClassification(
            severity=ErrorSeverity.RECLASSIFICATION,
            user_message="Wrong capability",
            metadata={},
        )
        state = _create_react_state(
            task_current_task="Find beam current",
            planning_active_capabilities=["channel_finding"],
            control_has_error=True,
            control_error_info={
                "classification": classification,
                "capability_name": "channel_finding",
                "retry_policy": {},
            },
        )
        patches = _patch_react_mode(mock_registry)

        with patches[0], patches[1]:
            result = router_conditional_edge(state)
            assert result == "reactive_orchestrator"

    def test_critical_error_routes_to_reactive_orchestrator(self, mock_registry):
        """CRITICAL error routes to reactive_orchestrator (not error node)."""
        classification = ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message="Critical failure",
            metadata={},
        )
        state = _create_react_state(
            task_current_task="Find beam current",
            planning_active_capabilities=["channel_finding"],
            control_has_error=True,
            control_error_info={
                "classification": classification,
                "capability_name": "channel_finding",
                "retry_policy": {},
            },
        )
        patches = _patch_react_mode(mock_registry)

        with patches[0], patches[1]:
            result = router_conditional_edge(state)
            assert result == "reactive_orchestrator"


class TestReactiveRoutingMaxIterations:
    """Test max iterations guard."""

    def test_max_iterations_routes_to_error(self, mock_registry):
        """Exceeding max iterations routes to error node."""
        state = _create_react_state(
            task_current_task="Find beam current",
            planning_active_capabilities=["channel_finding"],
            react_step_count=100,
        )
        patches = _patch_react_mode(mock_registry)

        with patches[0], patches[1]:
            result = router_conditional_edge(state)
            assert result == "error"

    def test_below_max_iterations_continues(self, mock_registry):
        """Below max iterations continues to reactive_orchestrator."""
        state = _create_react_state(
            task_current_task="Find beam current",
            planning_active_capabilities=["channel_finding"],
            react_step_count=99,
        )
        patches = _patch_react_mode(mock_registry)

        with patches[0], patches[1]:
            result = router_conditional_edge(state)
            assert result == "reactive_orchestrator"


class TestReactiveRoutingDirectResponse:
    """Test routing when reactive orchestrator generated response directly."""

    def test_react_response_generated_routes_to_end(self, mock_registry):
        """react_response_generated=True routes directly to END."""
        state = _create_react_state(
            task_current_task="Find beam current",
            planning_active_capabilities=["channel_finding"],
            react_response_generated=True,
        )
        patches = _patch_react_mode(mock_registry)

        with patches[0], patches[1]:
            result = router_conditional_edge(state)
            assert result == "END"

    def test_react_response_not_generated_continues(self, mock_registry):
        """react_response_generated=False continues normal routing."""
        state = _create_react_state(
            task_current_task="Find beam current",
            planning_active_capabilities=["channel_finding"],
            react_response_generated=False,
        )
        patches = _patch_react_mode(mock_registry)

        with patches[0], patches[1]:
            result = router_conditional_edge(state)
            assert result == "reactive_orchestrator"


class TestPlanFirstRegression:
    """Ensure plan-first mode is unaffected by reactive routing changes."""

    def test_plan_first_mode_ignores_reactive_routing(self, mock_registry):
        """Plan-first mode bypasses reactive routing entirely."""
        state = _create_react_state(
            task_current_task="Find beam current",
            planning_active_capabilities=["channel_finding"],
            execution_start_time=1.0,
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
            result = router_conditional_edge(state)
            # Without an execution plan, plan-first routes to orchestrator
            assert result == "orchestrator"
