"""Unit Tests for XOpt Optimizer Service.

This module provides unit tests for the XOpt optimizer service including
service initialization, graph compilation, and basic workflow validation.

Test Coverage:
    - Service initialization and configuration
    - LangGraph compilation
    - State initialization
    - Basic workflow routing
"""

import os

import pytest

from osprey.services.xopt_optimizer import (
    MachineState,
    XOptExecutionRequest,
    XOptExecutionState,
    XOptServiceResult,
    XOptStrategy,
)
from osprey.services.xopt_optimizer.models import XOptError

# =============================================================================
# MODEL TESTS
# =============================================================================


class TestXOptModels:
    """Test XOpt data models."""

    def test_machine_state_enum(self):
        """MachineState enum should have expected values."""
        assert MachineState.READY.value == "ready"
        assert MachineState.NOT_READY.value == "not_ready"
        assert MachineState.UNKNOWN.value == "unknown"

    def test_xopt_strategy_enum(self):
        """XOptStrategy enum should have expected values."""
        assert XOptStrategy.EXPLORATION.value == "exploration"
        assert XOptStrategy.OPTIMIZATION.value == "optimization"
        assert XOptStrategy.ABORT.value == "abort"

    def test_xopt_execution_request_creation(self):
        """XOptExecutionRequest should be creatable with required fields."""
        request = XOptExecutionRequest(
            user_query="Optimize injection efficiency",
            optimization_objective="Maximize injection efficiency",
        )
        assert request.user_query == "Optimize injection efficiency"
        assert request.optimization_objective == "Maximize injection efficiency"
        assert request.max_iterations == 3  # Default
        assert request.require_approval is True  # Default

    def test_xopt_execution_request_custom_params(self):
        """XOptExecutionRequest should accept custom parameters."""
        request = XOptExecutionRequest(
            user_query="Test query",
            optimization_objective="Test objective",
            max_iterations=5,
            retries=2,
            require_approval=False,
        )
        assert request.max_iterations == 5
        assert request.retries == 2
        assert request.require_approval is False

    def test_xopt_error_dataclass(self):
        """XOptError should be creatable and formattable."""
        error = XOptError(
            error_type="test_error",
            error_message="Test error message",
            stage="yaml_generation",
            attempt_number=1,
            details={"key": "value"},
        )
        assert error.error_type == "test_error"
        assert error.error_message == "Test error message"
        assert error.stage == "yaml_generation"
        assert error.attempt_number == 1

        # Test prompt text formatting
        prompt_text = error.to_prompt_text()
        assert "YAML_GENERATION FAILED" in prompt_text
        assert "Test error message" in prompt_text

    def test_xopt_service_result_creation(self):
        """XOptServiceResult should be creatable with all fields."""
        result = XOptServiceResult(
            run_artifact={"status": "completed"},
            generated_yaml="test: yaml",
            strategy=XOptStrategy.EXPLORATION,
            total_iterations=3,
            analysis_summary={"summary": "test"},
            recommendations=("Recommendation 1", "Recommendation 2"),
        )
        assert result.run_artifact == {"status": "completed"}
        assert result.strategy == XOptStrategy.EXPLORATION
        assert result.total_iterations == 3
        assert len(result.recommendations) == 2


# =============================================================================
# SERVICE INITIALIZATION TESTS
# =============================================================================


class TestServiceInitialization:
    """Test service initialization and configuration."""

    def test_service_initializes(self, test_config):
        """Service should initialize without errors."""
        os.environ["CONFIG_FILE"] = str(test_config)

        from osprey.services.xopt_optimizer import XOptOptimizerService

        service = XOptOptimizerService()
        assert service is not None
        assert service.config is not None

    def test_service_builds_graph(self, test_config):
        """Service should build LangGraph on initialization."""
        os.environ["CONFIG_FILE"] = str(test_config)

        from osprey.services.xopt_optimizer import XOptOptimizerService

        service = XOptOptimizerService()
        graph = service.get_compiled_graph()
        assert graph is not None

    def test_service_creates_initial_state(self, test_config):
        """Service should create proper initial state from request."""
        os.environ["CONFIG_FILE"] = str(test_config)

        from osprey.services.xopt_optimizer import XOptOptimizerService

        service = XOptOptimizerService()
        request = XOptExecutionRequest(
            user_query="Test query",
            optimization_objective="Test objective",
            max_iterations=5,
        )

        state = service._create_initial_state(request)

        assert state["request"] == request
        assert state["max_iterations"] == 5
        assert state["iteration_count"] == 0
        assert state["is_successful"] is False
        assert state["is_failed"] is False
        assert state["current_stage"] == "state_id"
        assert state["error_chain"] == []


# =============================================================================
# NODE TESTS
# =============================================================================


class TestStateIdentificationNode:
    """Test state identification node."""

    @pytest.mark.asyncio
    async def test_state_identification_returns_ready(self, test_config):
        """State identification should return READY (mock mode)."""
        os.environ["CONFIG_FILE"] = str(test_config)

        # Ensure config cache is cleared after setting CONFIG_FILE
        from osprey.utils import config as config_module

        config_module._default_config = None
        config_module._default_configurable = None
        config_module._config_cache.clear()

        # Register MockConnector for channel access (if react mode runs)
        from osprey.connectors.control_system.mock_connector import MockConnector
        from osprey.connectors.factory import ConnectorFactory

        ConnectorFactory.register_control_system("mock", MockConnector)

        from osprey.services.xopt_optimizer.state_identification import (
            create_state_identification_node,
        )

        node = create_state_identification_node()

        # Create minimal state
        state = XOptExecutionState(
            request=XOptExecutionRequest(
                user_query="Test", optimization_objective="Test objective"
            ),
            capability_context_data=None,
            error_chain=[],
            yaml_generation_attempt=0,
            machine_state=None,
            machine_state_details=None,
            selected_strategy=None,
            decision_reasoning=None,
            generated_yaml=None,
            yaml_generation_failed=None,
            requires_approval=None,
            approval_interrupt_data=None,
            approval_result=None,
            approved=None,
            run_artifact=None,
            execution_error=None,
            execution_failed=None,
            analysis_result=None,
            recommendations=None,
            iteration_count=0,
            max_iterations=3,
            should_continue=False,
            is_successful=False,
            is_failed=False,
            failure_reason=None,
            current_stage="state_id",
        )

        result = await node(state)

        assert result["machine_state"] == MachineState.READY
        assert result["current_stage"] == "decision"
        assert "machine_state_details" in result


class TestDecisionNode:
    """Test decision node."""

    @pytest.mark.asyncio
    async def test_decision_routes_to_yaml_gen_when_ready(self, test_config):
        """Decision node should route to yaml_gen when machine is READY."""
        os.environ["CONFIG_FILE"] = str(test_config)

        from osprey.services.xopt_optimizer.decision import create_decision_node

        node = create_decision_node()

        state = XOptExecutionState(
            request=XOptExecutionRequest(
                user_query="Test", optimization_objective="Test objective"
            ),
            capability_context_data=None,
            error_chain=[],
            yaml_generation_attempt=0,
            machine_state=MachineState.READY,
            machine_state_details={"assessment": "test"},
            selected_strategy=None,
            decision_reasoning=None,
            generated_yaml=None,
            yaml_generation_failed=None,
            requires_approval=None,
            approval_interrupt_data=None,
            approval_result=None,
            approved=None,
            run_artifact=None,
            execution_error=None,
            execution_failed=None,
            analysis_result=None,
            recommendations=None,
            iteration_count=0,
            max_iterations=3,
            should_continue=False,
            is_successful=False,
            is_failed=False,
            failure_reason=None,
            current_stage="decision",
        )

        result = await node(state)

        assert result["selected_strategy"] == XOptStrategy.EXPLORATION
        assert result["current_stage"] == "yaml_gen"
        assert "decision_reasoning" in result

    @pytest.mark.asyncio
    async def test_decision_aborts_when_not_ready(self, test_config):
        """Decision node should abort when machine is NOT_READY."""
        os.environ["CONFIG_FILE"] = str(test_config)

        from osprey.services.xopt_optimizer.decision import create_decision_node

        node = create_decision_node()

        state = XOptExecutionState(
            request=XOptExecutionRequest(
                user_query="Test", optimization_objective="Test objective"
            ),
            capability_context_data=None,
            error_chain=[],
            yaml_generation_attempt=0,
            machine_state=MachineState.NOT_READY,
            machine_state_details={"reason": "Machine offline"},
            selected_strategy=None,
            decision_reasoning=None,
            generated_yaml=None,
            yaml_generation_failed=None,
            requires_approval=None,
            approval_interrupt_data=None,
            approval_result=None,
            approved=None,
            run_artifact=None,
            execution_error=None,
            execution_failed=None,
            analysis_result=None,
            recommendations=None,
            iteration_count=0,
            max_iterations=3,
            should_continue=False,
            is_successful=False,
            is_failed=False,
            failure_reason=None,
            current_stage="decision",
        )

        result = await node(state)

        assert result["selected_strategy"] == XOptStrategy.ABORT
        assert result["is_failed"] is True
        assert result["current_stage"] == "failed"


class TestAnalysisNode:
    """Test analysis node."""

    @pytest.mark.asyncio
    async def test_analysis_continues_when_under_max_iterations(self, test_config):
        """Analysis should continue when under max iterations."""
        os.environ["CONFIG_FILE"] = str(test_config)

        from osprey.services.xopt_optimizer.analysis import create_analysis_node

        node = create_analysis_node()

        state = XOptExecutionState(
            request=XOptExecutionRequest(
                user_query="Test", optimization_objective="Test objective"
            ),
            capability_context_data=None,
            error_chain=[],
            yaml_generation_attempt=0,
            machine_state=MachineState.READY,
            machine_state_details={},
            selected_strategy=XOptStrategy.EXPLORATION,
            decision_reasoning="test",
            generated_yaml="test: yaml",
            yaml_generation_failed=False,
            requires_approval=False,
            approval_interrupt_data=None,
            approval_result=None,
            approved=True,
            run_artifact={"status": "completed"},
            execution_error=None,
            execution_failed=False,
            analysis_result=None,
            recommendations=None,
            iteration_count=0,  # First iteration
            max_iterations=3,
            should_continue=False,
            is_successful=False,
            is_failed=False,
            failure_reason=None,
            current_stage="analysis",
        )

        result = await node(state)

        assert result["should_continue"] is True
        assert result["iteration_count"] == 1
        assert result["current_stage"] == "state_id"

    @pytest.mark.asyncio
    async def test_analysis_completes_at_max_iterations(self, test_config):
        """Analysis should complete when reaching max iterations."""
        os.environ["CONFIG_FILE"] = str(test_config)

        from osprey.services.xopt_optimizer.analysis import create_analysis_node

        node = create_analysis_node()

        state = XOptExecutionState(
            request=XOptExecutionRequest(
                user_query="Test", optimization_objective="Test objective"
            ),
            capability_context_data=None,
            error_chain=[],
            yaml_generation_attempt=0,
            machine_state=MachineState.READY,
            machine_state_details={},
            selected_strategy=XOptStrategy.EXPLORATION,
            decision_reasoning="test",
            generated_yaml="test: yaml",
            yaml_generation_failed=False,
            requires_approval=False,
            approval_interrupt_data=None,
            approval_result=None,
            approved=True,
            run_artifact={"status": "completed"},
            execution_error=None,
            execution_failed=False,
            analysis_result=None,
            recommendations=None,
            iteration_count=2,  # Already at iteration 2, next will be 3 (max)
            max_iterations=3,
            should_continue=False,
            is_successful=False,
            is_failed=False,
            failure_reason=None,
            current_stage="analysis",
        )

        result = await node(state)

        assert result["should_continue"] is False
        assert result["iteration_count"] == 3
        assert result["is_successful"] is True
        assert result["current_stage"] == "complete"


# =============================================================================
# ROUTING TESTS
# =============================================================================


class TestServiceRouting:
    """Test service routing logic."""

    def test_decision_router_continues_on_ready(self, test_config):
        """Decision router should return 'continue' when ready."""
        os.environ["CONFIG_FILE"] = str(test_config)

        from osprey.services.xopt_optimizer import XOptOptimizerService

        service = XOptOptimizerService()

        state = {
            "is_failed": False,
            "selected_strategy": XOptStrategy.EXPLORATION,
        }

        result = service._decision_router(state)
        assert result == "continue"

    def test_decision_router_aborts_on_failed(self, test_config):
        """Decision router should return 'abort' when failed."""
        os.environ["CONFIG_FILE"] = str(test_config)

        from osprey.services.xopt_optimizer import XOptOptimizerService

        service = XOptOptimizerService()

        state = {
            "is_failed": True,
            "selected_strategy": XOptStrategy.EXPLORATION,
        }

        result = service._decision_router(state)
        assert result == "abort"

    def test_decision_router_aborts_on_abort_strategy(self, test_config):
        """Decision router should return 'abort' when strategy is ABORT."""
        os.environ["CONFIG_FILE"] = str(test_config)

        from osprey.services.xopt_optimizer import XOptOptimizerService

        service = XOptOptimizerService()

        state = {
            "is_failed": False,
            "selected_strategy": XOptStrategy.ABORT,
        }

        result = service._decision_router(state)
        assert result == "abort"

    def test_approval_router_approved(self, test_config):
        """Approval router should return 'approved' when approved."""
        os.environ["CONFIG_FILE"] = str(test_config)

        from osprey.services.xopt_optimizer import XOptOptimizerService

        service = XOptOptimizerService()

        state = {"approved": True}
        result = service._approval_router(state)
        assert result == "approved"

    def test_approval_router_rejected(self, test_config):
        """Approval router should return 'rejected' when not approved."""
        os.environ["CONFIG_FILE"] = str(test_config)

        from osprey.services.xopt_optimizer import XOptOptimizerService

        service = XOptOptimizerService()

        state = {"approved": False}
        result = service._approval_router(state)
        assert result == "rejected"

    def test_loop_router_continues(self, test_config):
        """Loop router should return 'continue' when should_continue is True."""
        os.environ["CONFIG_FILE"] = str(test_config)

        from osprey.services.xopt_optimizer import XOptOptimizerService

        service = XOptOptimizerService()

        state = {"is_failed": False, "should_continue": True}
        result = service._loop_router(state)
        assert result == "continue"

    def test_loop_router_completes(self, test_config):
        """Loop router should return 'complete' when should_continue is False."""
        os.environ["CONFIG_FILE"] = str(test_config)

        from osprey.services.xopt_optimizer import XOptOptimizerService

        service = XOptOptimizerService()

        state = {"is_failed": False, "should_continue": False}
        result = service._loop_router(state)
        assert result == "complete"
