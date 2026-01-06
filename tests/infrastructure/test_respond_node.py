"""Tests for respond capability node."""

from unittest.mock import Mock, patch

from osprey.base.errors import ErrorSeverity
from osprey.infrastructure.respond_node import (
    RespondCapability,
    ResponseContext,
    _determine_response_mode,
    _gather_information,
    _get_base_system_prompt,
    _get_capabilities_overview,
    _get_execution_history,
)
from osprey.state import AgentState


class TestResponseContext:
    """Test ResponseContext dataclass."""

    def test_response_context_initialization(self):
        """Test basic ResponseContext creation."""
        context = ResponseContext(
            current_task="Test task",
            execution_history=[{"step": 1}],
            relevant_context=[{"context": "test"}],
            is_killed=False,
            kill_reason=None,
            capabilities_overview="Test capabilities",
            total_steps_executed=5,
            execution_start_time=1234.56,
            reclassification_count=0,
            current_date="2025-01-01",
            figures_available=2,
            commands_available=3,
            notebooks_available=1,
            interface_context="cli",
        )

        assert context.current_task == "Test task"
        assert len(context.execution_history) == 1
        assert context.is_killed is False
        assert context.total_steps_executed == 5
        assert context.figures_available == 2
        assert context.interface_context == "cli"

    def test_response_context_with_minimal_data(self):
        """Test ResponseContext with minimal required fields."""
        context = ResponseContext(
            current_task="Task",
            execution_history=[],
            relevant_context=[],
            is_killed=False,
            kill_reason=None,
            capabilities_overview=None,
            total_steps_executed=0,
            execution_start_time=None,
            reclassification_count=0,
            current_date="2025-01-01",
            figures_available=0,
            commands_available=0,
            notebooks_available=0,
            interface_context="cli",
        )

        assert context.execution_history == []
        assert context.capabilities_overview is None
        assert context.figures_available == 0


class TestRespondCapability:
    """Test RespondCapability node."""

    def test_capability_exists_and_is_callable(self):
        """Verify RespondCapability can be instantiated."""
        capability = RespondCapability()
        assert capability is not None
        assert hasattr(capability, "execute")

    def test_has_langgraph_node_attribute(self):
        """Test that RespondCapability has langgraph_node from decorator."""
        assert hasattr(RespondCapability, "langgraph_node")
        assert callable(RespondCapability.langgraph_node)

    def test_capability_name_and_description(self):
        """Test capability has correct name and description."""
        assert RespondCapability.name == "respond"
        assert RespondCapability.description is not None
        assert len(RespondCapability.description) > 0

    def test_capability_provides(self):
        """Test capability provides FINAL_RESPONSE."""
        assert "FINAL_RESPONSE" in RespondCapability.provides

    def test_capability_requires_empty(self):
        """Test capability requires no specific inputs."""
        assert RespondCapability.requires == []

    def test_classify_error(self):
        """Test error classification for respond capability."""
        exc = RuntimeError("LLM error")
        context = {"operation": "response_generation"}

        classification = RespondCapability.classify_error(exc, context)

        assert classification.severity == ErrorSeverity.CRITICAL
        assert "Failed to generate response" in classification.user_message
        assert "technical_details" in classification.metadata


class TestDetermineResponseMode:
    """Test _determine_response_mode helper function."""

    def test_conversational_mode_no_inputs_or_data(self):
        """Test conversational mode when no inputs or capability data."""
        state = AgentState()
        current_step = {}

        mode = _determine_response_mode(state, current_step)

        assert mode == "conversational"

    def test_specific_context_mode_with_step_inputs(self):
        """Test specific context mode when step has inputs."""
        state = AgentState()
        current_step = {"inputs": ["context1", "context2"]}

        mode = _determine_response_mode(state, current_step)

        assert mode == "specific_context"

    def test_general_context_mode_with_capability_data(self):
        """Test general context mode when capability data exists."""
        state = AgentState()
        state["capability_context_data"] = {"data": "value"}
        current_step = {}

        mode = _determine_response_mode(state, current_step)

        assert mode == "general_context"

    def test_specific_context_takes_precedence(self):
        """Test that specific context mode takes precedence over general."""
        state = AgentState()
        state["capability_context_data"] = {"data": "value"}
        current_step = {"inputs": ["context1"]}

        mode = _determine_response_mode(state, current_step)

        assert mode == "specific_context"

    def test_handles_none_current_step(self):
        """Test handling of None current_step."""
        state = AgentState()

        mode = _determine_response_mode(state, None)

        assert mode == "conversational"


class TestGetCapabilitiesOverview:
    """Test _get_capabilities_overview helper function."""

    def test_get_capabilities_overview_success(self):
        """Test successful retrieval of capabilities overview."""
        with patch("osprey.infrastructure.respond_node.get_registry") as mock_registry:
            mock_reg = Mock()
            mock_reg.get_capabilities_overview.return_value = "Test capabilities"
            mock_registry.return_value = mock_reg

            overview = _get_capabilities_overview()

        assert overview == "Test capabilities"

    def test_get_capabilities_overview_fallback_on_error(self):
        """Test fallback when registry raises exception."""
        with patch("osprey.infrastructure.respond_node.get_registry") as mock_registry:
            mock_registry.side_effect = Exception("Registry error")

            overview = _get_capabilities_overview()

        assert "General AI Assistant" in overview


class TestGetExecutionHistory:
    """Test _get_execution_history helper function."""

    def test_get_execution_history_empty(self):
        """Test getting execution history when state is empty."""
        state = AgentState()

        history = _get_execution_history(state)

        assert history == []

    def test_get_execution_history_single_step(self):
        """Test getting execution history with one step."""
        state = AgentState()
        state["execution_step_results"] = {
            "step_0": {
                "step_index": 0,
                "capability": "test",
                "success": True,
            }
        }

        history = _get_execution_history(state)

        assert len(history) == 1
        assert history[0]["step_index"] == 0

    def test_get_execution_history_ordered(self):
        """Test that execution history is ordered by step_index."""
        state = AgentState()
        state["execution_step_results"] = {
            "step_2": {
                "step_index": 2,
                "capability": "third",
            },
            "step_0": {
                "step_index": 0,
                "capability": "first",
            },
            "step_1": {
                "step_index": 1,
                "capability": "second",
            },
        }

        history = _get_execution_history(state)

        assert len(history) == 3
        assert history[0]["capability"] == "first"
        assert history[1]["capability"] == "second"
        assert history[2]["capability"] == "third"

    def test_get_execution_history_missing_step_index(self):
        """Test handling of steps without step_index."""
        state = AgentState()
        state["execution_step_results"] = {
            "step_0": {"capability": "test"},  # No step_index
        }

        history = _get_execution_history(state)

        assert len(history) == 1


class TestGetBaseSystemPrompt:
    """Test _get_base_system_prompt helper function."""

    def test_get_base_system_prompt_simple(self):
        """Test building base system prompt with just task."""
        with patch("osprey.infrastructure.respond_node.get_framework_prompts") as mock_get_prompts:
            mock_builder = Mock()
            mock_builder.get_system_instructions.return_value = "System prompt"
            mock_prompts = Mock()
            mock_prompts.get_response_generation_prompt_builder.return_value = mock_builder
            mock_get_prompts.return_value = mock_prompts

            prompt = _get_base_system_prompt("Test task")

        assert prompt == "System prompt"
        mock_builder.get_system_instructions.assert_called_once_with(
            current_task="Test task", info=None
        )

    def test_get_base_system_prompt_with_context(self):
        """Test building prompt with response context."""
        response_context = ResponseContext(
            current_task="Task",
            execution_history=[],
            relevant_context=[],
            is_killed=False,
            kill_reason=None,
            capabilities_overview="Caps",
            total_steps_executed=0,
            execution_start_time=None,
            reclassification_count=0,
            current_date="2025-01-01",
            figures_available=0,
            commands_available=0,
            notebooks_available=0,
            interface_context="cli",
        )

        with patch("osprey.infrastructure.respond_node.get_framework_prompts") as mock_get_prompts:
            mock_builder = Mock()
            mock_builder.get_system_instructions.return_value = "Detailed prompt"
            mock_prompts = Mock()
            mock_prompts.get_response_generation_prompt_builder.return_value = mock_builder
            mock_get_prompts.return_value = mock_prompts

            prompt = _get_base_system_prompt("Task", response_context)

        assert prompt == "Detailed prompt"
        mock_builder.get_system_instructions.assert_called_once()
        call_kwargs = mock_builder.get_system_instructions.call_args.kwargs
        assert call_kwargs["current_task"] == "Task"
        assert call_kwargs["info"] == response_context


class TestGatherInformation:
    """Test _gather_information helper function."""

    def test_gather_information_conversational_mode(self):
        """Test gathering information in conversational mode."""
        state = AgentState()
        state["task_current_task"] = "Test task"
        logger = Mock()

        with (
            patch("osprey.infrastructure.respond_node.ContextManager") as mock_cm,
            patch("osprey.infrastructure.respond_node.StateManager") as mock_sm,
            patch("osprey.infrastructure.respond_node._determine_response_mode") as mock_mode,
            patch("osprey.infrastructure.respond_node._get_capabilities_overview") as mock_caps,
            patch("osprey.utils.config.get_interface_context") as mock_interface,
        ):
            # Setup mocks
            mock_cm_instance = Mock()
            mock_cm_instance.get_summaries.return_value = []
            mock_cm.return_value = mock_cm_instance

            mock_sm.get_current_step.return_value = {}
            mock_sm.get_current_step_index.return_value = 0

            mock_mode.return_value = "conversational"
            mock_caps.return_value = "Test capabilities"
            mock_interface.return_value = "cli"

            context = _gather_information(state, logger)

        assert isinstance(context, ResponseContext)
        assert context.current_task == "Test task"
        assert context.execution_history == []
        assert context.capabilities_overview == "Test capabilities"
        assert context.interface_context == "cli"

    def test_gather_information_technical_mode(self):
        """Test gathering information in technical mode."""
        state = AgentState()
        state["task_current_task"] = "Technical task"
        state["execution_step_results"] = {"step_0": {"step_index": 0, "capability": "test"}}
        logger = Mock()

        with (
            patch("osprey.infrastructure.respond_node.ContextManager") as mock_cm,
            patch("osprey.infrastructure.respond_node.StateManager") as mock_sm,
            patch("osprey.infrastructure.respond_node._determine_response_mode") as mock_mode,
            patch("osprey.infrastructure.respond_node._get_execution_history") as mock_history,
            patch("osprey.utils.config.get_interface_context") as mock_interface,
        ):
            mock_cm_instance = Mock()
            mock_cm_instance.get_summaries.return_value = []
            mock_cm.return_value = mock_cm_instance

            mock_sm.get_current_step.return_value = {}
            mock_sm.get_current_step_index.return_value = 1

            mock_mode.return_value = "specific_context"
            mock_history.return_value = [{"step_index": 0}]
            mock_interface.return_value = "openwebui"

            context = _gather_information(state, logger)

        assert context.execution_history == [{"step_index": 0}]
        assert context.capabilities_overview is None  # Not needed in technical mode

    def test_gather_information_includes_ui_elements(self):
        """Test that gathering includes UI figures, commands, and notebooks."""
        state = AgentState()
        state["task_current_task"] = "Test"
        state["capability_context_data"] = {}
        state["ui_captured_figures"] = [{"id": 1}, {"id": 2}]
        state["ui_launchable_commands"] = [{"cmd": "test"}]
        state["ui_captured_notebooks"] = [{"nb": "test.ipynb"}]

        with (
            patch("osprey.context.context_manager.ContextManager") as mock_cm_class,
            patch("osprey.infrastructure.respond_node.StateManager") as mock_sm,
            patch("osprey.infrastructure.respond_node._determine_response_mode") as mock_mode,
            patch("osprey.infrastructure.respond_node._get_capabilities_overview") as mock_caps,
            patch("osprey.utils.config.get_interface_context") as mock_interface,
        ):
            mock_cm_instance = Mock()
            mock_cm_instance.get_summaries.return_value = []
            mock_cm_class.return_value = mock_cm_instance

            mock_sm.get_current_step.return_value = {}
            mock_sm.get_current_step_index.return_value = 0
            mock_mode.return_value = "conversational"
            mock_caps.return_value = "Test"
            mock_interface.return_value = "cli"

            context = _gather_information(state)

        assert context.figures_available == 2
        assert context.commands_available == 1
        assert context.notebooks_available == 1

    def test_gather_information_includes_kill_status(self):
        """Test that kill status is included in context."""
        state = AgentState()
        state["task_current_task"] = "Test"
        state["capability_context_data"] = {}
        state["control_is_killed"] = True
        state["control_kill_reason"] = "Timeout"

        with (
            patch("osprey.context.context_manager.ContextManager") as mock_cm_class,
            patch("osprey.infrastructure.respond_node.StateManager") as mock_sm,
            patch("osprey.infrastructure.respond_node._determine_response_mode") as mock_mode,
            patch("osprey.infrastructure.respond_node._get_capabilities_overview") as mock_caps,
            patch("osprey.utils.config.get_interface_context") as mock_interface,
        ):
            mock_cm_instance = Mock()
            mock_cm_instance.get_summaries.return_value = []
            mock_cm_class.return_value = mock_cm_instance

            mock_sm.get_current_step.return_value = {}
            mock_sm.get_current_step_index.return_value = 0
            mock_mode.return_value = "conversational"
            mock_caps.return_value = "Test"
            mock_interface.return_value = "cli"

            context = _gather_information(state)

        assert context.is_killed is True
        assert context.kill_reason == "Timeout"

    def test_gather_information_includes_current_date(self):
        """Test that current date is included in context."""
        state = AgentState()
        state["task_current_task"] = "Test"
        state["capability_context_data"] = {}

        with (
            patch("osprey.context.context_manager.ContextManager") as mock_cm_class,
            patch("osprey.infrastructure.respond_node.StateManager") as mock_sm,
            patch("osprey.infrastructure.respond_node._determine_response_mode") as mock_mode,
            patch("osprey.infrastructure.respond_node._get_capabilities_overview") as mock_caps,
            patch("osprey.utils.config.get_interface_context") as mock_interface,
            patch("osprey.infrastructure.respond_node.datetime") as mock_dt,
        ):
            mock_cm_instance = Mock()
            mock_cm_instance.get_summaries.return_value = []
            mock_cm_class.return_value = mock_cm_instance

            mock_sm.get_current_step.return_value = {}
            mock_sm.get_current_step_index.return_value = 0
            mock_mode.return_value = "conversational"
            mock_caps.return_value = "Test"
            mock_interface.return_value = "cli"

            # Mock datetime.now()
            mock_now = Mock()
            mock_now.strftime.return_value = "2025-12-23"
            mock_dt.now.return_value = mock_now

            context = _gather_information(state)

        assert context.current_date == "2025-12-23"
