"""
Tests for orchestrator prompt generation.

This test suite validates that the orchestrator prompt builder correctly:
1. Includes chat history when task_depends_on_chat_history=True
2. Excludes chat history when task_depends_on_chat_history=False
3. Formats chat history appropriately for the orchestrator
4. Includes capability context data with task_objective metadata
5. Maintains the correct prompt structure
"""

from typing import ClassVar

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field

from osprey.context.base import CapabilityContext
from osprey.context.context_manager import ContextManager
from osprey.prompts.defaults.orchestrator import DefaultOrchestratorPromptBuilder

# ===================================================================
# Test Context Classes
# ===================================================================


class TestPVAddressesContext(CapabilityContext):
    """Test context for PV addresses."""

    CONTEXT_TYPE: ClassVar[str] = "PV_ADDRESSES"
    pvs: list[str] = Field(description="List of PV addresses")

    def get_summary(self) -> dict:
        return {"type": "PV Addresses", "count": len(self.pvs)}

    def get_access_details(self, key: str) -> dict:
        return {"pvs": self.pvs}


class TestArchiverDataContext(CapabilityContext):
    """Test context for archiver data."""

    CONTEXT_TYPE: ClassVar[str] = "ARCHIVER_DATA"
    channels: list[str] = Field(description="Channel names")
    data_points: int = Field(description="Number of data points")

    def get_summary(self) -> dict:
        return {"type": "Archiver Data", "channels": self.channels, "points": self.data_points}

    def get_access_details(self, key: str) -> dict:
        return {"channels": self.channels, "data_points": self.data_points}


# ===================================================================
# Test Fixtures
# ===================================================================


@pytest.fixture
def sample_messages():
    """Create sample conversation messages."""
    return [
        HumanMessage(content="What's the weather in San Francisco?"),
        AIMessage(content="The weather in San Francisco is 6.0°C with clear skies."),
        HumanMessage(content="What did I just ask you?"),
    ]


@pytest.fixture
def multi_turn_messages():
    """Create multi-turn conversation with data analysis context."""
    return [
        HumanMessage(content="Plot SR beam current from 10/10/25 to 10/12/25"),
        AIMessage(
            content="Here's the plot of SR beam current for the requested time range. "
            "The data shows stable operation with an average current of 250mA."
        ),
        HumanMessage(content="Now make a correlation analysis using the same time range"),
    ]


# ===================================================================
# Tests for Chat History Inclusion
# ===================================================================


class TestOrchestratorChatHistoryInclusion:
    """Test suite for chat history inclusion in orchestrator prompt."""

    def test_chat_history_included_when_task_depends_on_history(self, sample_messages):
        """Test that chat history is included when task_depends_on_chat_history=True.

        This is the critical test for GitHub issue #111 - ensuring the orchestrator
        can see previous conversation context to understand follow-up queries.
        """
        builder = DefaultOrchestratorPromptBuilder()

        prompt = builder.get_planning_instructions(
            active_capabilities=[],
            context_manager=None,
            task_depends_on_chat_history=True,
            task_depends_on_user_memory=False,
            messages=sample_messages,
        )

        # Verify chat history section is present (with visual separators)
        assert "**CONVERSATION HISTORY**" in prompt
        assert "What's the weather in San Francisco?" in prompt
        assert "6.0°C with clear skies" in prompt
        assert "What did I just ask you?" in prompt

    def test_chat_history_excluded_when_task_does_not_depend_on_history(self, sample_messages):
        """Test that chat history is excluded when task_depends_on_chat_history=False."""
        builder = DefaultOrchestratorPromptBuilder()

        prompt = builder.get_planning_instructions(
            active_capabilities=[],
            context_manager=None,
            task_depends_on_chat_history=False,
            task_depends_on_user_memory=False,
            messages=sample_messages,
        )

        # Verify chat history section is NOT present
        assert "**CONVERSATION HISTORY**" not in prompt
        # The actual message content should also not be present
        assert "What's the weather in San Francisco?" not in prompt

    def test_chat_history_excluded_when_messages_none(self):
        """Test that chat history section is not added when messages is None."""
        builder = DefaultOrchestratorPromptBuilder()

        prompt = builder.get_planning_instructions(
            active_capabilities=[],
            context_manager=None,
            task_depends_on_chat_history=True,
            task_depends_on_user_memory=False,
            messages=None,
        )

        # Verify chat history section is NOT present even with flag=True
        assert "**CONVERSATION HISTORY**" not in prompt

    def test_chat_history_excluded_when_messages_empty(self):
        """Test that chat history section is not added when messages list is empty."""
        builder = DefaultOrchestratorPromptBuilder()

        prompt = builder.get_planning_instructions(
            active_capabilities=[],
            context_manager=None,
            task_depends_on_chat_history=True,
            task_depends_on_user_memory=False,
            messages=[],
        )

        # Verify chat history section is NOT present
        assert "**CONVERSATION HISTORY**" not in prompt

    def test_multi_turn_conversation_preserved(self, multi_turn_messages):
        """Test that multi-turn conversation context is fully preserved.

        This tests the scenario from issue #111 where follow-up queries
        reference previous time ranges or data.
        """
        builder = DefaultOrchestratorPromptBuilder()

        prompt = builder.get_planning_instructions(
            active_capabilities=[],
            context_manager=None,
            task_depends_on_chat_history=True,
            task_depends_on_user_memory=False,
            messages=multi_turn_messages,
        )

        # Verify all conversation turns are present
        assert "Plot SR beam current from 10/10/25 to 10/12/25" in prompt
        assert "250mA" in prompt
        assert "same time range" in prompt

    def test_chat_history_section_has_guidance_text(self, sample_messages):
        """Test that chat history section includes helpful guidance."""
        builder = DefaultOrchestratorPromptBuilder()

        prompt = builder.get_planning_instructions(
            active_capabilities=[],
            context_manager=None,
            task_depends_on_chat_history=True,
            messages=sample_messages,
        )

        # Verify guidance text is present
        assert "conversation history that this task builds upon" in prompt
        assert "references to previous queries" in prompt.lower() or "previous" in prompt


# ===================================================================
# Tests for Context Reuse Guidance
# ===================================================================


class TestOrchestratorContextGuidance:
    """Test suite for context reuse guidance in orchestrator prompt."""

    def test_context_reuse_guidance_included_when_depends_on_history(self):
        """Test that context reuse guidance is included when task depends on history."""
        builder = DefaultOrchestratorPromptBuilder()

        prompt = builder.get_planning_instructions(
            active_capabilities=[],
            context_manager=None,
            task_depends_on_chat_history=True,
            task_depends_on_user_memory=False,
            messages=[HumanMessage(content="test")],
        )

        assert "**CONTEXT REUSE GUIDANCE:**" in prompt
        assert "PRIORITIZE CONTEXT REUSE" in prompt

    def test_context_reuse_guidance_excluded_when_no_dependencies(self):
        """Test that context reuse guidance is excluded when no dependencies."""
        builder = DefaultOrchestratorPromptBuilder()

        prompt = builder.get_planning_instructions(
            active_capabilities=[],
            context_manager=None,
            task_depends_on_chat_history=False,
            task_depends_on_user_memory=False,
            messages=None,
        )

        assert "**CONTEXT REUSE GUIDANCE:**" not in prompt


# ===================================================================
# Tests for Prompt Structure
# ===================================================================


class TestOrchestratorPromptStructure:
    """Test suite for overall orchestrator prompt structure."""

    def test_base_prompt_always_present(self):
        """Test that base prompt sections are always present."""
        builder = DefaultOrchestratorPromptBuilder()

        prompt = builder.get_planning_instructions(
            active_capabilities=[],
            context_manager=None,
            task_depends_on_chat_history=False,
            messages=None,
        )

        # Verify base sections
        assert "expert execution planner" in prompt
        assert "TASK:" in prompt
        assert "PlannedStep" in prompt

    def test_chat_history_before_context_reuse_guidance(self, sample_messages):
        """Test that chat history appears before context reuse guidance.

        This ensures all context-related sections (guidance + available context + principle)
        are kept together, with chat history preceding them for better readability.
        """
        builder = DefaultOrchestratorPromptBuilder()

        prompt = builder.get_planning_instructions(
            active_capabilities=[],
            context_manager=None,
            task_depends_on_chat_history=True,
            messages=sample_messages,
        )

        # Find positions of sections
        history_pos = prompt.find("**CONVERSATION HISTORY**")
        guidance_pos = prompt.find("**CONTEXT REUSE GUIDANCE:**")

        # Chat history should come BEFORE context reuse guidance
        assert history_pos < guidance_pos, (
            "Chat history should appear before context reuse guidance"
        )


# ===================================================================
# Tests for Context Section with Task Objective Metadata
# ===================================================================


class TestOrchestratorContextSection:
    """Test suite for context section including task_objective metadata.

    This tests the feature where task_objective from execution plan steps
    is displayed in the orchestrator prompt to enable intelligent context reuse.
    """

    @pytest.fixture
    def context_manager_with_metadata(self):
        """Create a ContextManager with contexts that have task_objective metadata."""
        state = {"capability_context_data": {}}
        cm = ContextManager(state)

        # Add context with task_objective metadata
        cm.set_context(
            "PV_ADDRESSES",
            "beam_current_pvs",
            TestPVAddressesContext(pvs=["SR:CURRENT:RB", "SR:LIFETIME:RB"]),
            skip_validation=True,
            task_objective="Find PV addresses for beam current monitoring",
        )
        cm.set_context(
            "ARCHIVER_DATA",
            "historical_beam_data",
            TestArchiverDataContext(channels=["SR:CURRENT:RB"], data_points=8640),
            skip_validation=True,
            task_objective="Retrieve historical beam current data from archiver for the last 24 hours",
        )

        return cm

    @pytest.fixture
    def context_manager_without_metadata(self):
        """Create a ContextManager with contexts that have no metadata."""
        state = {"capability_context_data": {}}
        cm = ContextManager(state)

        # Add context without task_objective metadata
        cm.set_context(
            "PV_ADDRESSES",
            "some_pvs",
            TestPVAddressesContext(pvs=["TEST:PV"]),
            skip_validation=True,
        )

        return cm

    def test_context_section_includes_task_objective(self, context_manager_with_metadata):
        """Test that context section includes task_objective when available."""
        builder = DefaultOrchestratorPromptBuilder()

        prompt = builder.get_planning_instructions(
            active_capabilities=[],
            context_manager=context_manager_with_metadata,
            task_depends_on_chat_history=False,
            messages=None,
        )

        # Verify the context section header
        assert "**AVAILABLE CONTEXT (from previous queries):**" in prompt

        # Verify task objectives are included
        assert "Find PV addresses for beam current monitoring" in prompt
        assert "Retrieve historical beam current data from archiver for the last 24 hours" in prompt

        # Verify context keys are present in execution plan input format
        assert '{"PV_ADDRESSES": "beam_current_pvs"}' in prompt
        assert '{"ARCHIVER_DATA": "historical_beam_data"}' in prompt

    def test_context_section_shows_key_without_metadata(self, context_manager_without_metadata):
        """Test that context keys are shown even without metadata."""
        builder = DefaultOrchestratorPromptBuilder()

        prompt = builder.get_planning_instructions(
            active_capabilities=[],
            context_manager=context_manager_without_metadata,
            task_depends_on_chat_history=False,
            messages=None,
        )

        # Verify context key is present even without task_objective
        assert '{"PV_ADDRESSES": "some_pvs"}' in prompt
        assert "**AVAILABLE CONTEXT (from previous queries):**" in prompt

    def test_context_section_includes_reuse_principle(self, context_manager_with_metadata):
        """Test that context section includes the reuse principle guidance."""
        builder = DefaultOrchestratorPromptBuilder()

        prompt = builder.get_planning_instructions(
            active_capabilities=[],
            context_manager=context_manager_with_metadata,
            task_depends_on_chat_history=False,
            messages=None,
        )

        # Verify reuse principle is present with softened tone
        assert "**CONTEXT REUSE PRINCIPLE:**" in prompt
        assert "PREFER reusing existing context" in prompt

    def test_no_context_section_when_empty(self):
        """Test that no context section appears when there's no context data."""
        state = {"capability_context_data": {}}
        empty_cm = ContextManager(state)

        builder = DefaultOrchestratorPromptBuilder()

        prompt = builder.get_planning_instructions(
            active_capabilities=[],
            context_manager=empty_cm,
            task_depends_on_chat_history=False,
            messages=None,
        )

        # Verify no context section when empty
        assert "**AVAILABLE CONTEXT (from previous queries):**" not in prompt

    def test_context_section_skips_internal_keys(self):
        """Test that internal keys like _execution_config are not shown."""
        state = {"capability_context_data": {}}
        cm = ContextManager(state)

        # Add a normal context
        cm.set_context(
            "PV_ADDRESSES",
            "pvs",
            TestPVAddressesContext(pvs=["PV1"]),
            skip_validation=True,
            task_objective="Find PVs",
        )

        # Manually add internal key (simulating framework behavior)
        cm._data["_execution_config"] = {"some": "config"}

        builder = DefaultOrchestratorPromptBuilder()

        prompt = builder.get_planning_instructions(
            active_capabilities=[],
            context_manager=cm,
            task_depends_on_chat_history=False,
            messages=None,
        )

        # Verify PV_ADDRESSES is shown but _execution_config is not
        assert "PV_ADDRESSES" in prompt
        assert "_execution_config" not in prompt

    def test_context_section_multiple_contexts_same_type(self):
        """Test that multiple contexts of same type are all shown with their tasks."""
        state = {"capability_context_data": {}}
        cm = ContextManager(state)

        # Add two contexts of same type with different tasks
        cm.set_context(
            "ARCHIVER_DATA",
            "beam_current_data",
            TestArchiverDataContext(channels=["SR:CURRENT:RB"], data_points=1000),
            skip_validation=True,
            task_objective="Retrieve beam current data",
        )
        cm.set_context(
            "ARCHIVER_DATA",
            "rf_data",
            TestArchiverDataContext(channels=["SR:RF:POWER"], data_points=2000),
            skip_validation=True,
            task_objective="Retrieve RF power data",
        )

        builder = DefaultOrchestratorPromptBuilder()

        prompt = builder.get_planning_instructions(
            active_capabilities=[],
            context_manager=cm,
            task_depends_on_chat_history=False,
            messages=None,
        )

        # Both contexts should be shown with their respective task objectives
        assert '{"ARCHIVER_DATA": "beam_current_data"}' in prompt
        assert "Retrieve beam current data" in prompt
        assert '{"ARCHIVER_DATA": "rf_data"}' in prompt
        assert "Retrieve RF power data" in prompt


# ===================================================================
# Tests for Split Methods (get_step_format, get_planning_strategy, etc.)
# ===================================================================


class TestSplitMethods:
    """Test the decomposed prompt builder methods."""

    def test_get_step_format_contains_planned_step_fields(self):
        """get_step_format() contains all PlannedStep field definitions."""
        builder = DefaultOrchestratorPromptBuilder()
        step_format = builder.get_step_format()

        assert "context_key" in step_format
        assert "capability" in step_format
        assert "task_objective" in step_format
        assert "expected_output" in step_format
        assert "success_criteria" in step_format
        assert "inputs" in step_format
        assert "parameters" in step_format
        assert "PlannedStep" in step_format

    def test_get_planning_strategy_contains_guidelines(self):
        """get_planning_strategy() contains multi-step planning guidelines."""
        builder = DefaultOrchestratorPromptBuilder()
        strategy = builder.get_planning_strategy()

        assert "sequencing" in strategy.lower() or "Dependencies" in strategy
        assert "Cost optimization" in strategy or "cost" in strategy.lower()
        assert "respond" in strategy
        assert "clarify" in strategy

    def test_get_planning_strategy_does_not_contain_reactive(self):
        """get_planning_strategy() does NOT contain reactive mode constraints."""
        builder = DefaultOrchestratorPromptBuilder()
        strategy = builder.get_planning_strategy()

        assert "REACTIVE MODE" not in strategy
        assert "EXACTLY ONE step" not in strategy

    def test_get_reactive_strategy_contains_constraints(self):
        """get_reactive_strategy() contains single-step ReAct constraints."""
        builder = DefaultOrchestratorPromptBuilder()
        strategy = builder.get_reactive_strategy()

        assert "REACTIVE MODE" in strategy
        assert "ONE capability tool" in strategy
        assert "respond" in strategy
        assert "clarify" in strategy

    def test_get_reactive_strategy_does_not_contain_planning(self):
        """get_reactive_strategy() does NOT contain multi-step planning guidelines."""
        builder = DefaultOrchestratorPromptBuilder()
        strategy = builder.get_reactive_strategy()

        assert "ensure proper sequencing" not in strategy
        assert "Cost optimization" not in strategy

    def test_get_instructions_combines_step_format_and_planning_strategy(self):
        """get_instructions() returns get_step_format() + get_planning_strategy()."""
        builder = DefaultOrchestratorPromptBuilder()
        instructions = builder.get_instructions()
        step_format = builder.get_step_format()
        planning_strategy = builder.get_planning_strategy()

        assert step_format in instructions
        assert planning_strategy in instructions


# ===================================================================
# Tests for Reactive Instructions
# ===================================================================


class TestReactiveInstructions:
    """Test suite for get_reactive_instructions() composition."""

    def test_includes_role_definition(self):
        """Reactive instructions include the role definition."""
        builder = DefaultOrchestratorPromptBuilder()
        prompt = builder.get_reactive_instructions(active_capabilities=[])

        assert "expert execution planner" in prompt

    def test_includes_step_format(self):
        """Reactive instructions include step format."""
        builder = DefaultOrchestratorPromptBuilder()
        prompt = builder.get_reactive_instructions(active_capabilities=[])

        assert "PlannedStep" in prompt
        assert "context_key" in prompt

    def test_includes_reactive_strategy(self):
        """Reactive instructions include reactive strategy."""
        builder = DefaultOrchestratorPromptBuilder()
        prompt = builder.get_reactive_instructions(active_capabilities=[])

        assert "REACTIVE MODE" in prompt
        assert "ONE capability tool" in prompt

    def test_does_not_include_planning_strategy(self):
        """Reactive instructions do NOT include multi-step planning guidelines."""
        builder = DefaultOrchestratorPromptBuilder()
        prompt = builder.get_reactive_instructions(active_capabilities=[])

        assert "ensure proper sequencing" not in prompt
        assert "Cost optimization" not in prompt

    def test_does_not_include_task_definition(self):
        """Reactive instructions do NOT include the plan-first task definition."""
        builder = DefaultOrchestratorPromptBuilder()
        prompt = builder.get_reactive_instructions(active_capabilities=[])

        assert "TASK: Create a detailed execution plan" not in prompt

    def test_does_not_include_chat_history(self):
        """Reactive instructions do NOT include chat history sections."""
        builder = DefaultOrchestratorPromptBuilder()
        prompt = builder.get_reactive_instructions(active_capabilities=[])

        assert "**CONVERSATION HISTORY**" not in prompt

    def test_does_not_include_error_context(self):
        """Reactive instructions do NOT include replanning context."""
        builder = DefaultOrchestratorPromptBuilder()
        prompt = builder.get_reactive_instructions(active_capabilities=[])

        assert "**REPLANNING CONTEXT:**" not in prompt

    def test_does_not_include_context_reuse_guidance(self):
        """Reactive instructions do NOT include context reuse guidance."""
        builder = DefaultOrchestratorPromptBuilder()
        prompt = builder.get_reactive_instructions(active_capabilities=[])

        assert "**CONTEXT REUSE GUIDANCE:**" not in prompt

    def test_includes_execution_history(self):
        """Reactive instructions include the execution_history parameter."""
        builder = DefaultOrchestratorPromptBuilder()
        history = "- Step 'beam_channels' (channel_finding): SUCCESS - Find channels"
        prompt = builder.get_reactive_instructions(
            active_capabilities=[],
            execution_history=history,
        )

        assert "# EXECUTION HISTORY" in prompt
        assert history in prompt

    def test_default_execution_history(self):
        """Reactive instructions use default execution history when not provided."""
        builder = DefaultOrchestratorPromptBuilder()
        prompt = builder.get_reactive_instructions(active_capabilities=[])

        assert "No steps executed yet" in prompt

    def test_includes_context_section(self):
        """Reactive instructions include context data when available."""
        state = {"capability_context_data": {}}
        cm = ContextManager(state)
        cm.set_context(
            "PV_ADDRESSES",
            "beam_pvs",
            TestPVAddressesContext(pvs=["SR:CURRENT:RB"]),
            skip_validation=True,
            task_objective="Find beam current PVs",
        )

        builder = DefaultOrchestratorPromptBuilder()
        prompt = builder.get_reactive_instructions(
            active_capabilities=[],
            context_manager=cm,
        )

        assert "**AVAILABLE CONTEXT (from previous queries):**" in prompt
        assert '{"PV_ADDRESSES": "beam_pvs"}' in prompt

    def test_includes_capabilities(self):
        """Reactive instructions include capability sections when present."""
        from unittest.mock import MagicMock

        from osprey.base import OrchestratorGuide

        cap = MagicMock()
        cap.__class__.__name__ = "TestCapability"
        cap.orchestrator_guide = OrchestratorGuide(
            instructions="Use this for testing.",
            examples=[],
            priority=10,
        )

        builder = DefaultOrchestratorPromptBuilder()
        prompt = builder.get_reactive_instructions(active_capabilities=[cap])

        assert "# CAPABILITY PLANNING GUIDELINES" in prompt
        assert "Use this for testing." in prompt
