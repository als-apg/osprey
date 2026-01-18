"""
Tests for orchestrator prompt generation.

This test suite validates that the orchestrator prompt builder correctly:
1. Includes chat history when task_depends_on_chat_history=True
2. Excludes chat history when task_depends_on_chat_history=False
3. Formats chat history appropriately for the orchestrator
4. Includes capability context data
5. Maintains the correct prompt structure
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from osprey.prompts.defaults.orchestrator import DefaultOrchestratorPromptBuilder

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

        prompt = builder.get_system_instructions(
            active_capabilities=[],
            context_manager=None,
            task_depends_on_chat_history=True,
            task_depends_on_user_memory=False,
            messages=sample_messages,
        )

        # Verify chat history section is present
        assert "**CONVERSATION HISTORY:**" in prompt
        assert "What's the weather in San Francisco?" in prompt
        assert "6.0°C with clear skies" in prompt
        assert "What did I just ask you?" in prompt

    def test_chat_history_excluded_when_task_does_not_depend_on_history(self, sample_messages):
        """Test that chat history is excluded when task_depends_on_chat_history=False."""
        builder = DefaultOrchestratorPromptBuilder()

        prompt = builder.get_system_instructions(
            active_capabilities=[],
            context_manager=None,
            task_depends_on_chat_history=False,
            task_depends_on_user_memory=False,
            messages=sample_messages,
        )

        # Verify chat history section is NOT present
        assert "**CONVERSATION HISTORY:**" not in prompt
        # The actual message content should also not be present
        assert "What's the weather in San Francisco?" not in prompt

    def test_chat_history_excluded_when_messages_none(self):
        """Test that chat history section is not added when messages is None."""
        builder = DefaultOrchestratorPromptBuilder()

        prompt = builder.get_system_instructions(
            active_capabilities=[],
            context_manager=None,
            task_depends_on_chat_history=True,
            task_depends_on_user_memory=False,
            messages=None,
        )

        # Verify chat history section is NOT present even with flag=True
        assert "**CONVERSATION HISTORY:**" not in prompt

    def test_chat_history_excluded_when_messages_empty(self):
        """Test that chat history section is not added when messages list is empty."""
        builder = DefaultOrchestratorPromptBuilder()

        prompt = builder.get_system_instructions(
            active_capabilities=[],
            context_manager=None,
            task_depends_on_chat_history=True,
            task_depends_on_user_memory=False,
            messages=[],
        )

        # Verify chat history section is NOT present
        assert "**CONVERSATION HISTORY:**" not in prompt

    def test_multi_turn_conversation_preserved(self, multi_turn_messages):
        """Test that multi-turn conversation context is fully preserved.

        This tests the scenario from issue #111 where follow-up queries
        reference previous time ranges or data.
        """
        builder = DefaultOrchestratorPromptBuilder()

        prompt = builder.get_system_instructions(
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

        prompt = builder.get_system_instructions(
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

        prompt = builder.get_system_instructions(
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

        prompt = builder.get_system_instructions(
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

        prompt = builder.get_system_instructions(
            active_capabilities=[],
            context_manager=None,
            task_depends_on_chat_history=False,
            messages=None,
        )

        # Verify base sections
        assert "expert execution planner" in prompt
        assert "TASK:" in prompt
        assert "PlannedStep" in prompt

    def test_chat_history_before_capability_sections(self, sample_messages):
        """Test that chat history appears before capability sections."""
        builder = DefaultOrchestratorPromptBuilder()

        prompt = builder.get_system_instructions(
            active_capabilities=[],
            context_manager=None,
            task_depends_on_chat_history=True,
            messages=sample_messages,
        )

        # Find positions of sections
        history_pos = prompt.find("**CONVERSATION HISTORY:**")
        guidance_pos = prompt.find("**CONTEXT REUSE GUIDANCE:**")

        # Chat history should come after context reuse guidance
        assert history_pos > guidance_pos, "Chat history should appear after context reuse guidance"
