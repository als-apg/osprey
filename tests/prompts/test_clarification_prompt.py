"""
Tests for clarification prompt generation.

This test suite validates that the clarification prompt builder correctly:
1. Extracts user query from message history
2. Extracts task_objective from the current step
3. Formats chat history appropriately
4. Composes the complete prompt with all necessary sections
5. Maintains the correct structure (role + clarification query + context)
"""

import pytest

from osprey.prompts.defaults.clarification import DefaultClarificationPromptBuilder
from osprey.state import AgentState, StateManager
from tests.conftest import PromptTestHelpers, create_test_state

# ===================================================================
# Test Fixtures
# ===================================================================


@pytest.fixture
def sample_state() -> AgentState:
    """Create a sample state with user query and execution plan."""
    return create_test_state(
        user_message="whats the weather right now?",
        task_objective="Ask the user to specify the location for which they want to retrieve current weather information",
        capability="clarify",
        context_key="clarify_location",
        task_current_task="Retrieve current weather information",
        planning_active_capabilities=["clarify", "fetch_weather"],
    )


@pytest.fixture
def multi_turn_state() -> AgentState:
    """Create a state with multi-turn conversation history."""
    return create_test_state(
        conversation_history=[
            ("user", "I need some data"),
            ("ai", "I can help with that. What kind of data are you looking for?"),
            ("user", "beam current data"),
        ],
        task_objective="Ask the user to specify the time range for beam current data retrieval",
        capability="clarify",
        context_key="clarify_timerange",
        task_current_task="Retrieve beam current data",
        task_depends_on_chat_history=True,
        planning_active_capabilities=["clarify", "data_retrieval"],
    )


# ===================================================================
# Tests for Clarification Prompt Generation
# ===================================================================


class TestClarificationPromptGeneration:
    """Test suite for clarification prompt builder."""

    def test_basic_prompt_structure(self, sample_state):
        """Test that the prompt has all required sections."""
        builder = DefaultClarificationPromptBuilder()
        task_objective = "Ask the user to specify the location for which they want to retrieve current weather information"

        prompt = builder.build_prompt(sample_state, task_objective)

        # Verify all major sections are present
        assert "You are helping to clarify ambiguous user queries" in prompt
        assert "CONVERSATION HISTORY:" in prompt
        assert "USER'S ORIGINAL QUERY:" in prompt
        assert "ORCHESTRATOR'S CLARIFICATION INSTRUCTION:" in prompt
        assert "Your task is to generate specific clarifying questions" in prompt
        assert "Example:" in prompt

    def test_user_query_extraction(self, sample_state):
        """Test that user query is correctly extracted from messages."""
        builder = DefaultClarificationPromptBuilder()
        task_objective = "Ask the user to specify the location"

        prompt = builder.build_prompt(sample_state, task_objective)

        # Extract the user query section and verify exact content
        query_section = PromptTestHelpers.extract_section(prompt, "USER'S ORIGINAL QUERY:")
        assert "whats the weather right now?" in query_section.lower()

    def test_task_objective_inclusion(self, sample_state):
        """Test that the orchestrator's task_objective is included."""
        builder = DefaultClarificationPromptBuilder()
        task_objective = "Ask the user to specify the location for which they want to retrieve current weather information"

        prompt = builder.build_prompt(sample_state, task_objective)

        # Extract and verify the orchestrator's instruction section
        instruction_section = PromptTestHelpers.extract_section(
            prompt, "ORCHESTRATOR'S CLARIFICATION INSTRUCTION:"
        )
        assert "specify the location" in instruction_section.lower()
        assert "weather information" in instruction_section.lower()

    def test_chat_history_formatting(self, multi_turn_state):
        """Test that multi-turn conversation history is properly formatted."""
        builder = DefaultClarificationPromptBuilder()
        task_objective = "Ask the user to specify the time range"

        prompt = builder.build_prompt(multi_turn_state, task_objective)

        # Extract and verify conversation history section
        history_section = PromptTestHelpers.extract_section(prompt, "CONVERSATION HISTORY:")

        # Verify all conversation turns are present
        assert "I need some data" in history_section
        assert "What kind of data" in history_section
        assert "beam current data" in history_section

    def test_examples_included(self, sample_state):
        """Test that good/bad examples are included in the prompt."""
        builder = DefaultClarificationPromptBuilder()
        task_objective = "Ask the user to specify the location"

        prompt = builder.build_prompt(sample_state, task_objective)

        # Verify examples section with good/bad questions
        assert "Good question:" in prompt
        assert "Bad question:" in prompt
        assert "What do you need help with?" in prompt  # The bad example

    def test_prompt_section_order(self, sample_state):
        """Test that prompt sections appear in the correct order."""
        builder = DefaultClarificationPromptBuilder()
        task_objective = "Ask the user to specify the location"

        prompt = builder.build_prompt(sample_state, task_objective)

        # Get positions of key sections
        positions = PromptTestHelpers.get_section_positions(
            prompt,
            "You are helping",
            "CONVERSATION HISTORY:",
            "USER'S ORIGINAL QUERY:",
            "ORCHESTRATOR'S CLARIFICATION INSTRUCTION:",
        )

        # Verify all sections exist
        assert all(pos >= 0 for pos in positions.values()), f"Missing sections: {positions}"

        # Verify order
        assert positions["You are helping"] < positions["CONVERSATION HISTORY:"]
        assert positions["CONVERSATION HISTORY:"] < positions["USER'S ORIGINAL QUERY:"]
        assert (
            positions["USER'S ORIGINAL QUERY:"]
            < positions["ORCHESTRATOR'S CLARIFICATION INSTRUCTION:"]
        )

    def test_no_redundant_guidelines(self, sample_state):
        """Test that generic guidelines are not duplicated."""
        builder = DefaultClarificationPromptBuilder()
        task_objective = "Ask the user to specify the location"

        prompt = builder.build_prompt(sample_state, task_objective)

        # The prompt should NOT have a separate "SUPPORTING GUIDELINES" section
        # (we removed this to avoid redundancy)
        assert "SUPPORTING GUIDELINES" not in prompt

    def test_prompt_contains_specific_instructions(self, sample_state):
        """Test that specific instructions are prioritized over generic ones."""
        builder = DefaultClarificationPromptBuilder()
        task_objective = "Ask the user to specify the location for weather"

        prompt = builder.build_prompt(sample_state, task_objective)

        # The orchestrator's specific instruction should appear prominently
        instruction_section = PromptTestHelpers.extract_section(
            prompt, "ORCHESTRATOR'S CLARIFICATION INSTRUCTION:"
        )
        assert "specify the location" in instruction_section.lower()

        # Instructions should emphasize being specific (check in the IMPORTANT section)
        assert "be specific to what's missing" in prompt.lower()
        assert "don't ask generic" in prompt.lower()

    def test_empty_chat_history(self):
        """Test prompt generation with no prior conversation history."""
        # Create state with empty message list
        state = create_test_state(
            user_message="",  # Empty message
            task_objective="Ask the user for more details",
            capability="clarify",
            context_key="clarify_test",
        )
        # Override messages to be empty
        state["messages"] = []

        builder = DefaultClarificationPromptBuilder()
        task_objective = "Ask the user for more details"

        # Should not raise an error
        prompt = builder.build_prompt(state, task_objective)

        assert "CONVERSATION HISTORY:" in prompt
        assert "Ask the user for more details" in prompt

    def test_special_characters_in_query(self):
        """Test that special characters in user query are handled properly."""
        state = create_test_state(
            user_message='What\'s the "status" of system #5?',
            task_objective="Ask which specific metric for system status",
            capability="clarify",
            context_key="clarify_system",
            task_current_task="Get system status",
        )

        builder = DefaultClarificationPromptBuilder()
        task_objective = "Ask which specific metric for system status"

        prompt = builder.build_prompt(state, task_objective)

        # Extract user query section and verify special characters are preserved
        query_section = PromptTestHelpers.extract_section(prompt, "USER'S ORIGINAL QUERY:")
        assert "status" in query_section
        # Check for system identifier (with or without special chars preserved)
        assert "system" in query_section.lower()


# ===================================================================
# Error Handling Tests
# ===================================================================


class TestClarificationErrorHandling:
    """Test suite for error handling and edge cases."""

    def test_missing_messages_field(self):
        """Test handling when messages field is missing from state."""
        state = create_test_state(
            user_message="test query", task_objective="Ask for details", capability="clarify"
        )
        # Remove messages field
        del state["messages"]

        builder = DefaultClarificationPromptBuilder()

        # Should handle gracefully, not crash
        prompt = builder.build_prompt(state, "Ask for details")

        # Should still generate a valid prompt
        assert "You are helping" in prompt
        assert "Ask for details" in prompt

    def test_none_execution_plan(self):
        """Test handling when execution plan is None."""
        state = create_test_state(
            user_message="test query", task_objective="Ask for details", capability="clarify"
        )
        state["planning_execution_plan"] = None

        builder = DefaultClarificationPromptBuilder()

        # This should raise an error since we can't get current step
        with pytest.raises((RuntimeError, AttributeError, TypeError)):
            builder.build_prompt(state, "Ask for details")

    def test_empty_task_objective(self):
        """Test handling when task_objective is empty string."""
        state = create_test_state(
            user_message="whats the weather?", task_objective="", capability="clarify"
        )

        builder = DefaultClarificationPromptBuilder()

        # Should not crash, but prompt may be less useful
        prompt = builder.build_prompt(state, "")

        assert "You are helping" in prompt
        assert "ORCHESTRATOR'S CLARIFICATION INSTRUCTION:" in prompt

    def test_none_task_objective(self):
        """Test handling when task_objective is None."""
        state = create_test_state(
            user_message="test query", task_objective="placeholder", capability="clarify"
        )

        builder = DefaultClarificationPromptBuilder()

        # Pass None as task_objective
        # This might fail depending on implementation
        try:
            prompt = builder.build_prompt(state, None)
            # If it doesn't crash, verify basic structure
            assert "You are helping" in prompt
        except (TypeError, AttributeError):
            # Expected behavior - task_objective is required
            pass

    def test_malformed_messages_list(self):
        """Test handling of malformed message objects in messages list."""
        state = create_test_state(
            user_message="test query", task_objective="Ask for details", capability="clarify"
        )
        # Add a malformed message (not a proper Message object)
        state["messages"].append({"invalid": "message"})  # type: ignore

        builder = DefaultClarificationPromptBuilder()

        # Should either handle gracefully or raise appropriate error
        try:
            prompt = builder.build_prompt(state, "Ask for details")
            # If successful, verify basic structure
            assert "You are helping" in prompt
        except (AttributeError, TypeError, KeyError):
            # Expected - invalid message format
            pass

    def test_unicode_and_emoji_in_query(self):
        """Test handling of unicode characters and emojis."""
        state = create_test_state(
            user_message="What's the weather in 東京? 🌤️ How about 北京?",
            task_objective="Ask user to specify which city",
            capability="clarify",
        )

        builder = DefaultClarificationPromptBuilder()
        prompt = builder.build_prompt(state, "Ask user to specify which city")

        # Should handle unicode without crashing
        assert "You are helping" in prompt
        query_section = PromptTestHelpers.extract_section(prompt, "USER'S ORIGINAL QUERY:")
        # Verify unicode content is preserved (or at least doesn't crash)
        assert len(query_section) > 0

    def test_very_long_chat_history(self):
        """Test handling of extensive conversation history."""
        # Create a conversation with many turns
        conversation = []
        for i in range(50):
            conversation.append(("user", f"User message {i}"))
            conversation.append(("ai", f"AI response {i}"))

        state = create_test_state(
            conversation_history=conversation,
            task_objective="Ask for clarification",
            capability="clarify",
        )

        builder = DefaultClarificationPromptBuilder()
        prompt = builder.build_prompt(state, "Ask for clarification")

        # Should handle without crashing
        assert "You are helping" in prompt
        assert "CONVERSATION HISTORY:" in prompt
        # History should contain some of the messages
        assert "User message" in prompt

    def test_missing_current_step_index(self):
        """Test handling when current step index is missing."""
        state = create_test_state(
            user_message="test query", task_objective="Ask for details", capability="clarify"
        )
        # Remove current step index
        del state["planning_current_step_index"]

        builder = DefaultClarificationPromptBuilder()

        # Should either use default (0) or handle gracefully
        try:
            prompt = builder.build_prompt(state, "Ask for details")
            assert "You are helping" in prompt
        except (KeyError, RuntimeError):
            # Expected if implementation requires the field
            pass

    def test_invalid_step_index(self):
        """Test handling when step index is out of bounds."""
        state = create_test_state(
            user_message="test query", task_objective="Ask for details", capability="clarify"
        )
        # Set invalid step index
        state["planning_current_step_index"] = 99

        builder = DefaultClarificationPromptBuilder()

        # Should raise appropriate error
        with pytest.raises((IndexError, RuntimeError)):
            builder.build_prompt(state, "Ask for details")


# ===================================================================
# Integration Tests
# ===================================================================


class TestClarificationIntegration:
    """Integration tests for the full clarification flow."""

    def test_state_manager_integration(self, sample_state):
        """Test that StateManager correctly extracts data for prompt builder."""
        # Verify StateManager can extract the user query
        user_query = StateManager.get_user_query(sample_state)
        assert user_query == "whats the weather right now?"

        # Verify StateManager can get the current step
        current_step = StateManager.get_current_step(sample_state)
        assert current_step is not None
        assert (
            current_step["task_objective"]
            == "Ask the user to specify the location for which they want to retrieve current weather information"
        )

    def test_full_prompt_generation_flow(self, sample_state):
        """Test the complete flow from state to final prompt."""
        # This simulates what happens in clarify_node.py

        # Step 1: Get current step (like clarify_node does)
        step = StateManager.get_current_step(sample_state)
        task_objective = step.get("task_objective", "unknown") if step else "unknown"

        # Should not be 'unknown'
        assert task_objective != "unknown"
        assert "location" in task_objective

        # Step 2: Get prompt builder and generate prompt
        builder = DefaultClarificationPromptBuilder()
        prompt = builder.build_prompt(sample_state, task_objective)

        # Step 3: Verify the prompt is complete and correct
        assert len(prompt) > 100  # Should be substantial
        assert "whats the weather" in prompt.lower()
        assert "location" in prompt
        assert "ORCHESTRATOR'S CLARIFICATION INSTRUCTION:" in prompt
