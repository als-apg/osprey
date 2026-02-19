"""Tests for task extraction prompt builder — bug fix verification and section builders."""

from osprey.prompts.defaults.task_extraction import (
    DefaultTaskExtractionPromptBuilder,
)
from osprey.state import MessageUtils


class TestTaskExtractionBugFix:
    """Verify that get_role() and get_instructions() overrides take effect."""

    def test_default_role_appears_in_prompt(self):
        """Default role definition appears in the final prompt."""
        builder = DefaultTaskExtractionPromptBuilder()
        messages = [MessageUtils.create_user_message("test query")]
        prompt = builder.build_prompt(messages)
        assert "Convert chat conversations into actionable task descriptions" in prompt

    def test_default_instructions_appear_in_prompt(self):
        """Default instructions appear in the final prompt."""
        builder = DefaultTaskExtractionPromptBuilder()
        messages = [MessageUtils.create_user_message("test query")]
        prompt = builder.build_prompt(messages)
        assert "self-contained task descriptions" in prompt

    def test_subclass_role_override_takes_effect(self):
        """Subclass overriding get_role() produces a prompt with the custom role."""

        class CustomBuilder(DefaultTaskExtractionPromptBuilder):
            def get_role(self) -> str:
                return "You are a weather task extraction specialist."

        builder = CustomBuilder()
        messages = [MessageUtils.create_user_message("What's the weather?")]
        prompt = builder.build_prompt(messages)
        assert "weather task extraction specialist" in prompt
        # Original hardcoded role should NOT appear
        assert "task extraction system that analyzes chat history" not in prompt

    def test_subclass_instructions_override_takes_effect(self):
        """Subclass overriding get_instructions() produces a prompt with custom instructions."""

        class CustomBuilder(DefaultTaskExtractionPromptBuilder):
            def get_instructions(self) -> str:
                return "Focus on weather-related task extraction only."

        builder = CustomBuilder()
        messages = [MessageUtils.create_user_message("What's the weather?")]
        prompt = builder.build_prompt(messages)
        assert "weather-related task extraction" in prompt

    def test_include_default_examples_false(self):
        """Setting include_default_examples=False produces prompt with no examples."""
        builder = DefaultTaskExtractionPromptBuilder(include_default_examples=False)
        assert len(builder.examples) == 0
        messages = [MessageUtils.create_user_message("test")]
        prompt = builder.build_prompt(messages)
        # Should still have the structure but no example content
        assert "## Examples:" in prompt


class TestTaskExtractionSectionBuilders:
    """Test the individual section builder methods."""

    def test_build_examples_section(self):
        """build_examples_section() formats examples."""
        builder = DefaultTaskExtractionPromptBuilder()
        section = builder.build_examples_section()
        assert "## Example 1:" in section
        assert "Expected Output:" in section

    def test_build_chat_history_section(self):
        """build_chat_history_section() formats messages."""
        builder = DefaultTaskExtractionPromptBuilder()
        messages = [
            MessageUtils.create_user_message("Hello"),
            MessageUtils.create_assistant_message("Hi there"),
        ]
        section = builder.build_chat_history_section(messages)
        assert "Hello" in section
        assert "Hi there" in section

    def test_build_data_source_section_empty(self):
        """build_data_source_section() returns empty string when no data."""
        builder = DefaultTaskExtractionPromptBuilder()
        result = builder.build_data_source_section(None)
        assert result == ""

    def test_prompt_structure_has_all_sections(self):
        """Final prompt contains all expected sections."""
        builder = DefaultTaskExtractionPromptBuilder()
        messages = [MessageUtils.create_user_message("test query")]
        prompt = builder.build_prompt(messages)

        assert "## Guidelines:" in prompt
        assert "## Examples:" in prompt
        assert "## Current Chat History:" in prompt
        assert "## User Memory:" in prompt
        assert "Now extract the task" in prompt
