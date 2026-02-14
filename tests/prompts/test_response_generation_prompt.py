"""Tests for response generation prompt builder — public API verification."""

from types import SimpleNamespace

from osprey.prompts.defaults.response_generation import (
    DefaultResponseGenerationPromptBuilder,
)


class TestResponseGenerationPublicAPI:
    """Test that public methods are accessible and work correctly."""

    def test_get_conversational_guidelines(self):
        """get_conversational_guidelines() returns a list of strings."""
        builder = DefaultResponseGenerationPromptBuilder()
        guidelines = builder.get_conversational_guidelines()
        assert isinstance(guidelines, list)
        assert len(guidelines) > 0
        assert all(isinstance(g, str) for g in guidelines)

    def test_build_execution_section_successful(self):
        """build_execution_section() handles successful execution."""
        builder = DefaultResponseGenerationPromptBuilder()
        info = SimpleNamespace(
            execution_history=[
                {"success": True, "task_objective": "Fetch data", "result_summary": "OK"}
            ]
        )
        section = builder.build_execution_section(info)
        assert "EXECUTION SUMMARY" in section
        assert "Fetch data" in section

    def test_build_execution_section_terminated(self):
        """build_execution_section() handles terminated execution."""
        builder = DefaultResponseGenerationPromptBuilder()
        info = SimpleNamespace(
            is_killed=True,
            kill_reason="Timeout",
            execution_history=[
                {"success": False, "task_objective": "Slow task", "result_summary": "Timed out"}
            ],
            total_steps_executed=1,
            execution_start_time="2024-01-01",
            reclassification_count=0,
        )
        section = builder.build_execution_section(info)
        assert "Terminated" in section
        assert "Timeout" in section

    def test_build_data_section(self):
        """build_data_section() formats relevant context."""
        builder = DefaultResponseGenerationPromptBuilder()
        context = [{"type": "TestData", "value": "42"}]
        section = builder.build_data_section(context)
        assert "RETRIEVED DATA" in section

    def test_format_context_data_empty(self):
        """format_context_data() handles empty list."""
        builder = DefaultResponseGenerationPromptBuilder()
        result = builder.format_context_data([])
        assert "No context data" in result

    def test_build_capabilities_section(self):
        """build_capabilities_section() wraps overview text."""
        builder = DefaultResponseGenerationPromptBuilder()
        section = builder.build_capabilities_section("Can do X and Y")
        assert "SYSTEM CAPABILITIES" in section
        assert "Can do X and Y" in section

    def test_build_chat_history_section(self):
        """build_chat_history_section() includes conversation history."""
        builder = DefaultResponseGenerationPromptBuilder()
        section = builder.build_chat_history_section("User: Hello\nAI: Hi")
        assert "CONVERSATION HISTORY" in section
        assert "User: Hello" in section

    def test_build_guidelines_section(self):
        """build_guidelines_section() produces guidelines."""
        builder = DefaultResponseGenerationPromptBuilder()
        info = SimpleNamespace()
        section = builder.build_guidelines_section(info)
        assert "GUIDELINES" in section
        assert "clear, accurate response" in section

    def test_build_dynamic_context_integrates_sections(self):
        """build_dynamic_context() assembles sections from info object."""
        builder = DefaultResponseGenerationPromptBuilder()
        info = SimpleNamespace(
            chat_history="User: Hello",
            relevant_context=[{"type": "Data", "value": "test"}],
        )
        result = builder.build_dynamic_context(current_task="Test task", info=info)
        assert "CURRENT TASK: Test task" in result
        assert "CONVERSATION HISTORY" in result
        assert "RETRIEVED DATA" in result


class TestResponseGenerationOverride:
    """Test that subclass overrides of public methods work."""

    def test_override_conversational_guidelines(self):
        """Subclass can override get_conversational_guidelines()."""

        class CustomBuilder(DefaultResponseGenerationPromptBuilder):
            def get_conversational_guidelines(self):
                return ["Be domain-specific", "Use technical jargon"]

        builder = CustomBuilder()
        guidelines = builder.get_conversational_guidelines()
        assert "Be domain-specific" in guidelines

    def test_override_build_chat_history_section(self):
        """Subclass can override build_chat_history_section()."""

        class CustomBuilder(DefaultResponseGenerationPromptBuilder):
            def build_chat_history_section(self, chat_history):
                return f"CUSTOM HISTORY:\n{chat_history}"

        builder = CustomBuilder()
        section = builder.build_chat_history_section("test")
        assert "CUSTOM HISTORY" in section
