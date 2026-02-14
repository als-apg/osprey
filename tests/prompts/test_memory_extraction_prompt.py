"""Tests for memory extraction prompt builder — examples, guides, and system instructions."""

from osprey.prompts.defaults.memory_extraction import (
    DefaultMemoryExtractionPromptBuilder,
    MemoryContentExtraction,
    MemoryExtractionExample,
)
from osprey.state import MessageUtils


class TestMemoryExtractionPromptAttributes:
    """Test memory extraction prompt builder class attributes."""

    def test_prompt_type(self):
        builder = DefaultMemoryExtractionPromptBuilder()
        assert builder.PROMPT_TYPE == "memory_extraction"

    def test_role_definition(self):
        builder = DefaultMemoryExtractionPromptBuilder()
        role = builder.get_role()
        assert "extraction" in role.lower()

    def test_task_definition(self):
        builder = DefaultMemoryExtractionPromptBuilder()
        task = builder.get_task()
        assert "Extract" in task

    def test_instructions_content(self):
        builder = DefaultMemoryExtractionPromptBuilder()
        instructions = builder.get_instructions()
        assert "save this:" in instructions.lower() or "save" in instructions.lower()
        assert "remember" in instructions.lower()
        assert "found=false" in instructions.lower()


class TestMemoryExtractionExamples:
    """Test that default examples load correctly."""

    def test_examples_loaded(self):
        builder = DefaultMemoryExtractionPromptBuilder()
        assert len(builder.examples) == 6

    def test_positive_examples_exist(self):
        builder = DefaultMemoryExtractionPromptBuilder()
        positive = [e for e in builder.examples if e.expected_output.found]
        assert len(positive) == 4

    def test_negative_examples_exist(self):
        builder = DefaultMemoryExtractionPromptBuilder()
        negative = [e for e in builder.examples if not e.expected_output.found]
        assert len(negative) == 2

    def test_get_examples_returns_list(self):
        builder = DefaultMemoryExtractionPromptBuilder()
        examples = builder.get_examples()
        assert isinstance(examples, list)
        assert len(examples) == 6

    def test_format_examples_produces_string(self):
        builder = DefaultMemoryExtractionPromptBuilder()
        examples = builder.get_examples()
        formatted = builder.format_examples(examples)
        assert isinstance(formatted, str)
        assert "Expected Output:" in formatted

    def test_example_format_for_prompt(self):
        example = MemoryExtractionExample(
            messages=[MessageUtils.create_user_message("Remember that X is important")],
            expected_output=MemoryContentExtraction(
                content="X is important",
                found=True,
                explanation="User asked to remember a fact",
            ),
        )
        formatted = example.format_for_prompt()
        assert "Chat History:" in formatted
        assert "Expected Output:" in formatted
        assert "X is important" in formatted


class TestMemoryExtractionDynamicContext:
    """Test build_dynamic_context for response format section."""

    def test_returns_json_format_instructions(self):
        builder = DefaultMemoryExtractionPromptBuilder()
        ctx = builder.build_dynamic_context()
        assert "JSON" in ctx
        assert '"content"' in ctx
        assert '"found"' in ctx
        assert '"explanation"' in ctx


class TestMemoryExtractionSystemInstructions:
    """Test build_prompt composition."""

    def test_includes_role(self):
        builder = DefaultMemoryExtractionPromptBuilder()
        prompt = builder.build_prompt()
        assert "extraction" in prompt.lower()

    def test_includes_examples(self):
        builder = DefaultMemoryExtractionPromptBuilder()
        prompt = builder.build_prompt()
        assert "EXAMPLES:" in prompt

    def test_includes_dynamic_context(self):
        builder = DefaultMemoryExtractionPromptBuilder()
        prompt = builder.build_prompt()
        assert "JSON" in prompt


class TestMemoryExtractionGuides:
    """Test orchestrator and classifier guides."""

    def test_orchestrator_guide_exists(self):
        builder = DefaultMemoryExtractionPromptBuilder()
        guide = builder.get_orchestrator_guide()
        assert guide is not None
        assert guide.instructions
        assert len(guide.examples) == 2

    def test_orchestrator_guide_has_save_example(self):
        builder = DefaultMemoryExtractionPromptBuilder()
        guide = builder.get_orchestrator_guide()
        save_example = guide.examples[0]
        assert "save" in save_example.step["task_objective"].lower()

    def test_orchestrator_guide_has_show_example(self):
        builder = DefaultMemoryExtractionPromptBuilder()
        guide = builder.get_orchestrator_guide()
        show_example = guide.examples[1]
        assert "show" in show_example.step["task_objective"].lower()

    def test_classifier_guide_exists(self):
        builder = DefaultMemoryExtractionPromptBuilder()
        guide = builder.get_classifier_guide()
        assert guide is not None
        assert guide.instructions
        assert len(guide.examples) == 5

    def test_classifier_all_positive(self):
        builder = DefaultMemoryExtractionPromptBuilder()
        guide = builder.get_classifier_guide()
        assert all(e.result is True for e in guide.examples)


class TestMemoryClassificationPrompt:
    """Test get_memory_classification_prompt helper."""

    def test_returns_string(self):
        builder = DefaultMemoryExtractionPromptBuilder()
        prompt = builder.get_memory_classification_prompt()
        assert isinstance(prompt, str)

    def test_includes_save_and_retrieve(self):
        builder = DefaultMemoryExtractionPromptBuilder()
        prompt = builder.get_memory_classification_prompt()
        assert "SAVE" in prompt
        assert "RETRIEVE" in prompt


class TestMemoryExtractionOverrides:
    """Test that subclass overrides work."""

    def test_subclass_role_override(self):
        class CustomBuilder(DefaultMemoryExtractionPromptBuilder):
            def get_role(self) -> str:
                return "You are a facility memory specialist."

        builder = CustomBuilder()
        prompt = builder.build_prompt()
        assert "facility memory specialist" in prompt

    def test_subclass_custom_examples(self):
        class CustomBuilder(DefaultMemoryExtractionPromptBuilder):
            def get_examples(self, **kwargs):
                return [self.examples[0]]  # Only first example

        builder = CustomBuilder()
        examples = builder.get_examples()
        assert len(examples) == 1
