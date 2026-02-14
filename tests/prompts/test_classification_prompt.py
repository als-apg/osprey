"""Tests for classification prompt builder — public API and system instructions."""

from osprey.prompts.defaults.classification import DefaultClassificationPromptBuilder


class TestClassificationPromptAttributes:
    """Test classification prompt builder class attributes."""

    def test_prompt_type(self):
        builder = DefaultClassificationPromptBuilder()
        assert builder.PROMPT_TYPE == "classification"

    def test_role_definition(self):
        builder = DefaultClassificationPromptBuilder()
        role = builder.get_role()
        assert "classification" in role.lower()

    def test_task_definition(self):
        builder = DefaultClassificationPromptBuilder()
        task = builder.get_task()
        assert "capability" in task.lower()

    def test_instructions_require_json(self):
        builder = DefaultClassificationPromptBuilder()
        instructions = builder.get_instructions()
        assert "JSON" in instructions
        assert "is_match" in instructions


class TestClassificationDynamicContext:
    """Test build_dynamic_context with various input combinations."""

    def test_capability_instructions_included(self):
        builder = DefaultClassificationPromptBuilder()
        ctx = builder.build_dynamic_context(capability_instructions="Test capability description")
        assert "Test capability description" in ctx

    def test_classifier_examples_included(self):
        builder = DefaultClassificationPromptBuilder()
        ctx = builder.build_dynamic_context(classifier_examples="Example 1: ...")
        assert "Example 1:" in ctx

    def test_context_dict_included(self):
        builder = DefaultClassificationPromptBuilder()
        ctx = builder.build_dynamic_context(context={"key": "value"})
        assert "key" in ctx
        assert "value" in ctx

    def test_previous_failure_included(self):
        builder = DefaultClassificationPromptBuilder()
        ctx = builder.build_dynamic_context(previous_failure="Timed out on last attempt")
        assert "Timed out on last attempt" in ctx

    def test_empty_inputs_returns_empty(self):
        builder = DefaultClassificationPromptBuilder()
        ctx = builder.build_dynamic_context()
        assert ctx == ""

    def test_all_inputs_combined(self):
        builder = DefaultClassificationPromptBuilder()
        ctx = builder.build_dynamic_context(
            capability_instructions="Cap instructions",
            classifier_examples="Examples here",
            context={"prior": "data"},
            previous_failure="Last approach failed",
        )
        assert "Cap instructions" in ctx
        assert "Examples here" in ctx
        assert "prior" in ctx
        assert "Last approach failed" in ctx


class TestClassificationSystemInstructions:
    """Test build_prompt composition."""

    def test_includes_role_and_task(self):
        builder = DefaultClassificationPromptBuilder()
        prompt = builder.build_prompt()
        assert builder.get_role() in prompt
        assert builder.get_task() in prompt

    def test_includes_instructions(self):
        builder = DefaultClassificationPromptBuilder()
        prompt = builder.build_prompt()
        assert "is_match" in prompt

    def test_includes_dynamic_context(self):
        builder = DefaultClassificationPromptBuilder()
        prompt = builder.build_prompt(capability_instructions="Detect weather queries")
        assert "Detect weather queries" in prompt

    def test_subclass_role_override(self):
        class CustomClassifier(DefaultClassificationPromptBuilder):
            def get_role(self) -> str:
                return "You are a specialized weather classifier."

        builder = CustomClassifier()
        prompt = builder.build_prompt()
        assert "weather classifier" in prompt

    def test_subclass_instructions_override(self):
        class CustomClassifier(DefaultClassificationPromptBuilder):
            def get_instructions(self) -> str:
                return "Return only true or false."

        builder = CustomClassifier()
        prompt = builder.build_prompt()
        assert "Return only true or false." in prompt
