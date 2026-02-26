"""Tests for Python capability prompt builder — public API, guides, and overrides."""

from osprey.prompts.defaults.python import DefaultPythonPromptBuilder


class TestPythonPromptAttributes:
    """Test Python prompt builder class attributes."""

    def test_prompt_type(self):
        builder = DefaultPythonPromptBuilder()
        assert builder.PROMPT_TYPE == "python"

    def test_role_definition(self):
        builder = DefaultPythonPromptBuilder()
        role = builder.get_role()
        assert "Python" in role

    def test_task_definition_is_none(self):
        builder = DefaultPythonPromptBuilder()
        assert builder.get_task() is None

    def test_instructions_content(self):
        builder = DefaultPythonPromptBuilder()
        instructions = builder.get_instructions()
        assert "imports" in instructions.lower()
        assert "results" in instructions
        assert "executable" in instructions.lower()


class TestPythonOrchestratorGuide:
    """Test orchestrator guide content."""

    def test_guide_exists(self):
        builder = DefaultPythonPromptBuilder()
        guide = builder.get_orchestrator_guide()
        assert guide is not None

    def test_guide_has_three_examples(self):
        builder = DefaultPythonPromptBuilder()
        guide = builder.get_orchestrator_guide()
        assert len(guide.examples) == 3

    def test_guide_instructions_content(self):
        builder = DefaultPythonPromptBuilder()
        guide = builder.get_orchestrator_guide()
        assert "python" in guide.instructions.lower()
        assert "calculation" in guide.instructions.lower()

    def test_guide_priority(self):
        builder = DefaultPythonPromptBuilder()
        guide = builder.get_orchestrator_guide()
        assert guide.priority == 40

    def test_examples_use_python_capability(self):
        builder = DefaultPythonPromptBuilder()
        guide = builder.get_orchestrator_guide()
        for example in guide.examples:
            assert example.step["capability"] == "python"

    def test_examples_output_type(self):
        builder = DefaultPythonPromptBuilder()
        guide = builder.get_orchestrator_guide()
        for example in guide.examples:
            assert "PYTHON_RESULTS" in example.step["expected_output"]


class TestPythonClassifierGuide:
    """Test classifier guide content."""

    def test_guide_exists(self):
        builder = DefaultPythonPromptBuilder()
        guide = builder.get_classifier_guide()
        assert guide is not None

    def test_guide_has_examples(self):
        builder = DefaultPythonPromptBuilder()
        guide = builder.get_classifier_guide()
        assert len(guide.examples) == 10

    def test_guide_has_positive_examples(self):
        builder = DefaultPythonPromptBuilder()
        guide = builder.get_classifier_guide()
        positive = [e for e in guide.examples if e.result is True]
        assert len(positive) >= 5

    def test_guide_has_negative_examples(self):
        builder = DefaultPythonPromptBuilder()
        guide = builder.get_classifier_guide()
        negative = [e for e in guide.examples if e.result is False]
        assert len(negative) >= 4


class TestPythonPromptOverrides:
    """Test that subclass overrides work."""

    def test_subclass_role_override(self):
        class CustomPython(DefaultPythonPromptBuilder):
            def get_role(self) -> str:
                return "You are a scientific computing Python developer."

        builder = CustomPython()
        assert "scientific computing" in builder.get_role()

    def test_subclass_instructions_override(self):
        class CustomPython(DefaultPythonPromptBuilder):
            def get_instructions(self) -> str:
                return "Generate NumPy-based code for data analysis."

        builder = CustomPython()
        prompt = builder.build_prompt()
        assert "NumPy-based" in prompt
