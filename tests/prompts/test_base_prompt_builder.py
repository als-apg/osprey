"""Tests for base prompt builder deprecation bridges and public API."""

import warnings

from osprey.prompts.base import FrameworkPromptBuilder


class TestDeprecationBridges:
    """Test that old private method overrides still work but emit DeprecationWarning."""

    def test_old_get_dynamic_context_triggers_warning(self):
        """Subclass overriding _get_dynamic_context still works but warns."""

        class LegacyBuilder(FrameworkPromptBuilder):
            def get_role(self):
                return "test role"

            def get_instructions(self):
                return "test instructions"

            def _get_dynamic_context(self, **kwargs):
                return "legacy context"

        builder = LegacyBuilder()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = builder.build_dynamic_context()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "_get_dynamic_context" in str(w[0].message)
        assert result == "legacy context"

    def test_new_build_dynamic_context_no_warning(self):
        """Subclass overriding build_dynamic_context works without warning."""

        class ModernBuilder(FrameworkPromptBuilder):
            def get_role(self):
                return "test role"

            def get_instructions(self):
                return "test instructions"

            def build_dynamic_context(self, **kwargs):
                return "modern context"

        builder = ModernBuilder()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = builder.build_dynamic_context()
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 0
        assert result == "modern context"

    def test_old_get_examples_triggers_warning(self):
        """Subclass overriding _get_examples still works but warns."""

        class LegacyBuilder(FrameworkPromptBuilder):
            def get_role(self):
                return "test role"

            def get_instructions(self):
                return "test instructions"

            def _get_examples(self, **kwargs):
                return ["example1"]

        builder = LegacyBuilder()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = builder.get_examples()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "_get_examples" in str(w[0].message)
        assert result == ["example1"]

    def test_new_get_examples_no_warning(self):
        """Subclass overriding get_examples works without warning."""

        class ModernBuilder(FrameworkPromptBuilder):
            def get_role(self):
                return "test role"

            def get_instructions(self):
                return "test instructions"

            def get_examples(self, **kwargs):
                return ["example1"]

        builder = ModernBuilder()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = builder.get_examples()
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 0
        assert result == ["example1"]

    def test_old_format_examples_triggers_warning(self):
        """Subclass overriding _format_examples still works but warns."""

        class LegacyBuilder(FrameworkPromptBuilder):
            def get_role(self):
                return "test role"

            def get_instructions(self):
                return "test instructions"

            def _format_examples(self, examples):
                return "custom format"

        builder = LegacyBuilder()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = builder.format_examples([])
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "_format_examples" in str(w[0].message)
        assert result == "custom format"

    def test_new_format_examples_no_warning(self):
        """Subclass overriding format_examples works without warning."""

        class ModernBuilder(FrameworkPromptBuilder):
            def get_role(self):
                return "test role"

            def get_instructions(self):
                return "test instructions"

            def format_examples(self, examples):
                return "modern format"

        builder = ModernBuilder()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = builder.format_examples([])
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 0
        assert result == "modern format"


class TestGetSystemInstructions:
    """Test that build_prompt() calls new public method names."""

    def test_composition_includes_role_and_instructions(self):
        """build_prompt() composes from get_role() and get_instructions()."""

        class TestBuilder(FrameworkPromptBuilder):
            def get_role(self):
                return "CUSTOM ROLE"

            def get_instructions(self):
                return "CUSTOM INSTRUCTIONS"

        builder = TestBuilder()
        prompt = builder.build_prompt()
        assert "CUSTOM ROLE" in prompt
        assert "CUSTOM INSTRUCTIONS" in prompt

    def test_build_dynamic_context_called(self):
        """build_prompt() calls build_dynamic_context()."""

        class TestBuilder(FrameworkPromptBuilder):
            def get_role(self):
                return "role"

            def get_instructions(self):
                return "instructions"

            def build_dynamic_context(self, **kwargs):
                return "DYNAMIC CONTEXT HERE"

        builder = TestBuilder()
        prompt = builder.build_prompt()
        assert "DYNAMIC CONTEXT HERE" in prompt

    def test_default_returns_none_for_optional_methods(self):
        """Base class defaults return None for optional methods."""

        class MinimalBuilder(FrameworkPromptBuilder):
            def get_role(self):
                return "role"

            def get_instructions(self):
                return "instructions"

        builder = MinimalBuilder()
        assert builder.get_examples() is None
        assert builder.build_dynamic_context() is None
        assert builder.get_task() is None
