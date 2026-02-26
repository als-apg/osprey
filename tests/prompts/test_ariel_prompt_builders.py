"""Tests for ARIEL default prompt builders.

Validates that the 2 prompt builders (agent, rag) provide valid content
and implement the required interface.
"""

import pytest

from osprey.prompts.defaults.ariel.agent import DefaultARIELAgentPromptBuilder
from osprey.prompts.defaults.ariel.rag import DefaultARIELRAGPromptBuilder


class TestDefaultARIELAgentPromptBuilder:
    """Test DefaultARIELAgentPromptBuilder."""

    @pytest.fixture
    def builder(self):
        return DefaultARIELAgentPromptBuilder()

    def test_get_facility_context(self, builder):
        """Test facility context is a non-empty string."""
        ctx = builder.get_facility_context()
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_get_response_guidelines(self, builder):
        """Test response guidelines is a non-empty string."""
        guidelines = builder.get_response_guidelines()
        assert isinstance(guidelines, str)
        assert len(guidelines) > 0

    def test_get_system_prompt(self, builder):
        """Test system prompt assembles both parts."""
        prompt = builder.get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        # Should contain content from both facility context and guidelines
        assert builder.get_facility_context() in prompt
        assert builder.get_response_guidelines() in prompt

    def test_system_prompt_contains_ariel(self, builder):
        """Test system prompt mentions ARIEL."""
        prompt = builder.get_system_prompt()
        assert "ARIEL" in prompt

    def test_role_definition_delegates_to_facility_context(self, builder):
        """Test that get_role delegates to get_facility_context."""
        assert builder.get_role() == builder.get_facility_context()

    def test_instructions_delegates_to_response_guidelines(self, builder):
        """Test that get_instructions delegates to get_response_guidelines."""
        assert builder.get_instructions() == builder.get_response_guidelines()


class TestDefaultARIELRAGPromptBuilder:
    """Test DefaultARIELRAGPromptBuilder."""

    @pytest.fixture
    def builder(self):
        return DefaultARIELRAGPromptBuilder()

    def test_get_facility_context(self, builder):
        """Test facility context is a non-empty string."""
        ctx = builder.get_facility_context()
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_get_response_guidelines(self, builder):
        """Test response guidelines is a non-empty string."""
        guidelines = builder.get_response_guidelines()
        assert isinstance(guidelines, str)
        assert len(guidelines) > 0

    def test_get_prompt_template(self, builder):
        """Test prompt template assembles correctly."""
        template = builder.get_prompt_template()
        assert isinstance(template, str)
        assert len(template) > 0

    def test_prompt_template_contains_placeholders(self, builder):
        """Test prompt template has {context} and {question} placeholders."""
        template = builder.get_prompt_template()
        assert "{context}" in template
        assert "{question}" in template

    def test_prompt_template_format_works(self, builder):
        """Test that .format() works with context and question."""
        template = builder.get_prompt_template()
        result = template.format(context="test context", question="test question")
        assert "test context" in result
        assert "test question" in result
        # Placeholders should be consumed
        assert "{context}" not in result
        assert "{question}" not in result

    def test_prompt_template_contains_citation_guidance(self, builder):
        """Test prompt template includes citation format guidance."""
        template = builder.get_prompt_template()
        assert "[#" in template

    def test_role_definition_delegates_to_facility_context(self, builder):
        """Test that get_role delegates to get_facility_context."""
        assert builder.get_role() == builder.get_facility_context()

    def test_instructions_delegates_to_response_guidelines(self, builder):
        """Test that get_instructions delegates to get_response_guidelines."""
        assert builder.get_instructions() == builder.get_response_guidelines()


class TestARIELPromptBuilderInterface:
    """Test that all ARIEL builders implement the same interface correctly."""

    @pytest.fixture(
        params=[
            DefaultARIELAgentPromptBuilder,
            DefaultARIELRAGPromptBuilder,
        ]
    )
    def builder(self, request):
        return request.param()

    def test_has_required_methods(self, builder):
        """All builders must have the core interface methods."""
        assert hasattr(builder, "get_facility_context")
        assert hasattr(builder, "get_response_guidelines")
        assert hasattr(builder, "get_role")
        assert hasattr(builder, "get_instructions")

    def test_facility_context_not_empty(self, builder):
        """Facility context should never be empty."""
        ctx = builder.get_facility_context()
        assert ctx.strip(), f"{type(builder).__name__} returned empty facility context"

    def test_response_guidelines_not_empty(self, builder):
        """Response guidelines should never be empty."""
        guidelines = builder.get_response_guidelines()
        assert guidelines.strip(), f"{type(builder).__name__} returned empty response guidelines"

    def test_can_instantiate(self, builder):
        """All builders can be instantiated without arguments."""
        assert builder is not None
