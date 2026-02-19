"""Tests for channel finder default prompt builders.

Validates that the 3 prompt builders (in_context, hierarchical, middle_layer)
provide valid content and implement the required interface.
"""

import pytest

from osprey.prompts.defaults.channel_finder.hierarchical import DefaultHierarchicalPromptBuilder
from osprey.prompts.defaults.channel_finder.in_context import DefaultInContextPromptBuilder
from osprey.prompts.defaults.channel_finder.middle_layer import DefaultMiddleLayerPromptBuilder


class TestDefaultInContextPromptBuilder:
    """Test DefaultInContextPromptBuilder."""

    @pytest.fixture
    def builder(self):
        return DefaultInContextPromptBuilder()

    def test_get_facility_description(self, builder):
        """Test facility description is a non-empty string."""
        desc = builder.get_facility_description()
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_get_matching_rules(self, builder):
        """Test matching rules is a non-empty string."""
        rules = builder.get_matching_rules()
        assert isinstance(rules, str)
        assert len(rules) > 0

    def test_get_combined_description(self, builder):
        """Test combined description includes both parts."""
        combined = builder.get_combined_description()
        assert isinstance(combined, str)
        # Combined should include both facility description and matching rules
        assert len(combined) >= len(builder.get_facility_description())

    def test_role_definition_delegates_to_facility_description(self, builder):
        """Test that get_role delegates to get_facility_description."""
        assert builder.get_role() == builder.get_facility_description()

    def test_instructions_delegates_to_matching_rules(self, builder):
        """Test that get_instructions delegates to get_matching_rules."""
        assert builder.get_instructions() == builder.get_matching_rules()


class TestDefaultHierarchicalPromptBuilder:
    """Test DefaultHierarchicalPromptBuilder."""

    @pytest.fixture
    def builder(self):
        return DefaultHierarchicalPromptBuilder()

    def test_get_facility_description(self, builder):
        desc = builder.get_facility_description()
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_get_matching_rules(self, builder):
        rules = builder.get_matching_rules()
        assert isinstance(rules, str)
        assert len(rules) > 0

    def test_get_combined_description(self, builder):
        combined = builder.get_combined_description()
        assert isinstance(combined, str)
        assert len(combined) >= len(builder.get_facility_description())


class TestDefaultMiddleLayerPromptBuilder:
    """Test DefaultMiddleLayerPromptBuilder."""

    @pytest.fixture
    def builder(self):
        return DefaultMiddleLayerPromptBuilder()

    def test_get_facility_description(self, builder):
        desc = builder.get_facility_description()
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_get_matching_rules(self, builder):
        rules = builder.get_matching_rules()
        assert isinstance(rules, str)
        assert len(rules) > 0

    def test_get_combined_description(self, builder):
        combined = builder.get_combined_description()
        assert isinstance(combined, str)
        assert len(combined) >= len(builder.get_facility_description())


class TestPromptBuilderInterface:
    """Test that all builders implement the same interface correctly."""

    @pytest.fixture(
        params=[
            DefaultInContextPromptBuilder,
            DefaultHierarchicalPromptBuilder,
            DefaultMiddleLayerPromptBuilder,
        ]
    )
    def builder(self, request):
        return request.param()

    def test_has_required_methods(self, builder):
        """All builders must have the core interface methods."""
        assert hasattr(builder, "get_facility_description")
        assert hasattr(builder, "get_matching_rules")
        assert hasattr(builder, "get_combined_description")
        assert hasattr(builder, "get_role")
        assert hasattr(builder, "get_instructions")

    def test_combined_description_not_empty(self, builder):
        """Combined description should never be empty."""
        combined = builder.get_combined_description()
        assert combined.strip(), f"{type(builder).__name__} returned empty combined description"

    def test_can_instantiate(self, builder):
        """All builders can be instantiated without arguments."""
        # If we got here, instantiation worked
        assert builder is not None
