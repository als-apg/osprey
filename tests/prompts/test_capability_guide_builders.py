"""Tests for guide-only capability prompt builders.

Tests the 4 prompt builders that provide orchestrator and classifier guidance
for capabilities: channel_read, channel_write, channel_finding_orchestration,
and archiver_retrieval.

Note: PlannedStep is a TypedDict, so step fields are accessed via dict syntax.
"""

import pytest

from osprey.prompts.defaults.archiver_retrieval import DefaultArchiverRetrievalPromptBuilder
from osprey.prompts.defaults.channel_finding_orchestration import (
    DefaultChannelFindingOrchestrationPromptBuilder,
)
from osprey.prompts.defaults.channel_read import DefaultChannelReadPromptBuilder
from osprey.prompts.defaults.channel_write import DefaultChannelWritePromptBuilder

# ===================================================================
# Channel Read
# ===================================================================


class TestChannelReadPromptBuilder:
    """Test DefaultChannelReadPromptBuilder."""

    def test_prompt_type(self):
        builder = DefaultChannelReadPromptBuilder()
        assert builder.PROMPT_TYPE == "channel_read"

    def test_role_definition(self):
        builder = DefaultChannelReadPromptBuilder()
        assert "channel" in builder.get_role().lower()

    def test_instructions(self):
        builder = DefaultChannelReadPromptBuilder()
        assert "Read" in builder.get_instructions()

    def test_orchestrator_guide(self):
        builder = DefaultChannelReadPromptBuilder()
        guide = builder.get_orchestrator_guide()
        assert guide is not None
        assert len(guide.examples) == 2
        assert "channel_read" in guide.instructions

    def test_orchestrator_examples_capability(self):
        builder = DefaultChannelReadPromptBuilder()
        guide = builder.get_orchestrator_guide()
        for example in guide.examples:
            assert example.step["capability"] == "channel_read"
            assert example.step["expected_output"] == "CHANNEL_VALUES"

    def test_orchestrator_examples_require_channel_addresses(self):
        builder = DefaultChannelReadPromptBuilder()
        guide = builder.get_orchestrator_guide()
        for example in guide.examples:
            inputs = example.step["inputs"]
            assert any("CHANNEL_ADDRESSES" in inp for inp in inputs)

    def test_classifier_guide(self):
        builder = DefaultChannelReadPromptBuilder()
        guide = builder.get_classifier_guide()
        assert guide is not None
        assert len(guide.examples) == 5

    def test_classifier_positive_examples(self):
        builder = DefaultChannelReadPromptBuilder()
        guide = builder.get_classifier_guide()
        positive = [e for e in guide.examples if e.result is True]
        assert len(positive) >= 3

    def test_classifier_negative_examples(self):
        builder = DefaultChannelReadPromptBuilder()
        guide = builder.get_classifier_guide()
        negative = [e for e in guide.examples if e.result is False]
        assert len(negative) >= 2


# ===================================================================
# Channel Write
# ===================================================================


class TestChannelWritePromptBuilder:
    """Test DefaultChannelWritePromptBuilder."""

    def test_prompt_type(self):
        builder = DefaultChannelWritePromptBuilder()
        assert builder.PROMPT_TYPE == "channel_write"

    def test_role_definition(self):
        builder = DefaultChannelWritePromptBuilder()
        assert "write" in builder.get_role().lower()

    def test_instructions(self):
        builder = DefaultChannelWritePromptBuilder()
        instructions = builder.get_instructions()
        assert "CRITICAL RULES" in instructions
        assert "NEVER compute" in instructions

    def test_orchestrator_guide(self):
        builder = DefaultChannelWritePromptBuilder()
        guide = builder.get_orchestrator_guide()
        assert guide is not None
        assert len(guide.examples) == 2
        assert "channel_write" in guide.instructions

    def test_orchestrator_examples_capability(self):
        builder = DefaultChannelWritePromptBuilder()
        guide = builder.get_orchestrator_guide()
        for example in guide.examples:
            assert example.step["capability"] == "channel_write"
            assert example.step["expected_output"] == "CHANNEL_WRITE_RESULTS"

    def test_orchestrator_simple_write_example(self):
        builder = DefaultChannelWritePromptBuilder()
        guide = builder.get_orchestrator_guide()
        simple = guide.examples[0]
        assert "set" in simple.step["task_objective"].lower()

    def test_orchestrator_calculated_write_example(self):
        builder = DefaultChannelWritePromptBuilder()
        guide = builder.get_orchestrator_guide()
        calculated = guide.examples[1]
        inputs = calculated.step["inputs"]
        assert any("PYTHON_RESULTS" in inp for inp in inputs)

    def test_classifier_guide(self):
        builder = DefaultChannelWritePromptBuilder()
        guide = builder.get_classifier_guide()
        assert guide is not None
        assert len(guide.examples) == 7

    def test_classifier_positive_examples(self):
        builder = DefaultChannelWritePromptBuilder()
        guide = builder.get_classifier_guide()
        positive = [e for e in guide.examples if e.result is True]
        assert len(positive) >= 4

    def test_classifier_negative_examples(self):
        builder = DefaultChannelWritePromptBuilder()
        guide = builder.get_classifier_guide()
        negative = [e for e in guide.examples if e.result is False]
        assert len(negative) >= 2


class TestChannelWriteBuildPrompt:
    """Test build_prompt() composes static instructions with dynamic context."""

    def test_build_prompt_contains_role(self):
        builder = DefaultChannelWritePromptBuilder()
        prompt = builder.build_prompt()
        assert "expert at parsing" in prompt

    def test_build_prompt_contains_task_objective(self):
        builder = DefaultChannelWritePromptBuilder()
        prompt = builder.build_prompt(task_objective="Set HCM01 to 5.0")
        assert "Set HCM01 to 5.0" in prompt

    def test_build_prompt_contains_channel_mapping(self):
        builder = DefaultChannelWritePromptBuilder()
        mapping = '  "horizontal corrector" → HCM01:CURRENT:SP'
        prompt = builder.build_prompt(channel_mapping=mapping)
        assert "HCM01:CURRENT:SP" in prompt
        assert "AVAILABLE CHANNEL ADDRESSES" in prompt

    def test_build_prompt_contains_critical_rules(self):
        builder = DefaultChannelWritePromptBuilder()
        prompt = builder.build_prompt()
        assert "CRITICAL RULES" in prompt
        assert "NEVER compute" in prompt

    def test_build_prompt_contains_examples(self):
        builder = DefaultChannelWritePromptBuilder()
        prompt = builder.build_prompt()
        assert "VALUE EXTRACTION EXAMPLES" in prompt
        assert "PYTHON_RESULTS" in prompt

    def test_build_prompt_contains_available_data(self):
        builder = DefaultChannelWritePromptBuilder()
        prompt = builder.build_prompt(available_data="PYTHON_RESULTS: {}")
        assert "AVAILABLE DATA" in prompt
        assert "PYTHON_RESULTS: {}" in prompt

    def test_build_prompt_without_kwargs(self):
        builder = DefaultChannelWritePromptBuilder()
        prompt = builder.build_prompt()
        # Should still produce a valid prompt with role, instructions
        assert "expert at parsing" in prompt
        assert "YOUR JOB" in prompt
        # No dynamic sections without kwargs
        assert "TASK:" not in prompt


# ===================================================================
# Channel Finding Orchestration
# ===================================================================


class TestChannelFindingOrchestrationPromptBuilder:
    """Test DefaultChannelFindingOrchestrationPromptBuilder."""

    def test_prompt_type(self):
        builder = DefaultChannelFindingOrchestrationPromptBuilder()
        assert builder.PROMPT_TYPE == "channel_finding_orchestration"

    def test_role_definition(self):
        builder = DefaultChannelFindingOrchestrationPromptBuilder()
        role = builder.get_role()
        assert "channel" in role.lower()
        assert "find" in role.lower()

    def test_instructions(self):
        builder = DefaultChannelFindingOrchestrationPromptBuilder()
        assert "Find" in builder.get_instructions()

    def test_orchestrator_guide(self):
        builder = DefaultChannelFindingOrchestrationPromptBuilder()
        guide = builder.get_orchestrator_guide()
        assert guide is not None
        assert len(guide.examples) == 2

    def test_orchestrator_guide_priority(self):
        builder = DefaultChannelFindingOrchestrationPromptBuilder()
        guide = builder.get_orchestrator_guide()
        assert guide.priority == 1

    def test_orchestrator_examples_capability(self):
        builder = DefaultChannelFindingOrchestrationPromptBuilder()
        guide = builder.get_orchestrator_guide()
        for example in guide.examples:
            assert example.step["capability"] == "channel_finding"
            assert example.step["expected_output"] == "CHANNEL_ADDRESSES"

    def test_classifier_guide(self):
        builder = DefaultChannelFindingOrchestrationPromptBuilder()
        guide = builder.get_classifier_guide()
        assert guide is not None
        assert len(guide.examples) == 6

    def test_classifier_positive_examples(self):
        builder = DefaultChannelFindingOrchestrationPromptBuilder()
        guide = builder.get_classifier_guide()
        positive = [e for e in guide.examples if e.result is True]
        assert len(positive) >= 4

    def test_classifier_negative_examples(self):
        builder = DefaultChannelFindingOrchestrationPromptBuilder()
        guide = builder.get_classifier_guide()
        negative = [e for e in guide.examples if e.result is False]
        assert len(negative) >= 1


# ===================================================================
# Archiver Retrieval
# ===================================================================


class TestArchiverRetrievalPromptBuilder:
    """Test DefaultArchiverRetrievalPromptBuilder."""

    def test_prompt_type(self):
        builder = DefaultArchiverRetrievalPromptBuilder()
        assert builder.PROMPT_TYPE == "archiver_retrieval"

    def test_role_definition(self):
        builder = DefaultArchiverRetrievalPromptBuilder()
        role = builder.get_role()
        assert "historical" in role.lower() or "archiv" in role.lower()

    def test_instructions(self):
        builder = DefaultArchiverRetrievalPromptBuilder()
        instructions = builder.get_instructions()
        assert "historical" in instructions.lower() or "Retrieve" in instructions

    def test_orchestrator_guide(self):
        builder = DefaultArchiverRetrievalPromptBuilder()
        guide = builder.get_orchestrator_guide()
        assert guide is not None
        assert len(guide.examples) == 3

    def test_orchestrator_guide_priority(self):
        builder = DefaultArchiverRetrievalPromptBuilder()
        guide = builder.get_orchestrator_guide()
        assert guide.priority == 15

    def test_orchestrator_retrieval_example(self):
        builder = DefaultArchiverRetrievalPromptBuilder()
        guide = builder.get_orchestrator_guide()
        retrieval = guide.examples[0]
        assert retrieval.step["capability"] == "archiver_retrieval"
        assert retrieval.step["expected_output"] == "ARCHIVER_DATA"
        inputs = retrieval.step["inputs"]
        assert any("CHANNEL_ADDRESSES" in inp for inp in inputs)
        assert any("TIME_RANGE" in inp for inp in inputs)

    def test_orchestrator_workflow_examples(self):
        """Workflow examples show downstream python steps."""
        builder = DefaultArchiverRetrievalPromptBuilder()
        guide = builder.get_orchestrator_guide()
        plotting = guide.examples[1]
        analysis = guide.examples[2]
        assert plotting.step["capability"] == "python"
        assert analysis.step["capability"] == "python"
        assert any("ARCHIVER_DATA" in inp for inp in plotting.step["inputs"])

    def test_classifier_guide(self):
        builder = DefaultArchiverRetrievalPromptBuilder()
        guide = builder.get_classifier_guide()
        assert guide is not None
        assert len(guide.examples) == 5

    def test_classifier_positive_examples(self):
        builder = DefaultArchiverRetrievalPromptBuilder()
        guide = builder.get_classifier_guide()
        positive = [e for e in guide.examples if e.result is True]
        assert len(positive) >= 3

    def test_classifier_negative_examples(self):
        builder = DefaultArchiverRetrievalPromptBuilder()
        guide = builder.get_classifier_guide()
        negative = [e for e in guide.examples if e.result is False]
        assert len(negative) >= 2


# ===================================================================
# Cross-builder interface consistency
# ===================================================================


class TestGuideBuilderInterface:
    """Test that all guide-only builders share a consistent interface."""

    @pytest.fixture(
        params=[
            DefaultChannelReadPromptBuilder,
            DefaultChannelWritePromptBuilder,
            DefaultChannelFindingOrchestrationPromptBuilder,
            DefaultArchiverRetrievalPromptBuilder,
        ]
    )
    def builder(self, request):
        return request.param()

    def test_has_prompt_type(self, builder):
        assert hasattr(builder, "PROMPT_TYPE")
        assert isinstance(builder.PROMPT_TYPE, str)
        assert len(builder.PROMPT_TYPE) > 0

    def test_has_role_definition(self, builder):
        role = builder.get_role()
        assert isinstance(role, str)
        assert len(role) > 0

    def test_has_instructions(self, builder):
        instructions = builder.get_instructions()
        assert isinstance(instructions, str)
        assert len(instructions) > 0

    def test_has_orchestrator_guide(self, builder):
        guide = builder.get_orchestrator_guide()
        assert guide is not None
        assert guide.instructions
        assert len(guide.examples) > 0

    def test_has_classifier_guide(self, builder):
        guide = builder.get_classifier_guide()
        assert guide is not None
        assert guide.instructions
        assert len(guide.examples) > 0

    def test_orchestrator_examples_have_required_fields(self, builder):
        guide = builder.get_orchestrator_guide()
        for example in guide.examples:
            assert example.step["context_key"]
            assert example.step["capability"]
            assert example.step["task_objective"]
            assert example.step["expected_output"]

    def test_classifier_examples_have_required_fields(self, builder):
        guide = builder.get_classifier_guide()
        for example in guide.examples:
            assert example.query
            assert isinstance(example.result, bool)
            assert example.reason
