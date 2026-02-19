"""Tests for logbook search prompt builder — guide-only builder validation."""

from osprey.prompts.defaults.logbook_search import DefaultLogbookSearchPromptBuilder


class TestLogbookSearchPromptAttributes:
    """Test logbook search prompt builder class attributes."""

    def test_prompt_type(self):
        builder = DefaultLogbookSearchPromptBuilder()
        assert builder.PROMPT_TYPE == "logbook_search"

    def test_role_definition_is_stub(self):
        """Role definition is a minimal stub — not used at runtime."""
        builder = DefaultLogbookSearchPromptBuilder()
        role = builder.get_role()
        assert "delegates" in role.lower() or "ARIEL" in role

    def test_instructions_is_stub(self):
        """Instructions is a minimal stub — not used at runtime."""
        builder = DefaultLogbookSearchPromptBuilder()
        instructions = builder.get_instructions()
        assert "ARIEL" in instructions or "relay" in instructions.lower()


class TestLogbookSearchOrchestratorGuide:
    """Test orchestrator guide content."""

    def test_guide_exists(self):
        builder = DefaultLogbookSearchPromptBuilder()
        guide = builder.get_orchestrator_guide()
        assert guide is not None

    def test_guide_has_three_examples(self):
        builder = DefaultLogbookSearchPromptBuilder()
        guide = builder.get_orchestrator_guide()
        assert len(guide.examples) == 3

    def test_guide_priority(self):
        builder = DefaultLogbookSearchPromptBuilder()
        guide = builder.get_orchestrator_guide()
        assert guide.priority == 15

    def test_guide_instructions_content(self):
        builder = DefaultLogbookSearchPromptBuilder()
        guide = builder.get_orchestrator_guide()
        assert "logbook_search" in guide.instructions
        assert "Context Key Format" in guide.instructions

    def test_examples_use_correct_capability(self):
        builder = DefaultLogbookSearchPromptBuilder()
        guide = builder.get_orchestrator_guide()
        for example in guide.examples:
            assert example.step["capability"] == "logbook_search"

    def test_semantic_search_example(self):
        builder = DefaultLogbookSearchPromptBuilder()
        guide = builder.get_orchestrator_guide()
        semantic = guide.examples[0]
        assert "injector" in semantic.step["task_objective"].lower()

    def test_keyword_search_example(self):
        builder = DefaultLogbookSearchPromptBuilder()
        guide = builder.get_orchestrator_guide()
        keyword = guide.examples[1]
        assert "BTS chicane" in keyword.step["task_objective"]

    def test_rag_search_example(self):
        builder = DefaultLogbookSearchPromptBuilder()
        guide = builder.get_orchestrator_guide()
        rag = guide.examples[2]
        assert "RF trips" in rag.step["task_objective"]

    def test_search_strategy_documented(self):
        builder = DefaultLogbookSearchPromptBuilder()
        guide = builder.get_orchestrator_guide()
        assert "Keyword" in guide.instructions
        assert "Semantic" in guide.instructions
        assert "RAG" in guide.instructions


class TestLogbookSearchClassifierGuide:
    """Test classifier guide content."""

    def test_guide_exists(self):
        builder = DefaultLogbookSearchPromptBuilder()
        guide = builder.get_classifier_guide()
        assert guide is not None

    def test_guide_has_examples(self):
        builder = DefaultLogbookSearchPromptBuilder()
        guide = builder.get_classifier_guide()
        assert len(guide.examples) == 8

    def test_guide_has_positive_examples(self):
        builder = DefaultLogbookSearchPromptBuilder()
        guide = builder.get_classifier_guide()
        positive = [e for e in guide.examples if e.result is True]
        assert len(positive) == 5

    def test_guide_has_negative_examples(self):
        builder = DefaultLogbookSearchPromptBuilder()
        guide = builder.get_classifier_guide()
        negative = [e for e in guide.examples if e.result is False]
        assert len(negative) == 3

    def test_classifier_instructions(self):
        builder = DefaultLogbookSearchPromptBuilder()
        guide = builder.get_classifier_guide()
        assert "logbook" in guide.instructions.lower()
