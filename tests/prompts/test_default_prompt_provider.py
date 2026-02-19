"""Tests for DefaultPromptProvider — verify all getter methods return correct builder types."""

from osprey.prompts.base import FrameworkPromptBuilder
from osprey.prompts.defaults import DefaultPromptProvider
from osprey.prompts.defaults.archiver_retrieval import DefaultArchiverRetrievalPromptBuilder
from osprey.prompts.defaults.ariel.agent import DefaultARIELAgentPromptBuilder
from osprey.prompts.defaults.ariel.rag import DefaultARIELRAGPromptBuilder
from osprey.prompts.defaults.channel_finder.hierarchical import DefaultHierarchicalPromptBuilder
from osprey.prompts.defaults.channel_finder.in_context import DefaultInContextPromptBuilder
from osprey.prompts.defaults.channel_finder.middle_layer import DefaultMiddleLayerPromptBuilder
from osprey.prompts.defaults.channel_finding_orchestration import (
    DefaultChannelFindingOrchestrationPromptBuilder,
)
from osprey.prompts.defaults.channel_read import DefaultChannelReadPromptBuilder
from osprey.prompts.defaults.channel_write import DefaultChannelWritePromptBuilder
from osprey.prompts.defaults.clarification import DefaultClarificationPromptBuilder
from osprey.prompts.defaults.classification import DefaultClassificationPromptBuilder
from osprey.prompts.defaults.error_analysis import DefaultErrorAnalysisPromptBuilder
from osprey.prompts.defaults.logbook_search import DefaultLogbookSearchPromptBuilder
from osprey.prompts.defaults.memory_extraction import DefaultMemoryExtractionPromptBuilder
from osprey.prompts.defaults.orchestrator import DefaultOrchestratorPromptBuilder
from osprey.prompts.defaults.python import DefaultPythonPromptBuilder
from osprey.prompts.defaults.response_generation import DefaultResponseGenerationPromptBuilder
from osprey.prompts.defaults.task_extraction import DefaultTaskExtractionPromptBuilder
from osprey.prompts.defaults.time_range_parsing import DefaultTimeRangeParsingPromptBuilder


class TestDefaultPromptProviderInfrastructure:
    """Test infrastructure prompt builder getters."""

    def setup_method(self):
        self.provider = DefaultPromptProvider()

    def test_orchestrator(self):
        builder = self.provider.get_orchestrator_prompt_builder()
        assert isinstance(builder, DefaultOrchestratorPromptBuilder)

    def test_task_extraction(self):
        builder = self.provider.get_task_extraction_prompt_builder()
        assert isinstance(builder, DefaultTaskExtractionPromptBuilder)

    def test_response_generation(self):
        builder = self.provider.get_response_generation_prompt_builder()
        assert isinstance(builder, DefaultResponseGenerationPromptBuilder)

    def test_classification(self):
        builder = self.provider.get_classification_prompt_builder()
        assert isinstance(builder, DefaultClassificationPromptBuilder)

    def test_error_analysis(self):
        builder = self.provider.get_error_analysis_prompt_builder()
        assert isinstance(builder, DefaultErrorAnalysisPromptBuilder)

    def test_clarification(self):
        builder = self.provider.get_clarification_prompt_builder()
        assert isinstance(builder, DefaultClarificationPromptBuilder)


class TestDefaultPromptProviderCapabilities:
    """Test framework capability prompt builder getters."""

    def setup_method(self):
        self.provider = DefaultPromptProvider()

    def test_memory_extraction(self):
        builder = self.provider.get_memory_extraction_prompt_builder()
        assert isinstance(builder, DefaultMemoryExtractionPromptBuilder)

    def test_time_range_parsing(self):
        builder = self.provider.get_time_range_parsing_prompt_builder()
        assert isinstance(builder, DefaultTimeRangeParsingPromptBuilder)

    def test_python(self):
        builder = self.provider.get_python_prompt_builder()
        assert isinstance(builder, DefaultPythonPromptBuilder)


class TestDefaultPromptProviderChannelFinder:
    """Test channel finder prompt builder getters."""

    def setup_method(self):
        self.provider = DefaultPromptProvider()

    def test_in_context(self):
        builder = self.provider.get_channel_finder_in_context_prompt_builder()
        assert isinstance(builder, DefaultInContextPromptBuilder)

    def test_hierarchical(self):
        builder = self.provider.get_channel_finder_hierarchical_prompt_builder()
        assert isinstance(builder, DefaultHierarchicalPromptBuilder)

    def test_middle_layer(self):
        builder = self.provider.get_channel_finder_middle_layer_prompt_builder()
        assert isinstance(builder, DefaultMiddleLayerPromptBuilder)


class TestDefaultPromptProviderGuideBuilders:
    """Test capability guide prompt builder getters."""

    def setup_method(self):
        self.provider = DefaultPromptProvider()

    def test_channel_read(self):
        builder = self.provider.get_channel_read_prompt_builder()
        assert isinstance(builder, DefaultChannelReadPromptBuilder)

    def test_channel_write(self):
        builder = self.provider.get_channel_write_prompt_builder()
        assert isinstance(builder, DefaultChannelWritePromptBuilder)

    def test_channel_finding_orchestration(self):
        builder = self.provider.get_channel_finding_orchestration_prompt_builder()
        assert isinstance(builder, DefaultChannelFindingOrchestrationPromptBuilder)

    def test_archiver_retrieval(self):
        builder = self.provider.get_archiver_retrieval_prompt_builder()
        assert isinstance(builder, DefaultArchiverRetrievalPromptBuilder)

    def test_logbook_search(self):
        builder = self.provider.get_logbook_search_prompt_builder()
        assert isinstance(builder, DefaultLogbookSearchPromptBuilder)


class TestDefaultPromptProviderARIEL:
    """Test ARIEL prompt builder getters."""

    def setup_method(self):
        self.provider = DefaultPromptProvider()

    def test_ariel_agent(self):
        builder = self.provider.get_ariel_agent_prompt_builder()
        assert isinstance(builder, DefaultARIELAgentPromptBuilder)

    def test_ariel_rag(self):
        builder = self.provider.get_ariel_rag_prompt_builder()
        assert isinstance(builder, DefaultARIELRAGPromptBuilder)


class TestDefaultPromptProviderAllBuildersAreFrameworkPromptBuilders:
    """Verify every getter returns a FrameworkPromptBuilder subclass."""

    def setup_method(self):
        self.provider = DefaultPromptProvider()

    def test_all_getters_return_framework_prompt_builder(self):
        getters = [
            self.provider.get_orchestrator_prompt_builder,
            self.provider.get_task_extraction_prompt_builder,
            self.provider.get_response_generation_prompt_builder,
            self.provider.get_classification_prompt_builder,
            self.provider.get_error_analysis_prompt_builder,
            self.provider.get_clarification_prompt_builder,
            self.provider.get_memory_extraction_prompt_builder,
            self.provider.get_time_range_parsing_prompt_builder,
            self.provider.get_python_prompt_builder,
            self.provider.get_channel_finder_in_context_prompt_builder,
            self.provider.get_channel_finder_hierarchical_prompt_builder,
            self.provider.get_channel_finder_middle_layer_prompt_builder,
            self.provider.get_channel_read_prompt_builder,
            self.provider.get_channel_write_prompt_builder,
            self.provider.get_channel_finding_orchestration_prompt_builder,
            self.provider.get_archiver_retrieval_prompt_builder,
            self.provider.get_logbook_search_prompt_builder,
            self.provider.get_ariel_agent_prompt_builder,
            self.provider.get_ariel_rag_prompt_builder,
        ]
        for getter in getters:
            builder = getter()
            assert isinstance(builder, FrameworkPromptBuilder), (
                f"{getter.__name__} returned {type(builder).__name__}, not a FrameworkPromptBuilder"
            )

    def test_provider_returns_same_instance_per_call(self):
        """Each call to a getter should return the same cached instance."""
        builder1 = self.provider.get_orchestrator_prompt_builder()
        builder2 = self.provider.get_orchestrator_prompt_builder()
        assert builder1 is builder2
