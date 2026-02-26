"""Default prompt implementations package."""

from ..base import FrameworkPromptBuilder

# Import the interface
from ..loader import FrameworkPromptProvider
from .archiver_retrieval import DefaultArchiverRetrievalPromptBuilder
from .ariel import DefaultARIELAgentPromptBuilder, DefaultARIELRAGPromptBuilder
from .channel_finder import (
    DefaultHierarchicalPromptBuilder,
    DefaultInContextPromptBuilder,
    DefaultMiddleLayerPromptBuilder,
)
from .channel_finding_orchestration import DefaultChannelFindingOrchestrationPromptBuilder
from .channel_read import DefaultChannelReadPromptBuilder
from .channel_write import DefaultChannelWritePromptBuilder
from .clarification import DefaultClarificationPromptBuilder
from .classification import DefaultClassificationPromptBuilder
from .error_analysis import DefaultErrorAnalysisPromptBuilder
from .logbook_search import DefaultLogbookSearchPromptBuilder
from .memory_extraction import DefaultMemoryExtractionPromptBuilder
from .orchestrator import DefaultOrchestratorPromptBuilder
from .python import DefaultPythonPromptBuilder
from .response_generation import DefaultResponseGenerationPromptBuilder
from .task_extraction import (
    DefaultTaskExtractionPromptBuilder,
    ExtractedTask,
    TaskExtractionExample,
)
from .time_range_parsing import DefaultTimeRangeParsingPromptBuilder


class DefaultPromptProvider(FrameworkPromptProvider):
    """Default implementation of FrameworkPromptProvider using default builders."""

    def __init__(self):
        # Infrastructure prompt builders
        self._orchestrator_builder = DefaultOrchestratorPromptBuilder()
        self._task_extraction_builder = DefaultTaskExtractionPromptBuilder()
        self._response_generation_builder = DefaultResponseGenerationPromptBuilder()
        self._classification_builder = DefaultClassificationPromptBuilder()
        self._error_analysis_builder = DefaultErrorAnalysisPromptBuilder()
        self._clarification_builder = DefaultClarificationPromptBuilder()

        # Framework capability prompt builders
        self._memory_extraction_builder = DefaultMemoryExtractionPromptBuilder()
        self._time_range_parsing_builder = DefaultTimeRangeParsingPromptBuilder()
        self._python_builder = DefaultPythonPromptBuilder()

        # Channel finder prompt builders
        self._cf_in_context_builder = DefaultInContextPromptBuilder()
        self._cf_hierarchical_builder = DefaultHierarchicalPromptBuilder()
        self._cf_middle_layer_builder = DefaultMiddleLayerPromptBuilder()

        # Logbook search prompt builder
        self._logbook_search_builder = DefaultLogbookSearchPromptBuilder()

        # Capability guide prompt builders
        self._channel_read_builder = DefaultChannelReadPromptBuilder()
        self._channel_write_builder = DefaultChannelWritePromptBuilder()
        self._channel_finding_orchestration_builder = (
            DefaultChannelFindingOrchestrationPromptBuilder()
        )
        self._archiver_retrieval_builder = DefaultArchiverRetrievalPromptBuilder()

        # ARIEL prompt builders
        self._ariel_agent_builder = DefaultARIELAgentPromptBuilder()
        self._ariel_rag_builder = DefaultARIELRAGPromptBuilder()

    # =================================================================
    # Infrastructure prompts
    # =================================================================

    def get_orchestrator_prompt_builder(self) -> "FrameworkPromptBuilder":
        return self._orchestrator_builder

    def get_task_extraction_prompt_builder(self) -> "FrameworkPromptBuilder":
        return self._task_extraction_builder

    def get_response_generation_prompt_builder(self) -> "FrameworkPromptBuilder":
        return self._response_generation_builder

    def get_classification_prompt_builder(self) -> "FrameworkPromptBuilder":
        return self._classification_builder

    def get_error_analysis_prompt_builder(self) -> "FrameworkPromptBuilder":
        return self._error_analysis_builder

    def get_clarification_prompt_builder(self) -> "FrameworkPromptBuilder":
        return self._clarification_builder

    # =================================================================
    # Framework capability prompts
    # =================================================================

    def get_memory_extraction_prompt_builder(self) -> "FrameworkPromptBuilder":
        return self._memory_extraction_builder

    def get_time_range_parsing_prompt_builder(self) -> "FrameworkPromptBuilder":
        return self._time_range_parsing_builder

    def get_python_prompt_builder(self) -> "FrameworkPromptBuilder":
        return self._python_builder

    # =================================================================
    # Channel finder prompt builders
    # =================================================================

    def get_channel_finder_in_context_prompt_builder(self) -> "FrameworkPromptBuilder":
        return self._cf_in_context_builder

    def get_channel_finder_hierarchical_prompt_builder(self) -> "FrameworkPromptBuilder":
        return self._cf_hierarchical_builder

    def get_channel_finder_middle_layer_prompt_builder(self) -> "FrameworkPromptBuilder":
        return self._cf_middle_layer_builder

    # =================================================================
    # Logbook search prompt builder
    # =================================================================

    def get_logbook_search_prompt_builder(self) -> "FrameworkPromptBuilder":
        return self._logbook_search_builder

    # =================================================================
    # Capability guide prompt builders
    # =================================================================

    def get_channel_read_prompt_builder(self) -> "FrameworkPromptBuilder":
        return self._channel_read_builder

    def get_channel_write_prompt_builder(self) -> "FrameworkPromptBuilder":
        return self._channel_write_builder

    def get_channel_finding_orchestration_prompt_builder(self) -> "FrameworkPromptBuilder":
        return self._channel_finding_orchestration_builder

    def get_archiver_retrieval_prompt_builder(self) -> "FrameworkPromptBuilder":
        return self._archiver_retrieval_builder

    # =================================================================
    # ARIEL prompt builders
    # =================================================================

    def get_ariel_agent_prompt_builder(self) -> "FrameworkPromptBuilder":
        return self._ariel_agent_builder

    def get_ariel_rag_prompt_builder(self) -> "FrameworkPromptBuilder":
        return self._ariel_rag_builder


__all__ = [
    "DefaultARIELAgentPromptBuilder",
    "DefaultARIELRAGPromptBuilder",
    "DefaultArchiverRetrievalPromptBuilder",
    "DefaultChannelFindingOrchestrationPromptBuilder",
    "DefaultChannelReadPromptBuilder",
    "DefaultChannelWritePromptBuilder",
    "DefaultClassificationPromptBuilder",
    "DefaultResponseGenerationPromptBuilder",
    "DefaultTaskExtractionPromptBuilder",
    "DefaultClarificationPromptBuilder",
    "DefaultErrorAnalysisPromptBuilder",
    "DefaultMemoryExtractionPromptBuilder",
    "DefaultTimeRangeParsingPromptBuilder",
    "DefaultPythonPromptBuilder",
    "DefaultOrchestratorPromptBuilder",
    "DefaultInContextPromptBuilder",
    "DefaultHierarchicalPromptBuilder",
    "DefaultMiddleLayerPromptBuilder",
    "DefaultLogbookSearchPromptBuilder",
    "DefaultPromptProvider",
    "TaskExtractionExample",
    "ExtractedTask",
]
