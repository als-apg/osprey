"""Default prompt implementations package."""

from ..base import FrameworkPromptBuilder

# Import the interface
from ..loader import FrameworkPromptProvider
from .clarification import DefaultClarificationPromptBuilder
from .classification import DefaultClassificationPromptBuilder
from .error_analysis import DefaultErrorAnalysisPromptBuilder
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

    # =================================================================
    # Infrastructure prompts
    # =================================================================

    def get_orchestrator_prompt_builder(self) -> 'FrameworkPromptBuilder':
        return self._orchestrator_builder

    def get_task_extraction_prompt_builder(self) -> 'FrameworkPromptBuilder':
        return self._task_extraction_builder

    def get_response_generation_prompt_builder(self) -> 'FrameworkPromptBuilder':
        return self._response_generation_builder

    def get_classification_prompt_builder(self) -> 'FrameworkPromptBuilder':
        return self._classification_builder

    def get_error_analysis_prompt_builder(self) -> 'FrameworkPromptBuilder':
        return self._error_analysis_builder

    def get_clarification_prompt_builder(self) -> 'FrameworkPromptBuilder':
        return self._clarification_builder

    # =================================================================
    # Framework capability prompts
    # =================================================================

    def get_memory_extraction_prompt_builder(self) -> 'FrameworkPromptBuilder':
        return self._memory_extraction_builder

    def get_time_range_parsing_prompt_builder(self) -> 'FrameworkPromptBuilder':
        return self._time_range_parsing_builder

    def get_python_prompt_builder(self) -> 'FrameworkPromptBuilder':
        return self._python_builder

__all__ = [
    "DefaultClassificationPromptBuilder",
    "DefaultResponseGenerationPromptBuilder",
    "DefaultTaskExtractionPromptBuilder",
    "DefaultClarificationPromptBuilder",
    "DefaultErrorAnalysisPromptBuilder",
    "DefaultMemoryExtractionPromptBuilder",
    "DefaultTimeRangeParsingPromptBuilder",
    "DefaultPythonPromptBuilder",
    "DefaultOrchestratorPromptBuilder",
    "DefaultPromptProvider",
    "TaskExtractionExample",
    "ExtractedTask"
]
