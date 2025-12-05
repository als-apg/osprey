"""
Osprey Agentic Framework - Task Extraction Node

Converts chat conversation history into focused, actionable tasks.
Implemented using convention-based class architecture for LangGraph compatibility.
"""

from __future__ import annotations

import asyncio
from typing import Any

# Native LangGraph message types for checkpointing compatibility
from langchain_core.messages import BaseMessage

from osprey.base.decorators import infrastructure_node
from osprey.base.errors import ErrorClassification, ErrorSeverity
from osprey.base.nodes import BaseInfrastructureNode
from osprey.data_management import (
    DataSourceRequester,
    create_data_source_request,
    get_data_source_manager,
)
from osprey.models import get_chat_completion
from osprey.prompts.defaults.task_extraction import ExtractedTask
from osprey.prompts.loader import get_framework_prompts

# Updated imports for LangGraph compatibility with TypedDict state
from osprey.utils.config import get_model_config
from osprey.utils.logger import get_logger

# Module-level logger for helper functions
logger = get_logger("task_extraction")

# =============================================================================
# PROMPT BUILDING HELPER FUNCTIONS
# =============================================================================


def _format_task_context(messages: list[BaseMessage], retrieval_result, logger) -> ExtractedTask:
    """Format task context for bypass mode without LLM processing.

    Creates an ExtractedTask using the same context formatting as normal extraction
    but without LLM analysis. Returns the formatted context directly as the task.
    This combines chat history and data sources in the same way as normal extraction.

    :param messages: The native LangGraph messages
    :param retrieval_result: DataRetrievalResult containing data from all available sources
    :param logger: Logger instance
    :return: ExtractedTask with formatted context as task
    :rtype: ExtractedTask
    """
    from osprey.state.messages import ChatHistoryFormatter

    if retrieval_result and retrieval_result.has_data:
        logger.debug(f"Bypass mode: including data sources: {retrieval_result.get_summary()}")

    logger.info("Bypass mode: skipping LLM, using formatted context as task")

    # Format the chat history using native message formatter
    chat_formatted = ChatHistoryFormatter.format_for_llm(messages)

    # Add data source context if available
    data_context = ""
    if retrieval_result and retrieval_result.has_data:
        # Get the actual retrieved content formatted for LLM consumption
        try:
            formatted_contexts = []
            for source_name, context in retrieval_result.context_data.items():
                try:
                    formatted_content = context.format_for_prompt()
                    if formatted_content and formatted_content.strip():
                        formatted_contexts.append(f"**{source_name}:**\n{formatted_content}")
                except Exception as e:
                    logger.warning(f"Could not format content from source {source_name}: {e}")

            if formatted_contexts:
                data_context = "\n\n**Retrieved Data:**\n" + "\n\n".join(formatted_contexts)
            else:
                # Fallback to summary if no content could be formatted
                data_context = f"\n\n**Available Data Sources:**\n{retrieval_result.get_summary()}"

        except Exception as e:
            logger.warning(f"Could not process retrieval result: {e}")
            # Fallback to summary
            data_context = f"\n\n**Available Data Sources:**\n{retrieval_result.get_summary()}"

    formatted_context = f"{data_context}\n\nChat history:\n{chat_formatted}"

    # Return formatted context as the extracted task
    return ExtractedTask(
        task=formatted_context,
        depends_on_chat_history=True,  # Always true in bypass mode
        depends_on_user_memory=True,  # Always true in bypass mode
    )


def _build_task_extraction_prompt(messages: list[BaseMessage], retrieval_result) -> str:
    """Build the system prompt with examples, current chat, and integrated data sources context.

    :param messages: The native LangGraph messages to extract task from
    :param retrieval_result: Data retrieval result from external sources
    :return: Complete prompt for task extraction
    :rtype: str
    """

    prompt_provider = get_framework_prompts()
    task_extraction_builder = prompt_provider.get_task_extraction_prompt_builder()

    return task_extraction_builder.get_system_instructions(
        messages=messages, retrieval_result=retrieval_result
    )


def _extract_task(messages: list[BaseMessage], retrieval_result, logger) -> ExtractedTask:
    """Extract actionable task from native LangGraph messages with integrated data sources.

    Uses PydanticAI agent to analyze conversation and extract structured
    task information including context dependencies.

    :param messages: The native LangGraph messages
    :param retrieval_result: DataRetrievalResult containing data from all available sources
    :param logger: Logger instance
    :type logger: logging.Logger
    :return: ExtractedTask with parsed task and context information
    :rtype: ExtractedTask
    """
    if retrieval_result and retrieval_result.has_data:
        logger.debug(
            f"Injecting data sources into task extraction: {retrieval_result.get_summary()}"
        )

    prompt = _build_task_extraction_prompt(messages, retrieval_result)

    # Use structured LLM generation for task extraction
    task_extraction_config = get_model_config("task_extraction")
    response = get_chat_completion(
        message=prompt, model_config=task_extraction_config, output_model=ExtractedTask
    )

    return response


# =============================================================================
# CONVENTION-BASED TASK EXTRACTION NODE
# =============================================================================


@infrastructure_node
class TaskExtractionNode(BaseInfrastructureNode):
    """Convention-based task extraction node with sophisticated task processing logic.

    Extracts and processes user tasks with context analysis, dependency detection,
    and task refinement. Handles both initial task extraction and task updates
    from conversations.

    Features:
    - Configuration-driven error classification and retry policies
    - LLM-based task extraction with fallback mechanisms
    - Context-aware task processing
    - Dependency analysis for chat history and user memory
    - Sophisticated error handling for LLM operations
    """

    name = "task_extraction"
    description = "Task Extraction and Processing"

    @staticmethod
    def classify_error(exc: Exception, context: dict):
        """Built-in error classification for task extraction operations.

        :param exc: Exception that occurred during task extraction
        :type exc: Exception
        :param context: Execution context with task extraction details
        :type context: dict
        :return: Error classification for retry decisions
        :rtype: ErrorClassification
        """
        # Retry on network/API timeouts (LLM can be flaky)
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="Network timeout during task extraction, retrying...",
                metadata={"technical_details": str(exc)},
            )

        # Don't retry on validation or configuration errors
        if isinstance(exc, (ValueError, TypeError)):
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message="Task extraction configuration error",
                metadata={"technical_details": str(exc)},
            )

        # Don't retry on import/module errors (missing dependencies)
        if isinstance(exc, (ImportError, ModuleNotFoundError)):
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message="Task extraction dependencies not available",
                metadata={"technical_details": str(exc)},
            )

        # Default: CRITICAL for unknown errors (fail safe principle)
        # Only explicitly known errors should be RETRIABLE
        return ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message=f"Unknown task extraction error: {str(exc)}",
            metadata={
                "technical_details": f"Error type: {type(exc).__name__}, Details: {str(exc)}"
            },
        )

    @staticmethod
    def get_retry_policy() -> dict[str, Any]:
        """Custom retry policy for LLM-based task extraction operations.

        Task extraction uses LLM calls to parse user queries and can be flaky due to:
        - Network timeouts to LLM services
        - LLM provider rate limiting
        - Complex query parsing requirements

        Use standard retry attempts with moderate delays since task extraction
        is the entry point and should be reliable but not overly aggressive.
        """
        return {
            "max_attempts": 3,  # Standard attempts for entry point operation
            "delay_seconds": 1.0,  # Moderate delay for LLM service calls
            "backoff_factor": 1.5,  # Standard backoff for network issues
        }

    async def execute(self) -> dict[str, Any]:
        """Main task extraction logic with bypass support and error handling.

        Converts conversational exchanges into clear, actionable task descriptions.
        Analyzes native LangGraph messages and external data sources to extract the user's
        actual intent and dependencies on previous conversation context.

        Supports bypass mode where full chat history is passed directly as the task,
        skipping LLM-based extraction for performance optimization.

        :return: Dictionary of state updates to apply
        :rtype: Dict[str, Any]
        """
        state = self._state

        # Get unified logger with automatic streaming support
        logger = self.get_logger()

        # Get native LangGraph messages from flat state structure (move outside try block)
        messages = state["messages"]

        # Check if task extraction bypass is enabled
        bypass_enabled = state.get("agent_control", {}).get("task_extraction_bypass_enabled", False)

        if bypass_enabled:
            logger.info("Task extraction bypass enabled - using full context with data sources")
            logger.status("Bypassing task extraction - retrieving data and formatting full context")
        else:
            logger.status("Extracting actionable task from conversation")
        try:
            # Attempt to retrieve context from data sources if available
            retrieval_result = None
            try:
                data_manager = get_data_source_manager()
                requester = DataSourceRequester("task_extraction", "task_extraction")
                request = create_data_source_request(state, requester)
                retrieval_result = await data_manager.retrieve_all_context(request)
                logger.info(
                    f"Retrieved data from {retrieval_result.total_sources_attempted} sources"
                )
            except (ImportError, ModuleNotFoundError):
                logger.warning(
                    "Data source system not available - proceeding without external context"
                )
            except Exception as e:
                logger.warning(
                    f"Data source retrieval failed, proceeding without external context: {e}"
                )

            # Extract task using LLM or bypass mode with integrated data sources
            # Run sync function in thread pool to avoid blocking event loop for streaming
            if bypass_enabled:
                processed_task = await asyncio.to_thread(
                    _format_task_context, messages, retrieval_result, logger
                )
            else:
                processed_task = await asyncio.to_thread(
                    _extract_task, messages, retrieval_result, logger
                )

            if bypass_enabled:
                logger.info(
                    f" * Bypass mode: formatted context ({len(processed_task.task)} characters)"
                )
                logger.info(
                    f" * Builds on previous context: {processed_task.depends_on_chat_history}"
                )
                logger.info(f" * Uses memory context: {processed_task.depends_on_user_memory}")
                logger.success("Task extraction bypassed - full context ready")
            else:
                logger.info(f" * Extracted: '{processed_task.task[:100]}...'")
                logger.info(
                    f" * Builds on previous context: {processed_task.depends_on_chat_history}"
                )
                logger.info(f" * Uses memory context: {processed_task.depends_on_user_memory}")
                logger.success("Task extraction completed")

            # Create direct state update with correct field names
            return {
                "task_current_task": processed_task.task,
                "task_depends_on_chat_history": processed_task.depends_on_chat_history,
                "task_depends_on_user_memory": processed_task.depends_on_user_memory,
            }

        except Exception as e:
            # Task extraction failed - error() automatically streams
            logger.error(f"Task extraction failed: {e}")
            raise e  # Raise original error for better debugging
