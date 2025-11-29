"""Memory Capability - Framework-Native User Memory Management

This capability provides comprehensive user memory operations including persistent storage,
retrieval, and management of user information and context. It integrates seamlessly with
the framework's approval system, context management, and LangGraph execution model to
provide controlled memory modifications and context injection for other capabilities.

The memory capability handles two primary operations:
1. **Memory Storage**: Extracts memory-worthy content from chat history and stores it
2. **Memory Retrieval**: Fetches stored memory entries for context in other operations

Key Features:
    - LLM-based content extraction from chat conversations
    - Approval system integration for controlled memory modifications
    - Persistent storage through the memory storage manager
    - Context injection for other capabilities requiring user context
    - Structured memory operations with comprehensive error handling

The capability uses sophisticated LLM-based analysis to identify content worth saving
from chat interactions, then integrates with the approval system to ensure user control
over what information gets permanently stored.

.. note::
   Memory operations require user ID availability in the session configuration.
   All memory modifications go through the approval system unless configured otherwise.

.. seealso::
   :class:`osprey.services.memory_storage.MemoryStorageManager` : Memory persistence
   :class:`osprey.approval.ApprovalManager` : Memory approval workflows
   :class:`MemoryContext` : Context structure for memory operation results
"""
import asyncio
import textwrap
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar

from langchain_core.messages import BaseMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from osprey.approval import (
    clear_approval_state,
    create_approval_type,
    create_memory_approval_interrupt,
    get_approval_resume_data,
)
from osprey.approval.approval_manager import get_memory_evaluator

# Import from framework architecture
from osprey.base import (
    BaseCapability,
    BaseExample,
    OrchestratorGuide,
    TaskClassifierGuide,
)
from osprey.base.decorators import _is_graph_interrupt, capability_node
from osprey.base.errors import ErrorClassification, ErrorSeverity
from osprey.context import CapabilityContext
from osprey.context.context_manager import ContextManager
from osprey.models import get_chat_completion
from osprey.prompts.loader import get_framework_prompts
from osprey.registry import get_registry
from osprey.services.memory_storage import MemoryContent, get_memory_storage_manager
from osprey.state import AgentState, ChatHistoryFormatter, StateManager
from osprey.utils.config import get_model_config, get_session_info
from osprey.utils.logger import get_logger

# Module-level logger for helper functions
logger = get_logger("memory")


# ===========================================================
# Context Classes
# ===========================================================

class MemoryContext(CapabilityContext):
    """Framework memory context for storing and retrieving user memory data.

    Provides structured context for memory operations including save and retrieve
    operations. This context integrates with the execution context system to provide
    memory data access to other capabilities that need user context information.

    The context maintains operation metadata and results, allowing capabilities to
    understand both what memory operation was performed and access the resulting data.
    This enables sophisticated workflows where capabilities can build upon previously
    stored or retrieved user information.

    :param memory_data: Dictionary containing memory operation data and results
    :type memory_data: Dict[str, Any]
    :param operation_type: Type of memory operation performed ('store', 'retrieve', 'search')
    :type operation_type: str
    :param operation_result: Human-readable result message from the operation
    :type operation_result: Optional[str]

    .. note::
       The memory_data structure varies based on operation_type:
       - 'store': Contains saved_content and timestamp
       - 'retrieve': Contains memories list with all stored entries
       - 'search': Contains filtered results based on search criteria

    .. seealso::
       :class:`osprey.context.base.CapabilityContext` : Base context functionality
       :class:`osprey.services.memory_storage.MemoryContent` : Memory entry structure
       :meth:`MemoryOperationsCapability.execute` : Main capability that creates this context
       :func:`_perform_memory_save_operation` : Save operation that produces this context
       :func:`_perform_memory_retrieve_operation` : Retrieve operation that produces this context
    """
    CONTEXT_TYPE: ClassVar[str] = "MEMORY_CONTEXT"
    CONTEXT_CATEGORY: ClassVar[str] = "CONTEXTUAL_KNOWLEDGE"

    memory_data: dict[str, Any]
    operation_type: str  # 'store', 'retrieve', 'search'
    operation_result: str | None = None

    def get_access_details(self, key: str) -> dict[str, Any]:
        """Provide detailed access information for capability context integration.

        Generates comprehensive access details for other capabilities to understand
        how to interact with this memory context data. Includes access patterns,
        example usage, and data structure descriptions.

        :param key_name: Optional context key name for access pattern generation
        :type key_name: Optional[str]
        :return: Dictionary containing access details and usage examples
        :rtype: Dict[str, Any]

        .. note::
           This method is called by the framework's context management system
           to provide integration guidance for other capabilities.
        """
        return {
            "operation": self.operation_type,
            "data_keys": list(self.memory_data.keys()) if self.memory_data else [],
            "access_pattern": f"context.{self.CONTEXT_TYPE}.{key}.memory_data",
            "example_usage": f"context.{self.CONTEXT_TYPE}.{key}.memory_data['user_preferences'] gives stored user preferences",
            "operation_result": self.operation_result
        }

    def get_summary(self) -> dict[str, Any]:
        """Generate summary for response generation and UI display.

        Creates a formatted summary of the memory operation results suitable for
        display in user interfaces and inclusion in agent responses. Returns raw
        data structures for robust LLM processing rather than pre-formatted strings.

        :param key_name: Optional context key name for reference
        :type key_name: Optional[str]
        :return: Dictionary containing memory operation summary
        :rtype: Dict[str, Any]

        .. note::
           This method returns structured data rather than formatted strings
           to enable robust LLM processing and response generation.
        """
        # Just return the raw data - let the LLM handle formatting
        # This is much more robust than trying to hardcode string formatting
        return {
            "operation_type": self.operation_type,
            "operation_result": self.operation_result,
            "memory_data": self.memory_data
        }


# =============================================================================
# Memory extraction examples
# =============================================================================

class MemoryOperation(Enum):
    """Enumeration of supported memory operation types.

    Defines the available memory operations that can be performed by the
    memory capability. Used for operation classification and routing.

    :cvar SAVE: Store new content to user memory
    :cvar RETRIEVE: Fetch existing memory entries for context
    """
    SAVE = "save"
    RETRIEVE = "retrieve"

class MemoryOperationClassification(BaseModel):
    """Structured output model for memory operation classification.

    Pydantic model used by LLM to classify user requests into specific memory
    operations. Replaces fragile regex-based classification with robust LLM
    analysis that can understand context and intent.

    The model ensures structured output from LLM calls and provides reasoning
    for classification decisions to enable debugging and validation.

    :param operation: The classified memory operation type ('save' or 'retrieve')
    :type operation: str
    :param reasoning: Detailed explanation for the classification decision
    :type reasoning: str

    .. note::
       This model is used with structured LLM output to ensure reliable
       operation classification without regex pattern matching.

    .. seealso::
       :func:`_classify_memory_operation` : Function that uses this model for classification
       :class:`MemoryContentExtraction` : Related model for content extraction
       :class:`MemoryOperation` : Enum defining available operation types
    """
    operation: str = Field(description="The memory operation type: 'save' for storing new content, 'retrieve' for showing existing memories")
    reasoning: str = Field(description="Brief explanation of why this operation was selected")

class MemoryContentExtraction(BaseModel):
    """Structured output model for memory content extraction from chat history.

    Pydantic model used by LLM to analyze chat conversations and extract
    content worth saving to user memory. Uses sophisticated analysis to
    identify important information, preferences, and context that should
    be preserved for future interactions.

    The model provides both the extracted content and metadata about the
    extraction decision to enable validation and debugging.

    :param content: Extracted content to save to memory, empty string if none found
    :type content: str
    :param found: Whether memory-worthy content was identified in the conversation
    :type found: bool
    :param explanation: Detailed reasoning for the extraction decision
    :type explanation: str

    .. note::
       The LLM analyzes full chat history including context from previous
       capabilities to identify comprehensive memory-worthy content.
    """
    content: str = Field(description="The content that should be saved to memory, or empty string if no content identified")
    found: bool = Field(description="True if content to save was identified in the user message, False otherwise")
    explanation: str = Field(description="Brief explanation of what content was extracted and why")

@dataclass
class MemoryExtractionExample(BaseExample):
    """Structured example for memory content extraction training.

    Training example that demonstrates proper memory content extraction patterns
    to the LLM. Shows how to analyze chat conversations and identify content
    worth preserving in user memory.

    These examples are used in few-shot learning to teach the LLM how to
    distinguish between transient conversation content and information that
    has lasting value for the user relationship.

    :param messages: Example chat conversation using native LangGraph message format
    :type messages: List[BaseMessage]
    :param expected_output: Expected extraction result demonstrating correct analysis
    :type expected_output: MemoryContentExtraction

    .. note::
       Examples use native LangGraph message formats for consistency with
       the framework's chat history management system.

    .. seealso::
       :class:`osprey.base.examples.BaseExample` : Base example structure
       :class:`MemoryContentExtraction` : Expected output model
       :meth:`MemoryOperationsCapability._create_classifier_guide` : Uses this example class
       :func:`_extract_memory_content` : Function that leverages these examples
    """
    messages: list[BaseMessage]
    expected_output: MemoryContentExtraction

    def format_for_prompt(self) -> str:
        """Format this example for inclusion in LLM training prompts.

        Converts the structured example into a formatted string suitable for
        inclusion in LLM prompts. Uses the framework's chat history formatter
        for consistent message presentation.

        :return: Formatted example string ready for LLM prompt inclusion
        :rtype: str

        .. note::
           Uses ChatHistoryFormatter.format_for_llm() for consistent message
           formatting throughout the framework (architectural system).
        """
        # Format chat history using native message formatter
        chat_formatted = ChatHistoryFormatter.format_for_llm(self.messages)

        return textwrap.dedent(f"""
            **Chat History:**
            {textwrap.indent(chat_formatted, "  ")}

            **Expected Output:**
            {{
                "content": "{self.expected_output.content}",
                "found": {str(self.expected_output.found).lower()},
                "explanation": "{self.expected_output.explanation}"
            }}
            """).strip()

# Examples are provided by the application layer through the prompt builder system
# See: applications/als_assistant/framework_prompts/memory_extraction.py

def _get_memory_extraction_system_instructions() -> str:
    """Create system instructions for LLM-based memory content extraction.

    Retrieves comprehensive system instructions from the framework's prompt
    builder system. These instructions guide the LLM in analyzing chat history
    and identifying content worth saving to user memory.

    :return: Complete system instruction prompt for memory extraction operations
    :rtype: str

    .. note::
       Instructions are provided by the application layer through the prompt
       builder system, enabling domain-specific memory extraction guidance.

    .. seealso::
       :mod:`applications.als_assistant.osprey_prompts.memory_extraction` : Application prompts
    """

    prompt_provider = get_framework_prompts()
    memory_builder = prompt_provider.get_memory_extraction_prompt_builder()

    return memory_builder.get_system_instructions()

async def _classify_memory_operation(task_objective: str, logger) -> MemoryOperation:
    """Classify memory operation using LLM-based analysis.

    Uses sophisticated LLM analysis to determine whether a user request requires
    saving new content to memory or retrieving existing memories. This approach
    replaces fragile regex-based classification with robust natural language
    understanding that can handle complex and varied user expressions.

    The classification process uses structured LLM output with reasoning to ensure
    reliable operation identification and enable debugging of classification decisions.

    :param task_objective: The user task description to analyze and classify
    :type task_objective: str
    :param logger: Logger instance for debugging and operation tracking
    :type logger: logging.Logger
    :return: Classified memory operation type
    :rtype: MemoryOperation

    :raises ContentExtractionError: If LLM classification fails or returns invalid operation
    :raises LLMCallError: If the underlying LLM call fails

    .. note::
       Uses the framework's configuration system for model selection
       and structured output parsing for reliable classification.

    Examples:
        Typical classification scenarios::

            >>> await _classify_memory_operation("save my preferences", logger)
            MemoryOperation.SAVE

            >>> await _classify_memory_operation("show me what you remember", logger)
            MemoryOperation.RETRIEVE

    .. seealso::
       :class:`MemoryOperationClassification` : Output model used by this function
       :class:`MemoryOperation` : Enum values returned by this function
       :meth:`MemoryOperationsCapability.execute` : Main method that uses this classification
       :func:`_perform_memory_save_operation` : Save operation implementation
       :func:`_perform_memory_retrieve_operation` : Retrieve operation implementation
    """

    # Get classification prompt from framework prompt builder
    prompt_provider = get_framework_prompts()
    memory_builder = prompt_provider.get_memory_extraction_prompt_builder()
    system_prompt = memory_builder.get_memory_classification_prompt()

    user_prompt = f"Classify this memory task: '{task_objective}'"

    try:
        # Use config helper for model configuration
        classifier_config = get_model_config("classifier")

        # Create full message combining system and user prompts
        full_message = f"{system_prompt}\n\nUser task: {user_prompt}"

        # Set caller context for API call logging (propagates through asyncio.to_thread)
        from osprey.models import set_api_call_context
        set_api_call_context(
            function="_classify_memory_operation",
            module="memory",
            class_name="MemoryCapability",
            extra={"capability": "memory", "operation": "classification"}
        )

        response_data = await asyncio.to_thread(
            get_chat_completion,
            model_config=classifier_config,
            message=full_message,
            output_model=MemoryOperationClassification,
        )

        if isinstance(response_data, MemoryOperationClassification):
            classification = response_data
        else:
            logger.error(f"Memory operation classification did not return expected model. Got: {type(response_data)}")
            raise ContentExtractionError("Failed to classify memory operation")

        # Validate and convert to enum
        operation_str = classification.operation.lower()
        logger.info(f"Memory operation classified as: {operation_str} (reasoning: {classification.reasoning})")

        if operation_str == "save":
            return MemoryOperation.SAVE
        elif operation_str == "retrieve":
            return MemoryOperation.RETRIEVE
        else:
            logger.error(f"Invalid memory operation returned by LLM: {operation_str}")
            raise ContentExtractionError(f"Unknown memory operation: {operation_str}")

    except Exception as e:
        logger.error(f"Failed to classify memory operation: {e}")
        raise ContentExtractionError(f"Memory operation classification failed: {e}")

# =============================================================================
# Exception Classes
# =============================================================================

class MemoryCapabilityError(Exception):
    """Base exception class for memory capability-specific errors.

    Provides a hierarchy of memory-related exceptions to enable sophisticated
    error handling and classification. All memory capability errors inherit
    from this base class for consistent exception handling.

    .. seealso::
       :meth:`MemoryOperationsCapability.classify_error` : Error classification for recovery strategies
       :class:`UserIdNotAvailableError` : Specific error for missing user identification
       :class:`ContentExtractionError` : Specific error for failed content extraction
       :class:`MemoryFileError` : Specific error for storage system failures
    """
    pass

class UserIdNotAvailableError(MemoryCapabilityError):
    """Raised when user ID is not available in session configuration.

    Memory operations require a valid user ID to associate stored content
    with the correct user account. This error indicates that the session
    configuration does not contain the required user identification.
    """
    pass

class ContentExtractionError(MemoryCapabilityError):
    """Raised when content extraction from chat history fails.

    Indicates that the LLM-based content extraction process could not
    identify memory-worthy content in the conversation, or that the
    extraction process itself failed due to technical issues.
    """
    pass

class MemoryFileError(MemoryCapabilityError):
    """Raised when memory storage operations fail.

    Indicates that the underlying memory storage system encountered an error
    during save or load operations. This could be due to file system issues,
    permission problems, or storage system failures.
    """
    pass

class MemoryRetrievalError(MemoryCapabilityError):
    """Raised when memory retrieval operations fail.

    Indicates that the system could not successfully retrieve stored memory
    entries, either due to storage system issues or data corruption problems.
    """
    pass

class LLMCallError(MemoryCapabilityError):
    """Raised when LLM operations for memory processing fail.

    Indicates that calls to the language model for content extraction,
    operation classification, or other memory-related analysis failed
    due to model errors, configuration issues, or service unavailability.
    """
    pass

 # ===========================================================
# Convention-Based Capability
# ===========================================================

from osprey.base.capability import BaseCapability


@capability_node
class MemoryOperationsCapability(BaseCapability):
    """Memory operations capability for comprehensive user memory management.

    Provides sophisticated user memory management functionality including persistent
    storage of user information, intelligent content extraction, and memory retrieval
    for contextual enhancement of other capabilities. The capability integrates
    seamlessly with the framework's approval system, context management, and
    LangGraph execution model.

    The capability handles two primary workflows:
    1. **Memory Storage**: Analyzes chat history to extract memory-worthy content,
       optionally requests user approval, and stores the content persistently
    2. **Memory Retrieval**: Fetches stored memory entries and provides them as
       context for other capabilities to enhance their operations

    Key architectural features:
        - LLM-based content extraction with sophisticated analysis
        - Integration with approval system for controlled memory modifications
        - Context injection system for cross-capability memory access
        - Comprehensive error handling and classification
        - Support for both synchronous and asynchronous memory operations

    The capability uses the @capability_node decorator for full LangGraph integration
    including error handling, retry policies, execution tracking, and streaming support.

    .. note::
       Memory operations require valid user ID in session configuration.
       All memory modifications can be configured to require user approval.

    .. warning::
       Memory content is stored persistently and should be handled according
       to appropriate data privacy and security policies.

    .. seealso::
       :class:`osprey.base.capability.BaseCapability` : Base capability functionality
       :class:`MemoryContext` : Memory operation result context
       :class:`osprey.services.memory_storage.MemoryStorageManager` : Storage backend
    """

    name = "memory"
    description = "Save content to and retrieve content from user memory files"
    provides = ["MEMORY_CONTEXT"]
    requires = []

    async def execute(self) -> dict[str, Any]:
        """Execute memory operations with comprehensive approval and context integration.

        Implements a sophisticated 3-phase execution pattern that handles both
        approved operations and new requests with seamless approval system integration:

        1. **Approval Resume Phase**: Handles execution of previously approved operations
        2. **Operation Analysis Phase**: Classifies and processes new memory requests
        3. **Approval Integration Phase**: Manages approval workflows for memory modifications

        The method supports both memory storage (with LLM-based content extraction)
        and memory retrieval operations, automatically handling context creation and
        state management for seamless integration with other capabilities.

        :return: State updates with memory operation results and context data
        :rtype: Dict[str, Any]

        :raises UserIdNotAvailableError: If user ID is not available in session config
        :raises ContentExtractionError: If memory content extraction fails
        :raises MemoryFileError: If memory storage operations fail
        :raises MemoryRetrievalError: If memory retrieval operations fail
        :raises LLMCallError: If LLM operations for memory processing fail

        .. note::
           The method uses StateManager for context storage and supports both
           synchronous and asynchronous approval workflows through LangGraph interrupts.

        .. warning::
           Memory save operations may trigger approval interrupts that suspend
           execution until user approval is received.

        .. seealso::
           :func:`_classify_memory_operation` : Operation classification used by this method
           :func:`_perform_memory_save_operation` : Save operation implementation
           :func:`_perform_memory_retrieve_operation` : Retrieve operation implementation
           :class:`MemoryContext` : Context structure returned by this method
           :class:`osprey.approval.ApprovalManager` : Approval system integration
           :func:`osprey.approval.get_approval_resume_data` : Approval resume handling
        """

        # Get unified logger with automatic streaming support
        logger = self.get_logger()
        state = self._state

        # =====================================================================
        # PHASE 1: CHECK FOR APPROVED MEMORY OPERATION (HIGHEST PRIORITY)
        # =====================================================================

        has_approval_resume, approved_payload = get_approval_resume_data(self._state, create_approval_type("memory", "save"))

        if has_approval_resume and approved_payload:
            logger.success("Using approved memory operation from agent state")
            logger.status("Executing approved memory operation...")

            # Execute the approved memory operation
            content = approved_payload.get("content")
            user_id = approved_payload.get("user_id")

            # Perform the memory operation using unified helper
            memory_context = await _perform_memory_save_operation(content, user_id, logger)

            # Store context using StateManager
            step = self._step
            context_update = StateManager.store_context(
                self._state,
                "MEMORY_CONTEXT",
                step.get("context_key"),
                memory_context
            )
            approval_cleanup = clear_approval_state()

            # Combine context storage with approval state cleanup
            return {**context_update, **approval_cleanup}

        # =====================================================================
        # PHASE 2: NORMAL MEMORY OPERATION FLOW
        # =====================================================================

        # Extract current step from execution plan (single source of truth)
        step = self._step

        try:
            # Get user ID from config system
            session_info = get_session_info()
            user_id = session_info.get("user_id")

            if not user_id:
                raise UserIdNotAvailableError("Cannot perform memory operations: user ID not available in config")

            # Extract and classify the request
            task_objective = step.get('task_objective', '')

            # Use LLM-based classification
            operation = await _classify_memory_operation(task_objective, logger)

            if operation == MemoryOperation.RETRIEVE:
                logger.status("Retrieving user memory...")

                memory_context = await _perform_memory_retrieve_operation(user_id, logger)
                logger.status("Memory retrieval complete")

                # Store context using helper method
                return self.store_output_context(memory_context)

            elif operation == MemoryOperation.SAVE:
                logger.status("Extracting content to save...")

                # Extract content to save from chat history using correct LangGraph state access
                messages = state.get("messages", [])
                memory_extraction_result = None

                if messages:
                    # Build system instructions and user input
                    system_instructions = _get_memory_extraction_system_instructions()
                    chat_formatted = ChatHistoryFormatter.format_for_llm(messages)

                    # Check if we have context inputs from previous steps and include them
                    step = self._step
                    step_inputs = step.get('inputs', [])
                    context_section = ""

                    if step_inputs:
                        logger.info(f"Memory save: Including context from {len(step_inputs)} previous steps")
                        try:
                            context_manager = ContextManager(self._state)
                            context_summaries = context_manager.get_summaries(step)

                            if context_summaries:
                                # Format list as readable string
                                import json
                                formatted_summaries = json.dumps(context_summaries, indent=2, default=str)
                                context_section = f"\n\nAVAILABLE CONTEXT FROM PREVIOUS STEPS:\n{formatted_summaries}\n"
                                logger.debug(f"Added context summaries: {context_summaries}")
                        except Exception as e:
                            logger.warning(f"Failed to get context summaries: {e}")

                    query = f"Please analyze this chat history{' and available context' if context_section else ''} and extract any content to save:\n\n{chat_formatted}{context_section}"

                    try:
                        memory_model_config = get_model_config("memory")

                        # Use structured LLM generation for memory extraction
                        message = f"{system_instructions}\n\n{query}"

                        # Set caller context for API call logging (propagates through asyncio.to_thread)
                        from osprey.models import set_api_call_context
                        set_api_call_context(
                            function="_extract_memory_content",
                            module="memory",
                            class_name="MemoryCapability",
                            extra={"capability": "memory", "operation": "extraction"}
                        )

                        response_data = await asyncio.to_thread(
                            get_chat_completion,
                            message=message,
                            model_config=memory_model_config,
                            output_model=MemoryContentExtraction
                        )

                        # Enhanced logging pattern for debugging
                        logger.info("LLM extraction result:")
                        logger.info(f" found={response_data.found}")
                        logger.info(f" content='{response_data.content}'")
                        logger.info(f" explanation='{response_data.explanation}'")

                        if response_data.found and response_data.content.strip():
                            memory_extraction_result = response_data
                        else:
                            logger.debug(f"No content identified for saving: {response_data.explanation}")

                    except Exception as e:
                        logger.error(f"LLM call failed for memory content extraction: {e}")
                        raise LLMCallError(f"LLM call failed for memory content extraction: {e}")
                else:
                    logger.debug("No messages found in state for content extraction")

                if not memory_extraction_result:
                    raise ContentExtractionError("No content specified for memory save operation")

                logger.status("Preparing memory content...")

                # =====================================================================
                # PHASE 3: APPROVAL CHECK AND INTERRUPT HANDLING
                # =====================================================================

                # Check if approval is required using the new approval system
                evaluator = get_memory_evaluator()
                decision = evaluator.evaluate(operation_type="save")

                if decision.needs_approval:
                    logger.info(f"Memory operation requires approval: {decision.reasoning}")

                    logger.status("Requesting memory approval...")

                    # Create structured memory approval interrupt
                    interrupt_data = create_memory_approval_interrupt(
                        content=memory_extraction_result.content,
                        operation_type="save",
                        user_id=user_id,
                        step_objective=step.get('task_objective', 'Save content to memory')
                    )

                    logger.approval("Interrupting execution for memory approval")
                    logger.debug(f"Interrupt data created for memory content: '{memory_extraction_result.content[:100]}...'")

                    # LangGraph interrupt - execution stops here until user responds
                    interrupt(interrupt_data)
                else:
                    # No approval needed - proceed directly
                    logger.info("Memory save operation allowed without approval")

                    # Execute the memory save operation
                    logger.status("Saving to memory...")
                    memory_context = await _perform_memory_save_operation(
                        content=memory_extraction_result.content,
                        user_id=user_id,
                        logger=logger
                    )
                    logger.status("Memory saved successfully")

                    # Store context using helper method
                    return self.store_output_context(memory_context)
            else:
                raise ContentExtractionError("Unknown memory operation. Supported operations: save content to memory, show memory")

        except Exception as e:
            # Re-raise GraphInterrupt immediately - it's not an error!
            if _is_graph_interrupt(e):
                logger.info("GraphInterrupt detected in memory - re-raising for LangGraph to handle")
                raise e

            logger.error(f"Memory operation failed: {e}")
            raise

    @staticmethod
    def classify_error(exc: Exception, context: dict) -> ErrorClassification:
        """Classify memory operation errors for sophisticated recovery strategies.

        Provides domain-specific error classification for memory operations,
        enabling appropriate recovery strategies based on the specific failure mode.
        Maps memory-specific exceptions to framework error severities with
        appropriate user messages and technical details.

        :param exc: The exception that occurred during memory operation
        :type exc: Exception
        :param context: Error context including capability info and execution state
        :type context: dict
        :return: Error classification with recovery strategy and user messaging
        :rtype: ErrorClassification

        .. note::
           Classification determines framework response:
           - CRITICAL: End execution immediately
           - RETRIABLE: Retry with same parameters
           - REPLANNING: Create new execution plan

        .. seealso::
           :class:`osprey.base.errors.ErrorClassification` : Error classification structure
           :class:`osprey.base.errors.ErrorSeverity` : Available severity levels
           :class:`MemoryCapabilityError` : Base exception class for memory errors
           :class:`UserIdNotAvailableError` : Specific error handled by this method
           :class:`ContentExtractionError` : Specific error handled by this method
        """
        if isinstance(exc, UserIdNotAvailableError):
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message="Cannot perform memory operations: user ID not available",
                metadata={"technical_details": str(exc)}
            )
        elif isinstance(exc, ContentExtractionError):
            return ErrorClassification(
                severity=ErrorSeverity.REPLANNING,
                user_message="Need clarification on what to save to memory",
                metadata={"technical_details": str(exc)}
            )
        elif isinstance(exc, MemoryRetrievalError):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="Failed to retrieve memory, retrying...",
                metadata={"technical_details": str(exc)}
            )
        elif isinstance(exc, MemoryFileError):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="Memory file operation failed, retrying...",
                metadata={"technical_details": str(exc)}
            )
        else:
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message=f"Memory operation error: {str(exc)}",
                metadata={"technical_details": str(exc)}
            )

    def _create_orchestrator_guide(self) -> OrchestratorGuide | None:
        """Create orchestrator integration guide from prompt builder system.

        Retrieves sophisticated orchestration guidance from the application's
        prompt builder system. This guide teaches the orchestrator when and how
        to invoke memory operations within execution plans.

        :return: Orchestrator guide for memory capability integration
        :rtype: Optional[OrchestratorGuide]

        .. note::
           Guide content is provided by the application layer through the
           framework's prompt builder system for domain-specific customization.

        .. seealso::
           :class:`osprey.base.examples.OrchestratorGuide` : Guide structure returned by this method
           :meth:`_create_classifier_guide` : Complementary classifier guide creation
           :class:`osprey.prompts.loader.FrameworkPrompts` : Prompt system integration
        """
        prompt_provider = get_framework_prompts()
        memory_builder = prompt_provider.get_memory_extraction_prompt_builder()

        return memory_builder.get_orchestrator_guide()

    def _create_classifier_guide(self) -> TaskClassifierGuide | None:
        """Create task classification guide from prompt builder system.

        Retrieves task classification guidance from the application's prompt
        builder system. This guide teaches the classifier when user requests
        should be routed to memory operations.

        :return: Classification guide for memory capability activation
        :rtype: Optional[TaskClassifierGuide]

        .. note::
           Guide content is provided by the application layer through the
           framework's prompt builder system for domain-specific examples.
        """
        prompt_provider = get_framework_prompts()
        memory_builder = prompt_provider.get_memory_extraction_prompt_builder()

        return memory_builder.get_classifier_guide()


# Create instance for registration
memory_capability = MemoryOperationsCapability()


# =============================================================================
# BUSINESS LOGIC HELPERS
# =============================================================================




async def _perform_memory_save_operation(
    content: str,
    user_id: str,
    logger
) -> MemoryContext:
    """Execute memory save operation with comprehensive validation and error handling.

    Centralizes all memory save business logic into a reusable helper function
    that handles the complete save workflow including content validation, storage
    manager interaction, and result context creation.

    The function creates a structured memory entry, saves it through the storage
    manager, and returns a formatted context object for capability integration.

    :param content: Content to save to user memory
    :type content: str
    :param user_id: Unique user identifier for memory association
    :type user_id: str
    :param logger: Logger instance for operation tracking and debugging
    :type logger: logging.Logger
    :return: Memory context object containing save operation results
    :rtype: MemoryContext

    :raises MemoryFileError: If save operation fails or storage cannot be written

    .. note::
       Creates timestamped memory entries with metadata for comprehensive
       tracking and retrieval support.

    .. seealso::
       :class:`osprey.services.memory_storage.MemoryStorageManager` : Storage backend
       :class:`MemoryContent` : Memory entry structure
       :func:`_perform_memory_retrieve_operation` : Complementary retrieve operation
       :class:`MemoryContext` : Context structure for both save and retrieve operations
    """

    logger.info("Executing memory save operation")

    try:
        # Create memory entry from content
        memory_entry = MemoryContent(
            timestamp=datetime.now(),
            content=content
        )

        # Save to memory
        memory_manager = get_memory_storage_manager()
        success = memory_manager.add_memory_entry(user_id, memory_entry)

        if success:

            logger.success(f"Memory save completed for user {user_id}")

            # Return MemoryContext with save operation results
            return MemoryContext(
                memory_data={"saved_content": content, "timestamp": memory_entry.timestamp.isoformat()},
                operation_type="save",
                operation_result="Successfully saved content to memory"
            )
        else:
            raise MemoryFileError("Failed to save memory content to file")

    except Exception as e:
        logger.error(f"Memory operation failed: {e}")
        raise


async def _perform_memory_retrieve_operation(
    user_id: str,
    logger
) -> MemoryContext:
    """Execute memory retrieval operation with comprehensive error handling.

    Centralizes all memory retrieval business logic into a reusable helper
    function that handles the complete retrieval workflow including storage
    manager interaction, entry validation, and result context creation.

    The function retrieves all stored memory entries for the specified user
    and formats them into a structured context object for capability integration.

    :param user_id: Unique user identifier for memory retrieval
    :type user_id: str
    :param logger: Logger instance for operation tracking and debugging
    :type logger: logging.Logger
    :return: Memory context object containing retrieved memory entries
    :rtype: MemoryContext

    :raises MemoryRetrievalError: If retrieval operation fails or entries cannot be loaded

    .. note::
       Retrieved entries include all stored content with timestamps and
       metadata for comprehensive context provision.

    .. seealso::
       :class:`osprey.services.memory_storage.MemoryStorageManager` : Storage backend
       :class:`MemoryContext` : Result context structure
       :func:`_perform_memory_save_operation` : Complementary save operation
       :class:`MemoryContent` : Individual memory entry structure
    """

    logger.info("Executing memory retrieve operation")

    try:
        # Retrieve all memory entries
        memory_manager = get_memory_storage_manager()
        memory_entries = memory_manager.get_all_memory_entries(user_id)

        logger.info(f"Memory retrieval completed for user {user_id}: {len(memory_entries)} entries")

        # Return MemoryContext with retrieval results
        return MemoryContext(
            memory_data={"memories": memory_entries},
            operation_type="retrieve",
            operation_result=f"Retrieved {len(memory_entries)} memory entries"
        )

    except Exception as e:
        logger.error(f"Memory retrieve operation failed: {e}")
        raise MemoryRetrievalError(f"Failed to retrieve memory entries: {e}")


