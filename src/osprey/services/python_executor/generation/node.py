"""
Code Generator Node - LangGraph Architecture

Generates Python code using configurable code generation strategies.
Transformed for LangGraph integration with TypedDict state management.

This node now uses the factory pattern to select code generators based on
configuration, enabling pluggable generation strategies while maintaining
the same interface for the executor service.
"""

from typing import Any

from osprey.events.types import CodeGeneratedEvent, CodeGenerationStartEvent
from osprey.utils.logger import get_logger

from ..models import ExecutionError, PythonExecutionState
from .factory import create_code_generator

# Try to import LangGraph's stream writer for event emission
try:
    from langgraph.config import get_stream_writer

    LANGGRAPH_STREAMING_AVAILABLE = True
except ImportError:
    LANGGRAPH_STREAMING_AVAILABLE = False
    get_stream_writer = None

logger = get_logger("python_generator")


def _emit_code_event(event_name: str, **kwargs) -> None:
    """Emit code generation events if streaming is available.

    This helper emits both CodeGenerationStartEvent and CodeGeneratedEvent
    with proper serialization for LangGraph streaming.

    Args:
        event_name: The event class name ("CodeGenerationStartEvent" or "CodeGeneratedEvent")
        **kwargs: Event-specific parameters (attempt, is_retry, code, success, etc.)
    """
    if not LANGGRAPH_STREAMING_AVAILABLE:
        return

    try:
        stream_writer = get_stream_writer()
        if stream_writer:
            # Create appropriate event based on name
            if event_name == "CodeGenerationStartEvent":
                event = CodeGenerationStartEvent(component="python_code_generator", **kwargs)
            elif event_name == "CodeGeneratedEvent":
                event = CodeGeneratedEvent(component="python_code_generator", **kwargs)
            else:
                logger.warning(f"Unknown event type: {event_name}")
                return

            # Emit as dict for LangGraph streaming with proper event_class field
            event_dict = event.__dict__.copy()
            event_dict["event_class"] = event_name
            # Convert timestamp to ISO format if present
            if hasattr(event_dict.get("timestamp"), "isoformat"):
                event_dict["timestamp"] = event_dict["timestamp"].isoformat()
            stream_writer(event_dict)
    except Exception:
        # Not in a streaming context or streaming failed - silent fallback
        # This is expected when running outside of TUI (e.g., CLI, tests)
        pass


def create_generator_node():
    """Create the code generator node function.

    The node uses the factory pattern to create the appropriate code generator
    based on configuration. This enables pluggable generation strategies while
    maintaining the same interface for the executor service.

    Returns:
        Async function that implements the generator node logic
    """

    async def generator_node(state: PythonExecutionState) -> dict[str, Any]:
        """Generate Python code using configured code generator.

        Uses the factory pattern to create the appropriate generator based on
        configuration (legacy LLM, Claude Code, or custom generator).

        Args:
            state: LangGraph state containing request and execution tracking

        Returns:
            State update dict with generated code or error information
        """
        # Debug log what we received in state
        logger.debug(f"Generator node received state type: {type(state)}")
        logger.debug(
            f"Generator node state keys: {list(state.keys()) if hasattr(state, 'keys') else 'no keys method'}"
        )
        if hasattr(state, "keys") and "request" in state:
            logger.debug(f"Request found in state: {type(state['request'])}")
        else:
            logger.error(f"NO REQUEST FOUND IN STATE! State content: {state}")

        # Use unified logging system with streaming support
        streamer = get_logger("python_generator", state=state)
        streamer.status("Generating Python code...")

        # Create execution folder early if not already exists (for saving prompts)
        # This allows generators to save debug/prompt data during generation
        execution_folder = state.get("execution_folder")
        if not execution_folder:
            from osprey.utils.config import get_full_configuration

            from ..services import FileManager

            configurable = get_full_configuration()
            file_manager = FileManager(configurable)
            execution_context = file_manager.create_execution_folder(
                state["request"].execution_folder_name
            )
            execution_folder = execution_context
            logger.debug(
                f"Created execution folder for generation: {execution_context.folder_path}"
            )

        # Create generator via factory - configuration-driven selection
        generator = create_code_generator()

        logger.info(f"Using generator: {type(generator).__name__}")

        try:
            # Add execution folder path to request for generators that save prompts
            request = state["request"]
            if execution_folder and hasattr(execution_folder, "folder_path"):
                # Create a modified request with execution folder path
                # We use model_copy to avoid mutating the original
                request_dict = request.model_dump()
                request_dict["execution_folder_path"] = str(execution_folder.folder_path)
                from ..models import PythonExecutionRequest

                request = PythonExecutionRequest(**request_dict)

            # EMIT EVENT: Signal new generation attempt starting
            # This arrives BEFORE any tokens, preventing race conditions
            current_attempt = state.get("generation_attempt", 0) + 1
            is_retry = len(state.get("error_chain", [])) > 0
            _emit_code_event("CodeGenerationStartEvent", attempt=current_attempt, is_retry=is_retry)

            # Generate code with error feedback from previous attempts
            # Same interface for all generators - clean Protocol-based design
            # Native LangGraph streaming captures tokens automatically via subgraphs=True
            generated_code = await generator.generate_code(
                request,  # Pass request with execution_folder_path
                state.get("error_chain", []),  # Use service state for error tracking
            )

            # EMIT EVENT: Signal code generation completed successfully
            # This triggers TUI to finalize the widget (close stream, update title, auto-collapse)
            _emit_code_event(
                "CodeGeneratedEvent", code=generated_code, attempt=current_attempt, success=True
            )

            streamer.status(f"Generated {len(generated_code)} characters of code")

            logger.info(f"Code generated successfully: {len(generated_code)} characters")

            # Collect metadata if generator provides it
            metadata = None
            if hasattr(generator, "get_generation_metadata"):
                try:
                    metadata = generator.get_generation_metadata()
                    logger.debug(
                        f"Generator metadata: {list(metadata.keys()) if metadata else 'none'}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to get generator metadata: {e}")

            # Update state with generated code, metadata, and execution folder
            state_update = {
                "generated_code": generated_code,
                "generation_attempt": state.get("generation_attempt", 0) + 1,
                "current_stage": "analysis",
                "code_generator_metadata": metadata,
            }

            # Add execution folder to state if we created it
            if execution_folder and not state.get("execution_folder"):
                state_update["execution_folder"] = execution_folder

            return state_update

        except Exception as e:
            logger.error(f"Code generation failed: {e}")

            # Add structured error to chain and check retry limits
            import traceback

            error = ExecutionError(
                error_type="generation",
                error_message=str(e),
                traceback=traceback.format_exc(),
                attempt_number=state.get("generation_attempt", 0) + 1,
                stage="generation",
            )
            error_chain = state.get("error_chain", []) + [error]
            max_retries = state["request"].retries

            if len(error_chain) >= max_retries:
                return {
                    "error_chain": error_chain,
                    "is_failed": True,
                    "failure_reason": f"Code generation failed after {max_retries} attempts",
                }
            else:
                # EMIT EVENT: Signal retry attempt starting
                retry_attempt = state.get("generation_attempt", 0) + 1
                _emit_code_event("CodeGenerationStartEvent", attempt=retry_attempt, is_retry=True)

                return {
                    "error_chain": error_chain,
                    "generation_attempt": retry_attempt,
                }

    return generator_node
