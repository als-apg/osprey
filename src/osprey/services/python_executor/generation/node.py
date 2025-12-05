"""
Code Generator Node - LangGraph Architecture

Generates Python code using configurable code generation strategies.
Transformed for LangGraph integration with TypedDict state management.

This node now uses the factory pattern to select code generators based on
configuration, enabling pluggable generation strategies while maintaining
the same interface for the executor service.
"""

from typing import Any

from osprey.utils.logger import get_logger

from ..models import ExecutionError, PythonExecutionState
from .factory import create_code_generator

logger = get_logger("python_generator")


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

            # Generate code with error feedback from previous attempts
            # Same interface for all generators - clean Protocol-based design
            generated_code = await generator.generate_code(
                request,  # Pass request with execution_folder_path
                state.get("error_chain", []),  # Use service state for error tracking
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
                return {
                    "error_chain": error_chain,
                    "generation_attempt": state.get("generation_attempt", 0) + 1,
                }

    return generator_node
