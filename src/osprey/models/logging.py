"""LLM API Call Logging for Transparency and Debugging.

This module provides comprehensive logging of all LLM API interactions through the
get_chat_completion function. It captures raw input/output data along with rich
metadata including caller information, timestamps, and model parameters.

Key capabilities:
- Automatic capture of caller context using Python's inspect module
- Context variable propagation through async/thread boundaries
- Structured logging with metadata headers
- Integration with existing debug_print_prompt pattern
- Configurable output directory and file naming
- Support for both timestamped and latest-only file modes

.. note::
   This logging is controlled by development.api_calls configuration:

   - save_all: Enable file output to configured api_calls directory
   - latest_only: Use latest.txt filenames vs timestamped files
   - include_stack_trace: Include full stack trace in metadata

.. seealso::
   :func:`~completion.get_chat_completion` : Main chat completion interface
   :func:`~prompts.base.debug_print_prompt` : Similar pattern for prompt debugging
   :func:`set_api_call_context` : Set caller context for async/thread pool calls
"""

import contextvars
import inspect
import json
import textwrap
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from osprey.utils.config import get_agent_dir, get_config_value
from osprey.utils.logger import get_logger

logger = get_logger("osprey.models")

# Context variable for passing caller information across async/thread boundaries
# This automatically propagates through asyncio.to_thread() calls
_api_call_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "_api_call_context", default=None
)


def set_api_call_context(
    function: str,
    module: str,
    class_name: str | None = None,
    line: int = 0,
    extra: dict[str, Any] | None = None
):
    """Set caller context for API call logging across async/thread boundaries.

    Use this function before calling get_chat_completion from within asyncio.to_thread()
    or thread pools to provide accurate caller information in API call logs.

    The context is automatically propagated through asyncio.to_thread() and other
    async boundaries via Python's contextvars mechanism.

    :param function: Name of the calling function
    :type function: str
    :param module: Module name containing the caller
    :type module: str
    :param class_name: Optional class name if called from a method
    :type class_name: str | None
    :param line: Optional line number
    :type line: int
    :param extra: Optional extra metadata (e.g., capability, task_id, operation)
    :type extra: dict[str, Any] | None

    .. note::
       This context is thread-local and automatically propagated by asyncio.
       You don't need to manually clean it up.

    Examples:
        Before calling get_chat_completion in a threaded context::

            from osprey.models.logging import set_api_call_context

            # In ClassificationNode._perform_classification
            set_api_call_context(
                function="_perform_classification",
                module="classification_node",
                class_name="CapabilityClassifier",
                line=387,
                extra={"capability": "python"}
            )

            response = await asyncio.to_thread(
                get_chat_completion,
                message=message,
                model_config=model_config
            )
    """
    context = {
        "function": function,
        "module": module,
        "class": class_name,
        "line_number": line,
        "source": "context_var"
    }

    # Add extra metadata if provided
    if extra:
        context.update(extra)

    _api_call_context.set(context)


def _get_caller_info(skip_frames: int = 2) -> dict[str, Any]:
    """Extract detailed information about the calling function.

    Uses Python's inspect module to walk the call stack and extract
    information about where get_chat_completion was called from.
    Intelligently skips thread pool and other framework internals to
    find the actual business logic caller.

    :param skip_frames: Number of frames to skip (default 2: this func + log_api_call)
    :type skip_frames: int
    :return: Dictionary containing caller metadata
    :rtype: dict[str, Any]

    .. note::
       The function extracts:
       - Function name and full qualified name if available
       - File path (relative to project root if possible)
       - Line number
       - Module name
       - Class name if called from a method
       - Threading context if called from thread pool
    """
    try:
        # First, check if caller context was explicitly set via context variable
        # This handles async/thread pool cases where stack inspection doesn't work
        context = _api_call_context.get()
        if context is not None:
            logger.debug(
                f"API Call Logging: Using context variable for caller: "
                f"{context.get('function')}() in {context.get('module')}"
            )
            return context

        # Fall back to stack inspection for direct calls
        # Get the call stack
        stack = inspect.stack()

        # Skip internal frames to get to the actual caller
        # skip_frames: 0=this function, 1=log_api_call, 2=get_chat_completion, 3=actual caller
        if len(stack) <= skip_frames:
            return {
                "function": "Unknown",
                "filename": "Unknown",
                "line_number": 0,
                "module": "Unknown",
            }

        # Patterns to skip (framework internals that aren't the real caller)
        # These are file path patterns that indicate framework/stdlib machinery
        skip_patterns = [
            "/logging.py",  # This file!
            "/completion.py",  # The get_chat_completion wrapper
            "/concurrent/futures/",  # Thread/process pool executors
            "/threading.py",  # Threading module
            "/asyncio/",  # AsyncIO internals
            "/langgraph/",  # LangGraph execution engine
            "/pydantic_ai/",  # PydanticAI agent wrapper
            "/queue.py",  # Queue module
        ]

        # Start from frame 0 and find the first frame that's meaningful
        caller_frame = None
        threading_context = None
        skipped_frames = []

        for frame_idx, candidate in enumerate(stack):
            candidate_file = candidate.filename.replace("\\", "/")  # Normalize path separators

            # Always skip the first few frames (this function, log_api_call, etc.)
            if frame_idx < 3:  # Skip: _get_caller_info, log_api_call, get_chat_completion
                continue

            # Check if this is internal framework machinery
            is_internal = any(pattern in candidate_file for pattern in skip_patterns)

            if is_internal:
                # Track what we're skipping for debugging
                skipped_frames.append(f"{candidate.function} ({Path(candidate_file).name})")

                # Detect threading context
                if "/concurrent/futures/" in candidate_file or "/threading.py" in candidate_file:
                    threading_context = "ThreadPoolExecutor"
                elif "/asyncio/" in candidate_file:
                    threading_context = "AsyncIO"

                continue  # Keep looking

            # Also skip Python stdlib and site-packages except osprey
            is_stdlib_or_sitepackages = (
                "/lib/python" in candidate_file or "/site-packages/" in candidate_file
            )
            is_osprey_code = "/osprey/" in candidate_file

            if is_stdlib_or_sitepackages and not is_osprey_code:
                skipped_frames.append(f"{candidate.function} ({Path(candidate_file).name})")
                continue  # Keep looking

            # Found it! This is our actual caller
            caller_frame = candidate
            logger.debug(
                f"API Call Logging: Identified caller as {candidate.function}() "
                f"in {Path(candidate_file).name}:{candidate.lineno}"
            )
            break

        # If we still couldn't find a good frame, search backwards for ANY osprey frame
        if caller_frame is None:
            logger.debug(
                "API Call Logging: Forward search failed, attempting backward search for osprey code..."
            )
            for i in range(
                len(stack) - 1, 2, -1
            ):  # Start from end, stop before our internal frames
                candidate_file = stack[i].filename.replace("\\", "/")
                if "/osprey/" in candidate_file and "/logging.py" not in candidate_file:
                    caller_frame = stack[i]
                    threading_context = "Isolated (thread boundary)"
                    logger.debug(
                        f"API Call Logging: Found osprey code in backward search: "
                        f"{stack[i].function}() in {Path(stack[i].filename).name}"
                    )
                    break

        # Last resort fallback
        if caller_frame is None:
            # Temporarily log the full stack to help debug why we can't find the caller
            logger.debug(f"API Call Logging: Full stack trace ({len(stack)} frames):")
            for i, frame in enumerate(stack[:20]):  # Show first 20 frames
                logger.debug(
                    f"  Frame {i}: {frame.function}() @ {Path(frame.filename).name}:{frame.lineno}"
                )

            logger.warning(
                "API Call Logging: Unable to identify caller function (likely due to thread pool isolation). "
                "API call log will show thread machinery instead of business logic. "
                "This is a logging limitation, not an error in your code."
            )
            caller_frame = stack[3] if len(stack) > 3 else stack[-1]
            threading_context = "Unknown context"

        # Extract basic information
        caller_info = {
            "function": caller_frame.function,
            "filename": caller_frame.filename,
            "line_number": caller_frame.lineno,
            "module": inspect.getmodulename(caller_frame.filename) or "Unknown",
        }

        # Add threading/async context if detected
        if threading_context:
            caller_info["threading"] = threading_context

        # Add debug info about skipped frames (helpful for understanding complex call chains)
        if skipped_frames:
            caller_info["skipped_frames"] = skipped_frames[:5]  # Limit to first 5 for brevity

        # Try to make filename relative to project root for readability
        try:
            project_root = get_config_value("project_root", None)
            if project_root:
                caller_info["filename"] = str(Path(caller_frame.filename).relative_to(project_root))
        except (ValueError, TypeError):
            # Keep absolute path if we can't make it relative
            pass

        # Try to get class name if this is a method call
        if "self" in caller_frame.frame.f_locals:
            caller_info["class"] = caller_frame.frame.f_locals["self"].__class__.__name__
        elif "cls" in caller_frame.frame.f_locals:
            caller_info["class"] = caller_frame.frame.f_locals["cls"].__name__

        return caller_info

    except Exception as e:
        logger.debug(f"Error extracting caller info: {e}")
        return {
            "function": "Unknown",
            "filename": "Unknown",
            "line_number": 0,
            "module": "Unknown",
            "error": str(e),
        }


def _format_metadata_header(
    caller_info: dict,
    provider: str,
    model_id: str,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    budget_tokens: int | None,
    output_model: Any,
    include_stack_trace: bool = False,
) -> str:
    """Format a comprehensive metadata header for the log file.

    :param caller_info: Dictionary with caller information from _get_caller_info
    :param provider: AI provider name
    :param model_id: Model identifier
    :param max_tokens: Maximum tokens for response
    :param temperature: Model temperature setting
    :param enable_thinking: Whether extended thinking is enabled
    :param budget_tokens: Thinking budget tokens if applicable
    :param output_model: Structured output model if used
    :param include_stack_trace: Whether to include full stack trace
    :return: Formatted metadata header string
    :rtype: str
    """
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build class context if available
    class_context = ""
    if "class" in caller_info:
        class_context = f"\n# Class: {caller_info['class']}"

    # Build threading context if available
    threading_context = ""
    if "threading" in caller_info:
        threading_context = f"\n# Threading: {caller_info['threading']}"

    # Build capability context if available (for classifier)
    capability_context = ""
    if "capability" in caller_info:
        capability_context = f"\n# Capability: {caller_info['capability']}"

    # Build skipped frames info for debugging complex call chains
    skipped_info = ""
    if include_stack_trace and "skipped_frames" in caller_info:
        skipped_list = "\n# ".join(caller_info["skipped_frames"])
        skipped_info = f"\n# Skipped Frames: \n# {skipped_list}"

    # Build output model info
    output_model_info = "None"
    if output_model is not None:
        output_model_info = getattr(output_model, "__name__", str(output_model))

    header = textwrap.dedent(f"""
        # ==========================================
        # LLM API CALL LOG
        # ==========================================
        # Timestamp: {timestamp_str}
        #
        # CALLER INFORMATION
        # ------------------------------------------
        # Function: {caller_info.get("function", "Unknown")}
        # Module: {caller_info.get("module", "Unknown")}{class_context}
        # File: {caller_info.get("filename", "Unknown")}
        # Line: {caller_info.get("line_number", 0)}{threading_context}{capability_context}{skipped_info}
        #
        # MODEL CONFIGURATION
        # ------------------------------------------
        # Provider: {provider}
        # Model ID: {model_id}
        # Max Tokens: {max_tokens}
        # Temperature: {temperature}
        # Enable Thinking: {enable_thinking}
        # Budget Tokens: {budget_tokens}
        # Output Model: {output_model_info}
        #
        # ==========================================
    """).strip()

    # Add stack trace if requested
    if include_stack_trace:
        stack_trace = "".join(
            traceback.format_stack()[:-2]
        )  # Exclude this function and log_api_call
        header += "\n\n# FULL STACK TRACE\n# ------------------------------------------\n"
        header += "# " + stack_trace.replace("\n", "\n# ")

    return header


def _sanitize_result_for_logging(result: Any) -> str:
    """Convert API result to string format suitable for logging.

    Handles different result types (str, BaseModel, list, dict) and
    converts them to readable string format for file output.

    :param result: Result from get_chat_completion
    :return: String representation of result
    :rtype: str
    """
    if isinstance(result, str):
        return result
    elif hasattr(result, "model_dump"):
        # Pydantic BaseModel - use mode='json' to serialize datetime and other complex types
        return json.dumps(result.model_dump(mode='json'), indent=2)
    elif isinstance(result, (dict, list)):
        # For plain dicts/lists, use default JSON encoder with fallback for non-serializable types
        return json.dumps(result, indent=2, default=str)
    else:
        return str(result)


def log_api_call(
    message: str,
    result: Any,
    provider: str,
    model_id: str,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool = False,
    budget_tokens: int | None = None,
    output_model: Any = None,
) -> None:
    """Log complete LLM API call with input, output, and rich metadata.

    This function captures every interaction with LLM APIs through get_chat_completion,
    saving both the raw input message and the complete response along with context
    about where the call originated. This provides complete transparency and enables
    debugging, auditing, and analysis of LLM usage patterns.

    The function integrates with the development configuration system:
    - Controlled by development.api_calls.save_all flag
    - Respects development.api_calls.latest_only for file naming
    - Uses development.api_calls.include_stack_trace for detailed debugging

    Files are saved to the api_calls directory within _agent_data with structured
    naming based on the calling function and timestamp.

    :param message: Input message/prompt sent to the LLM
    :type message: str
    :param result: Response from the LLM (str, BaseModel, list, or dict)
    :type result: Any
    :param provider: AI provider name
    :type provider: str
    :param model_id: Model identifier
    :type model_id: str
    :param max_tokens: Maximum tokens configured for response
    :type max_tokens: int
    :param temperature: Model temperature setting
    :type temperature: float
    :param enable_thinking: Whether extended thinking was enabled
    :type enable_thinking: bool
    :param budget_tokens: Thinking budget tokens if applicable
    :type budget_tokens: int | None
    :param output_model: Structured output model class if used
    :type output_model: Any | None

    .. note::
       This function is designed to be called from get_chat_completion and
       will silently handle any errors to prevent debugging from breaking
       the main application flow.

    .. seealso::
       :func:`~completion.get_chat_completion` : Function that calls this logger
       :func:`_get_caller_info` : Extracts caller context from stack
       :func:`~prompts.base.debug_print_prompt` : Similar pattern for prompts

    Examples:
        Configuration in config.yml::

            development:
              api_calls:
                save_all: true          # Enable API call logging
                latest_only: false      # Create timestamped files
                include_stack_trace: true  # Add full stack traces
    """
    try:
        # Check if API call logging is enabled
        development_config = get_config_value("development", {})
        api_calls_config = development_config.get("api_calls", {})

        if not api_calls_config.get("save_all", False):
            return

        # Get caller information
        caller_info = _get_caller_info(
            skip_frames=3
        )  # Skip: this func, wrapper in completion.py, get_chat_completion

        # Create api_calls directory
        api_calls_dir = Path(get_agent_dir("api_calls_dir"))
        api_calls_dir.mkdir(exist_ok=True, parents=True)

        # Determine filename
        latest_only = api_calls_config.get("latest_only", True)
        caller_function = caller_info.get("function", "unknown")
        caller_module = caller_info.get("module", "unknown")
        caller_class = caller_info.get("class", None)

        # Build descriptive filename
        if caller_class:
            base_name = f"{caller_module}_{caller_class}_{caller_function}"
        else:
            base_name = f"{caller_module}_{caller_function}"

        # Add extra metadata to filename if provided (e.g., capability)
        if "capability" in caller_info:
            base_name = f"{base_name}_{caller_info['capability']}"

        if latest_only:
            filename = f"{base_name}_latest.txt"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_name}_{timestamp}.txt"

        filepath = api_calls_dir / filename

        # Format metadata header
        include_stack_trace = api_calls_config.get("include_stack_trace", False)
        header = _format_metadata_header(
            caller_info=caller_info,
            provider=provider,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
            budget_tokens=budget_tokens,
            output_model=output_model,
            include_stack_trace=include_stack_trace,
        )

        # Format output
        result_str = _sanitize_result_for_logging(result)

        # Build complete log entry
        log_content = f"{header}\n\n"
        log_content += "=" * 80 + "\n"
        log_content += "INPUT MESSAGE\n"
        log_content += "=" * 80 + "\n\n"
        log_content += message + "\n\n"
        log_content += "=" * 80 + "\n"
        log_content += "OUTPUT RESPONSE\n"
        log_content += "=" * 80 + "\n\n"
        log_content += result_str + "\n"

        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(log_content)

        logger.debug(f"API call logged to {filepath}")

    except Exception as e:
        # Don't break execution if logging fails, but make errors visible
        logger.warning(f"Failed to log API call: {e}")
        logger.debug(f"API logging traceback: {traceback.format_exc()}")
