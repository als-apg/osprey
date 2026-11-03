"""LLM API Call Recording for Transparency and Debugging.

This module records every LLM API interaction made through the
get_chat_completion function as an ``api_call`` entry in the agent record. Each
entry carries the raw input and output along with caller information and the
model parameters.

Key capabilities:
- Automatic capture of caller context using Python's inspect module
- Context variable propagation through async/thread boundaries
- One record entry per call, written through the record writer

.. note::
   This recording is controlled by development.api_calls configuration:

   - save_all: Record every API call
   - include_stack_trace: Include the full stack trace in the entry

   The record's off switch (``OSPREY_AGENT_RECORD_OFF=1``) suppresses it.

.. seealso::
   :func:`~completion.get_chat_completion` : Main chat completion interface
   :func:`set_api_call_context` : Set caller context for async/thread pool calls
"""

import contextvars
import inspect
import json
import traceback
import warnings
from pathlib import Path
from typing import Any

from osprey.agent_record import writer as record_writer
from osprey.agent_record.contract import KIND_API_CALL
from osprey_connectors.config import get_config_value
from osprey_connectors.logger import get_logger

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
    extra: dict[str, Any] | None = None,
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
        "source": "context_var",
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
            "/litellm/",  # LiteLLM adapter
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


def _sanitize_result_for_logging(result: Any) -> str:
    """Convert API result to string format suitable for logging.

    Handles different result types (str, BaseModel, list, dict) and
    converts them to readable string format for the entry's ``output`` field.

    Note: LiteLLM response objects (Message, Choices, StreamingChoices) have model_dump()
    but aren't standard Pydantic models - they have provider-specific fields that may not
    match the expected schema, causing Pydantic serialization warnings.

    :param result: Result from get_chat_completion
    :return: String representation of result
    :rtype: str
    """
    if isinstance(result, str):
        return result
    if hasattr(result, "model_dump"):
        # Pydantic BaseModel or Pydantic-like objects (LiteLLM responses)
        # Suppress warnings for non-standard Pydantic objects like LiteLLM's Message/Choices
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
                return json.dumps(result.model_dump(mode="json"), indent=2)
        except Exception:
            # Fallback for non-standard Pydantic-like objects that fail to serialize
            try:
                return json.dumps(result.__dict__, indent=2, default=str)
            except Exception:
                return str(result)
    elif isinstance(result, (dict, list)):
        # For plain dicts/lists, use default JSON encoder with fallback for non-serializable types
        return json.dumps(result, indent=2, default=str)
    else:
        return str(result)


#: The caller-info fields an ``api_call`` entry carries, in the order they are
#: written. ``skipped_frames`` is added only with ``include_stack_trace``.
_CALLER_FIELDS: tuple[str, ...] = (
    "function",
    "module",
    "class",
    "filename",
    "line_number",
    "threading",
    "capability",
)


def _api_call_body(
    *,
    caller_info: dict[str, Any],
    message: str,
    result: Any,
    provider: str,
    model_id: str,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    budget_tokens: int | None,
    output_model: Any,
    include_stack_trace: bool,
) -> dict[str, Any]:
    """The body of one ``api_call`` entry: who called, how, what went in and out."""
    caller = {key: caller_info[key] for key in _CALLER_FIELDS if key in caller_info}
    if include_stack_trace and "skipped_frames" in caller_info:
        caller["skipped_frames"] = list(caller_info["skipped_frames"])

    body: dict[str, Any] = {
        "caller": caller,
        "provider": provider,
        "model_id": model_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "enable_thinking": enable_thinking,
        "budget_tokens": budget_tokens,
        "output_model": (
            None if output_model is None else getattr(output_model, "__name__", str(output_model))
        ),
        "input": message,
        "output": _sanitize_result_for_logging(result),
    }
    if include_stack_trace:
        # Every frame above this helper and log_api_call.
        body["stack"] = "".join(traceback.format_stack()[:-2])
    return body


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
    """Record one LLM API call as an ``api_call`` entry in the agent record.

    Each call is one entry, written through the record writer: the caller's
    context, the model configuration, the input message and the response. The
    record's own off switch (``OSPREY_AGENT_RECORD_OFF``) is honoured by the
    writer, so a switched-off run records nothing.

    Gated by ``development.api_calls``:

    - ``save_all``: record every call (off by default)
    - ``include_stack_trace``: add the full stack and the skipped frames

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
       Called from get_chat_completion; it never raises, so recording can
       never break the call it records.

    .. seealso::
       :func:`~completion.get_chat_completion` : Function that calls this logger
       :func:`_get_caller_info` : Extracts caller context from stack

    Examples:
        Configuration in config.yml::

            development:
              api_calls:
                save_all: true             # Record every API call
                include_stack_trace: true  # Add full stack traces
    """
    try:
        development_config = get_config_value("development", {})
        api_calls_config = development_config.get("api_calls", {})

        if not api_calls_config.get("save_all", False) or record_writer.record_off():
            return

        # Skip: this func, wrapper in completion.py, get_chat_completion
        caller_info = _get_caller_info(skip_frames=3)

        body = _api_call_body(
            caller_info=caller_info,
            message=message,
            result=result,
            provider=provider,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
            budget_tokens=budget_tokens,
            output_model=output_model,
            include_stack_trace=api_calls_config.get("include_stack_trace", False),
        )
        entry_id = record_writer.write(KIND_API_CALL, body)
        if entry_id is None:
            logger.debug("API call was not recorded (the record writer stored nothing)")
        else:
            logger.debug(f"API call recorded as {entry_id}")

    except Exception as e:
        # Don't break execution if recording fails, but make errors visible
        logger.warning(f"Failed to record API call: {e}")
        logger.debug(f"API call recording traceback: {traceback.format_exc()}")
