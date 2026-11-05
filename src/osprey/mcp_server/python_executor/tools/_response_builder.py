"""Shared post-execution response builder for Python executor tools."""

import json
import logging
from typing import Any

import nbformat
from fastmcp.exceptions import ToolError

from osprey.audit.call import current_tool_use_id, note, write_stamps
from osprey.mcp_server.errors import make_error
from osprey.mcp_server.python_executor.executor import (
    FAILURE_KIND_APPROVAL_CONFIG_INVALID,
    FAILURE_KIND_APPROVAL_MISSING,
    FAILURE_KIND_GUARDED_RUN_BUSY,
    FAILURE_KIND_SETUP,
    FAILURE_KIND_SWITCH_IN_PROGRESS,
    FAILURE_KIND_TIMEOUT,
    FAILURE_KIND_UNREVIEWED,
    NOT_STARTED_FAILURE_KINDS,
    ExecutionResult,
)
from osprey.stores.artifact_manifest import read_collected_file

logger = logging.getLogger("osprey.mcp_server.tools.execute")

# Maximum characters of stdout/stderr returned inline in the summary.
_STDOUT_PREVIEW_LIMIT = 500
_STDERR_PREVIEW_LIMIT = 500

#: ``details.subsystem`` on every envelope this builder raises: the one word
#: that says which part of OSPREY the failure belongs to, so the agent can name
#: it instead of guessing between "the user's code" and "the control system".
SUBSYSTEM = "python_executor"

_SETUP_SUGGESTIONS = [
    "The Python execution sandbox could not be started, so none of the submitted "
    "code ran — this is not a fault in that code and rewriting it will not help.",
    "Report that the python_executor service is unavailable. An operator needs to "
    "check its MCP server (interpreter, workspace disk, config.yml execution "
    "settings) and its logs.",
]
_TIMEOUT_SUGGESTIONS = [
    "The sandbox killed the script at its configured timeout. Reduce the work one "
    "call does — fewer channels, a shorter time range — or split it across calls.",
    "Do not raise the timeout in config.yml; that is an operator decision.",
]
_SCRIPT_ERROR_SUGGESTIONS = [
    "The submitted code raised. Read the traceback in stderr, fix the code, and run it again.",
]
_SWITCH_IN_PROGRESS_SUGGESTIONS = [
    "The control target is being switched, so the run was declined before it started.",
    "Re-run the code after the target switch completes.",
]
_APPROVAL_MISSING_SUGGESTIONS = [
    "The approval prompt left no record for this call, so the run was declined before it started.",
    "Call the tool again so the approval prompt can ask.",
]
_APPROVAL_CONFIG_INVALID_SUGGESTIONS = [
    "Nothing ran: the deployment's approval switch is not true or false, and calling "
    "again will not help. Tell the operator to set approval.enabled in profile.yml and rebuild.",
]
_GUARDED_RUN_BUSY_SUGGESTIONS = [
    "Nothing ran: another guarded run holds this control target. Run it again once the "
    "holder named in the message (pid, since) has ended.",
]
_UNREVIEWED_SUGGESTIONS = [
    "Nothing ran: the code review had not read this code. Call the tool again; the review "
    "runs on the next call.",
]


def _exec_id() -> str:
    """The run's identity: the tool call in scope, or ``""`` (nothing links)."""
    return current_tool_use_id() or ""


def _run_status(exec_result: ExecutionResult) -> str:
    """The result head's stored status for *exec_result*."""
    if exec_result.success:
        return "succeeded"
    if exec_result.failure_kind in NOT_STARTED_FAILURE_KINDS:
        return "not_run"
    return "failed"


def _error_message(error: ToolError) -> str:
    """The envelope's ``error_message``, else the error's text."""
    text = str(error)
    try:
        envelope = json.loads(text)
    except ValueError:
        return text
    if isinstance(envelope, dict) and isinstance(envelope.get("error_message"), str):
        return envelope["error_message"]
    return text


def save_refusal_head(
    *,
    tool_source: str,
    code: str,
    description: str,
    execution_mode: str,
    error: ToolError,
    launched: bool,
    extra: dict[str, Any] | None = None,
) -> None:
    """Store the head of a run a tool refused: ``not_run`` before the launch, ``failed`` after.

    The head carries the proposed code and the refusal's message. Best-effort:
    a store failure is logged and never replaces the refusal the caller re-raises.
    """
    status = "failed" if launched else "not_run"
    message = _error_message(error)
    data: dict[str, Any] = {
        "description": description,
        "execution_mode": execution_mode,
        "code": code,
        "error": message,
        "has_errors": True,
        **(extra or {}),
    }
    try:
        from osprey.stores.artifact_store import RUN_RECORD_KEY, get_artifact_store

        get_artifact_store().save_data(
            tool=tool_source,
            data=data,
            title=description or tool_source,
            description=description,
            summary={
                "description": description,
                "status": "Failed" if launched else "Not run",
                "error": message[:_STDERR_PREVIEW_LIMIT],
                "has_errors": True,
            },
            access_details={"execution_mode": execution_mode, "code_lines": len(code.splitlines())},
            metadata={RUN_RECORD_KEY: True, "run_part": "result", "run_status": status},
            exec_id=_exec_id(),
        )
    except Exception:
        logger.debug("Refused-run head save failed (non-fatal)", exc_info=True)


def _note_writes(exec_result: ExecutionResult) -> None:
    """Note who the run wrote as and which channels it attempted. Never raises.

    ``ca_user`` and ``ca_host`` are read in this process: the sandbox is its
    child, running under the same account on the same host, so these are the
    stamps the control system saw. A stamp that cannot be read is left out
    rather than guessed. ``channels`` is the run's write ledger — empty for a
    run that attempted no write.
    """
    note(**write_stamps(), channels=list(exec_result.written_channels))


def _raise_failure(exec_result: ExecutionResult, stderr_text: str, payload: dict) -> None:
    """Raise the error envelope for a failed run, classed by who failed.

    A run the sandbox never started (:data:`FAILURE_KIND_SETUP`) is an
    infrastructure outage and goes out as ``service_unavailable`` — the type
    the error-guidance hook classes as *Connection*, whose protocol is "report
    the service as unavailable", not "help the user fix their code". A run
    that started and failed is ``execution_error`` whether its own code raised
    or the sandbox timed it out; ``details.kind`` tells those two apart. A run
    declined because the control target is mid-switch
    (:data:`FAILURE_KIND_SWITCH_IN_PROGRESS`) is the third ``kind``: nothing
    ran, nothing is broken, and the answer is to run it again once the switch
    lands — which is neither of the other two protocols. A run declined because
    the approval prompt left no record of the call
    (:data:`FAILURE_KIND_APPROVAL_MISSING`) is the fourth: nothing ran, nothing
    is broken, and the answer is to call the tool again so the prompt can ask.
    A run refused because the deployment's approval switch is not true or false
    (:data:`FAILURE_KIND_APPROVAL_CONFIG_INVALID`) has its own ``kind``: nothing
    ran, and only the operator can fix it, in the profile.
    A run declined because another guarded run holds its control target
    (:data:`FAILURE_KIND_GUARDED_RUN_BUSY`) is the fifth: nothing ran, and the
    answer is to run it again once the holder the message names has ended.
    A run refused because its code review was skipped on a decline that did
    not hold at the launch (:data:`FAILURE_KIND_UNREVIEWED`) is the sixth:
    nothing ran, and calling again reviews it.

    ``details`` carries the response the success path would have returned, so
    a failed run still surfaces its artifacts, plus :data:`SUBSYSTEM` and the
    ``kind`` — the same "one shared error_type, identity in details" contract
    the graph server documents on ``GraphStoreError``. A code review note goes
    there too, in every shape and for every ``kind``: the concern was raised
    about code that ran, so the run failing does not settle it.
    """
    if exec_result.code_review is not None:
        payload = {**payload, "code_review": exec_result.code_review}
    if exec_result.failure_kind == FAILURE_KIND_SETUP:
        make_error(
            "service_unavailable",
            exec_result.error_message or "Execution setup failed.",
            _SETUP_SUGGESTIONS,
            details={**payload, "subsystem": SUBSYSTEM, "kind": "setup_failed"},
        )
    if exec_result.failure_kind == FAILURE_KIND_TIMEOUT:
        kind, suggestions = "timeout", _TIMEOUT_SUGGESTIONS
    elif exec_result.failure_kind == FAILURE_KIND_SWITCH_IN_PROGRESS:
        kind, suggestions = FAILURE_KIND_SWITCH_IN_PROGRESS, _SWITCH_IN_PROGRESS_SUGGESTIONS
    elif exec_result.failure_kind == FAILURE_KIND_APPROVAL_MISSING:
        kind, suggestions = FAILURE_KIND_APPROVAL_MISSING, _APPROVAL_MISSING_SUGGESTIONS
    elif exec_result.failure_kind == FAILURE_KIND_APPROVAL_CONFIG_INVALID:
        kind = FAILURE_KIND_APPROVAL_CONFIG_INVALID
        suggestions = _APPROVAL_CONFIG_INVALID_SUGGESTIONS
    elif exec_result.failure_kind == FAILURE_KIND_GUARDED_RUN_BUSY:
        kind, suggestions = FAILURE_KIND_GUARDED_RUN_BUSY, _GUARDED_RUN_BUSY_SUGGESTIONS
    elif exec_result.failure_kind == FAILURE_KIND_UNREVIEWED:
        kind, suggestions = FAILURE_KIND_UNREVIEWED, _UNREVIEWED_SUGGESTIONS
    else:
        kind, suggestions = "script_error", _SCRIPT_ERROR_SUGGESTIONS
    make_error(
        "execution_error",
        (stderr_text or "Execution reported an error.")[:_STDERR_PREVIEW_LIMIT],
        suggestions,
        details={**payload, "subsystem": SUBSYSTEM, "kind": kind},
    )


async def build_execution_response(
    code: str,
    description: str,
    execution_mode: str,
    exec_result: ExecutionResult,
    patterns: dict,
    save_output: bool,
    tool_source: str = "execute",
) -> str:
    """Build the tool's reply from an execution result.

    Handles figure/artifact saving, notebook creation, summary building,
    ArtifactStore persistence, and gallery URL injection. A successful run
    returns the response dict as JSON text, which is the tool's declared
    ``str`` reply: fastmcp passes only its own ``ToolResult`` through and
    serializes any other return value as data against the declared schema.
    When the execution reported errors, raises ``ToolError`` carrying the OSPREY
    error envelope (via :func:`~osprey.mcp_server.errors.make_error`, see
    :func:`_raise_failure` for how the run's ``failure_kind`` picks the type) so
    fastmcp produces a wire-form ``CallToolResult(isError=True)`` (returning a
    CallToolResult with ``isError`` set is silently dropped by fastmcp's
    ``convert_result``).

    Args:
        code: Original source code (for notebook/metadata — not augmented).
        description: Human-readable description of what the code does.
        execution_mode: "readonly" or "readwrite".
        exec_result: Result from ``execute_code()``.
        patterns: Dict from ``detect_control_system_operations()``.
        save_output: If True, reply with the stored result record; otherwise
            return the result inline. The record (the run's head) is stored either way.
        tool_source: Tool identifier for stored metadata ("execute" or "execute_file").

    Returns:
        The execution summary as JSON text.
    """
    _note_writes(exec_result)

    artifact_ids: list[str] = []
    exec_id = _exec_id()

    stdout_text = exec_result.stdout
    stderr_text = exec_result.stderr

    # Auto-save figure artifacts collected by the adapter
    for fig_path in exec_result.figures:
        try:
            from osprey.stores.artifact_store import get_artifact_store

            store = get_artifact_store()
            fig_entry = store.save_file(
                file_content=read_collected_file(fig_path),
                filename=fig_path.name,
                artifact_type="image",
                title=f"Figure: {fig_path.stem}",
                description=f"Figure from: {description}",
                mime_type=f"image/{fig_path.suffix.lstrip('.')}",
                tool_source=tool_source,
                metadata={"run_part": "output", "figure": True},
                exec_id=exec_id,
            )
            artifact_ids.append(fig_entry.id)
        except Exception:
            logger.debug("Figure artifact save failed for %s (non-fatal)", fig_path, exc_info=True)

    # Auto-save artifacts collected from subprocess save_artifact() calls
    for art in exec_result.artifacts:
        try:
            from osprey.stores.artifact_store import get_artifact_store

            store = get_artifact_store()
            art_entry = store.save_file(
                file_content=read_collected_file(art["path"]),
                filename=art["path"].name,
                artifact_type=art["artifact_type"],
                title=art["title"],
                description=art["description"],
                mime_type=art["mime_type"],
                tool_source=tool_source,
                metadata={"run_part": "output"},
                exec_id=exec_id,
            )
            artifact_ids.append(art_entry.id)
        except Exception:
            logger.debug(
                "Subprocess artifact save failed for %s (non-fatal)",
                art.get("title", "unknown"),
                exc_info=True,
            )

    has_errors = not exec_result.success
    full_result = {
        "description": description,
        "execution_mode": execution_mode,
        "execution_method": exec_result.execution_method_used,
        "code": code,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "has_errors": has_errors,
        "detected_patterns": patterns,
        "artifact_ids": artifact_ids,
    }

    # Auto-save a notebook artifact for every execution
    notebook_artifact_id = None
    try:
        from osprey.stores.notebook_renderer import create_notebook_from_code

        nb = create_notebook_from_code(code, description, stdout_text, stderr_text)
        nb_bytes = nbformat.writes(nb).encode()

        from osprey.stores.artifact_store import RUN_RECORD_KEY, get_artifact_store

        store = get_artifact_store()
        nb_entry = store.save_file(
            file_content=nb_bytes,
            filename=f"{description[:40].replace(' ', '_')}.ipynb",
            artifact_type="notebook",
            title=f"Notebook: {description}",
            description=description,
            mime_type="application/x-ipynb+json",
            tool_source=tool_source,
            metadata={RUN_RECORD_KEY: True, "run_part": "code"},
            exec_id=exec_id,
        )
        notebook_artifact_id = nb_entry.id
        artifact_ids.append(notebook_artifact_id)
    except Exception:
        logger.debug("Notebook artifact creation failed (non-fatal)", exc_info=True)

    # Build compact summary inline
    summary: dict[str, Any] = {
        "description": description,
        "status": "Failed" if has_errors else "Success",
        "output": stdout_text[:_STDOUT_PREVIEW_LIMIT],
        "error": stderr_text[:_STDERR_PREVIEW_LIMIT] if stderr_text else None,
        "has_errors": has_errors,
        "detected_patterns": patterns,
    }
    if artifact_ids:
        summary["artifact_ids"] = artifact_ids
        summary["artifact_count"] = len(artifact_ids)
    if exec_result.code_review is not None:
        summary["code_review"] = exec_result.code_review
    access_details = {
        "execution_mode": execution_mode,
        "execution_method": exec_result.execution_method_used,
        "code_lines": len(code.splitlines()),
        "stdout_length": len(stdout_text),
        "stderr_length": len(stderr_text),
    }

    # The result record is the run's head, stored whether or not the reply
    # carries it: it holds the run's status.
    from osprey.stores.artifact_store import RUN_RECORD_KEY, get_artifact_store

    try:
        entry = get_artifact_store().save_data(
            tool=tool_source,
            data=full_result,
            title=description,
            description=description,
            summary=summary,
            access_details=access_details,
            metadata={
                RUN_RECORD_KEY: True,
                "run_part": "result",
                "run_status": _run_status(exec_result),
            },
            exec_id=exec_id,
        )
    except Exception:
        # The saved reply points at the record, so its save failing is the
        # tool's failure; an inline reply does without it.
        if save_output:
            raise
        logger.debug("Result head save failed (non-fatal)", exc_info=True)

    if not save_output:
        # Return inline (the path for callers that opt out of saving)
        result = {
            "description": description,
            "execution_mode": execution_mode,
            "execution_method": exec_result.execution_method_used,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "has_errors": has_errors,
            "detected_patterns": patterns,
        }
        if artifact_ids:
            result["artifact_ids"] = artifact_ids
        if exec_result.code_review is not None:
            result["code_review"] = exec_result.code_review
        if has_errors:
            _raise_failure(exec_result, stderr_text, result)
        return json.dumps(result, default=str)

    response = entry.to_tool_response()
    if artifact_ids:
        response["artifact_ids"] = artifact_ids
        try:
            from osprey.mcp_server.http import gallery_url

            response["gallery_url"] = gallery_url()
        except Exception:
            pass
    if notebook_artifact_id:
        response["notebook_artifact_id"] = notebook_artifact_id
    if has_errors:
        _raise_failure(exec_result, stderr_text, response)
    return json.dumps(response, default=str)
