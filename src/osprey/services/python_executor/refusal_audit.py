"""Durable audit record for control-system writes refused in a readonly run.

Every readonly refusal — whichever layer catches it — appends one JSON object
to ``var/audit/readonly-refusals.jsonl`` in the deployment repo. That file is
the answer to "did an agent ever try to write while the session was readonly,
and what exactly did it run": the offending source is recorded verbatim
(bounded), because a refusal whose source is not kept is an alert, not an audit
trail.

Why ``var/audit`` and not the server log: the audit zone is durable by
construction. ``osprey build`` re-renders ``build/`` wholesale and never touches
``var/``, and ``osprey reset`` keeps ``var/audit`` unless the operator passes
``--purge-audit``. A refusal recorded there outlives the render that produced
it; one recorded only in a process log does not.

The same zone holds a second log, ``protected-writes.jsonl``, for the framework
writers: the config routes, the ``setup_patch`` tool, the Claude-setup panel and
the scaffold gallery, which refuse edits to the protected set. Those refusals
answer a different question — "did an agent try to rewrite the framework that
constrains it" rather than "did an agent try to move the machine" — so they get
their own file rather than diluting the readonly trail.

**Recording never fails a refusal.** Every function here swallows its own
errors: the refusal itself is enforced by the caller, and an unwritable audit
directory must not turn a blocked write into a traceback that reads like the
gate malfunctioned.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from osprey.utils.logger import get_logger

logger = get_logger("readonly_refusal_audit")

#: Basename of the append-only refusal log inside the audit zone.
REFUSAL_LOG_FILENAME = "readonly-refusals.jsonl"

#: Basename of the append-only log of framework-writer refusals, beside the
#: readonly one in the same audit zone.
PROTECTED_WRITES_LOG_FILENAME = "protected-writes.jsonl"

#: Characters of offending source kept per record. Generous enough to hold a
#: whole ordinary script, bounded so a pathological submission cannot inflate
#: the log without limit. A truncated record says so via ``source_truncated``
#: rather than silently looking like the whole script.
MAX_SOURCE_CHARS = 8000

#: The layer that refused, recorded verbatim in the ``layer`` field. Static
#: layers refuse before the script is launched; ``runtime_guard`` is the one
#: that fires inside the subprocess, so it is the layer that catches the
#: spellings the static ones cannot see.
LAYER_IMPORT_DENYLIST = "import_denylist"
LAYER_PATTERN_DETECTION = "pattern_detection"
LAYER_PATH_POLICY = "path_policy"
LAYER_RUNTIME_GUARD = "runtime_guard"


def audit_dir() -> Path:
    """The deployment's audit zone (``<repo>/var/audit``).

    Anchored with the same resolver the executor uses to pick the subprocess
    ``cwd`` (:func:`osprey.utils.workspace.resolve_project_root`), so the
    refusal record lands in the repo whose script was refused rather than in
    whatever directory the MCP server process happens to run from.

    Imported lazily and kept as its own function so tests have one seam to
    redirect, rather than having to stand up a project root.
    """
    from osprey.utils.workspace import (
        AUDIT_DIR_RELPATH,
        load_osprey_config,
        resolve_project_root,
    )

    return resolve_project_root(load_osprey_config()) / AUDIT_DIR_RELPATH


def refusal_log_path() -> Path:
    """Full path of the append-only refusal log."""
    return audit_dir() / REFUSAL_LOG_FILENAME


def protected_writes_log_path() -> Path:
    """Full path of the append-only protected-write refusal log.

    Derived from :func:`audit_dir` rather than re-spelled, so the one test seam
    that redirects the audit zone redirects both logs.
    """
    return audit_dir() / PROTECTED_WRITES_LOG_FILENAME


def _truncate(source: str) -> tuple[str, bool]:
    """Return *source* bounded to :data:`MAX_SOURCE_CHARS`, and whether it was cut."""
    if len(source) <= MAX_SOURCE_CHARS:
        return source, False
    return source[:MAX_SOURCE_CHARS], True


def build_record(
    *,
    layer: str,
    trigger: Any,
    source: str,
    description: str | None = None,
    tool: str | None = None,
    execution_mode: str = "readonly",
) -> dict[str, Any]:
    """Assemble one refusal record without writing it.

    Split from :func:`record_refusal` so the record's shape can be asserted on
    directly, and so a caller that wants to log the same facts somewhere else
    does not have to re-derive them.

    Args:
        layer: Which layer refused — one of the ``LAYER_*`` constants.
        trigger: What matched. A list of import issues, the detected-pattern
            mapping, or the runtime refusal message; stored as given.
        source: The offending code, truncated to :data:`MAX_SOURCE_CHARS`.
        description: The description the agent supplied for the script.
        tool: MCP tool that was called (``execute`` / ``execute_file``).
        execution_mode: Mode the run was submitted under.
    """
    body, truncated = _truncate(source)
    record: dict[str, Any] = {
        "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "layer": layer,
        "mode": execution_mode,
        "trigger": trigger,
        "source": body,
    }
    if truncated:
        record["source_truncated"] = True
    if tool:
        record["tool"] = tool
    if description:
        record["description"] = description
    return record


def record_refusal(
    *,
    layer: str,
    trigger: Any,
    source: str,
    description: str | None = None,
    tool: str | None = None,
    execution_mode: str = "readonly",
) -> Path | None:
    """Append one refusal record to the audit log and warn on the server log.

    Returns the path written, or ``None`` when the record could not be stored —
    the caller does not branch on it, but a test can tell "wrote" from "gave up
    quietly" without reading the directory.

    Never raises: see the module docstring.
    """
    record = build_record(
        layer=layer,
        trigger=trigger,
        source=source,
        description=description,
        tool=tool,
        execution_mode=execution_mode,
    )

    # The server log carries the event even when the durable write fails, so a
    # read-only filesystem downgrades the audit trail instead of erasing it.
    logger.warning(
        "Refused a control-system write in %s mode (layer=%s, tool=%s): %s",
        record["mode"],
        record["layer"],
        record.get("tool", "-"),
        json.dumps(record["trigger"]),
    )

    try:
        path = refusal_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        return path
    except Exception:
        logger.warning("Could not append to the refusal audit log", exc_info=True)
        return None


def record_protected_refusal(
    *,
    surface: str,
    target_file: str,
    key_or_path: str,
    channel: str,
    reason: str,
) -> Path | None:
    """Append one framework-writer refusal to ``protected-writes.jsonl``.

    Called by every writer that guards the protected set, so a refused edit
    leaves a trace an operator can find later instead of only a 403 the agent
    saw and nobody else did.

    Args:
        surface: Which writer refused — ``setup_patch``, ``http_config``,
            ``claude_setup``, ``scaffold_gallery`` or ``scaffold_restore``.
        target_file: The file the write was aimed at, as the caller names it.
        key_or_path: What inside it was protected — a dotted config key, or the
            project-relative path when the whole file is the target.
        channel: The channel that owns the target, named the same way the
            refusal message names it, so the log and the message agree.
        reason: Short machine-ish reason (``protected_key``, ``reserved path``).

    Returns the path written, or ``None`` when the record could not be stored.

    Never raises: see the module docstring.
    """
    record = {
        "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "surface": surface,
        "target_file": target_file,
        "key_or_path": key_or_path,
        "channel": channel,
        "reason": reason,
    }

    # Logged before the durable write, for the same reason record_refusal does
    # it: an unwritable filesystem should degrade the audit trail, not erase it.
    logger.warning(
        "Refused a protected write (surface=%s, target=%s, key_or_path=%s, channel=%s): %s",
        surface,
        target_file,
        key_or_path,
        channel,
        reason,
    )

    try:
        path = protected_writes_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        return path
    except Exception:
        logger.debug("Could not append to the protected-write audit log", exc_info=True)
        return None
