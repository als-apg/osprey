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
