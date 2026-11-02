#!/usr/bin/env python3
"""
---
name: Code Review
description: Has a reviewer read the code cell an agent is about to write with NotebookEdit into the agent-data artifacts or notebooks trees, and denies it when the configured level says so
summary: Reviews agent-written notebook code cells before they land, under the code_review config section
event: PreToolUse
tools: NotebookEdit
safety_layer: 4
wiring: standalone
timeout: 90
---

## Flow

```
stdin ──► Parse JSON
              │
              ▼
         NotebookEdit into agent-data ──NO──► EXIT (no opinion)
         artifacts/ or notebooks/ ?
              │
             YES
              ▼
         osprey importable? ──NO──► DENY (review=failure kind=unavailable)
              │
             YES
              ▼
         code_review section usable? ──NO──► DENY (review=failure kind=config)
              │
             YES
              ▼
         enabled? ──NO──► EXIT (off: no record, no notice)
              │
             YES
              ▼
         Writes a code cell? ──NO──► EXIT (nothing to review)
              │
             YES
              ▼
         Run the reviewer, capped by --deadline
              │
              ▼
         Apply the level ──► Record ──► Notice (not on a pass)
              │
              ├── refused ──────────────► DENY with the reasons
              ├── proceeds with a note ─► additionalContext
              └── plain pass ───────────► EXIT
```

## Details

The notebook surface of the code review. The core in ``osprey.services.code_review`` builds the
request, runs the reviewer and words every result; this file adapts that to a ``PreToolUse`` hook.

**Config.** The section is read uncached from the file ``locate_osprey_config`` names, never from the
working directory, so a level change reaches the next edit without a rebuild. ``enabled: false`` is
off whatever else the section says. Unusable YAML, a non-bool ``enabled`` or a malformed enabled
section denies with no level, because no level could be read.

**Failure.** An exception once the settings are read is a ``runner`` failure under the configured
level. Before that it is a level-less ``unavailable`` denial, which also covers an interpreter that
cannot import ``osprey``. That denial holds even when the feature is off, because this hook cannot
read the setting without the core. A crash this hook cannot catch lets the write land; the proposal
accepts that remainder at this surface.

**Records.** Every reviewed edit writes one value-free audit record, passes included, so the record
shows that a review happened. A notice goes to the operator's activity strip for everything but a
pass, and its failure costs only the notice.

Module level is the standard library and the logging sibling only, as for every hook. The agent-data
trees come from ``osprey_hook_log.agent_data_subdirs``, the resolver the memory guard and the
notebook-update hook use, so the three agree on them. The memory guard itself is not imported: a
catalog row another row requires is a helper, and a profile may not select a helper.

The web terminal's port is written out for the reason ``osprey_notebook_update.py`` states: this
file may not import ``osprey.port_layout`` at module level, nor ``osprey.mcp_server.http``, whose
notice helper the POST below restates.
"""

import asyncio
import contextlib
import dataclasses
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osprey_hook_log import (
    AUDIT_DECISION_REFUSED,
    agent_data_subdirs,
    emit_audit,
    get_hook_input,
    get_repo_root,
    locate_osprey_config,
    log_hook,
)

_HOOK_NAME = "code-review"
_TOOL = "NotebookEdit"

#: Where the tool names its notebook. The same keys and order as the memory guard, so both judge
#: the same path.
_PATH_KEYS = ("notebook_path", "file_path")

#: The agent-data subdirectories reviewed.
_TREES = ("artifacts", "notebooks")

#: Seconds the harness allows when the render passes no ``--deadline``: the ``timeout`` above.
_DEFAULT_DEADLINE_S = 90.0

_DEADLINE_FLAG = "--deadline"

#: Spelled out for the denial that has no core to ask: pinned against the core by the tests.
_AUDIT_REASON = "code_review"
_UNAVAILABLE_DETAIL = "review=failure kind=unavailable"

_ACTIVITY_KIND = "review"
#: Seconds a notice may spend, and the longest ``detail`` the activity route accepts.
_ACTIVITY_TIMEOUT = 1
_ACTIVITY_DETAIL_LIMIT = 1024

#: The web terminal's port when ``OSPREY_WEB_PORT`` names none; the module docstring says why.
_DEFAULT_WEB_PORT = "10100"  # osprey:not-a-port — stdlib-only hook contract (see module docstring); equals default_port('web', 0)


def _activity_request(tool, kind, detail):
    """Build the ``POST /api/agent-activity`` request for one notice."""
    if len(detail) > _ACTIVITY_DETAIL_LIMIT:
        detail = detail[: _ACTIVITY_DETAIL_LIMIT - 1] + "…"
    payload = json.dumps({"tool": tool, "target": {"kind": kind, "detail": detail}}).encode()

    headers = {"Content-Type": "application/json"}
    token = (os.environ.get("OSPREY_PANEL_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    port = os.environ.get("OSPREY_WEB_PORT", _DEFAULT_WEB_PORT)
    return urllib.request.Request(
        f"http://127.0.0.1:{port}/api/agent-activity",
        data=payload,
        headers=headers,
        method="POST",
    )


def notify_activity(tool, kind, detail):
    """Post one notice to the operator's activity strip; a failure costs only the notice."""
    with contextlib.suppress(Exception):
        urllib.request.urlopen(
            _activity_request(tool, kind, detail), timeout=_ACTIVITY_TIMEOUT
        ).close()


def _parse_deadline(argv) -> float:
    """The ``--deadline N`` seconds in *argv*; absent, malformed or non-positive is the default."""
    for index, argument in enumerate(argv[:-1]):
        if argument == _DEADLINE_FLAG:
            try:
                deadline = float(argv[index + 1])
            except ValueError:
                return _DEFAULT_DEADLINE_S
            return deadline if deadline > 0 else _DEFAULT_DEADLINE_S
    return _DEFAULT_DEADLINE_S


def _target_path(tool_input) -> str:
    """The path the tool call names, or an empty string."""
    if not isinstance(tool_input, dict):
        return ""
    for key in _PATH_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _locate(target, hook_input):
    """Find *target* in the reviewed trees.

    Args:
        target: The path the tool call named.
        hook_input: The parsed hook payload, used to locate the project.

    Returns:
        ``(resolved file, audit subject)``, or ``None`` for a path outside both trees. A path the
        memory guard refuses for its traversal is outside them here too.
    """
    if ".." in target:
        return None
    try:
        resolved = Path(target).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    for tree in _TREES:
        for root in agent_data_subdirs(hook_input, tree):
            if resolved != root and resolved.is_relative_to(root):
                return resolved, f"{tree}/{resolved.relative_to(root).as_posix()}"
    return None


def _deny(reason) -> dict:
    """Build the PreToolUse deny payload carrying *reason*."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _note(message) -> dict:
    """Build the PreToolUse payload that lets the edit go on and tells the agent *message*."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": message,
        }
    }


def _refuse_unavailable(hook_input, subject, message) -> dict:
    """Deny with no level: the core is missing or broke before the settings were read.

    The wording matches what the core gives this failure, restated because the core may be what
    could not be imported.
    """
    log_hook(_HOOK_NAME, hook_input, status="deny", detail=_UNAVAILABLE_DETAIL)
    emit_audit(
        _HOOK_NAME,
        hook_input,
        decision=AUDIT_DECISION_REFUSED,
        subject=subject,
        reason=_AUDIT_REASON,
        detail=_UNAVAILABLE_DETAIL,
    )
    notify_activity(
        _TOOL,
        _ACTIVITY_KIND,
        f"Code review on notebook: failure (unavailable), refused: {subject}",
    )
    return _deny(
        f"The code review failed (kind unavailable), so the code was refused: {message}\n\n"
        "Changing the code will not help: report it and do not retry."
    )


def _conclude(hook_input, outcome, request):
    """Record an outcome, send its notice, and return what the agent is told."""
    from osprey.services.code_review.review import (
        ACTION_REFUSE,
        agent_message,
        audit_fields,
        notice_text,
    )

    fields = audit_fields(outcome, request)
    log_hook(_HOOK_NAME, hook_input, status=fields.decision, detail=fields.detail)
    emit_audit(
        _HOOK_NAME,
        hook_input,
        decision=fields.decision,
        subject=fields.subject,
        reason=fields.reason,
        detail=fields.detail,
    )
    notice = notice_text(outcome, request)
    if notice:
        notify_activity(_TOOL, _ACTIVITY_KIND, notice)

    message = agent_message(outcome)
    if outcome.action == ACTION_REFUSE:
        return _deny(message)
    return _note(message) if message else None


def _conclude_failure(hook_input, subject, kind, message, level):
    """Conclude a failure that has no request of its own to answer."""
    from osprey.services.code_review.contract import (
        SURFACE_NOTEBOOK,
        ReviewFailure,
        ReviewRequest,
    )
    from osprey.services.code_review.review import decide

    outcome = decide(ReviewFailure(kind, message), level)
    return _conclude(hook_input, outcome, ReviewRequest(SURFACE_NOTEBOOK, "", subject))


def _review(hook_input, tool_input, notebook_file, subject):
    """Read the settings and, when the feature is on, review the edit.

    The reviewer's own modules load only once the feature is known to be on, so an edit under a
    feature that is off does not pay for them.
    """
    from osprey.services.code_review.settings import (
        HOOK_OVERHEAD_S,
        ReviewConfigError,
        load_review_settings,
        read_review_config,
    )

    try:
        settings = load_review_settings(config=read_review_config(locate_osprey_config(hook_input)))
    except ReviewConfigError as exc:
        return _conclude_failure(hook_input, subject, "config", str(exc), None)
    if not settings.enabled:
        return None

    from osprey.services.code_review.notebook import notebook_review_request
    from osprey.services.code_review.review import review_code

    try:
        request = notebook_review_request(
            tool_input,
            notebook_file=notebook_file,
            subject=subject,
        )
        if request is None:
            return None
        capped = dataclasses.replace(
            settings,
            time_limit=min(settings.time_limit, _parse_deadline(sys.argv) - HOOK_OVERHEAD_S),
        )
        outcome = asyncio.run(
            review_code(
                request,
                capped,
                repo_root=Path(get_repo_root(hook_input)),
                render_dir=locate_osprey_config(hook_input).parent,
            )
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return _conclude_failure(hook_input, subject, "runner", message, settings.level)
    return _conclude(hook_input, outcome, request)


def _decide(hook_input):
    """Everything past the tool check: the payload to print, or ``None`` for no opinion."""
    tool_input = hook_input.get("tool_input", {})
    target = _target_path(tool_input)
    if not target:
        return None

    subject = _TOOL
    try:
        located = _locate(target, hook_input)
        if located is None:
            return None
        notebook_file, subject = located
        return _review(hook_input, tool_input, notebook_file, subject)
    except Exception as exc:
        return _refuse_unavailable(hook_input, subject, f"{type(exc).__name__}: {exc}")


def main():
    hook_input = get_hook_input()
    if not hook_input or hook_input.get("tool_name") != _TOOL:
        sys.exit(0)

    output = _decide(hook_input)
    if output is not None:
        json.dump(output, sys.stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()
