#!/usr/bin/env python3
"""
---
name: Human Approval Gate
description: Requires human approval for dangerous operations based on per-tool policy
summary: Requires human approval for dangerous operations
event: PreToolUse
tools: channel_write, execute, setup_patch, entry_create, queue_add, queue_start, queue_stop, stop_run
safety_layer: 2
---

## Flow

```
stdin ──► Parse JSON
              │
              ▼
         Is OSPREY tool?  ──NO──► EXIT (allow)
              │
             YES
              │
              ▼
         Load config.yml
         approval section
              │
              ▼
  enabled: false?  ──YES──► EXIT (allow)
     │
    NO
     │
     ▼
  Look up policy for tool
  (fallback: default_policy)
     │
     ├── skip ──────► EXIT (allow)
     ├── selective ──► Content analysis (execute: write patterns)
     └── always ────► ASK (with tool details)
```

## Details

Per-tool policies from `approval.tools` in config.yml:
- **skip**: tool allowed without prompt
- **always**: every call requires approval
- **selective**: content-aware (execute checks write patterns + exec mode)
- **enabled: false**: global toggle disables all approval
- **default_policy**: fail-closed default for unmapped tools

Creates a pre-execution notebook artifact for code review whenever
`execute` requires approval (write mode, write patterns, or always policy).
"""

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osprey_hook_log import (
    get_hook_input,
    load_hook_config,
    load_osprey_config,
    log_hook,
)

# Fallback write patterns: used when osprey is not importable (e.g., standalone hook).
# Must stay in sync with get_framework_standard_patterns()["write"] (19 patterns).
# The parity test in test_approval_hook.py enforces this.
_FALLBACK_WRITE_PATTERNS = [
    # osprey.runtime unified API
    r"\bwrite_channel\s*\(",
    r"\bwrite_channels\s*\(",
    # EPICS (PyEPICS)
    r"\bcaput\s*\(",
    r"epics\.caput\(",
    r"\.put\s*\(",
    r"\.set_value\s*\(",
    r"PV\([^)]*\)\.put",
    r"epics\.PV\([^)]*\)\.put",
    # PVAccess (p4p) - anchored to p4p; a bare r"\.post\s*\(" would flag
    # every requests.post() in ordinary analysis code
    r"\bp4p\b[\s\S]*?\.put\s*\(",
    r"\bp4p\b[\s\S]*?\.post\s*\(",
    r"\.rpc\s*\(",
    r"\bSharedPV\b",
    # Tango (PyTango)
    r"DeviceProxy\([^)]*\)\.write_attribute\(",
    r"\.write_attribute\s*\(",
    r"\.write_attribute_asynch\s*\(",
    r"tango\.DeviceProxy\([^)]*\)\.write",
    # LabVIEW
    r"labview\.set_control\(",
    r"\.SetControlValue\(",
    # Direct connector access
    r"connector\.write_channel\(",
]

# Pattern detection: prefer framework module (regex-based, config-driven, 19 patterns)
# with graceful fallback to regex matching against _FALLBACK_WRITE_PATTERNS
try:
    from osprey.services.python_executor.analysis.pattern_detection import (
        detect_control_system_operations,
    )

    def has_write_patterns(code: str, config: dict | None = None) -> bool:
        """Check if code contains control system write patterns (framework detection)."""
        patterns = None
        pattern_mode = None
        if config:
            pat_config = config.get("control_system", {}).get("patterns")
            if pat_config:
                patterns = pat_config
                pattern_mode = pat_config.get("mode")
        return detect_control_system_operations(code, patterns=patterns, pattern_mode=pattern_mode)[
            "has_writes"
        ]

except ImportError:

    def has_write_patterns(code: str, config: dict | None = None) -> bool:  # type: ignore[misc]
        """Check if code contains control system write patterns (fallback)."""
        patterns = list(_FALLBACK_WRITE_PATTERNS)
        if config:
            pat_config = config.get("control_system", {}).get("patterns", {})
            custom = pat_config.get("write")
            mode = pat_config.get("mode", "extend")
            if custom:
                if mode == "override":
                    patterns = list(custom)
                else:
                    patterns.extend(p for p in custom if p not in patterns)
        return any(re.search(p, code) for p in patterns)


def build_approval_output(reason_detail: str) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "ask",
            "permissionDecisionReason": (
                "\u26a0\ufe0f  OSPREY APPROVAL REQUIRED\n\n"
                f"{reason_detail}\n\n"
                "Review the operation above and approve to proceed."
            ),
        }
    }


def build_allow_output() -> dict:
    """Explicit allow decision — overrides static permission lists.

    In headless dispatch runs (``OSPREY_DISPATCH_RUN=1``, set by the dispatch
    worker) this returns ``{}`` — no decision — instead. An explicit allow
    would override the dispatch worker's per-trigger allowlist hook (CLI
    hook aggregation is not deny-dominates), silently widening a sandboxed
    run's tool surface. With no decision emitted, the call falls through to
    the permission flow, where the worker's own callback rules. Ask and deny
    outputs are unaffected.
    """
    if os.environ.get("OSPREY_DISPATCH_RUN") == "1":
        return {}
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
        }
    }


def _focus_artifact(gallery_base_url: str, artifact_id: str) -> None:
    """Fire-and-forget POST to bring an artifact into focus in the gallery.

    Non-fatal: silently swallows errors so focus failures never block approval.
    """
    try:
        import urllib.request

        data = json.dumps({"artifact_id": artifact_id}).encode()
        req = urllib.request.Request(
            f"{gallery_base_url}/api/focus",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass


# Default Bluesky bridge base URL — mirrors
# osprey.bluesky_bridge_connection.resolve_bridge_url's fallback
# (`DEFAULT_BRIDGE_URL`) without importing that module: this hook runs
# standalone, in a different process/venv from OSPREY's own package.
_DEFAULT_BRIDGE_URL = "http://127.0.0.1:8090"


def _resolve_bridge_url(config: dict) -> str:
    """Resolve the Bluesky bridge base URL: env wins outright over config.yml.

    Resolution order mirrors `osprey.bluesky_bridge_connection.resolve_bridge_url`
    exactly: ``BLUESKY_BRIDGE_URL`` env var, then ``bluesky.bridge_url`` in
    config.yml, then the loopback default above.
    """
    full = os.environ.get("BLUESKY_BRIDGE_URL")
    if full:
        return full.rstrip("/")
    url = config.get("bluesky", {}).get("bridge_url", _DEFAULT_BRIDGE_URL)
    return str(url).rstrip("/")


def _bridge_get_json(base_url: str, path: str, timeout: float = 3.0):
    """GET ``path`` off the bridge and return parsed JSON, or `None` on ANY failure.

    Fail-open (same pattern as `_focus_artifact` below): an unreachable
    bridge, a 404, a network hiccup, or a malformed response must never block
    the approval prompt — every caller here treats `None` as "render
    less detail", never as a reason to raise.
    """
    try:
        import urllib.request

        req = urllib.request.Request(f"{base_url}{path}", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except Exception:
        return None


def _bridge_post_json(base_url: str, path: str, body, timeout: float):
    """POST *body* as JSON to ``path`` and return the parsed answer, or `None`.

    The write-shaped sibling of `_bridge_get_json`, and fail-open in exactly
    the same way: every failure — unreachable bridge, non-2xx, unparseable
    answer, a body that will not serialize — is `None`, which every caller
    renders as "unavailable" rather than raising.
    """
    try:
        import urllib.request

        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except Exception:
        return None


# Unicode code points that render as a line/paragraph break in some terminals
# in addition to the C0 control range (< 0x20) and DEL (0x7f) escaped below.
_UNICODE_LINE_BREAKS = ("\x85", "\u2028", "\u2029")


def _sanitize_label(text) -> str:
    """Escape control characters in an agent-influenced single-line label.

    `plan_name` (and every channel name and setpoint the pre-flight relays
    beside it) originate from a plan file's ``PLAN_METADATA`` and from the
    parameters an agent staged — unconstrained strings for a session-tier,
    agent-authored plan — and reach this prompt RAW: the
    bridge's `PATCH /draft` gates `plan_name` only by registry membership, never
    by character content, and `PlanMetadata.name` carries no character
    constraint. An embedded newline would otherwise forge a fake enrichment line
    (e.g. a spoofed "Validation status: PASSED") on the human approval prompt.
    Escaping the C0 range, DEL, and the unicode line breaks to a visible
    ``\\xNN`` token keeps every label on one line and makes any tampering legible
    to the approver rather than silently dropped. The `Plan source` block is
    deliberately NOT run through this — it is rendered verbatim (and clearly
    delimited) so the approver sees the real, possibly multi-line, plan body.
    """
    return "".join(
        ch
        if (ch >= " " and ch != "\x7f" and ch not in _UNICODE_LINE_BREAKS)
        else f"\\x{ord(ch):02x}"
        for ch in str(text)
    )


def _revision_match_line(pinned, current_revision) -> str:
    """One line stating whether the live draft still matches the pinned revision.

    `queue_add` pins the `draft_revision` the agent staged; the human
    approving the enqueue must see whether the shared draft has moved on since
    then. A match is stated plainly; any mismatch (including a missing pin)
    renders LOUD and warns that what follows is the *current* draft, not
    necessarily what the agent saw.
    """
    if pinned is not None and current_revision == pinned:
        return f"Draft revision {current_revision} — matches pinned revision {pinned}."
    return (
        f"⚠️  DRAFT CHANGED since the agent pinned revision {pinned} "
        f"— the shared draft is now at revision {current_revision}. What follows "
        f"is the CURRENT draft, not necessarily what the agent staged."
    )


# ---------------------------------------------------------------------------
# Pre-flight: what a launch would actually move, before the human decides
# ---------------------------------------------------------------------------
# `POST /plans/{name}/preview` walks the plan without a RunEngine consuming it
# and answers with the channels the plan declares plus the setpoints it would
# drive them to. It is TOTAL — always HTTP 200, always the same keys — so this
# hook branches on `ok` alone and renders either the trajectory or the reason
# there is none. Nothing here may block, delay unboundedly, or fail an approval
# prompt: an approver who cannot see the trajectory still has to be able to
# decide.

# How many moves from each end of the trajectory the prompt names individually.
# The middle is elided with its own count, so the prompt states the exact total
# regardless. Five and five keeps a grid scan's trajectory to eleven lines,
# short enough that the warning lines above it stay on screen.
_TRAJECTORY_EDGE_MOVES = 5

# One pre-flight fetch's ceiling. Set just above the bridge's own budget for
# the call (~10 s of polling plus at most one in-flight RPC) so a slow
# pre-flight comes back as the bridge's own `preview_timed_out` — a reason word
# the approver can read — rather than as this client giving up with none.
_PREVIEW_TIMEOUT_S = 15.0

# Every pre-flight fetch behind ONE prompt, together. `queue_start` previews
# several items, so each fetch's timeout is clamped to what is left of this
# budget: the approver waits at most this long for trajectories, whatever the
# queue holds and however the bridge misbehaves.
_PREVIEW_BUDGET_S = 20.0

# How many of a start's pending items get a trajectory. A start drains the
# whole queue, but previewing every item would cost one bridge round trip each;
# the items that run first are the ones an approver can still act on.
_MAX_PREVIEWED_QUEUE_ITEMS = 3

# The roles the pre-flight labels channels with, in capability terms. An
# unrecognised role is rendered with its own word rather than dropped — the
# bridge's reading of the plan's declaration is what this prompt reports.
_ROLE_HEADLINES = {
    "movable": "Channels this launch would move",
    "readable": "Channels this launch would read",
}


def _fetch_preview(base_url: str, plan_name, plan_args, timeout: float):
    """The pre-flight summary for *plan_name* with *plan_args*, or `None`.

    The parameters go on the wire exactly as the queue item carries them (one
    JSON object), because that is what the route previews. The name is
    percent-encoded: it can be an agent-authored string, and a space or a slash
    in it must land as an unknown plan, never as a malformed request line.
    """
    import urllib.parse

    quoted = urllib.parse.quote(str(plan_name), safe="", errors="replace")
    # Relayed as-is, even if not a dict: a non-object body makes the route
    # answer reason="plan_error" -> rendered "unavailable", which is the
    # honest outcome. Coercing a non-dict to {} here would instead render a
    # trajectory computed with NO parameters as though it were the staged one.
    return _bridge_post_json(base_url, f"/plans/{quoted}/preview", plan_args, timeout)


def _declared_channel_lines(preview) -> list[str]:
    """The channels the launch declares, grouped by role, rendered verbatim.

    The grouping is the bridge's own reading of the plan's role declaration
    against these parameters — this hook does no schema walking and guesses
    nothing from a name. Groups render in whichever order the bridge sent
    them (the route contract puts movable first; this function preserves
    that order rather than establishing it).
    """
    if not isinstance(preview, dict):
        return []

    entries = preview.get("channels")
    entries = entries if isinstance(entries, list) else []

    grouped: dict[str, list[str]] = {}
    for entry in entries:
        if isinstance(entry, dict):
            grouped.setdefault(str(entry.get("role")), []).append(
                _sanitize_label(entry.get("channel"))
            )
    if not grouped:
        return ["Channels: none declared for these parameters."]

    return [
        f"{_ROLE_HEADLINES.get(role, f'Channels declared {_sanitize_label(role)}')}: "
        f"{', '.join(names)}"
        for role, names in grouped.items()
    ]


def _move_line(index: int, move) -> str:
    """One ``N. channel → target`` line, every part escaped to stay on it."""
    if not isinstance(move, dict):
        return f"  {index}. {_sanitize_label(move)}"
    return (
        f"  {index}. {_sanitize_label(move.get('channel'))} → {_sanitize_label(move.get('target'))}"
    )


def _bounded_move_lines(moves: list) -> list[str]:
    """The first and last `_TRAJECTORY_EDGE_MOVES` moves, with the middle counted.

    Numbering is the move's real position in the list the pre-flight returned,
    so the resumed numbering after the elision states how much was skipped a
    second way.
    """
    edge = _TRAJECTORY_EDGE_MOVES
    if len(moves) <= 2 * edge:
        return [_move_line(index, move) for index, move in enumerate(moves, start=1)]

    lines = [_move_line(index, move) for index, move in enumerate(moves[:edge], start=1)]
    hidden = len(moves) - 2 * edge
    noun = "move" if hidden == 1 else "moves"
    lines.append(f"  … {hidden} {noun} not shown …")
    first_tail = len(moves) - edge + 1
    lines.extend(_move_line(first_tail + offset, move) for offset, move in enumerate(moves[-edge:]))
    return lines


def _trajectory_lines(preview) -> list[str]:
    """The setpoint trajectory, or a plain statement that there is none to show.

    Unavailable is a rendering, never a refusal: the prompt still asks, the
    human still decides, and the reason word the pre-flight gave (if it gave
    one) is named so an operator can tell a denied pre-flight from a plan that
    does not build. What this line never does is imply anything about the
    launch itself — an unavailable trajectory is a gap in the evidence, not a
    verdict on the plan.
    """
    if not isinstance(preview, dict):
        return [
            "Setpoint trajectory: unavailable — the pre-flight could not be reached. "
            "Approval is not blocked; this prompt simply cannot show what the launch "
            "would move."
        ]

    if not preview.get("ok"):
        reason = preview.get("reason")
        named = f" (reason: {_sanitize_label(reason)})" if reason else ""
        lines = [
            f"Setpoint trajectory: unavailable{named} — approval is not blocked; this "
            f"prompt cannot show what the launch would move."
        ]
        detail = preview.get("detail")
        if detail:
            lines.append(f"  Pre-flight reported: {_sanitize_label(detail)}")
        return lines

    moves = preview.get("moves")
    moves = moves if isinstance(moves, list) else []
    total = preview.get("total_moves")
    if not total:
        return ["Setpoint trajectory: no moves — this launch would read only."]

    if preview.get("truncated"):
        headline = (
            f"Setpoint trajectory — {_sanitize_label(total)} moves in total, of which the "
            f"pre-flight captured the first {len(moves)} (truncated): the last move below "
            f"is NOT the last move of the launch."
        )
    else:
        headline = f"Setpoint trajectory — {_sanitize_label(total)} moves in total:"
    return [headline] + _bounded_move_lines(moves)


def _preview_lines(preview) -> list[str]:
    """The whole pre-flight block: declared channels, then the trajectory."""
    return _declared_channel_lines(preview) + _trajectory_lines(preview)


def _describe_plan_provenance(base_url: str, plan_name: str) -> list[str]:
    """Render a plan's authoring metadata, provenance/trust tier, validation, and source.

    This is the human backstop for the plan-validator's documented, accepted
    residual (a `getattr`/string-concat obfuscated body that passes the
    sandbox's AST import/pattern scan — see `plan_validation.py`'s module
    docstring): an approver who can actually SEE the plan's source has a
    chance to refuse it even where the earlier automated stages could not.
    Every bridge call goes through `_bridge_get_json`, so any fetch/parse
    failure just yields a shorter line list — never an exception.
    """
    lines: list[str] = []

    plans = _bridge_get_json(base_url, "/plans") or []
    plan_entry = next(
        (p for p in plans if isinstance(p, dict) and p.get("name") == plan_name), None
    )
    metadata = (plan_entry or {}).get("metadata")
    if metadata:
        # `writes` is the whole of the authoring declaration this prompt reads.
        # Which channels a launch touches is NOT authored metadata — it is read
        # off the plan's role-typed parameter fields, and reaches this prompt
        # through the pre-flight (see `_declared_channel_lines`), for the exact
        # parameters staged rather than as a plan-wide claim.
        lines.append(
            "Hazard: writes to hardware"
            if metadata.get("writes")
            else "Hazard: read-only (no hardware writes declared)"
        )
    else:
        lines.append("Hazard: unavailable (no authoring metadata — built-in plan)")

    source_info = _bridge_get_json(base_url, f"/plans/{plan_name}/source")
    provenance = (source_info or {}).get("provenance") or (plan_entry or {}).get("provenance")
    if provenance in ("session", "unreviewed"):
        lines.append(f"Provenance: {provenance.upper()} — AGENT-AUTHORED, NOT REVIEWED BY A HUMAN")
    elif provenance:
        lines.append(f"Provenance: {provenance} (operator-supplied)")
    else:
        lines.append("Provenance: unknown")

    if source_info is None:
        lines.append("Validation status: unknown (could not reach the plan-source endpoint)")
    elif provenance in ("session", "unreviewed"):
        lines.append(
            "Validation status: PASSED (content hash matches a recorded validation run)"
            if source_info.get("validated")
            else "Validation status: NO PASSING RECORD — would be refused at enqueue"
        )
    else:
        lines.append("Validation status: not applicable (operator-supplied plan)")

    if source_info is not None:
        source_text = source_info.get("source", "")
        if source_text:
            note = " (truncated)" if source_info.get("truncated") else ""
            lines.append(f"\nPlan source{note}:\n{source_text}")

    return lines


# How many queued items a start-approval prompt names individually before it
# summarises the rest. A start drains the WHOLE queue, so the approver needs
# the list; an unbounded list would let a large queue push the warning lines
# off the top of the prompt.
_MAX_LISTED_QUEUE_ITEMS = 10

# The manager's one quiescent state. Deliberately a single token tested
# NEGATIVELY (see `_queue_activity_lines`): everything that is not this word —
# including a state this hook has never heard of, and a missing value — counts
# as "the queue may be under way" and earns a warning. The inverse spelling (a
# replica of the manager's active-state list) would fail OPEN on any state
# added upstream, which is the wrong direction for a line that tells a human
# whether approving an enqueue is approving an execution.
_IDLE_MANAGER_STATE = "idle"


def _queue_snapshot(base_url: str):
    """`GET /queue`, or `None` on any failure. Fail-open like every fetch here."""
    snapshot = _bridge_get_json(base_url, "/queue")
    return snapshot if isinstance(snapshot, dict) else None


def _queue_activity_lines(snapshot) -> list[str]:
    """State whether the queue is already draining toward hardware.

    This is the fact the tool call itself cannot show. Adding to an idle queue
    stages work for a later, separately-approved start; adding to a draining
    one hands the item to the RunEngine with no further human action, and the
    approval prompt is the only place a human can tell those apart.

    Three tiers, most specific first. The two precise headlines are classified
    from what was OBSERVED — a running item, or autostart reported on. The
    third is a single-token NEGATIVE test, ``manager_state != "idle"``, which
    catches every other way the queue can be under way (``starting_queue`` most
    importantly: a start is already in flight but no item is running yet, so
    the observed tests alone would have shown the calm headline on the one line
    whose whole job is telling a human that an enqueue is really an execution).

    Testing for the ONE idle token rather than replicating the manager's set of
    six active states is the difference between failing closed and failing
    open: if the manager ever renames or adds an active state, an unknown token
    is simply "not idle" and still warns, whereas a stale copy of the active
    list would silently classify it as safe. This hook runs standalone, in a
    different process/venv, and cannot import OSPREY, so that direction matters.
    A missing ``manager_state`` counts as not-idle for the same reason — the
    calm sentence must never render around an unknown state.

    The raw ``manager_state`` is printed in every tier so the approver sees the
    manager's own word for it either way.
    """
    if snapshot is None:
        return ["Queue state: unavailable (the bridge could not be reached)."]

    status = snapshot.get("status")
    status = status if isinstance(status, dict) else {}
    raw_state = status.get("manager_state")
    state = _sanitize_label(raw_state)
    pending = status.get("items_in_queue")
    running = snapshot.get("running_item")

    if isinstance(running, dict) and running:
        headline = (
            f"⚠️  A PLAN IS ALREADY RUNNING (manager state: {state}) — the queue is "
            f"draining, so an item added now executes with no further approval."
        )
    elif status.get("queue_autostart_enabled"):
        headline = (
            f"⚠️  AUTOSTART IS ENABLED on the manager (state: {state}) — OSPREY never "
            f"enables it, so something armed this queue out of band. An item added "
            f"now can execute with no further approval."
        )
    elif raw_state != _IDLE_MANAGER_STATE:
        headline = (
            f"⚠️  THE QUEUE IS NOT IDLE (manager state: {state}) — it may already be "
            f"draining; an item added now may execute with no further approval."
        )
    else:
        headline = f"No plan is currently running (manager state: {state})."

    lines = [headline]
    if pending is not None:
        lines.append(f"Items already queued: {pending}")
    if status.get("queue_stop_pending"):
        lines.append("A stop is PENDING: the queue halts after the running item finishes.")
    return lines


def _item_trajectory_lines(base_url: str, item: dict, deadline: float) -> list[str]:
    """One pending item's pre-flight block, indented under its own line.

    *deadline* is the whole prompt's pre-flight budget as a `time.monotonic`
    instant, and each fetch is clamped to what is left of it — so a start's
    several previews cost the approver one bounded wait between them all, not
    one wait per item.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return [
            "    Setpoint trajectory: unavailable — the pre-flight budget for this "
            "prompt is spent. Approval is not blocked."
        ]
    preview = _fetch_preview(
        base_url, item.get("name"), item.get("kwargs"), min(_PREVIEW_TIMEOUT_S, remaining)
    )
    return [f"    {line}" for line in _preview_lines(preview)]


def _queue_item_lines(snapshot, base_url: str) -> list[str]:
    """Name what a start would actually run, and flag anything agent-authored in it.

    A start drains every pending item, not only the one the agent just added,
    so the approver has to see the whole list. Session/unreviewed plans are
    called out as a group from a single `GET /plans` — per-plan source fetches
    would multiply the prompt's bridge calls by the queue length.

    The items that run first also carry their pre-flight trajectory, capped at
    `_MAX_PREVIEWED_QUEUE_ITEMS` and sharing one time budget: the trajectories
    an approver can still act on, without making the prompt's cost grow with
    the queue.
    """
    items = snapshot.get("items") if snapshot else None
    items = [item for item in (items or []) if isinstance(item, dict)]
    running = snapshot.get("running_item") if snapshot else None

    lines: list[str] = []
    if isinstance(running, dict) and running:
        lines.append(f"Currently running: {_sanitize_label(running.get('name'))}")

    if not items:
        lines.append("Pending items: none — starting an empty queue runs nothing.")
        return lines

    lines.append(f"Pending items ({len(items)}), in execution order:")
    deadline = time.monotonic() + _PREVIEW_BUDGET_S
    for index, item in enumerate(items[:_MAX_LISTED_QUEUE_ITEMS], start=1):
        rendered = f"  {index}. {_sanitize_label(item.get('name'))}"
        kwargs = item.get("kwargs")
        if kwargs:
            rendered += f" {json.dumps(kwargs)}"
        lines.append(rendered)
        if index <= _MAX_PREVIEWED_QUEUE_ITEMS:
            lines.extend(_item_trajectory_lines(base_url, item, deadline))
    if len(items) > _MAX_PREVIEWED_QUEUE_ITEMS:
        lines.append(
            f"  Trajectories above cover the first {_MAX_PREVIEWED_QUEUE_ITEMS} items only; "
            f"the rest run without one shown here."
        )
    if len(items) > _MAX_LISTED_QUEUE_ITEMS:
        lines.append(f"  … and {len(items) - _MAX_LISTED_QUEUE_ITEMS} more.")

    plans = _bridge_get_json(base_url, "/plans") or []
    untrusted = {
        plan["name"]
        for plan in plans
        if isinstance(plan, dict)
        and isinstance(plan.get("name"), str)
        and plan.get("provenance") in ("session", "unreviewed")
    }
    queued_untrusted = sorted({item.get("name") for item in items if item.get("name") in untrusted})
    if queued_untrusted:
        names = ", ".join(_sanitize_label(name) for name in queued_untrusted)
        lines.append(f"⚠️  AGENT-AUTHORED, NOT REVIEWED BY A HUMAN — this start would run: {names}")
    return lines


def _describe_queue_add(tool_input: dict, config: dict) -> list[str]:
    """Render the enqueue-approval prompt's draft/plan/queue detail lines.

    `queue_add` carries only a pinned `draft_revision` (no run record exists
    yet — the bridge mints the run *from* the draft at enqueue). So this
    fetches the shared plan draft (`GET /draft`) and renders exactly what would
    be queued: the plan name and args currently staged, whether the draft still
    matches the pinned revision, the channels and setpoint trajectory the
    launch would actually drive (see :func:`_preview_lines`), the plan's
    provenance/validation/source (see :func:`_describe_plan_provenance`), and
    whether the queue is already draining — which is what decides whether
    approving this enqueue is approving an execution.

    Fail-open is guaranteed three ways: every bridge call goes through
    `_bridge_get_json` (any fetch/parse failure → `None`), a valid-but-misshaped
    `GET /draft` body is caught by the `isinstance` guard here (→ queue lines
    only), and `main` wraps the whole call in a final try/except. The
    approval prompt must always render, degraded if need be, never blocked.
    """
    base_url = _resolve_bridge_url(config)
    lines = _queue_activity_lines(_queue_snapshot(base_url))

    snapshot = _bridge_get_json(base_url, "/draft")
    if not isinstance(snapshot, dict):
        # Bridge unreachable, a malformed body, or an unexpected shape — fail
        # open with no draft detail; the queue lines and plain reason remain.
        return lines

    pinned = tool_input.get("draft_revision")
    current_revision = snapshot.get("revision")
    draft = snapshot.get("draft")

    lines.append(_revision_match_line(pinned, current_revision))

    if not isinstance(draft, dict) or not draft.get("plan_name"):
        lines.append(
            "Draft: EMPTY — no plan is staged in the shared draft; there is "
            "nothing to queue (the bridge would refuse this enqueue)."
        )
        return lines

    plan_name = draft["plan_name"]
    lines.append(f"Plan: {_sanitize_label(plan_name)}")

    plan_args = draft.get("plan_args")
    if plan_args:
        lines.append(f"Plan args: {json.dumps(plan_args)}")

    # Before the source block: the trajectory is the answer to "what would this
    # move", and burying it under a plan body the approver has to scroll past
    # is the same as not showing it.
    lines.extend(_preview_lines(_fetch_preview(base_url, plan_name, plan_args, _PREVIEW_TIMEOUT_S)))
    lines.extend(_describe_plan_provenance(base_url, plan_name))
    return lines


def _describe_queue_start(tool_input: dict, config: dict) -> list[str]:
    """Render the start-approval prompt: everything this start would run.

    `queue_start` takes no arguments, so without these lines the approver would
    be asked to approve motion with no statement of what moves. A start drains
    the whole queue in order, so the whole queue is the answer.
    """
    base_url = _resolve_bridge_url(config)
    snapshot = _queue_snapshot(base_url)
    lines = [
        "Starting the queue runs EVERY pending item below, in order — "
        "not only the most recently added one."
    ]
    if snapshot is None:
        lines.append(
            "Queue contents: unavailable (the bridge could not be reached) — "
            "approving means starting a queue nobody here can see."
        )
        return lines
    lines.extend(_queue_item_lines(snapshot, base_url))
    return lines


def _describe_queue_stop(tool_input: dict, config: dict) -> list[str]:
    """Render the stop-approval prompt, whose two directions are opposites.

    A plain stop halts the queue after the running item finishes. ``cancel:
    true`` WITHDRAWS a pending stop, which resumes draining toward hardware —
    the arming direction, and the one an approver must not skim past.
    """
    base_url = _resolve_bridge_url(config)
    snapshot = _queue_snapshot(base_url)
    if tool_input.get("cancel"):
        lines = [
            "⚠️  WITHDRAWS A PENDING STOP — this does not halt anything. It cancels a "
            "halt someone already requested and lets the queue keep draining toward "
            "hardware."
        ]
    else:
        lines = [
            "Requests a stop: the queue halts AFTER the running item finishes. It does "
            "NOT abort the item already in motion — if that item must stop NOW, the "
            "stop_run tool is what aborts it, and this approval is not that."
        ]
    lines.extend(_queue_activity_lines(snapshot))
    return lines


def _describe_stop_run(tool_input: dict, config: dict) -> list[str]:
    """Render the abort-approval prompt: what an abort costs, and what is running.

    This is the emergency halt for a plan already moving hardware, so the
    prompt has to be honest in BOTH directions. It is not a routine stop — the
    remaining points are discarded and the hardware stays wherever the plan
    left it — and it is also not something to hesitate over when a machine
    needs stopping. Naming what is running is what lets the approver tell which
    situation they are in.
    """
    base_url = _resolve_bridge_url(config)
    snapshot = _queue_snapshot(base_url)
    lines = [
        "⚠️  ABORTS THE PLAN THAT IS RUNNING NOW — this is the emergency stop, not a "
        "queue halt. The running plan's remaining points are discarded, the data "
        "already collected is kept, and the hardware is left wherever the plan moved "
        "it (an abort returns nothing to a starting position)."
    ]
    lines.extend(_queue_activity_lines(snapshot))
    return lines


# Tool short-name -> the describer that renders its approval detail. Keyed by
# the same names `osprey.bluesky_tool_names` registers (QUEUE_ADD/QUEUE_START/
# QUEUE_STOP/STOP_RUN); this hook is deployed standalone and cannot import
# them, so the drift guard in tests/registry/test_gate_wiring.py pins these
# literals against the constants.
_QUEUE_DESCRIBERS = {
    "queue_add": _describe_queue_add,
    "queue_start": _describe_queue_start,
    "queue_stop": _describe_queue_stop,
    "stop_run": _describe_stop_run,
}


def _gallery_base_url(config: dict) -> str:
    """Resolve the artifact gallery's base URL from *config* and the environment.

    Prefers the framework's shared derivation so the per-user
    ``OSPREY_ARTIFACT_SERVER_PORT`` override multi-user deployments export is
    honoured here exactly as it is by the launcher that binds the port.

    This hook is rendered into projects that may run against a different osprey
    install than the one it shipped with, so the import is lazy and a failure
    falls back to the same resolution order done inline. Never raises.
    """
    try:
        from osprey.registry.web import resolve_web_server_base_url

        return resolve_web_server_base_url("artifact", config)
    except Exception:
        art_config = config.get("artifact_server") or {}
        host = art_config.get("host") or "127.0.0.1"
        port = os.environ.get("OSPREY_ARTIFACT_SERVER_PORT") or art_config.get("port") or 8086
        return f"http://{host}:{port}"


def _create_pre_execution_notebook(code: str, exec_mode: str, config: dict) -> str | None:
    """Create a pre-execution notebook artifact for code review.

    Returns the gallery URL if successful, None otherwise.
    Failures are silently swallowed — notebook creation must never break approval.
    """
    try:
        import nbformat

        from osprey.stores.artifact_store import ArtifactStore

        # Build a minimal notebook with the code to be reviewed
        cells = []
        cells.append(
            nbformat.v4.new_markdown_cell(
                f"# Pre-Execution Review\n\n"
                f"**Mode:** `{exec_mode or 'unspecified'}`  \n"
                f"**Status:** Pending approval  \n"
            )
        )
        cells.append(nbformat.v4.new_code_cell(code))
        nb = nbformat.v4.new_notebook()
        nb.cells = cells

        nb_bytes = nbformat.writes(nb).encode()

        store = ArtifactStore()
        entry = store.save_file(
            file_content=nb_bytes,
            filename="pre_execution_review.ipynb",
            artifact_type="notebook",
            title="Pre-Execution Review",
            description=f"Code pending approval (mode: {exec_mode or 'unspecified'})",
            mime_type="application/x-ipynb+json",
            tool_source="osprey_approval",
        )

        # Build gallery URL and bring the notebook into focus
        base_url = _gallery_base_url(config)

        # Fire-and-forget POST to switch gallery focus to this notebook
        _focus_artifact(base_url, entry.id)

        return f"{base_url}#focus"
    except Exception:
        return None


def main():
    hook_input = get_hook_input()
    if not hook_input:
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "")

    # Only inspect OSPREY tools (prefixes loaded from hook_config.json)
    _prefixes = load_hook_config().get("approval_prefixes", [])
    matched_prefix = None
    for prefix in _prefixes:
        if tool_name.startswith(prefix):
            matched_prefix = prefix
            break
    if matched_prefix is None:
        sys.exit(0)

    tool_input = hook_input.get("tool_input", {})
    short_name = tool_name[len(matched_prefix) :]

    config = load_osprey_config(hook_input)
    approval_config = config.get("approval", {})

    # Deterministic short-circuit when writes are EXPLICITLY disabled at the
    # deployment level. Empirically (Claude Code SDK 2.x, not source-verified):
    # PreToolUse hook-decision aggregation does NOT honour writes_check's JSON
    # deny if this hook ALSO emits an "ask" decision — aggregation appears to
    # be any-ask-wins, not deny-dominates, when multiple hooks return decisions
    # for the same tool call. Without the short-circuit `can_use_tool` fires
    # and the deployment-disabled invariant is violated.
    #
    # For `mcp__controls__channel_write` the renderer's `permissions.deny`
    # augmentation in `src/osprey/cli/templates/claude_code.py` is the PRIMARY
    # mechanism — it hard-blocks before any PreToolUse hook fires. The defer
    # below is intentional belt-and-braces against renderer drift (e.g., a
    # stale `settings.json` after a `writes_enabled` flip without regen).
    #
    # Gate on `is False` (not falsy) so unit tests that omit the
    # `control_system` block from their config see normal approval behaviour.
    control_system_cfg = config.get("control_system") or {}
    if control_system_cfg.get("writes_enabled") is False:
        if (
            tool_name == "mcp__python__execute"
            and tool_input.get("execution_mode", "readonly") != "readonly"
        ):
            log_hook("approval", hook_input, status="defer", detail="writes_disabled")
            sys.exit(0)
        elif tool_name == "mcp__controls__channel_write":
            log_hook(
                "approval",
                hook_input,
                status="defer",
                detail="writes_disabled_belt_and_braces",
            )
            sys.exit(0)

    # Global toggle — disabled means allow everything
    if not approval_config.get("enabled", True):
        log_hook("approval", hook_input, status="allow", detail="enabled=false")
        json.dump(build_allow_output(), sys.stdout)
        sys.exit(0)

    # Per-tool policy dispatch (default_policy applies when `tools` is absent
    # or the specific tool isn't listed; production default is "always").
    default_policy = approval_config.get("default_policy", "always")
    tool_policies = approval_config.get("tools", {})
    policy = tool_policies.get(short_name, default_policy)

    # Skip policy — no approval needed
    if policy == "skip":
        log_hook("approval", hook_input, status="allow", detail=f"policy=skip tool={short_name}")
        json.dump(build_allow_output(), sys.stdout)
        sys.exit(0)

    # Selective policy — content-aware analysis
    if policy == "selective":
        if short_name == "execute":
            exec_mode = tool_input.get("execution_mode", "")
            code = tool_input.get("code", "")

            writes_detected = has_write_patterns(code, config)
            # The agent asking for write mode is itself the signal: approval must not
            # rest on the regex recognising the spelling of the write.
            needs_approval = exec_mode == "readwrite" or writes_detected
            if needs_approval:
                reason_parts = [f"Python execution (mode: {exec_mode or 'unspecified'})"]
                if writes_detected:
                    reason_parts.append("Code contains control system write patterns.")
                if code.strip():
                    gallery_link = _create_pre_execution_notebook(code, exec_mode, config)
                    if gallery_link:
                        reason_parts.append(f"\nReview notebook: {gallery_link}")
                reason = "\n".join(reason_parts)
                log_hook("approval", hook_input, status="ask", detail="execute_selective")
                json.dump(build_approval_output(reason), sys.stdout)
                sys.exit(0)

            # Selective execute without write indicators — allow
            log_hook("approval", hook_input, status="allow", detail="execute_selective_readonly")
            json.dump(build_allow_output(), sys.stdout)
            sys.exit(0)

        # For channel_write under selective, treat as always (conservative)
        if short_name == "channel_write":
            channels = tool_input.get("operations", [])
            if not channels:
                ch = tool_input.get("channel")
                val = tool_input.get("value")
                if ch is not None:
                    channels = [{"channel": ch, "value": val}]
            channel_list = ", ".join(f"{op.get('channel')}={op.get('value')}" for op in channels)
            reason = f"Channel write: {channel_list or 'unknown'}"
            log_hook("approval", hook_input, status="ask", detail="channel_write_selective")
            json.dump(build_approval_output(reason), sys.stdout)
            sys.exit(0)

        # Other tools under selective: treat as always (conservative)
        # Falls through to "always" handling below

    # Always policy (explicit or fallback from selective for non-execute/write tools)
    reason_parts = [
        f"Tool: {short_name}",
        f"Approval policy: {policy}",
    ]
    if short_name == "execute":
        code = tool_input.get("code", "")
        exec_mode = tool_input.get("execution_mode", "")
        if code.strip():
            gallery_link = _create_pre_execution_notebook(code, exec_mode, config)
            if gallery_link:
                reason_parts.append(f"\nReview notebook: {gallery_link}")
    elif short_name == "channel_write":
        channels = tool_input.get("operations", [])
        if not channels:
            ch = tool_input.get("channel")
            val = tool_input.get("value")
            if ch is not None:
                channels = [{"channel": ch, "value": val}]
        channel_list = ", ".join(f"{op.get('channel')}={op.get('value')}" for op in channels)
        if channel_list:
            reason_parts.append(f"Channels: {channel_list}")
    elif short_name in _QUEUE_DESCRIBERS:
        try:
            reason_parts.extend(_QUEUE_DESCRIBERS[short_name](tool_input, config))
        except Exception:
            pass

    reason = "\n".join(reason_parts)
    log_hook(
        "approval",
        hook_input,
        status="ask",
        detail=f"policy={policy} tool={short_name}",
    )
    json.dump(build_approval_output(reason), sys.stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()
