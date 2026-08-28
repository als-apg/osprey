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
              │
              ▼
  Is this call a write, and is the deployment
  NOT armed for the target it would act on?
     │                    │
    YES                  NO / no posture stated
     │                    │
     ▼                    │
  EXIT, no decision       │
  (writes_check's deny    │
   or the tool's own      │
   lane gate refuses it)  │
                          ▼
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

## Write posture

Ahead of all of that, a write tool whose target this deployment has not armed
exits with NO decision at all, so that the layer which refuses it — the
`osprey_writes_check` deny, or `queue_start`'s own lane gate — is not reopened
by an approval prompt. Posture is per target, so the same tool on the same
deployment can defer on one target and prompt on the other. A config that
states no posture anywhere prompts exactly as it always has.

## Write-approval stamps

A `channel_write` ask also leaves a *stamp* beside the target state: the target
and generation this prompt was rendered against, written before the dialog is
shown. The controls server reads it back when the tool finally runs and refuses
the write if the session has moved since — see `_stamp_write_approval` for the
correlation key and the multi-session rule. The stamp is an enrichment, never a
gate: a render that cannot write one simply produces a prompt the server cannot
cross-check, exactly as every older render already does.

## Bluesky plan lanes

A deployment that renders two plan lanes — one Bluesky stack per control-system
target — turns "queue this" and "start the queue" into questions about WHICH
machine. The queue describers therefore name the lane an operation binds to,
fetch their detail from that lane's own bridge, and say out loud when a start
names a lane the session has switched away from (which the deployment refuses).
The lane map is render-time truth read from the config's ``services.<lane>``
blocks; the session's target still comes only from the state file. A single-lane
deployment has nothing to address and renders exactly what it always did.
"""

import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osprey_hook_log import (
    AUDIT_DECISION_ASK,
    emit_audit,
    get_hook_input,
    is_write_call,
    is_write_tool,
    load_hook_config,
    load_osprey_config,
    log_hook,
    short_tool_name,
    write_tools,
)

# The shared, stdlib-only reader for the control-system target state — the ONLY
# source this hook consults for target identity (see `_target_line`). Imported
# here, while this file's own directory is still the first `sys.path` entry the
# line above put there, because the reader is a sibling script rather than an
# installed module.
#
# The import is guarded because a project rendered before the switch capability
# existed has no such sibling, and an approval prompt that failed to render at
# all would be a far worse outcome than one that says the target is unknown. A
# missing reader therefore lands on exactly the same explicit baseline line as
# unreadable state.
try:
    import osprey_target_state as _target_state
except Exception:  # pragma: no cover - older render without the reader
    _target_state = None

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


def build_approval_output(reason_detail: str, hook_input=None, read_record=None) -> dict:
    """The ask envelope, with the target identity line above every detail.

    Every ask this hook emits goes out through here, which is why the identity
    line is added here and not at the three call sites: a line that is assembled
    per branch is a line that eventually goes missing from one of them, and the
    absence of a `Target:` line is indistinguishable from a target that is not
    the machine. It sits directly under the headline so a long enrichment block
    (a full queue listing, a plan's source) cannot push it out of view.

    The record is read ONCE here and handed to both consumers: the line the
    human reads and the stamp the server later cross-checks have to be the same
    claim, and two reads of a file another process is writing are two claims.

    *read_record* extends that guarantee to the whole prompt. A describer that
    also names a target — the plan lane a queue operation binds to, the
    destination of a switch — is handed the same reader, so one prompt describes
    one read of the state file rather than two reads a switch could land between.
    Left unset, this resolves its own, exactly as before.
    """
    record = (read_record or _record_reader(hook_input))()
    _stamp_write_approval(hook_input, record)
    _record_ask(hook_input)
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "ask",
            "permissionDecisionReason": (
                "\u26a0\ufe0f  OSPREY APPROVAL REQUIRED\n\n"
                f"{_target_line(hook_input, record=record)}\n\n"
                f"{reason_detail}\n\n"
                "Review the operation above and approve to proceed."
            ),
        }
    }


def _record_ask(hook_input):
    """Put this ask in the deployment's audit ledger, and never cost the prompt.

    Emitted from the ask *builder* for the same reason the target line is: an
    audit record assembled per branch is one that eventually goes missing from
    a branch. The three call sites differ only in the prompt text they compose,
    and the record names the tool rather than the prose, so one call here
    covers every ask this hook can emit.

    The reason is the same for all of them — an approval policy asked a human.
    Which policy did so is a debug fact and stays on the debug line; the ledger
    answers who was asked to approve what.
    """
    try:
        emit_audit(
            "approval",
            hook_input,
            decision=AUDIT_DECISION_ASK,
            subject=(hook_input or {}).get("tool_name", ""),
            reason="approval_required",
        )
    except Exception:
        pass  # the audit trail must never cost the prompt


def _read_record_once(hook_input=None):
    """The session's state record, or ``None`` \u2014 and never an exception.

    :func:`_target_line` already degrades to the baseline line on any trouble;
    this is the same tolerance moved one level up, so that hoisting the read out
    of it cannot turn an unreadable state directory into a prompt that fails to
    render at all.
    """
    try:
        return _session_state_record(hook_input)
    except Exception:
        return None


def _record_reader(hook_input=None):
    """A callable resolving this session's state record AT MOST once.

    Threading a reader rather than a record is what lets one prompt describe one
    read without making a prompt that needs no record pay for one. Every ask
    names the session's target, but a single-lane deployment's queue prompt has
    no lane to resolve and must not read the state directory a second time —
    nor, before the ask is even assembled, a first time.

    The cell is per call of this function, so nothing outlives the prompt: a
    hook process renders one prompt, and a memo any wider would hand a later
    call an answer taken before a switch it should have seen.
    """
    cell = []

    def read():
        if not cell:
            cell.append(_read_record_once(hook_input))
        return cell[0]

    return read


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

    Carries ``Authorization: Bearer <OSPREY_PANEL_TOKEN>`` whenever that
    variable holds a non-blank value in the hook's inherited environment — the
    web terminal exports it into the agent it launches, so the hook picks it up
    without reading a file or importing anything. When it is unset, empty or
    whitespace-only no ``Authorization`` header is sent at all, which is the
    unauthenticated loopback POST this call has always been, and which matches
    how ``mcp_server.http._panel_auth_headers`` reads the same carrier.

    Non-fatal: silently swallows errors so focus failures never block approval.
    A gallery that refuses the call — 401 included — is a focus failure like any
    other and is swallowed the same way.
    """
    try:
        import urllib.request

        data = json.dumps({"artifact_id": artifact_id}).encode()
        headers = {"Content-Type": "application/json"}
        token = (os.environ.get("OSPREY_PANEL_TOKEN") or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(
            f"{gallery_base_url}/api/focus",
            data=data,
            headers=headers,
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
    config.yml, then the port the deployment publishes its bridge on
    (``services.bluesky.port``, dialed on loopback), then the default above.
    """
    full = os.environ.get("BLUESKY_BRIDGE_URL")
    if full:
        return full.rstrip("/")
    bluesky = config.get("bluesky") or {}
    url = bluesky.get("bridge_url") if isinstance(bluesky, dict) else None
    if url:
        return str(url).rstrip("/")
    services = config.get("services") or {}
    block = services.get("bluesky") if isinstance(services, dict) else None
    port = block.get("port") if isinstance(block, dict) else None
    if port:
        return f"http://127.0.0.1:{port}"
    return _DEFAULT_BRIDGE_URL


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


# ---------------------------------------------------------------------------
# Identity: which control-system target this approval would act on
# ---------------------------------------------------------------------------
# One line, on every prompt, answering the question no tool argument answers:
# is this session pointed at the real machine or at a simulation? It is read
# EXCLUSIVELY from the target-state file the controls server writes. The
# rendered config.yml is in this hook's hand throughout and is deliberately not
# consulted: config states what the deployment STARTS as, so on a session that
# switched at run time it would produce a confident, stale, wrong safety claim —
# and a wrong `Target:` line is worse than none, because nobody audits a line
# that looks answered.

#: Rendered whenever the target cannot be resolved: no switch capability in this
#: render, no controls server running, two sessions sharing the checkout, or a
#: corrupt record. The wording is fixed — operators and tests both key off it —
#: and it is never omitted, because a missing line reads as "not the machine".
_TARGET_BASELINE_LINE = "Target: deployment baseline (state unavailable)"


def _real_machine_claim(meta):
    """Whether *meta* CLAIMS a real machine: ``True``, ``False``, or ``None``.

    Three states, not two. A ``real_machine`` key that is absent, null, or not a
    boolean is a record this hook cannot read — an older or newer writer, or a
    truncated one — and coercing that to ``False`` would print "virtual
    accelerator (simulation)" over a record that never said so. ``None`` sends
    the caller to the explicit baseline line, which is the honest answer: the
    identity is unknown, and unknown is not the same claim as simulated.
    """
    if not isinstance(meta, dict):
        return None
    claim = meta.get("real_machine")
    return claim if isinstance(claim, bool) else None


#: "the caller did not pass a record", which is a different thing from a caller
#: passing the ``None`` that means "no record resolved".
_UNREAD = object()

#: Short name of the tool whose approvals are stamped. A literal for the same
#: reason the describer tables' keys are: this hook is deployed standalone and
#: cannot import the name the controls server registers.
_CHANNEL_WRITE_TOOL = "channel_write"

#: Stamp files live beside the target state, under a prefix the state-file glob
#: (``target_state_*.json``) cannot match — a stamp must never be mistaken for a
#: server's state file by the reader here or the sweeper on the writer's side.
WRITE_APPROVAL_PREFIX = "write_approval_"
WRITE_APPROVAL_SUFFIX = ".json"

#: How long a stamp stays on disk. Long enough that a human can think before
#: clicking, short enough that the directory does not accumulate one file per
#: write for the life of a deployment. Expiry is housekeeping only: the server
#: matches a stamp by payload, so an expired one that survives is still correct.
WRITE_APPROVAL_TTL_S = 3600.0


def write_approval_key(tool_input) -> str | None:
    """Correlate a prompt with the tool call it will authorize, or ``None``.

    The key is a SHA-256 over the canonical JSON of the two fields of the write
    that both sides see — ``operations`` and ``verification_level`` — because
    the hook and the server share no identifier at all: Claude Code gives the
    hook a session id and a tool-call payload, and the MCP tool receives only
    its arguments. The payload is the one thing that provably crosses the gap,
    and the tool's own parameter names are the spelling of it, so the legacy
    single-channel shape (``channel``/``value``, which the tool does not accept)
    is deliberately not stamped.

    ``None`` whenever no key can be formed, which the caller renders as "do not
    stamp": no comparison is a far better failure than a wrong comparison.

    The server restates this derivation in its own module. That duplication is
    the same one the state-file path contract already carries — hooks run
    outside the osprey venv and cannot import it — and the two spellings are
    pinned against each other by a test.
    """
    if not isinstance(tool_input, dict):
        return None
    operations = tool_input.get("operations")
    if not isinstance(operations, list) or not operations:
        return None
    try:
        payload = json.dumps(
            {
                "operations": operations,
                "verification_level": tool_input.get("verification_level"),
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _channel_write_summary(tool_input) -> str:
    """``channel=value`` for every write in the call, or ``""`` if there is none.

    Both approval policies render the same list — one as the whole reason, one as
    a line under it — so the list is derived once. The legacy single-channel
    shape (``channel``/``value``) is folded in for display only: a prompt has to
    name what it was actually handed, whatever spelling the caller used.
    """
    channels = tool_input.get("operations", [])
    if not channels:
        ch = tool_input.get("channel")
        val = tool_input.get("value")
        if ch is not None:
            channels = [{"channel": ch, "value": val}]
    return ", ".join(f"{op.get('channel')}={op.get('value')}" for op in channels)


def _stamp_binding(record):
    """``(target, generation, server_pid)`` for a stamp, each possibly ``None``.

    Normalized exactly as the server normalizes the same record: a target that
    is not a non-empty string, or a generation that is not an integer, makes the
    binding unknown *as a whole* rather than half-known. Both sides collapsing
    those cases the same way is what keeps "unpublished" a stable answer instead
    of an intermittent mismatch.
    """
    if not isinstance(record, dict) or _target_state is None:
        return None, None, None
    try:
        server_pid = int(record.get("server_pid"))
    except (TypeError, ValueError):
        server_pid = None
    target = _target_state.selected_target(record)
    if not isinstance(target, str) or not target:
        return None, None, server_pid
    try:
        return target, int(record.get("generation")), server_pid
    except (TypeError, ValueError):
        return None, None, server_pid


def _prune_write_approvals(directory) -> None:
    """Drop stamps older than :data:`WRITE_APPROVAL_TTL_S`. Never raises."""
    cutoff = time.time() - WRITE_APPROVAL_TTL_S
    try:
        names = os.listdir(directory)
    except OSError:
        return
    for name in names:
        if not (name.startswith(WRITE_APPROVAL_PREFIX) and name.endswith(WRITE_APPROVAL_SUFFIX)):
            continue
        path = os.path.join(directory, name)
        try:
            if os.path.getmtime(path) < cutoff:
                os.unlink(path)
        except OSError:
            continue


def _stamp_write_approval(hook_input, record) -> None:
    """Record the binding a ``channel_write`` prompt is being rendered against.

    This closes the window the tool cannot see. Approval is enforced out here,
    before the server is called at all, so the server's earliest observation is
    its own entry — and between the moment this prompt is rendered and the
    moment the human clicks approve, a target switch can land. Writing the
    rendered binding down is what lets the server compare against what the human
    was actually shown rather than against a state it inherited.

    The stamp carries the ``server_pid`` of the record it was rendered from. Two
    sessions sharing one checkout write into one directory, and an identical
    write payload would otherwise collide; a stamp whose ``server_pid`` is not
    the reading server's is ignored by it, which costs that call its comparison
    and never produces a wrong one.

    Never raises, and never blocks the prompt: every failure — no reader, no
    resolvable state directory, an unwritable disk — simply leaves no stamp, and
    a missing stamp means the server does no comparison, exactly as it does for
    a project rendered before this existed.
    """
    try:
        if _target_state is None or not isinstance(hook_input, dict):
            return
        tool_name = str(hook_input.get("tool_name") or "")
        if tool_name.rsplit("__", 1)[-1] != _CHANNEL_WRITE_TOOL:
            return
        key = write_approval_key(hook_input.get("tool_input"))
        if key is None:
            return
        directory = _target_state.resolve_state_dir(hook_input)
        if not directory:
            return
        os.makedirs(directory, exist_ok=True)
        _prune_write_approvals(directory)

        target, generation, server_pid = _stamp_binding(record)
        path = os.path.join(directory, f"{WRITE_APPROVAL_PREFIX}{key}{WRITE_APPROVAL_SUFFIX}")
        payload = {
            "tool": _CHANNEL_WRITE_TOOL,
            "key": key,
            "target": target,
            "generation": generation,
            "server_pid": server_pid,
            "rendered_at": time.time(),
        }
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(tmp, path)
    except Exception:
        return


def _session_state_record(hook_input=None):
    """The raw state record this session resolves to, or ``None``.

    A one-line delegation to the reader's :func:`read_session_record`, which
    applies the identical liveness, parentage and ambiguity rules
    :func:`read_session_target` does. This hook deliberately holds NO copy of
    those rules: a second walk of the state directory would eventually differ
    from the first — dropping the liveness filter is the easy mistake, and it
    ends with a crashed server's stale file answering for the live one.

    ``None`` for every baseline outcome, for a render with no reader, and on any
    trouble at all; the caller renders the explicit baseline or "cannot be
    previewed", and the approval still reaches the human.
    """
    state = _target_state
    if state is None:
        return None
    return state.read_session_record(hook_input)


def _target_identity_phrase(record, target):
    """How one target is SPOKEN OF on this prompt, or ``None`` if it cannot be.

    The single place the two identity phrasings live, because more than one
    surface now names a target: the ``Target:`` line names the session's, and
    the plan-lane lines name the target a lane is wired to. A second spelling of
    "virtual accelerator (simulation)" would eventually drift from this one, and
    two different phrasings for one machine on one prompt is exactly the
    ambiguity the line exists to remove.

    ``None`` means the record makes no readable claim about *target* — the
    tri-state of :func:`_real_machine_claim`, propagated rather than flattened,
    so every caller renders its own explicit "unknown" instead of inheriting a
    silent "simulation".
    """
    if _target_state is None or record is None or not target:
        return None
    meta = _target_state.target_metadata(record, target)
    claim = _real_machine_claim(meta)
    if claim is None:
        return None
    if claim:
        # The endpoint is the selected-role one the writer chose (write_access
        # when writes are enabled, read_only otherwise). Rendered verbatim:
        # picking a role here would be a second opinion about which gateway
        # this session actually holds.
        endpoint = _sanitize_label(meta.get("endpoint") or "endpoint not recorded")
        # The label is the writer's too, for the same reason the destination
        # line below reads it rather than re-deriving one: a deployment may put
        # a stand-in behind the live role, and only the writer knows that. The
        # fallback covers a record from a writer that recorded no label at all;
        # it never softens the claim, which stays `real_machine`'s to make.
        label = _sanitize_label(meta.get("label") or "LIVE MACHINE")
        return f"{label} ({endpoint})"
    return "virtual accelerator (simulation)"


def _target_line(hook_input=None, record=_UNREAD) -> str:
    """Name the target this approval would act on, in one line that always renders.

    On EVERY ask, not only writes: an approver deciding about a queue start, a
    patch or an execution has to know where the session points, and a line that
    appears only on some prompts teaches them to read its absence as safe.

    *record* lets a caller that has already resolved the session's record pass it
    in, so the line and the write-approval stamp describe one read of the file
    rather than two. Left unset, the record is resolved here exactly as before.

    Fail-open like the rest of this hook — every failure, including the reader
    module being absent from an older render and a record whose identity field
    this hook cannot read, degrades to :data:`_TARGET_BASELINE_LINE` rather than
    raising or guessing.
    """
    try:
        state = _target_state
        if record is _UNREAD:
            record = _session_state_record(hook_input)
        if record is None:
            return _TARGET_BASELINE_LINE
        phrase = _target_identity_phrase(record, state.selected_target(record))
        if phrase is None:
            return _TARGET_BASELINE_LINE
        return f"Target: {phrase}"
    except Exception:
        return _TARGET_BASELINE_LINE


def _describe_control_target_set(
    tool_input, config, hook_input=None, read_record=None
) -> list[str]:
    """Render where a prospective target switch would leave the session.

    The identity line above names where the session is NOW; for a switch, where
    it would be is the whole decision, so this names the destination's label,
    the endpoint it would hold, and the channel the switch will probe to prove
    the destination is really there.

    Everything comes from the same state file as the identity line, and — when
    the caller hands one over — from the same READ of it, so the "where you are"
    line and the "where you would be" lines below it cannot straddle a switch
    that landed mid-prompt. Its per-target metadata is read schema-tolerantly: a
    record written by a writer that does not record a probe channel for this
    destination simply yields no probe line, rather than an empty one or an
    exception.
    """
    destination = tool_input.get("target")
    if not isinstance(destination, str) or not destination.strip():
        return ["Destination: not named in this call — the switch would be refused."]
    # The raw value is what the state file is keyed by; the escaped copy is what
    # this prompt prints. Sanitizing before the lookup would turn a hostile
    # target name into a silent miss instead of a visible one.
    destination = destination.strip()
    shown = _sanitize_label(destination)

    record = (read_record or _record_reader(hook_input))()
    if record is None:
        return [
            f"Destination: {shown} — cannot be previewed (state unavailable). "
            f"Approval is not blocked; this prompt simply cannot show the endpoint "
            f"the switch would land on."
        ]

    meta = _target_state.target_metadata(record, destination)
    if meta is None:
        return [
            f"Destination: {shown} — the state file records no metadata for it. "
            f"Approval is not blocked; the endpoint cannot be shown."
        ]

    lines = []
    claim = _real_machine_claim(meta)
    if claim is None:
        # Same tri-state as the identity line: silence about a destination's
        # nature must not read as "simulation". Saying so keeps the endpoint and
        # probe lines below useful without attaching a safety claim to them.
        lines.append(
            "⚠️  The state file does not record whether this destination is the real "
            "machine — treat it as unknown."
        )
    elif claim:
        lines.append(
            "⚠️  THIS SWITCH POINTS THE SESSION AT THE LIVE MACHINE — every write "
            "approved afterwards moves real hardware."
        )
    label = _sanitize_label(meta.get("label") or destination)
    endpoint = _sanitize_label(meta.get("endpoint") or "endpoint not recorded")
    lines.append(f"Destination: {label} ({endpoint})")
    probe = meta.get("probe_channel")
    if probe:
        lines.append(f"Destination probe channel: {_sanitize_label(probe)}")
    return lines


#: Tool short name -> describer, for tools whose approval detail comes from the
#: target state rather than from the bridge. A table of its own rather than an
#: entry in `_QUEUE_DESCRIBERS` further down, because the two groups answer from
#: different places: these tools are described entirely from the state file the
#: controls server writes, while the queue tools describe a bridge they reach
#: over HTTP. Both tables' describers take the same three arguments — resolving
#: the state file means resolving the repo root, and the hook payload's ``cwd``
#: is part of that derivation.
#:
#: The key is a literal for the same reason the queue table's keys are — this
#: hook is deployed standalone and cannot import the name the controls server
#: registers. It stays inert until that tool actually reaches this hook.
_TARGET_DESCRIBERS = {
    "control_target_set": _describe_control_target_set,
}


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


# ---------------------------------------------------------------------------
# Plan lanes: WHICH Bluesky stack a queue operation would act on
# ---------------------------------------------------------------------------
# A plan lane is a whole Bluesky stack — bridge, queue manager, worker — wired
# at RENDER time to one control-system target. Nearly every deployment renders
# one; a deployment that opted into `bluesky.second_lane` renders two, one per
# target, and then "start the queue" is an ambiguous instruction about hardware
# motion until a lane is named.
#
# Two sources, and the split matters:
#
# * the lane MAP (which lanes exist, and which target each serves) is
#   render-time truth and comes from the rendered config's `services.<lane>`
#   blocks — the deployment's own statement of its shape, and the same key the
#   host reads (`mcp_server/bluesky/lanes.discover_lanes`). Session state could
#   not answer it: a lane's target does not move when a session switches;
# * the SESSION target still comes exclusively from the state file, exactly as
#   the `Target:` line does. Reading it from config would produce a confident,
#   stale answer on the one deployment shape this axis exists for.
#
# The active lane is the lane whose target equals the session target — the same
# exact-match-on-exactly-one rule the host applies, so this prompt and the tool
# that later refuses agree about which lane is addressed. Zero matches and two
# matches are both "no answer", and no answer is rendered as an explicit
# unresolved line rather than as a guess.

#: The lane every deployment has had since the plan stack shipped.
_LANE_ONE = "bluesky"

#: Every service key that can name a lane, in render order. Literals for the
#: same reason the describer tables' keys are: this hook is deployed standalone
#: and cannot import `osprey.bluesky_bridge_connection.LANE_KEYS`.
_LANE_KEYS = ("bluesky", "bluesky_va", "bluesky_live")


#: The control-system type whose deployments run against a simulator, and the
#: two target names every lane is wired to. Literals for the same reason the
#: lane keys are: this hook is deployed standalone and can import neither
#: ``osprey_connectors.types`` nor ``target_state``'s ``TARGET_VA``/``TARGET_LIVE``.
_VIRTUAL_ACCELERATOR_TYPE = "virtual_accelerator"
_TARGET_VA = "va"
_TARGET_LIVE = "live"


def _baseline_target(config: dict) -> str:
    """The target this deployment's config declares as its baseline.

    The stdlib restatement of ``target_banner.resolve_baseline_target``, and the
    fallback a lane with no declared ``target`` takes — the same substitution
    ``lanes.discover_lanes`` makes host-side and ``queue_backend`` makes
    bridge-side, so all three answer "which target does this lane serve"
    identically. Without it a hand-degraded config (a lane block whose ``target``
    key was dropped) would render "no lane serves your target" over a deployment
    whose lanes cover it perfectly.

    This is lane TOPOLOGY, not identity: it answers what the deployment is wired
    to, which is a render-time fact config is the authority on. Where the SESSION
    is pointed is a different question, and only the state file may answer it.
    """
    section = config.get("control_system") if isinstance(config, dict) else None
    cs_type = section.get("type") if isinstance(section, dict) else None
    return _TARGET_VA if cs_type == _VIRTUAL_ACCELERATOR_TYPE else _TARGET_LIVE


def _lane_block(config: dict, lane_key: str) -> dict:
    """A lane's ``services.<lane>`` block, or ``{}`` if it has none."""
    services = config.get("services") if isinstance(config, dict) else None
    block = services.get(lane_key) if isinstance(services, dict) else None
    return block if isinstance(block, dict) else {}


def _rendered_lanes(config: dict) -> list[tuple[str, str]]:
    """``[(lane key, the target it serves)]`` in render order.

    A lane exists in the rendered config as its own ``services.<lane>`` block,
    and carries a ``target`` key only on a two-lane deploy — the single-lane
    block has never needed one, which is what keeps a single-lane prompt
    byte-for-byte what it always was. A lane without one therefore serves the
    deployment BASELINE (:func:`_baseline_target`), which is the substitution
    both the host and the bridge make for the same block.

    Lane 1 is reported whether or not the config names it, because a config this
    hook could not read must not be able to make the default lane disappear.
    """
    baseline = _baseline_target(config)
    lanes = [
        (key, _declared_lane_target(config, key) or baseline)
        for key in _LANE_KEYS
        if _lane_block(config, key)
    ]
    if not any(key == _LANE_ONE for key, _ in lanes):
        lanes.insert(0, (_LANE_ONE, baseline))
    return lanes


def _declared_lane_target(config: dict, lane_key: str) -> str | None:
    """The control target a lane's own config block declares, or ``None``.

    Read exactly as ``bluesky_bridge_connection.lane_declared_target`` reads it,
    whitespace included: the tool this prompt gates resolves the same key
    through that function, and a value one of them trims and the other does not
    is a lane the two answer for differently.
    """
    target = _lane_block(config, lane_key).get("target")
    return target if isinstance(target, str) and target else None


def _lane_situation(config: dict, hook_input=None, read_record=None) -> dict:
    """Everything the lane lines are rendered from, resolved once.

    Keys: ``lanes`` (the rendered map), ``multi`` (whether there is anything to
    address at all), ``record`` (the session's state record, or ``None``),
    ``session_target`` and ``active`` (the lane serving it, or ``None``).

    Never raises: an unreadable config is a single-lane deployment and an
    unreadable state file is an unknown session target, which are the two
    answers that degrade the prompt rather than blocking it.

    The state file is read only when there is more than one lane. A single-lane
    deployment has nothing to address, renders no lane line at all, and must
    therefore do exactly the work — and exactly the file reads — it did before
    lanes existed. That is why *read_record* is a reader and not a record: the
    prompt's one read happens here when a lane has to be resolved, and does not
    happen at all when none does.
    """
    try:
        lanes = _rendered_lanes(config)
    except Exception:
        # Unreachable in practice (`_rendered_lanes` reads only mappings), and a
        # single-lane answer either way — no lane line is rendered from it. The
        # target named is the loud one, so a bug here cannot quietly say
        # "simulation".
        lanes = [(_LANE_ONE, _TARGET_LIVE)]
    record = (read_record or _record_reader(hook_input))() if len(lanes) > 1 else None
    session_target = None
    if _target_state is not None and record is not None:
        try:
            session_target = _target_state.selected_target(record)
        except Exception:
            session_target = None
    matches = [key for key, target in lanes if session_target and target == session_target]
    return {
        "lanes": lanes,
        "multi": len(lanes) > 1,
        "record": record,
        "session_target": session_target,
        "active": matches[0] if len(matches) == 1 else None,
    }


def _lane_target_of(situation: dict, lane_key) -> str | None:
    """The target the named lane serves, per the rendered config."""
    for key, target in situation["lanes"]:
        if key == lane_key:
            return target
    return None


def _lane_target_phrase(situation: dict, lane_target) -> str:
    """How a lane's target is spoken of, with an explicit word for every gap.

    The identity voice is :func:`_target_identity_phrase`'s, so a lane and the
    ``Target:`` line above it name one machine the same way. What this adds is
    the way a lane can have no phrasable identity — a state file that records
    nothing about the target the lane serves — said out loud rather than left
    blank or guessed at.

    The empty-target guard is defensive only: every lane :func:`_rendered_lanes`
    reports serves a target, declared or baseline. It is here so that a caller
    naming a lane this deployment does not render gets a word rather than the
    string ``None``.
    """
    if not lane_target:
        return "target unknown to this prompt"
    phrase = _target_identity_phrase(situation["record"], lane_target)
    if phrase:
        return phrase
    return f"{_sanitize_label(lane_target)} (identity not recorded in the state file)"


def _lane_roster_text(situation: dict) -> str:
    """Every rendered lane and the target it serves, for a refusal line."""
    return ", ".join(f"{key!r} ({_sanitize_label(target)})" for key, target in situation["lanes"])


def _session_target_phrase(situation: dict) -> str:
    """How this session's own target is spoken of in a lane line."""
    session_target = situation["session_target"]
    if not session_target:
        return "unavailable (the session's target state could not be read)"
    return _lane_target_phrase(situation, session_target)


def _unresolved_lane_lines(situation: dict, action: str) -> list[str]:
    """Why no lane could be named, in the operator's terms. Never empty.

    Both branches state the consequence — this deployment refuses an unaddressed
    plan operation rather than picking a machine — because an approver reading a
    lane line has to know whether approving would achieve anything.
    """
    if not situation["session_target"]:
        return [
            f"Bluesky PLAN lane: unresolved — the session's target state could not be "
            f"read, so this prompt cannot name the lane {action} would act on. Approval "
            f"is not blocked; this deployment renders {len(situation['lanes'])} lanes "
            f"({_lane_roster_text(situation)})."
        ]
    return [
        f"⚠️  NO ACTIVE BLUESKY PLAN LANE — this session is on target "
        f"{_sanitize_label(situation['session_target'])}, which no single rendered lane "
        f"serves (rendered lanes: {_lane_roster_text(situation)}). {action} will be "
        f"REFUSED."
    ]


def _lane_bridge_url(situation: dict, lane_key, config: dict) -> str | None:
    """The base URL of one lane's bridge, or ``None`` when it cannot be resolved.

    Lane 1 resolves exactly as it always has (:func:`_resolve_bridge_url`), so a
    single-lane deployment makes the same call to the same place. A second lane
    has no ``bluesky.bridge_url`` of its own: its address is the loopback URL of
    the port the build published for it, which is the same derivation
    ``bluesky_bridge_connection.resolve_bridge_url`` applies, plus the per-lane
    ``<LANE>_BRIDGE_URL`` override the framework sets per bridge instance.

    ``None`` rather than lane 1's URL when a second lane's port is missing: a
    queue listing fetched from the WRONG lane would describe a different machine
    than the one being approved, which is worse than no listing at all. The
    caller renders :func:`_unaddressable_lane_line` for it, which says a
    different thing from "the bridge could not be reached" — one is a config
    that never published an address, the other an address nobody answered.
    """
    if not lane_key or lane_key == _LANE_ONE:
        return _resolve_bridge_url(config)
    full = os.environ.get(f"{lane_key.upper()}_BRIDGE_URL")
    if full:
        return full.rstrip("/")
    port = _lane_block(config, lane_key).get("port")
    return f"http://127.0.0.1:{port}" if port else None


def _unaddressable_lane_line(lane_key) -> str:
    """One line for a lane this deployment publishes no address for.

    Deliberately distinct from the "could not be reached" line: an operator who
    reads "unavailable" reaches for the bridge's logs, and this case has no
    bridge to look at — the rendered config simply carries no port for the lane,
    so nothing was ever asked.
    """
    return (
        f"Queue state: unavailable — this deployment's config publishes no port for the "
        f"{lane_key!r} lane, so its bridge cannot be reached from here. Approval is not "
        f"blocked."
    )


def _describe_queue_add(
    tool_input: dict, config: dict, hook_input=None, read_record=None
) -> list[str]:
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

    On a deployment with two plan lanes the first line names the lane the item
    would BIND to — the active one, the lane serving the session's target — and
    everything below it is fetched from THAT lane's bridge: each lane holds its
    own draft and its own queue, so lane 1's answers would describe a different
    machine than the one being approved. A single-lane deployment has no lane to
    choose and renders exactly what it always did.
    """
    situation = _lane_situation(config, hook_input, read_record)
    lines: list[str] = []
    lane_key = None
    if situation["multi"]:
        lane_key = situation["active"]
        if lane_key is None:
            return _unresolved_lane_lines(situation, "Queuing a Bluesky PLAN") + [
                "The staged draft and the queue are NOT previewed below: this prompt "
                "cannot tell which lane's queue the item would land in, and previewing "
                "the wrong lane would describe a different machine."
            ]
        lines.append(
            f"Bluesky PLAN lane: {lane_key} "
            f"(target: {_lane_target_phrase(situation, _lane_target_of(situation, lane_key))}) "
            f"— the lane serving this session's target, which is where this plan binds."
        )

    base_url = _lane_bridge_url(situation, lane_key, config)
    if base_url is None:
        lines.append(_unaddressable_lane_line(lane_key))
        return lines
    lines.extend(_queue_activity_lines(_queue_snapshot(base_url)))

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


def _lane_start_lines(situation: dict, tool_input: dict) -> tuple[list[str], str | None]:
    """The lane block for a start, and the lane whose queue to preview.

    Returns ``([], None)`` on a single-lane deployment — nothing to address, and
    the queue listing comes from the one bridge it always came from.

    On a two-lane deployment the start is an ADDRESS, and this renders whether
    the address is one the deployment will honour. The mismatch case is the
    reason the block exists: a session that switched targets between the enqueue
    and the start would otherwise be shown a queue listing with no hint that
    starting it drives the machine the session has just left. The host refuses
    such a start (``lane_mismatch``), so the prompt says so rather than asking a
    human to approve something that cannot happen.

    The second element is the lane whose queue the prompt should show: the bound
    one when it is real, and ``None`` when no lane can be named — in which case
    a listing would be a listing of the wrong queue.
    """
    if not situation["multi"]:
        return [], None

    requested = tool_input.get("lane")
    requested = requested.strip() if isinstance(requested, str) and requested.strip() else None
    active = situation["active"]

    if requested is None:
        lines = [
            f"⚠️  NO LANE NAMED — this deployment renders {len(situation['lanes'])} Bluesky "
            f"PLAN lanes ({_lane_roster_text(situation)}), so a start has to name which "
            f"one (lane=<the lane id queue_add returned>). Starting will be REFUSED."
        ]
        if active:
            lines.append(
                f"The lane serving this session's target is {active!r} "
                f"(target: {_lane_target_phrase(situation, _lane_target_of(situation, active))})."
            )
        return lines, None

    shown = _sanitize_label(requested)
    if all(key != requested for key, _ in situation["lanes"]):
        return [
            f"⚠️  UNKNOWN LANE — this deployment renders no {shown!r} Bluesky PLAN lane "
            f"(rendered lanes: {_lane_roster_text(situation)}). Starting will be REFUSED."
        ], None

    bound_phrase = _lane_target_phrase(situation, _lane_target_of(situation, requested))
    bound_line = f"Bluesky PLAN lane: {shown} (target: {bound_phrase})"

    if active == requested:
        return [f"{bound_line} — the lane this session's target is on."], requested

    if situation["session_target"] is None:
        # The lane is real and the queue below is genuinely its own; what cannot
        # be said is whether it is the lane the session is on. Saying "mismatch"
        # here would be a claim the state file never made.
        return [
            bound_line,
            "This session's target state could not be read, so this prompt cannot say "
            "whether that is the lane this session is on. Approval is not blocked; a "
            "start on a lane the session has left is refused by the deployment.",
        ], requested

    lines = [
        f"⚠️  LANE MISMATCH — the {shown!r} lane serves {bound_phrase}; this session's "
        f"target is {_session_target_phrase(situation)}. Starting will be REFUSED."
    ]
    if active:
        lines.append(
            f"The lane serving this session's target is {active!r} — start that one "
            f"instead, but only if its queue is what should run."
        )
    else:
        lines.append(
            f"No single rendered lane serves this session's target "
            f"(rendered lanes: {_lane_roster_text(situation)})."
        )
    # Last, directly above the listing it owns: the warning leads, and the label
    # under it is what tells the approver whose queue the items below are.
    lines.append(f"{bound_line} — the queue listed below is this lane's.")
    return lines, requested


def _describe_queue_start(
    tool_input: dict, config: dict, hook_input=None, read_record=None
) -> list[str]:
    """Render the start-approval prompt: everything this start would run.

    `queue_start` carries at most a lane id, so without these lines the approver
    would be asked to approve motion with no statement of what moves. A start
    drains the whole queue in order, so the whole queue is the answer.

    On a deployment with two plan lanes it also has to be the RIGHT queue: the
    lane block above the listing names the lane the start is addressed to, and
    the listing is fetched from that lane's bridge. Where no lane can be named —
    none given, one this deployment does not render — there is no queue to show
    that would be honest, and the block says why instead.
    """
    situation = _lane_situation(config, hook_input, read_record)
    lane_lines, lane_key = _lane_start_lines(situation, tool_input)
    lines = list(lane_lines)
    lines.append(
        "Starting the queue runs EVERY pending item below, in order — "
        "not only the most recently added one."
    )
    if situation["multi"] and lane_key is None:
        lines.append(
            "Queue contents: not shown — this prompt cannot tell which lane's queue "
            "would start, and listing the other lane's would describe a different "
            "machine."
        )
        return lines

    base_url = _lane_bridge_url(situation, lane_key, config)
    if base_url is None:
        lines.append(_unaddressable_lane_line(lane_key))
        return lines
    snapshot = _queue_snapshot(base_url)
    if snapshot is None:
        lines.append(
            "Queue contents: unavailable (the bridge could not be reached) — "
            "approving means starting a queue nobody here can see."
        )
        return lines
    lines.extend(_queue_item_lines(snapshot, base_url))
    return lines


def _describe_queue_stop(
    tool_input: dict, config: dict, hook_input=None, read_record=None
) -> list[str]:
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


def _describe_stop_run(
    tool_input: dict, config: dict, hook_input=None, read_record=None
) -> list[str]:
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


# ---------------------------------------------------------------------------
# Write posture: is this deployment armed for the target THIS call would act on
# ---------------------------------------------------------------------------
# Posture is per target (`control_system.connector.<type>.writes_enabled` over
# `control_system.writes_enabled`), so "are writes disabled here" is only
# answerable once the call has been pointed at a target. The rules themselves
# live in `osprey_target_state.writes_posture`, shared with `osprey_writes_check`
# so the deny and the missing prompt can never disagree about one deployment.
#
# What this file owns is the TARGETING: which target a given call would act on.
#
# * every ordinary write tool follows the SESSION target, from the state file;
# * a queue START binds to a plan lane instead, and the lane's target is
#   render-time truth in the rendered config — the same map the prompt's lane
#   lines read. A start naming no lane binds to the single rendered lane where
#   there is only one, which is what the tool's own `_bind_lane` does;
# * every other queue tool is left alone entirely.
#
# THE RULE THIS FILE OBEYS: DEFER IS ONLY SAFE WHERE A REFUSAL IS GUARANTEED.
# Deferring emits no decision, so the call proceeds unless some other layer
# refuses it. There are exactly two guarantors, and they cover different tools:
#
# * `osprey_writes_check` denies the session-targeted write tools. That deny is
#   the whole reason this short-circuit exists (an "ask" from here reopens
#   `can_use_tool` over it), and it is what makes a defer safe for them;
# * `queue_start` refuses in-tool, before its bridge is called, whenever the
#   bound lane's target is not armed. `writes_check` deliberately allows every
#   lane-addressed tool, so for a start THAT gate is the only guarantor — and it
#   only fires once the lane is placed. A start whose lane cannot be placed
#   therefore keeps its prompt rather than deferring into a gap.

#: Queue tools are addressed by lane rather than by the session target, and only
#: a START names one. `queue_add` composes onto an idle queue and starts nothing
#: (the server withholds the launch token, and the bridge refuses
#: `launch_token_required` the moment the queue drains), and a plain stop is
#: ungated everywhere by design. Neither has a target to resolve, and neither is
#: ever short-circuited: a prompt that vanished on a tool nobody denies would be
#: a gate removed rather than a gate applied.
_QUEUE_TOOL_PREFIX = "queue_"
_LANE_ADDRESSED_TOOL = "queue_start"


def _unanswerable_posture(short_name):
    """The posture to assume when the question could not be answered at all.

    Not armed — a defer — wherever `osprey_writes_check` is guaranteed to deny,
    so the two layers compose into a refusal rather than into a gap. The
    lane-addressed start is the exception: `writes_check` allows every
    lane-addressed tool, and the tool's own gate cannot fire on a lane nobody
    placed, so a defer there would be an allow with no prompt at all.
    """
    return None if short_name == _LANE_ADDRESSED_TOOL else False


def _lane_posture(config, section, tool_input):
    """Posture for a queue start, or ``None`` when its lane cannot be placed.

    The lane map is render-time truth from the config's `services.<lane>`
    blocks. A start that names no lane still binds to the one lane a single-lane
    deployment has — that is what `_bind_lane` does server-side, so it is the
    lane the bridge will use and its posture is the honest answer. A start that
    names no lane on a two-lane render, or one naming a lane this deployment
    does not carry, is genuinely unplaced and keeps its prompt.
    """
    lanes = _rendered_lanes(config)
    lane = tool_input.get("lane") if isinstance(tool_input, dict) else None
    if not lane:
        if len(lanes) != 1:
            return None
        target = lanes[0][1]
    else:
        target = next((declared for key, declared in lanes if key == lane), None)
    if target not in (_TARGET_LIVE, _TARGET_VA):
        # Includes the unplaced lane. A `services.<lane>.target` naming
        # something else is a config this hook cannot read as a target, and
        # `writes_posture` would answer it from the deployment-wide key — a
        # posture for a machine nobody identified.
        return None
    return _target_state.writes_posture(section, target)


def _session_posture(section, hook_input):
    """Posture for the target this SESSION is pointed at."""
    result = _target_state.read_session_target(hook_input)
    if _target_state.is_baseline(result):
        # A baseline fallback still NAMES the deployment's baseline target, and
        # answering for it would state a posture for a session that may have
        # switched away from it. The posture every target this session could
        # REACH agrees on is the only one that cannot become a guess in favour
        # of hardware — the same call `osprey_writes_check` makes on the same
        # shape, so the two cannot part company over an unidentified session.
        return _target_state.most_restrictive_posture(section)
    return _target_state.writes_posture(section, result.get("target"))


def _call_write_posture(config, tool_name, short_name, tool_input, hook_input):
    """Whether writes are armed for what this call would touch. Never raises.

    ``True`` armed, ``False`` explicitly not armed, and ``None`` for every shape
    that must keep normal approval: a tool that is not a write, a readonly
    execution, a queue tool this hook does not target, a start whose lane cannot
    be placed, and a config that states no posture anywhere.

    Everything the answer depends on — the generated write-tool list, the config
    section, the state file, the lane map — sits inside one ``try``, and a
    failure resolves the way :func:`_unanswerable_posture` describes rather than
    propagating: an uncaught exception here would exit non-zero with no JSON and
    let the call through with neither a prompt nor a decision.
    """
    try:
        if not is_write_tool(tool_name, write_tools()):
            return None
        if not is_write_call(tool_name, tool_input, short_name):
            return None
        if short_name.startswith(_QUEUE_TOOL_PREFIX) and short_name != _LANE_ADDRESSED_TOOL:
            return None

        if _target_state is None:
            # The reader is also where the posture RULES live, so a render
            # without it cannot answer this at all. The sibling is rendered
            # beside this file by the same build, so this is theoretical rather
            # than an upgrade path.
            return _unanswerable_posture(short_name)

        section = config.get("control_system") if isinstance(config, dict) else None
        if short_name == _LANE_ADDRESSED_TOOL:
            return _lane_posture(config, section, tool_input)
        return _session_posture(section, hook_input)
    except Exception:
        return _unanswerable_posture(short_name)


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
    short_name = short_tool_name(tool_name, _prefixes)

    config = load_osprey_config(hook_input)
    approval_config = config.get("approval", {})

    # Deterministic short-circuit when this deployment is NOT armed for the
    # target this call would act on. Empirically (Claude Code SDK 2.x, not
    # source-verified): PreToolUse hook-decision aggregation does NOT honour
    # writes_check's JSON deny if this hook ALSO emits an "ask" decision —
    # aggregation appears to be any-ask-wins, not deny-dominates, when multiple
    # hooks return decisions for the same tool call. Without the short-circuit
    # `can_use_tool` fires and the unarmed invariant is violated.
    #
    # How much this defer carries depends on the render, and posture is per
    # target, so `src/osprey/cli/templates/claude_code.py` renders three ways:
    #
    # * NO target may write — every write tool is hard-denied in
    #   `permissions.deny`, which blocks before any PreToolUse hook fires. There
    #   the deny is primary and this defer is belt-and-braces against renderer
    #   drift, such as a stale `settings.json` after a posture flip without
    #   regen;
    # * EVERY target may write — nothing is rendered and nothing defers here;
    # * the targets DISAGREE — settings.json is rendered once, before any
    #   session picks a target, so a static deny would be wrong on the armed
    #   target and a static `ask` would be wrong on the unarmed one. The render
    #   therefore steps aside entirely: it denies nothing and pulls every
    #   writes-check-gated matcher OUT of `ask`. On that render there is no
    #   static gate at all, and this defer together with writes_check's
    #   per-target deny IS the whole gate — for the lane-addressed queue tools
    #   writes_check skips, the bluesky tools' own lane-bound posture re-read is.
    #
    # Three-way, and the third way is the point: a config that states no posture
    # at all falls through to normal approval, so every deployment that predates
    # the per-target key keeps exactly the prompt it has today. Only an explicit
    # `False` defers, and only where a refusal is guaranteed — see
    # `_call_write_posture`, which answers every part of that question.
    if _call_write_posture(config, tool_name, short_name, tool_input, hook_input) is False:
        log_hook(
            "approval",
            hook_input,
            status="defer",
            detail=f"writes_not_armed tool={short_name}",
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
                json.dump(build_approval_output(reason, hook_input), sys.stdout)
                sys.exit(0)

            # Selective execute without write indicators — allow
            log_hook("approval", hook_input, status="allow", detail="execute_selective_readonly")
            json.dump(build_allow_output(), sys.stdout)
            sys.exit(0)

        # For channel_write under selective, treat as always (conservative)
        if short_name == "channel_write":
            channel_list = _channel_write_summary(tool_input)
            reason = f"Channel write: {channel_list or 'unknown'}"
            log_hook("approval", hook_input, status="ask", detail="channel_write_selective")
            json.dump(build_approval_output(reason, hook_input), sys.stdout)
            sys.exit(0)

        # Other tools under selective: treat as always (conservative)
        # Falls through to "always" handling below

    # Always policy (explicit or fallback from selective for non-execute/write tools)
    reason_parts = [
        f"Tool: {short_name}",
        f"Approval policy: {policy}",
    ]
    # One reader for this whole prompt: whatever a describer learns about the
    # session's target, the `Target:` line and the write-approval stamp below are
    # rendered from the same read. It resolves nothing until something asks, so a
    # prompt with no target-aware describer costs exactly what it always did.
    read_record = _record_reader(hook_input)
    if short_name == "execute":
        code = tool_input.get("code", "")
        exec_mode = tool_input.get("execution_mode", "")
        if code.strip():
            gallery_link = _create_pre_execution_notebook(code, exec_mode, config)
            if gallery_link:
                reason_parts.append(f"\nReview notebook: {gallery_link}")
    elif short_name == "channel_write":
        channel_list = _channel_write_summary(tool_input)
        if channel_list:
            reason_parts.append(f"Channels: {channel_list}")
    elif short_name in _QUEUE_DESCRIBERS:
        try:
            reason_parts.extend(
                _QUEUE_DESCRIBERS[short_name](tool_input, config, hook_input, read_record)
            )
        except Exception:
            pass
    elif short_name in _TARGET_DESCRIBERS:
        try:
            reason_parts.extend(
                _TARGET_DESCRIBERS[short_name](tool_input, config, hook_input, read_record)
            )
        except Exception:
            pass

    reason = "\n".join(reason_parts)
    log_hook(
        "approval",
        hook_input,
        status="ask",
        detail=f"policy={policy} tool={short_name}",
    )
    json.dump(build_approval_output(reason, hook_input, read_record), sys.stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()
