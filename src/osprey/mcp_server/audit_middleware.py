"""The MCP audit middleware — every ``tools/call`` recorded, writes clamped.

Osprey's write gating used to live entirely in the ``osprey_writes_check.py``
PreToolUse hook, which is a *client-side* layer: Claude Code runs it, it fails
open by design (an uncaught exception exits non-zero and the tool proceeds), and
nothing outside Claude Code consults it at all. This middleware is the
server-side half. It sits on every framework MCP server a deployment launches,
so the sandbox posture is enforced where the tool actually executes and every
call — refused or not — leaves a record.

**What it decides.** Under the sandbox posture
(``OSPREY_EXECUTION_MODE=readonly``, matched by *value*, never by presence) a
call whose fully-qualified name is in the clamp set is refused before the tool
runs. The clamp set is the rendered ``hook_config.json``'s ``write_tools``
*minus* its ``mixed_read_write_tools`` — the read/write tools, of which
``mcp__python__execute`` is the framework's one, keep their own in-tool clamp
because a *readonly* execution is exactly what a sandboxed session is for.
Refusing them here would take the reads with the writes.

**Where it reads that from.** Exclusively from ``OSPREY_CONFIG``, which
``registry/mcp.py`` sets to ``<root>/build/config.yml`` for every framework
server it launches: the hook config is its sibling at
:data:`HOOK_CONFIG_RELPATH`. Never a project-root resolver (that answers the
repo root, one level up from the render zone) and never a cwd guess — a server
started with no ``OSPREY_CONFIG`` gets the degraded floor and one loud warning
rather than a plausible-looking file from somewhere else.

**Both lists fall together.** A hook config that is missing, unreadable,
malformed, or that parses without either list yields :data:`_FALLBACK_WRITE_TOOLS`
minus :data:`_FALLBACK_MIXED_TOOLS` — never a loaded write list against a floor
exclusion, which would clamp ``execute`` as a pure write tool and break readonly
executions. The stated consequence: on a pre-feature render, which parses
without the mixed key, the clamp is the floor only and a facility's other write
tools stay hook-covered until the operator re-renders. Containers self-heal via
the entrypoint's drift check; bare hosts get the warning naming ``osprey build``.

**Fresh without a restart.** The file is stat'ed once per call and re-parsed
only when :data:`_STAT_FIELDS` moved, so a re-render lands on an open session
instead of waiting for a server restart. A re-parse that *fails* keeps the last
good clamp set **plus** the floor: the previous answer is still the best
evidence available, and adding the floor to it can only narrow.

**Fails closed on an unrecognised prefix.** The subject a record names is
``mcp__<prefix>__<tool>``, where the prefix comes from
:data:`TOOL_PREFIX_ENV` — assigned post-merge by the registry on every launch
path so a server spec cannot pin it. If that prefix is not among the rendered
``server_prefixes`` (which lists every enabled server), this process is a stale
clone, a mis-prefixed one, or a server the render has never heard of; the
rendered write set **plus** the floor is clamped, with one warning. A server
that is *known* but contributes no write tools is not a mismatch: it runs, and
it says nothing.

**A clamp is matched fully-qualified only when the prefix was verified.** The
one branch that proves this process is the server the render is talking about
is "the prefix env is set *and* ``mcp__<prefix>__`` is among the rendered
``server_prefixes``". Only there is membership an exact match on the
fully-qualified name, so two servers cannot collide on a shared tool name.
Everywhere else — a degraded parse, a prefix the render does not list, no
prefix at all — the qualification is exactly what is untrustworthy, so the
*bare* tool name is matched against each clamp entry's own bare half.
Without that, every fail-closed branch is a no-op for any server whose rendered
name is not one the clamp entries happen to spell: an ``extends`` clone is
launched with its own name as the prefix, so ``mcp__controls_ring__`` qualifies
nothing the floor names. The cost is bounded over-refusal — a read tool sharing
a bare name with some write tool — under the sandbox posture, in a state this
module already calls a framework bug.

**The innermost recorder owns the decision.** A tool's own gates run inside
this layer, and one of them — the executor's session-posture clamp — files its
own record on the ``executor`` surface. When it does, it leaves a marker
(:mod:`osprey.audit.dedup`) and this layer defers: no second record, and in
particular no ``allowed`` stamped on a call an inner guard refused while still
returning a successful result. Same process, same call, cleared on the way in
and out; a marker inherited across a ``fork`` is not believed. The deferral is
transitive — the marker is re-asserted once the scope closes — so a layer
stacked outside this one defers to the same answer.

**Nothing here may cost the call it records.** Records go through
:func:`~osprey.audit.writer.record`, which builds the envelope and appends it
inside its own never-raises boundary; the one place this module could still
raise on its own (reading the config) is wrapped. The clamp *decision* is
deliberately not: an internal error there costs the call rather than becoming
an allowed write. A refusal that could not be audited is still a refusal.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastmcp.exceptions import ToolError
from fastmcp.server.middleware.middleware import CallNext, Middleware, MiddlewareContext

from osprey.audit import posture
from osprey.audit.dedup import decision_scope, mark_recorded, recorded_decision
from osprey.audit.envelope import DECISION_ALLOWED, DECISION_REFUSED
from osprey.audit.writer import record
from osprey.mcp_server.errors import make_error

if TYPE_CHECKING:  # pragma: no cover - typing only
    import mcp.types as mt
    from fastmcp.tools.base import ToolResult

logger = logging.getLogger(__name__)

__all__ = [
    "AuditMiddleware",
    "CLAMP_SOURCE_FLOOR",
    "CLAMP_SOURCE_LOADED",
    "CLAMP_SOURCE_UNVERIFIED",
    "CONFIG_ENV",
    "HOOK_CONFIG_RELPATH",
    "POSTURE_ENV",
    "POSTURE_SANDBOX",
    "POSTURE_SESSION_ENV",
    "POSTURE_SOURCE_ENV",
    "POSTURE_WRITES",
    "REASON_POSTURE",
    "REASON_TOOL_CALL",
    "REASON_TOOL_ERROR",
    "SANDBOX_MODE",
    "SURFACE_UNPREFIXED",
    "TOOL_PREFIX_ENV",
    "hook_config_path",
    "reset_audit_state",
]

# --------------------------------------------------------------------------
# The environment this process is told about itself
# --------------------------------------------------------------------------

#: The rendered config the launching registry hands every framework server. The
#: hook config is found relative to *this*, and to nothing else.
CONFIG_ENV: str = "OSPREY_CONFIG"

#: Where the render puts the hook config, relative to the rendered config's own
#: directory (``build/``). ``osprey build`` writes both in one pass.
HOOK_CONFIG_RELPATH: str = ".claude/hooks/hook_config.json"

#: This server's name, for composing the ``mcp__<name>__<tool>`` subject.
#: Re-spelled rather than imported from ``osprey.registry.mcp``: the registry is
#: the render/launch side and a running server should not import it to learn its
#: own name. Pinned against the registry by a test in this module's test file,
#: the same way the registry pins the audit-writer marker it cannot import.
TOOL_PREFIX_ENV: str = "OSPREY_MCP_TOOL_PREFIX"

#: The session posture, its provenance and its posture-store key, and the two
#: ledger spellings — :mod:`osprey.audit.posture`'s, re-exported under this
#: module's names so the wiring and the tests keep addressing them here.
POSTURE_ENV: str = posture.POSTURE_ENV_VAR
POSTURE_SOURCE_ENV: str = posture.POSTURE_SOURCE_ENV_VAR
POSTURE_SESSION_ENV: str = posture.POSTURE_SESSION_ENV_VAR
SANDBOX_MODE: str = posture.SANDBOX_MODE
POSTURE_SANDBOX: str = posture.POSTURE_SANDBOX
POSTURE_WRITES: str = posture.POSTURE_WRITES

# --------------------------------------------------------------------------
# Floors
# --------------------------------------------------------------------------

#: The write tools clamped when the render cannot be read. Deliberately the
#: same literal as ``osprey_writes_check.py``'s floor of the same name — the two
#: layers must not disagree about what a degraded deployment refuses — and
#: pinned against it by a test rather than imported, because the hook ships as a
#: template file that is copied into a project, not as an importable module.
#:
#: It covers **every** write-gated tool the framework ships, not the ones a
#: default render happens to enable. ``bluesky``'s arming pair used to be left
#: out on the grounds that bluesky is opt-in — which has it backwards: a
#: deployment that opted in is precisely the one whose sandboxed session can
#: call ``queue_add``, and a degraded render is exactly when nothing else is
#: left to refuse it. ``registry.mcp.framework_write_tools()`` is the derived
#: answer this list and the hook's are both pinned against
#: (``tests/registry/test_mixed_floor_driftguard.py``), so the next framework
#: write tool cannot strand the floor the same way.
_FALLBACK_WRITE_TOOLS: list[str] = [
    "mcp__bluesky__queue_add",
    "mcp__bluesky__queue_start",
    "mcp__controls__channel_write",
    "mcp__python__execute",
]

#: The read/write tools excluded from the floor clamp. Task 1.8's drift guard
#: asserts this list against ``registry.mcp.framework_mixed_read_write_tools()``,
#: which derives the same names from the ``_WRITES_CHECK`` matchers of every
#: ``MIXED_READ_WRITE_TEMPLATES`` server — so the canonical source stays the
#: registry and this stays the copy that must match it.
#:
#: It is applied to :data:`_FALLBACK_WRITE_TOOLS` and never to a loaded write
#: list: the two always degrade together. A loaded write list minus this floor
#: would clamp whatever the deployment renders as mixed, and a readonly
#: ``execute`` would stop working under exactly the posture it exists for.
_FALLBACK_MIXED_TOOLS: list[str] = ["mcp__python__execute"]

#: The clamp set used whenever the render cannot be trusted.
_FLOOR_CLAMP: frozenset[str] = frozenset(_FALLBACK_WRITE_TOOLS) - frozenset(_FALLBACK_MIXED_TOOLS)

# --------------------------------------------------------------------------
# Record vocabulary
# --------------------------------------------------------------------------

#: The surface a record names when this process was never told its own name.
#: A real name would be a guess; ``mcp`` at least files the record somewhere an
#: operator will look, next to the warning that says why it is unnamed.
SURFACE_UNPREFIXED: str = "mcp"

#: Reasons, in the ledger's short machine-ish vocabulary.
#:
#: :data:`REASON_POSTURE` is the one word three layers spell for the same
#: refusal — this middleware, the executor's in-tool session clamp
#: (``_execution_gates``), and the client-side ``osprey_writes_check.py`` hook.
#: A sandboxed session's refusals must join on one spelling, or
#: ``grep '"reason": "posture"'`` over an identity's ledgers answers "what did
#: the sandbox posture refuse?" with only the layer that happened to be asked.
#: A cross-layer test pins all three (the hook by AST, since it ships as a
#: template file that imports nothing from ``osprey``).
REASON_TOOL_CALL: str = "tool_call"
REASON_POSTURE: str = "posture"
REASON_TOOL_ERROR: str = "tool_error"

#: What a clamped record's ``detail`` says about *which* set refused it. The
#: distinction is the whole forensic value of a degraded refusal: "the render
#: says this is a write tool" and "we could not read the render" are different
#: findings for the operator reading the ledger.
CLAMP_SOURCE_LOADED: str = "clamp=hook_config"
CLAMP_SOURCE_FLOOR: str = "clamp=fallback_floor"

#: The third case, between the two: the render parsed cleanly but says nothing
#: about *this* server, so its write set is clamped together with the floor and
#: matched by bare name. "We could not read the render" and "the render does not
#: know this server" are different findings, and the second is the one that
#: means a stale clone or a stale render — worth its own spelling rather than
#: being filed under the floor. Which of the two unverified causes it was (no
#: prefix at all, or a prefix the render does not list) is in the log, keyed
#: ``no_prefix`` and ``prefix_mismatch:<prefix>``.
CLAMP_SOURCE_UNVERIFIED: str = "clamp=unverified_prefix"

# --------------------------------------------------------------------------
# Parsed state
# --------------------------------------------------------------------------


#: The ``os.stat`` fields the freshness check compares, in key order.
#:
#: Not mtime and size alone. A replacement can preserve both — a restore with
#: ``cp -p``, ``rsync -t`` or ``tar -p``, an image-layer or bind-mount swap, an
#: in-place editor on a filesystem with coarse mtime — and the direction that
#: matters is a re-render that *adds* a write tool: the running server would
#: keep clamping the old, narrower set for the life of the process, with no
#: warning, while the module advertises freshness without a restart.
#: ``st_ino`` moves on any atomic rename (the ordinary way a file is replaced);
#: ``st_ctime_ns`` moves on any change to the inode, including the ``utime``
#: that would put a forged mtime back.
_STAT_FIELDS: tuple[str, ...] = ("st_mtime_ns", "st_size", "st_ino", "st_ctime_ns")

#: ``(str(path), *_STAT_FIELDS values)``.
_StatKey = tuple[Any, ...]


@dataclass(frozen=True)
class _ClampState:
    """One resolved answer about what this server clamps, and how sure it is."""

    clamp: frozenset[str]
    server_prefixes: frozenset[str]
    degraded: bool
    stat_key: _StatKey | None


#: The state the last call resolved, and the last state that came from a hook
#: config that actually parsed. Module-level rather than instance-level: one
#: process hosts one server, and the middleware is constructed by the wiring
#: without arguments, so a per-instance cache would only mean a second
#: construction silently re-reading everything.
_STATE: _ClampState | None = None
_LAST_GOOD_CLAMP: frozenset[str] = frozenset()

#: Warning keys already emitted. Every degrade in this module is a *standing*
#: condition — an unset env var, a stale render — so warning per call would put
#: one line in the log per tool call for as long as it lasts.
_WARNED: set[str] = set()


def reset_audit_state() -> None:
    """Forget the parsed hook config and the warnings already emitted.

    A test seam, and the only supported way to make the next call re-read
    everything. Nothing in the server calls it: freshness on the running path is
    the per-call stat, not an invalidation someone has to remember to perform.
    """
    global _STATE, _LAST_GOOD_CLAMP
    _STATE = None
    _LAST_GOOD_CLAMP = frozenset()
    _WARNED.clear()


def _warn_once(key: str, message: str, *args: Any) -> None:
    """Log *message* at WARNING the first time *key* is seen."""
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(message, *args)


def hook_config_path() -> Path | None:
    """The rendered ``hook_config.json`` this server reads, or ``None``.

    ``None`` means :data:`CONFIG_ENV` is unset, blank, or not absolute — the
    caller degrades to the floor. Deliberately not a resolver:
    ``resolve_project_root`` answers the repo root, one directory above the
    render zone, and a fallback to it would read a *different* deployment's hook
    config while looking entirely healthy. A *relative* ``OSPREY_CONFIG`` does
    the same thing by a shorter route — it resolves against whatever directory
    this server happens to have been started in — so it is refused here rather
    than turned into a plausible-looking path that merely fails to open. Every
    launch path sets an absolute ``{project_root}/build/config.yml``.
    """
    configured = (os.environ.get(CONFIG_ENV) or "").strip()
    if not configured:
        return None
    path = Path(configured)
    if not path.is_absolute():
        _warn_once(
            "config_not_absolute",
            "%s is %r, which is not an absolute path, so the MCP audit middleware will not "
            "guess a hook config from the working directory and is clamping the fallback "
            "write floor. Every framework server is launched with an absolute %s.",
            CONFIG_ENV,
            configured,
            CONFIG_ENV,
        )
        return None
    return path.parent / HOOK_CONFIG_RELPATH


def _read_hook_config(path: Path) -> dict | None:
    """Parse *path*, or ``None`` if it cannot be read as a JSON object."""
    try:
        with open(path, encoding="utf-8") as handle:
            parsed = json.load(handle)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _string_list(value: Any) -> list[str] | None:
    """*value* as a list of non-empty strings, or ``None`` if it is not exactly that.

    A list carrying anything that is not a non-empty string is rejected whole
    rather than filtered: ``[null, 123]`` is the same evidence about the render
    as ``"everything"`` is, and filtering it would hand the caller an EMPTY
    clamp with ``degraded=False`` and no warning — the widest possible answer,
    arrived at quietly.
    """
    if not isinstance(value, list):
        return None
    items = [item for item in value if isinstance(item, str) and item]
    return items if len(items) == len(value) else None


def _degraded_state(stat_key: _StatKey | None) -> _ClampState:
    """The clamp for a render we could not read: last good, plus the floor.

    Adding the floor to the last good set can only narrow, which is the right
    direction to move when the evidence just got worse. With no last good set —
    the first call after a bad render — it is the floor alone.
    """
    return _ClampState(
        clamp=_LAST_GOOD_CLAMP | _FLOOR_CLAMP,
        server_prefixes=frozenset(),
        degraded=True,
        stat_key=stat_key,
    )


def _resolve_state() -> _ClampState:
    """The clamp state for this call, re-parsing only when the file moved."""
    global _STATE, _LAST_GOOD_CLAMP

    path = hook_config_path()
    if path is None:
        # `hook_config_path` has already warned if the value was present but
        # unusable; only claim "unset" when it actually is.
        if (os.environ.get(CONFIG_ENV) or "").strip():
            return _degraded_state(None)
        _warn_once(
            "no_config_env",
            "%s is unset, so the MCP audit middleware cannot locate %s and is clamping "
            "the fallback write floor. Every framework server is launched with %s set; "
            "an unset one means this server was started outside `osprey`.",
            CONFIG_ENV,
            HOOK_CONFIG_RELPATH,
            CONFIG_ENV,
        )
        return _degraded_state(None)

    try:
        stat = os.stat(path)
        stat_key: _StatKey | None = (
            str(path),
            *(getattr(stat, field) for field in _STAT_FIELDS),
        )
    except OSError:
        stat_key = None

    if stat_key is not None and _STATE is not None and _STATE.stat_key == stat_key:
        return _STATE

    if stat_key is None:
        _warn_once(
            "unreadable",
            "Could not read %s; the MCP audit middleware is clamping the last known write "
            "set plus the fallback floor. Re-render it with `osprey build`.",
            path,
        )
        state = _degraded_state(None)
        _STATE = state
        return state

    parsed = _read_hook_config(path)
    write_tools = _string_list((parsed or {}).get("write_tools"))
    mixed_tools = _string_list((parsed or {}).get("mixed_read_write_tools"))

    if parsed is None or write_tools is None or mixed_tools is None:
        # A file that parsed but says nothing usable is a stale render, which
        # `osprey build` fixes; one that did not parse is a broken one. Both
        # degrade the same way, and both are worth naming the remedy for.
        _warn_once(
            "unusable_lists",
            "%s does not carry usable 'write_tools' and 'mixed_read_write_tools' lists; "
            "the MCP audit middleware is clamping the fallback write floor only. "
            "Re-render it with `osprey build`.",
            path,
        )
        state = _degraded_state(stat_key)
        _STATE = state
        return state

    clamp = frozenset(write_tools) - frozenset(mixed_tools)
    _LAST_GOOD_CLAMP = clamp
    state = _ClampState(
        clamp=clamp,
        server_prefixes=frozenset(_string_list(parsed.get("server_prefixes")) or ()),
        degraded=False,
        stat_key=stat_key,
    )
    _STATE = state
    return state


# --------------------------------------------------------------------------
# This process's identity and posture
# --------------------------------------------------------------------------


def _tool_prefix() -> str | None:
    """This server's rendered name, or ``None`` with one warning."""
    prefix = (os.environ.get(TOOL_PREFIX_ENV) or "").strip()
    if prefix:
        return prefix
    _warn_once(
        "no_prefix",
        "%s is unset, so the MCP audit middleware records tool names unqualified and "
        "matches the fallback write floor by bare tool name. The registry assigns it on "
        "every launch path; an unset one means this server was not launched by `osprey`.",
        TOOL_PREFIX_ENV,
    )
    return None


def _clamp_for(state: _ClampState, prefix: str | None) -> tuple[frozenset[str], str, bool]:
    """The clamp to apply, what a refusal by it says it was, and whether it is *verified*.

    Verified means one thing only: the prefix env is set **and** the render
    lists it, so this process is provably the server the loaded write set is
    about and membership can be an exact match on the fully-qualified name. A
    *known* server that contributes no write tools is still verified — no write
    tools is the ordinary shape of a read-only server, not a mismatch — so it
    takes the loaded (possibly empty) set and says nothing.

    Every other branch is unverified and clamps the widest defensible set: the
    render's own write set (or the last good one) **plus** the floor. Throwing
    the loaded names away at the moment the evidence got worse would make "fail
    closed" strictly weaker than the healthy path — a render naming
    ``mcp__sitectl__set_value`` would clamp it for ``sitectl`` but not for a
    server the render does not recognise, which is the more suspicious of the
    two. :func:`_is_clamped` matches that set by bare tool name, because the
    qualification is precisely what could not be verified.
    """
    if state.degraded:
        return state.clamp, CLAMP_SOURCE_FLOOR, False
    if prefix is not None and f"mcp__{prefix}__" in state.server_prefixes:
        return state.clamp, CLAMP_SOURCE_LOADED, True
    if prefix is not None:
        _warn_once(
            f"prefix_mismatch:{prefix}",
            "This server's tool prefix %r is not among the server_prefixes in %s, so the "
            "MCP audit middleware is clamping the rendered write set plus the fallback "
            "floor, matched by bare tool name. Re-render with `osprey build`, or check "
            "that the server's name matches the render.",
            prefix,
            HOOK_CONFIG_RELPATH,
        )
    return state.clamp | _FLOOR_CLAMP, CLAMP_SOURCE_UNVERIFIED, False


def _is_clamped(clamp: frozenset[str], subject: str, tool: str, *, verified: bool) -> bool:
    """Whether *subject* is refused by *clamp*.

    On the verified path an exact match on the fully-qualified name, so two
    servers cannot collide on a shared tool name. Otherwise the bare tool name
    is matched against each entry's own bare half: with no trustworthy prefix a
    fully-qualified test is a test of the very thing that is broken, and it
    would quietly turn every fail-closed branch into a no-op for any server
    whose rendered name is not one the clamp entries happen to spell — which is
    every ``extends`` clone.
    """
    if subject in clamp:
        return True
    if verified:
        return False
    return any(entry.endswith(f"__{tool}") for entry in clamp)


def _record(
    *,
    surface: str,
    subject: str,
    decision: str,
    reason: str,
    detail: str | None = None,
) -> None:
    """File one record, and never let filing it cost the call it describes.

    :func:`~osprey.audit.writer.record` is the never-raises boundary: it builds
    the envelope inside it, fills ``actor`` from the identity ladder, and
    swallows everything the construction or the append can raise.
    """
    record(
        surface=surface,
        posture=posture.posture(),
        posture_source=posture.posture_source(),
        session=posture.posture_session(),
        subject=subject,
        decision=decision,
        reason=reason,
        detail=detail,
    )


# --------------------------------------------------------------------------
# The middleware
# --------------------------------------------------------------------------


class AuditMiddleware(Middleware):
    """Audit every ``tools/call``, and clamp write tools under the sandbox posture.

    Constructed with no arguments: everything it needs is in the environment the
    registry launched this server with, so the wiring in
    ``startup.run_mcp_server`` stays a single ``add_middleware`` call and there
    is no second place a deployment could configure the clamp differently from
    the render.

    Stateless per instance — the parsed hook config lives at module level (see
    :func:`reset_audit_state`) — and it defers *transitively*: when an inner
    layer has already recorded the call, this one re-asserts that marker on the
    way out (see :mod:`osprey.audit.dedup`), so a second instance stacked
    outside it defers to the same answer instead of stamping ``allowed`` over a
    refusal. ``startup.run_mcp_server`` installs exactly one; the transitivity
    is there so that any layer which one day wraps this one inherits the
    invariant rather than breaking it.
    """

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        """Record the call, refuse it if the posture says so, else pass it on."""
        tool = getattr(context.message, "name", "") or ""
        prefix = _tool_prefix()
        subject = f"mcp__{prefix}__{tool}" if prefix else tool
        surface = prefix or SURFACE_UNPREFIXED

        # Deliberately NOT wrapped in a try/except: an internal error here must
        # not become an allowed write. `_record` swallows everything because a
        # lost record is not a lost decision; the decision itself fails closed
        # by costing the call, and a test pins that.
        clamp, clamp_source, verified = _clamp_for(_resolve_state(), prefix)
        if posture.posture() == POSTURE_SANDBOX and _is_clamped(
            clamp, subject, tool, verified=verified
        ):
            _record(
                surface=surface,
                subject=subject,
                decision=DECISION_REFUSED,
                reason=REASON_POSTURE,
                detail=clamp_source,
            )
            # `make_error` raises: fastmcp turns a raised ToolError into a
            # CallToolResult with isError=True and the message verbatim, which
            # returning a result directly does not.
            suggestions = [
                "Switch the session to writes posture from the terminal card; "
                "config.yml is not the gate here.",
            ]
            if clamp_source != CLAMP_SOURCE_LOADED:
                # Off the verified path the match is by bare tool name, so a
                # read tool sharing a name with some write tool lands here too.
                # Say why, and what fixes it, rather than let a refusal of a
                # read look like the posture doing its job.
                suggestions.append(
                    "This clamp is degraded (the rendered hook config could not be read or "
                    "used, or this server is not one it names), so tools are matched by bare "
                    "name; re-render with `osprey build`, and relaunch this server under "
                    "`osprey` if its name or launch changed."
                )
            make_error(
                "safety_error",
                f"{tool} is refused: this terminal session is in the sandbox posture.",
                suggestions,
            )

        # One dedup scope per call: the innermost layer that recorded a
        # decision owns it, and this layer defers. Entering clears any marker
        # the task carried in and leaving drops the one set inside, so a
        # refusal that propagates out of `call_next` cannot silence the next
        # call on the same task. See `osprey.audit.dedup`.
        #
        # `outward` is what a layer OUTSIDE this one should defer to, as
        # (decision, reason, stored). The scope resets the marker on the way
        # out, so deferring here would otherwise stop one layer deep and a
        # second AuditMiddleware — or any future layer wrapping this one —
        # would stamp `allowed` over the refusal this one just honoured. The
        # `finally` re-asserts it once the scope has closed.
        outward: tuple[str, str, bool] | None = None
        try:
            with decision_scope():
                try:
                    result = await call_next(context)
                except ToolError:
                    # The tool refused for its own reasons — a limits violation, a
                    # deployment kill switch, an in-tool posture clamp. That is a
                    # refusal of the same call and belongs in the same ledger,
                    # unless the layer that refused already filed it. The message
                    # is deliberately NOT copied into the record: a ToolError
                    # carries whatever the tool put in it, and the ledger holds
                    # identifiers, not payloads.
                    inner = recorded_decision()
                    if inner is not None and inner.decision != DECISION_REFUSED:
                        # Deferred to all the same — the inner layer is still the
                        # one that decided — but a layer that recorded `allowed`
                        # and then raised has a bug worth naming.
                        logger.warning(
                            "%s recorded %r/%r and then raised a ToolError; the ledger keeps "
                            "the inner record",
                            subject,
                            inner.decision,
                            inner.reason,
                        )
                    if inner is not None and inner.stored:
                        outward = (inner.decision, inner.reason, inner.stored)
                    else:
                        # Either nothing inside recorded, or something did and
                        # its write did not land. Both leave the ledger with no
                        # line for a call that raised, and this record says
                        # `refused` — which contradicts no inner decision worth
                        # protecting. Filing it is the difference between a
                        # duplicate and a silence.
                        _record(
                            surface=surface,
                            subject=subject,
                            decision=DECISION_REFUSED,
                            reason=REASON_TOOL_ERROR,
                        )
                        outward = (DECISION_REFUSED, REASON_TOOL_ERROR, True)
                    raise

                inner = recorded_decision()
                if inner is not None:
                    outward = (inner.decision, inner.reason, inner.stored)
        finally:
            if outward is not None:
                decision, reason, stored = outward
                mark_recorded(decision, reason, stored=stored)

        if inner is not None:
            # The call succeeded on the wire and an inner layer already
            # recorded what happened. This is the runtime-guard shape: the
            # guard refused mid-run, the tool reported it and returned the
            # output the script legitimately produced. Stamping `allowed` on
            # top of that would put a record in the ledger that contradicts a
            # refusal which really happened.
            #
            # Deferred even when the inner write did NOT land (`stored=False`),
            # which is the one place this path is deliberately asymmetric with
            # the `ToolError` branch above: there, the middleware files its own
            # `refused`/`tool_error`, because the call raising is a first-person
            # observation this layer made and can honestly sign. Here there is
            # no such statement left to make. The call succeeded, so `allowed`
            # is false; `refused` is a decision this layer did not take; and
            # copying the inner marker's decision and reason would file another
            # layer's finding under THIS surface and this layer's vocabulary —
            # a `runtime_guard` line on `mcp__<prefix>` rather than on
            # `executor`, without the `source` the executor surface carries and
            # that the refusal is actually about. An operator grepping the
            # executor ledger would still find nothing, and would now also find
            # a lookalike filed somewhere it never happened. Silence plus the
            # marker's `stored=False`, which has already travelled outward, is
            # the honest answer; `test_a_successful_call_over_an_unstored_marker`
            # `_files_no_substitute_record` pins it against the ToolError twin.
            logger.debug(
                "Deferring the audit record for %s to the inner layer (%s/%s)",
                subject,
                inner.decision,
                inner.reason,
            )
            return result

        _record(
            surface=surface,
            subject=subject,
            decision=DECISION_ALLOWED,
            reason=REASON_TOOL_CALL,
        )
        return result
