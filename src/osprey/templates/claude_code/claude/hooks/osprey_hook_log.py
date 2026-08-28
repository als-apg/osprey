"""Shared hook logging utility. No external dependencies."""

import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

_hook_config_cache = None


def load_hook_config():
    """Load hook_config.json (generated at regen time) with prefix lists.

    Lookup order:
    1. ``OSPREY_HOOK_CONFIG`` env var (for tests)
    2. ``hook_config.json`` next to this script (deployed projects)

    Returns ``{}`` if the file is missing or unparseable — hooks degrade
    gracefully to "match nothing" rather than crashing.
    """
    global _hook_config_cache
    if _hook_config_cache is not None:
        return _hook_config_cache
    config_path = os.environ.get("OSPREY_HOOK_CONFIG")
    if config_path is None:
        config_path = Path(__file__).parent / "hook_config.json"
    try:
        with open(config_path) as f:
            _hook_config_cache = json.load(f)
    except Exception:
        _hook_config_cache = {}
    return _hook_config_cache


#: The one wildcard form a ``write_tools`` entry may use. ``settings.json``
#: PreToolUse matchers spell a whole self-gated MCP server as
#: ``mcp__<server>__.*``, and ``hook_config.json`` carries those matchers
#: verbatim, so that suffix is the only regex syntax that can reach here. It is
#: compared as a literal prefix rather than compiled: a hook may run under a
#: bare ``python3``, and a hand-edited matcher must not make a write gate raise.
#: Honouring it is a behaviour change for the deployments that carry one: the
#: gates used to compare entries for equality, so a whole-server matcher matched
#: nothing at all, and a server declaring ``writes_check`` at server level now
#: has every one of its tools gated where before it had none of them.
_MATCHER_WILDCARD = ".*"

#: The one write tool whose calls are not all writes: the python server's
#: executor, which carries an ``execution_mode``. A SHORT name, because an
#: ``extends`` clone of that server renames the prefix and keeps the tool.
_EXECUTE_TOOL = "execute"

#: The write tools refused when the generated list cannot be read. Fail-closed:
#: a hook that gated nothing because ``hook_config.json`` was missing would be
#: a write gate that quietly stopped being one.
#:
#: It covers EVERY write-gated tool the framework ships, not the ones a default
#: render happens to enable: the bluesky plan-queue arming pair used to be left
#: out because bluesky is opt-in, which has it backwards — a deployment that
#: opted in is precisely the one whose sandboxed session can reach those tools,
#: and a degraded render is exactly when nothing else is left to refuse them.
#: Facility-declared extras are absent by construction — they are only knowable
#: from the generated list — so a fallback run gates less than a real render,
#: never more.
#:
#: The one floor every write gate shares: ``osprey_writes_check`` and
#: ``osprey_approval`` read it through :func:`write_tools`, the MCP audit
#: middleware and ``osprey.agent_runner.write_tools`` carry copies of the same
#: literal (this file ships as a copied-in project template run by a bare
#: ``python3``, so nothing can import it), and all of them are pinned against
#: ``registry.mcp.framework_write_tools()`` by
#: ``tests/registry/test_mixed_floor_driftguard.py`` so registry growth cannot
#: strand any copy.
FALLBACK_WRITE_TOOLS = [
    "mcp__bluesky__queue_add",
    "mcp__bluesky__queue_start",
    "mcp__controls__channel_write",
    "mcp__python__execute",
    "mcp__python__execute_file",
]


def write_tools():
    """The write-tool matchers this deployment generated, or the fallback.

    One accessor so every write gate reads the same list from the same place
    with the same fallback. Two gates disagreeing about which tools are writes
    is one of them refusing a call the other waves through.
    """
    return load_hook_config().get("write_tools", FALLBACK_WRITE_TOOLS)


def short_tool_name(tool_name, prefixes):
    """*tool_name* with its MCP server prefix stripped.

    The LONGEST matching prefix wins, since one server prefix can be a prefix of
    another and stripping the shorter one would leave half a server name glued
    to the tool. A name matching none of *prefixes* falls back to the
    ``mcp__<server>__<tool>`` shape the prefix lists are themselves generated
    from, and a name in no MCP shape at all is already its own short name.

    Non-string entries are skipped rather than raising, for the same reason
    :func:`is_write_tool` skips them: the lists are generated into a deployment
    and a gate must survive a malformed one.

    Args:
        tool_name: The full tool name as the hook payload carries it.
        prefixes: Server prefixes to consider — ``server_prefixes``,
            ``approval_prefixes``, or both, as the caller's question needs.
    """
    matched = [
        prefix
        for prefix in prefixes or ()
        if isinstance(prefix, str) and tool_name.startswith(prefix)
    ]
    if matched:
        return tool_name[len(max(matched, key=len)) :]
    parts = tool_name.split("__")
    if tool_name.startswith("mcp__") and len(parts) > 2:
        return "__".join(parts[2:])
    return tool_name


def is_write_tool(tool_name, write_tools):
    """Is *tool_name* covered by *write_tools*, the hook_config write list?

    Entries are exact tool names plus ``mcp__<server>__.*`` matchers for servers
    that gate their own writes. An entry matches when it equals *tool_name*, or
    when it ends in :data:`_MATCHER_WILDCARD` and *tool_name* starts with the
    text before it — so ``foo.*`` matches ``foo`` itself as well as ``foobar``.

    Non-string entries are skipped rather than raising: the list is generated
    into a deployment and a write gate must survive a malformed one.
    """
    for entry in write_tools or ():
        if not isinstance(entry, str):
            continue
        if entry == tool_name:
            return True
        if entry.endswith(_MATCHER_WILDCARD) and tool_name.startswith(
            entry[: -len(_MATCHER_WILDCARD)]
        ):
            return True
    return False


def is_write_call(tool_name, tool_input, short_name=None):
    """Does *this call* write, judged from the tool name and its arguments?

    Distinct from :func:`is_write_tool`, which answers whether a tool is *able*
    to write. Only the python server's ``execute`` separates the two: it carries
    an ``execution_mode``, and a readonly execution writes nothing. A missing
    mode counts as readonly, matching the server's own default. Every other
    write tool writes whenever it is called, and a *tool_input* that is not a
    mapping is read as an empty one.

    The carve-out keys on the SHORT name, so an ``extends`` clone of the python
    server (``mcp__pyva__execute``) keeps it rather than having every readonly
    analysis on it refused. *short_name* is the caller's own resolution against
    its prefix list; without one the ``mcp__<server>__<tool>`` shape the lists
    are generated from is used.
    """
    if short_name is None:
        short_name = short_tool_name(tool_name, ())
    if short_name != _EXECUTE_TOOL:
        return True
    if not isinstance(tool_input, dict):
        tool_input = {}
    return tool_input.get("execution_mode", "readonly") != "readonly"


def get_hook_input():
    """Read and return hook input JSON from stdin. Returns {} on failure."""
    try:
        parsed = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def get_project_dir(hook_input=None):
    """Resolve project directory: CLAUDE_PROJECT_DIR env var > hook_input['cwd'].

    This is the directory *Claude Code* runs in, which under the four-zone repo
    layout is the RENDER — ``<repo>/build`` — not the repo. It is the right
    answer for anything that belongs to the render (the agent's own settings,
    the hook debug log beside them) and the wrong one for durable agent state,
    which outlives the render: see :func:`get_repo_root`.

    *hook_input* is optional because hooks that read no stdin (``UserPromptSubmit``)
    still need the env var.
    """
    return os.environ.get("CLAUDE_PROJECT_DIR") or (hook_input or {}).get("cwd", "")


#: The marker that makes a directory a deployment repo root, and the name of the
#: render zone inside it. Kept in step with ``osprey.cli.repo_resolver`` and
#: ``osprey.utils.workspace``; spelled out here rather than imported because a
#: hook can be executed by a bare system ``python3`` with no ``osprey`` on its
#: path, and the derivation below has to work under exactly that interpreter.
PROFILE_MARKER = "profile.yml"
BUILD_DIR_NAME = "build"

#: ``project_root`` as a TOP-LEVEL config key with a real value. Line-scanned
#: rather than parsed: PyYAML is not guaranteed to be importable in a hook (a raw
#: ``claude`` launch runs hooks under whatever ``python3`` is on PATH), and a
#: scan also survives a config whose *other* keys are malformed. Anchoring at
#: column 0 is what makes it safe — the shipped config templates mention
#: ``project_root`` in comments and under nested blocks, and both are indented or
#: commented, so neither can match.
_PROJECT_ROOT_LINE = re.compile(r"^project_root:[ \t]*(\S.*?)[ \t]*$", re.MULTILINE)

#: An unquoted YAML scalar ends where a whitespace-preceded ``#`` begins. A ``#``
#: without leading whitespace is part of the value, so the whitespace is required.
_INLINE_COMMENT = re.compile(r"\s+#")

#: Unquoted spellings YAML resolves to a non-string type. The framework's
#: ``dotted_config_str`` answers ``None`` for every one of them, because a value
#: that is not a string cannot be a path; a text scan has to make the same call
#: itself or it reads them as literal directory names — ``~`` would expand to the
#: user's home and ``null`` would become a relative directory called ``null``.
#: Quoted, they are ordinary strings and are left alone, exactly as YAML has it.
_NULL_SCALARS = frozenset({"~", "null", "true", "false"})


def _config_path(hook_input=None):
    """The config.yml a hook should read: ``OSPREY_CONFIG``, else the project's.

    One reader for the question "which config file", so the config *loader*, the
    debug switch and the zone anchor cannot answer it differently. Every launch
    path that starts an agent (``osprey chat``, ``osprey web``, the dispatch
    worker, the container) exports ``OSPREY_CONFIG``; the ``<project_dir>/config.yml``
    fallback covers a raw ``claude`` launch, where the render's own config is the
    one beside the ``.claude/`` directory the session is obeying.
    """
    configured = os.environ.get("OSPREY_CONFIG")
    if configured:
        return Path(os.path.expandvars(configured))
    project_dir = get_project_dir(hook_input)
    return (Path(project_dir) if project_dir else Path.cwd()) / "config.yml"


def _project_root_from_config(config_path):
    """Read ``project_root`` out of *config_path*, or ``None`` if it says nothing.

    The LAST occurrence wins, as it does in YAML: a profile's ``config:`` block
    can append a key the config template already emitted, and a scanner stopping
    at the first match would return the value the config does not have.

    Only a *string* value answers. An unquoted :data:`_NULL_SCALARS` spelling is
    YAML's way of writing "unset", and the framework's ``dotted_config_str``
    returns ``None`` for it, so this returns ``None`` too and the caller falls
    through to the on-disk rules.
    """
    try:
        text = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    matches = _PROJECT_ROOT_LINE.findall(text)
    if not matches:
        return None
    raw = matches[-1]
    if raw[0] in ("'", '"'):
        closing = raw.find(raw[0], 1)
        return (raw[1:closing] if closing > 0 else raw[1:]).strip() or None
    value = _INLINE_COMMENT.split(raw, 1)[0].strip()
    if value.lower() in _NULL_SCALARS:
        return None
    return value or None


def get_repo_root(hook_input=None):
    """Resolve the deployment REPO root — the anchor for durable agent state.

    A hook's working directory is the render (see :func:`get_project_dir`), and
    ``build/`` is re-created wholesale by every ``osprey build``. Anything a hook
    writes for the rest of OSPREY to read — the pending-review store the feedback
    app serves, the focus/artifact state the gallery owns — must anchor one level
    up, on the repo, exactly as ``osprey.utils.workspace.resolve_project_root``
    does for code that can import osprey. This is that rule, restated with the
    standard library only.

    Resolution order, mirroring the framework's:

    1. ``project_root`` in the config :func:`_config_path` names. Authoritative,
       and the only answer that is right inside a container, where the repo was
       rendered at one path and runs at another (``/app/<name>``), so no walk on
       the running filesystem could recover it.
    2. The config's own directory — or its parent when that directory is the
       build zone. Same rule as ``workspace.repo_root_for_config``, and it comes
       *before* any walk because that is where ``resolve_project_root`` puts it:
       when a config file exists, the framework never looks at the filesystem
       around the cwd, and a hook that did would answer a different repo than
       the app it shares state with.
    3. The nearest ancestor holding ``profile.yml``, found the way every OSPREY
       verb finds a repo. Reached only when the config named no root and does
       not exist, where the framework has nothing left to consult either.
    4. The project directory itself. A legacy flat layout has no zones to
       separate, so anchoring on it is the right answer.

    Returns:
        The repo root as a string, matching :func:`get_project_dir`'s type.
    """
    project_dir = get_project_dir(hook_input)
    base = Path(project_dir) if project_dir else Path.cwd()
    config_path = _config_path(hook_input)

    configured = _project_root_from_config(config_path)
    if configured:
        return str(Path(configured).expanduser())

    if config_path.is_file():
        parent = config_path.parent
        return str(parent.parent if parent.name == BUILD_DIR_NAME else parent)

    for candidate in (base, *base.parents):
        if (candidate / PROFILE_MARKER).is_file():
            return str(candidate)

    return str(base)


_osprey_config_cache = None


def load_osprey_config(hook_input=None):
    """Load project config.yml with caching. Falls back to {} on any failure."""
    global _osprey_config_cache
    if _osprey_config_cache is not None:
        return _osprey_config_cache
    try:
        import yaml

        config_path = _config_path(hook_input)
        if config_path.exists():
            with open(config_path) as f:
                _osprey_config_cache = yaml.safe_load(f) or {}
        else:
            _osprey_config_cache = {}
    except Exception:
        _osprey_config_cache = {}
    return _osprey_config_cache


_debug_from_config = None  # module-level cache


def _is_debug_enabled(hook_input):
    """Check whether hook debug logging is enabled.

    Fast path: ``OSPREY_HOOK_DEBUG`` env var.
    Fallback: read ``hooks.debug`` from ``config.yml`` (cached after first read).
    """
    if os.environ.get("OSPREY_HOOK_DEBUG"):
        return True

    global _debug_from_config
    if _debug_from_config is not None:
        return _debug_from_config

    _debug_from_config = False
    try:
        import yaml

        config_path = _config_path(hook_input)
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            _debug_from_config = bool(cfg.get("hooks", {}).get("debug"))
    except Exception:
        pass  # never break a hook for logging
    return _debug_from_config


def log_hook(hook_name, hook_input, status="ok", detail=""):
    """Log one debug line to stderr AND a JSONL file when hook debug is enabled.

    Enabled by the ``OSPREY_HOOK_DEBUG`` env var or, failing that, ``hooks.debug``
    in the project's ``config.yml`` (see ``_is_debug_enabled``).

    Dual output makes debugging possible even when stderr is swallowed by
    the PTY layer.  The JSONL file lands at ``<project>/.claude/hooks/hook_debug.jsonl``
    and is append-only — nothing rotates or truncates it — and it also backs the
    web terminal's Safety-panel hook-activity feed.
    """
    if not _is_debug_enabled(hook_input):
        return
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    tool = hook_input.get("tool_name", "-")

    # stderr (visible in the terminal that launched Claude Code)
    line = f"{ts} [{hook_name}] tool={tool} status={status}"
    if detail:
        line += f" {detail}"
    print(line, file=sys.stderr)

    # JSONL file (always available for tail -f)
    project_dir = get_project_dir(hook_input)
    if project_dir:
        try:
            from pathlib import Path

            log_path = Path(project_dir) / ".claude" / "hooks" / "hook_debug.jsonl"
            record = {"ts": ts, "hook": hook_name, "tool": tool, "status": status}
            tool_use_id = hook_input.get("tool_use_id")
            if tool_use_id:
                record["tool_use_id"] = tool_use_id
            if detail:
                record["detail"] = detail
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass  # never break a hook for logging


# ---------------------------------------------------------------------------
# Audit emission — the hook half of the unified ledger
# ---------------------------------------------------------------------------
#
# `log_hook` above is a DEBUG facility: it is off unless someone turned it on,
# it files under the render (`build/.claude/hooks/hook_debug.jsonl`, wiped by
# the next `osprey build`), and it records allows and denies alike. None of
# that is an audit trail. `emit_audit` below is the sibling that is one: always
# on, anchored on the REPO through `get_repo_root`, and emitted only where the
# hook layer actually decides something — a deny or an ask.
#
# It writes the same record `osprey.audit.writer` writes, into the same
# `var/audit/<identity>/<surface>.jsonl` layout, and it may not import a line
# of it: a hook is a fresh process launched by whatever `python3` is on PATH,
# with no `osprey` importable and no third-party package guaranteed. So the
# minimal subset is restated here in the standard library, and
# `tests/hooks/test_hook_audit_emitter.py` pins every constant below against
# `osprey.audit.envelope`, `osprey.audit.writer` and `osprey.audit.posture` so
# the two cannot drift into two formats sharing one directory.
#
# Nothing here may cost a decision. Hooks fail OPEN — an uncaught exception
# exits non-zero with no JSON and the tool proceeds — so `emit_audit` swallows
# everything and answers `None`, and every call site puts it BEFORE the
# `sys.exit(0)` that carries the decision, never after.

#: The durable audit zone, relative to the repo root. Mirrors
#: ``osprey_connectors.workspace.AUDIT_DIR_RELPATH``; the hook cannot import it.
AUDIT_DIR_RELPATH = "var/audit"

#: Every hook surface starts with this. The MCP middleware records under the
#: server's own name and the HTTP layer under its route family, so the prefix
#: is what keeps the hook subprocess's ledger disjoint from theirs without any
#: cross-process coordination — which is the reason the hook needs no dedup
#: marker: nothing else writes these files.
AUDIT_SURFACE_PREFIX = "hook_"

#: The identity ladder, mirroring ``osprey.utils.identity``: the multi-user
#: deployment's per-container user, then the identity of a container that hosts
#: no single user, then the local account, then an honest floor. Never the
#: hostname — see that module for why.
TERMINAL_USER_ENV = "OSPREY_TERMINAL_USER"
AUDIT_IDENTITY_ENV = "OSPREY_AUDIT_IDENTITY"
IDENTITY_ENV_LADDER = (TERMINAL_USER_ENV, AUDIT_IDENTITY_ENV)
UNKNOWN_IDENTITY = "unknown"

#: The spawn-time posture markers a session child inherits. Absent means this
#: process was not launched by a posture-carrying spawn site at all.
POSTURE_SOURCE_ENV = "OSPREY_POSTURE_SOURCE"
POSTURE_SESSION_ENV = "OSPREY_POSTURE_SESSION"

#: The closed set of provenances a record may claim for its posture, and the
#: one a child-side emitter uses when the marker is absent, blank or
#: unrecognised. `process` is the honest "we were not told" answer — a dispatch
#: worker, a CLI run, a container-level execution mode — and deriving anything
#: else from the posture VALUE would turn that into a confident guess.
POSTURE_SOURCE_PROCESS = "process"
AUDIT_POSTURE_SOURCES = ("spawn", "live", "app", POSTURE_SOURCE_PROCESS)

#: The session posture, as the rest of OSPREY spells it. A hook sees it only
#: through the execution mode it inherits, and only the exact ``readonly``
#: string sandboxes a session — the same value comparison the posture deny
#: itself makes, so the record and the decision read one variable one way.
EXECUTION_MODE_ENV = "OSPREY_EXECUTION_MODE"
READONLY_MODE = "readonly"
POSTURE_SANDBOX = "sandbox"
POSTURE_WRITES = "writes"

#: Decision words. ``refused`` is the envelope's own spelling; ``ask`` is the
#: third word this surface needs and the envelope deliberately does not enforce
#: its two — an approval prompt is neither allowed nor refused by the hook, and
#: recording it as either would be a false statement about what happened.
AUDIT_DECISION_REFUSED = "refused"
AUDIT_DECISION_ASK = "ask"

#: Record fields that are always present, in envelope order after ``ts``.
AUDIT_REQUIRED_FIELDS = (
    "surface",
    "actor",
    "posture",
    "posture_source",
    "session",
    "subject",
    "decision",
    "reason",
)

#: Bounds, matching the envelope's. Identifier-shaped fields are generous
#: enough for any real name and bounded so a caller-supplied one cannot inflate
#: the ledger; ``detail`` gets the larger free-form bound.
AUDIT_MAX_FIELD_CHARS = 256
AUDIT_MAX_DETAIL_CHARS = 1024

#: The size one append stays inside, restated from ``osprey.audit.writer``.
#: POSIX makes an ``O_APPEND`` write atomic with respect to other appenders at
#: this scale, and hooks share the ``var/audit/<identity>/`` directory with the
#: MCP middleware and the HTTP layers — so a hook record that ignored the bound
#: would be the one line another emitter could land inside of. The per-field
#: bounds above do NOT imply it: six 256-character identifiers plus a
#: 1024-character ``detail`` encode to roughly 2.4 KB.
AUDIT_MAX_RECORD_BYTES = 2048

#: What an oversize ``detail`` is replaced by, restated from the writer. A
#: fixed marker rather than a truncation: a record that says its context was
#: dropped is honest, while a silently shortened one reads like the whole
#: context. ``detail`` is the field that gives way because it is the only
#: supplementary one — trimming identifiers would leave a record that names the
#: wrong thing, so an identifier-only record over budget is written whole.
AUDIT_DETAIL_DROPPED = "<dropped: record over the append bound>"

#: Modes the writer uses, restated. Owner writes and group reads the ledger;
#: the per-identity directory is setgid group-writable so the deploy-path
#: provisioning and this fallback agree.
AUDIT_FILE_MODE = 0o640
AUDIT_DIR_MODE = 0o2770

#: Append-only by construction: the descriptor cannot seek backwards.
_AUDIT_OPEN_FLAGS = os.O_WRONLY | os.O_APPEND | os.O_CREAT

# What disqualifies a string from being one path component.
_AUDIT_UNSAFE = ("/", "\\", "\0")
_AUDIT_RESERVED = (".", "..")


def _audit_component(value):
    """Return *value* stripped if it can serve as one path component, else ``""``."""
    if not isinstance(value, str):
        return ""
    candidate = value.strip()
    if not candidate or candidate in _AUDIT_RESERVED:
        return ""
    if any(bad in candidate for bad in _AUDIT_UNSAFE):
        return ""
    return candidate


def acting_identity():
    """Who a record names — the ladder of ``osprey.utils.identity``, restated.

    The answer is used twice, as the record's ``actor`` and as the
    ``var/audit/<identity>/`` directory, so a rung only counts when its value
    can serve as both: that is what keeps the two from drifting apart. Reads
    the environment on every call, because the markers are set per process by
    compose and by the entrypoint.

    Never raises. :func:`getpass.getuser` fails for a uid with no passwd entry,
    which is ordinary in a slim image rather than exceptional.
    """
    for env_name in IDENTITY_ENV_LADDER:
        candidate = _audit_component(os.environ.get(env_name))
        if candidate:
            return candidate
    try:
        import getpass

        return _audit_component(getpass.getuser()) or UNKNOWN_IDENTITY
    except Exception:
        return UNKNOWN_IDENTITY


def audit_posture_source():
    """The provenance marker this process inherited, or ``process``.

    Read, never inferred. A value outside :data:`AUDIT_POSTURE_SOURCES` is
    treated as absent for the same reason the writer refuses one: a record
    whose provenance is unrecognised reads as authoritative and is worse than
    one that admits it was not told.
    """
    marker = _audit_component(os.environ.get(POSTURE_SOURCE_ENV))
    return marker if marker in AUDIT_POSTURE_SOURCES else POSTURE_SOURCE_PROCESS


def audit_ledger_path(hook_name, hook_input=None, identity=None):
    """Where records from *hook_name* land: ``<repo>/var/audit/<identity>/<surface>.jsonl``.

    Anchored on :func:`get_repo_root`, never on :func:`get_project_dir`: the
    render is re-created wholesale by every ``osprey build`` and an audit trail
    that a rebuild erases is not one.
    """
    who = identity or acting_identity()
    root = Path(get_repo_root(hook_input))
    return root / AUDIT_DIR_RELPATH / who / (audit_surface(hook_name) + ".jsonl")


def audit_surface(hook_name):
    """The surface name records from *hook_name* carry.

    Derived from the same string the hook passes to :func:`log_hook`, so a hook
    has exactly one name and the debug line and the audit record cannot come to
    disagree about it. Underscores because the surface is also a file stem.
    """
    return AUDIT_SURFACE_PREFIX + str(hook_name).replace("-", "_")


def _audit_field(value, limit):
    """Bound one record field. Raises on a non-string, which the caller swallows."""
    return value if len(value) <= limit else value[:limit]


def _audit_encode(record):
    """Serialize *record* to the bytes of one JSONL line.

    Compact separators because the byte budget is the point, and the default
    ``ensure_ascii`` because a ledger of pure ASCII lines is readable by every
    consumer that will ever tail it. The writer's ``_encode``, restated.
    """
    return (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8", "replace")


def _audit_line(record):
    """The bytes to append for *record*, degraded to fit the append bound.

    The writer's degrade order, minus the exception a hook cannot reach (the
    executor's ``source``, which no hook record carries): supplementary context
    gives way first, identifiers never give way. An identifier-only record that
    is still over budget is written whole — one large line is better than a
    record that names the wrong thing, and the reader tolerates it.
    """
    line = _audit_encode(record)
    if len(line) <= AUDIT_MAX_RECORD_BYTES or record.get("detail") is None:
        return line
    record["detail"] = AUDIT_DETAIL_DROPPED
    return _audit_encode(record)


def emit_audit(hook_name, hook_input, decision, subject, reason, detail=None):
    """Record one hook decision in the deployment's audit ledger.

    Call this at a deny or an ask, before the ``sys.exit(0)`` that carries the
    decision. Allows are not recorded here: the hook layer refuses and defers,
    and the MCP middleware already records the calls it admits.

    :param hook_name: The hook's own name, exactly as given to :func:`log_hook`.
    :param hook_input: The parsed stdin payload, used only to anchor the repo.
    :param decision: :data:`AUDIT_DECISION_REFUSED` or :data:`AUDIT_DECISION_ASK`.
    :param subject: What was acted on — a tool name, a path. An identifier.
    :param reason: Short machine-ish reason (``posture``, ``writes_disabled``).
    :param detail: Optional supplementary context. Identifiers and config keys
        only: never a config value, a channel value, a prompt or agent text.
    :returns: The ledger path as a string, or ``None`` when nothing was stored.

    Never raises, and never blocks a decision on the audit trail: an unwritable
    zone, an unresolvable repo root or a malformed field costs the record and
    nothing else.
    """
    try:
        record = {"ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}
        identity = acting_identity()
        session = _audit_component(os.environ.get(POSTURE_SESSION_ENV))
        sandboxed = os.environ.get(EXECUTION_MODE_ENV) == READONLY_MODE
        record["surface"] = audit_surface(hook_name)
        record["actor"] = identity
        record["posture"] = POSTURE_SANDBOX if sandboxed else POSTURE_WRITES
        record["posture_source"] = audit_posture_source()
        record["session"] = _audit_field(session, AUDIT_MAX_FIELD_CHARS) if session else None
        record["subject"] = _audit_field(subject or UNKNOWN_IDENTITY, AUDIT_MAX_FIELD_CHARS)
        record["decision"] = _audit_field(decision, AUDIT_MAX_FIELD_CHARS)
        record["reason"] = _audit_field(reason or UNKNOWN_IDENTITY, AUDIT_MAX_FIELD_CHARS)
        if detail:
            record["detail"] = _audit_field(detail, AUDIT_MAX_DETAIL_CHARS)

        path = audit_ledger_path(hook_name, hook_input, identity=identity)
        line = _audit_line(record)

        try:
            fd = os.open(path, _AUDIT_OPEN_FLAGS, AUDIT_FILE_MODE)
        except FileNotFoundError:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                # mkdir's mode is umask-masked and the setgid bit does not
                # survive it at all, so the chmod is what actually sets the
                # mode the deploy path provisions host-side. Best-effort: a
                # narrower directory still beats a dropped record.
                os.chmod(path.parent, AUDIT_DIR_MODE)
            except OSError:
                pass
            fd = os.open(path, _AUDIT_OPEN_FLAGS, AUDIT_FILE_MODE)

        try:
            written = os.write(fd, line)
            if written != len(line) and written > 0 and not line[:written].endswith(b"\n"):
                # Terminate the fragment while the descriptor is still open, so
                # the NEXT record starts on a line of its own instead of being
                # appended onto a half-record and lost with it. One torn write
                # must cost one record, not every record after it. The writer
                # does exactly this, for exactly this reason.
                try:
                    os.write(fd, b"\n")
                except OSError:
                    pass
        finally:
            os.close(fd)

        # A short write leaves a torn line, so the record is not stored — say
        # so rather than hand back a path that suggests it is.
        return str(path) if written == len(line) else None
    except Exception:
        return None
