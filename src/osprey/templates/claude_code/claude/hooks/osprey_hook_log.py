"""Shared hook logging utility. No external dependencies."""

import contextlib
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
    "mcp__phoebus__phoebus_drive",
    "mcp__python__execute",
    "mcp__python__execute_file",
    "mcp__python__pyaml_measure",
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
#: render zone inside it. Kept in step with ``osprey.cli.repo_resolver``,
#: ``osprey.utils.workspace`` and ``osprey_connectors.workspace``'s private
#: copy; spelled out
#: here rather than imported because a hook can be executed by a bare system
#: ``python3`` with no ``osprey`` on its path, and the derivation below has to
#: work under exactly that interpreter.
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
    up, on the repo, exactly as ``osprey_connectors.workspace.resolve_project_root``
    does for code that can import osprey. This is that rule, restated with the
    standard library only.

    Resolution order, mirroring the framework's:

    1. ``project_root`` in the config :func:`_config_path` names, WHEN it names a
       directory that exists here. The qualifier is the rule, not a safety net:
       a deployed service reads a config STAGED on the host, whose
       ``project_root`` is a host path naming nothing inside the container, and
       anchoring on it would put every store this resolver anchors — the
       control-context record, the spool, the pending-review store — outside
       every mount, where the writer that shares those files never looks.
       Qualified the same way, and for the same reason, as
       ``osprey_connectors.workspace.resolve_project_root``.
    2. The config's own directory — or its parent when that directory is the
       build zone. Same rule as ``workspace.repo_root_for_config``, and it comes
       *before* any walk because that is where ``resolve_project_root`` puts it:
       when a config file exists, the framework never looks at the filesystem
       around the cwd, and a hook that did would answer a different repo than
       the app it shares state with. This is the rung a container lands on: the
       mount target names the render, so unwrapping it names the repo the
       service actually runs in.
    3. The nearest ancestor holding ``profile.yml``, found the way every OSPREY
       verb finds a repo. Reached only when the config named no usable root and
       does not exist. The framework takes the same walk at the same point,
       from its working directory.
    4. The recorded ``project_root`` after all, unqualified. A build rendered
       with ``--runtime-root`` records a path that exists only on the machine it
       will run on, and with no config file to unwrap there is nothing better to
       anchor on — the framework keeps the same value as its own late rung.
    5. The project directory itself. A legacy flat layout has no zones to
       separate, so anchoring on it is the right answer. The framework's last
       rung is its working directory instead. It does not read the harness
       variable, so the two agree whenever that variable is unset or names the
       working directory.

    Returns:
        The repo root as a string, matching :func:`get_project_dir`'s type.
    """
    project_dir = get_project_dir(hook_input)
    base = Path(project_dir) if project_dir else Path.cwd()
    config_path = _config_path(hook_input)

    configured = _project_root_from_config(config_path)
    recorded = Path(configured).expanduser() if configured else None
    if recorded is not None and recorded.is_dir():
        return str(recorded)

    if config_path.is_file():
        parent = config_path.parent
        return str(parent.parent if parent.name == BUILD_DIR_NAME else parent)

    for candidate in (base, *base.parents):
        if (candidate / PROFILE_MARKER).is_file():
            return str(candidate)

    if recorded is not None:
        return str(recorded)

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


# The framework default agent-data root, imported so the hooks and the framework
# cannot drift apart. The literal serves a hook running with osprey off the
# path, the one case where guessing beats crashing.
try:
    from osprey_connectors.workspace import DEFAULT_AGENT_DATA_BASE_DIR as _DEFAULT_AGENT_DATA_ROOT
except Exception:  # pragma: no cover - hooks must never crash the agent
    _DEFAULT_AGENT_DATA_ROOT = "var/agent_data"


def agent_data_base_dir(config: dict | None) -> str:
    """Read ``agent_data.base_dir`` out of an already-loaded config mapping.

    A restatement of ``osprey_connectors.workspace.agent_data_base_dir`` using
    the standard library only, because a hook runs in user projects where
    ``osprey`` may not be importable. This is the only hook-side reader of the
    key, so every hook that touches agent data follows a relocated root the same
    way the framework and the rendered ``Edit(...)`` allow rules do.

    Args:
        config: Loaded ``config.yml`` mapping, or ``None``.

    Returns:
        The configured base directory, possibly relative to a project anchor.
    """
    section = (config or {}).get("agent_data") or {}
    if not isinstance(section, dict):
        return _DEFAULT_AGENT_DATA_ROOT
    return str(section.get("base_dir") or _DEFAULT_AGENT_DATA_ROOT)


def agent_data_root_at(anchor, config) -> str:
    """Anchor the configured agent-data root on *anchor*.

    The same three-way choice as ``osprey_connectors.workspace.anchored_path``:
    ``~`` is expanded, an absolute ``base_dir`` is returned as is, and a relative
    one is joined onto the anchor.

    Args:
        anchor: The directory a relative ``base_dir`` is anchored on.
        config: Loaded ``config.yml`` mapping, or ``None``.

    Returns:
        The agent-data root as a string.
    """
    base = Path(agent_data_base_dir(config)).expanduser()
    if base.is_absolute():
        return str(base)
    return str(Path(anchor).expanduser() / base)


def repo_agent_data_root(hook_input=None) -> str:
    """The agent-data root the gallery and the channel-finder app write under.

    ``agent_data.base_dir`` anchored on the repo (:func:`get_repo_root`) rather
    than the render, because ``build/`` is disposable. It does not read the
    ``OSPREY_AGENT_DATA_ROOT`` stamp: the writers it pairs with resolve through
    the config alone.

    Args:
        hook_input: The parsed hook payload, used to locate the project.

    Returns:
        The agent-data root as a string.
    """
    return agent_data_root_at(get_repo_root(hook_input), load_osprey_config(hook_input))


def agent_data_subdirs(hook_input, subdir) -> list[Path]:
    """Resolve one agent-data subdirectory under every anchor it can have.

    A relative ``agent_data.base_dir`` — the normal case — needs an anchor, and
    under the four-zone layout there are two plausible ones: the repo root that
    owns durable agent state (:func:`get_repo_root`) and the render Claude Code
    actually runs in (:func:`get_project_dir`). They coincide in a flat layout
    and diverge in a zoned one, and the gallery can write under either, so
    both are accepted rather than picking one and denying the agent its
    own notebooks under the other. An absolute ``base_dir`` needs no anchor and
    yields exactly one directory.

    Args:
        hook_input: The parsed hook payload, used to locate the project.
        subdir: The agent-data subdirectory to resolve, e.g. ``artifacts``.

    Returns:
        Resolved ``.../<subdir>`` directories, deduplicated, possibly empty.
    """
    config = load_osprey_config(hook_input)
    candidates = [
        Path(agent_data_root_at(anchor, config))
        for anchor in (get_repo_root(hook_input), get_project_dir(hook_input))
        if anchor
    ]

    subdirs: list[Path] = []
    for candidate in candidates:
        try:
            resolved = (candidate / subdir).resolve()
        except (OSError, ValueError):
            continue
        if resolved not in subdirs:
            subdirs.append(resolved)
    return subdirs


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
    """Record one debug line when hook debug is enabled; do nothing otherwise.

    Enabled by the ``OSPREY_HOOK_DEBUG`` env var or, failing that, ``hooks.debug``
    in the project's ``config.yml`` (see ``_is_debug_enabled``).

    The line goes to stderr (the terminal that launched Claude Code) and, as a
    ``debug`` entry whose body names ``source: "hook"``, onto the agent record
    through the same sealing spool emitter :func:`emit_audit` uses. The web
    terminal's Safety-panel hook-activity feed reads those entries back. Never
    raises: a hook decision never depends on its debug line.
    """
    if not _is_debug_enabled(hook_input):
        return
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    tool = hook_input.get("tool_name", "-") if isinstance(hook_input, dict) else "-"

    line = f"{ts} [{hook_name}] tool={tool} status={status}"
    if detail:
        line += f" {detail}"
    print(line, file=sys.stderr)  # noqa: T201 -- stderr carries the hook log line

    emit_debug(hook_name, hook_input, tool, status, detail)


# ---------------------------------------------------------------------------
# Audit emission — the hook half of the agent record
# ---------------------------------------------------------------------------
#
# `log_hook` above is a DEBUG facility: it is off unless someone turned it on,
# its entries are ``debug`` entries marked ``source: "hook"``, and it records
# allows and denies alike. None of that is an audit trail. `emit_audit` below
# is the sibling that is one: always on (but for the record off switch), and
# emitted only where the hook layer actually decides something — a deny, an
# ask, or an ask's answer. Both write through the one spool emitter here.
#
# It writes the entry `osprey.audit.writer` writes — an ``audit`` entry whose
# body is the envelope — as one line of agent-record's spool, in the exact
# line format of `osprey.agent_record.writer`, and it may not import a line of
# either: a hook is a fresh process launched by whatever `python3` is on PATH,
# with no `osprey` importable and no third-party package guaranteed. So the
# minimal subset is restated here in the standard library, and
# `tests/hooks/test_hook_audit_emitter.py` pins every constant below against
# `osprey.agent_record.contract`, `osprey.agent_record.writer`,
# `osprey.audit.envelope`, `osprey.audit.writer`, `osprey.audit.posture` and
# `osprey.audit.call`, and consumes what the hook writes through the spool
# consumer, so the hook cannot drift into a format the record refuses.
#
# The spool protocol is the writer's, cut to a process that lives for one
# decision: every call creates a segment of its own with ``O_CREAT|O_EXCL``,
# writes its one line, and renames the segment to ``.sealed`` before it
# returns. A hook therefore never leaves an ``.open`` segment behind for the
# consumer to report as an abandoned writer, and needs no exit handler.
#
# Nothing here may cost a decision. Hooks fail OPEN — an uncaught exception
# exits non-zero with no JSON and the tool proceeds — so `emit_audit` swallows
# everything and answers `None`, and every call site puts it BEFORE the
# `sys.exit(0)` that carries the decision, never after.

#: The spool zone, relative to the repo root. Mirrors
#: ``osprey.utils.workspace.SPOOL_DIR_RELPATH``; the hook cannot import it.
SPOOL_DIR_RELPATH = "var/spool"

#: The acting identity's own spool directory, when the launcher names it. A
#: container binds only its own directory, so there is nothing above it to
#: resolve. Mirrors ``osprey.agent_record.contract.SPOOL_DIR_ENV``.
SPOOL_DIR_ENV = "OSPREY_SPOOL_DIR"

#: The record off switch, restated from the contract. Exactly this value turns
#: every writer of the record off: nothing is resolved, created or appended.
RECORD_OFF_ENV = "OSPREY_AGENT_RECORD_OFF"
RECORD_OFF_VALUE = "1"

#: Segment suffixes and the writer's failure log, restated from the contract.
SPOOL_OPEN_SUFFIX = ".open"
SPOOL_SEALED_SUFFIX = ".sealed"
SPOOL_ERRORS_FILE = ".errors"

#: Modes the spool writer uses, restated. A segment is readable by the spool
#: directory's group (the consumer); a directory the hook has to create itself
#: is setgid group-writable so the consumer can remove sealed segments.
SPOOL_SEGMENT_MODE = 0o640
SPOOL_DIR_MODE = 0o2770

#: A segment is created, never reopened: ``O_EXCL`` refuses a name that is
#: already taken, so no other process's segment can be appended to.
_SPOOL_CREATE_FLAGS = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL

#: Name collisions retried before a record is given up. A collision needs the
#: same host, pid and nanosecond, so one retry is already generous.
_SPOOL_CREATE_ATTEMPTS = 3

#: The entry kinds hook lines carry, restated from the contract: a decision
#: is an ``audit`` entry, a hook debug line a ``debug`` entry.
KIND_AUDIT = "audit"
KIND_DEBUG = "debug"

#: The body field naming the writer of an entry (the reader's ``source``
#: filter), and the value every hook debug line carries in it. The transcript
#: also produces ``debug`` entries; this marker is what keeps them apart.
SOURCE_FIELD = "source"
DEBUG_SOURCE_HOOK = "hook"

#: The optional id fields of a spool line, in the order the writer writes them
#: between ``ts`` and ``body``. ``key`` heads the list in the contract but is
#: honoured only in a receiver-owned directory, so a hook never writes it.
SPOOL_LINE_ID_FIELDS = ("session_id", "terminal_session", "run_id", "tool_use_id", "on_behalf_of")

#: The conversation id env names, restated from ``osprey.audit.call``: the id
#: Osprey forces first, then the one the harness stamps on its children. Read
#: only when the hook input carries no ``session_id`` of its own.
CONVERSATION_ENV = "OSPREY_TELEMETRY_SESSION_ID"
HARNESS_CONVERSATION_ENV = "CLAUDE_CODE_SESSION_ID"

#: The identity ladder, restated from ``osprey_connectors.identity``: the
#: multi-user deployment's per-container user, then the identity of a container
#: that hosts no single user, then the local account, then an honest floor.
#: Never the hostname — see that module for why. Restated rather than imported
#: because a hook runs outside the osprey venv; the two are held together by
#: ``tests/connectors/test_identity_ladder.py``.
TERMINAL_USER_ENV = "OSPREY_TERMINAL_USER"
AUDIT_IDENTITY_ENV = "OSPREY_AUDIT_IDENTITY"
IDENTITY_ENV_LADDER = (TERMINAL_USER_ENV, AUDIT_IDENTITY_ENV)
UNKNOWN_IDENTITY = "unknown"

#: Every hook surface starts with this. The MCP middleware records under the
#: server's own name and the HTTP layer under its route family, so the prefix
#: is what keeps the hook's surfaces disjoint from theirs.
AUDIT_SURFACE_PREFIX = "hook_"

#: The spawn-time posture markers a session child inherits. Absent means this
#: process was not launched by a posture-carrying spawn site at all.
POSTURE_SOURCE_ENV = "OSPREY_POSTURE_SOURCE"
POSTURE_SESSION_ENV = "OSPREY_POSTURE_SESSION"
#: The launch-scoped terminal session, for a process with no posture session of
#: its own (a dispatched job, a host chat). Mirrors
#: ``osprey.audit.envelope.TERMINAL_SESSION_ENV``; read by exact name.
TERMINAL_SESSION_ENV = "OSPREY_TERMINAL_SESSION"

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

#: An ask's answer is its own record, on the same ``tool_use_id`` as the ask:
#: ``approved`` when the harness reports the call ran, ``denied`` when the turn
#: ended without it. The envelope spells both the same way.
AUDIT_DECISION_APPROVED = "approved"
AUDIT_DECISION_DENIED = "denied"

#: The shape a tool-use id is recorded in, restated from ``osprey.audit.call``.
#: Anything else in the hook input is dropped rather than recorded.
_AUDIT_TOOL_USE_ID = re.compile(r"\A[A-Za-z0-9_-]{1,128}\Z")

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
#: the record; ``detail`` gets the larger free-form bound.
AUDIT_MAX_FIELD_CHARS = 256
AUDIT_MAX_DETAIL_CHARS = 1024

#: The size an envelope body stays inside, restated from ``osprey.audit.writer``,
#: so a hook entry is bounded exactly like every other audit entry. The
#: per-field bounds above do NOT imply it: six 256-character identifiers plus a
#: 1024-character ``detail`` encode to roughly 2.4 KB.
AUDIT_MAX_RECORD_BYTES = 2048

#: What an oversize ``detail`` is replaced by, restated from the writer. A
#: fixed marker rather than a truncation: a record that says its context was
#: dropped is honest, while a silently shortened one reads like the whole
#: context. ``detail`` is the field that gives way because it is the only
#: supplementary one — trimming identifiers would leave a record that names the
#: wrong thing, so an identifier-only record over budget is written whole.
AUDIT_DETAIL_DROPPED = "<dropped: record over the append bound>"

# What disqualifies a string from being one path component.
_AUDIT_UNSAFE = ("/", "\\", "\0")
_AUDIT_RESERVED = (".", "..")
# What else disqualifies a string from naming a source identity — the
# contract's ``check_identity``, restated: its forbidden characters and space.
_SPOOL_IDENTITY_FORBIDDEN = "/\\:$\0"


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
    """Who a record names — the ladder of ``osprey_connectors.identity``, restated.

    The answer is used twice, as the record's ``actor`` and, when the launcher
    names no spool directory, as the ``var/spool/<identity>/`` directory, so a
    rung only counts when its value can serve as both: that is what keeps the
    two from drifting apart. Reads the environment on every call, because the
    markers are set per process by compose and by the entrypoint.

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


def record_off():
    """Whether the record off switch is set for this call; read on every call."""
    return os.environ.get(RECORD_OFF_ENV) == RECORD_OFF_VALUE


def audit_spool_dir(hook_input=None, identity=None):
    """The spool directory records from this process land in.

    ``$OSPREY_SPOOL_DIR`` when it is set and non-empty — the launcher's answer,
    exactly as the record writer takes it. Otherwise
    ``<repo>/var/spool/<identity>``, anchored on :func:`get_repo_root` and never
    on :func:`get_project_dir`: the render is re-created wholesale by every
    ``osprey build``, and the consumer never looks there.

    Raises ``ValueError`` when the identity cannot name a spool directory; the
    caller swallows it.
    """
    configured = os.environ.get(SPOOL_DIR_ENV, "").strip()
    if configured:
        return Path(configured)
    who = identity or acting_identity()
    if any(ch in who for ch in _SPOOL_IDENTITY_FORBIDDEN) or any(ch.isspace() for ch in who):
        raise ValueError(f"{who!r} cannot be a source identity")
    return Path(get_repo_root(hook_input)) / SPOOL_DIR_RELPATH / who


def audit_surface(hook_name):
    """The surface name records from *hook_name* carry.

    Derived from the same string the hook passes to :func:`log_hook`, so a hook
    has exactly one name and the debug line and the audit record cannot come to
    disagree about it.
    """
    return AUDIT_SURFACE_PREFIX + str(hook_name).replace("-", "_")


def _audit_field(value, limit):
    """Bound one record field. Raises on a non-string, which the caller swallows."""
    return value if len(value) <= limit else value[:limit]


def _audit_encode(record):
    """The bytes the audit writer measures a body by, restated.

    Compact separators and the default ``ensure_ascii``, exactly as
    ``osprey.audit.writer`` encodes a body to hold it to
    :data:`AUDIT_MAX_RECORD_BYTES`. Only the bound is measured this way; the
    line itself is encoded by :func:`_spool_line`.
    """
    return (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode(
        "utf-8", "replace"
    )


def _audit_body(record):
    """*record*, degraded to fit the body bound the way the audit writer does.

    Supplementary context gives way first, identifiers never give way. An
    identifier-only record that is still over budget is written whole — one
    large entry is better than a record that names the wrong thing.
    """
    if len(_audit_encode(record)) <= AUDIT_MAX_RECORD_BYTES or record.get("detail") is None:
        return record
    record["detail"] = AUDIT_DETAIL_DROPPED
    return record


def _audit_tool_use_id(value):
    """*value* when it is a well-formed tool-use id, else ``None``."""
    if isinstance(value, str) and _AUDIT_TOOL_USE_ID.match(value):
        return value
    return None


def _audit_conversation(hook_input):
    """The conversation the entry belongs to: the hook input's, else the process's."""
    if isinstance(hook_input, dict):
        given = hook_input.get("session_id")
        if isinstance(given, str) and given:
            return given
    return os.environ.get(CONVERSATION_ENV) or os.environ.get(HARNESS_CONVERSATION_ENV) or None


def _terminal_session():
    """``osprey.audit.envelope.terminal_session``, restated: stripped, bounded, or ``None``."""
    value = (os.environ.get(TERMINAL_SESSION_ENV) or "").strip()
    return value[:AUDIT_MAX_FIELD_CHARS] if value else None


def _spool_line(entry_id, ts, body, ids, kind):
    """One spool line, byte for byte as ``SpoolLine.to_json`` writes it, plus its newline."""
    line = {"entry_id": entry_id, "kind": kind, "ts": ts}
    for name in SPOOL_LINE_ID_FIELDS:
        value = ids.get(name)
        if value is not None:
            line[name] = value
    line["body"] = dict(body)
    return (json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _spool_host():
    """This host's name, made safe as the first part of a segment file name."""
    import socket

    try:
        name = socket.gethostname()
    except OSError:
        name = ""
    for ch in _AUDIT_UNSAFE:
        name = name.replace(ch, "_")
    return name if name not in ("", *_AUDIT_RESERVED) else "host"


def _spool_ensure_dir(directory):
    """Create *directory*; the mode is best-effort, exactly as the writer's."""
    directory.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        # mkdir's mode is umask-masked and the setgid bit does not survive it
        # at all, so the chmod is what actually sets the provisioned mode. A
        # narrower directory still beats a dropped record.
        os.chmod(directory, SPOOL_DIR_MODE)


def _spool_create(directory):
    """Create a fresh ``.open`` segment in *directory*; ``(path, fd)``."""
    import time

    host, pid = _spool_host(), os.getpid()
    last = _SPOOL_CREATE_ATTEMPTS - 1
    for attempt in range(_SPOOL_CREATE_ATTEMPTS):
        path = directory / f"{host}-{pid}-{time.time_ns()}{SPOOL_OPEN_SUFFIX}"
        try:
            return path, os.open(path, _SPOOL_CREATE_FLAGS, SPOOL_SEGMENT_MODE)
        except FileNotFoundError:
            # The directory is created only after an open fails for want of
            # it, so the ordinary record is one open, one write, one rename.
            if attempt == last:
                raise
            _spool_ensure_dir(directory)
        except FileExistsError:
            if attempt == last:
                raise
    raise AssertionError("unreachable")


def _spool_write_sealed(directory, data):
    """Write *data* as one segment of its own and seal it; raise on any failure.

    The segment is removed again when anything goes wrong before the seal, so a
    failed record leaves nothing behind for the consumer to half-read.
    """
    path, fd = _spool_create(directory)
    try:
        try:
            written = os.write(fd, data)
        finally:
            os.close(fd)
        if written != len(data):
            raise OSError(f"short append to {path}: {written} of {len(data)} bytes")
        os.rename(path, path.with_name(path.name[: -len(SPOOL_OPEN_SUFFIX)] + SPOOL_SEALED_SUFFIX))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(path)
        raise


def _spool_record_failure(directory, entry_id, exc, kind):
    """Append *exc* to *directory*'s ``.errors`` in the writer's shape; best-effort."""
    if directory is None:
        return
    with contextlib.suppress(Exception):
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "kind": kind,
            "entry_id": entry_id if isinstance(entry_id, str) else repr(entry_id),
            "error": f"{type(exc).__name__}: {exc}",
        }
        data = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        fd = os.open(
            directory / SPOOL_ERRORS_FILE,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            SPOOL_SEGMENT_MODE,
        )
        try:
            os.write(fd, data)
        finally:
            os.close(fd)


def _emit(kind, hook_input, make_body):
    """Write one spool line of *kind*; its ``entry_id``, or ``None`` when nothing was stored.

    *make_body* is called with the line's ``ts`` and answers ``(body,
    tool_use_id, identity)``; *identity* (``None`` to resolve it) picks the
    spool directory. The line names the conversation, the terminal session
    and the tool call. Never raises: off switch on, nothing is written; any
    failure, the body's own included, is noted in the spool directory's
    ``.errors``.
    """
    if record_off():
        return None
    directory = None
    entry_id = None
    try:
        import uuid

        entry_id = uuid.uuid4().hex
        ts = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        body, tool_use_id, identity = make_body(ts)
        ids = {
            "session_id": _audit_conversation(hook_input),
            "terminal_session": _terminal_session(),
            "tool_use_id": tool_use_id,
        }
        data = _spool_line(entry_id, ts, body, ids, kind)
        directory = audit_spool_dir(hook_input, identity=identity)
        _spool_write_sealed(directory, data)
        return entry_id
    except Exception as exc:
        _spool_record_failure(directory, entry_id, exc, kind)
        return None


def emit_audit(hook_name, hook_input, decision, subject, reason, detail=None, tool_use_id=None):
    """Record one hook decision on the agent record, through the spool.

    Call this at a deny or an ask, before the ``sys.exit(0)`` that carries the
    decision. Allows are not recorded here: the hook layer refuses and defers,
    and the MCP middleware already records the calls it admits.

    :param hook_name: The hook's own name, exactly as given to :func:`log_hook`.
    :param hook_input: The parsed stdin payload: it anchors the repo and names
        the conversation (``session_id``) and the tool call (``tool_use_id``).
    :param decision: :data:`AUDIT_DECISION_REFUSED` or :data:`AUDIT_DECISION_ASK`
        (or an ask's answer).
    :param subject: What was acted on — a tool name, a path. An identifier.
    :param reason: Short machine-ish reason (``posture``, ``writes_disabled``).
    :param detail: Optional supplementary context. Identifiers and config keys
        only: never a config value, a channel value, a prompt or agent text.
    :param tool_use_id: The tool call this record is about; defaults to the
        hook input's ``tool_use_id``. A malformed id is left off the record.
    :returns: The entry's ``entry_id``, or ``None`` when nothing was stored —
        the record off switch, an unwritable spool, anything at all.

    Never raises, and never blocks a decision on the record: a failure costs
    the entry and is noted in the spool directory's ``.errors`` for the
    consumer's health report, and nothing else.
    """

    def audit_body(ts):
        record = {"ts": ts}
        identity = acting_identity()
        session = _audit_component(os.environ.get(POSTURE_SESSION_ENV)) or _audit_component(
            os.environ.get(TERMINAL_SESSION_ENV)
        )
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
        raw = tool_use_id
        if raw is None and isinstance(hook_input, dict):
            raw = hook_input.get("tool_use_id")
        use_id = _audit_tool_use_id(raw)
        if use_id:
            record["tool_use_id"] = _audit_field(use_id, AUDIT_MAX_FIELD_CHARS)
        return _audit_body(record), use_id, identity

    return _emit(KIND_AUDIT, hook_input, audit_body)


def emit_debug(hook_name, hook_input, tool, status, detail=""):
    """Record one hook debug line on the agent record, through the spool.

    The body carries what the hook-activity feed shows —
    ``{source, ts, hook, tool, status, tool_use_id?, detail?}`` — with
    ``source`` always :data:`DEBUG_SOURCE_HOOK`. The line names the
    conversation, the terminal session and the tool call the way an audit
    line does, so a reader can narrow to one turn.

    :returns: The entry's ``entry_id``, or ``None`` when nothing was stored.

    Never raises; a failure is noted in the spool directory's ``.errors``,
    exactly as :func:`emit_audit` notes one.
    """

    def debug_body(ts):
        body = {
            SOURCE_FIELD: DEBUG_SOURCE_HOOK,
            "ts": ts,
            "hook": str(hook_name),
            "tool": str(tool),
            "status": str(status),
        }
        raw_tool_use_id = hook_input.get("tool_use_id") if isinstance(hook_input, dict) else None
        tool_use_id = _audit_tool_use_id(raw_tool_use_id)
        if tool_use_id:
            body["tool_use_id"] = tool_use_id
        if detail:
            body["detail"] = str(detail)
        return body, tool_use_id, None

    return _emit(KIND_DEBUG, hook_input, debug_body)
