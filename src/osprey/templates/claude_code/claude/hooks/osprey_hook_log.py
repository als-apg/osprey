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
