"""Shared hook logging utility. No external dependencies."""

import json
import os
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
        return json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return {}


def get_project_dir(hook_input):
    """Resolve project directory: CLAUDE_PROJECT_DIR env var > hook_input['cwd']."""
    return os.environ.get("CLAUDE_PROJECT_DIR") or hook_input.get("cwd", "")


_osprey_config_cache = None


def load_osprey_config(hook_input=None):
    """Load project config.yml with caching. Falls back to {} on any failure."""
    global _osprey_config_cache
    if _osprey_config_cache is not None:
        return _osprey_config_cache
    try:
        import yaml

        project_dir = get_project_dir(hook_input) if hook_input else ""
        default = (
            str(Path(project_dir) / "config.yml") if project_dir else str(Path.cwd() / "config.yml")
        )
        config_path = Path(os.path.expandvars(os.environ.get("OSPREY_CONFIG", default)))
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
    project_dir = get_project_dir(hook_input)
    if not project_dir:
        return False
    try:
        from pathlib import Path

        import yaml

        default = str(Path(project_dir) / "config.yml")
        config_path = Path(os.path.expandvars(os.environ.get("OSPREY_CONFIG", default)))
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            _debug_from_config = bool(cfg.get("hooks", {}).get("debug"))
    except Exception:
        pass  # never break a hook for logging
    return _debug_from_config


def log_hook(hook_name, hook_input, status="ok", detail=""):
    """Log one debug line to stderr AND a JSONL file if OSPREY_HOOK_DEBUG is set.

    Dual output makes debugging possible even when stderr is swallowed by
    the PTY layer.  The JSONL file lands at ``<project>/.claude/hooks/hook_debug.jsonl``.
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
            if detail:
                record["detail"] = detail
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass  # never break a hook for logging
