"""MCP tools: setup_inspect and setup_patch.

Diagnostic tools for inspecting and modifying OSPREY agent configuration.
Used by the /setup-mode skill to help operators troubleshoot setup issues.
"""

import json
import logging
import os
import re
from pathlib import Path

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.workspace.server import mcp
from osprey.utils.workspace import load_osprey_config, resolve_config_path

logger = logging.getLogger("osprey.mcp_server.tools.setup")

# Keys whose values should be masked in environment output
_SENSITIVE_PATTERNS = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)", re.IGNORECASE)

# Files that setup_patch is allowed to modify
_PATCHABLE_FILES = {"config.yml", ".mcp.json"}

# Changes that take effect without restarting the MCP server
_HOT_CHANGE_PATHS = {
    "config.yml": {
        "control_system.writes_enabled",
        "control_system.limits_checking.enabled",
        "control_system.limits_checking.allow_unlisted_channels",
        "approval.global_mode",
    },
}


def _mask_env(env: dict[str, str]) -> dict[str, str]:
    """Return a copy of env vars with sensitive values replaced by '***'."""
    masked = {}
    for key, value in env.items():
        if _SENSITIVE_PATTERNS.search(key):
            masked[key] = "***"
        else:
            masked[key] = value
    return masked


def _read_json_file(path: Path) -> dict | list | None:
    """Read and parse a JSON file, returning None if missing or invalid."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _read_text_file(path: Path) -> str | None:
    """Read a text file, returning None if missing."""
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _list_files_in(directory: Path) -> list[str]:
    """List file names in a directory, returning empty list if missing."""
    if not directory.is_dir():
        return []
    return sorted(f.name for f in directory.iterdir() if f.is_file())


def _list_dirs_in(directory: Path) -> list[str]:
    """List subdirectory names in a directory, returning empty list if missing."""
    if not directory.is_dir():
        return []
    return sorted(d.name for d in directory.iterdir() if d.is_dir())


@mcp.tool()
async def setup_inspect() -> str:
    """Inspect OSPREY agent configuration.

    Returns a comprehensive JSON snapshot of the current agent configuration
    including config.yml, MCP servers, settings, hooks, rules, agents, skills,
    environment variables, and workspace structure.

    Environment variables containing KEY, TOKEN, SECRET, or PASSWORD in their
    names have their values masked.

    Returns:
        JSON object with all configuration sections.
    """
    try:
        project_root = resolve_config_path().parent
        claude_dir = project_root / ".claude"

        # Config
        config = load_osprey_config()

        # MCP servers (.mcp.json)
        mcp_servers = _read_json_file(project_root / ".mcp.json")

        # Settings (.claude/settings.json)
        settings = _read_json_file(claude_dir / "settings.json")

        # Hooks (extracted from settings)
        hooks = settings.get("hooks", {}) if isinstance(settings, dict) else {}

        # Rules
        rules_dir = claude_dir / "rules"
        rules = _list_files_in(rules_dir)

        # Agents
        agents_dir = claude_dir / "agents"
        agents = _list_files_in(agents_dir)

        # Skills
        skills_dir = claude_dir / "skills"
        skills = _list_dirs_in(skills_dir)

        # Environment (OSPREY-related vars only, masked)
        osprey_env = {
            k: v
            for k, v in os.environ.items()
            if k.startswith("OSPREY_") or k.startswith("CLAUDE_")
        }
        masked_env = _mask_env(osprey_env)

        # Workspace
        ws_dir = project_root / "osprey-workspace"
        workspace = {
            "exists": ws_dir.is_dir(),
            "subdirs": _list_dirs_in(ws_dir) if ws_dir.is_dir() else [],
        }

        return json.dumps(
            {
                "config": config,
                "mcp_servers": mcp_servers,
                "settings": settings,
                "hooks": hooks,
                "rules": rules,
                "agents": agents,
                "skills": skills,
                "environment": masked_env,
                "workspace": workspace,
                "project_root": str(project_root),
            },
            indent=2,
        )

    except Exception as exc:
        logger.exception("setup_inspect failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to inspect configuration: {exc}",
                ["Check that config.yml exists and is readable."],
            )
        )


def _validate_key_path(key_path: str) -> str | None:
    """Validate a dot-notation key path. Returns error message or None."""
    if not key_path:
        return "key_path must not be empty"
    if ".." in key_path:
        return "key_path must not contain '..'"
    if key_path.startswith("/") or key_path.startswith("~"):
        return "key_path must not be an absolute path"
    # Only allow alphanumeric, dots, underscores, hyphens
    if not re.match(r"^[a-zA-Z0-9_.][a-zA-Z0-9_.\-]*$", key_path):
        return "key_path contains invalid characters"
    return None


def _classify_change(file: str, key_path: str) -> str:
    """Classify whether a config change is hot (live) or cold (requires restart)."""
    hot_paths = _HOT_CHANGE_PATHS.get(file, set())
    if key_path in hot_paths:
        return "hot — takes effect immediately (hooks re-read config on each call)"
    return "cold — requires MCP server restart (`osprey claude restart` or new session)"


def _set_nested(data: dict, keys: list[str], value) -> None:
    """Set a value in a nested dict using a list of keys."""
    for key in keys[:-1]:
        if key not in data or not isinstance(data[key], dict):
            data[key] = {}
        data = data[key]
    data[keys[-1]] = value


def _get_nested(data: dict, keys: list[str], default=None):
    """Get a value from a nested dict using a list of keys."""
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return default
        data = data[key]
    return data


@mcp.tool()
async def setup_patch(file: str, key_path: str, value: str) -> str:
    """Modify an OSPREY configuration file.

    Patches a single key in config.yml or .mcp.json using dot-notation paths.
    Values are YAML-parsed for automatic type conversion (e.g., "true" becomes
    boolean True, "42" becomes integer 42).

    Only config.yml and .mcp.json are allowed targets. YAML files preserve
    formatting via ruamel.yaml round-trip mode.

    Args:
        file: Target file — must be "config.yml" or ".mcp.json".
        key_path: Dot-notation path to the key (e.g., "control_system.writes_enabled").
        value: New value as a string (YAML-parsed for type conversion).

    Returns:
        JSON with before/after values and hot/cold change classification.
    """
    try:
        # Validate file whitelist
        if file not in _PATCHABLE_FILES:
            return json.dumps(
                make_error(
                    "validation_error",
                    f"File '{file}' is not patchable. Allowed: {sorted(_PATCHABLE_FILES)}",
                    ["Only config.yml and .mcp.json can be patched."],
                )
            )

        # Validate key_path
        key_error = _validate_key_path(key_path)
        if key_error:
            return json.dumps(
                make_error(
                    "validation_error",
                    f"Invalid key_path: {key_error}",
                    ["Use dot-notation like 'control_system.writes_enabled'."],
                )
            )

        project_root = resolve_config_path().parent
        file_path = project_root / file

        if not file_path.exists():
            return json.dumps(
                make_error(
                    "not_found",
                    f"File not found: {file}",
                    [f"Create {file} first, or run `osprey claude regen`."],
                )
            )

        keys = key_path.split(".")

        # Parse value using YAML for type conversion
        import yaml as pyyaml

        parsed_value = pyyaml.safe_load(value)

        if file.endswith(".json"):
            # JSON file — standard json
            data = json.loads(file_path.read_text(encoding="utf-8"))
            before = _get_nested(data, keys)
            _set_nested(data, keys, parsed_value)
            file_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        else:
            # YAML file — ruamel.yaml round-trip mode
            from ruamel.yaml import YAML

            ryaml = YAML()
            ryaml.preserve_quotes = True
            with open(file_path, encoding="utf-8") as f:
                data = ryaml.load(f)
            if data is None:
                data = {}

            before = _get_nested(data, keys)
            _set_nested(data, keys, parsed_value)

            with open(file_path, "w", encoding="utf-8") as f:
                ryaml.dump(data, f)

        note = _classify_change(file, key_path)

        return json.dumps(
            {
                "file": file,
                "key_path": key_path,
                "before": before,
                "after": parsed_value,
                "note": note,
            },
            indent=2,
            default=str,
        )

    except Exception as exc:
        logger.exception("setup_patch failed")
        return json.dumps(
            make_error(
                "internal_error",
                f"Failed to patch {file}: {exc}",
                ["Check that the file is valid YAML/JSON."],
            )
        )
