"""Utilities for project path resolution and management.

This module provides helper functions for resolving project directories
across all CLI commands, supporting the --project flag and environment
variable for flexible project location specification.
"""

import os
from pathlib import Path

from .styles import Messages, console


def _clear_claude_code_project_state(project_path: Path) -> None:
    """Remove Claude Code's cached state for a project path.

    Claude Code stores trust decisions and session data in two places:
    - ~/.claude.json  →  projects.<absolute-path>.hasTrustDialogAccepted
    - ~/.claude/projects/<encoded-path>/  →  session transcripts & memory

    Removing both ensures the trust prompt appears on next launch.
    """
    import json
    import shutil

    project_key = str(project_path)
    cleared = False

    # 1. Remove trust entry from ~/.claude.json
    claude_json = Path.home() / ".claude.json"
    if claude_json.exists():
        try:
            data = json.loads(claude_json.read_text())
            if project_key in data.get("projects", {}):
                del data["projects"][project_key]
                claude_json.write_text(json.dumps(data, indent=2) + "\n")
                cleared = True
        except (json.JSONDecodeError, OSError):
            pass  # Don't fail init over this

    # 2. Remove session/memory directory from ~/.claude/projects/
    encoded_key = project_key.replace("/", "-")
    claude_project_dir = Path.home() / ".claude" / "projects" / encoded_key
    if claude_project_dir.exists():
        shutil.rmtree(claude_project_dir)
        cleared = True

    if cleared:
        console.print(f"  {Messages.success('Cleared Claude Code project state')}")


def resolve_project_path(project_arg: str | None = None) -> Path:
    """Resolve project directory from multiple sources.

    Resolution priority:
    1. --project CLI argument (if provided)
    2. OSPREY_PROJECT environment variable (if set)
    3. Current working directory (default)

    Args:
        project_arg: Project directory from --project flag (optional)

    Returns:
        Resolved project directory as Path object

    Examples:
        >>> # Using --project flag
        >>> resolve_project_path("~/projects/my-agent")
        Path('/Users/user/projects/my-agent')

        >>> # Using environment variable
        >>> os.environ['OSPREY_PROJECT'] = '/tmp/test-project'
        >>> resolve_project_path()
        Path('/tmp/test-project')

        >>> # Default to current directory
        >>> resolve_project_path()
        Path('/current/working/directory')
    """
    # Priority 1: --project CLI argument
    if project_arg:
        return Path(project_arg).expanduser().resolve()

    # Priority 2: OSPREY_PROJECT environment variable
    env_project = os.environ.get("OSPREY_PROJECT")
    if env_project:
        return Path(env_project).expanduser().resolve()

    # Priority 3: Current working directory
    return Path.cwd()


def resolve_config_path(project_arg: str | None = None, config_arg: str | None = None) -> str:
    """Resolve configuration file path.

    If --config is provided, uses it directly.
    Otherwise, looks for config.yml in the resolved project directory.

    Args:
        project_arg: Project directory from --project flag (optional)
        config_arg: Config file path from --config flag (optional)

    Returns:
        Path to configuration file as string

    Examples:
        >>> # Explicit config file
        >>> resolve_config_path(config_arg="custom.yml")
        'custom.yml'

        >>> # Config in project directory
        >>> resolve_config_path(project_arg="~/my-project")
        '/Users/user/my-project/config.yml'

        >>> # Default: ./config.yml
        >>> resolve_config_path()
        '/current/directory/config.yml'
    """
    # If explicit config provided, use it
    if config_arg and config_arg != "config.yml":
        return config_arg

    # Otherwise, resolve project and find config.yml in it
    project_path = resolve_project_path(project_arg)
    config_path = project_path / "config.yml"

    return str(config_path)
