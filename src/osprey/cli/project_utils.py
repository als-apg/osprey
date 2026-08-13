"""Project-path resolution for the commands that take ``--project``.

Not the deployment-repo discovery rule — that is
:mod:`osprey.cli.repo_resolver`, which every repo-scoped verb uses. What is
left here serves the handful of commands that resolve their own inputs and
carry a ``--project`` flag instead (``channel-finder``, ``health``).

:func:`_clear_claude_code_project_state` has no caller. It is kept only because
removing it would also remove this module's re-export of
``encode_claude_project_path``, which a test still imports from here. Both
belong to a dead-code sweep, not to this module.
"""

from pathlib import Path

from osprey.agent_runner.project_paths import encode_claude_project_path

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
    encoded_key = encode_claude_project_path(project_path)
    claude_project_dir = Path.home() / ".claude" / "projects" / encoded_key
    if claude_project_dir.exists():
        shutil.rmtree(claude_project_dir)
        cleared = True

    if cleared:
        console.print(f"  {Messages.success('Cleared Claude Code project state')}")


def resolve_project_path(project_arg: str | None = None) -> Path:
    """Resolve project directory from the flag, else the working directory.

    Two answers, in priority order: what ``--project`` named, or where the
    command was typed.

    No environment variable sits between the two, and none belongs there: a
    variable that silently redirects a command to another directory means the
    same command line acts on different deployments depending on a shell the
    operator cannot see in the invocation. Discovery is one rule
    (:func:`osprey.cli.repo_resolver.find_repo_root`), with ``--repo`` as its
    only override; the commands calling this function are the ones that
    resolve their own inputs and take ``--project`` instead.

    Args:
        project_arg: Project directory from --project flag (optional)

    Returns:
        Resolved project directory as Path object

    Examples:
        >>> # Using --project flag
        >>> resolve_project_path("~/projects/my-agent")
        Path('/Users/user/projects/my-agent')

        >>> # Default to current directory
        >>> resolve_project_path()
        Path('/current/working/directory')
    """
    if project_arg:
        return Path(project_arg).expanduser().resolve()

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
