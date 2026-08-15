"""Project-path resolution for the commands that take ``--project``.

The commands here (``health``, ``channel-finder``) accept something
:mod:`osprey.cli.repo_resolver` alone cannot answer for: a RENDERED project
directory, which holds a ``config.yml`` at its own root and no ``profile.yml``
anywhere above it. That is the whole reason ``--project`` exists beside
``--repo``, and the reason this module exists beside that one.

What it does NOT get to be is a second discovery rule. One question decides
between the two answers — *is the directory in hand itself a project?* — and
when it is not, the deployment repo enclosing it is found by
:func:`osprey.cli.repo_resolver.find_repo_root`, the same upward walk every
repo-scoped verb uses. So ``osprey health`` typed in ``<repo>/data/raw``
reports on that repo, exactly as ``git status`` typed there reports on the
repository, while ``--project`` pointed straight at a render still reports on
the render.

The bug this fixes was the absence of the walk: the stance was literally
``Path.cwd()``, so a subdirectory looked like a deployment with no config at
all, and ``osprey health`` exited 2 with an error row — "this deployment is
broken" to a CI gate and an operator both — over a working directory.

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


def project_config_path(project_path: Path) -> Path:
    """The ``config.yml`` a resolved project directory answers with.

    Two spellings, in the order :func:`osprey.utils.workspace.resolve_config_path`
    reads them: the render at ``build/config.yml`` for a deployment repo, then
    the flat ``config.yml`` a rendered or container project directory holds at
    its own root.

    When NEITHER exists the answer is still a path, because the caller reports
    "not found" against it — and it is the one this directory would hold once
    built. A deployment repo renders into ``build/``, so answering with its root
    would send an operator to a file no build ever writes.

    Spelled here rather than at each caller because a command that resolved the
    project directory one way and its config another would report on two
    different deployments in one run.

    Args:
        project_path: A directory already resolved by :func:`resolve_project_path`.

    Returns:
        The config path, which is not required to exist.
    """
    from osprey.utils.workspace import rendered_config_path

    from .repo_resolver import PROFILE_FILENAME

    rendered: Path = rendered_config_path(project_path)
    if rendered.is_file():
        return rendered
    flat = project_path / "config.yml"
    if flat.is_file():
        return flat
    return rendered if (project_path / PROFILE_FILENAME).is_file() else flat


def _is_project_directory(candidate: Path) -> bool:
    """Whether *candidate* is itself the project, rather than a place inside one.

    True for anything holding a config the commands here can read — a repo root
    with a render in ``build/``, or a rendered/container project directory whose
    own root is the render. A render is the case the upward walk cannot serve:
    it has no ``profile.yml``, and one that happens to sit under a repo (every
    persona render does, in ``build/``) would be silently redirected to that
    repo — reporting on the wrong project rather than on the one named.
    """
    from osprey.utils.workspace import rendered_config_path

    return rendered_config_path(candidate).is_file() or (candidate / "config.yml").is_file()


def resolve_project_path(project_arg: str | None = None) -> Path:
    """Resolve the project directory ``--project`` named, else the enclosing one.

    One question decides it: is the directory in hand — what ``--project``
    named, or where the command was typed — itself a project? If it is, that is
    the answer. If it is not, the deployment repo enclosing it is, found by the
    same upward walk every repo-scoped verb uses
    (:func:`osprey.cli.repo_resolver.find_repo_root`). A directory that is
    neither is returned as-is, so the caller reports a missing config against
    the place the operator actually stood.

    The walk is what makes these commands behave like every other verb: run
    from ``<repo>/data/raw``, ``osprey health`` reports on that repo instead of
    calling it a deployment with no configuration.

    No environment variable sits between the two answers, and none belongs
    there: a variable that silently redirects a command to another directory
    means the same command line acts on different deployments depending on a
    shell the operator cannot see in the invocation.

    Args:
        project_arg: Project directory from the ``--project`` flag (optional).

    Returns:
        Resolved project directory.

    Examples:
        >>> # Using --project flag
        >>> resolve_project_path("~/projects/my-agent")
        Path('/Users/user/projects/my-agent')

        >>> # Typed inside a deployment repo: the repo, not the subdirectory
        >>> resolve_project_path()
        Path('/Users/user/projects/my-agent')
    """
    from .repo_resolver import RepoNotFoundError, find_repo_root

    stance = Path(project_arg).expanduser().resolve() if project_arg else Path.cwd()
    if _is_project_directory(stance):
        return stance
    try:
        return find_repo_root(stance)
    except RepoNotFoundError:
        return stance


def resolve_config_path(project_arg: str | None = None, config_arg: str | None = None) -> str:
    """Resolve configuration file path.

    If ``--config`` is provided, uses it directly. Otherwise resolves the
    project (:func:`resolve_project_path`) and takes the config that project
    answers with (:func:`project_config_path`) — the render first, then the flat
    spelling.

    Args:
        project_arg: Project directory from the ``--project`` flag (optional).
        config_arg: Config file path from the ``--config`` flag (optional).

    Returns:
        Path to configuration file as string.

    Examples:
        >>> # Explicit config file
        >>> resolve_config_path(config_arg="custom.yml")
        'custom.yml'

        >>> # A deployment repo, from anywhere inside it
        >>> resolve_config_path()
        '/Users/user/projects/my-agent/build/config.yml'
    """
    # If explicit config provided, use it
    if config_arg and config_arg != "config.yml":
        return config_arg

    return str(project_config_path(resolve_project_path(project_arg)))
