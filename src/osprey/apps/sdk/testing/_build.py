"""The two verbs a synthetic deployment is made with: ``osprey init`` and ``osprey build``.

Both run in process through Click's runner, so a test pays no interpreter start
and sees what the command printed. The build skips the dependency install and
the lifecycle hooks: the test process already holds the framework, and nothing
is started.
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner, Result

__all__ = ["PRESET", "init_repo", "invoke_build"]

#: The preset a synthetic deployment starts from: the simulator served in process,
#: no containers of its own.
PRESET = "hello-world"


def init_repo(repo: Path) -> Result:
    """Run ``osprey init`` for :data:`PRESET` into *repo*, without git.

    Args:
        repo: The repository to create; it must not exist.

    Returns:
        The runner's result.
    """
    from osprey.cli.init_cmd import init

    return CliRunner().invoke(init, [str(repo), "--preset", PRESET, "--no-git"])


def invoke_build(repo: Path | None = None, *args: str) -> Result:
    """Run ``osprey build`` with no dependency install and no lifecycle hooks.

    Args:
        repo: The deployment repository, the directory holding ``profile.yml``,
            passed as ``--repo``. ``None`` passes none, so the build finds its
            repository from the working directory.
        *args: Further arguments, after the flags.

    Returns:
        The runner's result.
    """
    from osprey.cli.build_cmd import build

    repo_args = [] if repo is None else ["--repo", str(repo)]
    return CliRunner().invoke(build, [*repo_args, "--skip-deps", "--skip-lifecycle", *args])
