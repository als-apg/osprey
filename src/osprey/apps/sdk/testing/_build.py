"""The two verbs a synthetic deployment is made with: ``osprey init`` and ``osprey build``.

Both run in process through Click's runner, so a test pays no interpreter start
and sees what the command printed. The build skips the dependency install and
the lifecycle hooks: the test process already holds the framework, and nothing
is started.

Each runs its subcommand directly, not through the root ``osprey`` group, so the
caller's process keeps its own environment, logging and console. A facility
error still prints as the command line prints it: the subcommand runs inside
:func:`osprey.cli.main.facility_errors_as_click`, as the root group runs it.
"""

from __future__ import annotations

import copy
import functools
from pathlib import Path

import click
from click.testing import CliRunner, Result

__all__ = ["PRESET", "init_repo", "invoke_build"]

#: The preset a synthetic deployment starts from: the simulator served in process,
#: no containers of its own.
PRESET = "hello-world"


def _as_command_line(command: click.Command) -> click.Command:
    """*command*, with its callback run inside ``facility_errors_as_click()``.

    Args:
        command: The subcommand to run.

    Returns:
        A copy of *command*; the original is left as it is.
    """
    from osprey.cli.main import facility_errors_as_click

    callback = command.callback
    assert callback is not None

    @functools.wraps(callback)
    def adapted(*args: object, **kwargs: object) -> object:
        with facility_errors_as_click():
            return callback(*args, **kwargs)

    wrapped = copy.copy(command)
    wrapped.callback = adapted
    return wrapped


def init_repo(repo: Path) -> Result:
    """Run ``osprey init`` for :data:`PRESET` into *repo*, without git.

    Args:
        repo: The repository to create; it must not exist.

    Returns:
        The runner's result.
    """
    from osprey.cli.init_cmd import init

    return CliRunner().invoke(_as_command_line(init), [str(repo), "--preset", PRESET, "--no-git"])


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
    return CliRunner().invoke(
        _as_command_line(build), [*repo_args, "--skip-deps", "--skip-lifecycle", *args]
    )
