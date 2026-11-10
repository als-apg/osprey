"""Suite for the by-hand lines of the project-image how-to.

The reader types every command on the page at the deployment repo root, where
``.env`` and the sender login files live. A ``docker build`` there names the
container repo ``osprey build`` rendered (``build/.image/<name>/``) as its
context and the recipe inside it by path, so no line moves the shell and none
has to be undone before ``docker run`` finds its ``--env-file`` arguments.
"""

from __future__ import annotations

import shlex
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PAGE = _REPO_ROOT / "docs" / "source" / "how-to" / "deploy-project" / "project-image.rst"

_CONTEXT = "build/.image/my-project"
_RECIPE = f"{_CONTEXT}/build/Dockerfile"


def _bash_commands(text: str) -> list[str]:
    """Return every command of the page's bash blocks, continuations joined."""
    commands: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() != ".. code-block:: bash":
            i += 1
            continue
        i += 1
        pending = ""
        while i < len(lines) and (not lines[i].strip() or lines[i].startswith(" ")):
            line = lines[i].strip()
            i += 1
            if not line:
                continue
            if line.endswith("\\"):
                pending += line[:-1] + " "
                continue
            commands.append(pending + line)
            pending = ""
        if pending:
            commands.append(pending)
    return commands


def _docker_builds() -> list[list[str]]:
    builds = []
    for command in _bash_commands(_PAGE.read_text()):
        if command.startswith("docker build"):
            builds.append(shlex.split(command, comments=True))
    return builds


def test_no_by_hand_line_changes_directory():
    commands = _bash_commands(_PAGE.read_text())
    assert commands
    offenders = [c for c in commands if c.startswith("cd ") or c == "cd"]
    assert not offenders, offenders


def test_every_docker_build_names_the_rendered_context_and_recipe():
    builds = _docker_builds()
    assert len(builds) >= 3, builds
    for argv in builds:
        assert "-f" in argv, argv
        assert argv[argv.index("-f") + 1] == _RECIPE, argv
        assert argv[-1] == _CONTEXT, argv
