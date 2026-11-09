"""The facility-knowledge guide shows the help ``osprey knowledge`` prints.

``docs/source/how-to/facility-knowledge/okf-bundle.rst`` includes
``examples/knowledge-help.out`` as the group's help screen. The file must be
exactly what the CLI prints, so a verb added, removed or reworded fails here
first. The page's count of operations must match the group's verbs too.
"""

from __future__ import annotations

import re
from pathlib import Path

import click
from click.testing import CliRunner

from osprey.cli.main import cli

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PAGE = _REPO_ROOT / "docs" / "source" / "how-to" / "facility-knowledge" / "okf-bundle.rst"
_OUT = _PAGE.parent / "examples" / "knowledge-help.out"

#: Width the help is wrapped at, pinned so the comparison does not depend on the terminal.
_WIDTH = 80

_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def test_help_file_is_what_the_cli_prints() -> None:
    result = CliRunner().invoke(
        cli, ["knowledge", "--help"], prog_name="osprey", terminal_width=_WIDTH
    )
    assert result.exit_code == 0, result.output
    assert _OUT.read_text(encoding="utf-8") == result.output


def test_page_includes_the_help_file() -> None:
    assert ".. literalinclude:: examples/knowledge-help.out" in _PAGE.read_text(encoding="utf-8")


def test_page_counts_the_groups_verbs() -> None:
    ctx = click.Context(cli)
    group = cli.get_command(ctx, "knowledge")
    assert isinstance(group, click.Group)
    text = " ".join(_PAGE.read_text(encoding="utf-8").split())
    match = re.search(r"``osprey knowledge`` command group provides (\w+) operations", text)
    assert match is not None
    assert _NUMBERS[match.group(1)] == len(group.list_commands(ctx))
