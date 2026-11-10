"""Every ``osprey`` verb a plugin skill cites is a live command.

A skill is prose an agent follows literally: a step that names a removed or
retired verb fails in front of a user, mid-interview, with nothing in CI to say
so beforehand. The docs pages are held to the click tree by
``tests/docs/test_cli_page_commands.py``; this holds the plugin's skills to it.

What is read: every inline code span and every line of a fenced block in the
Markdown under ``plugins/osprey/skills/``. A span or line that starts with
``osprey`` is walked down the command tree word by word. The walk stops at the
first word that is an option, a placeholder or an argument, so only the command
path is judged:

* a word after a command *group* that is not one of its subcommands is a dead
  verb;
* a command whose help opens with ``Retired`` is a stub kept to print a remedy,
  not something to send a user to.
"""

from __future__ import annotations

import re
from pathlib import Path

import click
import pytest

from osprey.cli.main import cli

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SKILLS = _REPO_ROOT / "plugins" / "osprey" / "skills"

_INLINE = re.compile(r"`([^`\n]+)`")
_FENCE = re.compile(r"^\s*```")
#: A word that can be a command name. Anything else (``--flag``, ``<name>``,
#: ``KEY=VALUE``, a path) ends the command path.
_VERB = re.compile(r"^[a-z][a-z0-9-]*$")


def _cited_commands(text: str) -> list[tuple[int, str]]:
    """Return ``(line number, command text)`` for every ``osprey ...`` citation."""
    cited: list[tuple[int, str]] = []
    fenced = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _FENCE.match(line):
            fenced = not fenced
            continue
        candidates = [line.strip().removeprefix("$ ")] if fenced else _INLINE.findall(line)
        cited.extend((lineno, c) for c in candidates if c.startswith("osprey "))
    return cited


def _problem(command_text: str) -> str | None:
    """Say what is wrong with one cited command, or ``None`` when it is live."""
    command: click.Command = cli
    ctx = click.Context(cli, info_name="osprey")
    path = ["osprey"]
    for word in command_text.split()[1:]:
        if not _VERB.match(word) or not isinstance(command, click.Group):
            break
        sub = command.get_command(ctx, word)
        if sub is None:
            return f"`{' '.join(path)}` has no subcommand `{word}`"
        command = sub
        ctx = click.Context(command, info_name=word, parent=ctx)
        path.append(word)
    if (command.help or "").lstrip().startswith("Retired"):
        return f"`{' '.join(path)}` is retired: {(command.help or '').strip().splitlines()[0]}"
    return None


def _offenders(root: Path) -> list[str]:
    return [
        f"{path.relative_to(root)}:{lineno}: {problem}"
        for path in sorted(root.rglob("*.md"))
        for lineno, command_text in _cited_commands(path.read_text(encoding="utf-8"))
        if (problem := _problem(command_text)) is not None
    ]


def test_every_verb_a_skill_cites_is_live() -> None:
    offenders = _offenders(_SKILLS)
    assert not offenders, "Plugin skills cite dead or retired verbs:\n" + "\n".join(offenders)


@pytest.mark.parametrize(
    ("citation", "expected"),
    [
        ("Run `osprey scaffold apps --from demo`.", "is retired"),
        ("Run `osprey scaffold no-such-verb`.", "has no subcommand"),
        ("```\nosprey no-such-group list\n```", "has no subcommand"),
    ],
)
def test_the_sweep_would_catch_a_dead_verb(tmp_path: Path, citation: str, expected: str) -> None:
    """The guard is only worth having if it fails on a dead citation."""
    (tmp_path / "SKILL.md").write_text(citation + "\n", encoding="utf-8")

    offenders = _offenders(tmp_path)

    assert len(offenders) == 1 and expected in offenders[0], offenders


def test_the_sweep_reads_past_options_and_arguments(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text(
        "`osprey init <dir> --preset hello-world`, `osprey set config.a.b=1`,\n"
        "`osprey scaffold pull demo:data/facility/knowledge` and `osprey --version`.\n",
        encoding="utf-8",
    )

    assert _offenders(tmp_path) == []
