"""The CLI reference page and click's command tree name the same verbs.

``docs/source/reference/cli.rst`` is written by hand, so it drifts from the
command tree in both directions: a verb is retired and its row stays behind,
sending a reader to a command that no longer exists, or a verb is added and
never gets a row, so nobody learns it is there. This check walks the tree
``osprey.cli.main.cli`` registers and compares it with the page both ways.

Matching rule
-------------
Only two places on the page are read as commands, never prose:

* an inline literal that starts with the word ``osprey`` (````osprey build````);
* a line inside a literal block (``::`` or ``.. code-block::``) that starts with
  the word ``osprey``, after an optional ``$`` prompt.

From each, the words after ``osprey`` are followed down the tree while they
name a subcommand of a group. The walk stops at a leaf command, at an option,
or at a placeholder (an upper-case word, a bracket, a pipe). A lower-case word
under a group that names none of its subcommands is a verb the tree does not
register, and fails.

A verb counts as documented when some reference resolves to exactly that
verb. A group is a verb of its own: a row for ``osprey web sessions clear``
does not document ``osprey web sessions``.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import click
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The page this check reads, relative to the repo root.
_PAGE = "docs/source/reference/cli.rst"

#: An RST inline literal; the content may wrap onto the next line.
_LITERAL_PATTERN = re.compile(r"``([^`]+)``")

#: A word that can name a subcommand.
_VERB_WORD = re.compile(r"^[a-z][a-z0-9-]*$")

#: A code directive, or a paragraph ending in ``::``, that opens a literal block.
#: Other directives (``.. note::``) hold prose and open nothing.
_BLOCK_OPENER = re.compile(r"^(\s*)(\.\. (code-block|code|sourcecode)::.*|(?!\.\. ).*::)\s*$")


@lru_cache(maxsize=1)
def _registered() -> dict[tuple[str, ...], click.Command]:
    """Every command path click registers under ``osprey``, groups included."""
    from osprey.cli.main import cli

    found: dict[tuple[str, ...], click.Command] = {}

    def walk(group: click.Group, prefix: tuple[str, ...]) -> None:
        ctx = click.Context(group, info_name=prefix[-1] if prefix else "osprey")
        for name in group.list_commands(ctx):
            command = group.get_command(ctx, name)
            if command is None:
                continue
            path = (*prefix, name)
            found[path] = command
            if isinstance(command, click.Group):
                walk(command, path)

    walk(cli, ())
    return found


def _block_lines(lines: list[str]) -> list[tuple[int, str]]:
    """``(line number, text)`` for every line inside a literal block."""
    inside: list[tuple[int, str]] = []
    index = 0
    while index < len(lines):
        opener = _BLOCK_OPENER.match(lines[index])
        index += 1
        if opener is None:
            continue
        base = len(opener.group(1))
        # Directive options and blank lines precede the content.
        while index < len(lines) and (
            not lines[index].strip() or lines[index].strip().startswith(":")
        ):
            index += 1
        while index < len(lines):
            text = lines[index]
            if text.strip() and len(text) - len(text.lstrip()) <= base:
                break
            if text.strip():
                inside.append((index + 1, text.strip()))
            index += 1
    return inside


def _references(page_text: str) -> list[tuple[int, list[str]]]:
    """``(line number, words after osprey)`` for every command reference."""
    found: list[tuple[int, list[str]]] = []
    for match in _LITERAL_PATTERN.finditer(page_text):
        words = match.group(1).split()
        if words and words[0] == "osprey":
            line = page_text.count("\n", 0, match.start()) + 1
            found.append((line, words[1:]))
    for line, text in _block_lines(page_text.splitlines()):
        words = text.split()
        if words and words[0] == "$":
            words = words[1:]
        if words and words[0] == "osprey":
            found.append((line, words[1:]))
    return sorted(found)


def _resolve(
    words: list[str], registered: dict[tuple[str, ...], click.Command]
) -> tuple[tuple[str, ...], str | None]:
    """The command path ``words`` names, and the word that broke the walk if any."""
    path: tuple[str, ...] = ()
    for word in words:
        if not _VERB_WORD.match(word):
            break
        current = registered.get(path)
        if path and not isinstance(current, click.Group):
            break
        candidate = (*path, word)
        if candidate not in registered:
            return path, word
        path = candidate
    return path, None


def _page_text() -> str:
    return (_REPO_ROOT / _PAGE).read_text(encoding="utf-8")


def test_every_reference_resolves() -> None:
    """Every ``osprey …`` the page writes names a registered verb."""
    registered = _registered()
    broken = []
    for line, words in _references(_page_text()):
        path, stray = _resolve(words, registered)
        if stray is not None:
            named = " ".join(("osprey", *path, stray))
            broken.append(f"{_PAGE}:{line}: {named}")
    assert not broken, "the page names verbs click does not register:\n" + "\n".join(broken)


def test_every_registered_verb_is_documented() -> None:
    """Every registered verb has a reference that resolves to exactly it."""
    registered = _registered()
    documented = set()
    for _line, words in _references(_page_text()):
        path, stray = _resolve(words, registered)
        if stray is None and path:
            documented.add(path)
    missing = sorted(
        " ".join(("osprey", *path))
        for path, command in registered.items()
        if path not in documented and not command.hidden
    )
    assert not missing, f"{_PAGE} has no row for:\n" + "\n".join(missing)


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        (["build", "--repo", "x"], (("build",), None)),
        (["set", "config.system.timezone=UTC"], (("set",), None)),
        (["web", "sessions", "clear"], (("web", "sessions", "clear"), None)),
        (["facility", "import", "mml", "EXPORT..."], (("facility", "import", "mml"), None)),
        (["facility", "nonexistent"], (("facility",), "nonexistent")),
        (["nonexistent"], ((), "nonexistent")),
        (["--version"], ((), None)),
    ],
)
def test_resolve_walks_the_tree(
    words: list[str], expected: tuple[tuple[str, ...], str | None]
) -> None:
    """Placeholders and options stop the walk; an unknown verb under a group breaks it."""
    assert _resolve(words, _registered()) == expected


def test_prose_is_not_a_reference() -> None:
    """A sentence that begins with the word osprey is not read as a command."""
    text = "osprey rendered it.\n\nRun ``osprey build`` first.\n\n::\n\n   osprey up -d\n"
    assert _references(text) == [(3, ["build"]), (7, ["up", "-d"])]
