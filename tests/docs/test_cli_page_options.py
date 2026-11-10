"""The CLI reference page names every option click accepts, in the right section.

``docs/source/reference/cli.rst`` is written by hand. ``test_cli_page_commands``
checks that it names the same verbs as the click tree; this check goes one level
down and asks whether each verb's section names every option the verb takes. An
option added to a command and never written up is one nobody learns about; the
overview list at the top of the page drifts the same way when a verb is added.

Sectioning rule
---------------
A heading is a title line followed by an underline of one repeated punctuation
character at least as long as the title. A title with an overline as well (the
page title) is not a section. Levels follow reStructuredText: each underline
character gets the next level the first time it appears.

A heading's *section* is the text from the heading to the next heading of the
same or a higher level, so a section holds its subsections. The *lead* of a
heading is the text from the heading to its first heading of any level.

A definition-list *entry* is a term line that is a single inline literal
starting ``osprey`` (````osprey ariel enhance [--force]````), together with the
indented body under it: everything up to the next non-blank line indented no
deeper than the term. The entry documents the command its words resolve to.

The text that documents a command is:

* the section whose title is ``osprey <command path>``, when there is one;
* otherwise, when the command has entries in the section of its nearest
  titled ancestor (``osprey ariel`` for ``osprey ariel enhance``), those
  entries plus that section's text before its first entry, where a note shared
  by every subcommand lives (``osprey scaffold`` names ``--repo`` there once);
* otherwise the whole section of that ancestor;
* plus, in every case, the lead of every heading that encloses the section,
  where a note shared by several titled verbs lives ("Lifecycle verbs" names
  ``--repo`` once for ``up``, ``down`` and the rest).

Remaining false negatives: a command with a heading of its own is checked
against its whole section, subsections and subcommand entries included, so a
group's option is found if only a subcommand's text names it. A command with
neither a heading nor an entry is checked against its ancestor's whole section.

Matching rule
-------------
Every name of every non-hidden ``click.Option`` of a non-hidden command
(``opts`` and ``secondary_opts``: ``-d`` as well as ``--detached``) must occur
in that text as a whole token: not preceded or followed by a word character
or a hyphen, so ``--dev`` is not found inside ``--dev-mode``. Hidden options and
hidden commands are left out. Arguments are not checked.
"""

from __future__ import annotations

import re
from pathlib import Path

import click
import pytest

from tests.docs.test_cli_page_commands import _registered, _resolve

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The page this check reads, relative to the repo root.
_PAGE = "docs/source/reference/cli.rst"

#: A heading underline (or overline): one punctuation character, repeated.
_ADORNMENT = re.compile(r"^([=\-~^*\"#+`'.:_])\1{2,}\s*$")

#: A verb in the overview block, at the start of a block line.
_OVERVIEW_VERB = re.compile(r"^\s+osprey ([a-z][a-z0-9-]*)")

#: A definition-list term that is one inline literal naming a command.
_ENTRY_TERM = re.compile(r"^(\s*)``(osprey [^`]+)``\s*$")


def _headings(lines: list[str]) -> list[tuple[int, str, int]]:
    """``(line index, title, level)`` for every section heading, in page order."""
    found: list[tuple[int, str, int]] = []
    styles: list[str] = []
    for index in range(1, len(lines)):
        underline = _ADORNMENT.match(lines[index])
        title = lines[index - 1].strip()
        if underline is None or not title or _ADORNMENT.match(lines[index - 1]):
            continue
        if len(lines[index].rstrip()) < len(title):
            continue
        if index >= 2 and lines[index - 2].rstrip() == lines[index].rstrip():
            continue  # overlined: the page title, not a section
        style = underline.group(1)
        if style not in styles:
            styles.append(style)
        found.append((index - 1, title, styles.index(style)))
    return found


def _section(lines: list[str], title: str) -> tuple[tuple[int, int], list[str]] | None:
    """The ``(start, end)`` line span titled ``title``, and its enclosing leads."""
    headings = _headings(lines)
    for position, (start, heading, level) in enumerate(headings):
        if heading != title:
            continue
        end = next(
            (line for line, _t, other in headings[position + 1 :] if other <= level),
            len(lines),
        )
        leads = []
        wanted = level
        for back in range(position - 1, -1, -1):
            outer_start, _t, outer_level = headings[back]
            if outer_level < wanted:
                leads.append("\n".join(lines[outer_start : headings[back + 1][0]]))
                wanted = outer_level
        return (start, end), leads
    return None


def _command_text(lines: list[str], title: str) -> str | None:
    """The section titled ``title`` plus the leads of its enclosing headings."""
    found = _section(lines, title)
    if found is None:
        return None
    (start, end), leads = found
    return "\n".join(["\n".join(lines[start:end]), *leads])


def _entries(lines: list[str], start: int, end: int) -> list[tuple[int, int, tuple[str, ...]]]:
    """``(first line, end line, command path)`` of each entry in ``lines[start:end]``."""
    registered = _registered()
    found = []
    for index in range(start, end):
        term = _ENTRY_TERM.match(lines[index])
        if term is None:
            continue
        depth = len(term.group(1))
        stop = index + 1
        while stop < end:
            text = lines[stop]
            if text.strip() and len(text) - len(text.lstrip()) <= depth:
                break
            stop += 1
        path, stray = _resolve(term.group(2).split()[1:], registered)
        if stray is None:
            found.append((index, stop, path))
    return found


def _documenting_text(lines: list[str], path: tuple[str, ...]) -> tuple[str, str] | None:
    """``(section title, text)`` that documents the command at ``path``, per the rules."""
    titles = {title for _i, title, _l in _headings(lines)}
    owner = path
    while owner and " ".join(("osprey", *owner)) not in titles:
        owner = owner[:-1]
    if not owner:
        return None
    title = " ".join(("osprey", *owner))
    found = _section(lines, title)
    assert found is not None
    (start, end), leads = found
    parts = ["\n".join(lines[start:end])]
    if owner != path:
        entries = _entries(lines, start, end)
        own = [(first, stop) for first, stop, entry in entries if entry == path]
        if own:
            parts = ["\n".join(lines[start : entries[0][0]])]
            parts += ["\n".join(lines[first:stop]) for first, stop in own]
    return title, "\n".join([*parts, *leads])


def _names_option(text: str, name: str) -> bool:
    """Whether ``name`` (``--repo``, ``-d``) occurs in ``text`` as a whole token."""
    return re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text) is not None


def _undocumented(lines: list[str]) -> list[str]:
    """One line per visible option of a visible command the page does not name."""
    missing = []
    for path, command in sorted(_registered().items()):
        if command.hidden:
            continue
        command_name = " ".join(("osprey", *path))
        found = _documenting_text(lines, path)
        if found is None:
            missing.append(f"{command_name}: no section titled for it or a parent")
            continue
        title, text = found
        for param in command.params:
            if not isinstance(param, click.Option) or param.hidden:
                continue
            for name in (*param.opts, *param.secondary_opts):
                if not _names_option(text, name):
                    missing.append(f"{command_name} {name} (section {title!r})")
    return missing


def _lines() -> list[str]:
    return (_REPO_ROOT / _PAGE).read_text(encoding="utf-8").splitlines()


def test_every_visible_option_is_named_in_its_section() -> None:
    """Each non-hidden option of each non-hidden command appears in its text."""
    missing = _undocumented(_lines())
    assert not missing, f"{_PAGE} does not name these options:\n" + "\n".join(missing)


def test_a_sibling_entry_does_not_document_an_option() -> None:
    """Negative control: ``ariel ingest`` naming ``--limit`` does not cover ``enhance``."""
    lines = _lines()
    term = "``osprey ariel enhance [-m, --module NAME] [--force] [--limit N]"
    edited = [line.replace(" [--limit N]", "") if line.startswith(term) else line for line in lines]
    assert edited != lines, "the ariel enhance entry this control edits has moved"
    assert "osprey ariel enhance --limit (section 'osprey ariel')" in _undocumented(edited)


def test_overview_lists_every_top_level_verb() -> None:
    """The overview block at the top of the page names every top-level verb."""
    lines = _lines()
    overview = _command_text(lines, "Overview")
    assert overview is not None, f"{_PAGE} has no Overview section"
    listed = {
        match.group(1) for line in overview.splitlines() if (match := _OVERVIEW_VERB.match(line))
    }
    missing = sorted(
        path[0]
        for path, command in _registered().items()
        if len(path) == 1 and not command.hidden and path[0] not in listed
    )
    assert not missing, f"the {_PAGE} overview block does not list:\n" + "\n".join(missing)


_SAMPLE = """\
=====
Title
=====

Intro naming ``--title-only``.

Group
=====

Group lead naming ``--shared``.

osprey a
--------

``--alpha`` and ``-a``.

osprey b
--------

``--beta``.

Other
=====

``--other``.
"""


def test_headings_skip_the_overlined_title() -> None:
    assert [(title, level) for _i, title, level in _headings(_SAMPLE.splitlines())] == [
        ("Group", 0),
        ("osprey a", 1),
        ("osprey b", 1),
        ("Other", 0),
    ]


def test_command_text_is_the_section_plus_enclosing_leads() -> None:
    text = _command_text(_SAMPLE.splitlines(), "osprey a")
    assert text is not None
    assert "--alpha" in text and "--shared" in text
    assert "--beta" not in text and "--other" not in text


@pytest.mark.parametrize(
    ("text", "name", "expected"),
    [
        ("``--dev`` — dev mode", "--dev", True),
        ("``--dev-mode``", "--dev", False),
        ("``--no-dev``", "--dev", False),
        ("``-d, --detached``", "-d", True),
        ("``-d/--detached``", "--detached", True),
        ("``--depth``", "-d", False),
    ],
)
def test_names_option_matches_whole_tokens(text: str, name: str, expected: bool) -> None:
    assert _names_option(text, name) is expected
