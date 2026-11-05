"""Every ``osprey`` command the docs write is one click registers.

The pages are written by hand, so they drift from the command tree: a verb is
retired or renamed and a page keeps sending readers to it, an option is
dropped, a choice loses a value. Two checks read the pages against the tree
``osprey.cli.main.cli`` registers:

* every page under ``docs/source`` names only registered verbs, options of the
  command it names, and values a fixed ``click.Choice`` accepts;
* ``docs/source/reference/cli.rst`` also has a row for every visible verb, so a
  new verb cannot ship without one.

Matching rule
-------------
Only two places on a page are read as commands, never prose:

* an inline literal that starts with the word ``osprey`` (````osprey build````);
* a line inside a literal block (``::`` or ``.. code-block::``) that starts with
  the word ``osprey``, after an optional ``$`` prompt. A line ending in ``\\``
  continues on the next one. A shell line is split at ``&&``, ``||``, ``;`` and
  ``|``, and each part that starts with ``osprey`` (after ``uv run``, ``sudo``
  or ``VAR=value`` prefixes) is a reference of its own.

A ``#`` word ends the reference: the rest is a comment.

From each, the words after ``osprey`` are followed down the tree while they
name a subcommand of a group. The walk stops at a leaf command, at an option,
or at a placeholder (an upper-case word, a bracket, a pipe). A lower-case word
under a group that names none of its subcommands is a verb the tree does not
register, and fails.

The words after the walk are then read for options. A word that starts with
``-`` (brackets stripped, ``-d|--detached`` split at the pipe, ``--opt=value``
split at the ``=``) must be an option of the command the walk reached, or a
global option of ``osprey``. When that option's type is a fixed
``click.Choice``, the value after it, or each ``|`` alternative of it, must be
one of its choices unless it is a placeholder. Choice types that resolve their
values at parse time (a registry an application extends) are not checked.

A verb counts as documented on the CLI page when some reference resolves to
exactly that verb. A group is a verb of its own: a row for
``osprey web sessions clear`` does not document ``osprey web sessions``.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import click
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The docs tree every page is read from.
_DOCS = _REPO_ROOT / "docs" / "source"

#: The page that must document every verb, relative to the repo root.
_PAGE = "docs/source/reference/cli.rst"

#: Problems the guard reports that are correct as written, keyed by
#: ``"<page relative to docs/source>: <problem>"``, each with the reason.
#: Empty by design: fix the page instead. An entry that no longer matches a
#: problem fails ``test_every_exemption_is_still_needed``.
_EXEMPTIONS: dict[str, str] = {}

#: An RST inline literal; the content may wrap onto the next line.
_LITERAL_PATTERN = re.compile(r"``([^`]+)``")

#: A word that can name a subcommand.
_VERB_WORD = re.compile(r"^[a-z][a-z0-9-]*$")

#: A word that names an option: ``-d``, ``--repo``, ``--no-cache``.
_OPTION_WORD = re.compile(r"^--?[A-Za-z][A-Za-z0-9-]*$")

#: A literal choice value; anything else (``MODE``, ``<x>``, ``$VAR``) is a
#: placeholder.
_CHOICE_WORD = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")

#: Shell words that end one command and may start the next.
_SHELL_SEPARATORS = frozenset({"&&", "||", ";", "|"})

#: Shell words that end a command without starting another.
_SHELL_ENDS = frozenset({">", ">>", "2>", "2>&1", "&", "<"})

#: Words that may precede ``osprey`` on a shell line.
_PREFIXES = (("uv", "run"), ("sudo",))

#: A ``VAR=value`` environment assignment before a command.
_ENV_ASSIGNMENT = re.compile(r"^[A-Z_][A-Z0-9_]*=\S*$")

#: A code directive, or a paragraph ending in ``::``, that opens a literal block.
#: Other directives (``.. note::``) hold prose and open nothing.
_BLOCK_OPENER = re.compile(
    r"^(\s*)(\.\. (code-block|code|sourcecode)::.*|(?!\.\.\s)(?:\S.*)?::)\s*$"
)

#: The opening of a command substitution, ``$(`` or ``(``, optionally after a
#: ``var=`` assignment: ``result=$(osprey query --json ...)``.
_SUBSTITUTION = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_]*=)?\$?\(")


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


def _joined_block_lines(lines: list[str]) -> list[tuple[int, str]]:
    """Block lines with ``\\`` continuations joined, numbered by their first line."""
    joined: list[tuple[int, str]] = []
    pending: tuple[int, str] | None = None
    previous = -1
    for line, text in _block_lines(lines):
        if pending is not None and line == previous + 1:
            pending = (pending[0], f"{pending[1]} {text}")
        else:
            if pending is not None:
                joined.append(pending)
            pending = (line, text)
        previous = line
        if not pending[1].endswith("\\"):
            joined.append(pending)
            pending = None
        else:
            pending = (pending[0], pending[1][:-1].rstrip())
    if pending is not None:
        joined.append(pending)
    return joined


def _commands(words: list[str]) -> list[list[str]]:
    """The words after ``osprey`` of every command on one shell line."""
    found: list[list[str]] = []
    segment: list[str] = []
    for word in [*words, ";"]:
        if word not in _SHELL_SEPARATORS:
            segment.append(word)
            continue
        segment = _unwrap_substitution(segment)
        while segment and _ENV_ASSIGNMENT.match(segment[0]):
            segment = segment[1:]
        for prefix in _PREFIXES:
            if tuple(segment[: len(prefix)]) == prefix:
                segment = segment[len(prefix) :]
        if segment and segment[0] == "osprey":
            found.append(_until_comment(segment[1:]))
        segment = []
    return found


def _unwrap_substitution(segment: list[str]) -> list[str]:
    """The command inside a leading ``$(...)`` or ``(...)``, else ``segment``."""
    if not segment or not _SUBSTITUTION.match(segment[0]):
        return segment
    inner = [_SUBSTITUTION.sub("", segment[0], count=1), *segment[1:]]
    inner = [word for word in inner[:1] if word] + inner[1:]
    for index, word in enumerate(inner):
        if word.endswith(")"):
            return [*inner[:index], word[:-1]] if word[:-1] else inner[:index]
    return inner


def _until_comment(words: list[str]) -> list[str]:
    """``words`` up to a ``#`` comment or an output redirection."""
    for index, word in enumerate(words):
        if word.startswith("#") or word in _SHELL_ENDS:
            return words[:index]
    return words


def _references(page_text: str) -> list[tuple[int, list[str]]]:
    """``(line number, words after osprey)`` for every command reference."""
    found: list[tuple[int, list[str]]] = []
    for match in _LITERAL_PATTERN.finditer(page_text):
        commands = _commands(match.group(1).split())
        if commands:
            line = page_text.count("\n", 0, match.start()) + 1
            found.extend((line, command) for command in commands)
    for line, text in _joined_block_lines(page_text.splitlines()):
        words = text.split()
        if words and words[0] == "$":
            words = words[1:]
        found.extend((line, command) for command in _commands(words))
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


def _options(command: click.Command) -> dict[str, click.Option]:
    """Every spelling of every option ``command`` declares, ``--help`` included."""
    found: dict[str, click.Option] = {}
    for param in command.get_params(click.Context(command)):
        if isinstance(param, click.Option):
            for spelling in (*param.opts, *param.secondary_opts):
                found[spelling] = param
    return found


def _option_problems(command: click.Command, words: list[str]) -> list[str]:
    """What is wrong with the options and choice values in ``words``.

    ``words`` follow the command the walk reached. Global options of
    ``osprey`` are accepted anywhere, as the docs write them either side of
    the verb.
    """
    from osprey.cli.main import cli

    known = {**_options(cli), **_options(command)}
    problems: list[str] = []
    for index, raw in enumerate(words):
        for word in raw.strip("[](){},").split("|"):
            spelling, equals, value = word.partition("=")
            if not _OPTION_WORD.match(spelling):
                continue
            option = known.get(spelling)
            if option is None:
                problems.append(f"no option {spelling}")
                continue
            # Only a fixed choice list: a subclass may resolve its values late.
            if type(option.type) is not click.Choice or option.is_flag:
                continue
            if not equals:
                value = words[index + 1] if index + 1 < len(words) else ""
            for choice in value.strip("[](){},").split("|"):
                if _CHOICE_WORD.match(choice) and choice not in option.type.choices:
                    problems.append(
                        f"{spelling} {choice} is not one of {', '.join(option.type.choices)}"
                    )
    return problems


def _problems(page_text: str) -> list[tuple[int, str]]:
    """``(line, problem)`` for every reference that does not resolve in full."""
    from osprey.cli.main import cli

    registered = _registered()
    global_flags = {spelling for spelling, option in _options(cli).items() if option.is_flag}
    found: list[tuple[int, str]] = []
    for line, words in _references(page_text):
        # ``osprey --verbose build``: a global flag before the verb.
        leading = 0
        while leading < len(words) and words[leading] in global_flags:
            leading += 1
        path, stray = _resolve(words[leading:], registered)
        named = " ".join(("osprey", *path))
        if stray is not None:
            found.append((line, f"{named} has no verb {stray}"))
            continue
        command = registered[path] if path else cli
        found.extend(
            (line, f"{named}: {problem}")
            for problem in _option_problems(command, words[leading + len(path) :])
        )
    return found


def _pages() -> list[Path]:
    return sorted(_DOCS.rglob("*.rst"))


@lru_cache(maxsize=1)
def _all_problems() -> tuple[tuple[str, int, str], ...]:
    """``(page relative to docs/source, line, problem)`` for every page."""
    return tuple(
        (page.relative_to(_DOCS).as_posix(), line, problem)
        for page in _pages()
        for line, problem in _problems(page.read_text(encoding="utf-8"))
    )


def _exemption_key(page: str, problem: str) -> str:
    return f"{page}: {problem}"


def _page_text() -> str:
    return (_REPO_ROOT / _PAGE).read_text(encoding="utf-8")


def test_every_reference_on_every_page_resolves() -> None:
    """Every ``osprey …`` any page writes names a registered verb, its options and choices."""
    broken = [
        f"docs/source/{page}:{line}: {problem}"
        for page, line, problem in _all_problems()
        if _exemption_key(page, problem) not in _EXEMPTIONS
    ]
    assert not broken, "the docs name commands click does not accept:\n" + "\n".join(broken)


def test_every_exemption_is_still_needed() -> None:
    """An exemption whose problem is gone is removed, so it cannot hide a new one."""
    stale = sorted(
        set(_EXEMPTIONS)
        - {_exemption_key(page, problem) for page, _line, problem in _all_problems()}
    )
    assert not stale, "exemptions that match no problem:\n" + "\n".join(stale)


def test_every_registered_verb_is_documented() -> None:
    """Every registered verb has a reference on the CLI page that resolves to exactly it."""
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


def test_shell_lines_split_into_commands() -> None:
    """Chains, prefixes, continuations and comments are read the way a shell reads them."""
    text = (
        ".. code-block:: bash\n"
        "\n"
        "   cd x && uv run osprey build --repo . | tee log  # then\n"
        "   osprey query \\\n"
        "      --json  # comment --bogus\n"
        "   OSPREY_X=1 osprey up -d > out\n"
        "   echo osprey up\n"
    )
    assert _references(text) == [
        (3, ["build", "--repo", "."]),
        (4, ["query", "--json"]),
        (6, ["up", "-d"]),
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("``osprey up [-d|--detached]``", []),
        ("``osprey --verbose build --repo x``", []),
        ("``osprey chat --effort high``", []),
        ("``osprey validate --drift error|warn``", []),
        ("``osprey chat --effort EFFORT``", []),
        ("``osprey deploy``", [(1, "osprey has no verb deploy")]),
        ("``osprey up --bogus``", [(1, "osprey up: no option --bogus")]),
        ("``osprey up -q``", [(1, "osprey up: no option -q")]),
        (
            "``osprey chat --effort=extreme``",
            [(1, "osprey chat: --effort extreme is not one of ")],
        ),
        (
            "``osprey validate --drift error|ignore``",
            [(1, "osprey validate: --drift ignore is not one of ")],
        ),
    ],
)
def test_bad_verbs_options_and_choices_fail(text: str, expected: list[tuple[int, str]]) -> None:
    """A dead verb, an undeclared option and a value outside a fixed choice are reported.

    Each expected message is a prefix, so the test does not restate a choice list.
    """
    found = _problems(text)
    assert [line for line, _problem in found] == [line for line, _prefix in expected]
    for (_line, problem), (_line_expected, prefix) in zip(found, expected, strict=True):
        assert problem.startswith(prefix), problem


def test_a_prose_directive_is_not_a_literal_block() -> None:
    """An indented ``.. note::`` holds prose: a sentence in it is not a command."""
    text = "Text\n\n   .. note::\n\n      osprey renders the files.\n"
    assert _references(text) == []


def test_substitutions_and_inline_prefixes_are_read() -> None:
    """``$(osprey ...)`` and an inline ``uv run osprey ...`` are references too."""
    text = (
        "Run ``uv run osprey build`` here.\n"
        "\n"
        ".. code-block:: bash\n"
        "\n"
        '   result=$(osprey query --json "x")\n'
        "   code=$(osprey health --bogus)\n"
    )
    assert _references(text) == [
        (1, ["build"]),
        (5, ["query", "--json", '"x"']),
        (6, ["health", "--bogus"]),
    ]
    assert _problems(text) == [(6, "osprey health: no option --bogus")]
