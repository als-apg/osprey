"""Every code block on every docs page parses in the language it declares.

A ``.. code-block:: bash`` that bash cannot parse, or a ``json`` block that is
not JSON, renders without a warning: Sphinx highlights what it can and moves
on. A reader who pastes it gets a syntax error instead of the result the page
promised. This check reads every ``*.rst`` under ``docs/source`` and hands each
block to its language's own parser. It never reads prose, so a page changes
its result only by adding or editing a block.

Block scanning
--------------
A block opens on a ``.. code-block::``/``.. code::``/``.. sourcecode::``
directive, whose argument names the language, or on a paragraph ending in
``::`` (language ``literal``). Directive options (``:caption:``) and blank
lines before the content are not content. The block ends at the first
non-blank line indented no deeper than its opener, so a nested directive such
as ``.. note::`` that follows it closes it. The shape is the one
``tests/docs/test_cli_page_commands.py`` uses.

Checked languages
-----------------
* ``bash`` -- a leading ``$ `` prompt is stripped from each line and every
  ``<placeholder>`` (an angle-bracketed phrase starting with a letter, such
  as ``<package>`` or ``<the digest show printed>``) becomes one word, since
  the reader replaces it; then ``bash -n`` parses the block. Skipped, with the reason, where no ``bash`` is
  on ``PATH``.
* ``console`` -- only lines starting with ``$ `` are commands (with the lines
  a trailing ``\\`` continues); everything else is output. The commands go to
  ``bash -n`` as a ``bash`` block's lines do.
* ``json`` -- :func:`json.loads`.
* ``toml`` -- :func:`tomllib.loads`.
* ``ini`` -- :class:`configparser.ConfigParser` with ``strict=False`` (a systemd
  unit may repeat a key) and no interpolation (a ``%`` is literal).
* ``python`` -- :func:`ast.parse`.
* ``jinja`` -- ``jinja2.Environment().parse``.
* ``dockerfile`` -- every non-blank, non-comment logical line (``\\``
  continuations joined) starts with a Dockerfile instruction word.

Not checked
-----------
``yaml`` (its keys are checked against the config key manifest by the YAML
guard, which owns those blocks), ``sql``, ``cypher``, ``html``,
``javascript``, ``markdown``, ``text``, ``bibtex`` and ``literal`` (``::``
blocks, which carry no language). ``literalinclude`` files are not page
blocks; their own tests load them. A language outside both lists fails
:func:`test_every_language_is_classified`, so a new one is placed on purpose.
"""

from __future__ import annotations

import ast
import configparser
import json
import re
import shutil
import subprocess
import textwrap
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import jinja2
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS_SOURCE = _REPO_ROOT / "docs" / "source"

#: A code directive, or a paragraph ending in ``::``, that opens a literal block.
#: Other directives (``.. note::``) hold prose and open nothing.
_BLOCK_OPENER = re.compile(r"^(\s*)(\.\. (code-block|code|sourcecode)::(.*)|(?!\.\. )\S.*::)\s*$")

#: A directive option line, ``:caption: foo`` or ``:linenos:``.
_OPTION_LINE = re.compile(r"^:[\w-]+:")

#: A ``<placeholder>`` the reader replaces; bash would read its ``<`` and ``>``
#: as redirections.
_PLACEHOLDER = re.compile(r"<[A-Za-z][^<>\n]*>")

#: Languages this check does not parse; see the module docstring for why.
_UNCHECKED = frozenset(
    {"yaml", "sql", "cypher", "html", "javascript", "markdown", "text", "bibtex", "literal"}
)

#: Dockerfile instruction words (https://docs.docker.com/reference/dockerfile/).
_DOCKERFILE_INSTRUCTIONS = frozenset(
    {
        "ADD",
        "ARG",
        "CMD",
        "COPY",
        "ENTRYPOINT",
        "ENV",
        "EXPOSE",
        "FROM",
        "HEALTHCHECK",
        "LABEL",
        "MAINTAINER",
        "ONBUILD",
        "RUN",
        "SHELL",
        "STOPSIGNAL",
        "USER",
        "VOLUME",
        "WORKDIR",
    }
)

#: Blocks known not to parse, keyed ``(page, first content line)`` with the
#: reason.
_ALLOWED_FAILURES: dict[tuple[str, str], str] = {
    ("how-to/ariel/index.rst", "osprey chat"): (
        "a chat transcript declared as bash: its second line is the prompt"
        " ``>>> What does the logbook say ...``, not a command; the page is"
        " owned by an open docs PR, which can declare it ``text``"
    ),
}


@dataclass(frozen=True)
class Block:
    """One literal block: where it is, what it declares, what it holds."""

    page: str
    line: int
    language: str
    text: str

    @property
    def first_line(self) -> str:
        return self.text.splitlines()[0].strip()


def scan(text: str, page: str) -> list[Block]:
    """Every non-empty literal block on one page, in order."""
    lines = text.splitlines()
    blocks: list[Block] = []
    index = 0
    while index < len(lines):
        opener = _BLOCK_OPENER.match(lines[index])
        index += 1
        if opener is None:
            continue
        base = len(opener.group(1))
        is_directive = opener.group(3) is not None
        language = (opener.group(4) or "").strip() if is_directive else "literal"
        while index < len(lines) and (
            not lines[index].strip() or (is_directive and _OPTION_LINE.match(lines[index].strip()))
        ):
            index += 1
        start = index
        body: list[str] = []
        while index < len(lines):
            line = lines[index]
            if line.strip() and len(line) - len(line.lstrip()) <= base:
                break
            body.append(line)
            index += 1
        content = textwrap.dedent("\n".join(body)).strip("\n")
        if content:
            blocks.append(Block(page, start + 1, language or "literal", content))
    return blocks


def scan_tree(root: Path) -> list[Block]:
    """Every block on every page under ``root``."""
    found: list[Block] = []
    for path in sorted(root.rglob("*.rst")):
        page = path.relative_to(root).as_posix()
        found.extend(scan(path.read_text(encoding="utf-8"), page))
    return found


def _bash_n(source: str) -> str | None:
    result = subprocess.run(
        ["bash", "-n"],
        input=_PLACEHOLDER.sub("PLACEHOLDER", source),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return None
    return result.stderr.strip() or f"bash -n exited {result.returncode}"


def check_bash(text: str) -> str | None:
    return _bash_n("\n".join(re.sub(r"^\$ ", "", line) for line in text.splitlines()))


def check_console(text: str) -> str | None:
    commands: list[str] = []
    continued = False
    for line in text.splitlines():
        if line.startswith("$ "):
            commands.append(line[2:])
        elif continued:
            commands.append(line)
        else:
            continue
        continued = line.rstrip().endswith("\\")
    return _bash_n("\n".join(commands))


def _raises(parse: Callable[[str], object]) -> Callable[[str], str | None]:
    def check(text: str) -> str | None:
        try:
            parse(text)
        except Exception as error:  # each parser raises its own error type
            return f"{type(error).__name__}: {error}"
        return None

    return check


def _parse_ini(text: str) -> None:
    configparser.ConfigParser(strict=False, interpolation=None).read_string(text)


def check_dockerfile(text: str) -> str | None:
    logical: list[str] = []
    pending = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not pending and (not stripped or stripped.startswith("#")):
            continue
        pending = f"{pending} {stripped}" if pending else stripped
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        logical.append(pending)
        pending = ""
    if pending:
        logical.append(pending)
    for line in logical:
        word = line.split()[0]
        if word.upper() not in _DOCKERFILE_INSTRUCTIONS:
            return f"not a Dockerfile instruction: {word!r} in {line!r}"
    return None


#: Language -> checker returning ``None`` when the block parses, else the error.
CHECKERS: dict[str, Callable[[str], str | None]] = {
    "bash": check_bash,
    "console": check_console,
    "json": _raises(json.loads),
    "toml": _raises(tomllib.loads),
    "ini": _raises(_parse_ini),
    "python": _raises(ast.parse),
    "jinja": _raises(jinja2.Environment().parse),
    "dockerfile": check_dockerfile,
}

#: Languages whose checker runs ``bash``.
_NEEDS_BASH = frozenset({"bash", "console"})

_NO_BASH = pytest.mark.skipif(shutil.which("bash") is None, reason="no bash on PATH")


@lru_cache(maxsize=1)
def _live_blocks() -> tuple[Block, ...]:
    return tuple(scan_tree(_DOCS_SOURCE))


def _checked_params() -> list[object]:
    params: list[object] = []
    for block in _live_blocks():
        if block.language not in CHECKERS:
            continue
        marks = [_NO_BASH] if block.language in _NEEDS_BASH else []
        params.append(
            pytest.param(block, id=f"{block.page}:{block.line}:{block.language}", marks=marks)
        )
    return params


@pytest.mark.parametrize("block", _checked_params())
def test_block_parses(block: Block) -> None:
    """The block parses in the language its directive declares."""
    reason = _ALLOWED_FAILURES.get((block.page, block.first_line))
    if reason is not None:
        pytest.skip(f"allowlisted: {reason}")
    error = CHECKERS[block.language](block.text)
    assert error is None, (
        f"{block.page}:{block.line}: the {block.language} block does not parse:\n{error}"
    )


def test_every_language_is_classified() -> None:
    """Every language a page declares is either checked or listed as unchecked."""
    unknown = sorted(
        {
            f"{block.language} ({block.page}:{block.line})"
            for block in _live_blocks()
            if block.language not in CHECKERS and block.language not in _UNCHECKED
        }
    )
    assert not unknown, "add a checker to CHECKERS or the language to _UNCHECKED: " + ", ".join(
        unknown
    )


def test_the_scan_finds_the_checked_blocks() -> None:
    """A scanner that silently found nothing would pass every block test."""
    counts: dict[str, int] = {}
    for block in _live_blocks():
        counts[block.language] = counts.get(block.language, 0) + 1
    assert counts.get("bash", 0) >= 100
    assert counts.get("json", 0) >= 5
    assert counts.get("python", 0) >= 5


def test_allowlist_entries_are_live_failures() -> None:
    """An allowlisted block that is gone, or now parses, is removed from the list."""
    by_key = {(block.page, block.first_line): block for block in _live_blocks()}
    stale: list[str] = []
    for key in _ALLOWED_FAILURES:
        block = by_key.get(key)
        if block is None:
            stale.append(f"{key}: no such block")
        elif block.language not in CHECKERS or CHECKERS[block.language](block.text) is None:
            stale.append(f"{key}: parses now")
    assert not stale, "stale _ALLOWED_FAILURES entries: " + "; ".join(stale)


# -- negative controls ---------------------------------------------------------

#: One block per checked language that its parser refuses.
_BROKEN = {
    "bash": "if true; then\n  echo missing fi",
    "console": "$ for x in a b; do echo $x\nsome output line",
    "json": '{"key": "value",}',
    "toml": "[table\nkey = 1",
    "ini": "key_before_any_section = 1",
    "python": "def broken(:\n    pass",
    "jinja": "{% if x %}never closed",
    "dockerfile": "FROM python:3.11-slim\nRUN echo \\\n  continued\npip install foo",
}


def _page(language: str, body: str) -> str:
    indented = textwrap.indent(body, "   ")
    return f"Title\n=====\n\n.. code-block:: {language}\n   :caption: planted\n\n{indented}\n\nAfter.\n"


@pytest.mark.parametrize("language", sorted(_BROKEN))
def test_a_planted_error_fails(language: str, tmp_path: Path) -> None:
    """Each checker refuses a block its parser refuses, found through the scanner."""
    if language in _NEEDS_BASH and shutil.which("bash") is None:
        pytest.skip("no bash on PATH")
    assert set(_BROKEN) == set(CHECKERS)
    (tmp_path / "planted.rst").write_text(_page(language, _BROKEN[language]), encoding="utf-8")
    [block] = scan_tree(tmp_path)
    assert block.language == language
    assert block.text == _BROKEN[language]
    assert CHECKERS[language](block.text) is not None


def test_console_output_lines_are_not_parsed() -> None:
    """Output that is not shell syntax does not fail a console block."""
    if shutil.which("bash") is None:
        pytest.skip("no bash on PATH")
    text = "$ osprey knowledge --help \\\n    --flag\nUsage: osprey (OPTIONS) [ARGS]...\n  )( not shell"
    assert check_console(text) is None


def test_a_placeholder_is_a_word_not_a_redirection() -> None:
    """``<package>`` parses; a real syntax error next to it still fails."""
    if shutil.which("bash") is None:
        pytest.skip("no bash on PATH")
    assert check_bash("uv add <package>\ngit revert <accept commit>") is None
    assert check_bash("uv add <package> |") is not None


def test_the_scanner_stops_at_a_nested_directive_and_skips_options() -> None:
    """Options are not content; a less-indented directive closes the block."""
    text = (
        ".. tab-item:: One\n\n"
        "   .. code-block:: bash\n"
        "      :caption: run it\n\n"
        "      echo one\n\n"
        "      echo two\n\n"
        "   .. note::\n\n"
        "      prose, not a block\n\n"
        "A paragraph::\n\n"
        "   literal text\n"
    )
    blocks = scan(text, "page.rst")
    assert [(b.language, b.line, b.text) for b in blocks] == [
        ("bash", 6, "echo one\n\necho two"),
        ("literal", 16, "literal text"),
    ]
