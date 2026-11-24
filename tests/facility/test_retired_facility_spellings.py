"""Shipped text names no facility source at a retired path.

A facility tree's files moved into groups (``vocabulary/``, ``simulation/``,
``provenance/``) and the benchmark queries into ``benchmarks/channel_finder``.
A deployment still holding a retired path is refused, so prose, a skill, a
template or a docstring that tells a reader to write one sends them straight
into that refusal. This sweep reads the shipped text for each retired path in
its path-shaped spellings, anchored at the facility root:

- ``data/facility/<old>``, the root as messages name it;
- a ``deck:`` value beginning with the retired decks directory;
- a retired directory as a leading segment, ``scenarios/<name>.yaml``, that no
  other path segment precedes (``simulation/scenarios/`` and
  ``imported/<layer>/decks`` are the current places).

A bare file name, ``classes.yaml`` in a sentence, is not path-shaped and is not
read. The retired paths come from :func:`osprey_facility.source_tree.retired_moves`
alone; the sweep keeps no list of its own.

The swept text is the prose under ``docs/source`` and ``plugins``, the
non-facility-tree templates, the profiles, the channel-finder static pages and
every docstring under ``src`` and ``packages/*/src``. Code strings are the
spelling guard's (``test_facility_spelling_guard.py``), and a tree listing
(a line that is only a path) is checked by hand, not here.

Exempt as whole files: the changelogs by name, the frozen floor schema, the
source-tree table that declares the retired paths, and the simulator view
modules (the view the build writes keeps ``decks/`` and ``scenarios/`` at its
own root). A line that must name a retired path on purpose is listed in
:data:`LINE_ALLOWLIST` with its reason; an entry that no longer matches its
line fails, so the list cannot rot.
"""

from __future__ import annotations

import ast
import functools
import re
from collections.abc import Collection, Iterable, Iterator, Mapping
from pathlib import Path

from osprey_facility import source_tree
from tests._text_sweep import text_files

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The roots whose prose the sweep reads.
PROSE_ROOTS = (
    "docs/source",
    "plugins",
    "src/osprey/templates",
    "src/osprey/profiles",
    "src/osprey/interfaces/channel_finder/static",
)

#: The suffixes of shipped prose, pages and templates.
PROSE_SUFFIXES = (".rst", ".md", ".txt", ".html", ".js", ".j2", ".yml", ".yaml")

#: The roots whose Python docstrings the sweep reads.
DOCSTRING_ROOTS = (
    "src",
    *sorted(f"packages/{p.name}/src" for p in (REPO_ROOT / "packages").iterdir() if p.is_dir()),
)

#: Whole files that declare the retired paths, describe the simulator view, or
#: are fixed at the version they describe.
EXEMPT_FILES = frozenset(
    {
        "packages/osprey-facility/src/osprey_facility/source_tree.py",
        "packages/osprey-facility/src/osprey_facility/schema/_generated/facility-4.schema.json",
        "src/osprey/facility/views/simulator.py",
        "packages/osprey-connectors/src/osprey_connectors/simulation/view.py",
    }
)

#: Whole files exempt by name wherever they sit.
EXEMPT_NAMES = frozenset({"CHANGELOG.md"})

#: ``(path, text the line contains) -> reason`` for lines that name a retired
#: path on purpose.
LINE_ALLOWLIST: Mapping[tuple[str, str], str] = {
    (
        "docs/source/reference/cli.rst",
        "``scenarios/``  ",
    ): "the migration table names each retired path",
    (
        "docs/source/reference/cli.rst",
        "``measurement/``  ",
    ): "the migration table names each retired path",
    (
        "docs/source/reference/cli.rst",
        "``decks/``  ",
    ): "the migration table names each retired path",
    (
        "docs/source/reference/cli.rst",
        "``benchmarks/cross_paradigm/queries/``  ",
    ): "the migration table names each retired path",
    (
        "docs/source/reference/cli.rst",
        "that begins ``decks/`` is",
    ): "the migration names the deck value it rewrites",
    (
        "docs/source/how-to/describe-your-facility.rst",
        "``scenarios/`` in place of",
    ): "the refusal note names a retired path it refuses",
    (
        "docs/source/how-to/describe-your-facility.rst",
        "value that names ``decks/``",
    ): "the refusal note names the deck value it rewrites",
    (
        "docs/source/_diagrams/va-layer-map.html",
        "the view osprey build writes:",
    ): "the simulator view's deck copies, not the facility source",
    (
        "docs/source/_diagrams/va-layer-map.html",
        "decks/&lt;model&gt;.json",
    ): "the simulator view's deck copy, not the facility source",
    (
        "docs/source/architecture/virtual-accelerator.rst",
        "``decks/<model>.<its suffix>``",
    ): "the simulator view's deck copy, not the facility source",
    (
        "docs/source/contributing/extending-osprey.rst",
        "``decks/<model>.<its suffix>``",
    ): "the simulator view's deck copy an engine plug-in is given",
    (
        "docs/source/how-to/control-systems/use-virtual-accelerator.rst",
        "``decks/<model>.json``",
    ): "the simulator view's deck copy, not the facility source",
    (
        "src/osprey/simulation/apply.py",
        "view's copies under ``scenarios/<name>/``",
    ): "the simulator view's scenario copies, not the facility source",
    (
        "src/osprey/simulation/model_request.py",
        "simulator view's ``decks/<model>.json``",
    ): "the simulator view's deck copy, not the facility source",
    (
        "src/osprey/mcp_server/python_executor/tools/model_response_matrix.py",
        "simulator view's ``decks/<model>.json``",
    ): "the simulator view's deck copy, not the facility source",
}


def retired_paths() -> tuple[tuple[str, bool], ...]:
    """Each retired path with whether it was a directory, from the source-tree table."""
    return tuple(
        (old, "." not in old.rsplit("/", 1)[-1])
        for old, _new, _content in source_tree.retired_moves()
    )


def _patterns(retired: Iterable[tuple[str, bool]]) -> tuple[re.Pattern[str], ...]:
    """The path-shaped spellings of the retired paths."""
    retired = tuple(retired)
    olds = "|".join(re.escape(old) for old, _ in sorted(retired, key=lambda r: -len(r[0])))
    dirs = "|".join(
        re.escape(old) for old, is_dir in sorted(retired, key=lambda r: -len(r[0])) if is_dir
    )
    rooted = re.compile(rf"{re.escape(source_tree.display_path(''))}(?:{olds})(?![\w.-])")
    deck_old, _deck_new = source_tree.retired_deck_prefix()
    deck = re.compile(rf"\bdeck:\s*[\"'`]*{re.escape(deck_old)}")
    patterns = [rooted, deck]
    if dirs:
        patterns.append(re.compile(rf"(?<![\w./-])(?:{dirs})/"))
    return tuple(patterns)


#: The indent and tree-drawing characters a tree listing entry starts after.
_LISTING_INDENT = re.compile(r"[ \t│├└─|]+")


def _line_offenses(line: str, patterns: Iterable[re.Pattern[str]]) -> list[str]:
    """The retired-path spellings on one line, less a tree listing's entry.

    A spelling inside a longer one (the ``decks/`` of ``deck: decks/``) is
    reported once, as the longer. A listing entry is the first thing on an indented line (``    decks/   the
    decks``); listings are checked by hand.
    """
    indent = _LISTING_INDENT.match(line)
    entry_at = indent.end() if indent else -1
    spans = sorted(
        {m.span() for pattern in patterns for m in pattern.finditer(line) if m.start() != entry_at}
    )
    return [
        line[start:end]
        for start, end in spans
        if not any(s <= start and end <= e and (s, e) != (start, end) for s, e in spans)
    ]


def _docstring_spans(source: str) -> Iterator[tuple[int, int]]:
    """The ``(first, last)`` line numbers of each docstring in *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            first = body[0]
            yield first.lineno, first.end_lineno or first.lineno


def _swept_lines(
    repo_root: Path,
    prose_roots: Iterable[str],
    docstring_roots: Iterable[str],
    exempt_files: Collection[str],
) -> Iterator[tuple[str, int, str]]:
    """Every ``(rel, line number, line)`` the sweep reads."""

    def skip(path: Path) -> bool:
        rel = path.relative_to(repo_root).as_posix()
        return rel in exempt_files or path.name in EXEMPT_NAMES or "/data/facility/" in f"/{rel}"

    for path in text_files(repo_root, prose_roots, suffixes=PROSE_SUFFIXES, skip=skip):
        rel = path.relative_to(repo_root).as_posix()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(lines, start=1):
            yield rel, number, line

    for path in text_files(repo_root, docstring_roots, suffixes=(".py",), skip=skip):
        rel = path.relative_to(repo_root).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        lines = source.splitlines()
        for first, last in _docstring_spans(source):
            for number in range(first, last + 1):
                yield rel, number, lines[number - 1]


def retired_spelling_offenders(
    repo_root: Path,
    prose_roots: Iterable[str] = PROSE_ROOTS,
    docstring_roots: Iterable[str] = DOCSTRING_ROOTS,
    *,
    retired: Iterable[tuple[str, bool]] | None = None,
    exempt_files: Collection[str] = EXEMPT_FILES,
    line_allowlist: Mapping[tuple[str, str], str] = LINE_ALLOWLIST,
) -> tuple[list[str], set[tuple[str, str]]]:
    """Every retired-path spelling under the roots, and the allowlist entries used.

    Returns:
        ``path:line: spelling`` offenders, and the allowlist keys that matched
        a line holding a spelling.
    """
    patterns = _patterns(retired_paths() if retired is None else retired)
    offenders: list[str] = []
    used: set[tuple[str, str]] = set()
    for rel, number, line in _swept_lines(repo_root, prose_roots, docstring_roots, exempt_files):
        offenses = _line_offenses(line, patterns)
        if not offenses:
            continue
        allowed = [key for key in line_allowlist if key[0] == rel and key[1] in line]
        if allowed:
            used.update(allowed)
            continue
        offenders.extend(f"{rel}:{number}: {offense}" for offense in offenses)
    return offenders, used


@functools.cache
def _repo_sweep() -> tuple[tuple[str, ...], frozenset[tuple[str, str]]]:
    """:func:`retired_spelling_offenders` over the real repository, read once."""
    offenders, used = retired_spelling_offenders(REPO_ROOT)
    return tuple(offenders), frozenset(used)


def test_retired_paths_are_known() -> None:
    """The sweep knows what it looks for; an empty list would pass anything."""
    retired = dict(retired_paths())
    assert retired["classes.yaml"] is False
    assert retired["decks"] is True
    assert retired["benchmarks/cross_paradigm/queries"] is True
    assert source_tree.retired_deck_prefix()[0] == "decks/"


def test_no_retired_facility_spellings() -> None:
    offenders, _ = _repo_sweep()
    assert offenders == (), (
        "a facility source lives at its current path (vocabulary/, simulation/, "
        "provenance/, benchmarks/channel_finder); rewrite these spellings (or, "
        "for a line that must name a retired path, add it to LINE_ALLOWLIST with "
        "its reason):\n" + "\n".join(offenders)
    )


def test_line_allowlist_entries_are_live() -> None:
    """Every allowlisted line still holds a retired-path spelling."""
    _, used = _repo_sweep()
    assert sorted(set(LINE_ALLOWLIST) - used) == []


def _plant(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _planted_sweep(root: Path) -> list[str]:
    offenders, _ = retired_spelling_offenders(
        root,
        ("docs/source", "plugins", "src/osprey/templates"),
        ("src",),
        exempt_files=frozenset(),
        line_allowlist={},
    )
    return sorted(offenders)


def test_planted_retired_spellings_fail(tmp_path: Path) -> None:
    """Each path-shaped spelling fails in prose, pages, templates and docstrings."""
    _plant(
        tmp_path,
        "docs/source/page.rst",
        "Edit ``data/facility/classes.yaml`` to add a class.\n"
        "Scenarios sit in ``data/facility/scenarios/`` and models in\n"
        "``data/facility/models.yaml``.\n"
        "Write ``scenarios/<name>.yaml`` beside ``measurement/<model>.yaml``.\n"
        "Queries live in ``benchmarks/cross_paradigm/queries/``.\n",
    )
    _plant(tmp_path, "plugins/skill/SKILL.md", "Set `deck: decks/SR.json` on the model.\n")
    _plant(tmp_path, "docs/source/_diagrams/map.html", "<text>data/facility/decks/</text>\n")
    _plant(tmp_path, "docs/source/_static/app.js", "const DECKS = 'decks/' + name;\n")
    _plant(tmp_path, "src/osprey/templates/app/README.md.j2", 'deck: "decks/SR.json"\n')
    _plant(
        tmp_path,
        "src/pkg/mod.py",
        '"""Reads ``data/facility/seeds.yaml``."""\n'
        "\n"
        "PATH = 'data/facility/fixes.yaml'\n"
        "\n"
        "\n"
        "def load():\n"
        '    """Load every file under ``scenarios/``."""\n',
    )

    assert _planted_sweep(tmp_path) == sorted(
        [
            "docs/source/page.rst:1: data/facility/classes.yaml",
            "docs/source/page.rst:2: data/facility/scenarios",
            "docs/source/page.rst:3: data/facility/models.yaml",
            "docs/source/page.rst:4: scenarios/",
            "docs/source/page.rst:4: measurement/",
            "docs/source/page.rst:5: benchmarks/cross_paradigm/queries/",
            "plugins/skill/SKILL.md:1: deck: decks/",
            "docs/source/_diagrams/map.html:1: data/facility/decks",
            "docs/source/_static/app.js:1: decks/",
            'src/osprey/templates/app/README.md.j2:1: deck: "decks/',
            "src/pkg/mod.py:1: data/facility/seeds.yaml",
            "src/pkg/mod.py:7: scenarios/",
        ]
    )


def test_current_spellings_pass(tmp_path: Path) -> None:
    """The current paths, a bare file name, a listing and a code string stay legal."""
    _plant(
        tmp_path,
        "docs/source/page.rst",
        "Edit ``data/facility/vocabulary/classes.yaml``; ``classes.yaml`` holds classes.\n"
        "Scenarios sit in ``data/facility/simulation/scenarios/<name>.yaml``.\n"
        "A layer's decks sit in ``imported/mml/decks/``, the source's in\n"
        "``simulation/decks/`` and the queries in ``benchmarks/channel_finder/``.\n"
        "Set ``deck: simulation/decks/SR.json``; ``data/facility/decksets`` is other.\n"
        "The ``apps/demo/scenarios/`` folder is not a facility root.\n"
        "\n"
        "   simulation/\n"
        "     decks/                the decks those models name\n"
        "     scenarios/<name>.yaml ways the machine departs from its baseline\n",
    )
    _plant(tmp_path, "src/osprey/templates/app/data/facility/README.md", "``decks/``\n")
    _plant(tmp_path, "src/pkg/mod.py", "PATH = 'data/facility/fixes.yaml'\n")

    assert _planted_sweep(tmp_path) == []


def test_allowlisted_line_passes_and_is_used(tmp_path: Path) -> None:
    """An allow-listed line is not an offender, and the entry counts as live."""
    _plant(tmp_path, "docs/source/page.rst", "``decks/`` moved to ``simulation/decks/``\n")
    key = ("docs/source/page.rst", "``decks/`` moved")
    offenders, used = retired_spelling_offenders(
        tmp_path,
        ("docs/source",),
        (),
        exempt_files=frozenset(),
        line_allowlist={key: "names the move"},
    )
    assert offenders == []
    assert used == {key}
