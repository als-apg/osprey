"""Shipped text names no deployment-root entry at a retired path.

The deployment root's agent material moved under ``agent/``, its operations
material under ``ops/`` and its runtime state under ``var/``. A deployment still
holding a retired path is refused, so prose, a skill, a template or a docstring
that tells a reader to write one sends them straight into that refusal. This
sweep reads the shipped text for each retired path in its root-anchored
spellings:

- the retired path run from the deployment root, a command line that begins
  ``./scripts/verify.sh``;
- the retired path as the possession of the deployment, the repository or the
  profile, ``the profile's ``project/`` mirror``;
- the retired path as the value of a profile key that holds a repo-relative
  path (:data:`osprey.repo_retired.PATH_KEYS`), ``template: services/site_api``;
- a retired directory whose name is not a plain word (``web-terminal-context``,
  ``data/code_review``) as a leading segment that no other path segment
  precedes.

A retired path that is a plain word (``rules``, ``scripts``, ``services``) is
read only in the anchored spellings: the same word names the convention
directories inside ``agent/`` and ``ops/``, an artifact key (``rules/safety``),
a directory of an application, of the build output or of the OSPREY repository
itself, and none of those moved. A retired file (``osprey.service``) is read only
in the anchored spellings too: its bare name is the name it keeps. The retired
paths come from :func:`osprey.repo_layout.retired_moves` alone; the sweep keeps
no list of its own.

The swept text is the prose under ``docs/source`` and ``plugins``, the
templates, the profiles and presets, the channel-finder static pages and every
docstring under ``src`` and ``packages/*/src``. Code strings are the spelling
guard's (``test_repo_layout_spelling_guard.py``), and a tree listing (a line
that is only a path) is checked by hand, not here.

Exempt as whole files: the changelogs by name, the layout table that declares
the retired paths and the refusal that names them. A line that must name a
retired path on purpose is listed in :data:`LINE_ALLOWLIST` with its reason; an
entry that no longer matches its line fails, so the list cannot rot.
"""

from __future__ import annotations

import ast
import functools
import re
from collections.abc import Collection, Iterable, Iterator, Mapping
from pathlib import Path

from osprey import repo_layout
from osprey.repo_retired import PATH_KEYS
from tests._text_sweep import text_files

REPO_ROOT = Path(__file__).resolve().parents[1]

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

#: Whole files that declare the retired paths or refuse them.
EXEMPT_FILES = frozenset(
    {
        "src/osprey/repo_layout.py",
        "src/osprey/repo_retired.py",
    }
)

#: Whole files exempt by name wherever they sit.
EXEMPT_NAMES = frozenset({"CHANGELOG.md"})

#: ``(path, text the line contains) -> reason`` for lines that name a retired
#: path on purpose.
LINE_ALLOWLIST: Mapping[tuple[str, str], str] = {
    (
        "docs/source/how-to/build-profiles.rst",
        "* - ``web-terminal-context/``",
    ): "the migration table names each retired path",
    (
        "docs/source/how-to/build-profiles.rst",
        "* - ``data/code_review/``",
    ): "the migration table names each retired path",
    (
        "docs/source/how-to/build-profiles.rst",
        "* - ``data/.runtime/``",
    ): "the migration table names each retired path",
    (
        "docs/source/how-to/build-profiles.rst",
        "perl -pi -e 's{    template: services/site_api}",
    ): "the sample refusal block rewrites a retired template value",
    (
        "docs/source/how-to/build-profiles.rst",
        "* - ``data/facility/``",
    ): "the migration table names each retired path",
    (
        "docs/source/how-to/build-profiles.rst",
        "* - ``data/canopy/library/``, ``data/canopy/collections/``,",
    ): "the migration table names each retired path",
    (
        "docs/source/how-to/build-profiles.rst",
        "       ``data/canopy/theme/``",
    ): "the migration table names each retired path",
    (
        "docs/source/how-to/build-profiles.rst",
        "* - ``data/landing/``",
    ): "the migration table names each retired path",
    (
        "docs/source/how-to/build-profiles.rst",
        "git mv data/facility facility",
    ): "the sample refusal block moves a retired path",
    (
        "docs/source/how-to/build-profiles.rst",
        "git mv data/canopy/library web/canopy/library",
    ): "the sample refusal block moves a retired path",
    (
        "docs/source/how-to/build-profiles.rst",
        "git mv data/landing web/landing",
    ): "the sample refusal block moves a retired path",
    (
        "docs/source/how-to/build-profiles.rst",
        "facility_knowledge\\.bundle_path: data/facility/knowledge}",
    ): "the sample refusal block rewrites a profile value naming a retired path",
    (
        "docs/source/how-to/build-profiles.rst",
        "        - data/landing/working-safely\\.md}",
    ): "the sample refusal block rewrites a profile value naming a retired path",
    (
        "docs/source/reference/cli.rst",
        "``data/`` is dissolved: ``data/facility/`` moves to",
    ): "the migration command names each path it moves",
    (
        "docs/source/reference/cli.rst",
        "``facility/``, ``data/landing/`` and ``data/canopy/`` to ``web/``",
    ): "the migration command names each path it moves",
    (
        "docs/source/architecture/facility-file.rst",
        "       ``data/facility/ariel/vocabulary.yml``",
    ): "names the render's copy, which the render still places under data/",
    (
        "docs/source/how-to/ariel/search-modes.rst",
        "``data/facility/ariel/vocabulary.yml``. The example",
    ): "names the render's copy, which the render still places under data/",
    (
        "docs/source/how-to/ariel/search-modes.rst",
        "path: data/facility/ariel/vocabulary.yml     # stated by the build",
    ): "names the render's copy, which the render still places under data/",
    (
        "src/osprey/profiles/config_key_manifest.yml",
        "The build states data/facility/ariel/vocabulary.yml",
    ): "names the render's copy, which the render still places under data/",
    (
        "src/osprey/templates/claude_code/claude/skills/pyaml-tools/SKILL.md.j2",
        "exceeds half its `max_step` in `data/facility/limits.yaml`",
    ): "the agent reads the render's facility tree",
    (
        "src/osprey/cli/build_profile_load.py",
        "keep saying ``data/facility``, because the",
    ): "names the render's facility tree",
    (
        "src/osprey/facility/views/__init__.py",
        "facility_dir: The build's machine root, its ``data/facility``",
    ): "names the render's facility tree",
    (
        "src/osprey/facility/views/outputs.py",
        "facility_dir: The build's machine root, its ``data/facility``",
    ): "names the render's facility tree",
    (
        "src/osprey/cli/templates/scaffolding.py",
        "The runtime reads one facility tree, ``data/facility``",
    ): "names the render's facility tree",
    (
        "src/osprey/cli/templates/scaffolding.py",
        "``data/facility/identity.yaml`` and ``limits.yaml`` all expect it",
    ): "names the render's facility tree",
    (
        "src/osprey/cli/templates/scaffolding.py",
        "dst_facility: The render's ``data/facility`` directory.",
    ): "names the render's facility tree",
    (
        "src/osprey/cli/templates/manager.py",
        "project's ``data/facility`` beside the deployment's own content",
    ): "names the render's facility tree",
    (
        "docs/source/how-to/build-profiles.rst",
        "- web-terminal-context/base.md   #",
    ): "an exclude entry names an artifact by its convention key, not a path",
}

#: Who a retired path is spelled as belonging to when it is the deployment's.
_OWNERS = ("deployment", "repository", "repo", "profile")

#: A plain word: a retired path spelled so is read only when anchored.
_PLAIN_WORD = re.compile(r"[a-z]+")

#: Quoting that may sit between an anchor and the path it anchors.
_QUOTES = r"[\"'`*]*"


def retired_paths() -> tuple[tuple[str, bool], ...]:
    """Each retired path with whether it was a directory, from the layout table."""
    return tuple(
        (move.old, move.kind is not repo_layout.EntryKind.FILE)
        for move in repo_layout.retired_moves()
    )


def kept_names() -> frozenset[str]:
    """The retired paths whose successor ends in the same name (``rules`` -> ``agent/rules``)."""
    return frozenset(
        move.old for move in repo_layout.retired_moves() if move.new.endswith(f"/{move.old}")
    )


def _path_key_names() -> tuple[str, ...]:
    """The last key of each profile key that holds a repo-relative path."""
    return tuple(sorted({key.rsplit(".", 1)[-1] for key in PATH_KEYS}))


def _alternatives(retired: Iterable[tuple[str, bool]]) -> str:
    """The retired paths as one alternation, longest first, each ending at a path boundary.

    A directory is a path only before a ``/``: ``the profile's project`` is
    prose and ``the profile's mcp_servers:`` a key, ``the profile's project/`` a
    directory.
    """
    alternatives = []
    for old, is_dir in sorted(retired, key=lambda r: -len(r[0])):
        end = "(?=/)" if is_dir else r"(?![\w.-])"
        alternatives.append(rf"{re.escape(old)}{end}")
    return "|".join(alternatives)


def _patterns(
    retired: Iterable[tuple[str, bool]], kept: Collection[str] = frozenset()
) -> tuple[tuple[re.Pattern[str], ...], re.Pattern[str] | None]:
    """The anchored spellings of the retired paths, and their leading-segment spelling.

    A path in *kept* kept its name under its successor, so as a leading segment
    it may be that successor's convention directory spoken of from inside its
    group; it is read only when anchored.
    """
    retired = tuple(retired)
    if not retired:
        return (), None
    olds = _alternatives(retired)
    owners = "|".join(_OWNERS)
    keys = "|".join(re.escape(key) for key in _path_key_names())
    anchored = (
        re.compile(rf"^[ \t]*(?:\$[ \t]+)?\./(?:{olds})"),
        re.compile(
            rf"\b(?i:the|a|this|that|your|each|every)\s+(?:{owners})'s\s+{_QUOTES}(?:{olds})"
        ),
        re.compile(rf"\b(?:{keys}):\s*{_QUOTES}(?:{olds})"),
    )
    leading = sorted((old for old, _ in retired if old not in kept), key=lambda old: -len(old))
    if not leading:
        return anchored, None
    names = "|".join(rf"{re.escape(old)}(?=/|[\s`\"',;:)\]]|$)" for old in leading)
    return anchored, re.compile(rf"(?<![\w./<>{{}}$-])(?:{names})")


#: The indent and tree-drawing characters a tree listing entry starts after.
_LISTING_INDENT = re.compile(r"[ \t│├└─|]+")


def _line_offenses(
    line: str,
    patterns: tuple[tuple[re.Pattern[str], ...], re.Pattern[str] | None],
) -> list[str]:
    """The retired-path spellings on one line, less a tree listing's entry.

    A spelling inside a longer one (the ``web-terminal-context`` of
    ``the repo's web-terminal-context``) is reported once, as the longer. A
    listing entry is a leading segment that is the first thing on an indented
    line (``    web-terminal-context/   the context``); listings are checked
    by hand.
    """
    anchored, leading = patterns
    indent = _LISTING_INDENT.match(line)
    entry_at = indent.end() if indent else -1
    found = {m.span() for pattern in anchored for m in pattern.finditer(line)}
    if leading is not None:
        found |= {m.span() for m in leading.finditer(line) if m.start() != entry_at}
    spans = sorted(found)
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
        return rel in exempt_files or path.name in EXEMPT_NAMES

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
    kept: Collection[str] = frozenset(),
    exempt_files: Collection[str] = EXEMPT_FILES,
    line_allowlist: Mapping[tuple[str, str], str] = LINE_ALLOWLIST,
) -> tuple[list[str], set[tuple[str, str]]]:
    """Every retired-path spelling under the roots, and the allowlist entries used.

    Returns:
        ``path:line: spelling`` offenders, and the allowlist keys that matched
        a line holding a spelling.
    """
    patterns = (
        _patterns(retired_paths(), kept_names()) if retired is None else _patterns(retired, kept)
    )
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
    assert retired["rules"] is True
    assert retired["web-terminal-context"] is True
    assert retired["osprey.service"] is False
    assert retired["data/.runtime"] is True
    for old in (
        "data/facility",
        "data/landing",
        "data/canopy/library",
        "data/canopy/collections",
        "data/canopy/theme",
        "data/bluesky_curve",
        "data/bluesky_va_curve",
        "data/bluesky_live_curve",
        "data/bluesky_standin_curve",
    ):
        assert retired[old] is True, old
    assert "template" in _path_key_names()
    kept = kept_names()
    assert "rules" in kept and "osprey.service" in kept
    renamed = {
        "web-terminal-context",
        "data/code_review",
        "data/.runtime",
        "data/facility",
        "data/landing",
        "data/bluesky_curve",
    }
    assert not renamed & kept


def test_every_retired_move_is_covered() -> None:
    """Each retired path is caught in an anchored spelling, plain word, file or not."""
    patterns = _patterns(retired_paths(), kept_names())
    uncovered = [
        old
        for old, is_dir in retired_paths()
        if _line_offenses(f"Edit the deployment's ``{old}{'/' if is_dir else ''}``.", patterns)
        != [f"the deployment's ``{old}"]
    ]
    assert uncovered == []


def test_no_retired_root_spellings() -> None:
    offenders, _ = _repo_sweep()
    assert offenders == (), (
        "a deployment-root entry lives at its current path (agent/, ops/, var/); "
        "rewrite these spellings (or, for a line that must name a retired path, "
        "add it to LINE_ALLOWLIST with its reason):\n" + "\n".join(offenders)
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
        ("docs/source", "plugins", "src/osprey/templates", "src/osprey/profiles"),
        ("src",),
        exempt_files=frozenset(),
        line_allowlist={},
    )
    return sorted(offenders)


def test_planted_retired_spellings_fail(tmp_path: Path) -> None:
    """Each root-anchored spelling fails in prose, presets, templates and docstrings."""
    _plant(
        tmp_path,
        "docs/source/page.rst",
        "After a deploy, run:\n"
        "\n"
        "   ./scripts/verify.sh containers\n"
        "Per-user notes sit in ``web-terminal-context/<user>/extra.md``.\n"
        "The reviewer reads ``data/code_review/reviewer.md`` and the runtime\n"
        "certificates sit in ``data/.runtime/``.\n"
        "Edit the deployment's ``rules/`` and the profile's ``project/`` mirror.\n"
        "The profile's ``osprey.service`` and the repository's ``ci-extra.yml``.\n"
        "   $ ./osprey.service\n"
        "Describe the machine under ``data/facility/devices/``.\n",
    )
    _plant(
        tmp_path,
        "src/osprey/profiles/presets/site.yml",
        "services:\n  site_api:\n    template: services/site_api\n",
    )
    _plant(tmp_path, "plugins/skill/SKILL.md", "Write styles into your repo's `output-styles/`.\n")
    _plant(
        tmp_path,
        "src/osprey/templates/app/README.md.j2",
        "This deployment's `mcp_servers/` holds {{ name }}.\n",
    )
    _plant(
        tmp_path,
        "src/pkg/mod.py",
        '"""Reads ``data/.runtime/certs``."""\n'
        "\n"
        "PATH = './skills/beam-check'\n"
        "\n"
        "\n"
        "def load():\n"
        '    """Load the repo\'s ``watchdogs/`` directory."""\n'
        "\n"
        "\n"
        "def facility():\n"
        '    """Read the sources under data/facility/records."""\n',
    )

    assert _planted_sweep(tmp_path) == sorted(
        [
            "docs/source/page.rst:3:    ./scripts",
            "docs/source/page.rst:4: web-terminal-context",
            "docs/source/page.rst:5: data/code_review",
            "docs/source/page.rst:6: data/.runtime",
            "docs/source/page.rst:7: the deployment's ``rules",
            "docs/source/page.rst:7: the profile's ``project",
            "docs/source/page.rst:8: The profile's ``osprey.service",
            "docs/source/page.rst:8: the repository's ``ci-extra.yml",
            "docs/source/page.rst:9:    $ ./osprey.service",
            "docs/source/page.rst:10: data/facility",
            "src/osprey/profiles/presets/site.yml:3: template: services",
            "plugins/skill/SKILL.md:1: your repo's `output-styles",
            "src/osprey/templates/app/README.md.j2:1: This deployment's `mcp_servers",
            "src/pkg/mod.py:1: data/.runtime",
            "src/pkg/mod.py:7: the repo's ``watchdogs",
            "src/pkg/mod.py:11: data/facility",
        ]
    )


def test_current_spellings_pass(tmp_path: Path) -> None:
    """Current paths, plain-word keys, other roots, listings and code strings stay legal."""
    _plant(
        tmp_path,
        "docs/source/page.rst",
        "Run ``./ops/scripts/verify.sh``; per-user notes sit in ``agent/context/``.\n"
        "Edit the deployment's ``agent/rules/`` and the profile's ``ops/project/``.\n"
        "List ``rules/safety`` under ``rules:`` and claim ``services/postgresql``.\n"
        "Run ``scripts/check.py`` in the OSPREY repository; an app has ``skills/``.\n"
        "The build writes ``build/.claude/output-styles/`` and ``.claude/rules/``.\n"
        "The template tree is ``claude_code/web-terminal-context/``.\n"
        "The unit is ``osprey.service`` and the CI file ``ops/ci-extra.yml``.\n"
        "The runtime state sits in ``var/runtime/``; the render reads ``build/data/facility/``.\n"
        "A style is named ``output-styles/terse``; the profile's ``mcp_servers:`` key.\n"
        "The build injector writes ``path: ./services/qmd`` and the profile's project.\n"
        "Ship ``rules/facility.md`` in the convention directory ``rules/``.\n"
        "\n"
        "   agent/\n"
        "     mcp_servers/  the MCP server packages the preset ships\n",
    )
    _plant(
        tmp_path,
        "src/osprey/profiles/presets/site.yml",
        "services:\n  site_api:\n    template: ops/services/site_api\n",
    )
    _plant(tmp_path, "src/pkg/mod.py", "PATH = 'web-terminal-context/base.md'\n")

    assert _planted_sweep(tmp_path) == []


def test_allowlisted_line_passes_and_is_used(tmp_path: Path) -> None:
    """An allow-listed line is not an offender, and the entry counts as live."""
    _plant(tmp_path, "docs/source/page.rst", "The repo's ``rules/`` moved to ``agent/rules/``\n")
    key = ("docs/source/page.rst", "``rules/`` moved")
    offenders, used = retired_spelling_offenders(
        tmp_path,
        ("docs/source",),
        (),
        exempt_files=frozenset(),
        line_allowlist={key: "names the move"},
    )
    assert offenders == []
    assert used == {key}
