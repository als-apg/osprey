"""The engine's own words stay inside the engine's zones, and retired words stay gone.

The engine words (``ENGINE_WORDS``) are a permanent token: a file
outside ``ENGINE_ZONES`` that names a pyAT attribute or the ``axis`` key is
classifying a binding by the engine's words rather than by the ``role`` and
``plane`` the simulator view states.

The roster's retired word (``RETIRED_ROSTER_WORD``) is a second token, scanned
over every tracked file by content and by path. Only the code that reads the old
spelling, the frozen previous-release repo and the released notes keep it
(``RETIRED_ROSTER_READERS``); ``RETIRED_ROSTER_INTERIM`` names the trees a later
task still rewords, each row deleted by that task. A data line that must spell
the old key carries ``RETIRED_SPELLING_MARKER``. An allowlist row that stops
matching anything fails, so the list only shrinks.

The example deployment name that named a site (``SITE_EXAMPLE_WORD``) is a
third token, scanned over the core text under ``SITE_SCAN_PATHS``.

The phrase that named two login kinds (``RETIRED_TWO_KIND_PHRASE``) is a fourth
token, scanned over the folded text of ``TWO_KIND_SCAN_PATHS``, so a wrapped
phrase is found as an unwrapped one is. A file that must keep it is a
``RETIRED_TWO_KIND_READERS`` row with its reason; an allowlist row that stops
matching anything fails, so the list only shrinks.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from tests._vocabulary import (
    ENGINE_WORDS,
    RETIRED_ROSTER_WORD,
    RETIRED_TWO_KIND_PHRASE,
    SITE_EXAMPLE_WORD,
    Phrase,
    fold_lines,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# --- the fold a phrase row is matched over -----------------------------------------


@pytest.mark.parametrize(
    ("text", "folded"),
    [
        ("the tier\nor app", "the tier or app"),
        ("# the tier\n# or app", "# the tier or app"),
        ("    the tier\n    or app", "    the tier or app"),
        ("// the tier\n  // or app", "// the tier or app"),
        (" * the tier\n * or app", " * the tier or app"),
        ("; the tier\n;or app", "; the tier or app"),
        ("-- the tier\n-- or app", "-- the tier or app"),
        (".. the tier\n.. or app", ".. the tier or app"),
        ("> the tier\n> or app", "> the tier or app"),
        ("| the tier\n| or app", "| the tier or app"),
        ("\tthe tier\r\n\tor app", "\tthe tier or app"),
    ],
)
def test_fold_lines_joins_a_wrapped_line(text: str, folded: str) -> None:
    assert fold_lines(text).text == folded


@pytest.mark.parametrize(
    "text",
    [
        "the tier\n\nAnd app",
        "# the tier\n#\n# And app",
        "the tier\n   \nAnd app",
    ],
)
def test_fold_lines_keeps_a_break(text: str) -> None:
    lines = [line for _, line in fold_lines(text).lines()]

    assert len(lines) > 1
    assert not any(re.search(r"tier\s+(?:And\s+)?app", line) for line in lines)


@pytest.mark.parametrize("bullet", ["-", "+"])
def test_fold_lines_keeps_a_list_bullet(bullet: str) -> None:
    folded = fold_lines(f"{bullet} a tier\n{bullet} and an app")

    assert folded.text == f"{bullet} a tier {bullet} and an app"
    assert re.search(RETIRED_TWO_KIND_PHRASE, folded.text) is None


def test_fold_lines_folds_one_prefix_only() -> None:
    assert fold_lines("# the tier\n# # or app").text == "# the tier # or app"


def test_fold_lines_maps_an_offset_to_its_original_line() -> None:
    folded = fold_lines("one\n\n  # two\n  # three\nfour\n")

    for word, line in [("one", 1), ("two", 3), ("three", 4), ("four", 5)]:
        assert folded.line_at(folded.text.index(word)) == line
    assert [line for _, line in folded.lines()] == ["one", "", "  # two three four", ""]
    assert all(folded.text[start:].startswith(line) for start, line in folded.lines())


def test_the_two_kind_row_is_a_phrase_and_the_word_rows_are_not() -> None:
    assert isinstance(RETIRED_TWO_KIND_PHRASE, Phrase)
    for row in (ENGINE_WORDS, RETIRED_ROSTER_WORD, SITE_EXAMPLE_WORD):
        assert not isinstance(row, Phrase)


def _planted_repo(root: Path, files: Mapping[str, str]) -> Path:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root


# --- the engine words -------------------------------------------------------------

#: The trees the engine-word scan reads, ``*.py`` only: a vendored minified
#: script matches the token without classifying anything.
ENGINE_SCAN_PATHS: tuple[str, ...] = (
    "src",
    "packages",
    "scripts",
    "tests/e2e",
    "tests/va/e2e",
    "tests/connectors",
    "tests/interfaces",
)

#: Where the engine words belong, each with why.
ENGINE_ZONES: dict[str, str] = {
    "src/osprey/simulation/engines/**": "the engine plug-ins translate wiring into engine words",
    "src/osprey/facility/layers/**": "the importers author a model's wiring in its engine's words",
    "src/osprey/templates/**/data/**": "a shipped facility definition states its engine blocks",
    "src/osprey/templates/facilities/**": "a shipped facility definition states its engine blocks",
    "src/osprey/apps/sdk/testing/**": "the SDK's synthetic facility trees state their engine blocks",
    "src/osprey/interfaces/lattice_dashboard/state.py": "walks the pyAT elements of a deck",
    "src/osprey/interfaces/lattice_dashboard/workers/**": "walk the pyAT elements of a deck",
    "src/osprey/mcp_server/phoebus/tools/databrowser_tools.py": "Phoebus's own plot axis key",
    "scripts/facility_demo/**": "the demo generator authors the example facility's decks in the engine's format",
}


def engine_word_hits(root: Path, zones: Mapping[str, str] = ENGINE_ZONES) -> list[str]:
    """Every ``path:line`` under ``root`` outside ``zones`` naming an engine word."""
    includes = [f":(glob){path}/**/*.py" for path in ENGINE_SCAN_PATHS]
    excludes = [f":(exclude,glob){glob}" for glob in zones]
    result = subprocess.run(
        ["git", "grep", "-I", "-n", "-P", "-e", ENGINE_WORDS, "--", *includes, *excludes],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    return sorted(":".join(line.split(":", 2)[:2]) for line in result.stdout.splitlines() if line)


@pytest.mark.parametrize("text", ["KickAngle", "PolynomA", "PolynomB", "'axis'", '"axis"'])
def test_engine_words_pattern_finds_the_words(text: str) -> None:
    assert re.search(ENGINE_WORDS, text) is not None


@pytest.mark.parametrize("text", ["axis", "x_axis", "PolynomC", "KickAngles", "'axes'"])
def test_engine_words_pattern_leaves_other_words(text: str) -> None:
    assert re.search(ENGINE_WORDS, text) is None


def test_a_planted_engine_word_under_the_connectors_is_flagged(tmp_path: Path) -> None:
    line = "ATTRIBUTE = 'KickAngle'\n"
    root = _planted_repo(
        tmp_path,
        {
            "packages/osprey-connectors/src/osprey_connectors/simulation/planted.py": line,
            "src/osprey/simulation/engines/planted.py": line,
        },
    )

    assert engine_word_hits(root) == [
        "packages/osprey-connectors/src/osprey_connectors/simulation/planted.py:1"
    ]


def test_engine_words_survive_only_in_their_zones() -> None:
    assert engine_word_hits(REPO_ROOT) == []


# --- the roster's retired word ----------------------------------------------------

#: Where the retired word stays for good, each with why.
RETIRED_ROSTER_READERS: dict[str, str] = {
    "src/osprey/cli/build_profile_document.py": (
        "the retired-key rewrite, the one parse point that reads the old spelling"
    ),
    "tests/cli/test_retired_roster_keys.py": "the retired-key rewrite's own test",
    "tests/fixtures/previous_release_repo/**": "the frozen repo of the previous release",
    "tests/cli/test_init_revalidate.py": "the previous release's profile text, held as the swap table",
    "CHANGELOG.md": "released notes describe what shipped then",
}

#: Trees a later task still rewords; that task deletes its row.
RETIRED_ROSTER_INTERIM: dict[str, str] = {}

#: A line carrying this is not reported: a data line that must spell the old key.
RETIRED_SPELLING_MARKER = "retired-spelling"

#: The word, spelled without spelling it, for the tests below.
_WORD = "per" + "sona"


def retired_roster_hits(
    root: Path,
    allowed: Mapping[str, str] = {**RETIRED_ROSTER_READERS, **RETIRED_ROSTER_INTERIM},
) -> list[str]:
    """Every tracked ``path:line`` and ``path (name)`` naming the retired word.

    Content lines carrying ``RETIRED_SPELLING_MARKER`` and paths under an
    ``allowed`` glob are not reported.
    """
    excludes = [f":(exclude,glob){glob}" for glob in allowed]
    result = subprocess.run(
        ["git", "grep", "-I", "-n", "-P", "-e", RETIRED_ROSTER_WORD, "--", ".", *excludes],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    hits = [
        ":".join(line.split(":", 2)[:2])
        for line in result.stdout.splitlines()
        if line and RETIRED_SPELLING_MARKER not in line.split(":", 2)[2]
    ]
    paths = subprocess.run(
        ["git", "ls-files", "--", ".", *excludes],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    hits += [f"{path} (name)" for path in paths if re.search(RETIRED_ROSTER_WORD, path)]
    return sorted(hits)


def _hit_path(hit: str) -> str:
    return hit.removesuffix(" (name)") if hit.endswith(" (name)") else hit.rsplit(":", 1)[0]


@pytest.mark.parametrize(
    "text",
    [
        _WORD,
        f"{_WORD}s",
        f"default_{_WORD}",
        f"test_{_WORD}_permissions",
        f"effective{_WORD.capitalize()}",
        f"{_WORD.capitalize()}Card",
        f"per-{_WORD}",
        _WORD.upper(),
    ],
)
def test_retired_roster_word_finds_the_word(text: str) -> None:
    assert re.search(RETIRED_ROSTER_WORD, text) is not None


@pytest.mark.parametrize(
    "text", ["personal", "Personal", "PERSONAL", "personality", "impersonate", "personally"]
)
def test_retired_roster_word_leaves_other_words(text: str) -> None:
    assert re.search(RETIRED_ROSTER_WORD, text) is None


def test_a_planted_retired_word_is_flagged(tmp_path: Path) -> None:
    root = _planted_repo(
        tmp_path,
        {
            "packages/osprey-connectors/src/planted.py": f"KIND = '{_WORD}'\n",
            ".github/workflows/ci.yml": f"# one {_WORD} image\n",
            "tests/interfaces/planted.test.mjs": f"// a per-{_WORD} key\n",
            f"{_WORD}s/x.yml": "kind: tier\n",
            "src/marked.py": f"KEY = '{_WORD}s'  # {RETIRED_SPELLING_MARKER}\n",
            "CHANGELOG.md": f"- the {_WORD} catalog\n",
            "src/clean.py": "PERSONAL = 'personal personality'\n",
        },
    )

    assert retired_roster_hits(root) == [
        ".github/workflows/ci.yml:1",
        "packages/osprey-connectors/src/planted.py:1",
        f"{_WORD}s/x.yml (name)",
        "tests/interfaces/planted.test.mjs:1",
    ]


def test_no_tracked_file_names_the_retired_word() -> None:
    assert retired_roster_hits(REPO_ROOT) == []


def test_every_retired_word_allowlist_entry_still_matches() -> None:
    hit_paths = {_hit_path(hit) for hit in retired_roster_hits(REPO_ROOT, allowed={})}
    stale = []
    for glob in {**RETIRED_ROSTER_READERS, **RETIRED_ROSTER_INTERIM}:
        covered = subprocess.run(
            ["git", "ls-files", "--", f":(glob){glob}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        if not hit_paths.intersection(covered):
            stale.append(glob)
    assert stale == [], f"allowlist rows that cover no hit any more, delete them: {stale}"


# --- the site example name --------------------------------------------------------

#: The core text the site example name never enters.
SITE_SCAN_PATHS: tuple[str, ...] = ("src", "packages", "docs", "plugins", "README.md")


def site_example_hits(root: Path) -> list[str]:
    """Every ``path:line`` under ``SITE_SCAN_PATHS`` naming the site example name."""
    result = subprocess.run(
        [
            "git",
            "grep",
            "-I",
            "-n",
            "-P",
            "-e",
            SITE_EXAMPLE_WORD,
            "--",
            *SITE_SCAN_PATHS,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    return sorted(":".join(line.split(":", 2)[:2]) for line in result.stdout.splitlines() if line)


@pytest.mark.parametrize("text", ["als-assistant", "~/deployments/als-assistant/tiers"])
def test_site_example_word_finds_the_name(text: str) -> None:
    assert re.search(SITE_EXAMPLE_WORD, text) is not None


@pytest.mark.parametrize("text", ["my-assistant", "als-assistants-guide", "pals-assistant"])
def test_site_example_word_leaves_other_words(text: str) -> None:
    assert re.search(SITE_EXAMPLE_WORD, text) is None


def test_a_planted_site_example_is_flagged(tmp_path: Path) -> None:
    line = "osprey init als-assistant\n"
    root = _planted_repo(
        tmp_path,
        {
            "src/x.py": line,
            "docs/x.rst": line,
            "plugins/x/SKILL.md": line,
            "tests/x.py": line,
            "CHANGELOG.md": line,
        },
    )

    assert site_example_hits(root) == ["docs/x.rst:1", "plugins/x/SKILL.md:1", "src/x.py:1"]


def test_no_core_text_names_the_site_example() -> None:
    assert site_example_hits(REPO_ROOT) == []


# --- the retired two-kind phrase --------------------------------------------------

#: The core text the phrase never enters.
TWO_KIND_SCAN_PATHS: tuple[str, ...] = ("src", "docs/source", "plugins")

#: Where the phrase stays for good, each with why.
RETIRED_TWO_KIND_READERS: dict[str, str] = {}


def phrase_hits(root: Path, phrase: Phrase, pathspecs: list[str]) -> list[str]:
    """Every tracked text file's ``path:line`` where *phrase* starts in its folded text.

    ``git grep`` reads one line at a time, so the files are listed by git and
    each is folded and matched here. A file that is not text is skipped, as
    ``git grep -I`` skips it; hits starting on one line are one entry.
    """
    paths = subprocess.run(
        ["git", "ls-files", "-z", "--", *pathspecs],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    hits: set[str] = set()
    for path in filter(None, paths):
        file = root / path
        if not file.is_file():
            continue
        data = file.read_bytes()
        if b"\0" in data[:8000]:
            continue
        folded = fold_lines(data.decode("utf-8", errors="replace"))
        for offset, line in folded.lines():
            for match in re.finditer(phrase, line):
                hits.add(f"{path}:{folded.line_at(offset + match.start())}")
    return sorted(hits)


def two_kind_phrase_hits(
    root: Path, allowed: Mapping[str, str] = RETIRED_TWO_KIND_READERS
) -> list[str]:
    """Every ``path:line`` under ``TWO_KIND_SCAN_PATHS`` naming the two-kind phrase."""
    excludes = [f":(exclude,glob){glob}" for glob in allowed]
    return phrase_hits(root, RETIRED_TWO_KIND_PHRASE, [*TWO_KIND_SCAN_PATHS, *excludes])


@pytest.mark.parametrize(
    "text",
    [
        "a tier or app file",
        "every Tier and App render",
        "the tiers and apps its roster names",
        "tiers or apps share a project",
        "a tier and an app are",
        "one per tier/app",
        "Tier, app and image renders",
        "TIER OR APP",
    ],
)
def test_two_kind_phrase_finds_the_phrase(text: str) -> None:
    assert re.search(RETIRED_TWO_KIND_PHRASE, text) is not None


@pytest.mark.parametrize(
    "text",
    [
        "the login's tier",
        "an app's panel",
        "tiers, apps, and secrets",
        "tier and application",
        "tiers/ and apps/ folders",
        "tiers, apps = _roster_layers(config)",
        "the tier and write switch",
        "framework artifacts, tiers, apps and secrets",
        "data, tiers, apps and watchdogs",
    ],
)
def test_two_kind_phrase_leaves_other_words(text: str) -> None:
    assert re.search(RETIRED_TWO_KIND_PHRASE, text) is None


def test_a_planted_two_kind_phrase_is_flagged(tmp_path: Path) -> None:
    line = "names the tier or app file\n"
    root = _planted_repo(
        tmp_path,
        {
            "src/x.py": line,
            "docs/source/x.rst": line,
            "plugins/x/SKILL.md": line,
            "docs/other/x.md": line,
            "tests/x.py": line,
            "CHANGELOG.md": line,
            "src/clean.py": "# the login's tier; an app's panel\n",
            # Wrapped across lines, each reported at the line the phrase starts on.
            "src/comment.py": "x = 1\n# names the tier\n# or app file\n",
            "src/docstring.py": 'def f():\n    """Names the tier\n    or app file."""\n',
            "docs/source/paragraph.rst": "Intro.\n\nEach tier\nor app file is built.\n",
            "docs/source/directive.rst": ".. note::\n\n   Each tier\n   and app render.\n",
            "plugins/x/quote.md": "> names the tier\n> or app file\n",
            "src/block.js": "/**\n * names the tier\n * or app file\n */\n",
            "src/comment.yaml": "# names the tier\n# or app file\nkey: 1\n",
            "src/twice.py": "# tier or app; tier\n# or app\n",
            # Broken by a blank line or a list bullet: stays clean.
            "docs/source/break.rst": "The tier\n\nAnd app files are built.\n",
            "plugins/x/list.md": "Kinds:\n\n- a tier\n- and an app\n",
        },
    )

    assert two_kind_phrase_hits(root) == [
        "docs/source/directive.rst:3",
        "docs/source/paragraph.rst:3",
        "docs/source/x.rst:1",
        "plugins/x/SKILL.md:1",
        "plugins/x/quote.md:1",
        "src/block.js:2",
        "src/comment.py:2",
        "src/comment.yaml:1",
        "src/docstring.py:2",
        "src/twice.py:1",
        "src/x.py:1",
    ]


def test_no_core_text_names_the_two_kind_phrase() -> None:
    assert two_kind_phrase_hits(REPO_ROOT) == []


def test_every_two_kind_allowlist_entry_still_matches() -> None:
    hit_paths = {hit.rsplit(":", 1)[0] for hit in two_kind_phrase_hits(REPO_ROOT, allowed={})}
    stale = []
    for glob in RETIRED_TWO_KIND_READERS:
        covered = subprocess.run(
            ["git", "ls-files", "--", f":(glob){glob}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        if not hit_paths.intersection(covered):
            stale.append(glob)
    assert stale == [], f"allowlist rows that cover no hit any more, delete them: {stale}"
