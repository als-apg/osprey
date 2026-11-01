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
    SITE_EXAMPLE_WORD,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


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
