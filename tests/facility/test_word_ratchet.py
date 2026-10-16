"""The ``ring`` word is gone from the shipped code (lattice/deck instead), and the
engine's own words stay inside the engine's zones.

The scan runs ``git grep`` over the tracked text files under ``SCAN_PATHS``, and
every file it reports is a failure: no file carries the word.

A line quoting an outside format word for word carries ``QUOTE_MARKER`` and is
not reported.

The engine words (``ENGINE_WORDS``) are a second, permanent token: a file
outside ``ENGINE_ZONES`` that names a pyAT attribute or the ``axis`` key is
classifying a binding by the engine's words rather than by the ``role`` and
``plane`` the simulator view states.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from tests._vocabulary import ENGINE_WORDS, RATCHET_WORD

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The word the scan matches, shared with the agent-facing guard.
PATTERN = RATCHET_WORD

SCAN_PATHS: tuple[str, ...] = (
    "src",
    "packages",
    "scripts/facility_schema",
)

#: The MATLAB Middle Layer exporter the mml layer ships: MATLAB source written
#: in the Middle Layer's own words (``THERING``, "the ring"), an outside format.
OUTSIDE_FORMAT_FILES: tuple[str, ...] = ("src/osprey/facility/layers/mml/mml_export.m",)

EXCLUDED: tuple[str, ...] = (
    *OUTSIDE_FORMAT_FILES,
    "**/templates/apps/*/data/**",
    "**/templates/facilities/**",
    "**/data/facility/**",
    "**/static/**/vendor/**",
    "**/*.min.js",
    "**/*.svg",
)

QUOTE_MARKER = "outside-format-quote"


def scan(root: Path) -> set[str]:
    """Return the tracked files under ``root`` with an unmarked matching line."""
    pathspec = [*SCAN_PATHS, *(f":(exclude,glob){glob}" for glob in EXCLUDED)]
    result = subprocess.run(
        ["git", "grep", "-I", "-z", "-P", "-e", PATTERN, "--", *pathspec],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    hits: set[str] = set()
    for line in result.stdout.split("\n"):
        if not line:
            continue
        path, _, content = line.partition("\0")
        if QUOTE_MARKER not in content:
            hits.add(path)
    return hits


def violations(hits: set[str]) -> list[str]:
    """Return one line per reported file, sorted."""
    return [f"{path}: carries the word and has no allowlist entry" for path in sorted(hits)]


# --- the pattern ------------------------------------------------------------------


@pytest.mark.parametrize("text", ["strings", "wirings", "string_field", "Spring", "ringing"])
def test_pattern_leaves_other_words(text: str) -> None:
    assert re.search(PATTERN, text) is None


@pytest.mark.parametrize(
    "text", ["storage ring", "PyATRingModel", "ring_buffer", "RING", "load_ring", "Ring_Model"]
)
def test_pattern_finds_the_word(text: str) -> None:
    assert re.search(PATTERN, text) is not None


# --- the scan ---------------------------------------------------------------------


def test_the_exporter_is_the_only_excluded_file_under_the_facility_package() -> None:
    def listed(*excludes: str) -> set[str]:
        """The tracked non-empty files under the package the scan's pathspec keeps."""
        pathspec = ["src/osprey/facility", *(f":(exclude,glob){glob}" for glob in excludes)]
        result = subprocess.run(
            ["git", "grep", "-l", "-z", "-a", "-e", "", "--", *pathspec],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return set(filter(None, result.stdout.split("\0")))

    assert listed() - listed(*EXCLUDED) == set(OUTSIDE_FORMAT_FILES)


def test_scan_covers_the_facility_scripts() -> None:
    assert {"scripts/facility_schema"} <= set(SCAN_PATHS)


def test_binary_fonts_are_never_reported() -> None:
    fonts = subprocess.run(
        ["git", "ls-files", "--", "src/**/*.ttf"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert fonts, "no tracked .ttf under src/ to prove the binary skip on"
    assert [path for path in scan(REPO_ROOT) if path.endswith(".ttf")] == []


def _planted_repo(root: Path, files: Mapping[str, str]) -> Path:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root


def test_planted_word_under_facility_schema_is_flagged(tmp_path: Path) -> None:
    root = _planted_repo(
        tmp_path,
        {
            "scripts/facility_schema/planted.py": "LATTICE = 'storage ring'\n",
            "scripts/facility_schema/quoted.py": f"KEY = 'ring'  # {QUOTE_MARKER}\n",
            "src/pkg/templates/apps/demo/data/notes.md": "the ring\n",
            "src/pkg/data/facility/identity.yaml": "name: ring\n",
            "src/pkg/static/js/vendor/lib.js": "ring\n",
            "src/pkg/static/app.min.js": "ring\n",
            "src/pkg/static/icon.svg": "<svg>ring</svg>\n",
            "src/pkg/clean.py": "strings = wirings = 'string_field'\n",
            "docs/page.md": "storage ring\n",
        },
    )
    hits = scan(root)
    assert hits == {"scripts/facility_schema/planted.py"}
    assert violations(hits) == [
        "scripts/facility_schema/planted.py: carries the word and has no allowlist entry"
    ]


# --- the tree ---------------------------------------------------------------------


def test_no_shipped_file_carries_the_word() -> None:
    assert violations(scan(REPO_ROOT)) == []


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
