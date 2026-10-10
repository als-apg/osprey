"""The ``ring`` word shrinks to nothing in the shipped code (lattice/deck instead),
and the engine's own words stay inside the engine's zones.

The scan runs ``git grep`` over the tracked text files under ``SCAN_PATHS``.
Every file it reports must sit in ``ALLOWLIST``, whose tag names the stage that
clears the file: ``delete:<stage>`` when that stage deletes it, ``rename:<stage>``
when the file stays and loses the word. An entry fails once its stage is at or
before ``CURRENT_BATCH`` and the file still matches, and at any stage once the
file no longer matches (a stale entry). A new file never joins the list.

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
from tests.facility._batches import BATCHES, CURRENT_BATCH

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

#: Paths that never carry the word; no allowlist entry may lie under them.
CLEAN_PATHS: tuple[str, ...] = (
    "src/osprey/facility/",
    "src/osprey/simulation/engines/",
    "packages/osprey-connectors/src/osprey_connectors/simulation/values.py",
    "scripts/facility_schema/",
)

ALLOWLIST: dict[str, str] = {
    "src/osprey/interfaces/channel_finder/database_api.py": "rename:12",
    "src/osprey/interfaces/web_terminal/app.py": "rename:12",
    "src/osprey/interfaces/web_terminal/routes/agent_activity.py": "rename:12",
    "src/osprey/interfaces/web_terminal/routes/config.py": "rename:12",
    "src/osprey/interfaces/web_terminal/routes/panels.py": "rename:12",
    "src/osprey/interfaces/web_terminal/static/js/activity-history.js": "rename:12",
    "src/osprey/interfaces/web_terminal/static/js/activity-strip.js": "rename:12",
    "src/osprey/interfaces/web_terminal/static/js/panel-agent-attention.js": "rename:12",
    "src/osprey/interfaces/web_terminal/static/js/panel-sse.js": "rename:12",
    "src/osprey/mcp_server/artifact_activity.py": "rename:12",
    "src/osprey/mcp_server/channel_finder_graph/tools/examples_data.py": "rename:12",
    "src/osprey/mcp_server/channel_finder_hierarchical/tools/build_channels.py": "rename:12",
    "src/osprey/mcp_server/channel_finder_hierarchical/tools/get_options.py": "rename:12",
    "src/osprey/mcp_server/channel_finder_middle_layer/tools/list_families.py": "rename:12",
    "src/osprey/mcp_server/phoebus/plt_generator.py": "rename:12",
    "src/osprey/mcp_server/workspace/tools/artifact_register.py": "rename:12",
    "src/osprey/mcp_server/workspace/tools/setup.py": "rename:12",
    "src/osprey/profiles/presets/control-assistant.yml": "rename:12",
    "src/osprey/services/bluesky_bridge/bump_analysis.py": "rename:12",
    "src/osprey/services/bluesky_bridge/orm_analysis.py": "rename:12",
    "src/osprey/services/bluesky_bridge/plans_core/orbit_bump_sweep.py": "rename:12",
    "src/osprey/services/bluesky_bridge/plans_core/orm.py": "rename:12",
    "src/osprey/services/bluesky_bridge/substrate_devices.py": "rename:12",
    "src/osprey/services/channel_finder/feedback/pending_store.py": "rename:12",
    "src/osprey/services/facility_knowledge/okf/document.py": "rename:12",
    "src/osprey/templates/claude_code/CLAUDE.channel-finder.md.j2": "rename:12",
    "src/osprey/templates/claude_code/claude/agents/_terminology/graph.md.j2": "rename:12",
    "src/osprey/templates/claude_code/claude/agents/_terminology/middle_layer.md.j2": "rename:12",
    "src/osprey/templates/claude_code/claude/agents/channel-finder.md.j2": "rename:12",
    "src/osprey/templates/claude_code/claude/agents/facility-knowledge-graph.md.j2": "rename:12",
    "src/osprey/templates/claude_code/claude/agents/pyat-specialist.md.j2": "rename:12",
    "src/osprey/templates/claude_code/claude/hooks/osprey_target_state.py": "rename:12",
    "src/osprey/templates/claude_code/claude/hooks/osprey_writes_check.py": "rename:12",
    "src/osprey/templates/claude_code/claude/output-styles/control-operator.md.j2": "rename:12",
}

_TAG = re.compile(r"(delete|rename):(?P<stage>[0-9a-z]+)")


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


def violations(
    hits: set[str], allowlist: Mapping[str, str], current: int = CURRENT_BATCH
) -> list[str]:
    """Return one line per reported file or entry the ratchet refuses."""
    problems = [
        f"{path}: carries the word and has no allowlist entry" for path in hits - set(allowlist)
    ]
    for path, tag in allowlist.items():
        match = _TAG.fullmatch(tag)
        stage = match["stage"] if match else None
        if stage not in BATCHES:
            problems.append(f"{path}: tag {tag!r} names no stage")
        elif path not in hits:
            problems.append(f"{path}: no longer carries the word; drop its entry")
        elif BATCHES.index(stage) <= current:
            problems.append(f"{path}: tagged {tag}, still carries the word")
    return sorted(problems)


# --- the pattern ------------------------------------------------------------------


@pytest.mark.parametrize("text", ["strings", "wirings", "string_field", "Spring", "ringing"])
def test_pattern_leaves_other_words(text: str) -> None:
    assert re.search(PATTERN, text) is None


@pytest.mark.parametrize(
    "text", ["storage ring", "PyATRingModel", "ring_buffer", "RING", "load_ring", "Ring_Model"]
)
def test_pattern_finds_the_word(text: str) -> None:
    assert re.search(PATTERN, text) is not None


# --- the stage tuple and the allowlist ---------------------------------------------


def test_current_batch_indexes_the_tuple() -> None:
    assert isinstance(CURRENT_BATCH, int)
    assert 0 <= CURRENT_BATCH < len(BATCHES)
    assert len(set(BATCHES)) == len(BATCHES)


def test_every_tag_names_a_stage() -> None:
    bad = {path: tag for path, tag in ALLOWLIST.items() if not _TAG.fullmatch(tag)}
    bad |= {
        path: tag
        for path, tag in ALLOWLIST.items()
        if (match := _TAG.fullmatch(tag)) and match["stage"] not in BATCHES
    }
    assert bad == {}


def test_allowlist_is_sorted() -> None:
    assert list(ALLOWLIST) == sorted(ALLOWLIST)


def test_no_entry_lies_under_a_clean_path() -> None:
    assert [path for path in ALLOWLIST if path.startswith(CLEAN_PATHS)] == []


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
    assert violations(hits, {}) == [
        "scripts/facility_schema/planted.py: carries the word and has no allowlist entry"
    ]


def test_ratchet_refuses_due_and_stale_entries() -> None:
    hits = {"a.py", "b.py", "c.py"}
    allowlist = {
        "a.py": f"delete:{BATCHES[CURRENT_BATCH]}",
        "b.py": f"rename:{BATCHES[-1]}",
        "c.py": "delete:0z",
        "gone.py": f"rename:{BATCHES[-1]}",
    }
    assert violations(hits, allowlist) == [
        f"a.py: tagged delete:{BATCHES[CURRENT_BATCH]}, still carries the word",
        "c.py: tag 'delete:0z' names no stage",
        "gone.py: no longer carries the word; drop its entry",
    ]


# --- the tree ---------------------------------------------------------------------


def test_the_word_only_survives_where_the_allowlist_says() -> None:
    assert violations(scan(REPO_ROOT), ALLOWLIST) == []


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
