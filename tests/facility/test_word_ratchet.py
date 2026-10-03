"""The ``ring`` word shrinks to nothing in the shipped code (lattice/deck instead).

The scan runs ``git grep`` over the tracked text files under ``SCAN_PATHS``.
Every file it reports must sit in ``ALLOWLIST``, whose tag names the stage that
clears the file: ``delete:<stage>`` when that stage deletes it, ``rename:<stage>``
when the file stays and loses the word. An entry fails once its stage is at or
before ``CURRENT_BATCH`` and the file still matches, and at any stage once the
file no longer matches (a stale entry). A new file never joins the list.

A line quoting an outside format word for word carries ``QUOTE_MARKER`` and is
not reported.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from tests._vocabulary import RATCHET_WORD
from tests.facility._batches import BATCHES, CURRENT_BATCH

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The word the scan matches, shared with the agent-facing guard.
PATTERN = RATCHET_WORD

SCAN_PATHS: tuple[str, ...] = (
    "src",
    "packages",
    "scripts/facility_schema",
    "scripts/facility_demo",
)

#: The MATLAB Middle Layer exporter the mml layer ships: MATLAB source written
#: in the Middle Layer's own words (``THERING``, "the ring"), an outside format.
OUTSIDE_FORMAT_FILES: tuple[str, ...] = ("src/osprey/facility/layers/mml/mml_export.m",)

EXCLUDED: tuple[str, ...] = (
    *OUTSIDE_FORMAT_FILES,
    "**/templates/apps/*/data/**",
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
    "src/osprey/cli/build_cmd.py": "rename:12",
    "src/osprey/cli/mml_cmd.py": "delete:7e",
    "src/osprey/deployment/container_lifecycle.py": "rename:12",
    "src/osprey/interfaces/ariel/static/css/components.css": "rename:12",
    "src/osprey/interfaces/channel_finder/database_api.py": "rename:12",
    "src/osprey/interfaces/design_system/static/css/highlight.css": "rename:12",
    "src/osprey/interfaces/design_system/static/css/theme-lab.css": "rename:12",
    "src/osprey/interfaces/design_system/static/js/theme-lab-ui.js": "rename:12",
    "src/osprey/interfaces/design_system/static/theme-lab.html": "rename:12",
    "src/osprey/interfaces/health/static/dashboard.css": "rename:12",
    "src/osprey/interfaces/health/static/index.html": "rename:12",
    "src/osprey/interfaces/health/static/js/dashboard.js": "rename:12",
    "src/osprey/interfaces/lattice_dashboard/compute.py": "rename:12",
    "src/osprey/interfaces/lattice_dashboard/state.py": "rename:12",
    "src/osprey/interfaces/lattice_dashboard/static/dashboard.css": "rename:12",
    "src/osprey/interfaces/lattice_dashboard/static/js/render.js": "rename:12",
    "src/osprey/interfaces/lattice_dashboard/workers/_base.py": "rename:12",
    "src/osprey/interfaces/lattice_dashboard/workers/chromaticity.py": "rename:12",
    "src/osprey/interfaces/lattice_dashboard/workers/da.py": "rename:12",
    "src/osprey/interfaces/lattice_dashboard/workers/footprint.py": "rename:12",
    "src/osprey/interfaces/lattice_dashboard/workers/lma.py": "rename:12",
    "src/osprey/interfaces/lattice_dashboard/workers/optics.py": "rename:12",
    "src/osprey/interfaces/lattice_dashboard/workers/resonance.py": "rename:12",
    "src/osprey/interfaces/web_terminal/app.py": "rename:12",
    "src/osprey/interfaces/web_terminal/routes/agent_activity.py": "rename:12",
    "src/osprey/interfaces/web_terminal/routes/config.py": "rename:12",
    "src/osprey/interfaces/web_terminal/routes/panels.py": "rename:12",
    "src/osprey/interfaces/web_terminal/static/css/activity-strip.css": "rename:12",
    "src/osprey/interfaces/web_terminal/static/css/bars.css": "rename:12",
    "src/osprey/interfaces/web_terminal/static/css/terminal.css": "rename:12",
    "src/osprey/interfaces/web_terminal/static/css/tour.css": "rename:12",
    "src/osprey/interfaces/web_terminal/static/js/activity-history.js": "rename:12",
    "src/osprey/interfaces/web_terminal/static/js/activity-strip.js": "rename:12",
    "src/osprey/interfaces/web_terminal/static/js/panel-agent-attention.js": "rename:12",
    "src/osprey/interfaces/web_terminal/static/js/panel-sse.js": "rename:12",
    "src/osprey/mcp_server/artifact_activity.py": "rename:12",
    "src/osprey/mcp_server/channel_finder_graph/tools/capabilities.py": "rename:12",
    "src/osprey/mcp_server/channel_finder_graph/tools/examples_data.py": "rename:12",
    "src/osprey/mcp_server/channel_finder_hierarchical/tools/build_channels.py": "rename:12",
    "src/osprey/mcp_server/channel_finder_hierarchical/tools/get_options.py": "rename:12",
    "src/osprey/mcp_server/channel_finder_middle_layer/tools/list_families.py": "rename:12",
    "src/osprey/mcp_server/graph/tools/examples_data.py": "rename:12",
    "src/osprey/mcp_server/phoebus/plt_generator.py": "rename:12",
    "src/osprey/mcp_server/workspace/tools/artifact_register.py": "rename:12",
    "src/osprey/mcp_server/workspace/tools/setup.py": "rename:12",
    "src/osprey/profiles/presets/control-assistant.yml": "rename:12",
    "src/osprey/services/bluesky_bridge/bump_analysis.py": "rename:12",
    "src/osprey/services/bluesky_bridge/orm_analysis.py": "rename:12",
    "src/osprey/services/bluesky_bridge/plans_core/orbit_bump_sweep.py": "rename:12",
    "src/osprey/services/bluesky_bridge/plans_core/orm.py": "rename:12",
    "src/osprey/services/bluesky_bridge/substrate_devices.py": "rename:12",
    "src/osprey/services/channel_finder/benchmarks/generator.py": "rename:12",
    "src/osprey/services/channel_finder/feedback/pending_store.py": "rename:12",
    "src/osprey/services/channel_finder/naming.py": "rename:12",
    "src/osprey/services/channel_finder/tools/generate_from_spec.py": "delete:7d2",
    "src/osprey/services/facility_knowledge/okf/document.py": "rename:12",
    "src/osprey/services/facility_knowledge/ttl_generator/emitter.py": "delete:7e",
    "src/osprey/services/facility_knowledge/ttl_generator/mml_source.py": "delete:7e",
    "src/osprey/services/facility_knowledge/ttl_generator/model.py": "delete:7e",
    "src/osprey/services/mml/census.py": "delete:7e",
    "src/osprey/services/mml/emit/va.py": "delete:7e",
    "src/osprey/services/mml/judgments.py": "delete:7e",
    "src/osprey/services/mml/loaders/__init__.py": "delete:7e",
    "src/osprey/services/mml/loaders/mat.py": "delete:7e",
    "src/osprey/services/mml/mapping/skeleton.py": "delete:7e",
    "src/osprey/services/mml/va/elements.py": "delete:7e",
    "src/osprey/services/mml/va/fingerprint.py": "delete:7e",
    "src/osprey/services/mml/va/verdicts.py": "delete:7e",
    "src/osprey/services/mml/va/verify.py": "delete:7e",
    "src/osprey/services/virtual_accelerator/__init__.py": "rename:12",
    "src/osprey/services/virtual_accelerator/bindings.py": "delete:7e",
    "src/osprey/services/virtual_accelerator/entrypoint.py": "rename:12",
    "src/osprey/services/virtual_accelerator/ioc/__init__.py": "delete:7d",
    "src/osprey/services/virtual_accelerator/ioc/physics_bridge.py": "delete:7d",
    "src/osprey/services/virtual_accelerator/lattice/__init__.py": "delete:7e",
    "src/osprey/services/virtual_accelerator/lattice/calibration.py": "delete:7e",
    "src/osprey/services/virtual_accelerator/lattice/response.py": "delete:7e",
    "src/osprey/services/virtual_accelerator/lattice/ring.py": "delete:7e",
    "src/osprey/services/virtual_accelerator/lattice/solve.py": "delete:7e",
    "src/osprey/services/virtual_accelerator/manifest/__init__.py": "delete:7e",
    "src/osprey/services/virtual_accelerator/manifest/build.py": "delete:7e",
    "src/osprey/services/virtual_accelerator/manifest/channel_manifest.json": "delete:7e",
    "src/osprey/services/virtual_accelerator/manifest/classify.py": "delete:7e",
    "src/osprey/services/virtual_accelerator/manifest/loaders.py": "delete:7e",
    "src/osprey/services/virtual_accelerator/manifest/paths.py": "delete:7e",
    "src/osprey/services/virtual_accelerator/model/__init__.py": "delete:7e",
    "src/osprey/services/virtual_accelerator/model/bindings.py": "delete:7e",
    "src/osprey/services/virtual_accelerator/model/catalog.py": "delete:7e",
    "src/osprey/services/virtual_accelerator/model/pyat.py": "delete:7e",
    "src/osprey/services/virtual_accelerator/model/variables.py": "delete:7e",
    "src/osprey/services/virtual_accelerator/serving/pvdb.py": "delete:7d",
    "src/osprey/services/virtual_accelerator/serving/write_path.py": "delete:7d",
    "src/osprey/simulation/channel_schema.py": "delete:7d2",
    "src/osprey/simulation/facility_spec.py": "delete:7d2",
    "src/osprey/simulation/lattice/__init__.py": "delete:7d2",
    "src/osprey/simulation/lattice/artifact.py": "delete:7d2",
    "src/osprey/simulation/lattice/build.py": "delete:7d2",
    "src/osprey/simulation/lattice/ring.py": "delete:7d2",
    "src/osprey/templates/claude_code/CLAUDE.channel-finder.md.j2": "rename:12",
    "src/osprey/templates/claude_code/claude/agents/_terminology/graph.md.j2": "rename:12",
    "src/osprey/templates/claude_code/claude/agents/_terminology/in_context.md.j2": "rename:12",
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
    assert {"scripts/facility_schema", "scripts/facility_demo"} <= set(SCAN_PATHS)


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
