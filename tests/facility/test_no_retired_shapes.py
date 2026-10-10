"""Retired symbols, paths and environment names never come back.

``RETIRED`` maps each retired token to the stage that deleted its LAST producer
or reader; a token joins the map in that stage's close. Once its stage is at or
before ``CURRENT_BATCH``, the token must not appear in any tracked text file
under ``SCAN_ROOTS``. A token whose stage is still ahead is not enforced yet.

The map holds symbols, dotted module paths, file paths and environment names
only. Retired CLI verb spellings live in ``_RETIRED_SPELLINGS``
(``tests/cli/test_lifecycle_invariants.py``) and retired configuration keys in
the ``deleted:`` section of ``src/osprey/profiles/config_key_manifest.yml``.

History keeps the names on purpose: ``changelog.d/`` and ``CHANGELOG.md`` are
never swept. ``SELF_EXEMPT`` names the files that spell retired tokens as data
(this guard and its sibling guards, and the key manifest); every entry must be a
tracked file, so a stale exemption fails the guard.

A token matches as a whole name: a letter, digit or underscore on either side
means a different, longer name.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest

from tests.facility._batches import BATCHES, CURRENT_BATCH

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Tracked trees the sweep reads: tests (``tests/e2e`` included), scripts, docs,
#: the rendered templates, the profile presets and catalogs, and the deployment
#: code and images.
SCAN_ROOTS: tuple[str, ...] = (
    "docker",
    "docs",
    "scripts",
    "src/osprey/deployment",
    "src/osprey/profiles",
    "src/osprey/templates",
    "tests",
)

#: History records that the retired names existed and when they went.
EXCLUDED: tuple[str, ...] = ("changelog.d/", "CHANGELOG.md")

#: Files that spell retired tokens as data.
SELF_EXEMPT: tuple[str, ...] = (
    "src/osprey/profiles/config_key_manifest.yml",
    "tests/config/test_facility_keys_retired.py",
    "tests/config/test_simulation_keys_retired.py",
    "tests/connectors/test_limits_mode.py",
    "tests/deployment/test_va_compose_instances.py",
    "tests/docs/test_environment_variable_page.py",
    "tests/docs/test_mml_converter_retired.py",
    "tests/facility/golden/cf_index_pre_line/MANIFEST.json",
    "tests/facility/golden/demo_fingerprint.json",
    "tests/facility/golden/in_context_size.json",
    "tests/facility/golden/nominal_va.json",
    "tests/facility/test_deleted_surfaces.py",
    "tests/facility/test_no_retired_shapes.py",
    "tests/facility/test_schema_loosenings.py",
    "tests/simulation/test_apply_imports.py",
    "tests/utils/fixtures/legacy_config_all_deleted_keys.yml",
)

#: Retired token -> the stage that deleted its last producer or reader.
RETIRED: dict[str, str] = {
    "/api/state/init": "8",
    "ACTIVE_SCENARIO_FILENAME": "7a0",
    "BUILD_TTL_COMMAND": "5",
    "CohostDriver": "7d",
    "CohostRunner": "7d",
    "DIRECTION_SOURCE_HEADER": "7e",
    "DIRECTION_UNDERIVABLE": "2",
    "DatabaseWriteError": "4b",
    "DirectionSource": "7e",
    "FACILITY_PREFIX_CONFIG_KEY": "5",
    "GRAPHDB_BUILD_INDEX_COMMAND": "5",
    "GRAPHDB_SEED_COMMAND": "5",
    "GRAPH_MALFORMED": "2",
    "GRAPH_NO_TTL": "2",
    "GRAPH_SOURCE_PARADIGM": "2",
    "ManifestPaths": "7e",
    "P_CONFIDENCE": "7e",
    "P_PROTOCOL": "7e",
    "SignalSentence": "8",
    "TemplateChannelDatabase": "7e",
    "TestKindAwareNoiseFloor": "7d",
    "TierSpec": "4a",
    "VA_CHANNELS_FILE": "7e",
    "VA_ENTRYPOINT_MODULE": "7d",
    "VA_NOISE_LEVEL": "7d2",
    "_DerivedDevices": "4b",
    "_GRAPHDB_RECOVERY_HINT": "5",
    "_check_empty_facility_prefix": "3b",
    "_control_system_simulation_file": "7a",
    "_extend_pvdb": "7d",
    "_hierarchical_write": "4b",
    "_manifest_tier": "7e",
    "_resolve_ttl": "5",
    "_with_derived_simulation_file": "7a",
    "allow_unlisted_channels": "3a",
    "als_u_ar": "7d2",
    "assign_readbacks": "2",
    "build_tiers": "4a",
    "channel_databases/tiers": "7e",
    "derive_bands": "7a0",
    "engine_from_connector_config": "7a",
    "facility_ontology.json": "3a",
    "facility_vocabulary": "3a",
    "fieldDescription": "7e",
    "google_sheets": "4a",
    "lattice_init": "8",
    "lattice_json": "7e",
    "load_canonical_ring": "7d2",
    "machine_state_channels.json": "7e",
    "middle_layer_duckdb": "4a",
    "narad_p:confidence": "7e",
    "narad_p:protocol": "7e",
    "ordinalInFacility": "7e",
    "ordinalInSection": "7e",
    "osprey.channel_roster.database": "2",
    "osprey.channel_roster.graph": "5",
    "osprey.connectors.channel_taxonomy": "7d2",
    "osprey.services.channel_finder.databases.template": "7e",
    "osprey.services.channel_finder.naming": "7d2",
    "osprey.services.channel_finder.tools.generate_from_spec": "7d2",
    "osprey.services.virtual_accelerator.manifest.standin_defaults": "7d2",
    "osprey.simulation.archiver_seed": "7d2",
    "osprey.simulation.channel_schema": "7d2",
    "osprey.simulation.engine": "7d2",
    "osprey.simulation.expressions": "7d2",
    "osprey.simulation.facility_spec": "7d2",
    "osprey.simulation.lattice": "7d2",
    "osprey.simulation.machine": "7d2",
    "osprey.simulation.procedural": "7d2",
    "osprey.simulation.series": "7d2",
    "osprey_connectors.channel_taxonomy": "7d2",
    "osprey_connectors.simulation.archiver_seed": "7a",
    "osprey_connectors.simulation.engine": "7d2",
    "osprey_connectors.simulation.expressions": "7d2",
    "osprey_connectors.simulation.machine": "7d2",
    "osprey_connectors.simulation.procedural": "7d2",
    "pairing.py": "2",
    "parse_direction_source": "5",
    "prune_csv_build_artifacts": "4a",
    "read_database_roster": "2",
    "read_direction_source": "5",
    "read_graph_roster": "5",
    "resolve_facility_name": "3a",
    "resolved_tier": "7e",
    "ringDescription": "7e",
    "scripts/facility_demo/_limits.py": "7d2",
    "scripts/facility_demo/_measurement.py": "5",
    "scripts/facility_demo/_models.py": "5",
    "scripts/facility_demo/_records.py": "5",
    "scripts/facility_demo/_scenarios.py": "7d2",
    "scripts/facility_demo/_seeds.py": "5",
    "scripts/facility_demo/capture_nominals.py": "7d2",
    "scripts/facility_demo/fingerprint.py": "7d2",
    "scripts/va/derive_bands.py": "7a0",
    "scripts/va/pyat_model_demo.py": "7d2",
    "scripts/va/run_va.sh": "7d2",
    "sourceSectionId": "7e",
    "src/osprey/templates/apps/channel_finder_standalone/data/channel_databases/hierarchical.json": (
        "7d2"
    ),
    "src/osprey/templates/apps/control_assistant/data/lattice/als_u_ar.mat": "7d2",
    "src/osprey/templates/apps/control_assistant/data/simulation/lattice.json": "7d2",
    "src/osprey/templates/apps/control_assistant/data/simulation/machine.json": "7d2",
    "src/osprey/templates/apps/control_assistant/data/simulation/va_bindings.json": "7d2",
    "subfieldDescription": "7e",
    "tests/connectors/test_mock_archiver_simfile_fallback.py": "7a",
    "tests/templates/test_boot_band_invariant.py": "7a0",
    "tests/templates/test_channel_limits_va.py": "7a0",
    "tests/va/test_derive_bands_floor.py": "7a0",
    "tier_dir": "7e",
    "tier_mode_conflict": "4a",
    "va_graph_deferred": "2",
}

_NAME_CHAR = "A-Za-z0-9_"


def token_pattern(tokens: Iterable[str]) -> str | None:
    """Return one pattern matching any of ``tokens`` as a whole name, or None."""
    ordered = sorted(set(tokens), key=lambda token: (-len(token), token))
    if not ordered:
        return None
    alternation = "|".join(re.escape(token) for token in ordered)
    return rf"(?<![{_NAME_CHAR}])(?:{alternation})(?![{_NAME_CHAR}])"


def enforced(retired: Mapping[str, str], current: int = CURRENT_BATCH) -> set[str]:
    """Return the tokens whose stage is at or before ``current``."""
    return {
        token
        for token, stage in retired.items()
        if stage in BATCHES and BATCHES.index(stage) <= current
    }


def scan(root: Path, tokens: Iterable[str], exempt: Iterable[str] = SELF_EXEMPT) -> list[str]:
    """Return ``path:line:token`` for every enforced token in a swept file."""
    pattern = token_pattern(tokens)
    if pattern is None:
        return []
    pathspec = [
        *SCAN_ROOTS,
        *(f":(exclude){path}" for path in EXCLUDED),
        *(f":(exclude){path}" for path in exempt),
    ]
    result = subprocess.run(
        ["git", "grep", "-I", "-n", "-z", "-P", "-e", pattern, "--", *pathspec],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    compiled = re.compile(pattern)
    offenders: list[str] = []
    for line in result.stdout.split("\n"):
        if not line:
            continue
        path, lineno, content = line.split("\0", 2)
        offenders.extend(f"{path}:{lineno}:{hit}" for hit in compiled.findall(content))
    return sorted(offenders)


def stage_problems(retired: Mapping[str, str]) -> list[str]:
    """Return one line per entry whose tag names no stage."""
    return sorted(
        f"{token}: tag {stage!r} names no stage"
        for token, stage in retired.items()
        if stage not in BATCHES
    )


def stale_exemptions(root: Path, exempt: Iterable[str] = SELF_EXEMPT) -> list[str]:
    """Return the exempt paths that are not tracked files under ``root``."""
    tracked = set(
        subprocess.run(
            ["git", "ls-files", "--", *exempt],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    )
    return sorted(path for path in exempt if path not in tracked)


# --- the pattern ------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "from .pairing import assign_readbacks",
        "osprey.channel_roster.database",
        "src/osprey/channel_roster/pairing.py",
        "os.environ['OLD_ENV_NAME']",
        "data/channel_databases/tiers/tier3/hierarchical.json",
    ],
)
def test_pattern_finds_a_whole_name(text: str) -> None:
    pattern = token_pattern(
        [
            "assign_readbacks",
            "osprey.channel_roster.database",
            "pairing.py",
            "OLD_ENV_NAME",
            "channel_databases/tiers",
        ]
    )
    assert pattern is not None
    assert re.search(pattern, text) is not None


@pytest.mark.parametrize(
    "text",
    ["assign_readbacks_v2", "_assign_readbacks", "OLD_ENV_NAME_SUFFIX", "repairing.py"],
)
def test_pattern_leaves_longer_names(text: str) -> None:
    pattern = token_pattern(["assign_readbacks", "OLD_ENV_NAME", "pairing.py"])
    assert pattern is not None
    assert re.search(pattern, text) is None


def test_an_empty_map_sweeps_nothing() -> None:
    assert token_pattern([]) is None
    assert scan(REPO_ROOT, []) == []


# --- the map and the exemptions ---------------------------------------------------


def test_every_tag_names_a_stage() -> None:
    assert stage_problems(RETIRED) == []


def test_map_is_sorted() -> None:
    assert list(RETIRED) == sorted(RETIRED)


def test_exemptions_are_sorted() -> None:
    assert list(SELF_EXEMPT) == sorted(SELF_EXEMPT)


def test_every_exemption_is_a_tracked_file() -> None:
    assert stale_exemptions(REPO_ROOT) == []


def test_only_reached_stages_are_enforced() -> None:
    retired = {
        "due_name": BATCHES[CURRENT_BATCH],
        "old_name": BATCHES[0],
        "later_name": BATCHES[-1],
        "unknown_name": "0z",
    }
    assert enforced(retired) == {"due_name", "old_name"}
    assert stage_problems(retired) == ["unknown_name: tag '0z' names no stage"]


# --- the sweep --------------------------------------------------------------------


def _planted_repo(root: Path, files: Mapping[str, str | bytes]) -> Path:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root


def test_planted_token_is_flagged_where_swept(tmp_path: Path) -> None:
    root = _planted_repo(
        tmp_path,
        {
            "tests/e2e/test_planted.py": "from x import retired_name\n",
            "docs/source/page.rst": "Set ``RETIRED_ENV``.\n",
            "src/osprey/templates/app.j2": "{{ retired_name }}\n",
            "src/osprey/profiles/presets/demo.yml": "retired_name: 1\n",
            "src/osprey/profiles/config_key_manifest.yml": "- retired_name\n",
            "src/osprey/cli/main.py": "retired_name = 1\n",
            "changelog.d/1.removed.md": "Removed retired_name.\n",
            "CHANGELOG.md": "retired_name\n",
            "tests/facility/test_no_retired_shapes.py": "RETIRED = {'retired_name': '1b'}\n",
            "tests/data/blob.bin": b"\x00retired_name\x00",
            "scripts/tool.py": "retired_name_v2 = 1\n",
        },
    )
    assert scan(root, ["retired_name", "RETIRED_ENV"]) == [
        "docs/source/page.rst:1:RETIRED_ENV",
        "src/osprey/profiles/presets/demo.yml:1:retired_name",
        "src/osprey/templates/app.j2:1:retired_name",
        "tests/e2e/test_planted.py:1:retired_name",
    ]


def test_a_stale_exemption_is_refused(tmp_path: Path) -> None:
    root = _planted_repo(tmp_path, {"tests/kept.py": "\n"})
    assert stale_exemptions(root, ("tests/gone.py", "tests/kept.py")) == ["tests/gone.py"]


# --- the tree ---------------------------------------------------------------------


def test_no_retired_token_survives() -> None:
    offenders = scan(REPO_ROOT, enforced(RETIRED))
    assert offenders == [], "retired tokens still present:\n" + "\n".join(offenders)
