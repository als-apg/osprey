"""Retired symbols, paths and environment names never come back.

``RETIRED`` lists each token whose LAST producer or reader is deleted; a token
joins the list with that deletion and must not appear in any tracked text file
under ``SCAN_ROOTS``.

The list holds symbols, dotted module paths, file paths and environment names
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

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Tracked trees the sweep reads: tests (``tests/e2e`` included), scripts, docs,
#: the agent plugins, the rendered templates, the profile presets and catalogs,
#: and the deployment code and images.
SCAN_ROOTS: tuple[str, ...] = (
    "docker",
    "docs",
    "plugins",
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

#: Every retired token, sorted.
RETIRED: tuple[str, ...] = (
    "/api/state/init",
    "ACTIVE_SCENARIO_FILENAME",
    "BUILD_TTL_COMMAND",
    "CohostDriver",
    "CohostRunner",
    "DIRECTION_SOURCE_HEADER",
    "DIRECTION_UNDERIVABLE",
    "DatabaseWriteError",
    "DirectionSource",
    "FACILITY_PREFIX_CONFIG_KEY",
    "GRAPHDB_BUILD_INDEX_COMMAND",
    "GRAPHDB_SEED_COMMAND",
    "GRAPH_MALFORMED",
    "GRAPH_NO_TTL",
    "GRAPH_SOURCE_PARADIGM",
    "ListSourceInvalid",
    "MODIFIED_REFUSAL",
    "ManifestPaths",
    "P_CONFIDENCE",
    "P_PROTOCOL",
    "SignalSentence",
    "TemplateChannelDatabase",
    "TestKindAwareNoiseFloor",
    "TierSpec",
    "VA_CHANNELS_FILE",
    "VA_ENTRYPOINT_MODULE",
    "VA_NOISE_LEVEL",
    "_DerivedDevices",
    "_GRAPHDB_RECOVERY_HINT",
    "_check_empty_facility_prefix",
    "_control_system_simulation_file",
    "_extend_pvdb",
    "_hierarchical_write",
    "_manifest_tier",
    "_resolve_ttl",
    "_with_derived_simulation_file",
    "allow_unlisted_channels",
    "als_u_ar",
    "approval_hook_wired",
    "assign_readbacks",
    "build_tiers",
    "channel_databases/tiers",
    "derive_bands",
    "engine_from_connector_config",
    "facility_ontology.json",
    "facility_vocabulary",
    "fieldDescription",
    "google_sheets",
    "lattice_init",
    "lattice_json",
    "load_canonical_ring",
    "machine_state_channels.json",
    "middle_layer_duckdb",
    "narad_p:confidence",
    "narad_p:protocol",
    "ordinalInFacility",
    "ordinalInSection",
    "osprey.channel_roster.database",
    "osprey.channel_roster.graph",
    "osprey.connectors.channel_taxonomy",
    "osprey.services.channel_finder.databases.template",
    "osprey.services.channel_finder.naming",
    "osprey.services.channel_finder.tools.generate_from_spec",
    "osprey.services.virtual_accelerator.manifest.standin_defaults",
    "osprey.simulation.archiver_seed",
    "osprey.simulation.channel_schema",
    "osprey.simulation.engine",
    "osprey.simulation.expressions",
    "osprey.simulation.facility_spec",
    "osprey.simulation.lattice",
    "osprey.simulation.machine",
    "osprey.simulation.procedural",
    "osprey.simulation.series",
    "osprey_connectors.channel_taxonomy",
    "osprey_connectors.simulation.archiver_seed",
    "osprey_connectors.simulation.engine",
    "osprey_connectors.simulation.expressions",
    "osprey_connectors.simulation.machine",
    "osprey_connectors.simulation.procedural",
    "pairing.py",
    "parse_direction_source",
    "prune_csv_build_artifacts",
    "pyaml.configuration",
    "read_database_roster",
    "read_direction_source",
    "read_graph_roster",
    "resolve_facility_name",
    "resolved_tier",
    "ringDescription",
    "scripts/facility_demo/_limits.py",
    "scripts/facility_demo/_measurement.py",
    "scripts/facility_demo/_models.py",
    "scripts/facility_demo/_records.py",
    "scripts/facility_demo/_scenarios.py",
    "scripts/facility_demo/_seeds.py",
    "scripts/facility_demo/capture_nominals.py",
    "scripts/facility_demo/fingerprint.py",
    "scripts/va/derive_bands.py",
    "scripts/va/pyat_model_demo.py",
    "scripts/va/run_va.sh",
    "sourceSectionId",
    "src/osprey/templates/apps/channel_finder_standalone/data/channel_databases/hierarchical.json",
    "src/osprey/templates/apps/control_assistant/data/channel_databases/examples",
    "src/osprey/templates/apps/control_assistant/data/lattice/als_u_ar.mat",
    "src/osprey/templates/apps/control_assistant/data/mml",
    "src/osprey/templates/apps/control_assistant/data/simulation/lattice.json",
    "src/osprey/templates/apps/control_assistant/data/simulation/machine.json",
    "src/osprey/templates/apps/control_assistant/data/simulation/va_bindings.json",
    "subfieldDescription",
    "tests/connectors/test_mock_archiver_simfile_fallback.py",
    "tests/templates/test_boot_band_invariant.py",
    "tests/templates/test_channel_limits_va.py",
    "tests/va/test_derive_bands_floor.py",
    "tier_dir",
    "tier_mode_conflict",
    "va_graph_deferred",
)

_NAME_CHAR = "A-Za-z0-9_"


def token_pattern(tokens: Iterable[str]) -> str | None:
    """Return one pattern matching any of ``tokens`` as a whole name, or None."""
    ordered = sorted(set(tokens), key=lambda token: (-len(token), token))
    if not ordered:
        return None
    alternation = "|".join(re.escape(token) for token in ordered)
    return rf"(?<![{_NAME_CHAR}])(?:{alternation})(?![{_NAME_CHAR}])"


def scan(root: Path, tokens: Iterable[str], exempt: Iterable[str] = SELF_EXEMPT) -> list[str]:
    """Return ``path:line:token`` for every token in a swept file."""
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


def test_an_empty_list_sweeps_nothing() -> None:
    assert token_pattern([]) is None
    assert scan(REPO_ROOT, []) == []


# --- the list and the exemptions --------------------------------------------------


def test_list_is_sorted() -> None:
    assert list(RETIRED) == sorted(RETIRED)


def test_exemptions_are_sorted() -> None:
    assert list(SELF_EXEMPT) == sorted(SELF_EXEMPT)


def test_every_exemption_is_a_tracked_file() -> None:
    assert stale_exemptions(REPO_ROOT) == []


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
            "tests/facility/test_no_retired_shapes.py": "RETIRED = ('retired_name',)\n",
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
    offenders = scan(REPO_ROOT, RETIRED)
    assert offenders == [], "retired tokens still present:\n" + "\n".join(offenders)
