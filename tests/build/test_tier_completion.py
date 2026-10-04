"""The tier model survives only in the files a later stage removes it from.

Channel-finder indexes are views the build writes; nothing selects a tier any
more. ``PATTERN`` matches what is left of the tier model: the staged
``tiers/`` tree and its readers, the tier selector in the manifest paths, and
the template channel database those readers load. Every tracked file under
``SCAN_PATHS`` other than the two guards that matches is in ``ALLOWLIST``,
tagged with the stage that removes its last match. An entry whose stage is at
or before ``CURRENT_BATCH`` must no longer match, and an entry that no longer
matches must be dropped.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path

from tests.facility._batches import BATCHES, CURRENT_BATCH

REPO_ROOT = Path(__file__).resolve().parents[2]

PATTERN = (
    r"tier_dir|resolved_tier|TierSpec|tiers/|databases\.template|TemplateChannelDatabase"
    r"|from \.template import|_manifest_tier"
)

SCAN_PATHS: tuple[str, ...] = ("src", "scripts", "packages", "tests")

#: The guards that spell the pattern's tokens as data; the scan never reads them.
SELF_EXEMPT: tuple[str, ...] = (
    "tests/build/test_tier_completion.py",
    "tests/facility/test_no_retired_shapes.py",
)

#: Matching file -> the stage that removes its last match.
ALLOWLIST: dict[str, str] = {
    "scripts/facility_demo/fingerprint.py": "7d2",
    "src/osprey/cli/build_cmd.py": "7d",
    "src/osprey/cli/mml_cmd.py": "7e",
    "src/osprey/cli/templates/scaffolding.py": "7d2",
    "src/osprey/services/channel_finder/__init__.py": "7e",
    "src/osprey/services/channel_finder/benchmarks/generator.py": "7d2",
    "src/osprey/services/channel_finder/databases/__init__.py": "7e",
    "src/osprey/services/channel_finder/tools/generate_from_spec.py": "7d2",
    "src/osprey/services/virtual_accelerator/manifest/build.py": "7e",
    "src/osprey/services/virtual_accelerator/manifest/loaders.py": "7e",
    "src/osprey/services/virtual_accelerator/manifest/paths.py": "7e",
    "src/osprey/simulation/channel_schema.py": "7d2",
    "src/osprey/templates/apps/control_assistant/data/README.md": "7d2",
    "tests/build/test_modes.py": "7e",
    "tests/cli/test_build_cmd.py": "7d2",
    "tests/cli/test_build_graph_index.py": "7e",
    "tests/cli/test_lifecycle_repo_fixture.py": "7d2",
    "tests/cli/test_mml_build_recipes.py": "7e",
    "tests/cli/test_mml_emit.py": "7e",
    "tests/cli/test_profile_data_root.py": "7d2",
    "tests/cli/test_scaffold_pull.py": "7d2",
    "tests/facility/golden/cf_index_pre_line/MANIFEST.json": "7d2",
    "tests/facility/golden/demo_fingerprint.json": "7d2",
    "tests/facility/golden/in_context_size.json": "7d2",
    "tests/facility/test_generator_records.py": "7d2",
    "tests/fixtures/lifecycle_repo.py": "7d2",
    "tests/services/channel_finder/benchmarks/test_benchmark_datasets.py": "7d2",
    "tests/services/channel_finder/benchmarks/test_generator.py": "7d2",
    "tests/services/channel_finder/databases/test_template_presentation.py": "7e",
    "tests/services/channel_finder/databases/test_template_suffix_map.py": "7e",
    "tests/services/channel_finder/test_generate_from_spec.py": "7d2",
    "tests/services/channel_finder/test_tier_db_drift.py": "7e",
    "tests/services/channel_finder/tools/test_preview_database.py": "7d2",
    "tests/services/facility_knowledge/test_demo_ttl_consistency.py": "7e",
    "tests/services/facility_knowledge/test_ttl_generator_direction.py": "7e",
    "tests/services/facility_knowledge/test_ttl_generator_emitter.py": "7e",
    "tests/services/facility_knowledge/test_ttl_generator_model.py": "7e",
    "tests/services/facility_knowledge/test_ttl_generator_ontology.py": "7e",
    "tests/simulation/test_seed_logbook_naming.py": "7d2",
    "tests/templates/test_hierarchical_db_preset_copy.py": "7d2",
    "tests/va/test_build_time_manifest.py": "7d2",
}


def scan(root: Path) -> set[str]:
    """Return the tracked files under ``SCAN_PATHS`` that match ``PATTERN``."""
    result = subprocess.run(
        [
            "git",
            "grep",
            "-l",
            "-E",
            "-e",
            PATTERN,
            "--",
            *SCAN_PATHS,
            *(f":(exclude){path}" for path in SELF_EXEMPT),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    return set(result.stdout.split())


def violations(
    hits: set[str], allowlist: Mapping[str, str], current: int = CURRENT_BATCH
) -> list[str]:
    """Return one line per matching file or allowlist entry the guard refuses."""
    problems = [
        f"{path}: matches the tier pattern and has no allowlist entry"
        for path in hits - set(allowlist)
    ]
    for path, stage in allowlist.items():
        if stage not in BATCHES:
            problems.append(f"{path}: tag {stage!r} names no stage")
        elif path not in hits:
            problems.append(f"{path}: no longer matches; drop its entry")
        elif BATCHES.index(stage) <= current:
            problems.append(f"{path}: tagged {stage}, still matches")
    return sorted(problems)


def test_allowlist_is_sorted() -> None:
    assert list(ALLOWLIST) == sorted(ALLOWLIST)


def test_every_tag_names_a_stage() -> None:
    assert sorted(path for path, stage in ALLOWLIST.items() if stage not in BATCHES) == []


def test_guard_refuses_due_stale_and_unlisted_entries() -> None:
    hits = {"a.py", "b.py", "c.py", "new.py"}
    allowlist = {
        "a.py": BATCHES[CURRENT_BATCH],
        "b.py": BATCHES[-1],
        "c.py": "0z",
        "gone.py": BATCHES[-1],
    }
    assert violations(hits, allowlist) == [
        f"a.py: tagged {BATCHES[CURRENT_BATCH]}, still matches",
        "c.py: tag '0z' names no stage",
        "gone.py: no longer matches; drop its entry",
        "new.py: matches the tier pattern and has no allowlist entry",
    ]


def test_tier_model_survives_only_where_the_allowlist_says() -> None:
    assert violations(scan(REPO_ROOT), ALLOWLIST) == []
