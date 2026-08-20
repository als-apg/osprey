"""Tests for the build-time plan index builder.

The index is rendered into a bundled skill during ``osprey build``, so the
properties under test are as much about what it must *not* do — raise, import
bluesky, emit a half-read row — as about what it lists.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from osprey.cli.templates import plan_index
from osprey.cli.templates.plan_index import (
    DESCRIPTION_LIMIT,
    MAX_ROWS,
    build_plan_index,
)
from osprey.services.bluesky_bridge.plan_loader import _SHIPPED_PLANS_DIR, _iter_plan_files
from osprey.services.bluesky_bridge.plan_metadata import extract_plan_metadata_from_source


def shipped_plan_names() -> set[str]:
    """The shipped tier's plan names, derived the same way the builder derives them.

    Deliberately not a hardcoded literal: a test that pins today's three plan
    names would pass while the builder ignored a newly shipped plan.
    """
    return {
        extract_plan_metadata_from_source(path).name
        for path in _iter_plan_files(_SHIPPED_PLANS_DIR)
    }


def write_plan(directory: Path, stem: str, *, name: str | None = None, description: str) -> Path:
    """Write a minimal plan file declaring a literal ``PLAN_METADATA``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.py"
    path.write_text(
        textwrap.dedent(
            f'''
            """A test plan."""

            PLAN_METADATA = {{
                "name": {name or stem!r},
                "description": {description!r},
                "writes": False,
            }}
            '''
        ).lstrip(),
        encoding="utf-8",
    )
    return path


def row_names(index: plan_index.PlanIndex) -> set[str]:
    return {row.name for row in index.rows}


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    return root


def test_every_shipped_plan_gets_exactly_one_row(repo_root: Path) -> None:
    index = build_plan_index({}, repo_root)

    expected = shipped_plan_names()
    assert expected, "the shipped plans_core/ directory should not be empty"
    assert row_names(index) == expected
    assert len(index.rows) == len(expected)
    assert {row.provenance for row in index.rows} == {"shipped"}
    assert index.overflow == 0
    assert index.warnings == ()
    assert index.unreadable_dirs == ()


def test_excluded_plans_bluesky_spelling_drops_the_row(repo_root: Path) -> None:
    victim = sorted(shipped_plan_names())[0]

    index = build_plan_index({"bluesky": {"excluded_plans": [victim]}}, repo_root)

    assert victim not in row_names(index)
    assert row_names(index) == shipped_plan_names() - {victim}


def test_excluded_plans_service_spelling_is_pathsep_joined(repo_root: Path) -> None:
    victims = sorted(shipped_plan_names())[:2]
    assert len(victims) == 2

    config = {"services": {"bluesky": {"excluded_plans": os.pathsep.join(victims)}}}
    index = build_plan_index(config, repo_root)

    assert row_names(index) == shipped_plan_names() - set(victims)


def test_excluded_plans_bare_string_is_one_name(repo_root: Path) -> None:
    victim = sorted(shipped_plan_names())[0]

    index = build_plan_index({"bluesky": {"excluded_plans": victim}}, repo_root)

    assert victim not in row_names(index)


def test_preset_plan_dirs_accepts_a_list(repo_root: Path, tmp_path: Path) -> None:
    plans = tmp_path / "preset_plans"
    write_plan(plans, "extra_plan", description="An extra plan.")

    index = build_plan_index({"bluesky": {"plan_dirs": [str(plans)]}}, repo_root)

    assert "extra_plan" in row_names(index)
    assert row_names(index) == shipped_plan_names() | {"extra_plan"}


def test_preset_plan_dirs_accepts_a_bare_string(repo_root: Path, tmp_path: Path) -> None:
    plans = tmp_path / "preset_plans"
    write_plan(plans, "extra_plan", description="An extra plan.")

    index = build_plan_index({"bluesky": {"plan_dirs": str(plans)}}, repo_root)

    assert "extra_plan" in row_names(index)


def test_preset_plans_carry_the_preset_tier(repo_root: Path, tmp_path: Path) -> None:
    plans = tmp_path / "preset_plans"
    write_plan(plans, "extra_plan", description="An extra plan.")

    index = build_plan_index({"bluesky": {"plan_dirs": str(plans)}}, repo_root)

    by_name = {row.name: row for row in index.rows}
    assert by_name["extra_plan"].provenance == "preset"


def test_service_plan_dir_carries_the_facility_tier(repo_root: Path, tmp_path: Path) -> None:
    plans = tmp_path / "facility_plans"
    write_plan(plans, "site_plan", description="A facility plan.")

    config = {"services": {"bluesky": {"plan_dir": str(plans)}}}
    index = build_plan_index(config, repo_root)

    by_name = {row.name: row for row in index.rows}
    assert by_name["site_plan"].provenance == "facility"


def test_rows_are_sorted_by_trust_tier_then_name(repo_root: Path, tmp_path: Path) -> None:
    preset = tmp_path / "preset_plans"
    write_plan(preset, "zzz_preset", description="Preset plan.")
    write_plan(preset, "aaa_preset", description="Preset plan.")
    facility = tmp_path / "facility_plans"
    write_plan(facility, "aaa_facility", description="Facility plan.")

    config = {
        "bluesky": {"plan_dirs": str(preset)},
        "services": {"bluesky": {"plan_dir": str(facility)}},
    }
    index = build_plan_index(config, repo_root)

    tiers = [row.provenance for row in index.rows]
    assert tiers == sorted(tiers, key=["shipped", "preset", "facility"].index)
    preset_names = [row.name for row in index.rows if row.provenance == "preset"]
    assert preset_names == ["aaa_preset", "zzz_preset"]
    assert index.rows[-1].name == "aaa_facility"


def test_relative_plan_dir_resolves_against_repo_root_not_cwd(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_plan(repo_root / "plans", "repo_plan", description="Lives under the repo root.")
    cwd = tmp_path / "elsewhere"
    write_plan(cwd / "plans", "cwd_plan", description="Lives under the CWD.")
    monkeypatch.chdir(cwd)

    index = build_plan_index({"bluesky": {"plan_dirs": "plans"}}, repo_root)

    assert "repo_plan" in row_names(index)
    assert "cwd_plan" not in row_names(index)


def test_absolute_plan_dir_is_used_unchanged(repo_root: Path, tmp_path: Path) -> None:
    plans = tmp_path / "absolute_plans"
    write_plan(plans, "absolute_plan", description="Reached by absolute path.")
    assert plans.is_absolute()

    index = build_plan_index({"bluesky": {"plan_dirs": str(plans)}}, repo_root)

    assert "absolute_plan" in row_names(index)


def test_missing_absolute_plan_dir_flags_unreadable_without_raising(
    repo_root: Path, tmp_path: Path
) -> None:
    missing = tmp_path / "does_not_exist"
    assert not missing.exists()

    index = build_plan_index({"bluesky": {"plan_dirs": str(missing)}}, repo_root)

    assert index.has_unreadable_dirs
    assert index.unreadable_dirs == (str(missing),)
    assert row_names(index) == shipped_plan_names()


def test_missing_plan_dir_does_not_suppress_the_other_configured_tier(
    repo_root: Path, tmp_path: Path
) -> None:
    missing = tmp_path / "does_not_exist"
    facility = tmp_path / "facility_plans"
    write_plan(facility, "site_plan", description="A facility plan.")

    config = {
        "bluesky": {"plan_dirs": str(missing)},
        "services": {"bluesky": {"plan_dir": str(facility)}},
    }
    index = build_plan_index(config, repo_root)

    assert "site_plan" in row_names(index)
    assert index.unreadable_dirs == (str(missing),)


def test_absent_shipped_dir_is_not_reported_as_unreadable(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plan_index, "_SHIPPED_PLANS_DIR", tmp_path / "no_shipped_plans")

    index = build_plan_index({}, repo_root)

    assert index.rows == ()
    assert index.unreadable_dirs == ()


def test_zero_plans_returns_an_empty_index_not_none(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plan_index, "_SHIPPED_PLANS_DIR", tmp_path / "no_shipped_plans")

    index = build_plan_index({}, repo_root)

    assert isinstance(index, plan_index.PlanIndex)
    assert index.rows == ()
    assert index.overflow == 0


def test_malformed_metadata_is_skipped_and_warned_about(repo_root: Path, tmp_path: Path) -> None:
    plans = tmp_path / "preset_plans"
    write_plan(plans, "good_plan", description="A well-formed plan.")
    (plans / "bad_plan.py").write_text(
        'PLAN_METADATA = {"name": "bad_plan", "description": "No writes key."}\n',
        encoding="utf-8",
    )

    index = build_plan_index({"bluesky": {"plan_dirs": str(plans)}}, repo_root)

    assert "good_plan" in row_names(index)
    assert "bad_plan" not in row_names(index)
    assert len(index.warnings) == 1
    assert "bad_plan.py" in index.warnings[0]
    assert index.unreadable_dirs == ()


def test_malformed_metadata_never_emits_a_partial_row(repo_root: Path, tmp_path: Path) -> None:
    plans = tmp_path / "preset_plans"
    plans.mkdir(parents=True, exist_ok=True)
    (plans / "partial.py").write_text(
        'PLAN_METADATA = {"name": "partial", "description": "Missing writes."}\n',
        encoding="utf-8",
    )
    (plans / "unknown_key.py").write_text(
        "PLAN_METADATA = {\n"
        '    "name": "unknown_key",\n'
        '    "description": "Declares a key outside the contract.",\n'
        '    "writes": False,\n'
        '    "provenance": "facility",\n'
        "}\n",
        encoding="utf-8",
    )

    index = build_plan_index({"bluesky": {"plan_dirs": str(plans)}}, repo_root)

    assert row_names(index) == shipped_plan_names()
    assert len(index.warnings) == 2


def test_no_description_exceeds_the_limit(repo_root: Path, tmp_path: Path) -> None:
    plans = tmp_path / "preset_plans"
    write_plan(plans, "long_plan", description="word " * 200)
    write_plan(
        plans,
        "long_sentence",
        description="A single very long sentence " + "that keeps going " * 20 + "and ends.",
    )

    index = build_plan_index({"bluesky": {"plan_dirs": str(plans)}}, repo_root)

    assert index.rows
    for row in index.rows:
        assert len(row.description) <= DESCRIPTION_LIMIT, row.name


def test_description_is_truncated_to_the_first_sentence(repo_root: Path, tmp_path: Path) -> None:
    plans = tmp_path / "preset_plans"
    write_plan(
        plans,
        "two_sentences",
        description="The first sentence. The second one should not survive.",
    )

    index = build_plan_index({"bluesky": {"plan_dirs": str(plans)}}, repo_root)

    by_name = {row.name: row for row in index.rows}
    assert by_name["two_sentences"].description == "The first sentence."


def test_description_whitespace_is_collapsed(repo_root: Path, tmp_path: Path) -> None:
    plans = tmp_path / "preset_plans"
    write_plan(plans, "wrapped", description="Wrapped\n   across   lines.")

    index = build_plan_index({"bluesky": {"plan_dirs": str(plans)}}, repo_root)

    by_name = {row.name: row for row in index.rows}
    assert by_name["wrapped"].description == "Wrapped across lines."


def test_row_count_is_capped_and_the_rest_are_counted_as_overflow(
    repo_root: Path, tmp_path: Path
) -> None:
    plans = tmp_path / "many_plans"
    total = MAX_ROWS + 1
    for i in range(total):
        write_plan(plans, f"plan_{i:03d}", description=f"Generated plan {i}.")

    index = build_plan_index({"bluesky": {"plan_dirs": str(plans)}}, repo_root)

    expected_total = total + len(shipped_plan_names())
    assert len(index.rows) == MAX_ROWS
    assert index.overflow == expected_total - MAX_ROWS
    assert index.overflow > 0


def test_higher_trust_layer_wins_a_name_collision(repo_root: Path, tmp_path: Path) -> None:
    preset = tmp_path / "preset_plans"
    write_plan(preset, "shared", description="Preset version.")
    facility = tmp_path / "facility_plans"
    write_plan(facility, "shared", description="Facility version.")

    config = {
        "bluesky": {"plan_dirs": str(preset)},
        "services": {"bluesky": {"plan_dir": str(facility)}},
    }
    index = build_plan_index(config, repo_root)

    matches = [row for row in index.rows if row.name == "shared"]
    assert len(matches) == 1
    assert matches[0].provenance == "facility"
    assert matches[0].description == "Facility version."


def test_plan_name_comes_from_metadata_not_the_file_stem(repo_root: Path, tmp_path: Path) -> None:
    plans = tmp_path / "preset_plans"
    write_plan(plans, "file_stem", name="declared_name", description="Named by metadata.")

    index = build_plan_index({"bluesky": {"plan_dirs": str(plans)}}, repo_root)

    assert "declared_name" in row_names(index)
    assert "file_stem" not in row_names(index)


def test_importing_the_builder_does_not_import_bluesky(tmp_path: Path) -> None:
    """A clean interpreter must not pull bluesky in when this module is imported.

    Run out of process on purpose: by the time this file runs, another test may
    already have imported the bridge, which would make an in-process
    ``sys.modules`` assertion pass no matter what this module imports.
    """
    script = textwrap.dedent(
        """
        import sys

        import osprey.cli.templates.plan_index as module

        forbidden = {"bluesky", "ophyd", "tiled", "bluesky_queueserver"}
        leaked = sorted({name.split(".")[0] for name in sys.modules} & forbidden)
        print("FILE=" + module.__file__)
        print("LEAKED=" + ",".join(leaked))
        """
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    reported = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert Path(reported["FILE"]) == Path(plan_index.__file__), (
        "the subprocess imported a different checkout's module"
    )
    assert reported["LEAKED"] == "", f"importing plan_index leaked: {reported['LEAKED']}"
