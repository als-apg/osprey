"""The build's read of the channel-limits database (``build_limits_check``).

The bridge refuses to start writable on a limits file it cannot parse, and it
used to be the only thing that read the file — twenty minutes into ``osprey
up``, inside a container. These pin the same gate at build time: the same
posture readers decide whether the file is loaded, the same loader reads it,
and its own message is what the build reports.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from osprey.cli.build_limits_check import limits_database_errors

VALID_DB = {
    "defaults": {"min_value": -10.0, "max_value": 10.0, "max_step": 1.0, "confirm": False},
    "SR:C01:MAG:SP": {"writable": True},
}

#: The shape that motivated the check: a ``defaults`` block written against the
#: schema that ``confirm:`` replaced.
STALE_DB = {
    "defaults": {"verification": {"required": True}, "min_value": -1.0, "max_value": 1.0},
}


def _control_system(**overrides) -> dict:
    section = {
        "type": "epics",
        "writes_enabled": True,
        "limits_checking": {
            "enabled": True,
            "allow_unlisted_channels": False,
            "database_path": "data/channel_limits.json",
        },
    }
    section.update(overrides)
    return section


def _render(tmp_path: Path, control_system: dict, database: dict | None = VALID_DB) -> Path:
    """A render holding a ``config.yml`` and, unless told otherwise, a limits file."""
    render_dir = tmp_path / "render"
    (render_dir / "data").mkdir(parents=True, exist_ok=True)
    (render_dir / "data" / "channel_limits.json").unlink(missing_ok=True)
    (render_dir / "config.yml").write_text(
        yaml.safe_dump({"project_name": "demo", "control_system": control_system}),
        encoding="utf-8",
    )
    if database is not None:
        (render_dir / "data" / "channel_limits.json").write_text(json.dumps(database))
    return render_dir


def test_a_valid_database_passes(tmp_path: Path) -> None:
    assert limits_database_errors(_render(tmp_path, _control_system())) == []


def test_a_stale_schema_is_refused_with_the_validators_own_message(tmp_path: Path) -> None:
    errors = limits_database_errors(_render(tmp_path, _control_system(), STALE_DB))

    assert len(errors) == 1
    (line,) = errors
    assert "control_system.limits_checking.database_path" in line
    assert "data/channel_limits.json" in line
    # The validator's wording, not a paraphrase: it names the field that moved.
    assert "verification" in line
    assert "confirm" in line
    # And the posture that made the file load, so the operator can see why.
    assert "writes_enabled" in line
    assert "limits_checking.enabled" in line
    assert "live" in line


def test_a_missing_file_is_the_same_refusal_the_bridge_makes(tmp_path: Path) -> None:
    errors = limits_database_errors(_render(tmp_path, _control_system(), database=None))

    assert len(errors) == 1
    assert "not found" in errors[0]


def test_a_file_that_is_not_json_is_refused(tmp_path: Path) -> None:
    render_dir = _render(tmp_path, _control_system(), database=None)
    (render_dir / "data" / "channel_limits.json").write_text("{not json", encoding="utf-8")

    errors = limits_database_errors(render_dir)

    assert len(errors) == 1
    assert "could not be read or parsed" in errors[0]


def test_a_relative_path_resolves_against_the_render_not_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The render's own config.yml is the anchor — the one the bridge uses."""
    render_dir = _render(tmp_path, _control_system())
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert limits_database_errors(render_dir) == []


def test_an_absolute_path_is_read_where_it_points(tmp_path: Path) -> None:
    db = tmp_path / "shared" / "limits.json"
    db.parent.mkdir()
    db.write_text(json.dumps(STALE_DB))
    section = _control_system()
    section["limits_checking"]["database_path"] = str(db)

    errors = limits_database_errors(_render(tmp_path, section, database=None))

    assert len(errors) == 1
    assert "verification" in errors[0]


def test_a_read_only_deployment_may_stage_the_file_later(tmp_path: Path) -> None:
    """Writes-gated, like the bridge and the compose mount: no target arms writes,
    so a stale or absent database is nothing the deploy will ever read."""
    section = _control_system(writes_enabled=False)

    assert limits_database_errors(_render(tmp_path, section, STALE_DB)) == []
    assert limits_database_errors(_render(tmp_path, section, database=None)) == []


def test_limits_checking_off_reads_nothing(tmp_path: Path) -> None:
    section = _control_system()
    section["limits_checking"]["enabled"] = False

    assert limits_database_errors(_render(tmp_path, section, STALE_DB)) == []


def test_an_unstated_database_path_is_left_to_the_mount_refusal(tmp_path: Path) -> None:
    """The compose renderer already refuses an armed deployment with no path,
    naming the key; a second line here would say it twice."""
    section = _control_system()
    del section["limits_checking"]["database_path"]

    assert limits_database_errors(_render(tmp_path, section, STALE_DB)) == []


def test_a_per_type_block_that_switches_checking_off_wins(tmp_path: Path) -> None:
    """The per-type block overrides whole, and it is the block the live target
    resolves to — the same reading the bridge makes for its lane."""
    section = _control_system(
        connector={
            "epics": {
                "limits_checking": {"enabled": False, "allow_unlisted_channels": False},
            }
        }
    )

    assert limits_database_errors(_render(tmp_path, section, STALE_DB)) == []


def test_a_half_written_per_type_block_reads_nothing_here(tmp_path: Path) -> None:
    """An incomplete block is the ``unrunnable`` gate's refusal, already raised
    before this runs; this check does not build a second opinion on it."""
    section = _control_system(connector={"epics": {"limits_checking": {"enabled": True}}})

    assert limits_database_errors(_render(tmp_path, section, STALE_DB)) == []


def test_a_render_with_no_control_system_section_is_silent(tmp_path: Path) -> None:
    render_dir = tmp_path / "render"
    render_dir.mkdir()
    (render_dir / "config.yml").write_text("project_name: demo\n", encoding="utf-8")

    assert limits_database_errors(render_dir) == []
