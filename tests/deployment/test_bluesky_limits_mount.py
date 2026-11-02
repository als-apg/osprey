"""The Bluesky render and the build's limits check decide from the login tier.

``control_system.writes`` is the render's write ceiling: ``allowed`` or
``never``, target-blind. Everything the build derives from write posture for a
Bluesky deployment — the per-lane ``writes_enabled`` the compose template gates
its limits-database and control-tree mounts on, the mount refusal in
``resolve_limits_mount``, and the build's "limits database readable?" check —
reads that tier, so an ``allowed`` render mounts and checks the limits database
and the tree, and a ``never`` render does neither.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_limits_check import limits_database_errors
from osprey.deployment.compose_generator import (
    drives_target_write_posture,
    resolve_limits_mount,
)
from osprey.deployment.errors import DeploymentPreconditionError
from osprey_connectors.types import tier_limits_database_targets
from tests.deployment.test_lane_compose import (
    BLUESKY_PORT,
    CA_NAME_SERVERS_REQUIRED_PREFIX,
    CONTROL_TREE_SOURCE,
    LIMITS_DB_MOUNT,
    SECOND_LANE_PORT,
    _context,
    _lane_block,
    _limits_mounts,
    _render,
)

_LIMITS_ON = {
    "enabled": True,
    "mode": "exclusive",
    "database_path": "data/channel_limits.json",
}

#: An entry written against the schema ``confirm:`` replaced: a file the
#: bridge refuses, so the build check has something to report.
_STALE_DB = {
    "SR:C01:MAG:SP": {
        "writable": True,
        "verification": {"required": True},
        "min_value": -1.0,
        "max_value": 1.0,
    },
}


def _section(writes: Any, **extra: Any) -> dict[str, Any]:
    section: dict[str, Any] = {"type": "epics", "limits_checking": dict(_LIMITS_ON)}
    if writes is not None:
        section["writes"] = writes
    section.update(extra)
    return section


def _two_lanes() -> dict[str, dict[str, Any]]:
    return {
        "bluesky": _lane_block(
            BLUESKY_PORT,
            target="live",
            ca_name_servers=f"{CA_NAME_SERVERS_REQUIRED_PREFIX}set it}}",
        ),
        "bluesky_va": _lane_block(SECOND_LANE_PORT, target="va"),
    }


def _tree_mounts(rendered: dict[str, Any], service: str) -> list[str]:
    return [v for v in rendered["services"][service]["volumes"] if CONTROL_TREE_SOURCE in v]


def _environment(rendered: dict[str, Any], service: str) -> dict[str, Any]:
    return rendered["services"][service].get("environment") or {}


# ---------------------------------------------------------------------------
# The per-lane posture the template gates on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("section", "armed"),
    [
        ({"writes": "allowed"}, True),
        ({"writes": "never"}, False),
        # Only the literal string arms.
        ({"writes": True}, False),
        ({"writes": "Allowed"}, False),
        # The tier key decides over any retired leaf it sits beside.
        ({"writes": "never", "writes_enabled": True}, False),
        ({"writes": "allowed", "writes_enabled": False}, True),
        # A render with no tier key is `never`: retired leaves are not read at
        # run time (D2; rows `tier-absent-old-armed-flat`, `tier-absent-unarmed`).
        ({"writes_enabled": True}, False),
        ({"writes_enabled": False}, False),
        (None, False),
    ],
)
def test_a_lanes_posture_is_the_renders_tier(section: Any, armed: bool) -> None:
    lane = {"bluesky": {"port": BLUESKY_PORT}}

    assert drives_target_write_posture(lane, section) == {"bluesky": armed}


def test_the_tier_arms_every_lane_whatever_its_target() -> None:
    """The tier is a ceiling, not a selection: an ``allowed`` tier arms every
    lane, whichever machine each one serves, and a ``never`` tier arms none."""
    lanes = _two_lanes()
    allowed = {"type": "epics", "writes": "allowed"}

    assert drives_target_write_posture(lanes, allowed) == {"bluesky": True, "bluesky_va": True}
    assert drives_target_write_posture(lanes, {"type": "epics", "writes": "never"}) == {
        "bluesky": False,
        "bluesky_va": False,
    }


# ---------------------------------------------------------------------------
# The rendered compose file
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lanes",
    [
        pytest.param({"bluesky": _lane_block(BLUESKY_PORT)}, id="single-lane"),
        pytest.param(_two_lanes(), id="two-lanes"),
    ],
)
def test_an_allowed_render_mounts_the_limits_db_and_the_tree(lanes: dict[str, Any]) -> None:
    rendered = _render(
        _context(
            lanes=lanes,
            deployed_services=[*lanes, "virtual_accelerator"],
            control_system={"type": "epics", "writes": "allowed"},
        )
    )

    services = (
        ["bluesky-bridge", "queueserver", "bluesky-va-bridge", "bluesky-va-queueserver"]
        if "bluesky_va" in lanes
        else ["bluesky-bridge", "queueserver"]
    )
    for service in services:
        assert _limits_mounts(rendered, service) == [LIMITS_DB_MOUNT], service
    for queueserver in services[1::2]:
        assert len(_tree_mounts(rendered, queueserver)) == 1, queueserver
        assert "OSPREY_CONTROL_CONTEXT_TREE" in _environment(rendered, queueserver)


@pytest.mark.parametrize(
    "control_system",
    [
        pytest.param({"type": "epics", "writes": "never"}, id="never"),
        pytest.param(
            {"type": "epics", "writes": "never", "writes_enabled": True}, id="never-over-retired"
        ),
    ],
)
def test_a_never_render_mounts_neither(control_system: dict[str, Any]) -> None:
    context = _context(
        lanes=_two_lanes(),
        deployed_services=["bluesky", "bluesky_va", "virtual_accelerator"],
        control_system=control_system,
    )
    assert "limits_mount" not in context
    # Even a mount the generator handed over is not rendered on a never lane.
    context["limits_mount"] = {
        "source": "./data/channel_limits.json",
        "target": "/app/project/data/channel_limits.json",
    }
    rendered = _render(context)

    for service in ("bluesky-bridge", "queueserver", "bluesky-va-bridge", "bluesky-va-queueserver"):
        assert _limits_mounts(rendered, service) == [], service
        assert _tree_mounts(rendered, service) == [], service
        assert "OSPREY_CONTROL_CONTEXT_TREE" not in _environment(rendered, service), service


# ---------------------------------------------------------------------------
# The mount refusal
# ---------------------------------------------------------------------------


def _config(tmp_path: Path, section: dict[str, Any]) -> dict[str, Any]:
    return {"control_system": section, "config_dir": str(tmp_path)}


def test_an_allowed_render_with_checking_on_refuses_a_missing_database(tmp_path: Path) -> None:
    with pytest.raises(DeploymentPreconditionError) as excinfo:
        resolve_limits_mount(_config(tmp_path, _section("allowed")), str(tmp_path), "")

    assert "control_system.writes: allowed" in excinfo.value.reason
    assert "writes_enabled" not in excinfo.value.reason
    assert "limits_checking.enabled: false" in excinfo.value.remedy


def test_an_allowed_render_refuses_an_unset_path(tmp_path: Path) -> None:
    section = _section("allowed")
    del section["limits_checking"]["database_path"]

    with pytest.raises(DeploymentPreconditionError) as excinfo:
        resolve_limits_mount(_config(tmp_path, section), str(tmp_path), "")

    assert "control_system.writes: allowed" in excinfo.value.reason
    assert "control_system.writes: never" in excinfo.value.remedy


@pytest.mark.parametrize(
    "section",
    [
        pytest.param(_section("never"), id="never"),
        pytest.param(_section("never", writes_enabled=True), id="never-over-retired"),
        pytest.param(
            {**_section("allowed"), "limits_checking": {**_LIMITS_ON, "enabled": False}},
            id="allowed-checking-off",
        ),
    ],
)
def test_nothing_that_never_opens_the_database_refuses_its_absence(
    tmp_path: Path, section: dict[str, Any]
) -> None:
    assert resolve_limits_mount(_config(tmp_path, section), str(tmp_path), "") is None


def test_an_allowed_render_with_the_file_present_is_mounted(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "channel_limits.json").write_text("{}")

    assert resolve_limits_mount(_config(tmp_path, _section("allowed")), str(tmp_path), "build") == {
        "source": "./build/data/channel_limits.json",
        "target": "/app/project/data/channel_limits.json",
    }


def test_the_tier_arms_limits_for_every_reachable_target() -> None:
    """A per-type block that checks limits on a machine whose own retired leaf
    is unarmed still needs the database once the tier allows writes."""
    section = {
        "type": "epics",
        "writes": "allowed",
        "connector": {
            "epics": {"writes_enabled": False},
            "virtual_accelerator": {
                "port": 5064,
                "limits_checking": {"enabled": True, "mode": "exclusive"},
            },
        },
    }

    assert tier_limits_database_targets(section, path_configured=False) == ["va"]
    assert tier_limits_database_targets(section, path_configured=True) == ["va"]
    assert tier_limits_database_targets({**section, "writes": "never"}, path_configured=True) == []


# ---------------------------------------------------------------------------
# The build's "limits database readable?" check
# ---------------------------------------------------------------------------


def _render_dir(tmp_path: Path, section: dict[str, Any]) -> Path:
    render_dir = tmp_path / "render"
    (render_dir / "data").mkdir(parents=True)
    (render_dir / "config.yml").write_text(
        yaml.safe_dump({"project_name": "demo", "control_system": section}), encoding="utf-8"
    )
    (render_dir / "data" / "channel_limits.json").write_text(json.dumps(_STALE_DB))
    return render_dir


def test_an_allowed_render_checks_the_database(tmp_path: Path) -> None:
    errors = limits_database_errors(_render_dir(tmp_path, _section("allowed")))

    assert len(errors) == 1
    assert errors[0].startswith("control_system.writes and ")
    assert "verification" in errors[0]


@pytest.mark.parametrize(
    "section",
    [
        pytest.param(_section("never"), id="never"),
        pytest.param(_section("never", writes_enabled=True), id="never-over-retired"),
    ],
)
def test_a_never_render_checks_nothing(tmp_path: Path, section: dict[str, Any]) -> None:
    assert limits_database_errors(_render_dir(tmp_path, section)) == []


def test_a_retired_leaf_render_checks_nothing(tmp_path: Path) -> None:
    """A render that still spells only a retired per-type leaf has no tier key,
    so its tier is ``never`` (D2; row ``tier-absent-old-armed-connector``):
    nothing opens the database and the build reads nothing."""
    section = _section(
        None, connector={"virtual_accelerator": {"port": 5064, "writes_enabled": True}}
    )

    assert limits_database_errors(_render_dir(tmp_path, section)) == []
