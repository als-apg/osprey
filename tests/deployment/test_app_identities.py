"""A service's roles decide what the build stages into it and which identities it provisions.

A container gets the staged login→tier map because its service declares
``drives_target``, and the control-identity module because it declares
``control_identity`` — in its ``services.<key>`` block (an application's
manifest projects both there) or, for a bundled template, in
:data:`~osprey.deployment.compose_generator.BUNDLED_SERVICE_ROLES`. A projected
service's audit identity is its block's ``audit_identity``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from osprey.bluesky_bridge_connection import LANE_KEYS
from osprey.deployment import compose_generator, status_display
from osprey.deployment.compose_generator import (
    FIXED_SERVICE_AUDIT_IDENTITIES,
    SERVICE_ROLE_CONTROL_IDENTITY,
    SERVICE_ROLE_DRIVES_TARGET,
    drives_target_services,
    service_audit_identities,
    service_role,
)
from osprey.deployment.errors import DeploymentPreconditionError
from osprey.utils.workspace import SPOOL_DIR_RELPATH, STATE_DIR_RELPATH

ROLES = (SERVICE_ROLE_DRIVES_TARGET, SERVICE_ROLE_CONTROL_IDENTITY)

#: The map a build would stage; its content is not under test here.
LOGIN_TIERS = {"alice": "allowed", "bob": "never"}

_TEMPLATE = "services:\n  {key}:\n    image: demo\n"


def _block(**roles: object) -> dict:
    return {"services": {"knob": dict(roles)}}


# ---------------------------------------------------------------------------
# (1) service_role
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ROLES)
def test_a_projected_block_holds_a_role_only_for_the_literal_true(role: str) -> None:
    assert service_role(_block(**{role: True}), "knob", role) is True


@pytest.mark.parametrize("value", ["true", 1, "yes", None, False])
@pytest.mark.parametrize("role", ROLES)
def test_any_other_value_fails_closed(role: str, value: object) -> None:
    assert service_role(_block(**{role: value}), "knob", role) is False


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize(
    "config",
    [{}, {"services": {}}, {"services": {"knob": {}}}, {"services": {"knob": "x"}}, None],
)
def test_a_missing_key_or_a_non_mapping_block_fails_closed(role: str, config: object) -> None:
    assert service_role(config, "knob", role) is False


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("key", [*LANE_KEYS, "dispatch_worker"])
def test_every_bundled_lane_and_the_worker_hold_both_roles_with_no_block(
    role: str, key: str
) -> None:
    assert service_role({}, key, role) is True


def test_drives_target_services_lists_role_holders_in_services_order() -> None:
    config = {
        "services": {
            "knob": {"drives_target": True},
            "postgresql": {},
            "dispatch_worker": {},
            "dial": {"drives_target": "true"},
            "bluesky": {},
        }
    }
    assert drives_target_services(config) == ["knob", "dispatch_worker", "bluesky"]
    assert drives_target_services({}) == []


# ---------------------------------------------------------------------------
# (2) the login map is staged into a drives_target context
# ---------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A deployment repo with two projected service templates, entered as the build does."""
    repo = tmp_path / "repo"
    for key in ("knob", "plain"):
        service = repo / "services" / key
        service.mkdir(parents=True)
        (service / "docker-compose.yml.j2").write_text(_TEMPLATE.format(key=key))
    (repo / "config.yml").write_text("project_name: demo\n")
    monkeypatch.chdir(repo)
    return repo


def _deploy_config(repo: Path, services: dict) -> dict:
    return {
        "project_name": "demo",
        "project_root": str(repo),
        "config_dir": str(repo),
        "build_dir": "./build",
        "services": services,
        "deployment": {},
        "system": {"timezone": "UTC"},
        "deployed_services": list(services),
    }


def _staged_document(key: str) -> dict:
    return yaml.safe_load((Path("build") / "services" / key / "config.yml").read_text())


def test_the_map_is_staged_only_into_a_projected_drives_target_context(repo: Path) -> None:
    from osprey.deployment.compose_generator import setup_build_dir

    config = _deploy_config(repo, {"knob": {"drives_target": True}, "plain": {}})
    for key in ("knob", "plain"):
        setup_build_dir(
            f"services/{key}/docker-compose.yml.j2", config, {}, login_tiers=LOGIN_TIERS
        )

    assert _staged_document("knob")["control_system"]["login_tiers"] == LOGIN_TIERS
    plain = _staged_document("plain")
    assert "login_tiers" not in ((plain or {}).get("control_system") or {})


# ---------------------------------------------------------------------------
# (3) the control-identity module is staged into a control_identity context
# ---------------------------------------------------------------------------


def test_the_module_is_staged_only_into_a_projected_control_identity_context(
    tmp_path: Path,
) -> None:
    config = {"services": {"knob": {"control_identity": True}, "plain": {}}}
    knob, plain = tmp_path / "knob", tmp_path / "plain"
    knob.mkdir()
    plain.mkdir()

    assert compose_generator._stage_control_identity_module(config, "services/knob", str(knob))
    assert (knob / compose_generator.CONTROL_IDENTITY_DIRNAME).is_dir()
    assert not compose_generator._stage_control_identity_module(
        config, "services/plain", str(plain)
    )
    assert list(plain.iterdir()) == []


def test_a_projected_control_identity_context_refuses_when_the_module_cannot_land(
    tmp_path: Path,
) -> None:
    config = {"services": {"knob": {"control_identity": True}}}
    out_dir = tmp_path / "knob"
    out_dir.mkdir()
    # A file where the directory goes: nothing can be staged into it.
    (out_dir / compose_generator.CONTROL_IDENTITY_DIRNAME).write_text("")

    with pytest.raises(DeploymentPreconditionError) as excinfo:
        compose_generator._stage_control_identity_module(config, "services/knob", str(out_dir))
    assert "knob" in excinfo.value.reason


# ---------------------------------------------------------------------------
# (4) service_audit_identities
# ---------------------------------------------------------------------------


def _two_container_app(*, deployed: tuple[str, ...] = ("knob", "knob_worker")) -> dict:
    return {
        "services": {
            "knob": {"app": "knob", "audit_identity": "knob"},
            "knob_worker": {"app": "knob", "audit_identity": "knob"},
            "dial": {"app": "dial", "audit_identity": "dial"},
        },
        "deployed_services": list(deployed),
    }


def test_two_containers_of_one_application_share_one_identity() -> None:
    assert service_audit_identities(_two_container_app()) == ["knob"]


def test_an_undeployed_or_blank_identity_is_not_provisioned() -> None:
    config = _two_container_app(deployed=("knob",))
    config["services"]["blank"] = {"audit_identity": ""}
    config["services"]["number"] = {"audit_identity": 7}
    config["deployed_services"] += ["blank", "number"]
    assert service_audit_identities(config) == ["knob"]


def test_the_bundled_rows_come_first_and_unchanged() -> None:
    config = _two_container_app()
    config["deployed_services"] = [*FIXED_SERVICE_AUDIT_IDENTITIES, *config["deployed_services"]]
    assert service_audit_identities(config) == [*FIXED_SERVICE_AUDIT_IDENTITIES.values(), "knob"]


# ---------------------------------------------------------------------------
# (5) provisioning a projected identity
# ---------------------------------------------------------------------------


def test_a_projected_identity_gets_its_spool_state_and_control_tree_directories(
    tmp_path: Path,
) -> None:
    config = {"project_root": str(tmp_path), **_two_container_app()}

    compose_generator._ensure_agent_data_structure(config)

    assert (tmp_path / SPOOL_DIR_RELPATH / "knob").is_dir()
    assert (tmp_path / STATE_DIR_RELPATH / "knob").is_dir()
    assert compose_generator.control_target_identity_dir(config, tmp_path, "knob").is_dir()


# ---------------------------------------------------------------------------
# (6) status reads the map from a projected drives_target context
# ---------------------------------------------------------------------------


def test_status_finds_the_map_in_a_projected_drives_target_context(tmp_path: Path) -> None:
    build = tmp_path / "build"
    (build / "services" / "knob").mkdir(parents=True)
    (build / "config.yml").write_text(
        yaml.safe_dump({"services": {"knob": {"drives_target": True}}})
    )
    section = {"writes": "allowed", "login_tiers": LOGIN_TIERS}
    (build / "services" / "knob" / "config.yml").write_text(
        yaml.safe_dump({"control_system": section})
    )

    assert status_display._staged_login_tiers(tmp_path) == section


def test_status_ignores_a_staged_map_in_a_context_that_does_not_drive_the_target(
    tmp_path: Path,
) -> None:
    build = tmp_path / "build"
    (build / "services" / "knob").mkdir(parents=True)
    (build / "config.yml").write_text(yaml.safe_dump({"services": {"knob": {}}}))
    (build / "services" / "knob" / "config.yml").write_text(
        yaml.safe_dump({"control_system": {"login_tiers": LOGIN_TIERS}})
    )

    assert status_display._staged_login_tiers(tmp_path) is None
