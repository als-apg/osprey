"""The trajectory service's deploy lifecycle: its secrets, its image, its directories.

A deployment that selects the trajectory service and nothing else has to come up
with both of the credentials its doors check, run the project image ``osprey up``
builds, and find its audit subdirectory and its state directory provisioned on
the host before the container binds them.
"""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

import pytest
import yaml
from jinja2 import Environment, FileSystemLoader

from osprey.cli.templates.scaffolding import service_token_var_entries
from osprey.deployment import container_lifecycle
from osprey.deployment.compose_generator import (
    FIXED_SERVICE_AUDIT_IDENTITIES,
    _ensure_agent_data_structure,
    _inject_project_metadata,
    audit_identity_dir,
    ensure_trajectory_state_dir,
    service_audit_identities,
    trajectory_state_relpath,
)
from osprey.utils.dotenv import parse_dotenv_file

_TEMPLATES_ROOT = Path(__file__).resolve().parents[2] / "src" / "osprey" / "templates"
_TEMPLATE = "services/trajectory/docker-compose.yml.j2"

_SECRETS = ("TRAJECTORY_LAUNCH_TOKEN", "OSPREY_TERMINAL_SECRET")


# ── the secrets ──────────────────────────────────────────────────────────────


@pytest.fixture
def _clean_secret_env(monkeypatch):
    for var in (*_SECRETS, "ARIEL_DSN"):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.usefixtures("_clean_secret_env")
def test_a_deployment_with_only_trajectory_mints_both_secrets(tmp_path):
    env_path = tmp_path / ".env"

    container_lifecycle._ensure_service_tokens(
        {"deployed_services": ["trajectory"]}, False, env_path
    )

    env = parse_dotenv_file(env_path)
    for var in _SECRETS:
        assert env.get(var), f"{var} was not minted for a trajectory-only deployment"
        assert len(env[var]) >= 32, f"{var} is implausibly short"
    assert env["TRAJECTORY_LAUNCH_TOKEN"] != env["OSPREY_TERMINAL_SECRET"]


@pytest.mark.usefixtures("_clean_secret_env")
def test_an_operator_supplied_secret_is_kept(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("TRAJECTORY_LAUNCH_TOKEN=operator-chose-this-one-xxxxxxxxxxxx\n")

    container_lifecycle._ensure_service_tokens(
        {"deployed_services": ["trajectory"]}, False, env_path
    )

    env = parse_dotenv_file(env_path)
    assert env["TRAJECTORY_LAUNCH_TOKEN"] == "operator-chose-this-one-xxxxxxxxxxxx"
    assert env.get("OSPREY_TERMINAL_SECRET")


@pytest.mark.usefixtures("_clean_secret_env")
def test_no_launch_token_without_the_service(tmp_path):
    env_path = tmp_path / ".env"

    container_lifecycle._ensure_service_tokens({"deployed_services": ["mongodb"]}, False, env_path)

    env = parse_dotenv_file(env_path) if env_path.is_file() else {}
    assert "TRAJECTORY_LAUNCH_TOKEN" not in env


def test_the_env_example_documents_the_launch_token():
    entries = {entry["var"]: entry for entry in service_token_var_entries()}

    assert "trajectory" in entries["TRAJECTORY_LAUNCH_TOKEN"]["services"]
    assert entries["TRAJECTORY_LAUNCH_TOKEN"]["note"]
    assert "trajectory" in entries["OSPREY_TERMINAL_SECRET"]["services"]
    assert "trajectory" in entries["OSPREY_TERMINAL_SECRET"]["note"]


# ── the image ────────────────────────────────────────────────────────────────


def test_the_service_runs_the_project_image(monkeypatch):
    """No ``build:`` block: the project-image build is the only thing that makes
    the image it runs, so a trajectory-only deployment must still build it."""
    monkeypatch.setattr(container_lifecycle, "_resolve_prebuilt_images", lambda config: False)
    config = {"project_name": "proj", "deployed_services": ["trajectory"]}

    assert "trajectory" in container_lifecycle._PROJECT_IMAGE_SERVICES
    assert container_lifecycle._project_image_build_target(config, {}) is not None


# ── the audit identity ───────────────────────────────────────────────────────


def _render(config: dict[str, Any]) -> dict[str, Any]:
    context = _inject_project_metadata(config)
    environment = Environment(loader=FileSystemLoader(str(_TEMPLATES_ROOT)), autoescape=False)
    rendered = yaml.safe_load(environment.get_template(_TEMPLATE).render(**context))
    return rendered["services"]["trajectory"]


def _config(**extra: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "project_name": "proj",
        "project_root": "/deploy/proj",
        "deployment": {},
        "system": {"timezone": "UTC"},
        "services": {"trajectory": {"network": "host"}},
        "deployed_services": ["trajectory"],
    }
    config.update(extra)
    return config


def test_the_audit_identity_is_fixed():
    assert FIXED_SERVICE_AUDIT_IDENTITIES["trajectory"] == "trajectory"
    assert service_audit_identities({"deployed_services": ["trajectory"]}) == ["trajectory"]
    assert service_audit_identities({"deployed_services": []}) == []


def test_the_render_names_the_identity_and_binds_its_subdirectory():
    service = _render(_config())

    assert service["environment"]["OSPREY_AUDIT_IDENTITY"] == "trajectory"
    audit_dir = service["environment"]["OSPREY_AUDIT_DIR"]
    assert audit_dir == "/app/proj/var/audit/trajectory"
    assert f"./var/audit/trajectory:{audit_dir}" in service["volumes"]


# ── provisioning ─────────────────────────────────────────────────────────────


def test_the_build_path_provisions_the_audit_and_state_directories(tmp_path):
    _ensure_agent_data_structure(
        {"project_root": str(tmp_path), "deployed_services": ["trajectory"]}
    )

    for target in (audit_identity_dir(tmp_path, "trajectory"), tmp_path / "var" / "trajectory"):
        assert target.is_dir(), target
        assert stat.S_IMODE(target.stat().st_mode) & stat.S_ISGID, target


def test_the_state_directory_is_the_one_the_render_binds(tmp_path):
    ensure_trajectory_state_dir(tmp_path, {"deployed_services": ["trajectory"]})

    source = _render(_config())["volumes"]
    assert f"./{trajectory_state_relpath()}:/app/proj/{trajectory_state_relpath()}" in source
    assert (tmp_path / trajectory_state_relpath()).is_dir()


def test_nothing_is_provisioned_without_the_service(tmp_path):
    _ensure_agent_data_structure({"project_root": str(tmp_path), "deployed_services": []})

    assert not (tmp_path / "var" / "trajectory").exists()
    assert not audit_identity_dir(tmp_path, "trajectory").exists()
    assert ensure_trajectory_state_dir(tmp_path, {"deployed_services": ["mongodb"]}) is None
    assert not (tmp_path / "var" / "trajectory").exists()


def test_an_existing_state_directory_keeps_its_contents(tmp_path):
    state = tmp_path / "var" / "trajectory"
    state.mkdir(parents=True)
    (state / "session.json").write_text("{}")

    ensure_trajectory_state_dir(tmp_path, {"deployed_services": ["trajectory"]})

    assert (state / "session.json").read_text() == "{}"
    assert stat.S_IMODE(state.stat().st_mode) & stat.S_ISGID
