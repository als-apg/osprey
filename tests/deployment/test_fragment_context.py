"""What an application's ``image:`` fragment may read when it renders.

A projected ``image:`` fragment renders from exactly
:data:`~osprey.deployment.compose_generator.FRAGMENT_CONTEXT`, strictly, through
both setup paths: a name outside the list fails the render. Every other service
render, the bundled ``app_runtime`` included, keeps the lenient full config.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.apps import loader
from osprey.apps.projection import project_deployment
from osprey.deployment.compose_generator import (
    APP_RUNTIME_CONTEXT,
    FRAGMENT_CONTEXT,
    _incremental_setup_build_dir,
    setup_build_dir,
)
from osprey.deployment.errors import DeploymentPreconditionError
from osprey.deployment.web_terminals.auth_credentials import terminal_secret_var
from tests._renders import tier_render

FIXTURE = Path(__file__).parents[1] / "apps" / "fixtures" / "controls_admin_like"
PARTIALS = Path(__file__).parents[2] / "src" / "osprey" / "templates" / "services"
APP = "controls-admin-like"
KEY = "controls-admin"
TEMPLATE = f"services/{KEY}/docker-compose.yml.j2"
ENTRY_POINTS = ("full", "incremental")


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A deployment repo with the fixture assembled and its fragment landed, as a build does."""
    repo = tmp_path / "repo"
    shutil.copytree(FIXTURE, repo / "apps" / APP)
    (repo / "services").mkdir()
    for partial in ("_network_axis.j2", "_env_axis.j2"):
        shutil.copy(PARTIALS / partial, repo / "services" / partial)
    shutil.copytree(repo / "apps" / APP / "compose" / KEY, repo / "services" / KEY)
    project = repo / "build" / "demo-operator"
    project.mkdir(parents=True)
    (project / "config.yml").write_text(yaml.safe_dump({"web": {"panels": {KEY: {"url": "x"}}}}))
    (repo / "config.yml").write_text("project_name: demo\n")
    (repo / ".env").write_text("A=1\n")
    monkeypatch.chdir(repo)
    return repo


def _config(repo: Path, **axes: Any) -> dict[str, Any]:
    """The deploy config of the fixture, its service as the projection writes it."""
    projected = project_deployment(
        loader.load(repo),
        {"apps": {APP: {"services": {KEY: axes}}}, "deployment": {}},
        claimed_ports=frozenset(),
    )
    block = projected["services"][KEY]["config"]
    return {
        "project_name": "demo",
        "project_root": str(repo),
        "config_dir": str(repo),
        "build_dir": "./build",
        "deployment": {"bind_address": "10.0.0.5"},
        "system": {"timezone": "UTC"},
        "control_system": {
            "connector": {
                "epics": {"gateways": {"read_only": {"address": "gw.example", "port": 5064}}}
            }
        },
        "services": {KEY: {"path": f"./services/{KEY}", **block}},
        "deployed_services": [KEY],
        "modules": {
            "web_terminals": {
                "default_tier": "operator",
                "renders": {
                    "operator": tier_render(
                        "operator", "demo-operator", project_path="build/demo-operator"
                    )
                },
                "users": [{"name": "alice", "index": 0, "tier": "operator"}],
            }
        },
    }


def _render(entry_point: str, config: dict[str, Any], key: str = KEY) -> dict[str, Any]:
    template = f"services/{key}/docker-compose.yml.j2"
    out_dir = Path("build") / "services" / key
    if entry_point == "full":
        setup_build_dir(template, config, {}, service_key=key)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.yml").write_text("project_name: demo\n")
        _incremental_setup_build_dir(template, config, {}, str(out_dir), service_key=key)
    return yaml.safe_load((out_dir / "docker-compose.yml").read_text())


def _write_fragment(body: str) -> None:
    (Path("services") / KEY / "docker-compose.yml.j2").write_text(body)


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_the_fixture_renders_on_the_bridge_through_both_paths(repo: Path, entry_point: str) -> None:
    document = _render(entry_point, _config(repo, env=["EXTRA_SITE_VAR"]))
    service = document["services"][KEY]
    port = _config(repo)["services"][KEY]["port"]
    alice = terminal_secret_var("alice")
    suffix = alice.removeprefix("OSPREY_TERMINAL_SECRET_")
    assert service["image"] == "ghcr.io/example/controls-admin:1.0"
    assert service["container_name"] == f"demo-{KEY}"
    assert service["env_file"] == ["./.env"]
    assert service["environment"] == {
        "OSPREY_APP_NAME": APP,
        "CONTROLS_ADMIN_PORT": str(port),
        "CONTROLS_ADMIN_PUBLIC_ADDRESS": "10.0.0.5",
        "CONTROLS_ADMIN_HAS_ARCHIVER": "False",
        "TZ": "UTC",
        "EPICS_CA_ADDR_LIST": "gw.example",
        "EPICS_CA_SERVER_PORT": "5064",
        "OSPREY_TERMINAL_ACCEPT_ROSTER_SECRETS": "1",
        alice: f"${{{alice}:-}}",
        "OSPREY_TERMINAL_ROSTER_OWNERS": f"{suffix}=alice",
        "CONTROLS_ADMIN_SITE": "${CONTROLS_ADMIN_SITE}",
        "EXTRA_SITE_VAR": "${EXTRA_SITE_VAR}",
    }
    assert service["ports"] == [f"127.0.0.1:{port}:{port}"]
    assert service["volumes"] == [
        f"./build/services/{KEY}/app-config.yml:/etc/controls-admin/config.yml:ro",
        "/facility/controls/admin:/facility/admin:ro",
        f"{KEY}-data:/var/lib/controls-admin",
    ]
    assert service["networks"] == ["osprey-network"]
    assert f"{KEY}-data" in document["volumes"]


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_the_fixture_takes_host_networking_from_the_axis(repo: Path, entry_point: str) -> None:
    service = _render(entry_point, _config(repo, network="host"))["services"][KEY]
    assert service["network_mode"] == "host"
    assert "ports" not in service
    assert "networks" not in service


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_the_fixture_reads_the_single_user_alias_on_a_roster_less_deployment(
    repo: Path, entry_point: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OSPREY_AUDIT_IDENTITY", "ops")
    config = _config(repo)
    del config["modules"]["web_terminals"]
    environment = _render(entry_point, config)["services"][KEY]["environment"]
    assert environment[terminal_secret_var("ops")] == "${OSPREY_TERMINAL_SECRET}"


def test_the_fixture_reads_every_listed_name() -> None:
    source = (FIXTURE / "compose" / KEY / "docker-compose.yml.j2").read_text()
    assert [name for name in FRAGMENT_CONTEXT if name not in source] == []


def test_the_bind_address_is_always_defined(repo: Path) -> None:
    config = _config(repo)
    del config["deployment"]
    environment = _render("full", config)["services"][KEY]["environment"]
    assert environment["CONTROLS_ADMIN_PUBLIC_ADDRESS"] == "127.0.0.1"


@pytest.mark.parametrize(
    "name",
    [
        "services",
        "project_name",
        "osprey_images.worker",
        "osprey_version",
        "deployment.port_base",
        "system.other",
        "panel_grant.other",
        "service.other",
        "limits_mount",
        "osprey_guarded_run_mount",
        "osprey_control_tree_mount_source",
        "control_identity_staged",
        "bluesky_panel_roster_owners",
    ],
)
@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_a_name_outside_the_list_fails_the_render(repo: Path, entry_point: str, name: str) -> None:
    _write_fragment(
        f"services:\n  {KEY}:\n    image: demo\n    labels:\n      read: '{{{{ {name} }}}}'\n"
    )
    with pytest.raises(DeploymentPreconditionError, match=KEY) as refused:
        _render(entry_point, _config(repo))
    assert any(part in refused.value.reason for part in name.split("."))


def test_the_lenient_context_stays_for_the_bundled_runtime_and_a_plain_service(
    repo: Path,
) -> None:
    """A block with ``app`` and no ``image`` is the bundled runtime render."""
    config = _config(repo)
    del config["services"][KEY]["image"]
    config["services"]["plain"] = {"path": "./services/plain"}
    body = (
        "services:\n  {{ service_key }}:\n    image: demo\n    labels:\n"
        "      unlisted: '{{ no_such_name }}'\n      tree: '{{ project_name }}'\n"
    )
    _write_fragment(body)
    (Path("services") / "plain").mkdir()
    (Path("services") / "plain" / "docker-compose.yml.j2").write_text(body)
    for key in (KEY, "plain"):
        labels = _render("full", config, key)["services"][key]["labels"]
        assert labels == {"unlisted": "", "tree": "demo"}


def test_a_sibling_template_of_a_fragment_renders_from_the_same_list(repo: Path) -> None:
    (Path("services") / KEY / "site.yml.j2").write_text("tree: {{ project_name }}\n")
    with pytest.raises(DeploymentPreconditionError, match=KEY):
        _render("full", _config(repo))


GRANT_NAMES: tuple[str, ...] = (
    "panel_grant.secret_env_vars",
    "panel_grant.roster_owners",
    "panel_grant.secret_aliases",
)


@pytest.mark.parametrize("constant", [FRAGMENT_CONTEXT, APP_RUNTIME_CONTEXT])
def test_both_lists_state_the_three_grant_keys(constant: dict[str, str]) -> None:
    assert set(GRANT_NAMES) <= set(constant)
    assert constant["panel_grant.secret_aliases"] == (
        "a variable the panel's gate accepts, mapped to the deploy variable it reads; "
        "`{}` on every render but the panel's service"
    )


def test_the_fragment_list_is_the_contract() -> None:
    assert set(FRAGMENT_CONTEXT) == {
        "_network_axis.j2",
        "_env_axis.j2",
        "osprey_env_chain",
        "osprey_labels",
        "deployed_services",
        "deployment.bind_address",
        "dev_mode",
        "site_image_build_args",
        "service_key",
        "service",
        *GRANT_NAMES,
        "system.timezone",
        "control_system.connector.epics.gateways.read_only.address",
        "control_system.connector.epics.gateways.read_only.port",
    }
    assert "build-relative; a fragment binds it as `./build/…`" in FRAGMENT_CONTEXT["service"]


def test_the_runtime_list_names_what_the_bundled_template_owns() -> None:
    assert {
        "service_key",
        "service",
        "osprey_images.worker",
        "osprey_labels.project_name",
        "osprey_container_spool_dir",
        "osprey_spool_mount_source",
        "system.timezone",
        "_network_axis.j2",
        "_env_axis.j2",
        "control_identity_staged",
        "osprey_control_tree_mount_source",
        "osprey_container_control_tree_dir",
        "osprey_guarded_run_mount",
        "limits_mount",
        "services.<k>.state_mounts",
        "services.<k>.writes_enabled",
        "services.<k>.stop_grace_period",
        "services.<k>.ca_env",
        *GRANT_NAMES,
    } == set(APP_RUNTIME_CONTEXT)
    assert APP_RUNTIME_CONTEXT["services.<k>.writes_enabled"] == (
        "the render's `control_system.writes` tier is `allowed`"
    )
