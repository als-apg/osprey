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
    FRAGMENT_BUILD_KEYS,
    FRAGMENT_CONTEXT,
    FRAGMENT_SERVICE_KEYS,
    FRAGMENT_TOP_LEVEL_KEYS,
    _incremental_setup_build_dir,
    _inject_project_metadata,
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
    """A refused render also clears the compose file an earlier render left."""
    _render(entry_point, _config(repo))
    _write_fragment(
        f"services:\n  {KEY}:\n    image: demo\n    labels:\n      read: '{{{{ {name} }}}}'\n"
    )
    with pytest.raises(DeploymentPreconditionError, match=KEY) as refused:
        _render(entry_point, _config(repo))
    assert any(part in refused.value.reason for part in name.split("."))
    assert not (Path("build") / "services" / KEY / "docker-compose.yml").exists()


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
    assert not (Path("build") / "services" / KEY / "docker-compose.yml").exists()


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
        "osprey_container_project_dir",
        "osprey_container_guarded_run_dir",
        "services.<k>.state_mounts",
        "services.<k>.writes_enabled",
        "services.<k>.stop_grace_period",
        "services.<k>.ca_env",
        *GRANT_NAMES,
    } == set(APP_RUNTIME_CONTEXT)
    assert APP_RUNTIME_CONTEXT["services.<k>.writes_enabled"] == (
        "the render's `control_system.writes` tier arms writes (`asked` or `allowed`)"
    )
    assert "`drives_target`" in APP_RUNTIME_CONTEXT["service"]


def _fragment_with(*, volumes: list[Any] | None = None, ports: list[Any] | None = None) -> str:
    """A fragment whose rendered service carries these entries, as YAML."""
    service: dict[str, Any] = {"image": "demo"}
    if volumes is not None:
        service["volumes"] = volumes
    if ports is not None:
        service["ports"] = ports
    return yaml.safe_dump({"services": {"controls-admin-web": service}})


def _bind(source: str) -> str:
    return f"{source}:/mnt/target"


@pytest.mark.parametrize(
    "volume",
    [
        pytest.param(_bind("./var/guarded_run"), id="guarded-run"),
        pytest.param(_bind("./var/guarded_run/target"), id="inside-guarded-run"),
        pytest.param(_bind("./var"), id="above-guarded-run"),
        pytest.param(_bind("."), id="the-repo"),
        pytest.param(_bind("{repo}/var/guarded_run"), id="absolute"),
        pytest.param(_bind("./x/../var/guarded_run"), id="unnormalized"),
        pytest.param(
            {"type": "bind", "source": "./var/guarded_run", "target": "/mnt/target"},
            id="long-form",
        ),
        pytest.param(
            {"type": "${KIND}", "source": "./var/guarded_run", "target": "/mnt/target"},
            id="long-form-variable-type",
        ),
        pytest.param(_bind("./var/$$x/../guarded_run"), id="escaped-dollar"),
        pytest.param(_bind("./var/$${x/../guarded_run"), id="escaped-brace-hides-no-colon"),
        pytest.param(_bind("./data/limits.json"), id="limits-file"),
        pytest.param(_bind("./data"), id="above-limits-file"),
        pytest.param(_bind(f"./build/services/{KEY}/control_identity"), id="identity-dir"),
        pytest.param(_bind("./build"), id="above-identity-build"),
        pytest.param(_bind("./build/services"), id="above-identity-services"),
        pytest.param(_bind("./build/services/holder"), id="above-identity-context"),
        pytest.param(_bind("{control_tree}"), id="control-tree"),
    ],
)
@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_a_protected_mount_refuses_naming_the_service(
    repo: Path, entry_point: str, volume: Any
) -> None:
    config = _config(repo)
    if isinstance(volume, str):
        tree = _inject_project_metadata(config)["osprey_control_tree_mount_source"]
        volume = volume.replace("{repo}", str(repo)).replace("{control_tree}", tree)
    _write_fragment(_fragment_with(volumes=[volume]))
    config["limits_mount"] = {"source": "./data/limits.json", "target": "/limits.json"}
    config["services"]["holder"] = {"path": "./services/holder", "control_identity": True}
    with pytest.raises(DeploymentPreconditionError, match="controls-admin-web") as refused:
        _render(entry_point, config)
    assert "exposes" in refused.value.reason
    assert not (Path("build") / "services" / KEY / "docker-compose.yml").exists()


@pytest.mark.parametrize(
    "volume",
    [
        pytest.param(f"{KEY}-data:/var/lib/data", id="named-volume"),
        pytest.param("/var/lib/data", id="anonymous-volume"),
        pytest.param(_bind("/facility/data"), id="facility-bind"),
        pytest.param("/facility/data:${DATA_TARGET}", id="interpolated-target"),
        pytest.param(_bind("./var/spool/identity"), id="beside-the-guarded-run"),
        pytest.param({"type": "volume", "source": "var", "target": "/mnt/target"}, id="long-named"),
        pytest.param(_bind("./data$$dir"), id="escaped-dollar"),
        pytest.param(_bind("/facility/$${dir}"), id="escaped-brace"),
        pytest.param("$$named:/mnt/target", id="escaped-dollar-names-a-volume"),
        pytest.param("$$named", id="escaped-dollar-alone"),
        pytest.param(
            {"type": "bind", "source": "./data$$dir", "target": "/mnt/target"},
            id="long-form-escaped-dollar",
        ),
    ],
)
def test_an_unprotected_mount_renders(repo: Path, volume: Any) -> None:
    _write_fragment(_fragment_with(volumes=[volume]))
    document = _render("full", _config(repo))
    assert document["services"]["controls-admin-web"]["volumes"] == [volume]


@pytest.mark.parametrize(
    "volume",
    [
        pytest.param(_bind("${FACILITY_DATA}"), id="whole-source"),
        pytest.param(_bind("${PWD}/var/guarded_run"), id="prefix"),
        pytest.param(_bind("./${ZONE}/guarded_run"), id="inside"),
        pytest.param(_bind("${DATA:-/facility/data}"), id="default"),
        pytest.param(_bind("${DATA:-${ROOT:-/facility}/data}"), id="nested-default"),
        pytest.param("${DATA}", id="no-target"),
        pytest.param("${DATA:-./var/guarded_run:/mnt/target}", id="default-holds-the-bind"),
        pytest.param("$DATA", id="unbraced-no-target"),
        pytest.param("$$$DATA", id="escape-then-variable"),
        pytest.param("${DATA:-./var/guarded_run:/mnt/target", id="unbalanced"),
        pytest.param("./var/${DATA:/mnt/target", id="unbalanced-inside"),
        pytest.param("vol:${DATA", id="unbalanced-in-the-target"),
        pytest.param(
            {"type": "bind", "source": "${PWD}/var/guarded_run", "target": "/mnt/target"},
            id="long-form",
        ),
    ],
)
@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_a_bind_source_spelled_with_a_variable_refuses(
    repo: Path, entry_point: str, volume: Any
) -> None:
    """Its value is unknown at render time, so it cannot be told from a protected path."""
    _write_fragment(_fragment_with(volumes=[volume]))
    with pytest.raises(DeploymentPreconditionError, match="controls-admin-web") as refused:
        _render(entry_point, _config(repo))
    source = volume["source"] if isinstance(volume, dict) else volume.rsplit(":", 1)[0]
    assert source in refused.value.reason
    assert not (Path("build") / "services" / KEY / "docker-compose.yml").exists()


def test_a_bundled_lane_holds_the_identity_in_its_shared_context(repo: Path) -> None:
    """Every lane renders the one bundled directory, so that is the context to protect."""
    config = _config(repo)
    config["services"]["bluesky"] = {"path": "./services/bluesky"}
    _write_fragment(_fragment_with(volumes=[_bind("./build/services/bluesky")]))
    with pytest.raises(DeploymentPreconditionError, match="identity"):
        _render("full", config)


def _document_with(**sections: Any) -> str:
    """A fragment rendering one plain service beside these top-level sections, as YAML."""
    return yaml.safe_dump({"services": {"controls-admin-web": {"image": "demo"}}, **sections})


LIMITS_CONFIG = "configs.limits.file"
LIMITS_SECRET = "secrets.limits.file"


@pytest.mark.parametrize(
    ("sections", "path"),
    [
        pytest.param({"configs": {"limits": {"file": "./data/limits.json"}}}, LIMITS_CONFIG),
        pytest.param({"secrets": {"limits": {"file": "./data/limits.json"}}}, LIMITS_SECRET),
        pytest.param({"secrets": {"limits": {"file": "./var/$$x/../guarded_run"}}}, LIMITS_SECRET),
    ],
)
@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_a_protected_path_refuses_through_a_top_level_section(
    repo: Path, entry_point: str, sections: dict[str, Any], path: str
) -> None:
    _write_fragment(_document_with(**sections))
    config = _config(repo)
    config["limits_mount"] = {"source": "./data/limits.json", "target": "/limits.json"}
    with pytest.raises(DeploymentPreconditionError, match=KEY) as refused:
        _render(entry_point, config)
    assert path in refused.value.reason
    assert "exposes" in refused.value.reason
    assert not (Path("build") / "services" / KEY / "docker-compose.yml").exists()


@pytest.mark.parametrize(
    ("sections", "path"),
    [
        pytest.param({"configs": {"limits": {"file": "${DATA}/limits"}}}, LIMITS_CONFIG),
        pytest.param({"secrets": {"limits": {"file": "${DATA}/limits"}}}, LIMITS_SECRET),
    ],
)
@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_a_top_level_source_spelled_with_a_variable_refuses(
    repo: Path, entry_point: str, sections: dict[str, Any], path: str
) -> None:
    _write_fragment(_document_with(**sections))
    with pytest.raises(DeploymentPreconditionError, match=KEY) as refused:
        _render(entry_point, _config(repo))
    assert path in refused.value.reason
    assert "unknown at render time" in refused.value.reason


@pytest.mark.parametrize(
    "volume",
    [
        pytest.param(
            {"driver_opts": {"type": "overlay", "o": "lowerdir=./var/guarded_run,upperdir=/u"}},
            id="overlay-lowerdir",
        ),
        pytest.param({"driver_opts": {"type": "none", "o": "bind"}}, id="no-device"),
        pytest.param({"driver_opts": {"device": "/facility/data"}}, id="device"),
        pytest.param({"driver_opts": {}}, id="empty-opts"),
        pytest.param({"driver": "local"}, id="driver-local"),
        pytest.param({"driver": "rexray"}, id="driver-a-plugin"),
        pytest.param({"driver": "${DRIVER}"}, id="driver-spelled-with-a-variable"),
        pytest.param({"driver": "$$local"}, id="driver-an-escaped-dollar"),
    ],
)
@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_a_volume_driver_refuses_in_any_spelling(
    repo: Path, entry_point: str, volume: dict[str, Any]
) -> None:
    """A driver and its options can split across fragments and combine only at merge."""
    _write_fragment(_document_with(volumes={"lock": volume}))
    with pytest.raises(DeploymentPreconditionError, match=KEY) as refused:
        _render(entry_point, _config(repo))
    assert f"volumes.lock.{next(iter(volume))}" in refused.value.reason
    assert not (Path("build") / "services" / KEY / "docker-compose.yml").exists()


def _service_with(**fields: Any) -> str:
    return yaml.safe_dump({"services": {"controls-admin-web": {"image": "demo", **fields}}})


@pytest.mark.parametrize(
    ("build", "path"),
    [
        pytest.param({"context": "./var/guarded_run"}, "build.context", id="context"),
        pytest.param("./var/guarded_run", "build", id="short-form"),
        pytest.param({"context": "./data"}, "build.context", id="above-the-limits-file"),
        pytest.param({"context": "."}, "build.context", id="the-repo"),
        pytest.param({"args": {"A": "1"}}, "build.context", id="no-context-is-the-repo"),
        pytest.param(
            {"context": "./x", "dockerfile": "../var/guarded_run/Dockerfile"},
            "build.dockerfile",
            id="dockerfile-resolved-against-the-context",
        ),
        pytest.param(
            {"context": "./x", "additional_contexts": {"lock": "./var/guarded_run"}},
            "build.additional_contexts",
            id="additional",
        ),
        pytest.param(
            {"context": "./x", "additional_contexts": ["lock=./var/guarded_run"]},
            "build.additional_contexts",
            id="additional-a-list",
        ),
    ],
)
@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_a_build_path_on_a_protected_path_refuses(
    repo: Path, entry_point: str, build: Any, path: str
) -> None:
    _write_fragment(_service_with(build=build))
    config = _config(repo)
    config["limits_mount"] = {"source": "./data/limits.json", "target": "/limits.json"}
    with pytest.raises(DeploymentPreconditionError, match="controls-admin-web") as refused:
        _render(entry_point, config)
    assert path in refused.value.reason
    assert "exposes" in refused.value.reason
    assert not (Path("build") / "services" / KEY / "docker-compose.yml").exists()


def test_an_absolute_dockerfile_on_a_protected_path_refuses(repo: Path) -> None:
    _write_fragment(
        _service_with(build={"context": "./x", "dockerfile": f"{repo}/var/guarded_run/Dockerfile"})
    )
    with pytest.raises(DeploymentPreconditionError, match="build.dockerfile") as refused:
        _render("full", _config(repo))
    assert "exposes" in refused.value.reason


@pytest.mark.parametrize(
    "build",
    [
        pytest.param("${CONTEXT}", id="short-form"),
        pytest.param({"context": "./${CONTEXT}"}, id="context"),
        pytest.param({"context": "./x", "additional_contexts": {"a": "${A}"}}, id="additional"),
        pytest.param({"context": "./x", "additional_contexts": ["a=${A}"]}, id="additional-list"),
        pytest.param({"context": "./x", "dockerfile": "${DOCKERFILE}"}, id="dockerfile"),
    ],
)
def test_a_build_path_spelled_with_a_variable_refuses(repo: Path, build: Any) -> None:
    _write_fragment(_service_with(build=build))
    with pytest.raises(DeploymentPreconditionError, match="unknown at render time"):
        _render("full", _config(repo))


@pytest.mark.parametrize(
    "build",
    [
        pytest.param("./x", id="short-form"),
        pytest.param({"context": "./x"}, id="context"),
        pytest.param(
            {
                "context": "./x",
                "dockerfile": "docker/Dockerfile",
                "args": {"A": "1"},
                "target": "t",
            },
            id="dockerfile-args-target",
        ),
        pytest.param(
            {
                "context": "./x",
                "additional_contexts": {"a": "./y", "base": "docker-image://example/base:1"},
            },
            id="additional",
        ),
        pytest.param(
            {"context": "./x", "additional_contexts": ["a=./y"]},
            id="additional-a-list",
        ),
    ],
)
def test_a_build_path_that_exposes_nothing_renders(repo: Path, build: Any) -> None:
    _write_fragment(_service_with(build=build))
    assert _render("full", _config(repo))["services"]["controls-admin-web"]["build"] == build


@pytest.mark.parametrize(
    "env_file",
    [
        pytest.param(f"./build/services/{KEY}/control_identity/token.env", id="identity-string"),
        pytest.param(["./.env", "./var/guarded_run/x.env"], id="guarded-run-in-a-list"),
        pytest.param([{"path": "./data/limits.json", "required": False}], id="limits-long-form"),
        pytest.param({"path": "./data/limits.json"}, id="limits-a-mapping"),
    ],
)
@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_an_env_file_on_a_protected_path_refuses(
    repo: Path, entry_point: str, env_file: Any
) -> None:
    """Compose reads the file into the environment, which exposes it as a bind would."""
    _write_fragment(_service_with(env_file=env_file))
    config = _config(repo)
    config["limits_mount"] = {"source": "./data/limits.json", "target": "/limits.json"}
    with pytest.raises(DeploymentPreconditionError, match="controls-admin-web") as refused:
        _render(entry_point, config)
    assert "env_file" in refused.value.reason
    assert "exposes" in refused.value.reason
    assert not (Path("build") / "services" / KEY / "docker-compose.yml").exists()


@pytest.mark.parametrize(
    "env_file",
    [
        pytest.param("${X}", id="string"),
        pytest.param(["./.env", "${X}"], id="in-a-list"),
        pytest.param([{"path": "${X}"}], id="long-form"),
    ],
)
def test_an_env_file_spelled_with_a_variable_refuses(repo: Path, env_file: Any) -> None:
    _write_fragment(_service_with(env_file=env_file))
    with pytest.raises(DeploymentPreconditionError, match="env_file") as refused:
        _render("full", _config(repo))
    assert "unknown at render time" in refused.value.reason


def test_an_env_file_that_exposes_nothing_renders(repo: Path) -> None:
    env_file = ["./.env", {"path": "./site.env", "required": False}]
    _write_fragment(_service_with(env_file=env_file))
    assert _render("full", _config(repo))["services"]["controls-admin-web"]["env_file"] == env_file


def test_a_bare_config_file_in_a_top_level_section_refuses_with_the_build_spelling(
    repo: Path,
) -> None:
    source = f"./services/{KEY}/app-config.yml"
    _write_fragment(_document_with(configs={"site": {"file": source}}))
    with pytest.raises(DeploymentPreconditionError, match="configs.site.file") as refused:
        _render("full", _config(repo))
    assert f"./build/services/{KEY}/app-config.yml" in refused.value.remedy


def test_top_level_sections_that_expose_no_protected_path_render(repo: Path) -> None:
    sections = {
        "volumes": {
            "plain": None,
            "named": {"name": "elsewhere"},
            "shared": {"external": True},
        },
        "configs": {"site": {"file": "./site.yml"}, "inline": {"content": "a: 1"}},
        "secrets": {"token": {"environment": "TOKEN"}, "local": {"file": "./data$$token"}},
    }
    _write_fragment(_document_with(**sections))
    document = _render("full", _config(repo))
    assert {name: document[name] for name in sections} == sections


@pytest.mark.parametrize(
    "source",
    ["./build", "./build/services", "./build/services/other", f"./build/services/{KEY}"],
)
def test_a_build_path_binds_when_it_holds_no_staged_identity(repo: Path, source: str) -> None:
    """A role-less context stages no identity, so nothing under it is exposed."""
    config = _config(repo)
    config["services"]["other"] = {"path": "./services/other"}
    _write_fragment(_fragment_with(volumes=[_bind(source)]))
    _render("full", config)


def test_the_limits_file_binds_when_the_tier_arms_nothing(repo: Path) -> None:
    """With no limits mount there is no limits path to protect."""
    _write_fragment(_fragment_with(volumes=[_bind("./data/limits.json")]))
    _render("full", _config(repo))


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_a_bare_config_file_bind_refuses_with_the_build_spelling(
    repo: Path, entry_point: str
) -> None:
    _write_fragment(_fragment_with(volumes=[_bind(f"./services/{KEY}/app-config.yml")]))
    with pytest.raises(DeploymentPreconditionError, match="controls-admin-web") as refused:
        _render(entry_point, _config(repo))
    assert f"./build/services/{KEY}/app-config.yml" in refused.value.remedy


def _port_cases(port: int) -> list[Any]:
    return [
        f"0.0.0.0:{port}:{port}",
        f"10.0.0.5:{port}:{port}",
        f"{port}:{port}",
        f"{port}",
        port,
        f"0.0.0.0:{port}:{port}/tcp",
        f"[::1]:{port}:{port}",
        f"0.0.0.0:18000:{port}",
        f"0.0.0.0:{port}:8000",
        "${ADMIN_PORT}:8000",
        "${ADMIN_PORT:-8000}:8000",
        "${ADMIN_HOST:-127.0.0.1}:8000:8000",
        "0.0.0.0:${ADMIN_PORT:-8000}:8000",
        {"target": port, "published": port},
        {"target": port, "published": str(port), "host_ip": "0.0.0.0"},
        "0.0.0.0:8080:8080",
        "10.0.0.5:19100:9100",
        "9000:9000",
        {"target": 8000, "published": 19000, "host_ip": "0.0.0.0"},
        "${ADMIN_HOST:-127.0.0.1:8000:8000",
    ]


@pytest.mark.parametrize("case", range(len(_port_cases(0))))
@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_publishing_the_projected_port_off_loopback_refuses(
    repo: Path, entry_point: str, case: int
) -> None:
    config = _config(repo)
    _write_fragment(_fragment_with(ports=[_port_cases(config["services"][KEY]["port"])[case]]))
    with pytest.raises(DeploymentPreconditionError, match="controls-admin-web") as refused:
        _render(entry_point, config)
    assert "127.0.0.1" in refused.value.remedy


def test_a_fragment_publishing_on_the_bind_address_refuses(repo: Path) -> None:
    _write_fragment(
        "services:\n  web:\n    image: demo\n    ports:\n"
        '      - "{{ deployment.bind_address }}:{{ service.port }}:{{ service.port }}"\n'
    )
    with pytest.raises(DeploymentPreconditionError, match="web"):
        _render("full", _config(repo))


def test_loopback_publishes_render(repo: Path) -> None:
    config = _config(repo)
    port = config["services"][KEY]["port"]
    entries = [
        f"127.0.0.1:{port}:{port}",
        f"127.0.0.1:${{ADMIN_PORT:-{port}}}:{port}",
        f"127.0.0.1:${{ADMIN_PORT:-${{BASE_PORT:-{port}}}}}:{port}",
        {"target": port, "published": port, "host_ip": "127.0.0.1"},
    ]
    _write_fragment(_fragment_with(ports=entries))
    assert _render("full", config)["services"]["controls-admin-web"]["ports"] == entries


def test_a_container_with_no_projected_port_still_publishes_on_loopback_only(repo: Path) -> None:
    config = _config(repo)
    config["services"][KEY].pop("port")
    _write_fragment(_fragment_with(ports=["0.0.0.0:9000:9000"]))
    with pytest.raises(DeploymentPreconditionError, match="controls-admin-web"):
        _render("full", config)
    _write_fragment(_fragment_with(ports=["127.0.0.1:9000:9000"]))
    rendered = _render("full", config)["services"]["controls-admin-web"]["ports"]
    assert rendered == ["127.0.0.1:9000:9000"]


@pytest.mark.parametrize(
    "fragment",
    [
        pytest.param(
            "services:\n  web:\n    image: demo\n    writes_enabled: true\n", id="service-key"
        ),
        pytest.param(
            "services:\n  web:\n    image: demo\n    labels:\n      writes_enabled: 'true'\n",
            id="nested",
        ),
        pytest.param("writes_enabled: true\nservices: {}\n", id="top-level"),
    ],
)
@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_a_retired_write_key_in_the_output_refuses(
    repo: Path, entry_point: str, fragment: str
) -> None:
    _write_fragment(fragment)
    with pytest.raises(DeploymentPreconditionError, match="writes_enabled") as refused:
        _render(entry_point, _config(repo))
    assert KEY in refused.value.reason


@pytest.mark.parametrize("filename", ["site.yml", "site.yaml"])
@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_a_retired_write_key_in_a_sibling_output_refuses(
    repo: Path, entry_point: str, filename: str
) -> None:
    """The key is refused in any YAML the fragment renders, not only its compose file."""
    (Path("services") / KEY / f"{filename}.j2").write_text("a:\n  - writes_enabled: true\n")
    with pytest.raises(DeploymentPreconditionError, match="writes_enabled") as refused:
        _render(entry_point, _config(repo))
    assert KEY in refused.value.reason
    out_dir = Path("build") / "services" / KEY
    assert not (out_dir / filename).exists()
    assert not (out_dir / "docker-compose.yml").exists()


def test_a_sibling_output_that_is_not_yaml_refuses(repo: Path) -> None:
    """A YAML output that does not parse cannot be judged."""
    (Path("services") / KEY / "site.yml.j2").write_text("a: [\n")
    with pytest.raises(DeploymentPreconditionError, match=KEY):
        _render("full", _config(repo))


def test_a_sibling_output_of_any_yaml_shape_or_another_format_renders(repo: Path) -> None:
    directory = Path("services") / KEY
    (directory / "list.yml.j2").write_text("- a\n---\n- b\n")
    (directory / "notes.txt.j2").write_text("writes_enabled: [\n")
    _render("full", _config(repo))
    assert (Path("build") / "services" / KEY / "notes.txt").exists()


@pytest.mark.parametrize(
    ("fragment", "named"),
    [
        pytest.param("services: foo\n", KEY, id="services-a-string"),
        pytest.param("services: [web]\n", KEY, id="services-a-list"),
        pytest.param("services:\n  web: foo\n", "web", id="service-a-string"),
        pytest.param("services:\n  web:\n", "web", id="service-null"),
        pytest.param("services:\n  web:\n    volumes: 5\n", "web", id="volumes-not-a-list"),
        pytest.param("services:\n  web:\n    ports: 8000\n", "web", id="ports-not-a-list"),
        pytest.param("services:\n  web:\n    ports: '0:1:2'\n", "web", id="ports-a-string"),
        pytest.param("services:\n  web:\n    build: 5\n", "web", id="build-a-number"),
        pytest.param("services:\n  web:\n    build: [./x]\n", "web", id="build-a-list"),
        pytest.param(
            "services:\n  web:\n    build:\n      additional_contexts: ./x\n",
            "web",
            id="additional-contexts-a-string",
        ),
        pytest.param("services:\n  web:\n    env_file: 5\n", "web", id="env-file-a-number"),
        pytest.param(
            "services:\n  web:\n    env_file:\n      - required: false\n",
            "web",
            id="env-file-no-path",
        ),
        pytest.param("volumes: foo\n", "volumes", id="volumes-a-string"),
        pytest.param("configs: [a]\n", "configs", id="configs-a-list"),
        pytest.param("secrets: 5\n", "secrets", id="secrets-a-number"),
        pytest.param("volumes:\n  data: foo\n", "volumes.data", id="volume-a-string"),
        pytest.param("configs:\n  site: foo\n", "configs.site", id="config-a-string"),
    ],
)
@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_a_malformed_document_refuses_naming_the_service(
    repo: Path, entry_point: str, fragment: str, named: str
) -> None:
    _write_fragment(fragment)
    with pytest.raises(DeploymentPreconditionError, match=named):
        _render(entry_point, _config(repo))
    assert not (Path("build") / "services" / KEY / "docker-compose.yml").exists()


@pytest.mark.parametrize(
    ("document", "named"),
    [
        pytest.param(
            {"services": {"controls-admin-web": {"image": "demo", "volumes_from": ["worker"]}}},
            "volumes_from",
            id="volumes-from",
        ),
        pytest.param(
            {"services": {"controls-admin-web": {"image": "demo", "extends": "worker"}}},
            "extends",
            id="extends",
        ),
        pytest.param(
            {
                "services": {
                    "controls-admin-web": {
                        "image": "demo",
                        "extends": {"file": "../worker/docker-compose.yml", "service": "worker"},
                    }
                }
            },
            "extends",
            id="extends-a-file",
        ),
        pytest.param(
            {
                "services": {"controls-admin-web": {"image": "demo"}},
                "include": ["../worker/docker-compose.yml"],
            },
            "include",
            id="include",
        ),
    ],
)
@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_a_mount_the_document_does_not_state_refuses(
    repo: Path, entry_point: str, document: dict[str, Any], named: str
) -> None:
    """Compose would pull mounts in from elsewhere, where the rules cannot read them."""
    _write_fragment(yaml.safe_dump(document))
    with pytest.raises(DeploymentPreconditionError, match=KEY) as refused:
        _render(entry_point, _config(repo))
    assert named in refused.value.reason
    assert not (Path("build") / "services" / KEY / "docker-compose.yml").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("privileged", True, id="privileged"),
        pytest.param("privileged", False, id="privileged-false"),
        pytest.param("devices", ["/dev/mem:/dev/mem"], id="devices"),
        pytest.param("cap_add", ["SYS_ADMIN"], id="cap-add"),
        pytest.param("security_opt", ["no-new-privileges:true"], id="security-opt"),
        pytest.param("pid", "host", id="pid-host"),
        pytest.param("ipc", "host", id="ipc-host"),
        pytest.param("volumes_from", [], id="volumes-from-empty"),
        pytest.param("label_file", "./var/guarded_run/labels", id="label-file"),
        pytest.param("develop", {"watch": [{"path": ".", "action": "sync"}]}, id="develop"),
        pytest.param("no_such_key", 1, id="unknown"),
    ],
)
@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_a_service_key_outside_the_schema_refuses_by_name(
    repo: Path, entry_point: str, field: str, value: Any
) -> None:
    _write_fragment(_service_with(**{field: value}))
    with pytest.raises(DeploymentPreconditionError, match="controls-admin-web") as refused:
        _render(entry_point, _config(repo))
    assert f"{field}:" in refused.value.reason
    assert KEY in refused.value.reason
    assert not (Path("build") / "services" / KEY / "docker-compose.yml").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("ssh", ["default"], id="ssh"),
        pytest.param("secrets", ["token"], id="secrets"),
        pytest.param("cache_from", ["type=local,src=./var/guarded_run"], id="cache-from"),
        pytest.param("cache_to", ["type=local,dest=./var/guarded_run"], id="cache-to"),
        pytest.param("dockerfile_inline", "FROM scratch", id="dockerfile-inline"),
        pytest.param("no_such_key", 1, id="unknown"),
    ],
)
@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_a_build_key_outside_the_schema_refuses_by_name(
    repo: Path, entry_point: str, field: str, value: Any
) -> None:
    _write_fragment(_service_with(build={"context": "./x", field: value}))
    with pytest.raises(DeploymentPreconditionError, match="controls-admin-web") as refused:
        _render(entry_point, _config(repo))
    assert f"build.{field}:" in refused.value.reason
    assert not (Path("build") / "services" / KEY / "docker-compose.yml").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("include", [], id="include-empty"),
        pytest.param("name", "other", id="name"),
        pytest.param("version", "3.9", id="version"),
        pytest.param("x-shared", {"a": 1}, id="extension"),
    ],
)
@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_a_top_level_key_outside_the_schema_refuses_by_name(
    repo: Path, entry_point: str, field: str, value: Any
) -> None:
    _write_fragment(_document_with(**{field: value}))
    with pytest.raises(DeploymentPreconditionError, match=KEY) as refused:
        _render(entry_point, _config(repo))
    assert f"{field}:" in refused.value.reason
    assert not (Path("build") / "services" / KEY / "docker-compose.yml").exists()


def test_the_plain_keys_of_an_application_container_render(repo: Path) -> None:
    fields = {
        "container_name": "web",
        "command": ["serve"],
        "entrypoint": ["/entry"],
        "working_dir": "/app",
        "user": "1000:1000",
        "labels": {"a": "b"},
        "restart": "unless-stopped",
        "depends_on": ["db"],
        "healthcheck": {"test": ["CMD", "true"]},
        "stop_grace_period": "30s",
        "networks": ["osprey-network"],
    }
    _write_fragment(_service_with(**fields))
    service = _render("full", _config(repo))["services"]["controls-admin-web"]
    assert {field: service[field] for field in fields} == fields


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_host_networking_spelled_by_a_fragment_refuses_with_no_axis(
    repo: Path, entry_point: str
) -> None:
    _write_fragment(_service_with(network_mode="host"))
    with pytest.raises(DeploymentPreconditionError, match="controls-admin-web") as refused:
        _render(entry_point, _config(repo))
    assert "network_mode" in refused.value.reason
    assert not (Path("build") / "services" / KEY / "docker-compose.yml").exists()


@pytest.mark.parametrize("mode", ["bridge", "none", "service:db", "container:db", "${MODE}", None])
@pytest.mark.parametrize("axis", ["host", None])
def test_a_network_mode_but_the_axis_host_refuses(repo: Path, mode: str | None, axis: str) -> None:
    """Whatever the axis says: a namespace shared with another container is host-reach too."""
    _write_fragment(_service_with(network_mode=mode))
    with pytest.raises(DeploymentPreconditionError, match="network_mode"):
        _render("full", _config(repo, **({"network": axis} if axis else {})))


def test_host_networking_the_axis_asked_for_renders_when_spelled(repo: Path) -> None:
    _write_fragment(_service_with(network_mode="host"))
    service = _render("full", _config(repo, network="host"))["services"]["controls-admin-web"]
    assert service["network_mode"] == "host"


@pytest.mark.parametrize(
    "network",
    [
        pytest.param({"name": "host", "external": True}, id="name"),
        pytest.param({"external": {"name": "host"}}, id="external-name"),
        pytest.param({"driver": "host"}, id="driver"),
        pytest.param({"name": "${NET}", "external": True}, id="name-spelled-with-a-variable"),
        pytest.param({"driver": "${DRIVER}"}, id="driver-spelled-with-a-variable"),
    ],
)
@pytest.mark.parametrize("axis", ["host", None])
def test_a_top_level_network_on_the_host_refuses(
    repo: Path, network: dict[str, Any], axis: str | None
) -> None:
    """A service that joins it shares the host's network, with no network_mode to read."""
    _write_fragment(_document_with(networks={"lan": network}))
    with pytest.raises(DeploymentPreconditionError, match="networks.lan"):
        _render("full", _config(repo, **({"network": axis} if axis else {})))


def test_a_top_level_network_off_the_host_renders(repo: Path) -> None:
    networks = {
        "plain": None,
        "named": {"name": "elsewhere"},
        "shared": {"external": True},
        "bridge": {"driver": "bridge"},
    }
    _write_fragment(_document_with(networks=networks))
    assert _render("full", _config(repo))["networks"] == networks


@pytest.mark.parametrize(
    ("fragment", "named"),
    [
        pytest.param("networks: foo\n", "networks", id="networks-a-string"),
        pytest.param("networks:\n  lan: foo\n", "networks.lan", id="network-a-string"),
    ],
)
def test_a_malformed_network_refuses_naming_the_section(
    repo: Path, fragment: str, named: str
) -> None:
    _write_fragment(fragment)
    with pytest.raises(DeploymentPreconditionError, match=named):
        _render("full", _config(repo))


def test_the_schema_is_the_contract() -> None:
    assert set(FRAGMENT_TOP_LEVEL_KEYS) == {"services", "volumes", "networks", "configs", "secrets"}
    assert set(FRAGMENT_SERVICE_KEYS) == {
        "image",
        "container_name",
        "build",
        "env_file",
        "environment",
        "ports",
        "volumes",
        "networks",
        "network_mode",
        "command",
        "restart",
        "stop_grace_period",
        "healthcheck",
        "entrypoint",
        "working_dir",
        "user",
        "labels",
        "depends_on",
    }
    assert set(FRAGMENT_BUILD_KEYS) == {
        "context",
        "dockerfile",
        "args",
        "target",
        "additional_contexts",
    }


def test_the_rules_leave_a_bundled_render_alone(repo: Path) -> None:
    """The rules are about a projected ``image:`` fragment."""
    config = _config(repo)
    del config["services"][KEY]["image"]
    port = config["services"][KEY]["port"]
    _write_fragment(_fragment_with(volumes=[_bind("./var/guarded_run")], ports=[f"{port}:{port}"]))
    _render("full", config)
