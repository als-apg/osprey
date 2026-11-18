"""The server fleet service is deployed exactly when ``channel_directory.fleet`` is set.

The build injects it like the virtual accelerator; its compose fragment runs
the project image with one port published on loopback (or, on the host
network, bound to loopback), its staged config reaches the deployed machines
by compose name on the bridge, and the dispatch worker is told where it is.
A fleet service on the host beside a bridged worker is refused at build.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import pytest
import yaml
from jinja2 import Environment, FileSystemLoader

from osprey.cli.build_injectors import _declare_bundled_host_bindings, _inject_server_fleet
from osprey.deployment.compose_generator import (
    SERVICE_ROLE_CONTROL_IDENTITY,
    SERVICE_ROLE_DRIVES_TARGET,
    SERVICE_ROLE_READS_TARGET,
    _inject_project_metadata,
    service_audit_identities,
    service_role,
    setup_build_dir,
)
from osprey.deployment.host_binding import BUNDLED_HOST_BINDINGS
from osprey.deployment.reach import REACH_CONTRACTS, live_consumers, reach_dials, reach_errors
from osprey.deployment.web_terminals.lint import lint_web_terminals
from osprey.port_layout import default_port

FLEET_TEMPLATE = "services/server_fleet/docker-compose.yml.j2"
WORKER_TEMPLATE = "services/dispatch_worker/docker-compose.yml.j2"
PORT = 18084


# ── rendering ────────────────────────────────────────────────────────────────


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(resources.files("osprey").joinpath("templates"))),
        autoescape=False,
        keep_trailing_newline=True,
    )


def _config(*, fleet: dict | None = None, worker: dict | None = None, deployed=None) -> dict:
    services: dict[str, Any] = {"dispatch_worker": dict(worker or {})}
    if fleet is not None:
        services["server_fleet"] = {"path": "./services/server_fleet", "port": PORT, **fleet}
    if deployed is None:
        deployed = ["event_dispatcher", "dispatch_worker"]
        if fleet is not None:
            deployed.append("server_fleet")
    return {
        "project_name": "demo",
        "project_root": "/r/demo",
        "system": {"timezone": "UTC"},
        "channel_directory": {"fleet": {}} if fleet is not None else {},
        "deployed_services": deployed,
        "services": services,
    }


def _render(template: str, config: dict) -> dict:
    context = _inject_project_metadata(config)
    context["osprey_env_chain"] = ["./.env"]
    context["osprey_env_present"] = True
    context["osprey_env_shared_present"] = False
    return yaml.safe_load(_environment().get_template(template).render(**context))


def _fleet(config: dict) -> dict:
    return _render(FLEET_TEMPLATE, config)["services"]["server-fleet"]


def _worker_env(config: dict) -> dict:
    return _render(WORKER_TEMPLATE, config)["services"]["dispatch-worker-1"]["environment"]


def test_on_the_bridge_one_port_is_published_on_loopback_only() -> None:
    service = _fleet(_config(fleet={}))
    assert service["ports"] == [f"127.0.0.1:{PORT}:{PORT}"]
    assert service["networks"] == ["osprey-network"]
    assert "network_mode" not in service
    # Inside its own namespace it listens on every interface: the published
    # loopback port is the only door in.
    assert service["environment"]["SERVER_FLEET_BIND"] == "0.0.0.0"
    assert f"--port {PORT}" in service["command"][-1]
    # `$$` so compose leaves the variable for the container's shell to expand.
    assert '"$${SERVER_FLEET_BIND}"' in service["command"][-1]


def test_on_the_host_network_it_binds_loopback_and_publishes_nothing() -> None:
    rendered = _render(FLEET_TEMPLATE, _config(fleet={"network": "host"}))
    service = rendered["services"]["server-fleet"]
    assert service["network_mode"] == "host"
    assert "ports" not in service
    assert "networks" not in service
    assert "networks" not in rendered
    assert service["environment"]["SERVER_FLEET_BIND"] == "127.0.0.1"


def test_the_bind_address_is_read_from_the_declared_variable() -> None:
    service = _fleet(_config(fleet={"network": "host"}))
    assert BUNDLED_HOST_BINDINGS["server_fleet"].binding.bind_env == "SERVER_FLEET_BIND"
    assert "SERVER_FLEET_BIND" in service["command"][-1]


def test_it_runs_the_project_image_with_its_config_state_and_identity() -> None:
    service = _fleet(_config(fleet={}))
    env = service["environment"]
    assert service["image"].startswith("${OSPREY_WORKER_IMAGE:-")
    assert "python -m osprey.services.channel_directory.fleet" in service["command"][-1]
    assert env["OSPREY_AUDIT_IDENTITY"] == "server-fleet"
    assert env["SERVER_FLEET_STATE_DIR"].endswith("/server-fleet")
    assert env["OSPREY_STATE_DIR"] == env["SERVER_FLEET_STATE_DIR"]
    assert service["env_file"] == ["./.env"]
    mounts = {str(v).split(":")[1]: str(v) for v in service["volumes"]}
    assert mounts[env["OSPREY_CONFIG"]].startswith("./build/services/server_fleet/config.yml:")
    assert mounts[env["OSPREY_CONFIG"]].endswith(":ro")
    assert env["SERVER_FLEET_STATE_DIR"] in mounts
    assert f"localhost:{PORT}/health" in service["healthcheck"]["test"][-1]


def test_an_unset_port_renders_the_layout_slot() -> None:
    config = _config(fleet={})
    del config["services"]["server_fleet"]["port"]
    port = default_port("server_fleet")
    assert _fleet(config)["ports"] == [f"127.0.0.1:{port}:{port}"]


# ── the dispatch worker's URL (D4) ───────────────────────────────────────────


def test_a_bridged_worker_dials_a_bridged_service_by_compose_name() -> None:
    env = _worker_env(_config(fleet={}))
    assert env["SERVER_FLEET_URL"] == f"http://server-fleet:{PORT}"


def test_a_worker_on_the_host_gets_no_url_and_dials_loopback() -> None:
    env = _worker_env(_config(fleet={}, worker={"network": "host"}))
    assert "SERVER_FLEET_URL" not in env
    env = _worker_env(_config(fleet={"network": "host"}, worker={"network": "host"}))
    assert "SERVER_FLEET_URL" not in env


def test_without_the_fleet_the_worker_gets_no_url() -> None:
    assert "SERVER_FLEET_URL" not in _worker_env(_config())


# ── the build injects it from the predicate ─────────────────────────────────


def _project(tmp_path: Path, config: dict) -> Path:
    (tmp_path / "config.yml").write_text(yaml.safe_dump(config, sort_keys=False))
    return tmp_path


def _read(project: Path) -> dict:
    return yaml.safe_load((project / "config.yml").read_text())


def test_without_the_predicate_nothing_is_injected(tmp_path: Path) -> None:
    project = _project(tmp_path, {"project_name": "p", "channel_directory": {"facility": "x"}})
    before = (project / "config.yml").read_text()
    assert _inject_server_fleet(project) is False
    assert (project / "config.yml").read_text() == before
    assert not (project / "services" / "server_fleet").exists()


def test_with_the_predicate_the_service_is_deployed(tmp_path: Path) -> None:
    project = _project(
        tmp_path,
        {"project_name": "p", "channel_directory": {"fleet": {}}, "deployed_services": ["x"]},
    )
    assert _inject_server_fleet(project) is True
    config = _read(project)
    assert config["deployed_services"] == ["x", "server_fleet"]
    assert config["services"]["server_fleet"] == {
        "path": "./services/server_fleet",
        "port": default_port("server_fleet"),
    }
    assert (project / "services" / "server_fleet" / "docker-compose.yml.j2").is_file()
    assert (project / "services" / "_network_axis.j2").is_file()
    # Idempotent: a second build registers it once.
    assert _inject_server_fleet(project) is True
    assert _read(project)["deployed_services"] == ["x", "server_fleet"]


def test_an_authored_port_and_network_are_kept(tmp_path: Path) -> None:
    project = _project(
        tmp_path,
        {
            "project_name": "p",
            "channel_directory": {"fleet": {}},
            "services": {"server_fleet": {"port": 19999, "network": "host"}},
        },
    )
    _inject_server_fleet(project)
    block = _read(project)["services"]["server_fleet"]
    assert block["port"] == 19999
    assert block["network"] == "host"


def test_on_the_host_network_it_declares_its_bind_variable(tmp_path: Path) -> None:
    project = _project(
        tmp_path,
        {
            "project_name": "p",
            "channel_directory": {"fleet": {}},
            "services": {"server_fleet": {"network": "host"}},
        },
    )
    _inject_server_fleet(project)
    profile_dir = tmp_path / "profile"  # no services/server_fleet: not claimed
    profile_dir.mkdir()
    _declare_bundled_host_bindings(project, profile_dir, {})
    assert _read(project)["services"]["server_fleet"]["bind_env"] == "SERVER_FLEET_BIND"


# ── role, staged config, identity ───────────────────────────────────────────


def test_the_bundled_role_reads_the_target_and_drives_nothing() -> None:
    assert service_role({}, "server_fleet", SERVICE_ROLE_READS_TARGET)
    assert not service_role({}, "server_fleet", SERVICE_ROLE_DRIVES_TARGET)
    assert not service_role({}, "server_fleet", SERVICE_ROLE_CONTROL_IDENTITY)


def _stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, block: dict) -> dict:
    row = {"address": "localhost", "use_name_server": True}
    document = {
        "project_name": "p",
        "control_system": {
            "connector": {
                "virtual_accelerator": {
                    "gateways": {"read_only": dict(row), "write_access": dict(row)},
                }
            }
        },
    }
    (tmp_path / "config.yml").write_text(yaml.safe_dump(document))
    service_dir = tmp_path / "services" / "server_fleet"
    service_dir.mkdir(parents=True)
    (service_dir / "docker-compose.yml.j2").write_text("services:\n  server-fleet:\n    image: t\n")
    monkeypatch.chdir(tmp_path)
    config = {
        "project_name": "p",
        "build_dir": "./build",
        "deployed_services": ["virtual_accelerator", "server_fleet"],
        "services": {"server_fleet": block},
    }
    setup_build_dir(
        "services/server_fleet/docker-compose.yml.j2",
        config,
        {"copy_src": False},
        service_key="server_fleet",
    )
    staged = yaml.safe_load((tmp_path / "build/services/server_fleet/config.yml").read_text())
    gateways = staged["control_system"]["connector"]["virtual_accelerator"]["gateways"]
    assert "login_tiers" not in staged["control_system"]
    return {row["address"] for row in gateways.values()}


def test_on_the_bridge_its_staged_config_names_the_machines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _stage(tmp_path, monkeypatch, {}) == {"virtual-accelerator"}


def test_on_the_host_its_staged_config_keeps_loopback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _stage(tmp_path, monkeypatch, {"network": "host"}) == {"localhost"}


def test_its_identity_is_provisioned_only_when_deployed() -> None:
    assert "server-fleet" in service_audit_identities({"deployed_services": ["server_fleet"]})
    assert "server-fleet" not in service_audit_identities({"deployed_services": []})


def test_a_login_may_not_take_the_services_identity() -> None:
    config = {"modules": {"web_terminals": {"enabled": True, "users": ["alice", "server-fleet"]}}}
    codes = [
        (f.code, f.message)
        for f in lint_web_terminals(config)
        if f.code == "web_terminals.reserved_audit_identity"
    ]
    assert len(codes) == 1
    assert "'server-fleet'" in codes[0][1]


# ── reach (D4) ───────────────────────────────────────────────────────────────


def test_the_contract_names_the_web_terminals_and_the_worker() -> None:
    contract = REACH_CONTRACTS["server_fleet"]
    on = _config(fleet={})
    names = [c.name for k, c in live_consumers(on) if k.service == "server_fleet"]
    assert len(names) == 2
    assert any("web terminals" in n for n in names)
    assert any("dispatch worker" in n for n in names)
    assert [p.key for p in contract.projected] == ["services.server_fleet.port"]


def test_without_the_predicate_no_consumer_is_on() -> None:
    assert not [c for c, _ in live_consumers(_config()) if c.service == "server_fleet"]


def test_the_consumers_dial_the_published_port_on_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SERVER_FLEET_URL", raising=False)
    dials = {d for c, _, d in reach_dials(_config(fleet={})) if c.service == "server_fleet"}
    assert dials == {("127.0.0.1", PORT)}


@pytest.mark.parametrize(
    ("fleet_net", "worker_net"),
    [("bridge", "bridge"), ("bridge", "host"), ("host", "host")],
)
def test_three_network_combinations_resolve(fleet_net: str, worker_net: str) -> None:
    config = _config(fleet={"network": fleet_net}, worker={"network": worker_net})
    assert reach_errors(config) == []


def test_a_host_service_beside_a_bridged_worker_is_refused_naming_both_keys() -> None:
    errors = reach_errors(_config(fleet={"network": "host"}))
    assert len(errors) == 1
    assert "services.server_fleet.network" in errors[0]
    assert "services.dispatch_worker.network" in errors[0]


def test_a_deploying_render_without_the_service_is_refused() -> None:
    config = _config(fleet={}, deployed=["event_dispatcher", "dispatch_worker"])
    errors = reach_errors(config)
    assert errors and all("`server_fleet`" in e for e in errors)
