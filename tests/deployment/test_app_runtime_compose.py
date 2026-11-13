"""Render tests for the bundled ``app_runtime`` compose template.

Every ``runtime: osprey`` application container renders through this one file.
What is load-bearing about its shape:

- it runs the PROJECT image, which carries the application's package;
- its port is published on loopback only, and under host networking the
  process itself binds loopback (the bundled host-binding declaration names the
  variable that says so);
- it reads its container's block through the per-service names ``service`` /
  ``service_key`` and the panel's grant through ``panel_grant``, and nothing
  keyed by an application or panel name;
- each declared state directory is bound and named in ``OSPREY_APP_STATE_DIRS``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.deployment.compose_generator import _inject_project_metadata, service_render_context
from osprey.deployment.errors import DeploymentPreconditionError
from osprey.deployment.host_binding import BUNDLED_HOST_BINDINGS
from tests.deployment.test_compose_generator import _packaged_compose_template

TEMPLATE = "services/app_runtime/docker-compose.yml.j2"
KEY = "knob_backend"


def _row(**overrides: Any) -> dict[str, Any]:
    """A projected ``services.<k>`` row, as ``osprey.apps.projection`` writes it."""
    row: dict[str, Any] = {
        "template": "osprey.app_runtime",
        "app": "orbit-bump",
        "command": ["python", "-m", "orbit_bump.server"],
        "port": 10950,
        "url": "http://127.0.0.1:10950",
        "health": "/health",
        "env": ["KNOB_SITE_URL"],
        "drives_target": False,
        "control_identity": False,
        "audit_identity": "app-orbit-bump",
        "config_file": f"./services/{KEY}/app-config.yml",
    }
    row.update(overrides)
    return row


def _roster_credential(suffix: str, owner: str) -> dict[str, Any]:
    """A roster user's credential: it reads its own variable, empty by default."""
    var = f"OSPREY_TERMINAL_SECRET_{suffix}"
    return {"var": var, "source": var, "owner": owner, "required": False}


def _alias_credential(suffix: str, owner: str) -> dict[str, Any]:
    """The roster-less alias: the deployment-wide secret, read with no default."""
    var = f"OSPREY_TERMINAL_SECRET_{suffix}"
    return {"var": var, "source": "OSPREY_TERMINAL_SECRET", "owner": owner, "required": True}


def _render(
    row: dict[str, Any] | None = None, *, panel_grant: dict[str, Any] | None = None
) -> tuple[str, dict[str, Any]]:
    block = row if row is not None else _row()
    config = _inject_project_metadata(
        {
            "project_name": "fixture",
            "services": {KEY: copy.deepcopy(block)},
            "deployed_services": [KEY],
            "deployment": {},
            "system": {"timezone": "UTC"},
        }
    )
    context = {
        **config,
        "service_key": KEY,
        "service": config["services"][KEY],
        "panel_grant": panel_grant or {"credentials": []},
    }
    text = _packaged_compose_template(TEMPLATE).render(**context)
    return text, yaml.safe_load(text)["services"][KEY]


def test_it_runs_the_project_image_with_the_containers_command() -> None:
    _, svc = _render()

    assert svc["image"].startswith("${OSPREY_WORKER_IMAGE:-")
    assert svc["command"] == ["python", "-m", "orbit_bump.server"]
    assert "build" not in svc


def test_it_names_the_application_and_its_config() -> None:
    _, svc = _render()

    env = svc["environment"]
    assert env["OSPREY_APP_NAME"] == "orbit-bump"
    assert env["OSPREY_APP_CONFIG"] == "/osprey/app/config.yml"
    assert f"./build/services/{KEY}/app-config.yml:/osprey/app/config.yml:ro" in svc["volumes"]


def test_the_port_is_published_on_loopback_only() -> None:
    _, svc = _render()

    assert svc["ports"] == ["127.0.0.1:10950:10950"]
    assert svc["environment"]["OSPREY_APP_PORT"] == "10950"
    assert svc["environment"]["OSPREY_APP_BIND"] == "0.0.0.0"


def test_under_host_networking_the_process_binds_loopback() -> None:
    _, svc = _render(_row(network="host"))

    bind_env = BUNDLED_HOST_BINDINGS["app_runtime"].binding.bind_env
    assert svc["network_mode"] == "host"
    assert "ports" not in svc
    assert svc["environment"][bind_env] == "127.0.0.1"


def test_declared_host_variables_pass_through() -> None:
    _, svc = _render()

    assert svc["environment"]["KNOB_SITE_URL"] == "${KNOB_SITE_URL}"


def test_one_declared_state_renders_one_mount_and_one_entry() -> None:
    _, svc = _render(_row(state="var/knob"))

    entries = svc["environment"]["OSPREY_APP_STATE_DIRS"].split(":")
    targets = [v.split(":")[1] for v in svc["volumes"] if "/var/state/" in v.split(":")[0]]
    assert entries == targets == ["/app/fixture/var/knob"]


def test_no_declared_state_renders_neither_variable_nor_mount() -> None:
    _, svc = _render()

    assert "OSPREY_APP_STATE_DIRS" not in svc["environment"]
    assert not any("/var/state/" in v for v in svc["volumes"])


def test_its_own_spool_directory_is_bound() -> None:
    _, svc = _render()

    env = svc["environment"]
    assert env["OSPREY_AUDIT_IDENTITY"] == "app-orbit-bump"
    assert env["OSPREY_SPOOL_DIR"].endswith("/var/spool/app-orbit-bump")
    assert f"./var/spool/app-orbit-bump:{env['OSPREY_SPOOL_DIR']}" in svc["volumes"]


def test_the_launch_token_is_named_with_no_default() -> None:
    _, svc = _render(_row(launch_token_env="ORBIT_BUMP_LAUNCH_TOKEN"))

    assert svc["environment"]["ORBIT_BUMP_LAUNCH_TOKEN"] == "${ORBIT_BUMP_LAUNCH_TOKEN}"


def test_the_panel_grant_lists_each_roster_secret() -> None:
    grant = {"credentials": [_roster_credential("ALICE", "alice")]}
    _, svc = _render(panel_grant=grant)

    env = svc["environment"]
    assert env["OSPREY_TERMINAL_ACCEPT_ROSTER_SECRETS"] == "1"
    assert env["OSPREY_TERMINAL_SECRET_ALICE"] == "${OSPREY_TERMINAL_SECRET_ALICE:-}"
    assert env["OSPREY_TERMINAL_ROSTER_OWNERS"] == "ALICE=alice"


def test_a_roster_less_panel_grant_reads_the_deploy_variable_with_no_default() -> None:
    """The alias holds the one secret a roster-less deployment has; empty would lock it out."""
    grant = {"credentials": [_alias_credential("OPERATOR", "operator")]}
    text, svc = _render(panel_grant=grant)

    env = svc["environment"]
    assert env["OSPREY_TERMINAL_SECRET_OPERATOR"] == "${OSPREY_TERMINAL_SECRET}"
    assert env["OSPREY_TERMINAL_ACCEPT_ROSTER_SECRETS"] == "1"
    assert env["OSPREY_TERMINAL_ROSTER_OWNERS"] == "OPERATOR=operator"
    assert text.count("OSPREY_TERMINAL_ROSTER_OWNERS:") == 1


def test_a_grant_with_both_shapes_renders_each_by_its_own_rule() -> None:
    grant = {"credentials": [_roster_credential("ALICE", "alice"), _alias_credential("BOB", "bob")]}
    text, svc = _render(panel_grant=grant)

    env = svc["environment"]
    assert env["OSPREY_TERMINAL_SECRET_ALICE"] == "${OSPREY_TERMINAL_SECRET_ALICE:-}"
    assert env["OSPREY_TERMINAL_SECRET_BOB"] == "${OSPREY_TERMINAL_SECRET}"
    assert text.count("OSPREY_TERMINAL_ROSTER_OWNERS:") == 1


def test_an_empty_grant_renders_no_roster_secret() -> None:
    _, svc = _render(panel_grant={"credentials": []})

    env = svc["environment"]
    assert "OSPREY_TERMINAL_ACCEPT_ROSTER_SECRETS" not in env
    assert "OSPREY_TERMINAL_ROSTER_OWNERS" not in env
    assert not any(name.startswith("OSPREY_TERMINAL_SECRET") for name in env)


def test_the_health_endpoint_is_probed_on_the_containers_port() -> None:
    _, svc = _render()

    assert "http://localhost:10950/health" in svc["healthcheck"]["test"][1]


def test_the_env_chain_is_not_handed_to_application_code() -> None:
    _, svc = _render()

    assert "env_file" not in svc


def test_it_renders_with_every_key_it_reads_set() -> None:
    """The closed list of keys the template reads, each present."""
    row = _row(
        network="bridge",
        state="var/knob",
        launch_token_env="ORBIT_BUMP_LAUNCH_TOKEN",
        mounts=[{"source": "/nfs/oag/lattices", "target": "/lattices"}],
        roster_grant={"panel_id": "orbit-bump", "single_user_alias": None},
    )
    assert set(row) - {"template", "url"} == {
        "app",
        "audit_identity",
        "command",
        "config_file",
        "control_identity",
        "drives_target",
        "env",
        "health",
        "launch_token_env",
        "mounts",
        "network",
        "port",
        "roster_grant",
        "state",
    }

    _, svc = _render(row)

    assert svc["environment"]["OSPREY_APP_STATE_DIRS"]


def test_the_deployments_mounts_are_bound_read_only() -> None:
    mounts = [
        {"source": "/nfs/oag/lattices", "target": "/lattices"},
        {"source": "/usr/local/oag apps/bin", "target": "/opt/oag/bin"},
    ]

    _, svc = _render(_row(mounts=mounts))

    # The long form: compose refuses a source that is not there, where the short
    # form would create it as an empty root-owned directory.
    assert [volume for volume in svc["volumes"] if isinstance(volume, dict)] == [
        {"type": "bind", "source": "/nfs/oag/lattices", "target": "/lattices", "read_only": True},
        {
            "type": "bind",
            "source": "/usr/local/oag apps/bin",
            "target": "/opt/oag/bin",
            "read_only": True,
        },
    ]
    assert svc["security_opt"] == ["no-new-privileges:true"]


def test_mounts_alone_open_the_volumes_block() -> None:
    row = {"app": "knob", "mounts": [{"source": "/nfs/oag/lattices", "target": "/lattices"}]}

    _, svc = _render(row)

    assert svc["volumes"] == [
        {"type": "bind", "source": "/nfs/oag/lattices", "target": "/lattices", "read_only": True}
    ]


@pytest.mark.parametrize(
    ("source", "words"),
    [
        ("{repo}/var/guarded_run", "the guarded-run lock directory"),
        ("{repo}/var", "the guarded-run lock directory"),
        ("{repo}", "the guarded-run lock directory"),
        ("/", "the guarded-run lock directory"),
    ],
)
def test_a_mount_exposing_a_protected_path_is_refused(
    tmp_path: Path, source: str, words: str
) -> None:
    mount = {"source": source.format(repo=tmp_path), "target": "/lattices"}
    config = {
        "project_name": "fixture",
        "project_root": str(tmp_path),
        "services": {KEY: _row(mounts=[mount])},
        "deployed_services": [KEY],
        "deployment": {},
        "system": {"timezone": "UTC"},
    }

    with pytest.raises(DeploymentPreconditionError) as excinfo:
        _inject_project_metadata(config)

    assert f"binds {mount['source']} into {KEY}, which exposes {words}" in excinfo.value.reason
    assert "apps.orbit-bump.services" in excinfo.value.remedy


def test_the_declared_state_survives_the_render_a_build_takes(tmp_path: Path) -> None:
    """Through ``service_render_context``, the context a build's render takes."""
    config = {
        "project_name": "fixture",
        "services": {KEY: _row(state="var/knob")},
        "deployed_services": [KEY],
        "deployment": {},
        "system": {"timezone": "UTC"},
    }
    context = service_render_context(config, "services/app_runtime", str(tmp_path), service_key=KEY)
    text = _packaged_compose_template(TEMPLATE).render(**_inject_project_metadata(context))

    svc = yaml.safe_load(text)["services"][KEY]

    assert svc["environment"]["OSPREY_APP_STATE_DIRS"] == "/app/fixture/var/knob"
    assert any(volume.split(":")[1] == "/app/fixture/var/knob" for volume in svc["volumes"])


@pytest.mark.slow
def test_a_bridge_application_without_a_name_server_names_the_profile_keys(
    built_demo_orbit_bump,
) -> None:
    """The demo preset with orbit bump on, pointed at a Channel Access facility with no gateway.

    The refusal stands, and its remedy names only keys ``profile.yml`` can
    spell: the gateway address, the application's projected network axis, or
    dropping the application. ``services.<key>`` itself is a projected key.
    """
    config = yaml.safe_load((built_demo_orbit_bump.repo / "build" / "config.yml").read_text())
    config["control_system"]["type"] = "epics"

    with pytest.raises(DeploymentPreconditionError) as excinfo:
        _inject_project_metadata(config)

    remedy = excinfo.value.remedy
    assert "control_system.connector.epics.gateways." in remedy
    assert "apps.orbit-bump.services.orbit-bump.network: host" in remedy
    assert "apps.orbit-bump" in remedy.replace("apps.orbit-bump.services.orbit-bump.network", "")
    assert "services.orbit-bump" not in remedy.replace("apps.orbit-bump.services.orbit-bump", "")
    assert " on orbit-bump" not in remedy


def test_it_renders_with_none_of_the_keys_it_reads() -> None:
    text = _packaged_compose_template(TEMPLATE).render(
        **_inject_project_metadata(
            {
                "project_name": "fixture",
                "services": {KEY: {}},
                "deployed_services": [KEY],
                "deployment": {},
                "system": {"timezone": "UTC"},
            }
        ),
        service_key=KEY,
        service={},
        panel_grant={},
    )

    svc = yaml.safe_load(text)["services"][KEY]
    assert set(svc["environment"]) == {"TZ"}
    assert "ports" not in svc and "volumes" not in svc and "healthcheck" not in svc
