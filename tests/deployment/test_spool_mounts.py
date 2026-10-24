"""The spool and state mounts: each container binds its own two directories.

Every container that runs an OSPREY component appends its agent-record events
under ``var/spool/<identity>/`` and keeps its own state under
``var/state/<identity>/``, beside the audit directory it already binds. Each
section here pins one property of that:

* every container is mounted at, and told in ``OSPREY_SPOOL_DIR`` /
  ``OSPREY_STATE_DIR``, the same container path, derived from its identity;
* each container binds only its own directories, never the zone roots;
* the build path and the web deploy path provision those directories
  group-shared before any container starts;
* the entrypoint joins both directories' groups and leaves them out of the
  ownership hand-back;
* a roster user may not take a bridge's identity or ``operator``;
* the preflight-refused marker moved to the state directory on both its
  writing and its reading side.
"""

from __future__ import annotations

import ast
import copy
import datetime as dt
import inspect
import os
import re
import stat
import subprocess
import textwrap
from importlib import resources
from pathlib import Path

import pytest
import yaml

import osprey
from osprey.agent_record.identities import BRIDGE_IDENTITIES, OPERATOR_IDENTITY
from osprey.deployment.compose_generator import (
    _ensure_agent_data_structure,
    _inject_project_metadata,
    dispatch_worker_audit_identities,
    ensure_spool_dir,
    ensure_state_dir,
    lane_queueserver_audit_identities,
    spool_identity_dir,
    state_identity_dir,
)
from osprey.deployment.web_terminals import provision
from osprey.deployment.web_terminals.render import (
    AUTH_SIDECAR_AUDIT_IDENTITY,
    render_web_terminals,
)
from osprey_connectors.workspace import SPOOL_DIR_RELPATH, STATE_DIR_RELPATH

from .web_terminals.test_golden_render import EXAMPLE_CONFIG

ZONES = (("OSPREY_SPOOL_DIR", SPOOL_DIR_RELPATH), ("OSPREY_STATE_DIR", STATE_DIR_RELPATH))

TEMPLATES_ROOT = resources.files(osprey).joinpath("templates")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _env(service: dict) -> dict[str, str]:
    """A compose ``environment:`` block, list or mapping form, as a mapping."""
    environment = service.get("environment") or {}
    if isinstance(environment, dict):
        return {key: str(value) for key, value in environment.items()}
    return dict(item.split("=", 1) for item in environment)


def _zone_mounts(service: dict, relpath: str) -> list[str]:
    """Every bind this service holds whose source is under *relpath*."""
    return [
        mount
        for mount in service.get("volumes", [])
        if isinstance(mount, str)
        and mount.split(":", 1)[0].removeprefix("./").startswith(f"{relpath}/")
    ]


def _assert_owns_its_zones(service: dict, identity: str, container_root: str) -> None:
    """*service* binds and names exactly its own spool and state directories."""
    env = _env(service)
    for variable, relpath in ZONES:
        target = f"{container_root}/{relpath}/{identity}"
        assert _zone_mounts(service, relpath) == [f"./{relpath}/{identity}:{target}"], (
            variable,
            service.get("volumes"),
        )
        assert env[variable] == target, (variable, env.get(variable))


def _base_config(deployed: list[str]) -> dict:
    services = {
        Path(str(path)).parent.name: {}
        for path in TEMPLATES_ROOT.glob("services/*/docker-compose*.yml.j2")
    }
    return {
        "project_name": "demo",
        "project_root": "/r/demo",
        "services": services,
        "system": {"timezone": "UTC"},
        "deployment": {},
        "deployed_services": deployed,
    }


def _render(service: str, config: dict) -> dict:
    """The parsed ``services:`` of one template, through the production injection."""
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_ROOT)), autoescape=False)
    rendered = env.get_template(f"services/{service}/docker-compose.yml.j2").render(
        _inject_project_metadata(config)
    )
    return yaml.safe_load(rendered)["services"]


def _web_config(*, auth: bool) -> dict:
    config = copy.deepcopy(EXAMPLE_CONFIG)
    if auth:
        config["modules"]["web_terminals"]["auth"] = {
            "method": "password",
            "allow_insecure_http": True,
        }
    return config


def _web_services(*, auth: bool = False) -> dict:
    rendered = render_web_terminals(_web_config(auth=auth))["docker-compose.web.yml"]
    return yaml.safe_load(rendered)["services"]


# ---------------------------------------------------------------------------
# Each container's mounts
# ---------------------------------------------------------------------------


def test_every_web_terminal_binds_its_own_spool_and_state():
    services = _web_services()
    terminals = {key: svc for key, svc in services.items() if key.startswith("web-")}
    assert terminals
    for service in terminals.values():
        env = _env(service)
        identity = env["OSPREY_AUDIT_IDENTITY"]
        container_root = env["OSPREY_AUDIT_DIR"].removesuffix(f"/var/audit/{identity}")
        _assert_owns_its_zones(service, identity, container_root)


def test_the_auth_sidecar_binds_its_own_spool_and_state():
    sidecar = _web_services(auth=True)["auth"]
    _assert_owns_its_zones(sidecar, AUTH_SIDECAR_AUDIT_IDENTITY, "/app")


@pytest.mark.parametrize("worker_count", [1, 3])
def test_every_dispatch_worker_binds_its_own_spool_and_state(worker_count):
    config = _base_config(["dispatch_worker"])
    config["services"]["dispatch_worker"] = {"worker_count": worker_count}
    services = _render("dispatch_worker", config)

    identities = dispatch_worker_audit_identities(config)
    assert len(identities) == worker_count
    for identity in identities:
        _assert_owns_its_zones(services[identity], identity, "/app/demo")


def test_history_binds_its_own_spool_and_state():
    services = _render("history", _base_config([]))
    _assert_owns_its_zones(services["history"], "history", "/app/demo")


def test_bluesky_web_binds_its_own_spool_and_state():
    services = _render("bluesky_web", _base_config(["bluesky_web"]))
    (service,) = services.values()
    _assert_owns_its_zones(service, "bluesky-web", "/app/project")


@pytest.mark.parametrize("bridge", sorted(BRIDGE_IDENTITIES))
def test_every_bridge_binds_its_spool_and_state_and_carries_its_login(bridge):
    from osprey.agent_record.credentials import digest_env_name, login_file

    identity = BRIDGE_IDENTITIES[bridge]
    services = _render(bridge, _base_config([bridge]))
    (service,) = services.values()

    _assert_owns_its_zones(service, identity, "/app")
    assert service["env_file"] == [login_file(identity)]
    assert service["labels"]["osprey.agent-record.cred.digest"] == (
        f"${{{digest_env_name(bridge)}:-}}"
    )
    env = _env(service)
    assert env["OSPREY_AGENT_RECORD_URL"] == "http://agent-record:8000"
    # The bridge's own history file stays until the record replaces it.
    assert "HISTORY_PATH" in env


@pytest.mark.parametrize("bridge", sorted(BRIDGE_IDENTITIES))
def test_a_bridge_on_the_host_network_dials_the_loopback_publish(bridge):
    from osprey.agent_record.config import parse_agent_record_block
    from osprey.port_layout import resolve_port_base

    config = _base_config([bridge])
    config["services"][bridge] = {"network": "host"}
    (service,) = _render(bridge, config).values()

    port = parse_agent_record_block(config, base=resolve_port_base(config)).port
    assert _env(service)["OSPREY_AGENT_RECORD_URL"] == f"http://127.0.0.1:{port}"


def _lane_config() -> dict:
    config = _base_config(["bluesky", "bluesky_va", "bluesky_live"])
    config["services"]["bluesky_va"] = {"target": "va"}
    config["services"]["bluesky_live"] = {"target": "live"}
    return config


def test_each_lane_queueserver_gets_its_own_spool_dir():
    config = _lane_config()
    services = _render("bluesky", config)
    queueservers = {key: svc for key, svc in services.items() if key.endswith("queueserver")}

    assert set(queueservers) == set(lane_queueserver_audit_identities(config))
    assert len(queueservers) == 3
    spools = set()
    for identity, service in queueservers.items():
        _assert_owns_its_zones(service, identity, "/app/project")
        spools.add(_env(service)["OSPREY_SPOOL_DIR"])
    assert len(spools) == len(queueservers), "two queueservers share a spool"


def test_the_lane_bridge_containers_get_no_spool():
    services = _render("bluesky", _lane_config())
    for key, service in services.items():
        if key.endswith("queueserver"):
            continue
        assert not _zone_mounts(service, SPOOL_DIR_RELPATH), key


def test_no_container_binds_a_zone_root():
    """Per-identity binds only: agent-record alone binds the spool whole."""
    rendered = [
        *_web_services(auth=True).values(),
        *_render("dispatch_worker", _base_config(["dispatch_worker"])).values(),
        *_render("history", _base_config([])).values(),
        *_render("bluesky", _lane_config()).values(),
        *_render("bluesky_web", _base_config(["bluesky_web"])).values(),
        *(
            service
            for bridge in BRIDGE_IDENTITIES
            for service in _render(bridge, _base_config([bridge])).values()
        ),
    ]
    for service in rendered:
        for mount in service.get("volumes", []):
            if not isinstance(mount, str):
                continue
            source = mount.split(":", 1)[0].removeprefix("./").rstrip("/")
            assert source not in (SPOOL_DIR_RELPATH, STATE_DIR_RELPATH), mount


# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------


def _assert_group_shared(path: Path) -> None:
    assert path.is_dir(), path
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode & stat.S_ISGID, oct(mode)
    assert mode & stat.S_IWGRP, oct(mode)


def test_the_build_path_provisions_every_mounted_identity(tmp_path):
    deployed = [
        "dispatch_worker",
        "bluesky",
        "bluesky_va",
        "bluesky_web",
        *BRIDGE_IDENTITIES,
    ]
    config = {
        "project_root": str(tmp_path),
        "deployed_services": deployed,
        "services": {"dispatch_worker": {"worker_count": 2}, "bluesky_va": {"target": "va"}},
    }

    _ensure_agent_data_structure(config)

    expected = [
        "dispatch-worker-1",
        "dispatch-worker-2",
        "bluesky-web",
        "queueserver",
        "bluesky-va-queueserver",
        *BRIDGE_IDENTITIES.values(),
    ]
    for identity in expected:
        _assert_group_shared(spool_identity_dir(tmp_path, identity))
        _assert_group_shared(state_identity_dir(tmp_path, identity))


def test_the_build_path_provisions_nothing_for_an_undeployed_bridge(tmp_path):
    _ensure_agent_data_structure({"project_root": str(tmp_path), "deployed_services": []})

    for identity in BRIDGE_IDENTITIES.values():
        assert not (tmp_path / SPOOL_DIR_RELPATH / identity).exists()
        assert not (tmp_path / STATE_DIR_RELPATH / identity).exists()


def test_the_build_path_provisions_the_host_identity_spool(tmp_path):
    config = {
        "project_root": str(tmp_path),
        "deployed_services": [],
        "deployment": {"host_identity": "thorsten"},
    }

    _ensure_agent_data_structure(config)

    _assert_group_shared(spool_identity_dir(tmp_path, "thorsten"))


def test_no_host_identity_provisions_no_host_spool(tmp_path):
    _ensure_agent_data_structure({"project_root": str(tmp_path), "deployed_services": []})

    spool = tmp_path / SPOOL_DIR_RELPATH
    assert not spool.exists() or not any(p.name != ".otlp" for p in spool.iterdir())


@pytest.mark.parametrize("ensure", [ensure_spool_dir, ensure_state_dir])
def test_a_bad_identity_is_skipped_and_never_escapes_the_zone(tmp_path, ensure):
    assert ensure(tmp_path, "../escape") is None

    assert not (tmp_path / "escape").exists()
    assert not (tmp_path / "var" / "escape").exists()


@pytest.mark.parametrize("helper", [spool_identity_dir, state_identity_dir])
def test_the_identity_seam_refuses_a_path(helper, tmp_path):
    with pytest.raises(ValueError):
        helper(tmp_path, "a/b")


def test_the_web_deploy_path_provisions_both_zones_for_every_web_identity():
    """The loop that provisions the audit directory provisions the other two."""
    source = textwrap.dedent(inspect.getsource(provision.deploy_up_web_terminals))
    tree = ast.parse(source)
    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and getattr(node.iter.func, "id", None) == "_audit_identities"
    ]
    assert len(loops) == 1
    called = {
        node.func.id
        for node in ast.walk(loops[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {"ensure_audit_dir", "ensure_spool_dir", "ensure_state_dir"} <= called


def test_the_web_identities_provisioned_are_the_ones_mounted():
    config = _web_config(auth=True)
    services = yaml.safe_load(render_web_terminals(config)["docker-compose.web.yml"])["services"]
    mounted = {
        mount.split(":", 1)[0].removeprefix(f"./{SPOOL_DIR_RELPATH}/")
        for service in services.values()
        for mount in _zone_mounts(service, SPOOL_DIR_RELPATH)
    }
    assert mounted == set(provision._audit_identities(config))


# ---------------------------------------------------------------------------
# Reserved identities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [OPERATOR_IDENTITY, *sorted(BRIDGE_IDENTITIES.values())])
def test_a_roster_user_named_after_a_bridge_or_operator_is_refused_by_the_render(name):
    config = _web_config(auth=False)
    config["modules"]["web_terminals"]["users"][0] = name

    with pytest.raises(ValueError, match=re.escape(repr(name))):
        render_web_terminals(config)


@pytest.mark.parametrize("name", [OPERATOR_IDENTITY, *sorted(BRIDGE_IDENTITIES.values())])
def test_a_roster_user_named_after_a_bridge_or_operator_is_refused_by_lint(name):
    from osprey.deployment.web_terminals.lint import _check_reserved_audit_identities

    findings = _check_reserved_audit_identities([{"name": name}])

    assert [f.code for f in findings] == ["web_terminals.reserved_audit_identity"]
    assert findings[0].severity == "error"


def test_an_ordinary_name_is_still_allowed():
    from osprey.deployment.web_terminals.lint import _check_reserved_audit_identities

    assert _check_reserved_audit_identities([{"name": "operators"}]) == []
    assert _check_reserved_audit_identities([{"name": "bridge"}]) == []


# ---------------------------------------------------------------------------
# The entrypoint
# ---------------------------------------------------------------------------

ENTRYPOINT = TEMPLATES_ROOT.joinpath("project/entrypoint.sh")


def _function(text: str, name: str) -> str:
    start = text.index(f"{name}() {{")
    return text[start : text.index("\n}", start) + 2]


def test_the_entrypoint_joins_both_zone_groups():
    body = _function(ENTRYPOINT.read_text(), "join_mounted_groups")
    for variable, _ in ZONES:
        assert f'join_mounted_group {variable} "${{{variable}:-}}"' in body
        assert f'probe_mounted_dir {variable} "${{{variable}:-}}" write' in body


def test_the_hand_back_prunes_the_spool_and_state_dirs(tmp_path):
    """Run the shipped function: both directories are pruned, the rest swept."""
    root = tmp_path.resolve()
    state = root / "var"
    spool = state / "spool" / "alice"
    own_state = state / "state" / "alice"
    for directory in (spool, own_state):
        directory.mkdir(parents=True)
        (directory / "inside").write_text("x")
    elsewhere = state / "elsewhere"
    elsewhere.write_text("x")

    # A `find` that records the argv the function built instead of chowning;
    # the real `find` then walks the tree with that argv's prune expression.
    bindir = root / "bin"
    bindir.mkdir()
    find_stub = bindir / "find"
    find_stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$FIND_ARGS"\n')
    find_stub.chmod(0o755)
    script = (
        'log() { echo "$*" >&2; }\n'
        f"{_function(ENTRYPOINT.read_text(), 'hand_back_state_zone')}\n"
        "hand_back_state_zone\n"
    )
    args_file = root / "find-args"
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "STATE_DIR": str(state),
        "OSPREY_SPOOL_DIR": str(spool),
        "OSPREY_STATE_DIR": str(own_state),
        "FIND_ARGS": str(args_file),
    }
    result = subprocess.run(["sh", "-c", script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    argv = args_file.read_text().splitlines()
    joined = " ".join(argv)
    for directory in (spool, own_state):
        assert f"-path {directory} -prune -o" in joined, joined
    assert "! -user osprey" in joined

    # The outcome, not only the spelling: the real find with that prune
    # expression never visits either directory, and still visits the rest.
    head = argv[: argv.index("!")]
    visited = subprocess.run(
        ["/usr/bin/find", *head, "-print"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    assert str(elsewhere) in visited
    for directory in (spool, own_state):
        assert str(directory / "inside") not in visited


# ---------------------------------------------------------------------------
# The preflight-refused marker lives in the state directory
# ---------------------------------------------------------------------------


def test_the_marker_is_written_into_the_state_dir(tmp_path, monkeypatch):
    from osprey.cli import web_cmd

    monkeypatch.setenv("OSPREY_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("OSPREY_AUDIT_DIR", raising=False)

    web_cmd._write_preflight_marker("- the machine is unreachable")

    marker = tmp_path / web_cmd.PREFLIGHT_REFUSED_MARKER
    assert marker.read_text().splitlines()[1] == "- the machine is unreachable"


def test_an_audit_dir_alone_records_no_marker(tmp_path, monkeypatch):
    from osprey.cli import web_cmd

    monkeypatch.delenv("OSPREY_STATE_DIR", raising=False)
    monkeypatch.setenv("OSPREY_AUDIT_DIR", str(tmp_path))

    web_cmd._write_preflight_marker("- refused")

    assert not (tmp_path / web_cmd.PREFLIGHT_REFUSED_MARKER).exists()


def test_the_preflight_refused_status_still_shows(tmp_path, monkeypatch):
    from osprey.cli import web_cmd
    from osprey.deployment import status_display

    monkeypatch.setenv("OSPREY_STATE_DIR", str(state_identity_dir(tmp_path, "alice")))
    web_cmd._write_preflight_marker("- the machine is unreachable")
    monkeypatch.setattr(status_display, "_container_started_at", lambda *_: None)

    reason = status_display._preflight_refusal_reason(tmp_path, "alice", {}, None)

    assert reason == "the machine is unreachable"


def test_a_marker_left_in_the_audit_dir_is_not_read(tmp_path, monkeypatch):
    from osprey.cli import web_cmd
    from osprey.deployment import status_display

    audit = tmp_path / "var" / "audit" / "alice"
    audit.mkdir(parents=True)
    stamp = dt.datetime.now(dt.UTC).isoformat()
    (audit / web_cmd.PREFLIGHT_REFUSED_MARKER).write_text(f"{stamp}\n- old\n")
    monkeypatch.setattr(status_display, "_container_started_at", lambda *_: None)

    assert status_display._preflight_refusal_reason(tmp_path, "alice", {}, None) is None
