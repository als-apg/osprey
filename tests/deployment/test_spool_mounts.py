"""The spool and state mounts: each container binds its own two directories.

Every container that runs an OSPREY component appends its agent-record events
under ``var/spool/<identity>/`` and keeps its own state under
``var/state/<identity>/``. Each section here pins one property of that:

* every container is mounted at, and told in ``OSPREY_SPOOL_DIR`` /
  ``OSPREY_STATE_DIR``, the same container path, derived from its identity;
* each container binds only its own directories, never the zone roots;
* the identity a container is told and the directories it binds are one
  derivation, provisioned from the same rule the templates render;
* the build path and the web deploy path provision those directories
  group-shared before any container starts;
* the entrypoint joins both directories' groups and leaves them out of the
  ownership hand-back;
* a roster user may not take a service's identity, a bridge's or
  ``operator``, and must be one path segment in every auth posture;
* the preflight-refused marker moved to the state directory on both its
  writing and its reading side.
"""

from __future__ import annotations

import ast
import copy
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
    FIXED_SERVICE_AUDIT_IDENTITIES,
    _ensure_agent_data_structure,
    _inject_project_metadata,
    dispatch_worker_audit_identities,
    ensure_spool_dir,
    ensure_state_dir,
    lane_queueserver_audit_identities,
    service_audit_identities,
    spool_identity_dir,
    state_identity_dir,
)
from osprey.deployment.web_terminals import provision
from osprey.deployment.web_terminals.render import (
    AUTH_SIDECAR_AUDIT_IDENTITY,
    render_web_terminals,
)
from osprey.utils.workspace import PRE_UPGRADE_AUDIT_RELPATH, SPOOL_DIR_RELPATH, STATE_DIR_RELPATH

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
        container_root = env["OSPREY_SPOOL_DIR"].removesuffix(f"/{SPOOL_DIR_RELPATH}/{identity}")
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
    # Conversation history lives in the record: no history file is configured.
    assert "HISTORY_PATH" not in env


@pytest.mark.parametrize("bridge", sorted(BRIDGE_IDENTITIES))
def test_a_bridge_on_the_host_network_dials_the_loopback_publish(bridge):
    from osprey.agent_record.config import parse_agent_record_block
    from osprey.port_layout import resolve_port_base

    config = _base_config([bridge])
    config["services"][bridge] = {"network": "host"}
    (service,) = _render(bridge, config).values()

    port = parse_agent_record_block(config, base=resolve_port_base(config)).port
    assert _env(service)["OSPREY_AGENT_RECORD_URL"] == f"http://127.0.0.1:{port}"


# ---------------------------------------------------------------------------
# Each terminal's and worker's record login and URL
# ---------------------------------------------------------------------------


def _record_port(config: dict) -> int:
    from osprey.agent_record.config import parse_agent_record_block
    from osprey.port_layout import resolve_port_base

    return parse_agent_record_block(config, base=resolve_port_base(config)).port


def _record_logins(service: dict) -> list[str]:
    """The agent-record login files among *service*'s ``env_file`` entries."""
    from osprey.agent_record.credentials import LOGIN_FILE_PREFIX

    entries = service.get("env_file") or []
    if isinstance(entries, str):
        entries = [entries]
    return [entry for entry in entries if entry.startswith(LOGIN_FILE_PREFIX)]


def _worker_config(*, worker_count: int = 1, **settings) -> dict:
    config = _base_config(["dispatch_worker"])
    config["services"]["dispatch_worker"] = {"worker_count": worker_count, **settings}
    return config


@pytest.mark.parametrize("worker_count", [1, 3])
def test_every_dispatch_worker_carries_its_own_login_and_the_record_url(worker_count):
    from osprey.agent_record.credentials import digest_env_name, login_file, owner_files

    config = _worker_config(worker_count=worker_count)
    services = _render("dispatch_worker", config)
    owners = owner_files(config)

    for identity in dispatch_worker_audit_identities(config):
        service = services[identity]
        assert _record_logins(service) == [login_file(identity)]
        assert owners[identity] == (login_file(identity),)
        assert service["labels"]["osprey.agent-record.cred.digest"] == (
            f"${{{digest_env_name(identity)}:-}}"
        )
        assert _env(service)["OSPREY_AGENT_RECORD_URL"] == "http://agent-record:8000"


def test_shared_mode_workers_carry_the_shared_dispatch_login():
    from osprey.agent_record.credentials import login_file, owner_files
    from osprey.agent_record.identities import SHARED_DISPATCH_IDENTITY

    config = _worker_config(worker_count=2, workspace_mode="shared")
    services = _render("dispatch_worker", config)
    owners = owner_files(config)

    for identity in dispatch_worker_audit_identities(config):
        assert _record_logins(services[identity]) == [login_file(SHARED_DISPATCH_IDENTITY)]
        assert owners[identity] == (login_file(SHARED_DISPATCH_IDENTITY),)


def test_a_worker_on_the_host_network_dials_the_loopback_publish():
    config = _worker_config(network="host")
    (service,) = _render("dispatch_worker", config).values()

    assert _env(service)["OSPREY_AGENT_RECORD_URL"] == f"http://127.0.0.1:{_record_port(config)}"


def test_a_worker_with_no_env_chain_still_carries_its_login():
    from osprey.agent_record.credentials import login_file

    config = _worker_config()
    rendered = _inject_project_metadata(config)
    assert not rendered.get("osprey_env_chain")
    (service,) = _render("dispatch_worker", config).values()

    assert service["env_file"] == [login_file("dispatch-worker-1")]


def test_every_web_terminal_carries_its_own_login_and_the_loopback_url():
    from osprey.agent_record.credentials import login_file

    services = _web_services()
    terminals = {key: svc for key, svc in services.items() if key.startswith("web-")}
    assert terminals
    url = f"http://127.0.0.1:{_record_port(_web_config(auth=False))}"
    for service in terminals.values():
        assert service["network_mode"] == "host"
        env = _env(service)
        identity = env["OSPREY_AUDIT_IDENTITY"]
        assert service["env_file"][0] == ".env.users"
        assert _record_logins(service) == [login_file(identity)]
        assert env["OSPREY_AGENT_RECORD_URL"] == url


def test_the_web_render_bakes_each_terminal_login_digest():
    from osprey.agent_record.credentials import DIGEST_LABEL

    digests = {"alice": "a" * 64, "bob": "b" * 64}
    rendered = render_web_terminals(_web_config(auth=False), agent_record_login_digests=digests)[
        "docker-compose.web.yml"
    ]
    services = yaml.safe_load(rendered)["services"]
    terminals = {key: svc for key, svc in services.items() if key.startswith("web-")}
    assert terminals
    for service in terminals.values():
        identity = _env(service)["OSPREY_AUDIT_IDENTITY"]
        assert service["labels"][DIGEST_LABEL] == digests[identity]


def test_a_web_render_without_digests_bakes_no_record_label():
    from osprey.agent_record.credentials import DIGEST_LABEL

    for key, service in _web_services().items():
        assert DIGEST_LABEL not in (service.get("labels") or {}), key


@pytest.mark.parametrize("shared", [False, True])
def test_no_container_receives_another_identitys_login(shared):
    """Every record login a service is handed is its own identity's login."""
    from osprey.agent_record.credentials import login_file, owner_files

    worker_config = _worker_config(
        worker_count=3, **({"workspace_mode": "shared"} if shared else {})
    )
    owners = owner_files(worker_config)
    workers = _render("dispatch_worker", worker_config)
    for key, service in workers.items():
        assert _record_logins(service) == list(owners[key]), key

    web = _web_services(auth=True)
    for key, service in web.items():
        logins = _record_logins(service)
        if key.startswith("web-"):
            identity = _env(service)["OSPREY_AUDIT_IDENTITY"]
            assert logins == [login_file(identity)], key
        else:
            assert logins == [], key


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
    """One loop over the web identities provisions both zones."""
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
    assert {"ensure_spool_dir", "ensure_state_dir"} <= called


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

    web_cmd._write_preflight_marker("- the machine is unreachable")

    marker = tmp_path / web_cmd.PREFLIGHT_REFUSED_MARKER
    assert marker.read_text().splitlines()[1] == "- the machine is unreachable"


def test_the_preflight_refused_status_still_shows(tmp_path, monkeypatch):
    from osprey.cli import web_cmd
    from osprey.deployment import status_display

    monkeypatch.setenv("OSPREY_STATE_DIR", str(state_identity_dir(tmp_path, "alice")))
    web_cmd._write_preflight_marker("- the machine is unreachable")
    monkeypatch.setattr(status_display, "_container_started_at", lambda *_: None)

    reason = status_display._preflight_refusal_reason(tmp_path, "alice", {}, None)

    assert reason == "the machine is unreachable"


# ---------------------------------------------------------------------------
# The entrypoint's other named directories
# ---------------------------------------------------------------------------

#: The bundle the reference config mounts when a test needs the bundle half.
BUNDLE_PATH = "data/facility/knowledge"

#: The ARIEL qmd mirror the reference config mounts when a test needs that half.
MIRROR_PATH = "var/ariel_mirror"


def _corpus_web_services(*, bundle: bool = False, mirror: bool = False) -> dict:
    """The rendered web overlay, with the bundle and/or mirror the default tier is entitled to."""
    config = _web_config(auth=False)
    if bundle:
        config["facility_knowledge"] = {"bundle_path": BUNDLE_PATH}
    if mirror:
        config["ariel"] = {
            "enhancement_modules": {
                "qmd_export": {"enabled": True, "settings": {"mirror_path": MIRROR_PATH}}
            }
        }
    entitled = {"assistant"}
    rendered = render_web_terminals(
        config,
        facility_bundle_renders=entitled if bundle else None,
        ariel_mirror_renders=entitled if mirror else None,
    )["docker-compose.web.yml"]
    return yaml.safe_load(rendered)["services"]


def test_the_bundle_dir_is_named_when_a_bundle_is_mounted():
    """The entrypoint's root phase iterates exactly the directories the render
    names, so ``OSPREY_FACILITY_BUNDLE_DIR`` has to be the bundle's actual mount
    target — not a second derivation of it that could name a path nothing is
    mounted at."""
    service = _corpus_web_services(bundle=True)["web-alice"]
    bundle_targets = [
        mount.split(":")[1] for mount in service["volumes"] if BUNDLE_PATH in mount.split(":")[0]
    ]
    assert _env(service)["OSPREY_FACILITY_BUNDLE_DIR"] == bundle_targets[0]


def test_no_bundle_dir_without_a_bundle_mount():
    """A service with no bundle bound names no bundle directory, and the
    entrypoint's step skips what is unset."""
    assert "OSPREY_FACILITY_BUNDLE_DIR" not in _env(_corpus_web_services()["web-alice"])


def test_the_mirror_dir_is_named_when_a_mirror_is_mounted():
    """The exporter writing the ARIEL mirror runs as the dropped user, which
    `group_add:` alone never reaches. Named by the mount's actual target."""
    service = _corpus_web_services(mirror=True)["web-alice"]
    mirror_targets = [
        mount.split(":")[1] for mount in service["volumes"] if MIRROR_PATH in mount.split(":")[0]
    ]
    assert len(mirror_targets) == 1, service["volumes"]
    assert _env(service)["OSPREY_ARIEL_MIRROR_DIR"] == mirror_targets[0]


def test_no_mirror_dir_without_a_mirror_mount():
    """A deployment running no qmd export binds no mirror and names none."""
    assert "OSPREY_ARIEL_MIRROR_DIR" not in _env(_corpus_web_services()["web-alice"])


def test_a_worker_names_no_bundle_dir():
    """It mounts no facility bundle, so the entrypoint's group step finds
    nothing else to iterate — no special case needed there."""
    services = _render("dispatch_worker", _base_config(["dispatch_worker"]))

    assert "OSPREY_FACILITY_BUNDLE_DIR" not in _env(services["dispatch-worker-1"])


# ---------------------------------------------------------------------------
# Identities: the rendered service key is the provisioned identity
# ---------------------------------------------------------------------------


def _counted_worker_config(worker_count) -> dict:
    config = _base_config(["dispatch_worker"])
    config["services"]["dispatch_worker"] = {"worker_count": worker_count}
    return config


def test_the_sidecar_identity_is_the_fixed_name():
    """One service, not one per user: the users its login and denial events name
    are their SUBJECTS, never their writers."""
    services = _web_services(auth=True)

    assert _env(services["auth"])["OSPREY_AUDIT_IDENTITY"] == AUTH_SIDECAR_AUDIT_IDENTITY
    assert AUTH_SIDECAR_AUDIT_IDENTITY == "sidecar"


def test_every_user_is_told_its_own_identity():
    for key, service in _web_services().items():
        if key.startswith("web-"):
            assert _env(service)["OSPREY_AUDIT_IDENTITY"] == key.removeprefix("web-")


def test_a_workers_identity_is_its_compose_service_key():
    """A record traces back to the container that wrote it with no lookup table."""
    for key, service in _render("dispatch_worker", _counted_worker_config(2)).items():
        assert _env(service)["OSPREY_AUDIT_IDENTITY"] == key


def test_provisioned_worker_identities_are_exactly_the_rendered_service_keys():
    """The template names the service key while Python provisions the directory;
    a disagreement is a bind mount whose host side nobody created, which a
    rootful daemon then creates root-owned."""
    config = _counted_worker_config(4)

    assert dispatch_worker_audit_identities(config) == list(_render("dispatch_worker", config))


def test_no_worker_identities_when_the_worker_is_not_deployed():
    assert dispatch_worker_audit_identities({"services": {"dispatch_worker": {}}}) == []
    assert dispatch_worker_audit_identities({}) == []


@pytest.mark.parametrize("worker_count", [None, "", "two", 0])
def test_an_unusable_worker_count_falls_back_to_one(worker_count):
    """The template's own ``| default(1)`` for the same unset/unusable value."""
    config = {
        "services": {"dispatch_worker": {"worker_count": worker_count}},
        "deployed_services": ["dispatch_worker"],
    }

    assert dispatch_worker_audit_identities(config) == ["dispatch-worker-1"]


def test_provisioned_service_identities_are_exactly_the_rendered_service_keys():
    for service, identity in FIXED_SERVICE_AUDIT_IDENTITIES.items():
        assert service_audit_identities({"deployed_services": [service]}) == [identity]
        rendered = _render(service, _base_config([service]))
        assert list(rendered) == [identity]
        assert _env(rendered[identity])["OSPREY_AUDIT_IDENTITY"] == identity


def test_no_service_identities_when_the_service_is_not_deployed():
    assert service_audit_identities({"deployed_services": ["mongodb"]}) == []
    assert service_audit_identities({}) == []


def test_the_queueserver_identity_is_its_compose_key():
    config = _base_config(["bluesky", "bluesky_va", "bluesky_live"])
    config["services"]["bluesky_va"] = {"target": "va"}
    config["services"]["bluesky_live"] = {"target": "live"}
    queueservers = {
        key: service
        for key, service in _render("bluesky", config).items()
        if key.endswith("queueserver")
    }

    assert queueservers
    for key, service in queueservers.items():
        assert _env(service)["OSPREY_AUDIT_IDENTITY"] == key
    assert lane_queueserver_audit_identities(config) == list(queueservers)


# ---------------------------------------------------------------------------
# A roster name is never a service's identity, and is one path segment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "sidecar",
        "dispatch-worker-1",
        "dispatch-worker-12",
        *sorted(FIXED_SERVICE_AUDIT_IDENTITIES.values()),
    ],
)
@pytest.mark.parametrize("auth", [False, True])
def test_a_roster_name_that_is_a_service_identity_is_refused(name, auth):
    """The one place "your own subdirectory" must not be honoured literally: a
    user named for a service would get a read-write bind onto the spool and
    state of the component that records them. Unconditional on the auth
    method, because the name is a directory in every posture."""
    config = _web_config(auth=auth)
    config["modules"]["web_terminals"]["users"] = ["alice", name]

    with pytest.raises(ValueError, match="service's own identity") as exc_info:
        render_web_terminals(config)
    assert repr(name) in str(exc_info.value)
    assert PRE_UPGRADE_AUDIT_RELPATH not in str(exc_info.value)


def test_an_ordinary_hyphenated_name_is_still_fine():
    """The gate is narrow: only the service identities themselves."""
    config = _web_config(auth=False)
    config["modules"]["web_terminals"]["users"] = ["dispatch-worker", "sidecar-ops", "alice"]

    render_web_terminals(config)


@pytest.mark.parametrize("name", ["../../etc", "..", ".", "a/b", "Alice", "j.doe"])
def test_a_name_that_is_not_one_path_segment_is_refused_with_auth_off(name):
    """``auth.method: none`` is the default posture, and the render emits
    ``./var/spool/<name>:<container>/var/spool/<name>`` verbatim from the roster,
    so ``../../etc`` would be a read-write bind outside the repo. The
    provisioning seam declines to create such a source, but the container
    runtime would create it anyway; the refusal has to be the renderer's."""
    config = _web_config(auth=False)
    config["modules"]["web_terminals"]["users"] = ["alice", name]

    with pytest.raises(ValueError, match="spool and state subdirectories") as exc_info:
        render_web_terminals(config)
    assert repr(name) in str(exc_info.value)


def test_no_render_emits_a_zone_bind_that_leaves_its_zone():
    """The property the gate exists for, asserted on the artifact itself."""
    for auth in (False, True):
        for service in _web_services(auth=auth).values():
            for _variable, relpath in ZONES:
                for mount in _zone_mounts(service, relpath):
                    source = mount.split(":", 1)[0]
                    assert ".." not in Path(source).parts, mount
                    assert Path(source).resolve().is_relative_to(Path(relpath).resolve())


def test_the_render_and_the_provisioning_seam_agree_about_every_roster_name():
    """One charset, asked at both ends: the seam refusing an identity the
    renderer emitted anyway would leave compose to create the source."""
    for name in ("../../etc", "..", "a/b", "Alice"):
        config = _web_config(auth=False)
        config["modules"]["web_terminals"]["users"] = [name]

        for helper in (spool_identity_dir, state_identity_dir):
            with pytest.raises(ValueError):
                helper("/repo", name)
        with pytest.raises(ValueError):
            render_web_terminals(config)
