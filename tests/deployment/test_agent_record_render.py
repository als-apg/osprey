"""Render tests for the bundled ``agent_record`` compose template.

What is load-bearing about its shape:

**Every deployment renders it.** It is a derived service, never a
``deployed_services`` entry, so a profile with no store and no services at all
still renders it, and it is the only thing such a profile adds.

**It reads transcripts and nothing else of another container's.** Each terminal
and dispatch transcripts volume is mounted whole and read-only as the transcript
root the record names it by; no claude-config volume, no workspace, no
credential file and no other container's env file is mounted. Nothing of the
host's is: the host build's project directory, which Claude Code writes for the
host account alone, reaches the record through the host session's forwarder,
and agent-record runs as the image's user. A build writes nothing into the
host's Claude Code state.

**It declares the terminal volumes as their owner does.** The web-terminal
template renders into the other slice, so the same bare keys and labels are
re-declared here; a dispatch volume is its owner's alone in this slice.

**The ClickHouse backend adds a store no one else can read.** With
``agent_record.backend: clickhouse`` the template adds ``agent_record_clickhouse``
with a memory limit and a health check, no host port, its own data volume and the admin
file as its only env file; agent-record dials it over the bridge and never gets
the admin file. The embedded backend renders no store at all.
"""

from __future__ import annotations

import copy
import os
import stat
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.agent_record import credentials as creds
from osprey.agent_record.identities import transcript_roots
from osprey.agent_record.service import (
    ARCHIVE_DIR_ENV,
    CLICKHOUSE_URL_ENV,
    DEFAULT_TRANSCRIPTS_DIR,
    RECORD_DIR_ENV,
    SPOOL_ROOT_ENV,
    TRANSCRIPTS_DIR_ENV,
)
from osprey.agent_runner.project_paths import CLAUDE_CONFIG_DIR_ENV, claude_project_dir
from osprey.deployment.compose_generator import (
    AGENT_RECORD_CLICKHOUSE_VOLUME,
    AGENT_RECORD_TEMPLATE_DIR,
    _inject_project_metadata,
    agent_record_host_projects_dir,
    derived_compose_dirs,
    ensure_agent_record_dirs,
)
from osprey.deployment.web_terminals.render import render_web_terminals
from osprey.port_layout import default_port
from tests.deployment.test_compose_generator import _packaged_compose_template
from tests.deployment.web_terminals.test_golden_render import EXAMPLE_CONFIG

TEMPLATE = "services/agent_record/docker-compose.yml.j2"
SERVICE_KEY = "agent-record"
STORE_SERVICES = {"agent_record_clickhouse"}


def _config(
    tmp_path: Path,
    *,
    users: tuple[str, ...] = (),
    deployed: tuple[str, ...] = (),
    services: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "project_name": "dls",
        "project_root": str(tmp_path),
        "system": {"timezone": "UTC"},
        "deployment": {},
        "control_system": {},
        "services": dict(services or {}),
        "deployed_services": list(deployed),
    }
    if users:
        web = copy.deepcopy(EXAMPLE_CONFIG["modules"]["web_terminals"])
        web["users"] = list(users)
        config["modules"] = {"web_terminals": web}
        config["registry"] = copy.deepcopy(EXAMPLE_CONFIG["registry"])
        config["deploy"] = copy.deepcopy(EXAMPLE_CONFIG["deploy"])
    config.update(extra)
    return config


def _render(config: dict[str, Any], template: str = TEMPLATE) -> dict[str, Any]:
    context = _inject_project_metadata(copy.deepcopy(config))
    return yaml.safe_load(_packaged_compose_template(template).render(**context))


def _service(config: dict[str, Any]) -> dict[str, Any]:
    doc = _render(config)
    expected = [SERVICE_KEY]
    if (config.get("agent_record") or {}).get("backend") == "clickhouse":
        expected.append(creds.CLICKHOUSE_SERVICE)
    assert list(doc["services"]) == expected, doc["services"]
    return doc["services"][SERVICE_KEY]


def _string_mounts(service: dict[str, Any]) -> list[str]:
    return [mount for mount in service["volumes"] if isinstance(mount, str)]


def _transcript_mounts(service: dict[str, Any]) -> dict[str, str]:
    """Root name -> mounted volume, for every read-only transcripts mount."""
    mounts = {}
    for mount in _string_mounts(service):
        source, target, *mode = mount.split(":")
        if target.startswith(f"{DEFAULT_TRANSCRIPTS_DIR}/"):
            assert mode == ["ro"], mount
            mounts[target.removeprefix(f"{DEFAULT_TRANSCRIPTS_DIR}/")] = source
    return mounts


@pytest.fixture(autouse=True)
def _isolated_claude_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every host-projects lookup lands under the test's own tree."""
    claude = tmp_path / "claude-home"
    monkeypatch.setenv(CLAUDE_CONFIG_DIR_ENV, str(claude))
    return claude


# ── Every deployment renders it ─────────────────────────────────────────────


def test_a_store_less_profile_renders_agent_record_and_no_store(tmp_path: Path) -> None:
    config = _config(tmp_path)

    dirs = derived_compose_dirs(config)

    assert dirs == [AGENT_RECORD_TEMPLATE_DIR]
    assert not STORE_SERVICES & set(dirs)
    service = _service(config)
    assert service["container_name"] == "dls-agent-record"


@pytest.mark.parametrize(
    ("users", "deployed"),
    [
        # hello-world: nothing deployed, no web terminals.
        ((), ()),
        # control-assistant: web terminals and backend services.
        (("alice", "bob"), ("dispatch_worker", "event_dispatcher", "archive")),
    ],
    ids=["hello-world", "control-assistant"],
)
def test_agent_record_is_the_only_derived_render_and_names_only_itself(
    tmp_path: Path, users: tuple[str, ...], deployed: tuple[str, ...]
) -> None:
    """Without the history page, the derived renders are agent-record alone, and
    its template names no service but its own.

    That the other services' renders are byte-unchanged is pinned by the golden
    suites (``tests/templates/test_render_defaults_golden.py``,
    ``test_render_axis_shapes.py`` and the web-terminal golden), not here."""
    config = _config(tmp_path, users=users, deployed=deployed)

    assert derived_compose_dirs(config) == [AGENT_RECORD_TEMPLATE_DIR]
    assert list(_render(config)["services"]) == [SERVICE_KEY]


# ── The service shape ───────────────────────────────────────────────────────


def test_the_service_serves_on_the_project_image_on_loopback(tmp_path: Path) -> None:
    service = _service(_config(tmp_path))

    assert service["image"] == "${OSPREY_WORKER_IMAGE:-dls:local}"
    assert service["command"] == [
        "osprey",
        "agent-record",
        "serve",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    assert service["ports"] == [f"127.0.0.1:{default_port('agent_record')}:8000"]
    assert service["networks"] == ["osprey-network"]
    assert "build" not in service
    assert "/health" in " ".join(service["healthcheck"]["test"])
    assert "python" in " ".join(service["healthcheck"]["test"])


def test_the_port_follows_the_configured_agent_record_port(tmp_path: Path) -> None:
    service = _service(_config(tmp_path, agent_record={"port": 19053}))

    assert service["ports"] == ["127.0.0.1:19053:8000"]


def test_the_digest_label_is_the_credential_owners(tmp_path: Path) -> None:
    service = _service(_config(tmp_path))

    variable = creds.digest_env_name(creds.RECORD_SERVICE)
    assert service["labels"] == {creds.DIGEST_LABEL: f"${{{variable}:-}}"}


@pytest.mark.parametrize("backend", ["embedded", "clickhouse"])
def test_the_spool_record_dir_and_archive_are_the_writable_binds_named_in_the_env(
    tmp_path: Path, backend: str
) -> None:
    """The archive is bound at the project-relative path the host uses, so
    the record resolves the same tree in the container as on the host."""
    service = _service(_config(tmp_path, agent_record={"backend": backend}))
    env = service["environment"]

    assert env[SPOOL_ROOT_ENV] == "/app/dls/var/spool"
    assert env[RECORD_DIR_ENV] == "/app/dls/var/agent_record"
    assert env[ARCHIVE_DIR_ENV] == "/app/dls/var/archive"
    assert env[TRANSCRIPTS_DIR_ENV] == DEFAULT_TRANSCRIPTS_DIR
    writable = [m for m in _string_mounts(service) if not m.endswith(":ro")]
    assert writable == [
        f"./var/spool:{env[SPOOL_ROOT_ENV]}",
        f"./var/agent_record:{env[RECORD_DIR_ENV]}",
        f"./var/archive:{env[ARCHIVE_DIR_ENV]}",
    ]


def test_the_otlp_spool_is_mounted_only_inside_the_spool(tmp_path: Path) -> None:
    service = _service(_config(tmp_path, users=("alice",)))

    assert not [m for m in _string_mounts(service) if ".otlp" in m]


# ── Credentials ─────────────────────────────────────────────────────────────


def test_the_verifier_and_secret_set_arrive_by_env_file_alone(tmp_path: Path) -> None:
    config = _config(tmp_path, users=("alice",), deployed=("dispatch_worker",))
    service = _service(config)

    assert service["env_file"] == [creds.SERVICE_FILE]
    creds.ensure_agent_record_credentials(tmp_path, config, reader_keys=False)
    text = (tmp_path / creds.SERVICE_FILE).read_text(encoding="utf-8")
    assert f"{creds.VERIFIER_ENV}=" in text
    assert f"{creds.SECRETS_ENV}=" in text
    assert not [m for m in _string_mounts(service) if creds.CREDENTIAL_PREFIX in m]


def test_the_clickhouse_backend_adds_its_own_account_file(tmp_path: Path) -> None:
    service = _service(_config(tmp_path, agent_record={"backend": "clickhouse"}))

    assert service["env_file"] == [creds.SERVICE_FILE, creds.CLICKHOUSE_FILE]


def test_the_mounts_hold_no_credential_and_no_other_containers_env_file(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        users=("alice", "bob"),
        deployed=("dispatch_worker",),
        services={"dispatch_worker": {"worker_count": 2}},
    )
    service = _service(config)
    sources = [m.split(":", 1)[0] for m in _string_mounts(service)]

    for source in sources:
        assert ".credentials.json" not in source
        assert ".claude.json" not in source
        assert not Path(source).name.startswith(".env")
    others = {name for files in creds.owner_files(config).values() for name in files}
    others -= set(creds.owner_files(config)[creds.RECORD_SERVICE])
    assert not others & set(service["env_file"])


# ── Transcript roots ────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["isolated", "shared"])
def test_every_transcript_root_is_mounted_whole_and_read_only(tmp_path: Path, mode: str) -> None:
    config = _config(
        tmp_path,
        users=("alice", "bob"),
        deployed=("dispatch_worker",),
        services={"dispatch_worker": {"worker_count": 2, "workspace_mode": mode}},
    )

    mounts = _transcript_mounts(_service(config))

    assert set(mounts) == set(transcript_roots(config))
    for root, volume in mounts.items():
        assert root == volume


def test_no_claude_config_and_no_workspace_volume_is_mounted(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        users=("alice",),
        deployed=("dispatch_worker",),
        services={"dispatch_worker": {"worker_count": 2}},
    )

    for mount in _string_mounts(_service(config)):
        source = mount.split(":", 1)[0]
        assert "claude-config" not in source, mount
        assert not source.startswith("dispatch_workspace"), mount
        assert "agent-data" not in source, mount


def test_no_transcript_mount_lands_where_the_entrypoint_hands_transcripts_back(
    tmp_path: Path,
) -> None:
    for mount in _string_mounts(_service(_config(tmp_path, users=("alice",)))):
        assert not mount.split(":")[1].endswith("/claude-config/projects"), mount


def test_the_terminal_volumes_are_declared_as_the_web_template_declares_them(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, users=("alice", "bob"))

    declared = _render(config)["volumes"]
    web = yaml.safe_load(render_web_terminals(copy.deepcopy(config))["docker-compose.web.yml"])

    assert set(declared) == {"alice-claude-projects", "bob-claude-projects"}
    for key, declaration in declared.items():
        assert declaration == web["volumes"][key], key


@pytest.mark.parametrize("mode", ["isolated", "shared"])
def test_dispatch_volumes_are_their_owners_declarations_alone(tmp_path: Path, mode: str) -> None:
    config = _config(
        tmp_path,
        deployed=("dispatch_worker",),
        services={"dispatch_worker": {"worker_count": 2, "workspace_mode": mode}},
    )

    doc = _render(config)
    owner = _render(config, "services/dispatch_worker/docker-compose.yml.j2")
    mounted = set(_transcript_mounts(doc["services"][SERVICE_KEY]).values())

    assert mounted
    assert mounted <= set(owner["volumes"])
    assert "volumes" not in doc


def test_no_volume_is_declared_without_terminals(tmp_path: Path) -> None:
    assert "volumes" not in _render(_config(tmp_path))


# ── The host build's transcripts ────────────────────────────────────────────


def _host_config(tmp_path: Path) -> dict[str, Any]:
    return _config(tmp_path, deployment={"host_identity": "thorsten"})


def test_nothing_of_the_hosts_is_mounted_and_the_images_user_runs_it(tmp_path: Path) -> None:
    """Claude Code writes the host's transcripts for the host account alone; the
    host session forwards them through the spool instead."""
    config = _host_config(tmp_path)
    claude_project_dir(tmp_path / "build").mkdir(parents=True)

    service = _service(config)

    assert not [m for m in service["volumes"] if isinstance(m, dict)]
    assert "user" not in service
    assert "group_add" not in service


def test_the_host_projects_dir_is_the_shipped_build_zones_not_the_output_base(
    tmp_path: Path,
) -> None:
    # ``osprey build`` renders with ``build_dir`` replaced by its staging root
    # (``output_root="."``) and keeps the shipped value as ``deployed_build_dir``.
    config = {**_host_config(tmp_path), "build_dir": ".", "deployed_build_dir": "./build"}

    assert agent_record_host_projects_dir(config, tmp_path) == claude_project_dir(
        tmp_path / "build"
    )
    assert agent_record_host_projects_dir(_config(tmp_path), tmp_path) is None


# ── A real build of every preset ────────────────────────────────────────────

PRESETS_DIR = Path(__file__).resolve().parents[2] / "src" / "osprey" / "profiles" / "presets"
PRESETS = sorted(path.stem for path in PRESETS_DIR.glob("*.yml"))


@pytest.mark.parametrize("preset", PRESETS)
def test_every_preset_build_records_host_sessions_and_writes_nothing_of_the_hosts(
    preset: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests._builds import init_project, run_build

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    repo = init_project(tmp_path, preset, "demo")
    host_dir = claude_project_dir(repo / "build")

    result = run_build(repo)

    assert result.exit_code == 0, result.output
    as_built = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))
    assert as_built["deployment"]["host_identity"]
    # The up path reads the as-built config from the repo root: the same dir.
    assert agent_record_host_projects_dir(as_built, repo) == host_dir
    assert not host_dir.exists()
    compose = repo / "build" / "services" / AGENT_RECORD_TEMPLATE_DIR / "docker-compose.yml"
    service = yaml.safe_load(compose.read_text(encoding="utf-8"))["services"][SERVICE_KEY]
    assert not [m for m in service["volumes"] if isinstance(m, dict)]
    assert "user" not in service


# ── Host provisioning ───────────────────────────────────────────────────────


@pytest.mark.skipif(os.name != "posix", reason="setgid modes are POSIX")
def test_the_build_provisions_the_spool_and_record_dir_group_shared(tmp_path: Path) -> None:
    targets = ensure_agent_record_dirs(tmp_path)

    assert targets == [
        tmp_path / "var" / "spool",
        tmp_path / "var" / "spool" / ".otlp",
        tmp_path / "var" / "agent_record",
    ]
    for target in targets:
        mode = stat.S_IMODE(target.stat().st_mode)
        assert mode & stat.S_ISGID, (target, oct(mode))
        assert mode & stat.S_IWGRP, (target, oct(mode))
    assert not (tmp_path / "var" / "archive").exists()


# ── The ClickHouse backend ──────────────────────────────────────────────────

XML_FILES = ("osprey.xml", "osprey-defaults.xml", "zz-osprey-admin.xml")


def _clickhouse_config(tmp_path: Path, **block: Any) -> dict[str, Any]:
    return _config(tmp_path, agent_record={"backend": "clickhouse", **block})


def _store(config: dict[str, Any]) -> dict[str, Any]:
    return _render(config)["services"][creds.CLICKHOUSE_SERVICE]


def _render_sibling(config: dict[str, Any], name: str) -> str:
    context = _inject_project_metadata(copy.deepcopy(config))
    return _packaged_compose_template(f"services/agent_record/{name}.j2").render(**context)


def test_the_clickhouse_backend_renders_agent_record_and_its_store(tmp_path: Path) -> None:
    doc = _render(_clickhouse_config(tmp_path))

    assert list(doc["services"]) == [SERVICE_KEY, creds.CLICKHOUSE_SERVICE]
    store = doc["services"][creds.CLICKHOUSE_SERVICE]
    assert store["container_name"] == "dls-agent-record-clickhouse"
    assert store["image"] == "clickhouse/clickhouse-server:26.8.18.2"
    assert store["restart"] == "unless-stopped"
    assert store["networks"] == ["osprey-network"]


def test_the_store_publishes_no_host_port(tmp_path: Path) -> None:
    store = _store(_clickhouse_config(tmp_path))

    assert "ports" not in store
    assert "expose" not in store
    assert "network_mode" not in store


def test_the_store_carries_its_hardening(tmp_path: Path) -> None:
    store = _store(_clickhouse_config(tmp_path))

    assert store["mem_limit"] == "2g"
    assert store["healthcheck"] == {
        "test": ["CMD-SHELL", "wget -qO- http://127.0.0.1:8123/ping | grep -q Ok"],
        "interval": "5s",
        "timeout": "3s",
        "retries": 30,
    }
    assert store["volumes"] == [
        f"{AGENT_RECORD_CLICKHOUSE_VOLUME}:/var/lib/clickhouse",
        "./build/services/agent_record/osprey.xml:/etc/clickhouse-server/config.d/osprey.xml:ro",
        "./build/services/agent_record/osprey-defaults.xml:"
        "/etc/clickhouse-server/users.d/osprey-defaults.xml:ro",
        "./build/services/agent_record/zz-osprey-admin.xml:"
        "/etc/clickhouse-server/users.d/zz-osprey-admin.xml:ro",
    ]


def test_the_store_image_and_memory_follow_the_block(tmp_path: Path) -> None:
    config = _clickhouse_config(
        tmp_path, clickhouse={"image": "clickhouse/clickhouse-server:99.1", "memory": "3g"}
    )

    store = _store(config)

    assert store["image"] == "clickhouse/clickhouse-server:99.1"
    assert store["mem_limit"] == "3g"


def test_the_store_has_its_own_data_volume(tmp_path: Path) -> None:
    doc = _render(_clickhouse_config(tmp_path))

    assert AGENT_RECORD_CLICKHOUSE_VOLUME == "agent_record_clickhouse_data"
    assert set(doc["volumes"]) == {AGENT_RECORD_CLICKHOUSE_VOLUME}


def test_the_store_volume_is_declared_beside_the_terminal_volumes(tmp_path: Path) -> None:
    config = _config(tmp_path, users=("alice",), agent_record={"backend": "clickhouse"})

    declared = _render(config)["volumes"]

    assert set(declared) == {"alice-claude-projects", AGENT_RECORD_CLICKHOUSE_VOLUME}


def test_the_admin_file_goes_to_the_store_alone(tmp_path: Path) -> None:
    doc = _render(_clickhouse_config(tmp_path))
    store = doc["services"][creds.CLICKHOUSE_SERVICE]
    record = doc["services"][SERVICE_KEY]

    assert store["env_file"] == [creds.CLICKHOUSE_ADMIN_FILE]
    assert creds.CLICKHOUSE_ADMIN_FILE not in record["env_file"]
    assert record["env_file"] == [creds.SERVICE_FILE, creds.CLICKHOUSE_FILE]
    variable = creds.digest_env_name(creds.CLICKHOUSE_SERVICE)
    assert store["labels"] == {creds.DIGEST_LABEL: f"${{{variable}:-}}"}


def test_agent_record_dials_the_store_over_the_bridge(tmp_path: Path) -> None:
    service = _service(_clickhouse_config(tmp_path))

    assert service["environment"][CLICKHOUSE_URL_ENV] == "http://agent_record_clickhouse:8123"


def test_the_embedded_backend_renders_no_store(tmp_path: Path) -> None:
    for config in (_config(tmp_path), _config(tmp_path, agent_record={"backend": "embedded"})):
        doc = _render(config)

        assert list(doc["services"]) == [SERVICE_KEY]
        assert CLICKHOUSE_URL_ENV not in doc["services"][SERVICE_KEY]["environment"]
        assert "volumes" not in doc
        assert "clickhouse" not in yaml.safe_dump(doc)


def test_the_store_server_config_pins_limits_and_a_loopback_admin(tmp_path: Path) -> None:
    config = _clickhouse_config(tmp_path)

    server = _render_sibling(config, "osprey.xml")
    defaults = _render_sibling(config, "osprey-defaults.xml")
    admin = _render_sibling(config, "zz-osprey-admin.xml")

    assert "<max_server_memory_usage_to_ram_ratio>0.9<" in server
    assert "<max_concurrent_select_queries>8<" in server
    assert "<allow_implicit_no_password>0<" in server
    assert "<max_threads>1</max_threads>" in defaults
    assert f"<{creds.CLICKHOUSE_ADMIN_USER}>" in admin
    assert '<networks replace="replace">' in admin
    assert "<ip>127.0.0.1</ip>" in admin
    assert "<ip>::1</ip>" in admin


def test_the_build_renders_the_store_config_only_for_clickhouse(tmp_path: Path) -> None:
    """The derived render writes the store's config files beside agent-record's
    compose file on the ClickHouse backend alone."""
    from osprey.deployment.compose_generator import _derived_service_renders_config_files

    clickhouse = _clickhouse_config(tmp_path)

    assert _derived_service_renders_config_files(clickhouse, AGENT_RECORD_TEMPLATE_DIR)
    assert not _derived_service_renders_config_files(_config(tmp_path), AGENT_RECORD_TEMPLATE_DIR)
    assert not _derived_service_renders_config_files(clickhouse, "history")


def test_the_store_config_renders_into_the_paths_the_store_mounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osprey.deployment.compose_generator import render_service_templates

    templates_root = Path(_packaged_compose_template(TEMPLATE).filename).parents[2]
    monkeypatch.chdir(templates_root)
    out_dir = tmp_path / "build" / "services" / "agent_record"

    rendered = render_service_templates(
        os.path.join("services", "agent_record"), _clickhouse_config(tmp_path), str(out_dir)
    )

    assert sorted(Path(path).name for path in rendered) == sorted(XML_FILES)
    mounted = {
        mount.split(":")[0].removeprefix("./build/services/agent_record/")
        for mount in _store(_clickhouse_config(tmp_path))["volumes"]
        if mount.startswith("./build/")
    }
    assert mounted == set(XML_FILES)
    for name in XML_FILES:
        assert (out_dir / name).read_text(encoding="utf-8").startswith("<clickhouse>")
