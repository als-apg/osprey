"""Framework-guard tests for projects with empty deployed_services.

The hello-world preset (and any future "agent-only" preset) declares no
deployed_services. Two failure modes have to stay fixed:

1. ``osprey build`` must still copy the root ``services/docker-compose.yml.j2``
   into the project, because the renderer references it unconditionally.
2. ``osprey deploy up`` must succeed (graceful no-op) instead of dying with
   ``TemplateNotFound`` mid-render.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml
from ruamel.yaml import YAML

from osprey.cli.build_cmd import _copy_service_templates
from osprey.deployment.compose_generator import (
    prepare_compose_files,
    resolve_project_name,
    resolve_user_volume_names,
)


def _write_config(project_path: Path, deployed_services: list[str]) -> Path:
    """Write a minimal config.yml into ``project_path`` and return its path."""
    config_path = project_path / "config.yml"
    yaml = YAML()
    config: dict = {
        "project_name": "hwt-fixture",
        "build_dir": str(project_path / "build"),
        "deployed_services": deployed_services,
    }
    with open(config_path, "w") as fh:
        yaml.dump(config, fh)
    return config_path


def test_copy_service_templates_copies_root_when_no_services(tmp_path: Path) -> None:
    """Empty deployed_services must still copy the root compose template."""
    _write_config(tmp_path, deployed_services=[])

    result = _copy_service_templates(tmp_path)

    assert result == 0, "no per-service templates should be copied"
    root_template = tmp_path / "services" / "docker-compose.yml.j2"
    assert root_template.is_file(), (
        "root services/docker-compose.yml.j2 must be copied even when "
        "deployed_services is empty (a deploying project may enable services "
        "later without rebuilding; attached projects skip this copy entirely "
        "via deploy_services: false)"
    )


def test_prepare_compose_files_no_services_renders_nothing(tmp_path: Path) -> None:
    """With empty deployed_services, prepare_compose_files renders no files at all.

    An attached project (``deploy_services: false``) scaffolds no ``services/``
    templates, so the root render must be skipped too — every consumer is
    guarded on non-empty ``deployed_services`` before invoking compose
    (deploy_up's early return; the web branch's services sub-invocation).
    """
    config_path = _write_config(tmp_path, deployed_services=[])

    monkey_cwd = Path.cwd()
    try:
        # render_template resolves SERVICES_DIR relative to cwd; deliberately
        # no _copy_service_templates — the attached case has no services/ dir.
        import os

        os.chdir(tmp_path)
        config, compose_files = prepare_compose_files(str(config_path))
    finally:
        import os

        os.chdir(monkey_cwd)

    assert compose_files == [], (
        f"empty deployed_services must render no compose files, got {compose_files}"
    )


def test_copy_service_templates_no_config_returns_zero(tmp_path: Path) -> None:
    """Missing config.yml is a no-op, not a crash."""
    result = _copy_service_templates(tmp_path)
    assert result == 0
    assert not (tmp_path / "services").exists()


@pytest.mark.parametrize("services", [[], ["does.not.exist"]])
def test_copy_service_templates_root_always_present_with_valid_pkg_services(
    tmp_path: Path,
    services: list[str],
) -> None:
    """Whether deployed_services is empty or has unknown entries, the root is copied first."""
    _write_config(tmp_path, deployed_services=services)
    _copy_service_templates(tmp_path)
    assert (tmp_path / "services" / "docker-compose.yml.j2").is_file()


# ---------------------------------------------------------------------------
# Dispatch worker provider-auth wiring
#
# The worker container runs a headless agent that needs the LLM provider key.
# Its startup hook (``inject_provider_env``) resolves it from the process
# environment, so the worker compose service declares ``env_file: ../../.env``
# — read by the compose CLI on the HOST (as the file's owner) and injected
# into the container environment. This works even though the project ``.env``
# is deliberately 0600 and the worker runs as non-root ``osprey``: the
# container itself never opens the file, so a uid mismatch can't EACCES it
# (unlike a bind mount, which the non-root process must open itself). Gated
# on ``.env`` existence — an ``env_file:`` entry whose path is missing errors
# ``compose up`` outright.
# ---------------------------------------------------------------------------

# The worker container now runs the full PROJECT image, whose layout bakes the
# project at /app/<project> (Dockerfile ``COPY . /app/{{ project_name }}/``). The
# compose paths must track that same <project> name — resolved by the generator's
# ``_inject_project_metadata`` into ``osprey_labels.project_name`` (and the
# ``<project>:local`` image tag), both from a single ``resolve_project_name(config)``
# call — so the fixtures below drive the real injection rather than hardcoding "p".
_WORKER_PROJECT_NAME = "hwt-fixture"
_ENV_FILE_LINE = "- ../../.env"


def _render_worker_template(*, env_present: bool, project_name: str = _WORKER_PROJECT_NAME) -> str:
    """Render the worker compose through the real generator injection.

    Feeds a minimal config through ``_inject_project_metadata`` (the production
    code that sets ``osprey_labels.project_name`` and defaults the worker image to
    ``<project>:local``), then renders the packaged template with that config as
    context — exactly the ctx ``render_template`` passes in production. This proves
    the rendered layout path equals the injected project name (M1 alignment) rather
    than asserting against a value the test itself hardcoded into the ctx.
    """
    from importlib import resources

    from jinja2 import Template

    from osprey.deployment.compose_generator import _inject_project_metadata

    config = _inject_project_metadata(
        {
            "project_name": project_name,
            "project_root": f"/r/{project_name}",
            "services": {"dispatch_worker": {}},
            "system": {"timezone": "UTC"},
        }
    )
    # ``_inject_project_metadata`` sets osprey_env_present from the deploy CWD;
    # override it here so the mount gating is exercised deterministically.
    config["osprey_env_present"] = env_present

    tpl = resources.files("osprey").joinpath(
        "templates/services/dispatch_worker/docker-compose.yml.j2"
    )
    template = Template(tpl.read_text(encoding="utf-8"))
    return template.render(**config)


def _render_dispatcher_template() -> str:
    from importlib import resources

    from jinja2 import Template

    tpl = resources.files("osprey").joinpath(
        "templates/services/event_dispatcher/docker-compose.yml.j2"
    )
    template = Template(tpl.read_text(encoding="utf-8"))
    return template.render(
        services={"event_dispatcher": {}},
        deployment={},
        system={"timezone": "UTC"},
        osprey_labels={"project_name": "p", "project_root": "/r", "deployed_at": "now"},
        osprey_version="",
    )


def test_dispatcher_build_context_is_project_dir_relative() -> None:
    """The event-dispatcher image builds from ./event_dispatcher (project-dir relative).

    With multiple `-f` compose files, relative paths resolve against the first
    file's dir (build/services/), not each file's own subdir. File-relative
    contexts ('.', '../event_dispatcher') break a fresh `osprey deploy up` build
    with "unable to prepare context: path .../build/event_dispatcher not found".
    """
    assert "context: ./event_dispatcher" in _render_dispatcher_template()


def test_worker_does_not_build_shared_image() -> None:
    """The worker must NOT declare its own build for its image tag.

    The worker runs the project image (``<project>:local``) that `osprey deploy
    up` builds once before `compose up` (see ``_build_project_image``). If the
    worker also declared `build:`, two builders would race to tag the same
    image — one fails with ``ERROR: image ... already exists`` (deterministic
    once base layers are cached). The worker only references the prebuilt tag
    and depends_on the event-dispatcher for ordering.
    """
    rendered = _render_worker_template(env_present=True)
    # The build directive is identified by its context/dockerfile keys (the word
    # "build" also appears in explanatory comments, so don't match on that).
    assert "context:" not in rendered and "dockerfile:" not in rendered, (
        "dispatch worker must not build the shared image — that races the "
        "event-dispatcher build on the same tag"
    )
    assert "depends_on:" in rendered and "event-dispatcher" in rendered, (
        "worker must depend_on event-dispatcher so the shared image is built first"
    )


def test_worker_template_declares_env_file_when_present() -> None:
    rendered = _render_worker_template(env_present=True)
    assert "env_file:" in rendered and _ENV_FILE_LINE in rendered, (
        "dispatch worker must declare env_file: ../../.env so the agent can "
        "authenticate to the LLM provider"
    )
    assert f"/app/{_WORKER_PROJECT_NAME}/.env" not in rendered, (
        "the project .env must not be bind-mounted into the container — the "
        "non-root worker can't open a 0600 file owned by a different uid"
    )


def test_worker_template_omits_env_file_when_absent() -> None:
    rendered = _render_worker_template(env_present=False)
    assert "env_file:" not in rendered and _ENV_FILE_LINE not in rendered, (
        "no env_file should be emitted when the project has no .env "
        "(an env_file: pointing at a missing path errors compose up)"
    )


def test_inject_project_metadata_flags_env_presence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``osprey_env_present`` reflects whether a .env exists in the deploy CWD."""
    from osprey.deployment.compose_generator import _inject_project_metadata

    monkeypatch.chdir(tmp_path)
    assert _inject_project_metadata({})["osprey_env_present"] is False

    (tmp_path / ".env").write_text("ALS_APG_API_KEY=x\n")
    assert _inject_project_metadata({})["osprey_env_present"] is True


# The worker process reads OSPREY config directly (get_facility_timezone while
# building the agent system prompt) with CWD=/app (the image WORKDIR), so without
# CONFIG_FILE it falls back to /app/config.yml and every dispatch errors with
# "No config.yml found in current directory: /app".
def test_worker_template_sets_config_file() -> None:
    rendered = _render_worker_template(env_present=True)
    assert f"CONFIG_FILE: /app/{_WORKER_PROJECT_NAME}/config.yml" in rendered, (
        "dispatch worker must set CONFIG_FILE so the worker process (and the CLI "
        "subprocess it spawns) resolve config from the mounted project image layout"
    )


# The worker runs on the compose bridge network, so the OpenObserve store is
# reachable only by its service DNS name — never localhost. The compose declares
# the host explicitly (rather than sniffing the runtime) so telemetry emit works
# identically under docker and podman; the resolver reads it as an override.
def test_worker_template_declares_openobserve_host() -> None:
    rendered = _render_worker_template(env_present=True)
    assert "OSPREY_OTEL_OPENOBSERVE_HOST: openobserve" in rendered, (
        "dispatch worker must declare the in-network OpenObserve host so an "
        "in-container agent targets the service DNS name, not its own loopback"
    )


# ---------------------------------------------------------------------------
# Task 1.3: the worker container layout must match the PROJECT image
#
# The worker now runs ``<project>:local`` (the project image built by
# ``osprey deploy up``), which bakes the project at ``/app/<project>``
# (Dockerfile ``COPY . /app/{{ project_name }}/``, ``WORKDIR /app/<project>``,
# ``chown -R osprey:osprey /app/<project>``). Every worker path — OSPREY_PROJECT_DIR,
# CONFIG_FILE, the staged config bind-mount, the .env mount, the _agent_data volume
# — must point at that same ``/app/<project>`` root, or the worker points at an
# empty/absent directory (plan risk M1). The image tag prefix, the label project
# name, and the layout path all derive from one ``resolve_project_name(config)``
# call in ``_inject_project_metadata``, so they are provably the same string.
# ---------------------------------------------------------------------------


def test_worker_image_defaults_to_project_local_when_override_unset() -> None:
    """With OSPREY_WORKER_IMAGE unset, the worker image resolves to <project>:local.

    ``_inject_project_metadata`` defaults ``services.dispatch_worker.image`` to
    ``<project>:local`` (the tag ``osprey deploy up`` builds), so the rendered
    ``image:`` line falls back to it rather than the template literal default
    ``osprey-dispatch:local``.
    """
    rendered = _render_worker_template(env_present=True)
    assert f"image: ${{OSPREY_WORKER_IMAGE:-{_WORKER_PROJECT_NAME}:local}}" in rendered, (
        "worker image must default to the injected <project>:local project image"
    )
    assert "osprey-dispatch:local" not in rendered, (
        "the shared-dispatch fallback must not survive the <project>:local injection"
    )


def test_worker_layout_paths_track_injected_project_name() -> None:
    """OSPREY_PROJECT_DIR, CONFIG_FILE, and every mount target must live under
    ``/app/<project>`` — the exact path the project image bakes (M1 alignment).

    The expected path is derived from the SAME project name the fixture feeds the
    generator, so this asserts the rendered layout equals the injected name rather
    than a literal the test invented.
    """
    proj = _WORKER_PROJECT_NAME
    root = f"/app/{proj}"
    rendered = _render_worker_template(env_present=True)

    # Env: project dir + config file
    assert f"OSPREY_PROJECT_DIR: {root}" in rendered
    assert f"CONFIG_FILE: {root}/config.yml" in rendered

    # Staged config bind-mount (python_env_path stripped) — repointed target,
    # source unchanged (relative to build/services/).
    assert f"- ./dispatch_worker/config.yml:{root}/config.yml:ro" in rendered

    # Provider auth is delivered via env_file (host-side read), not a bind
    # mount, so there is no `/app/<project>/.env` path in the container layout.
    assert f"{root}/.env" not in rendered, (
        "the worker must not reference /app/<project>/.env — env_file: delivers "
        "provider auth without exposing the file inside the (non-root) container"
    )

    # _agent_data named-volume mount target (default isolated mode -> per-worker)
    assert f"- dispatch_workspace_1:{root}/_agent_data" in rendered

    # No stale hardcoded /app/project layout may survive anywhere.
    assert "/app/project" not in rendered, (
        "the worker template must not retain the hardcoded /app/project layout"
    )


def test_worker_agent_data_volume_shared_mode_targets_project_layout() -> None:
    """In shared workspace mode the single ``dispatch_workspace`` volume must also
    mount under ``/app/<project>/_agent_data``."""
    rendered = _render_worker_template(env_present=True)
    # Re-render with shared mode via a config that sets workspace_mode.
    from importlib import resources

    from jinja2 import Template

    from osprey.deployment.compose_generator import _inject_project_metadata

    config = _inject_project_metadata(
        {
            "project_name": _WORKER_PROJECT_NAME,
            "project_root": f"/r/{_WORKER_PROJECT_NAME}",
            "services": {"dispatch_worker": {"workspace_mode": "shared"}},
            "system": {"timezone": "UTC"},
        }
    )
    config["osprey_env_present"] = True
    tpl = resources.files("osprey").joinpath(
        "templates/services/dispatch_worker/docker-compose.yml.j2"
    )
    shared = Template(tpl.read_text(encoding="utf-8")).render(**config)
    assert f"- dispatch_workspace:/app/{_WORKER_PROJECT_NAME}/_agent_data" in shared
    # The isolated-mode default (from the other fixture) uses the per-worker name.
    assert f"- dispatch_workspace_1:/app/{_WORKER_PROJECT_NAME}/_agent_data" in rendered


def test_worker_template_inactivity_defaults_to_120() -> None:
    """With no inactivity_sec configured, the worker env pins the watchdog to the
    built-in 120s default — older configs missing the field still render cleanly."""
    rendered = _render_worker_template(env_present=True)
    assert 'DISPATCH_INACTIVITY_SEC: "120"' in rendered


def test_worker_template_inactivity_reflects_injected_value() -> None:
    """A configured services.dispatch_worker.inactivity_sec flows to the worker's
    DISPATCH_INACTIVITY_SEC env, so a long single step is not cut off at 120s."""
    from importlib import resources

    from jinja2 import Template

    from osprey.deployment.compose_generator import _inject_project_metadata

    config = _inject_project_metadata(
        {
            "project_name": "p",
            "project_root": "/r/p",
            "services": {"dispatch_worker": {"inactivity_sec": 600}},
            "system": {"timezone": "UTC"},
        }
    )
    tpl = resources.files("osprey").joinpath(
        "templates/services/dispatch_worker/docker-compose.yml.j2"
    )
    rendered = Template(tpl.read_text(encoding="utf-8")).render(**config)
    assert 'DISPATCH_INACTIVITY_SEC: "600"' in rendered


def test_worker_command_unchanged() -> None:
    """The worker overrides only ``command:`` — it must still launch the
    dispatch-worker MCP server, unchanged by the image/layout repoint."""
    rendered = _render_worker_template(env_present=True)
    assert 'command: ["python", "-m", "osprey.mcp_server.dispatch_worker"]' in rendered


def test_dev_wheel_build_uses_sys_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The --dev wheel build must invoke the running interpreter, not bare python3.

    In a non-activated venv, PATH ``python3`` is the system/pyenv interpreter,
    which lacks the ``build`` package — so ``python3 -m build`` failed and --dev
    silently fell back to the PyPI release, booting containers with stale osprey
    that lacked unreleased modules. ``sys.executable`` is the venv that has build.
    """
    import subprocess
    import sys

    from osprey.deployment import compose_generator

    captured: dict = {}

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        # Return non-zero so the function bails before trying to copy a wheel.
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="stop here")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    compose_generator._copy_local_framework_for_override("/tmp/ignored")

    assert captured.get("cmd"), "the wheel build subprocess was never invoked"
    assert captured["cmd"][0] == sys.executable, (
        f"wheel build must use sys.executable ({sys.executable}), not {captured['cmd'][0]!r}"
    )
    assert captured["cmd"][1:4] == ["-m", "build", "--wheel"]


def _render_bluesky_template(
    *,
    va_deployed: bool,
    services: dict | None = None,
    writes_enabled: bool | None = None,
) -> str:
    # Load the packaged template directly (CWD-independent — mirrors
    # _render_worker_template above). Bare jinja2.Template uses the default
    # Undefined, the same mode compose_generator's Environment uses, so this
    # faithfully reproduces production render behavior.
    from importlib import resources

    from jinja2 import Template

    tpl = resources.files("osprey").joinpath("templates/services/bluesky/docker-compose.yml.j2")
    template = Template(tpl.read_text(encoding="utf-8"))
    deployed = ["bluesky"] + (["virtual_accelerator"] if va_deployed else [])
    kwargs = {
        "services": services or {"bluesky": {"port": 8090}, "virtual_accelerator": {"port": 5064}},
        "deployment": {},
        "system": {"timezone": "UTC"},
        "deployed_services": deployed,
        "osprey_labels": {"project_name": "p", "project_root": "/r", "deployed_at": "now"},
        "osprey_version": "",
    }
    # Task 3.2: control_system is omitted by default (matching every
    # pre-existing call site below), so `control_system.writes_enabled |
    # default(false)` must resolve against Jinja2's default Undefined without
    # raising. Only pass it when a caller explicitly cares.
    if writes_enabled is not None:
        kwargs["control_system"] = {"writes_enabled": writes_enabled}
    return template.render(**kwargs)


def test_bluesky_wires_va_ca_env_and_ordering_only_when_va_co_deployed() -> None:
    """The bridge's EPICS_CA_* env + ``depends_on: virtual-accelerator`` must
    render IFF the Virtual Accelerator is co-deployed (Task 4.2's conditional).

    A bridge-only deploy that still emitted ``depends_on: virtual-accelerator``
    would make ``docker compose up`` fail ("depends_on undefined service"), and
    CA env pointing at an absent VA would be dead config — so the whole block is
    gated on ``'virtual_accelerator' in deployed_services``.
    """
    with_va = _render_bluesky_template(va_deployed=True)
    assert "EPICS_CA_NAME_SERVERS:" in with_va
    assert "EPICS_CA_AUTO_ADDR_LIST:" in with_va
    assert "condition: service_healthy" in with_va
    assert "virtual-accelerator" in with_va

    without_va = _render_bluesky_template(va_deployed=False)
    assert "EPICS_CA_NAME_SERVERS:" not in without_va
    assert "depends_on:" not in without_va
    assert "virtual-accelerator" not in without_va


def test_bluesky_va_ca_port_defaults_when_va_config_block_absent() -> None:
    """VA in ``deployed_services`` but no ``services.virtual_accelerator`` config
    block must still render the default CA port (5064), never raise.

    ``'virtual_accelerator' in deployed_services`` (a list membership) does not
    guarantee a populated ``services.virtual_accelerator`` mapping. The port
    lookup defaults the intermediate to ``{}`` so a missing config key falls back
    cleanly; without that, the chained access raises ``UndefinedError`` and
    aborts the whole compose render.
    """
    rendered = _render_bluesky_template(
        va_deployed=True,
        services={"bluesky": {"port": 8090}},  # no virtual_accelerator key
    )
    assert 'EPICS_CA_NAME_SERVERS: "virtual-accelerator:5064"' in rendered


# ---------------------------------------------------------------------------
# Task 3.2 / CC-2: read-only config + limits mounts under one /app/project
# root. The connector-backed bridge reads control_system.type/writes_enabled
# and control_system.limits_checking.database_path from config.yml at
# runtime, and (when writes are enabled) resolves a relative database_path
# against project_root -- so config.yml and channel_limits.json must land
# under the SAME /app/project root the connector expects.
# ---------------------------------------------------------------------------


def test_bluesky_template_sets_config_file() -> None:
    """Mirrors dispatch_worker's identical CONFIG_FILE convention: CWD is the
    image WORKDIR (/app), not the project dir, so without CONFIG_FILE the
    connector's config lookups fall back to /app/config.yml and error "No
    config.yml found in current directory: /app".
    """
    rendered = _render_bluesky_template(va_deployed=False)
    assert "CONFIG_FILE: /app/project/config.yml" in rendered


def test_bluesky_template_always_mounts_config_yml_read_only() -> None:
    """The config.yml :ro mount must be present unconditionally -- unlike the
    VA/tiled env blocks, the bridge needs control_system settings regardless
    of which optional services are co-deployed or whether writes are
    enabled.
    """
    rendered = _render_bluesky_template(va_deployed=False)
    assert "./bluesky/config.yml:/app/project/config.yml:ro" in rendered

    # And it must still be present when writes ARE enabled (the two mounts
    # are independent, not mutually exclusive).
    rendered_writable = _render_bluesky_template(va_deployed=False, writes_enabled=True)
    assert "./bluesky/config.yml:/app/project/config.yml:ro" in rendered_writable


def test_bluesky_template_mounts_channel_limits_when_writes_enabled() -> None:
    """control_system.writes_enabled=true must mount channel_limits.json
    read-only under /app/project/data/, the same /app/project root as
    config.yml, so a relative control_system.limits_checking.database_path
    (e.g. "data/channel_limits.json") resolves against project_root exactly
    as limits_validator.py / app.py's _assert_limits_readable_if_writable
    expect. Source path mirrors the Virtual Accelerator's ../../data/...
    mount convention (build/services/ -> project root's data/).
    """
    rendered = _render_bluesky_template(va_deployed=False, writes_enabled=True)
    assert "../../data/channel_limits.json:/app/project/data/channel_limits.json:ro" in rendered


def test_bluesky_template_omits_channel_limits_mount_when_writes_disabled() -> None:
    """A read-only deploy must never mount channel_limits.json -- a
    read-only posture never opens the limits DB. Both the explicit
    ``writes_enabled: false`` case and the default render (no
    ``control_system`` key at all in the context, matching every
    pre-existing call site in this module) must omit it.
    """
    rendered = _render_bluesky_template(va_deployed=False, writes_enabled=False)
    assert "channel_limits.json" not in rendered

    rendered_default = _render_bluesky_template(va_deployed=False)
    assert "channel_limits.json" not in rendered_default


def _render_bluesky_tiled(*, tiled_enabled: bool, va_deployed: bool = False) -> str:
    return _render_bluesky_template(
        va_deployed=va_deployed,
        services={"bluesky": {"port": 8090, "tiled_enabled": tiled_enabled}},
    )


def test_bluesky_tiled_service_renders_when_enabled() -> None:
    """Task 1.1: ``tiled_enabled: true`` must render a writable-catalog Tiled
    server wired for the bridge's TiledWriter subscription (Task 2.7).

    ``serve catalog`` aborts without a catalog DB argument and defaults to
    127.0.0.1 (unreachable from the bridge container) without ``--host``.
    ``TiledWriter`` appends event tables via ``create_appendable_table`` +
    ``append_partition``, which need SQL-family storage — hence the
    ``duckdb://`` writable target alongside the filesystem one.

    The duckdb:// target uses FOUR slashes (Task 1.5 fix), not three: this
    is the standard SQLAlchemy DBAPI URI convention where an empty host
    segment leaves the path relative (three slashes, resolved against the
    container's CWD) or absolute (four slashes). Three slashes resolved to
    the relative path "storage/data.duckdb" against /app and failed
    server-side ("The directory storage does not exist."), which
    ``_FaultIsolatedTiledWriter`` caught and silently latched
    ``tiled_degraded=True`` — the scan still completed, so nothing crashed
    and persistence just silently didn't happen. The client-visible symptom
    (a 409 on the run's metadata POST) points at TiledWriter's write logic,
    not at the storage URI — the real cause is visible only server-side.

    The catalog volume mounts at /storage, NOT /data (Task 1.3 fix):
    ``ghcr.io/bluesky/tiled:0.2.12`` ships /storage pre-owned by uid=999(app),
    the user the container runs as, so a fresh named volume inherits that
    ownership from the image. /data does not exist in the image, so Docker
    creates it root:root and the uid=999 tiled process can't open a catalog
    DB there — the container exits 1 immediately and /healthz never answers.
    This is a render-time assertion only: it pins the path the fix depends
    on, but rendering a template can't execute the image or verify volume
    ownership — that's the round-trip e2e's job.
    """
    rendered = _render_bluesky_tiled(tiled_enabled=True)

    assert "\n  tiled:\n" in rendered
    assert "ghcr.io/bluesky/tiled:0.2.12" in rendered

    assert "tiled serve catalog /storage/catalog.db" in rendered
    assert "--init" in rendered
    assert "--host 0.0.0.0" in rendered
    assert "--port 8000" in rendered
    assert "-w /storage/files" in rendered
    assert "bluesky_tiled_catalog:/storage" in rendered

    # Task 1.6: --api-key must be the QUOTED form with a `:?` fail-closed
    # default, never the bare unquoted `${BLUESKY_TILED_API_KEY}`. Compose
    # splits this string `command:` form shlex-style, so an unset/empty
    # value in the bare form contributes NO argument at all (not an empty
    # one) — `--api-key` then silently swallows the next token (`-w`) as
    # its operand and a writable target vanishes, with the resulting error
    # pointing at `-w`, never at the empty key. A substring check for just
    # "--api-key" passes for both forms, so it can't discriminate; these
    # two must.
    assert '--api-key "${BLUESKY_TILED_API_KEY:?must be a non-empty alphanumeric key}"' in rendered
    assert "--api-key ${BLUESKY_TILED_API_KEY}" not in rendered

    # Neither the mount point nor the command paths may regress to /data:
    # that was the Task 1.3 bug (a root-owned mount point uid=999 can't
    # write to), and together these two absent-assertions are what would
    # catch a regression back to it.
    #
    # ":/data" pins the volume MOUNT (the actual root cause — e.g. a
    # regressed "bluesky_tiled_catalog:/data" line, which has no trailing
    # slash so a bare "/data/" check would miss it entirely).
    # "/data/" pins the COMMAND paths (catalog.db, files, duckdb target).
    # Bare "/data" isn't usable for either: it false-positives on
    # "duckdb:////storage/data.duckdb", whose filename legitimately
    # contains "data" as a substring of "storage".
    assert ":/data" not in rendered
    assert "/data/" not in rendered

    # The duckdb writable target must use exactly FOUR slashes (Task 1.5
    # fix) for an absolute path, never three (which SQLAlchemy resolves as
    # a CWD-relative path and Tiled rejects server-side). A bare
    # "duckdb://" in rendered assertion is useless here: it passes for
    # both the correct four-slash form and the buggy three-slash form.
    assert "-w duckdb:////storage/data.duckdb" in rendered
    assert "duckdb:///storage/data.duckdb" not in rendered

    # /healthz must be probed in-image via python (curl is not in the image).
    assert "localhost:8000/healthz" in rendered
    assert "python -c" in rendered

    # bridge env, fail-closed (no `:-` default on the API key)
    assert 'BLUESKY_TILED_URI: "http://tiled:8000"' in rendered
    assert "BLUESKY_TILED_API_KEY: ${BLUESKY_TILED_API_KEY}" in rendered


def test_bluesky_tiled_absent_when_disabled() -> None:
    """``tiled_enabled: false`` (the default) must render neither the tiled
    service nor any ``BLUESKY_TILED_*`` bridge env — Tiled is fully optional.
    """
    rendered = _render_bluesky_tiled(tiled_enabled=False)

    assert "\n  tiled:\n" not in rendered
    assert "BLUESKY_TILED_URI" not in rendered
    assert "BLUESKY_TILED_API_KEY" not in rendered
    assert "bluesky_tiled_catalog" not in rendered


@pytest.mark.parametrize("tiled_enabled", [True, False])
def test_bluesky_bridge_never_depends_on_tiled(tiled_enabled: bool) -> None:
    """A Tiled outage must never block the bridge from starting (FR4): the
    bridge must never get ``depends_on: tiled`` / ``condition:
    service_healthy`` gating on the tiled service, whether or not Tiled
    itself is deployed.
    """
    rendered = _render_bluesky_tiled(tiled_enabled=tiled_enabled)
    assert "depends_on:\n      tiled:" not in rendered


# ---------------------------------------------------------------------------
# Task 4.3 / FR11: turn-key scan-stack deploy config
#
# A shipped, tested deploy configuration bringing up VA + bridge + Tiled with
# control_system.type=virtual_accelerator, execution.execution_method=
# container (so BLUESKY_LAUNCH_TOKEN mints safely and the agent can arm --
# see container_lifecycle.py's _local_exec_arming_unsafe), and the scan MCP
# server enabled. tests/e2e/_orm_stack.py is the single source of this
# config, reused by the real-container round-trip e2e (task 5.2) and the
# agentic-discovery e2e (5.3/5.4) -- this gate only exercises the Docker-free
# render path via its in-process `osprey build` helper.
# ---------------------------------------------------------------------------


def test_orm_stack_renders_va_bridge_tiled_with_arming_safe_exec_and_scan_mcp(
    tmp_path: Path,
) -> None:
    """FR11's turn-key deploy config, end to end without Docker:
    ``osprey build`` (in-process, via ``tests/e2e/_orm_stack``) followed by a
    Docker-free compose render, must produce:

      - the Virtual Accelerator + bluesky-bridge + co-deployed Tiled compose
        services (``control_system.type=virtual_accelerator`` +
        ``bluesky.tiled_enabled=true``),
      - ``execution.execution_method: container`` (the arming-safe exec
        method — a ``local`` exec method gates launch-token auto-minting
        off, per ``container_lifecycle.py``'s ``_local_exec_arming_unsafe``),
      - the ``scan`` MCP server enabled in the rendered ``.mcp.json`` (it is
        ``default_enabled=False`` in the framework registry — a project must
        opt in, and this deploy config does).
    """
    import json
    import os

    from click.testing import CliRunner

    from tests.e2e import _orm_stack

    runner = CliRunner()
    project_dir = _orm_stack.build_via_cli_runner(runner, tmp_path)

    # -- execution_method: container (arming-safe) --------------------------
    yaml = YAML()
    with open(project_dir / "config.yml") as fh:
        config = yaml.load(fh)
    assert config["execution"]["execution_method"] == "container", (
        "FR11 requires execution.execution_method=container so the launch "
        "token mints safely and the agent can arm"
    )
    assert config["control_system"]["type"] == "virtual_accelerator"

    # -- scan MCP server enabled in the rendered .mcp.json -------------------
    mcp_config = json.loads((project_dir / ".mcp.json").read_text(encoding="utf-8"))
    assert "bluesky" in mcp_config["mcpServers"], (
        "the scan MCP server must be enabled (claude_code.servers.bluesky.enabled: "
        f"true) so list_plans/launch_run are reachable: {mcp_config['mcpServers'].keys()}"
    )

    # -- VA + bridge + Tiled compose services --------------------------------
    monkey_cwd = Path.cwd()
    try:
        os.chdir(project_dir)
        _, compose_files = prepare_compose_files(str(project_dir / "config.yml"))
        # Read while still inside project_dir — prepare_compose_files returns
        # paths relative to it (SERVICES_DIR resolves relative to cwd).
        rendered = "\n".join(Path(f).read_text(encoding="utf-8") for f in compose_files)
    finally:
        os.chdir(monkey_cwd)

    assert "\n  virtual-accelerator:\n" in rendered, "VA service must be deployed"
    assert "\n  bluesky-bridge:\n" in rendered, "bridge service must be deployed"
    assert "\n  tiled:\n" in rendered, "Tiled must be co-deployed (bluesky.tiled_enabled=true)"

    # -- Task 3.2 / CC-2: read-only config + limits mounts under /app/project -
    # The control-assistant preset defaults control_system.writes_enabled to
    # true (this deploy config never overrides it off), so the real,
    # fully-flattened render must mount both config.yml and channel_limits.json
    # under the same /app/project root the connector resolves project_root
    # against.
    assert config["control_system"]["writes_enabled"] is True, (
        "this assertion block assumes the control-assistant preset's "
        "writes_enabled default -- if that default ever changes, this test's "
        "premise for asserting the channel_limits.json mount changes too"
    )
    assert "CONFIG_FILE: /app/project/config.yml" in rendered
    assert "./bluesky/config.yml:/app/project/config.yml:ro" in rendered, (
        "bridge must mount config.yml read-only under /app/project (Task 3.2)"
    )
    assert "../../data/channel_limits.json:/app/project/data/channel_limits.json:ro" in rendered, (
        "control_system.writes_enabled=true (preset default) must mount "
        "channel_limits.json under the same /app/project root as config.yml"
    )


# ---------------------------------------------------------------------------
# Task 1.4: preserve-staged-config-python-env-path
#
# The M2 concern: the dispatch worker's runtime config.yml must never carry
# the HOST build machine's ``execution.python_env_path`` (e.g.
# ``/Users/.../.venv/bin/python``) into the container — that path does not
# exist in-container, and Claude Code's MCP-server command generation prefers
# it over ``sys.executable`` when present, so every MCP server would fail to
# launch. This is already handled by two independent mechanisms:
#
# 1. ``setup_build_dir`` (compose_generator.py, ~L728-738) pops
#    ``execution.python_env_path`` from the flattened config it stages for
#    the worker's config.yml bind-mount.
# 2. ``build_claude_code_context`` (osprey.cli.templates.claude_code, L91)
#    resolves ``current_python_env`` as
#    ``config.execution.python_env_path or sys.executable`` — so a config
#    with the key stripped falls back to the container's own interpreter.
#
# These tests prove both mechanisms against the real generator entrypoints
# rather than re-asserting the ``or`` expression in isolation. Mechanism 2's
# companion (build-time ``osprey claude regen`` self-healing a *recorded*
# stale python_env_path baked into an existing project) is covered by
# tests/cli/test_claude_regen.py and is out of this file's scope.
# ---------------------------------------------------------------------------


def test_setup_build_dir_strips_python_env_path_from_staged_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``setup_build_dir`` must pop ``execution.python_env_path`` from the
    flattened config it writes for the service's ``config.yml`` bind-mount.

    Drives the real staging code path: a project ``config.yml`` on disk with
    a host-looking venv path, loaded internally via ``ConfigBuilder()``
    (which resolves against ``os.getcwd()``), flattened, and written to
    ``<build_dir>/<service_dir>/config.yml``. The written file must have the
    key removed entirely, not merely emptied.
    """
    from osprey.deployment.compose_generator import setup_build_dir

    host_python = "/Users/someone/.venv/bin/python"

    project_config_path = tmp_path / "config.yml"
    project_config_path.write_text(
        yaml.dump(
            {
                "project_name": "pep-fixture",
                "execution": {"python_env_path": host_python},
            }
        )
    )

    service_dir = tmp_path / "services" / "worker"
    service_dir.mkdir(parents=True)
    (service_dir / "docker-compose.yml.j2").write_text("services:\n  worker:\n    image: test\n")

    monkeypatch.chdir(tmp_path)

    template_path = str(Path("services") / "worker" / "docker-compose.yml.j2")
    config = {"project_name": "pep-fixture", "build_dir": "./build"}
    container_cfg = {"copy_src": False, "render_kernel_templates": False}

    setup_build_dir(template_path, config, container_cfg)

    staged_config_path = tmp_path / "build" / "services" / "worker" / "config.yml"
    assert staged_config_path.is_file(), (
        f"expected a staged config.yml at {staged_config_path} "
        "(flattening must have failed and fallen back to a verbatim copy)"
    )
    staged_config = yaml.safe_load(staged_config_path.read_text())
    assert "python_env_path" not in staged_config.get("execution", {}), (
        f"host python_env_path leaked into the staged config: {staged_config.get('execution')}"
    )


def test_setup_build_dir_staged_config_has_no_execution_python_env_path_key_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same as above, phrased as a positive contract on the whole ``execution``
    block: no key named ``python_env_path`` survives staging, regardless of
    whatever else lives under ``execution``.
    """
    from osprey.deployment.compose_generator import setup_build_dir

    project_config_path = tmp_path / "config.yml"
    project_config_path.write_text(
        yaml.dump(
            {
                "project_name": "pep-fixture-2",
                "execution": {
                    "python_env_path": "/Users/someone/.venv/bin/python3.11",
                    "execution_method": "local",
                },
            }
        )
    )

    service_dir = tmp_path / "services" / "worker"
    service_dir.mkdir(parents=True)
    (service_dir / "docker-compose.yml.j2").write_text("services:\n  worker:\n    image: test\n")

    monkeypatch.chdir(tmp_path)

    template_path = str(Path("services") / "worker" / "docker-compose.yml.j2")
    config = {"project_name": "pep-fixture-2", "build_dir": "./build"}
    container_cfg = {"copy_src": False, "render_kernel_templates": False}

    setup_build_dir(template_path, config, container_cfg)

    staged_config_path = tmp_path / "build" / "services" / "worker" / "config.yml"
    staged_config = yaml.safe_load(staged_config_path.read_text())
    assert list(staged_config["execution"].keys()) == ["execution_method"], (
        "only python_env_path should be dropped; sibling execution keys must survive"
    )


def test_missing_python_env_path_falls_back_to_sys_executable() -> None:
    """The real ``.mcp.json`` generation seam: with ``execution.python_env_path``
    absent (exactly what ``setup_build_dir`` staging produces), MCP-server
    commands must resolve to the CONTAINER's own ``sys.executable``, never a
    host path.

    Drives ``build_claude_code_context`` (the actual context-builder used by
    both ``osprey build`` and ``osprey claude regen``) followed by
    ``resolve_servers``'s real command resolution, rather than asserting the
    ``config.execution.python_env_path or sys.executable`` expression in
    isolation.
    """
    import tempfile

    from osprey.cli.templates import claude_code
    from osprey.cli.templates.manager import TemplateManager

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="pep-fallback",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        config = yaml.safe_load((project_dir / "config.yml").read_text())
        config.setdefault("execution", {}).pop("python_env_path", None)

        ctx = claude_code.build_claude_code_context(
            manager.template_root, manager.jinja_env, project_dir, config
        )

        assert ctx["current_python_env"] == sys.executable

        controls_server = next(s for s in ctx["servers"] if s["name"] == "controls")
        assert controls_server["command"] == sys.executable, (
            "MCP server command must fall back to sys.executable when "
            f"python_env_path is absent, got {controls_server['command']!r}"
        )


def test_host_python_env_path_would_bake_host_interpreter_into_mcp_command() -> None:
    """Companion to the above: proves what the strip in ``setup_build_dir``
    prevents. If a host-looking ``python_env_path`` survived staging (it does
    not, per the tests above), the exact same generator would bake that host
    path into every MCP server's ``command`` — the M2 failure mode.
    """
    import tempfile

    from osprey.cli.templates import claude_code
    from osprey.cli.templates.manager import TemplateManager

    host_python = "/Users/someone/.venv/bin/python"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="pep-host",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        config = yaml.safe_load((project_dir / "config.yml").read_text())
        config.setdefault("execution", {})["python_env_path"] = host_python

        ctx = claude_code.build_claude_code_context(
            manager.template_root, manager.jinja_env, project_dir, config
        )

        assert ctx["current_python_env"] == host_python

        controls_server = next(s for s in ctx["servers"] if s["name"] == "controls")
        assert controls_server["command"] == host_python
        assert controls_server["command"] != sys.executable


# ---------------------------------------------------------------------------
# OpenObserve telemetry add-on: Docker-free compose render gate
#
# The opt-in ``openobserve`` service pulls a single public OTLP-native store so
# the agent can emit telemetry to a local backend. It is a pure-image service
# (no Dockerfile, no wheel, no src) — the closest analog is ``postgresql``.
# These gates render the packaged template through the SAME deployed_services
# gating path the CLI uses (``_copy_service_templates`` -> ``prepare_compose_files``,
# mirroring ``test_prepare_compose_files_no_services_renders_root_only``), never
# a container: they assert the compose text alone.
#
# Shared constants that must match the telemetry resolver stream: compose service
# (and in-network DNS host) ``openobserve``, port ``5080``, root-cred env vars
# ``ZO_ROOT_USER_EMAIL`` / ``ZO_ROOT_USER_PASSWORD``. The pinned image tag
# (``v0.14.4``) is a to-confirm pin — kept overridable via
# ``OSPREY_OPENOBSERVE_IMAGE`` — so these gates assert the image *reference and
# override var*, not a specific tag.
# ---------------------------------------------------------------------------

_OPENOBSERVE_IMAGE_REF = "public.ecr.aws/zinclabs/openobserve"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CI_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_OPENOBSERVE_TEMPLATE = (
    _REPO_ROOT
    / "src"
    / "osprey"
    / "templates"
    / "services"
    / "openobserve"
    / "docker-compose.yml.j2"
)


def _pinned_openobserve_tag() -> str:
    """Return the openobserve tag the compose template pins."""
    text = _OPENOBSERVE_TEMPLATE.read_text(encoding="utf-8")
    match = re.search(
        r"\$\{OSPREY_OPENOBSERVE_IMAGE:-" + re.escape(_OPENOBSERVE_IMAGE_REF) + r":([^}\s]+)\}",
        text,
    )
    assert match, "compose template no longer pins the openobserve image in the expected form"
    return match.group(1)


def test_ci_openobserve_pinned_to_ghcr_mirror() -> None:
    """CI must pull openobserve from the ghcr mirror, pinned to the compose tag.

    Two foot-guns in one assertion:

    * If the override drifts back to ``public.ecr.aws`` the rate-limit flakiness
      returns — ECR Public throttles anonymous pulls per source IP and
      GitHub-hosted runners share egress IPs.
    * The mirror is populated by hand (mirror-openobserve.yml) for one tag at a
      time, so moving the compose template's pin without moving CI's leaves every
      deploy lane pulling a tag that was never mirrored — a hard, confusing
      failure far from the line that caused it.

    Both remain latent until a runner happens to hit them, so guard statically.
    """
    pinned = _pinned_openobserve_tag()
    overrides = re.findall(
        r"OSPREY_OPENOBSERVE_IMAGE:\s*(\S+)", _CI_WORKFLOW.read_text(encoding="utf-8")
    )
    assert overrides, (
        "CI no longer overrides OSPREY_OPENOBSERVE_IMAGE — deploy lanes fall back to ECR"
    )
    for ref in overrides:
        assert ref.startswith("ghcr.io/"), (
            f"CI pins {ref!r}, not the ghcr mirror. A non-ghcr ref (e.g. ECR Public) "
            "reintroduces the anonymous-pull rate-limit flakiness."
        )
        assert ref.endswith(f":{pinned}"), (
            f"CI pins {ref!r} but the compose template pins :{pinned}. Bump both "
            "together, and run the mirror workflow for the new tag first."
        )


def _write_openobserve_config(
    project_path: Path, deployed_services: list[str], retention_days: int | None = None
) -> Path:
    """Write a config.yml that declares ``services.openobserve`` and return its path.

    Unlike the module's ``_write_config`` helper (which declares no services),
    this always declares the ``openobserve`` service block — ``deployed_services``
    controls only whether it is *launched*, exercising the opt-in gating.
    """
    config_path = project_path / "config.yml"
    yaml_rt = YAML()
    oo_service: dict = {"path": "./services/openobserve", "port": 5080}
    if retention_days is not None:
        oo_service["retention_days"] = retention_days
    config: dict = {
        "project_name": "oo-fixture",
        "project_root": str(project_path),
        "build_dir": str(project_path / "build"),
        # Real configs always carry system.timezone; the compose template uses it
        # bare (the postgresql/VA precedent has no default), so declare it here.
        "system": {"timezone": "UTC"},
        "services": {"openobserve": oo_service},
        "deployed_services": deployed_services,
    }
    with open(config_path, "w") as fh:
        yaml_rt.dump(config, fh)
    return config_path


def _render_project_compose(config_path: Path, project_path: Path) -> str:
    """Copy service templates, render via ``prepare_compose_files``, return the text.

    Runs the real CLI gating path from inside the project dir (SERVICES_DIR and
    the service ``path`` both resolve relative to cwd), then joins every rendered
    compose file so callers can assert on the aggregate text.
    """
    import os

    _copy_service_templates(project_path)

    monkey_cwd = Path.cwd()
    try:
        os.chdir(project_path)
        _, compose_files = prepare_compose_files(str(config_path))
        return "\n".join(Path(f).read_text(encoding="utf-8") for f in compose_files)
    finally:
        os.chdir(monkey_cwd)


def test_openobserve_renders_when_deployed(tmp_path: Path) -> None:
    """With ``openobserve`` in deployed_services the compose renders (exit 0) with
    the expected image reference, port, named volume, and root-cred env vars.

    The service name, port, and env var names are the shared constants the
    telemetry resolver stream also depends on — a drift here silently breaks the
    agent's OTLP push against the local store.
    """
    config_path = _write_openobserve_config(tmp_path, deployed_services=["openobserve"])
    rendered = _render_project_compose(config_path, tmp_path)

    # The service block and its in-network DNS host name.
    assert "\n  openobserve:\n" in rendered

    # Image reference + overridable pin (assert the ref and the override var, not
    # the specific to-confirm tag).
    assert f"${{OSPREY_OPENOBSERVE_IMAGE:-{_OPENOBSERVE_IMAGE_REF}:" in rendered

    # Exposed port 5080 (host:container).
    assert ":5080:5080/tcp" in rendered

    # container_name is namespaced per-project (project_name is "oo-fixture"), not
    # the host-global "osprey-openobserve" — so two projects can run the store on
    # one host without colliding. In-network reach is by the service key
    # "openobserve", which is unaffected.
    assert "container_name: oo-fixture-openobserve" in rendered
    assert "osprey-openobserve" not in rendered


def test_openobserve_retention_env_default_rendered(tmp_path: Path) -> None:
    """Omitting retention_days renders the 14-day growth bound (the compose default)."""
    config_path = _write_openobserve_config(tmp_path, deployed_services=["openobserve"])
    rendered = _render_project_compose(config_path, tmp_path)
    assert 'ZO_COMPACT_DATA_RETENTION_DAYS: "14"' in rendered


def test_openobserve_retention_env_custom_rendered(tmp_path: Path) -> None:
    """A configured retention_days flows into ZO_COMPACT_DATA_RETENTION_DAYS."""
    config_path = _write_openobserve_config(
        tmp_path, deployed_services=["openobserve"], retention_days=30
    )
    rendered = _render_project_compose(config_path, tmp_path)
    assert 'ZO_COMPACT_DATA_RETENTION_DAYS: "30"' in rendered

    # Named data volume for persistence (both the mount and the top-level decl).
    assert "openobserve_data:/data" in rendered
    assert "\nvolumes:\n  openobserve_data:\n" in rendered

    # Root credentials sourced from compose env with overridable defaults, using
    # the exact env var names the resolver expects.
    assert "ZO_ROOT_USER_EMAIL: ${ZO_ROOT_USER_EMAIL:-" in rendered
    assert "ZO_ROOT_USER_PASSWORD: ${ZO_ROOT_USER_PASSWORD:-" in rendered


def test_openobserve_absent_when_not_deployed(tmp_path: Path) -> None:
    """With ``openobserve`` declared but NOT in deployed_services it must not
    render — the opt-in posture. Only the root compose is produced, with no
    openobserve service, image, or volume anywhere in the output.
    """
    config_path = _write_openobserve_config(tmp_path, deployed_services=[])
    rendered = _render_project_compose(config_path, tmp_path)

    assert "openobserve" not in rendered, (
        f"openobserve must stay off when absent from deployed_services (opt-in):\n{rendered}"
    )


def test_openobserve_service_config_lookup_succeeds(tmp_path: Path) -> None:
    """``osprey health`` accepts a service only when it resolves under both
    ``services:`` and (for deployment) ``deployed_services:``. Health is out of
    this module's scope, so this asserts the underlying seam it relies on:
    ``find_service_config`` resolves the declared ``openobserve`` entry to its
    packaged compose template path.
    """
    from osprey.deployment.compose_generator import find_service_config

    config_path = _write_openobserve_config(tmp_path, deployed_services=["openobserve"])
    yaml_rt = YAML()
    with open(config_path) as fh:
        config = yaml_rt.load(fh)

    service_config, template_path = find_service_config(config, "openobserve")
    assert service_config is not None, "openobserve must resolve as a declared service"
    assert service_config.get("port") == 5080
    assert template_path == "./services/openobserve/docker-compose.yml.j2"


# ---------------------------------------------------------------------------
# postgresql service: per-project container_name (concurrent-deploy safety)
#
# `container_name` is a HOST-GLOBAL docker identifier, so two OSPREY projects
# deploying postgres on one host collide on a hardcoded name (they serialize on
# the build/up). The service key `postgresql` doubles as the intra-network DNS
# name, but the ARIEL DSN (`postgresql://ariel:ariel@ariel-postgres:5432/ariel`)
# resolves the host `ariel-postgres`, which today works ONLY because it is the
# container_name. Namespacing the container_name per-project must therefore keep
# `ariel-postgres` resolvable via an explicit network alias, or every in-network
# DSN consumer breaks with "could not translate host name".
# ---------------------------------------------------------------------------
def _render_postgres_template(project_name: str) -> str:
    from importlib import resources

    from jinja2 import Template

    tpl = resources.files("osprey").joinpath("templates/services/postgresql/docker-compose.yml.j2")
    template = Template(tpl.read_text(encoding="utf-8"))
    return template.render(
        services={"postgresql": {"port_host": 5432}},
        deployment={},
        system={"timezone": "UTC"},
        osprey_labels={
            "project_name": project_name,
            "project_root": f"/r/{project_name}",
            "deployed_at": "now",
        },
        osprey_version="",
    )


def test_postgres_container_name_is_project_namespaced() -> None:
    """Two projects render distinct, project-scoped postgres container names.

    A hardcoded `container_name: ariel-postgres` is host-global, so two projects
    deploying postgres on one host collide and serialize. The name must carry the
    project so concurrent deploys don't fight over one identifier.
    """
    svc_a = yaml.safe_load(_render_postgres_template("proj-a"))["services"]["postgresql"]
    svc_b = yaml.safe_load(_render_postgres_template("proj-b"))["services"]["postgresql"]

    assert svc_a["container_name"] == "proj-a-ariel-postgres", svc_a["container_name"]
    assert svc_b["container_name"] == "proj-b-ariel-postgres", svc_b["container_name"]
    assert svc_a["container_name"] != svc_b["container_name"], (
        "two projects must get distinct postgres container names or they collide "
        "on one host and cannot deploy concurrently"
    )


def test_postgres_preserves_ariel_postgres_dns_alias() -> None:
    """`ariel-postgres` stays resolvable as a network alias after namespacing.

    The default ARIEL DSN hosts `ariel-postgres`; namespacing the container_name
    must not break that hostname. An explicit `ariel-postgres` alias on
    osprey-network keeps in-network DSN consumers resolving regardless of the
    now-unique container_name.
    """
    svc = yaml.safe_load(_render_postgres_template("proj-a"))["services"]["postgresql"]

    aliases = svc["networks"]["osprey-network"]["aliases"]
    assert "ariel-postgres" in aliases, (
        "postgres must keep the `ariel-postgres` network alias so the ARIEL DSN "
        f"hostname resolves after the container_name is namespaced; got {aliases}"
    )


# ---------------------------------------------------------------------------
# sibling system-1 services: per-project container_name (concurrent-deploy safety)
#
# Every deployed system-1 service shares postgres's problem: `container_name` is
# a HOST-GLOBAL docker identifier, so two OSPREY projects deploying the same
# service on one host collide on a hardcoded name and cannot come up
# concurrently. Unlike postgres, these siblings are reached IN-NETWORK only by
# their compose *service key* (`virtual-accelerator`, `event-dispatcher`,
# `dispatch-worker-1`, `bluesky-bridge`, `tiled`) — never by container_name — so
# namespacing the container_name needs no network alias. The service key (and
# thus every depends_on / EPICS_CA_NAME_SERVERS / tiled URI / dispatch route)
# is left untouched; only the host-global container_name is namespaced.
# ---------------------------------------------------------------------------
def _render_service_template(rel_path: str, project_name: str, **overrides: object) -> str:
    """Render a service compose template with a broad, sibling-agnostic context.

    Mirrors ``_render_postgres_template`` but generalized: it supplies enough
    context for any system-1 service template and lets a caller override any top
    key (e.g. ``services`` to flip ``tiled_enabled`` or bump ``worker_count``).
    """
    from importlib import resources

    from jinja2 import Template

    tpl = resources.files("osprey").joinpath(f"templates/services/{rel_path}")
    template = Template(tpl.read_text(encoding="utf-8"))
    ctx: dict = {
        "services": {
            "virtual_accelerator": {"port": 5064},
            "event_dispatcher": {"port": 8020},
            "dispatch_worker": {"worker_count": 1, "workspace_mode": "isolated"},
            "bluesky": {"port": 8090},
            "bluesky_panels": {"port": 8095},
        },
        "deployment": {},
        "system": {"timezone": "UTC"},
        "osprey_labels": {
            "project_name": project_name,
            "project_root": f"/r/{project_name}",
            "deployed_at": "now",
        },
        "osprey_version": "",
        "osprey_env_present": False,
        "deployed_services": [],
        "control_system": {},
    }
    ctx.update(overrides)
    return template.render(**ctx)


# (template rel-path, compose service key, expected container_name suffix, render overrides)
_SIBLING_SERVICES = [
    ("virtual_accelerator/docker-compose.yml.j2", "virtual-accelerator", "virtual-accelerator", {}),
    ("event_dispatcher/docker-compose.yml.j2", "event-dispatcher", "event-dispatcher", {}),
    ("dispatch_worker/docker-compose.yml.j2", "dispatch-worker-1", "dispatch-worker-1", {}),
    ("bluesky/docker-compose.yml.j2", "bluesky-bridge", "bluesky-bridge", {}),
    (
        "bluesky/docker-compose.yml.j2",
        "tiled",
        "bluesky-tiled",
        {"services": {"bluesky": {"port": 8090, "tiled_enabled": True}}},
    ),
]


@pytest.mark.parametrize(
    ("rel_path", "service_key", "suffix", "overrides"),
    _SIBLING_SERVICES,
    ids=[s[1] for s in _SIBLING_SERVICES],
)
def test_sibling_container_name_is_project_namespaced(
    rel_path: str, service_key: str, suffix: str, overrides: dict
) -> None:
    """Two projects render distinct, project-scoped sibling container names.

    A hardcoded `container_name: osprey-<svc>` is host-global, so two projects
    deploying the same service on one host collide and cannot deploy
    concurrently. Every system-1 service must carry the project in its
    container_name, exactly as postgres does.
    """
    svc_a = yaml.safe_load(_render_service_template(rel_path, "proj-a", **overrides))
    svc_b = yaml.safe_load(_render_service_template(rel_path, "proj-b", **overrides))

    name_a = svc_a["services"][service_key]["container_name"]
    name_b = svc_b["services"][service_key]["container_name"]

    assert name_a == f"proj-a-{suffix}", name_a
    assert name_b == f"proj-b-{suffix}", name_b
    assert name_a != name_b, (
        f"two projects must get distinct `{suffix}` container names or they "
        "collide on one host and cannot deploy concurrently"
    )


def test_sibling_container_names_carry_no_stale_osprey_prefix() -> None:
    """No system-1 sibling template ships a bare host-global `osprey-<svc>` name.

    Guards against a future service (or a reverted edit) reintroducing the
    static prefix that defeats compose's per-project scoping.
    """
    for rel_path, service_key, _suffix, overrides in _SIBLING_SERVICES:
        svc = yaml.safe_load(_render_service_template(rel_path, "proj-a", **overrides))
        container_name = svc["services"][service_key]["container_name"]
        assert not container_name.startswith("osprey-"), (
            f"{service_key} still renders a host-global `{container_name}`; "
            "namespace it with the project name"
        )


def test_multi_worker_dispatch_container_names_are_each_namespaced() -> None:
    """Every dispatch worker replica gets its own project-scoped container name.

    The worker service is a Jinja `for` loop over `worker_count`; namespacing
    must apply per-replica, not just to worker-1.
    """
    rendered = yaml.safe_load(
        _render_service_template(
            "dispatch_worker/docker-compose.yml.j2",
            "proj-a",
            services={"dispatch_worker": {"worker_count": 2, "workspace_mode": "isolated"}},
        )
    )
    names = {
        key: svc["container_name"]
        for key, svc in rendered["services"].items()
        if key.startswith("dispatch-worker-")
    }
    assert names == {
        "dispatch-worker-1": "proj-a-dispatch-worker-1",
        "dispatch-worker-2": "proj-a-dispatch-worker-2",
    }, names


# ---------------------------------------------------------------------------
# resolve_user_volume_names
#
# Web terminal per-user volumes are declared bare in the compose template, so
# compose namespaces them with COMPOSE_PROJECT_NAME. runtime_helper.runtime_env
# pins COMPOSE_PROJECT_NAME to resolve_project_name(config), so the real
# runtime volume name is always "<resolve_project_name(config)>_<bare-name>".
# resolve_user_volume_names must compute that same value so volume-targeting
# code (inspect/rm/archive) doesn't have to shell out to discover it.
# ---------------------------------------------------------------------------


def test_resolve_user_volume_names_uses_explicit_project_name() -> None:
    """An explicit project_name wins over any project_root fallback."""
    config = {"project_name": "proj-a", "project_root": "/somewhere/else"}
    claude_config, agent_data = resolve_user_volume_names(config, "alice")
    assert claude_config == "proj-a_alice-claude-config"
    assert agent_data == "proj-a_alice-agent-data"


def test_resolve_user_volume_names_falls_back_to_project_root_basename() -> None:
    """Without project_name, the project comes from basename(project_root)."""
    config = {"project_root": "/home/user/my-facility-project"}
    claude_config, agent_data = resolve_user_volume_names(config, "bob")
    assert claude_config == "my-facility-project_bob-claude-config"
    assert agent_data == "my-facility-project_bob-agent-data"


def test_resolve_user_volume_names_default_project_name() -> None:
    """With neither project_name nor project_root, the default project name applies."""
    claude_config, agent_data = resolve_user_volume_names({}, "carol")
    assert claude_config == "unnamed-project_carol-claude-config"
    assert agent_data == "unnamed-project_carol-agent-data"


def test_resolve_user_volume_names_none_config_uses_default_project_name() -> None:
    """A ``None`` config is treated as empty (resolving to ``unnamed-project``),
    symmetric with ``runtime_helper.runtime_env`` — it must not raise."""
    claude_config, agent_data = resolve_user_volume_names(None, "dave")
    assert claude_config == "unnamed-project_dave-claude-config"
    assert agent_data == "unnamed-project_dave-agent-data"


# ---------------------------------------------------------------------------
# resolve_project_name normalizes its result to a valid docker-compose project
# name (lowercase; only [a-z0-9_-]; must start with a letter or number), so the
# value OSPREY pins as COMPOSE_PROJECT_NAME matches what compose would derive on
# its own. Already-valid lowercase names must pass through byte-unchanged.
# ---------------------------------------------------------------------------


def test_resolve_project_name_valid_lowercase_unchanged() -> None:
    """An already-valid lowercase name passes through byte-for-byte."""
    assert resolve_project_name({"project_name": "my-facility-project"}) == "my-facility-project"
    assert resolve_project_name({"project_name": "als_beamline-7"}) == "als_beamline-7"


def test_resolve_project_name_lowercases_mixed_case() -> None:
    """Mixed-case names are lowercased, matching compose normalization."""
    assert resolve_project_name({"project_name": "MyProject"}) == "myproject"
    assert resolve_project_name({"project_name": "ALS-Booster"}) == "als-booster"


def test_resolve_project_name_drops_spaces() -> None:
    """Whitespace is dropped (not replaced), matching compose normalization."""
    assert resolve_project_name({"project_name": "my project"}) == "myproject"
    assert resolve_project_name({"project_name": "  Spaced  Name  "}) == "spacedname"


def test_resolve_project_name_drops_invalid_characters() -> None:
    """Characters outside [a-z0-9_-] are dropped."""
    assert resolve_project_name({"project_name": "proj@2024!"}) == "proj2024"
    assert resolve_project_name({"project_name": "a.b/c:d"}) == "abcd"


def test_resolve_project_name_strips_leading_invalid_chars() -> None:
    """Leading ``_``/``-`` are stripped so the name starts with a letter or number."""
    assert resolve_project_name({"project_name": "-leading-dash"}) == "leading-dash"
    assert resolve_project_name({"project_name": "__underscored"}) == "underscored"
    assert resolve_project_name({"project_name": "-_-mixed"}) == "mixed"


def test_resolve_project_name_normalizes_project_root_fallback() -> None:
    """The basename(project_root) fallback is normalized just like project_name."""
    assert resolve_project_name({"project_root": "/home/user/My Facility"}) == "myfacility"


def test_resolve_project_name_all_invalid_falls_back_to_default() -> None:
    """A candidate with no valid characters falls back to ``unnamed-project``."""
    assert resolve_project_name({"project_name": "@#$%"}) == "unnamed-project"
    assert resolve_project_name({"project_name": "---"}) == "unnamed-project"


def test_resolve_project_name_empty_config_uses_default() -> None:
    """With neither project_name nor project_root, the default name applies."""
    assert resolve_project_name({}) == "unnamed-project"


def test_resolve_project_name_strips_trailing_separators() -> None:
    """Trailing ``_``/``-`` are stripped, not just leading ones.

    The project name is used as an image-tag prefix (``<project>-dispatch:local``);
    a surviving trailing separator would produce invalid docker references like
    ``proj_-dispatch:local``.
    """
    assert resolve_project_name({"project_name": "trailing_"}) == "trailing"
    assert resolve_project_name({"project_name": "trailing-"}) == "trailing"
    assert resolve_project_name({"project_name": "_both_"}) == "both"


def test_resolve_project_name_project_root_trailing_separator_normalized() -> None:
    """The basename(project_root) fallback also gets trailing separators stripped."""
    assert resolve_project_name({"project_root": "/home/user/my-project_"}) == "my-project"
    assert resolve_project_name({"project_root": "/home/user/my-project-"}) == "my-project"


# ---------------------------------------------------------------------------
# Project-prefixed service image tags + build args (fast-dev-image-rebuilds)
#
# The locally-built service images used host-global default tags
# (osprey-dispatch:local, osprey-va:local, ...). Service `:local` tags are
# HOST-GLOBAL docker identifiers, so two OSPREY projects building the same
# service on one host raced to tag ONE image — a sibling clean/rebuild could
# delete or replace it mid-deploy. The defaults are now project-prefixed
# (`<project>-dispatch:local`, ...); the `${OSPREY_*_IMAGE:-...}` env override
# wrappers are unchanged. The build args additionally carry
# OSPREY_PROJECT_NAME (always) and OSPREY_DEV=1 (iff `osprey deploy up --dev`,
# via the dev_mode key setup_build_dir plumbs into the render context).
# The external Tiled image (a pulled upstream image, never built locally) is
# deliberately NOT project-prefixed.
# ---------------------------------------------------------------------------

# (template rel-path, compose service key, image override env var, local-tag suffix)
_PREFIXED_IMAGE_SERVICES = [
    (
        "event_dispatcher/docker-compose.yml.j2",
        "event-dispatcher",
        "OSPREY_DISPATCH_IMAGE",
        "dispatch",
    ),
    (
        "virtual_accelerator/docker-compose.yml.j2",
        "virtual-accelerator",
        "OSPREY_VA_IMAGE",
        "va",
    ),
    (
        "bluesky/docker-compose.yml.j2",
        "bluesky-bridge",
        "OSPREY_BLUESKY_BRIDGE_IMAGE",
        "bluesky-bridge",
    ),
    (
        "bluesky_panels/docker-compose.yml.j2",
        "bluesky-panels",
        "OSPREY_BLUESKY_PANELS_IMAGE",
        "bluesky-panels",
    ),
]

_PREFIXED_IDS = [s[1] for s in _PREFIXED_IMAGE_SERVICES]


@pytest.mark.parametrize(
    ("rel_path", "service_key", "env_var", "suffix"), _PREFIXED_IMAGE_SERVICES, ids=_PREFIXED_IDS
)
def test_service_image_default_is_project_prefixed(
    rel_path: str, service_key: str, env_var: str, suffix: str
) -> None:
    """Each locally-built service image defaults to ``<project>-<suffix>:local``,
    keeping the ``${OSPREY_*_IMAGE:-...}`` env override wrapper intact."""
    svc = yaml.safe_load(_render_service_template(rel_path, "proj-a"))["services"][service_key]
    assert svc["image"] == f"${{{env_var}:-proj-a-{suffix}:local}}", svc["image"]


def test_two_projects_render_disjoint_local_image_tag_sets() -> None:
    """Two project names must produce fully disjoint default local-tag sets —
    otherwise sibling deploys on one host still race on a shared tag."""

    def local_tags(project: str) -> set[str]:
        tags = set()
        for rel_path, service_key, _env_var, _suffix in _PREFIXED_IMAGE_SERVICES:
            svc = yaml.safe_load(_render_service_template(rel_path, project))["services"][
                service_key
            ]
            match = re.fullmatch(r"\$\{[A-Z_]+:-(.+)\}", svc["image"])
            assert match, f"unexpected image form: {svc['image']}"
            tags.add(match.group(1))
        return tags

    tags_a = local_tags("proj-a")
    tags_b = local_tags("proj-b")
    assert len(tags_a) == len(_PREFIXED_IMAGE_SERVICES)
    assert len(tags_b) == len(_PREFIXED_IMAGE_SERVICES)
    assert tags_a.isdisjoint(tags_b), f"shared tags: {tags_a & tags_b}"


@pytest.mark.parametrize(
    ("rel_path", "service_key", "env_var", "suffix"), _PREFIXED_IMAGE_SERVICES, ids=_PREFIXED_IDS
)
def test_service_build_args_carry_project_name_and_dev_flag(
    rel_path: str, service_key: str, env_var: str, suffix: str
) -> None:
    """build.args always carry OSPREY_PROJECT_NAME; OSPREY_DEV renders as "1"
    iff dev mode, and is entirely absent otherwise."""
    prod = yaml.safe_load(_render_service_template(rel_path, "proj-a"))["services"][service_key]
    prod_args = prod["build"]["args"]
    assert prod_args["OSPREY_PROJECT_NAME"] == "proj-a"
    assert "OSPREY_VERSION" in prod_args, "the existing OSPREY_VERSION arg must survive"
    assert "OSPREY_DEV" not in prod_args, "OSPREY_DEV must be absent outside dev mode"

    dev = yaml.safe_load(_render_service_template(rel_path, "proj-a", dev_mode=True))["services"][
        service_key
    ]
    dev_args = dev["build"]["args"]
    assert dev_args["OSPREY_PROJECT_NAME"] == "proj-a"
    assert dev_args["OSPREY_DEV"] == "1"


def test_tiled_external_image_stays_unprefixed() -> None:
    """The Tiled service pulls an external upstream image — it is never built
    locally, so it must NOT get a project-prefixed tag (or any build block)."""
    rendered = _render_service_template(
        "bluesky/docker-compose.yml.j2",
        "proj-a",
        services={"bluesky": {"port": 8090, "tiled_enabled": True}},
    )
    tiled = yaml.safe_load(rendered)["services"]["tiled"]
    assert tiled["image"] == "${OSPREY_TILED_IMAGE:-ghcr.io/bluesky/tiled:0.2.12}"
    assert "build" not in tiled


# ---------------------------------------------------------------------------
# Task 2.2: the dispatch worker's TEMPLATE-LEVEL image fallback
#
# The worker runs the PROJECT image (<project>:local — built by `osprey deploy
# up` from the project Dockerfile), never the dispatcher's image. Its rendered
# default normally comes from _inject_project_metadata's setdefault on
# services.dispatch_worker.image — but that setdefault only fires when
# `dispatch_worker:` is a mapping. A null `dispatch_worker:` key (legal YAML)
# skips it, exposing the template's own `| default(...)` literal, which used to
# be the WRONG image (osprey-dispatch:local). The template fallback must equal
# _worker_image_target's Python fallback for the same config.
# ---------------------------------------------------------------------------


def test_worker_template_fallback_matches_worker_image_target_python_fallback() -> None:
    """With a null ``dispatch_worker:`` key (setdefault can't fire), the rendered
    worker image fallback must equal ``_worker_image_target``'s ``<project>:local``."""
    from importlib import resources

    from jinja2 import Template

    from osprey.deployment.compose_generator import _inject_project_metadata
    from osprey.deployment.container_lifecycle import _worker_image_target

    config = {
        "project_name": "fbk-proj",
        "project_root": "/r/fbk-proj",
        # A null dispatch_worker: key — _inject_project_metadata's isinstance
        # guard skips the image setdefault, so the template's own default fires.
        "services": {"dispatch_worker": None},
        "system": {"timezone": "UTC"},
    }
    injected = _inject_project_metadata(config)
    assert injected["services"]["dispatch_worker"] is None, (
        "precondition: the setdefault path must NOT have injected an image"
    )

    tpl = resources.files("osprey").joinpath(
        "templates/services/dispatch_worker/docker-compose.yml.j2"
    )
    rendered = Template(tpl.read_text(encoding="utf-8")).render(**injected)

    expected = _worker_image_target(config, env={})
    assert expected == "fbk-proj:local"
    assert f"image: ${{OSPREY_WORKER_IMAGE:-{expected}}}" in rendered, (
        "the template-level fallback must match _worker_image_target's Python "
        "fallback — the worker runs the PROJECT image, not the dispatcher's"
    )
    assert "osprey-dispatch:local" not in rendered


# ---------------------------------------------------------------------------
# Wheel-build memo (Task 3.1) + Dockerfile-scoped staging (Task 3.2)
#
# One `--dev` deploy stages the SAME wheel into many build contexts (every
# Dockerfile-bearing service context, the project-image root, each persona
# project), and rebuild_deployment renders the compose files twice (its own
# prepare_compose_files + the delegated deploy_up's). The isolated-env
# `python -m build` takes tens of seconds, so it is memoized per process:
# built at most once, copied per context. Staging is also scoped to build
# contexts that actually contain a Dockerfile — pure-image services
# (postgresql, openobserve) never install the wheel.
# ---------------------------------------------------------------------------


def _write_dispatch_stack_config(project_path: Path, deployed: list[str]) -> Path:
    """Write a config.yml declaring event_dispatcher + postgresql; return its path."""
    config_path = project_path / "config.yml"
    yaml_rt = YAML()
    config: dict = {
        "project_name": "wheel-fixture",
        "project_root": str(project_path),
        "build_dir": str(project_path / "build"),
        "system": {"timezone": "UTC"},
        "services": {
            "event_dispatcher": {"path": "./services/event_dispatcher", "port": 8020},
            "postgresql": {"path": "./services/postgresql", "port_host": 5432},
        },
        "deployed_services": deployed,
    }
    with open(config_path, "w") as fh:
        yaml_rt.dump(config, fh)
    return config_path


# METADATA for the fixture wheel _write_fixture_wheel builds: two plain base
# deps, one dep kept behind a non-extra (python_version) marker, and two
# extra-gated deps that must stay OUT of the local-requirements manifest.
_FIXTURE_WHEEL_METADATA = (
    "Metadata-Version: 2.1\n"
    "Name: osprey-framework\n"
    "Version: 0.0.0\n"
    "Requires-Dist: softioc>=4.5\n"
    "Requires-Dist: aiohttp\n"
    'Requires-Dist: tomli>=2; python_version < "3.11"\n'
    'Requires-Dist: pytest>=8; extra == "dev"\n'
    'Requires-Dist: sphinx; extra == "docs"\n'
)

# The manifest _FIXTURE_WHEEL_METADATA must produce: extras excluded, non-extra
# markers verbatim, sorted, one per line, trailing newline.
_FIXTURE_WHEEL_EXPECTED_MANIFEST = (
    "aiohttp\nsoftioc>=4.5\ntomli>=2; python_version < \"3.11\"\n"
)


def _write_fixture_wheel(path: Path) -> None:
    """Write a minimal but structurally valid wheel zip with real METADATA."""
    import zipfile

    with zipfile.ZipFile(path, "w") as whl:
        whl.writestr("osprey_framework-0.0.0.dist-info/METADATA", _FIXTURE_WHEEL_METADATA)


@pytest.fixture
def spy_wheel_build(monkeypatch: pytest.MonkeyPatch) -> list:
    """Spy on the ``python -m build`` subprocess, faking a successful build.

    Records every build invocation and drops a minimal VALID wheel (real zip
    with a ``*.dist-info/METADATA``) into the requested ``--outdir`` so
    ``_copy_local_framework_for_override`` proceeds through both its copy step
    and its requirements-manifest step. Every other subprocess.run call passes
    through untouched.
    """
    import subprocess as subprocess_module

    calls: list = []
    real_run = subprocess_module.run

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(cmd, list) and cmd[1:3] == ["-m", "build"]:
            calls.append(list(cmd))
            outdir = cmd[cmd.index("--outdir") + 1]
            _write_fixture_wheel(Path(outdir, "osprey_framework-0.0.0-py3-none-any.whl"))
            return subprocess_module.CompletedProcess(cmd, 0, stdout="", stderr="")
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(subprocess_module, "run", _fake_run)
    return calls


def test_dev_wheel_builds_once_across_service_and_project_staging(
    tmp_path: Path, spy_wheel_build: list
) -> None:
    """One process, many staging targets, exactly ONE ``python -m build``.

    Drives the real service-context staging (``prepare_compose_files`` with
    dev_mode) plus two further ``_copy_local_framework_for_override`` calls
    with distinct out_dirs — the same helper the project-image build
    (``_build_project_image``) and persona builds (``build_persona_images``)
    invoke against their own contexts. Every context must still receive its
    own wheel copy (the memo caches the BUILD, not the copy).
    """
    import os

    from osprey.deployment.compose_generator import _copy_local_framework_for_override

    config_path = _write_dispatch_stack_config(tmp_path, deployed=["event_dispatcher"])
    _copy_service_templates(tmp_path)

    project_image_ctx = tmp_path / "project-image-ctx"
    persona_ctx = tmp_path / "persona-ctx"
    project_image_ctx.mkdir()
    persona_ctx.mkdir()

    monkey_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        prepare_compose_files(str(config_path), dev_mode=True)
        assert _copy_local_framework_for_override(str(project_image_ctx)) is True
        assert _copy_local_framework_for_override(str(persona_ctx)) is True
    finally:
        os.chdir(monkey_cwd)

    assert len(spy_wheel_build) == 1, (
        f"the wheel build subprocess must run exactly once, ran {len(spy_wheel_build)}x"
    )
    service_ctx = tmp_path / "build" / "services" / "event_dispatcher"
    for ctx in (service_ctx, project_image_ctx, persona_ctx):
        assert list(ctx.glob("*.whl")), f"no wheel staged into {ctx}"
        assert (ctx / "osprey-local-requirements.txt").is_file(), (
            f"no local-requirements manifest staged next to the wheel in {ctx}"
        )


def test_dev_wheel_builds_once_across_rebuild_deployment_renders(
    tmp_path: Path, spy_wheel_build: list
) -> None:
    """``rebuild_deployment`` runs ``prepare_compose_files`` twice (its own call
    plus the delegated ``deploy_up``'s) — the wheel build must still run once."""
    import os

    config_path = _write_dispatch_stack_config(tmp_path, deployed=["event_dispatcher"])
    _copy_service_templates(tmp_path)

    monkey_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        prepare_compose_files(str(config_path), dev_mode=True)
        prepare_compose_files(str(config_path), dev_mode=True)
    finally:
        os.chdir(monkey_cwd)

    assert len(spy_wheel_build) == 1, (
        f"the wheel build subprocess must run exactly once, ran {len(spy_wheel_build)}x"
    )


def test_dev_wheel_staged_only_into_dockerfile_build_contexts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dev-mode staging must target only build contexts that contain a
    Dockerfile: postgresql (pure-image, no Dockerfile) gets nothing, the
    event_dispatcher context gets the wheel."""
    import os

    from osprey.deployment import compose_generator

    staged: list[str] = []

    def _record_staging(out_dir: str) -> bool:
        staged.append(out_dir)
        return True

    monkeypatch.setattr(compose_generator, "_copy_local_framework_for_override", _record_staging)

    config_path = _write_dispatch_stack_config(
        tmp_path, deployed=["postgresql", "event_dispatcher"]
    )
    _copy_service_templates(tmp_path)

    monkey_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        prepare_compose_files(str(config_path), dev_mode=True)
    finally:
        os.chdir(monkey_cwd)

    assert len(staged) == 1, f"expected staging into exactly one context, got {staged}"
    assert staged[0].endswith(os.path.join("services", "event_dispatcher")), staged[0]


# ---------------------------------------------------------------------------
# Fail-closed OSPREY_DEV rendering: the pin-relaxing OSPREY_DEV=1 build arg is
# rendered into a service's compose build.args only when the dev wheel was
# actually staged into THAT context. A --dev deploy whose wheel build/staging
# failed must render WITHOUT the arg — with an unreleased pin, OSPREY_DEV=1
# would otherwise silently install the latest published release instead of
# the local code the flag promises.
# ---------------------------------------------------------------------------


def _rendered_dispatcher_build_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, staging_result: bool
) -> dict:
    """Run a --dev prepare_compose_files with a stubbed staging outcome and
    return the event-dispatcher's rendered build.args."""
    from osprey.deployment import compose_generator

    monkeypatch.setattr(
        compose_generator,
        "_copy_local_framework_for_override",
        lambda out_dir: staging_result,
    )
    config_path = _write_dispatch_stack_config(tmp_path, deployed=["event_dispatcher"])
    _copy_service_templates(tmp_path)
    monkeypatch.chdir(tmp_path)
    prepare_compose_files(str(config_path), dev_mode=True)
    compose_file = tmp_path / "build" / "services" / "event_dispatcher" / "docker-compose.yml"
    return yaml.safe_load(compose_file.read_text())["services"]["event-dispatcher"]["build"][
        "args"
    ]


def test_rendered_compose_carries_osprey_dev_when_wheel_staged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _rendered_dispatcher_build_args(tmp_path, monkeypatch, staging_result=True)
    assert args["OSPREY_DEV"] == "1"


def test_rendered_compose_omits_osprey_dev_when_wheel_staging_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _rendered_dispatcher_build_args(tmp_path, monkeypatch, staging_result=False)
    assert "OSPREY_DEV" not in args, (
        "a failed dev-wheel staging must not render the pin-relaxing "
        f"OSPREY_DEV build arg (fail-closed); got {args}"
    )


def test_rendered_compose_omits_osprey_dev_on_memoized_build_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive the REAL staging helper against a failing ``python -m build``:
    the memoized failure must leave OSPREY_DEV out of the rendered compose."""
    import subprocess as subprocess_module

    real_run = subprocess_module.run

    def _failing_build(cmd, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(cmd, list) and cmd[1:3] == ["-m", "build"]:
            return subprocess_module.CompletedProcess(
                cmd, 1, stdout="", stderr="No module named build"
            )
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(subprocess_module, "run", _failing_build)

    config_path = _write_dispatch_stack_config(tmp_path, deployed=["event_dispatcher"])
    _copy_service_templates(tmp_path)
    monkeypatch.chdir(tmp_path)
    prepare_compose_files(str(config_path), dev_mode=True)

    compose_file = tmp_path / "build" / "services" / "event_dispatcher" / "docker-compose.yml"
    args = yaml.safe_load(compose_file.read_text())["services"]["event-dispatcher"]["build"][
        "args"
    ]
    assert "OSPREY_DEV" not in args
    service_ctx = tmp_path / "build" / "services" / "event_dispatcher"
    assert not list(service_ctx.glob("*.whl")), "no wheel may be staged on a failed build"


# ---------------------------------------------------------------------------
# Wheel-cache temp dir lifecycle: the memo's mkdtemp dir must register an
# atexit cleanup when first created, so a real --dev deploy process doesn't
# leak a temp dir (with a wheel copy) on every run. The reset hook stays
# idempotent — it runs both explicitly (tests) and again at process exit.
# ---------------------------------------------------------------------------


def test_wheel_cache_dir_creation_registers_atexit_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spy_wheel_build: list
) -> None:
    from osprey.deployment import compose_generator
    from osprey.deployment.compose_generator import (
        _copy_local_framework_for_override,
        _reset_wheel_build_cache,
    )

    registered: list = []
    monkeypatch.setattr(compose_generator.atexit, "register", lambda fn: registered.append(fn))
    monkeypatch.setattr(compose_generator, "_wheel_cache_cleanup_registered", False)

    ctx = tmp_path / "ctx"
    ctx.mkdir()
    assert _copy_local_framework_for_override(str(ctx)) is True
    assert compose_generator._reset_wheel_build_cache in registered, (
        "creating the wheel cache dir must register the reset hook with atexit"
    )

    cache_dir = compose_generator._wheel_cache_dir
    assert cache_dir is not None and Path(cache_dir).is_dir()
    # The registered hook removes the dir and is safe to call twice
    # (explicitly now, and again at interpreter exit).
    _reset_wheel_build_cache()
    assert not Path(cache_dir).exists()
    _reset_wheel_build_cache()


# ---------------------------------------------------------------------------
# Task 3.4: dev-wheel reproducibility
#
# The service Dockerfiles COPY the staged wheel into an early layer; if two
# builds of identical source produced byte-different wheels, every --dev
# rebuild would invalidate the image layer cache even with no code change.
# This gate builds the wheel twice for real (memo reset in between) and pins
# byte-identity via sha256.
# ---------------------------------------------------------------------------


def test_dev_wheel_build_is_reproducible(tmp_path: Path) -> None:
    """Two dev-wheel builds from identical source yield identical sha256.

    Skips cleanly under exactly the conditions the production helper itself
    falls back on: osprey not editable-installed, no pyproject.toml at the
    source root, or the ``build`` package unavailable.
    """
    import hashlib
    import importlib.util

    import osprey
    from osprey.deployment.compose_generator import (
        _copy_local_framework_for_override,
        _reset_wheel_build_cache,
    )

    module_path = Path(osprey.__file__).parent
    if "site-packages" in str(module_path) or "dist-packages" in str(module_path):
        pytest.skip("osprey is not an editable install — dev wheel build unavailable")
    source_root = module_path.parent.parent
    if not (source_root / "pyproject.toml").exists():
        pytest.skip(f"no pyproject.toml at {source_root} — cannot build wheel from source")
    if importlib.util.find_spec("build") is None:
        pytest.skip("the 'build' package is not installed")

    digests = []
    for label in ("first", "second"):
        out_dir = tmp_path / label
        out_dir.mkdir()
        # Reset the memo so the second round is a genuinely fresh build, not
        # a copy of the first round's cached wheel.
        _reset_wheel_build_cache()
        if not _copy_local_framework_for_override(str(out_dir)):
            pytest.skip("dev wheel build unavailable in this environment")
        [wheel] = out_dir.glob("*.whl")
        digests.append(hashlib.sha256(wheel.read_bytes()).hexdigest())

    assert digests[0] == digests[1], (
        "two wheel builds from identical source must be byte-identical — a "
        "nondeterministic wheel invalidates the Docker layer cache on every "
        "--dev rebuild"
    )


# ---------------------------------------------------------------------------
# osprey-local-requirements.txt: staged next to every dev wheel so the service
# Dockerfiles' toolchain-equipped deps layer can install the local wheel's own
# base dependency set (the released PyPI pin's deps may lack e.g. softioc, a
# native sdist that cannot compile in the toolchain-less wheel layer). The
# content contract — extras excluded, non-extra markers verbatim, sorted, one
# per line, trailing newline — must be byte-deterministic for identical wheels
# because BuildKit content-hashes the COPY'd file.
# ---------------------------------------------------------------------------


def test_local_requirements_manifest_written_next_to_wheel(
    tmp_path: Path, spy_wheel_build: list
) -> None:
    """The REAL staging helper writes the manifest with exactly the expected
    content: extra-gated deps excluded, python_version marker kept verbatim,
    sorted, trailing newline."""
    from osprey.deployment.compose_generator import _copy_local_framework_for_override

    ctx = tmp_path / "ctx"
    ctx.mkdir()
    assert _copy_local_framework_for_override(str(ctx)) is True

    manifest = ctx / "osprey-local-requirements.txt"
    assert manifest.is_file(), "manifest must be staged next to the wheel"
    assert manifest.read_text(encoding="utf-8") == _FIXTURE_WHEEL_EXPECTED_MANIFEST
    assert list(ctx.glob("*.whl")), "the wheel itself must still be staged"


def test_local_requirements_manifest_is_deterministic(
    tmp_path: Path, spy_wheel_build: list
) -> None:
    """Two stagings of the same wheel produce byte-identical manifests —
    anything else busts BuildKit's content-hashed deps-layer cache on every
    deploy."""
    from osprey.deployment.compose_generator import _copy_local_framework_for_override

    contents = []
    for label in ("first", "second"):
        ctx = tmp_path / label
        ctx.mkdir()
        assert _copy_local_framework_for_override(str(ctx)) is True
        contents.append((ctx / "osprey-local-requirements.txt").read_bytes())

    assert contents[0] == contents[1]


def test_staging_fails_closed_when_manifest_cannot_be_derived(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wheel whose METADATA cannot be read (corrupt zip) must FAIL staging —
    and leave neither wheel nor manifest behind. A context holding a wheel
    without its manifest would trip the OSPREY_DEV gate while the deps layer
    still lacks the local wheel's added deps (half-staged)."""
    import subprocess as subprocess_module

    from osprey.deployment.compose_generator import _copy_local_framework_for_override

    real_run = subprocess_module.run

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(cmd, list) and cmd[1:3] == ["-m", "build"]:
            outdir = cmd[cmd.index("--outdir") + 1]
            Path(outdir, "osprey_framework-0.0.0-py3-none-any.whl").write_bytes(b"not a zip")
            return subprocess_module.CompletedProcess(cmd, 0, stdout="", stderr="")
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(subprocess_module, "run", _fake_run)

    ctx = tmp_path / "ctx"
    ctx.mkdir()
    assert _copy_local_framework_for_override(str(ctx)) is False
    assert not list(ctx.glob("*.whl")), "the half-staged wheel must be removed"
    assert not (ctx / "osprey-local-requirements.txt").exists()


def test_staging_fails_closed_when_manifest_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spy_wheel_build: list
) -> None:
    """Even with a valid wheel, a failed manifest WRITE must fail staging
    (return False) and remove the already-copied wheel."""
    from osprey.deployment import compose_generator
    from osprey.deployment.compose_generator import _copy_local_framework_for_override

    def _boom(cached_wheel, out_dir):  # type: ignore[no-untyped-def]
        raise OSError("disk full")

    monkeypatch.setattr(compose_generator, "_write_local_requirements_manifest", _boom)

    ctx = tmp_path / "ctx"
    ctx.mkdir()
    assert _copy_local_framework_for_override(str(ctx)) is False
    assert not list(ctx.glob("*.whl")), "the half-staged wheel must be removed"
    assert not (ctx / "osprey-local-requirements.txt").exists()
