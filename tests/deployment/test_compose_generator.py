"""Framework-guard tests for projects with empty deployed_services.

The hello-world preset (and any future "agent-only" preset) declares no
deployed_services. Two failure modes have to stay fixed:

1. ``osprey build`` must still copy the root ``services/docker-compose.yml.j2``
   into the project, because the renderer references it unconditionally.
2. ``osprey up`` must succeed (graceful no-op) instead of dying with
   ``TemplateNotFound`` mid-render.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import yaml
from ruamel.yaml import YAML

from osprey.cli.build_cmd import _copy_service_templates
from osprey.deployment.compose_generator import (
    prepare_compose_files,
    resolve_project_name,
    resolve_user_volume_names,
)
from osprey.utils.workspace import DEFAULT_AGENT_DATA_BASE_DIR, RENDERED_CONFIG_RELPATH


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


def test_prepare_compose_files_no_services_renders_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With empty deployed_services, prepare_compose_files renders no files at all.

    An attached project (``deploy_services: false``) scaffolds no ``services/``
    templates, so the root render must be skipped too — every consumer is
    guarded on non-empty ``deployed_services`` before invoking compose
    (deploy_up's early return; the web branch's services sub-invocation).
    """
    config_path = _write_config(tmp_path, deployed_services=[])

    # render_template resolves SERVICES_DIR relative to cwd; deliberately
    # no _copy_service_templates — the attached case has no services/ dir.
    monkeypatch.chdir(tmp_path)
    config, compose_files = prepare_compose_files(str(config_path))

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
# environment, so the worker compose service declares ``env_file: ./.env``
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
_ENV_FILE_LINE = "- ./.env"


def _render_worker_template(
    *,
    env_present: bool,
    project_name: str = _WORKER_PROJECT_NAME,
    deployed_services: list[str] | None = None,
) -> str:
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
            "deployed_services": list(deployed_services or []),
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
    """The event-dispatcher builds from ./build/services/event_dispatcher.

    Every relative path in every compose file resolves against ONE directory —
    the pinned compose project directory, which is the deployment repo root
    (``--project-directory``, see ``compose_base_cmd``) — never the file's own
    subdir. So a context naming the render has to spell the build zone.
    """
    assert "context: ./build/services/event_dispatcher" in _render_dispatcher_template()


def test_worker_does_not_build_shared_image() -> None:
    """The worker must NOT declare its own build for its image tag.

    The worker runs the project image (``<project>:local``) that `osprey up`
    builds once before `compose up` (see ``_build_project_image``). If the
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
        "dispatch worker must declare env_file: ./.env so the agent can "
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


# ---------------------------------------------------------------------------
# Virtual Accelerator scenario-state mount
# ---------------------------------------------------------------------------
# The VA container reads `active_scenarios` from a bind mount while the host
# rewrites it (`osprey sim apply`). The mount source is derived from the config
# rather than hardcoded, so a project that relocates its agent-data root does
# not end up mounting a directory nothing writes.


def _render_va_template(config: dict[str, Any]) -> str:
    """Render the packaged VA compose through the real generator injection."""
    from importlib import resources

    from jinja2 import Template

    from osprey.deployment.compose_generator import _inject_project_metadata

    ctx = _inject_project_metadata(
        {
            "system": {"timezone": "UTC"},
            "deployment": {"bind_address": "127.0.0.1"},
            "services": {"virtual_accelerator": {"port": 5064}},
            **config,
        }
    )
    tpl = resources.files("osprey").joinpath(
        "templates/services/virtual_accelerator/docker-compose.yml.j2"
    )
    return Template(tpl.read_text(encoding="utf-8")).render(**ctx)


def test_va_state_mount_defaults_to_the_agent_data_root() -> None:
    rendered = _render_va_template({})

    assert f"- ./{DEFAULT_AGENT_DATA_BASE_DIR}/simulation:/state/simulation:ro" in rendered
    assert "VA_STATE_DIR: /state/simulation" in rendered


def test_va_state_mount_follows_a_relocated_agent_data_root() -> None:
    rendered = _render_va_template({"agent_data": {"base_dir": "./scratch-data"}})

    assert "- ./scratch-data/simulation:/state/simulation:ro" in rendered
    mounts = [line for line in rendered.splitlines() if ":/state/simulation:ro" in line]
    assert mounts and not any("_agent_data" in line for line in mounts), (
        "the state mount source must come from agent_data.base_dir, not a literal"
    )


def test_va_state_mount_accepts_an_absolute_agent_data_root() -> None:
    """An absolute root is already anchored — the ../../ prefix would break it."""
    rendered = _render_va_template({"agent_data": {"base_dir": "/srv/osprey-state"}})

    assert "- /srv/osprey-state/simulation:/state/simulation:ro" in rendered


@pytest.mark.parametrize(
    "relocation",
    [
        pytest.param({}, id="default"),
        pytest.param({"agent_data": {"base_dir": "./scratch-data"}}, id="relocated-agent-data"),
        pytest.param({"simulation": {"state_dir": "run/scenarios"}}, id="explicit-state-dir"),
    ],
)
def test_va_state_mount_matches_what_the_engine_writes(
    tmp_path: Path, relocation: dict[str, Any]
) -> None:
    """The mount source and the writer's target are one directory.

    Every way a project can move that directory has to move both sides together:
    relocating the agent-data root, or naming the state dir outright. The mount
    is rendered relative to the pinned compose project directory — the repo root
    — so resolving it from there must land on exactly the path
    ``resolve_state_dir`` hands the engine.
    """
    from osprey.simulation.engine import resolve_state_dir

    config: dict[str, Any] = {"project_root": str(tmp_path), **relocation}
    rendered = _render_va_template(config)
    source = re.search(r"- (\S+):/state/simulation:ro", rendered).group(1)

    assert (tmp_path / source).resolve() == resolve_state_dir(config, tmp_path).resolve()


# The worker process reads OSPREY config directly (get_facility_timezone while
# building the agent system prompt) with CWD=/app (the image WORKDIR), so without
# CONFIG_FILE it falls back to /app/config.yml and every dispatch errors with
# "No config.yml found in current directory: /app".
#
# The path is the RENDER zone under the container's project root, because the
# worker runs the project image: its .mcp.json is rendered as
# `{project_root}/build/config.yml` (registry/mcp.py), so a flat config.yml here
# would be a file the agent's own MCP servers never read.
def test_worker_template_sets_config_file() -> None:
    rendered = _render_worker_template(env_present=True)
    assert f"CONFIG_FILE: /app/{_WORKER_PROJECT_NAME}/{RENDERED_CONFIG_RELPATH}" in rendered, (
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
# ``osprey up``), which bakes the project at ``/app/<project>``
# (Dockerfile ``COPY . /app/{{ project_name }}/``, ``WORKDIR /app/<project>``,
# ``chown -R osprey:osprey /app/<project>``). Every worker path — OSPREY_PROJECT_DIR,
# CONFIG_FILE, the staged config bind-mount, the .env mount, the agent-data volume
# — must point at that same ``/app/<project>`` root, or the worker points at an
# empty/absent directory (plan risk M1). The image tag prefix, the label project
# name, and the layout path all derive from one ``resolve_project_name(config)``
# call in ``_inject_project_metadata``, so they are provably the same string.
# ---------------------------------------------------------------------------


def test_worker_image_defaults_to_project_local_when_override_unset() -> None:
    """With OSPREY_WORKER_IMAGE unset, the worker image resolves to <project>:local.

    ``_inject_project_metadata`` defaults ``services.dispatch_worker.image`` to
    ``<project>:local`` (the tag ``osprey up`` builds), so the rendered
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

    # Env: project dir + config file. The config sits in the render zone under
    # that root, mirroring the repo's own three-zone shape — which is what the
    # image's .mcp.json resolves `{project_root}/build/config.yml` to.
    assert f"OSPREY_PROJECT_DIR: {root}" in rendered
    assert f"CONFIG_FILE: {root}/{RENDERED_CONFIG_RELPATH}" in rendered

    # Staged config bind-mount (the deploy-time config): rendered source under
    # build/services/, overlaying the image's own copy at the same path the
    # container's OSPREY_CONFIG names.
    assert (
        f"- ./build/services/dispatch_worker/config.yml:{root}/{RENDERED_CONFIG_RELPATH}:ro"
        in rendered
    )

    # Provider auth is delivered via env_file (host-side read), not a bind
    # mount, so there is no `/app/<project>/.env` path in the container layout.
    assert f"{root}/.env" not in rendered, (
        "the worker must not reference /app/<project>/.env — env_file: delivers "
        "provider auth without exposing the file inside the (non-root) container"
    )

    # Agent-data named-volume mount target (default isolated mode -> per-worker).
    # The in-container directory is the config's own ``agent_data.base_dir``, so
    # the volume lands exactly where the worker process writes.
    assert f"- dispatch_workspace_1:{root}/{DEFAULT_AGENT_DATA_BASE_DIR}" in rendered

    # No stale hardcoded /app/project layout may survive anywhere.
    assert "/app/project" not in rendered, (
        "the worker template must not retain the hardcoded /app/project layout"
    )


def test_worker_agent_data_volume_shared_mode_targets_project_layout() -> None:
    """In shared workspace mode the single ``dispatch_workspace`` volume must also
    mount at the agent-data root under ``/app/<project>``."""
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
    root = f"/app/{_WORKER_PROJECT_NAME}/{DEFAULT_AGENT_DATA_BASE_DIR}"
    assert f"- dispatch_workspace:{root}" in shared
    # The isolated-mode default (from the other fixture) uses the per-worker name.
    assert f"- dispatch_workspace_1:{root}" in rendered


def test_worker_agent_data_volume_honors_an_absolute_base_dir() -> None:
    """An absolute ``agent_data.base_dir`` names the same path inside the container.

    It must therefore be the mount target verbatim, NOT re-anchored under
    ``/app/<project>``. The failure this pins is silent in the worst way: the
    volume mounts at ``/app/<project>/data/agent`` while the worker writes to
    ``/data/agent`` — a plain directory in the container's writable layer, so
    every dispatch run's records are discarded at the next recreate with nothing
    logged at mount time to say so.
    """
    from importlib import resources

    from jinja2 import Template

    from osprey.deployment.compose_generator import _inject_project_metadata

    config = _inject_project_metadata(
        {
            "project_name": _WORKER_PROJECT_NAME,
            "project_root": f"/r/{_WORKER_PROJECT_NAME}",
            "agent_data": {"base_dir": "/data/agent"},
            "services": {"dispatch_worker": {}},
            "system": {"timezone": "UTC"},
        }
    )
    config["osprey_env_present"] = True
    tpl = resources.files("osprey").joinpath(
        "templates/services/dispatch_worker/docker-compose.yml.j2"
    )
    rendered = Template(tpl.read_text(encoding="utf-8")).render(**config)

    assert "- dispatch_workspace_1:/data/agent" in rendered
    assert f"/app/{_WORKER_PROJECT_NAME}//data/agent" not in rendered, (
        "an absolute base_dir must not be re-anchored under the project directory"
    )


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
    from osprey.deployment.errors import DevModeUnavailableError

    captured: dict = {}

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        # Return non-zero so the function bails before trying to copy a wheel.
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="stop here")

    monkeypatch.setattr(compose_generator.subprocess, "run", _fake_run)
    # A failed build now aborts the deploy rather than falling back to the
    # released package; the interpreter assertion below is what this test is for.
    with pytest.raises(DevModeUnavailableError):
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
    # Scoped to the VA specifically, not to `depends_on:` as a whole: the
    # queueserver service always waits on Redis, so a file-wide check for
    # `depends_on:` would fail on every render regardless of the VA.
    assert "virtual-accelerator" not in without_va


@pytest.mark.parametrize("va_deployed", [True, False])
def test_bluesky_bridge_waits_for_the_queueserver_to_answer(va_deployed: bool) -> None:
    """The bridge must start only after ``qserver ping`` answers — with or
    without the VA co-deployed.

    Not cosmetic ordering. The bridge opens the RE worker environment once at
    startup (``app.py``'s ``_open_environment_at_startup``), and
    ``ensure_environment`` gives that up WITHOUT retrying when ``capability()``
    reports ``manager_unreachable`` — it re-runs only on an armed
    ``POST /queue/start``. Since ``POST /queue/items`` validates against
    ``plans_allowed``, which the manager downloads from the worker at
    environment open, a bridge that wins the boot race against the manager
    refuses every enqueue with "not in the list of allowed plans" and no start
    ever gets the chance to self-heal it. Only container ordering closes that,
    so it is asserted here rather than left to whichever process imports its
    dependency stack faster.
    """
    bridge = yaml.safe_load(_render_bluesky_template(va_deployed=va_deployed))["services"][
        "bluesky-bridge"
    ]
    assert bridge["depends_on"]["queueserver"] == {"condition": "service_healthy"}


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
    assert "./build/services/bluesky/config.yml:/app/project/config.yml:ro" in rendered

    # And it must still be present when writes ARE enabled (the two mounts
    # are independent, not mutually exclusive).
    rendered_writable = _render_bluesky_template(va_deployed=False, writes_enabled=True)
    assert "./build/services/bluesky/config.yml:/app/project/config.yml:ro" in rendered_writable


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
    assert "./data/channel_limits.json:/app/project/data/channel_limits.json:ro" in rendered


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
    # "bluesky_tiled_catalog:/data" pins the volume MOUNT (the actual root
    # cause — the line has no trailing slash, so a bare "/data/" check would
    # miss it entirely). The named volume has to be part of the needle: Redis
    # legitimately mounts at ":/data", so a bare ":/data" check would fire on
    # every render.
    # The three per-argument needles pin the COMMAND paths (catalog.db,
    # files, duckdb target). A single bare "/data/" check would be shorter
    # but now false-positives on the CURVE certificate bind sources
    # ("./data/bluesky_curve/..."), which have nothing to do with Tiled.
    # Bare "/data" isn't usable anywhere here: it also false-positives on
    # "duckdb:////storage/data.duckdb", whose filename legitimately
    # contains "data" as a substring of "storage".
    assert "bluesky_tiled_catalog:/data" not in rendered
    assert "serve catalog /data/" not in rendered
    assert "-w /data/" not in rendered
    assert "duckdb:////data/" not in rendered

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
# control_system.type=virtual_accelerator and the scan MCP server enabled
# (BLUESKY_LAUNCH_TOKEN is minted unconditionally for the deployed bluesky
# service, so no execution-method override is needed to arm the agent).
# tests/e2e/_orm_stack.py is the single source of this
# config, reused by the real-container round-trip e2e (task 5.2) and the
# agentic-discovery e2e (5.3/5.4) -- this gate only exercises the Docker-free
# render path via its in-process `osprey build` helper.
# ---------------------------------------------------------------------------


def test_orm_stack_renders_va_bridge_tiled_and_scan_mcp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FR11's turn-key deploy config, end to end without Docker:
    ``osprey build`` (in-process, via ``tests/e2e/_orm_stack``) followed by a
    Docker-free compose render, must produce:

      - the Virtual Accelerator + bluesky-bridge + co-deployed Tiled compose
        services (``control_system.type=virtual_accelerator`` +
        ``bluesky.tiled_enabled=true``),
      - ``execution.execution_method: subprocess`` (the one execution backend
        OSPREY ships — the deploy config sets no override, so this is the
        rendered default, and no code path reads it for safety semantics),
      - the ``scan`` MCP server enabled in the rendered ``.mcp.json`` (it is
        ``default_enabled=False`` in the framework registry — a project must
        opt in, and this deploy config does).
    """
    import json

    from click.testing import CliRunner

    from tests.e2e import _orm_stack

    runner = CliRunner()
    project_dir = _orm_stack.build_via_cli_runner(runner, tmp_path)

    # -- execution_method: subprocess (the only backend OSPREY ships) --------
    yaml = YAML()
    with open(project_dir / "config.yml") as fh:
        config = yaml.load(fh)
    assert config["execution"]["execution_method"] == "subprocess", (
        "the rendered config must name the backend that actually runs agent "
        "Python; the deploy config overrides nothing here"
    )
    assert config["control_system"]["type"] == "virtual_accelerator"

    # -- scan MCP server enabled in the rendered .mcp.json -------------------
    mcp_config = json.loads((project_dir / ".mcp.json").read_text(encoding="utf-8"))
    assert "bluesky" in mcp_config["mcpServers"], (
        "the scan MCP server must be enabled (claude_code.servers.bluesky.enabled: "
        f"true) so list_plans/queue_add are reachable: {mcp_config['mcpServers'].keys()}"
    )

    # -- VA + bridge + Tiled compose services --------------------------------
    monkeypatch.chdir(project_dir)
    _, compose_files = prepare_compose_files(str(project_dir / "config.yml"))
    # Read while still inside project_dir — prepare_compose_files returns
    # paths relative to it (SERVICES_DIR resolves relative to cwd).
    rendered = "\n".join(Path(f).read_text(encoding="utf-8") for f in compose_files)

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
    assert "./build/services/bluesky/config.yml:/app/project/config.yml:ro" in rendered, (
        "bridge must mount config.yml read-only under /app/project (Task 3.2)"
    )
    assert "./data/channel_limits.json:/app/project/data/channel_limits.json:ro" in rendered, (
        "control_system.writes_enabled=true (preset default) must mount "
        "channel_limits.json under the same /app/project root as config.yml"
    )


# ---------------------------------------------------------------------------
# Container config staging: no host interpreter can reach the container
#
# The M2 concern: the dispatch worker's runtime config.yml must never carry
# the HOST build machine's interpreter (e.g. ``/Users/.../.venv/bin/python``)
# into the container — that path does not exist in-container, and MCP-server
# command generation used to prefer it over ``sys.executable``, so every MCP
# server would fail to launch. Two properties now make that unreachable, with
# no staging-time surgery:
#
# 1. A generated project's config records no interpreter at all — there is no
#    ``execution.python_env_path`` for staging to carry across. (An older
#    config that still carries the retired key stages verbatim, but
#    ``ConfigBuilder`` drops it on load.)
# 2. ``build_claude_code_context`` (osprey.cli.templates.claude_code) derives
#    ``current_python_env`` from the filesystem — the project's own ``.venv``
#    when it has one, else the generating interpreter — and consults the config
#    not at all. A ``project_root_override`` (the container case) forces the
#    generating interpreter, since the host's venv is not what exists in the
#    container. So no config, of any vintage, can name the interpreter an MCP
#    server launches with.
#
# ``setup_build_dir`` therefore stages the flattened config as-is; the strip it
# used to perform removed a key nothing writes and nothing reads. These tests
# prove the invariant against the real generator entrypoints. The full
# role-split contract (all three runtime launch sites, driven with a config
# that points somewhere else) lives in tests/cli/test_interpreter_role_split.py.
# ---------------------------------------------------------------------------


def test_setup_build_dir_stages_a_config_naming_no_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The config staged for a service's bind-mount must name no interpreter.

    Drives the real staging code path against a real generated project: its
    ``config.yml`` is loaded internally via ``ConfigBuilder()`` (which resolves
    against ``os.getcwd()``), flattened, and written to
    ``<build_dir>/<service_dir>/config.yml``. Neither the retired
    ``execution.python_env_path`` key nor any interpreter path may appear —
    and nothing strips them, because nothing writes them.
    """
    from osprey.cli.templates.manager import TemplateManager
    from osprey.deployment.compose_generator import setup_build_dir

    project_dir = TemplateManager().create_project(
        project_name="staged-no-interp",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )

    service_dir = project_dir / "services" / "worker"
    service_dir.mkdir(parents=True)
    (service_dir / "docker-compose.yml.j2").write_text("services:\n  worker:\n    image: test\n")

    monkeypatch.chdir(project_dir)

    template_path = str(Path("services") / "worker" / "docker-compose.yml.j2")
    config = {"project_name": "staged-no-interp", "build_dir": "./build"}
    container_cfg = {"copy_src": False, "render_kernel_templates": False}

    setup_build_dir(template_path, config, container_cfg)

    staged_config_path = project_dir / "build" / "services" / "worker" / "config.yml"
    assert staged_config_path.is_file(), (
        f"expected a staged config.yml at {staged_config_path} "
        "(flattening must have failed and fallen back to a verbatim copy)"
    )
    staged_text = staged_config_path.read_text()
    staged_config = yaml.safe_load(staged_text)
    assert "execution_method" in staged_config.get("execution", {}), (
        "the staged config must carry a real, flattened execution block — "
        "without one the interpreter assertions below would be vacuous"
    )
    assert "python_env_path" not in staged_config["execution"], (
        f"an interpreter path reached the staged config: {staged_config.get('execution')}"
    )
    assert sys.executable not in staged_text, (
        "the staging interpreter must not appear anywhere in the staged config"
    )


def test_setup_build_dir_stages_the_execution_block_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Staging edits no key under ``execution``.

    The strip that used to run here is gone, so a legacy config carrying the
    retired ``execution.python_env_path`` stages unchanged. That is safe rather
    than a regression: the loader drops the key and MCP-server generation never
    reads it (see the companion tests below), so the value is inert wherever it
    lands. What matters is that staging does not quietly lose sibling keys.
    """
    from osprey.deployment.compose_generator import setup_build_dir

    execution_block = {
        "python_env_path": "/Users/someone/.venv/bin/python3.11",
        "execution_method": "subprocess",
    }

    project_config_path = tmp_path / "config.yml"
    project_config_path.write_text(
        yaml.dump({"project_name": "pep-fixture-2", "execution": dict(execution_block)})
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
    assert staged_config["execution"] == execution_block, (
        "staging must pass the execution block through untouched"
    )


def test_missing_python_env_path_falls_back_to_sys_executable() -> None:
    """The real ``.mcp.json`` generation seam: with no interpreter recorded in
    the config (exactly what staging produces today), MCP-server commands must
    resolve to the CONTAINER's own ``sys.executable``, never a host path.

    Drives ``build_claude_code_context`` (the actual context-builder used by
    ``osprey build``) followed by
    ``resolve_servers``'s real command resolution, rather than asserting a
    single expression in isolation.
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


def test_host_python_env_path_cannot_bake_host_interpreter_into_mcp_command() -> None:
    """Companion to the above: the M2 failure mode is now unreachable by design.

    A host-looking ``python_env_path`` that survived staging used to be baked
    into every MCP server's ``command``. The generator no longer reads the key,
    so even a config that still carries it — exactly what an already-deployed
    project looks like — yields the container's own interpreter. This is what
    lets ``setup_build_dir`` stage the config untouched: nothing stands between
    the host path and a broken container because nothing consumes the path.
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

        assert ctx["current_python_env"] == sys.executable

        controls_server = next(s for s in ctx["servers"] if s["name"] == "controls")
        assert controls_server["command"] == sys.executable
        assert controls_server["command"] != host_python


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
    """Return the openobserve tag the compose template pins.

    The image line follows the uniform env → config → default chain
    (``${OSPREY_OPENOBSERVE_IMAGE:-{{ … | default('<ref>:<tag>') }}}``); the
    pinned tag lives in the innermost Jinja ``default('…')``.
    """
    text = _OPENOBSERVE_TEMPLATE.read_text(encoding="utf-8")
    match = re.search(
        r"default\('" + re.escape(_OPENOBSERVE_IMAGE_REF) + r":([^')\s]+)'\)",
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


def _render_project_compose(
    config_path: Path, project_path: Path, monkeypatch: pytest.MonkeyPatch
) -> str:
    """Copy service templates, render via ``prepare_compose_files``, return the text.

    Runs the real CLI gating path from inside the project dir (SERVICES_DIR and
    the service ``path`` both resolve relative to cwd), then joins every rendered
    compose file so callers can assert on the aggregate text. The caller's
    ``monkeypatch`` owns the cwd change, so it is undone at test teardown.
    """
    _copy_service_templates(project_path)

    monkeypatch.chdir(project_path)
    _, compose_files = prepare_compose_files(str(config_path))
    return "\n".join(Path(f).read_text(encoding="utf-8") for f in compose_files)


def test_openobserve_renders_when_deployed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``openobserve`` in deployed_services the compose renders (exit 0) with
    the expected image reference, port, named volume, and root-cred env vars.

    The service name, port, and env var names are the shared constants the
    telemetry resolver stream also depends on — a drift here silently breaks the
    agent's OTLP push against the local store.
    """
    config_path = _write_openobserve_config(tmp_path, deployed_services=["openobserve"])
    rendered = _render_project_compose(config_path, tmp_path, monkeypatch)

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


def test_openobserve_retention_env_default_rendered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitting retention_days renders the 14-day growth bound (the compose default)."""
    config_path = _write_openobserve_config(tmp_path, deployed_services=["openobserve"])
    rendered = _render_project_compose(config_path, tmp_path, monkeypatch)
    assert 'ZO_COMPACT_DATA_RETENTION_DAYS: "14"' in rendered


def test_openobserve_retention_env_custom_rendered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configured retention_days flows into ZO_COMPACT_DATA_RETENTION_DAYS."""
    config_path = _write_openobserve_config(
        tmp_path, deployed_services=["openobserve"], retention_days=30
    )
    rendered = _render_project_compose(config_path, tmp_path, monkeypatch)
    assert 'ZO_COMPACT_DATA_RETENTION_DAYS: "30"' in rendered

    # Named data volume for persistence (both the mount and the top-level decl).
    assert "openobserve_data:/data" in rendered
    assert "\nvolumes:\n  openobserve_data:\n" in rendered

    # Root credentials sourced from compose env with overridable defaults, using
    # the exact env var names the resolver expects.
    assert "ZO_ROOT_USER_EMAIL: ${ZO_ROOT_USER_EMAIL:-" in rendered
    assert "ZO_ROOT_USER_PASSWORD: ${ZO_ROOT_USER_PASSWORD:-" in rendered


def test_openobserve_absent_when_not_deployed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``openobserve`` declared but NOT in deployed_services it must not
    render — the opt-in posture. Only the root compose is produced, with no
    openobserve service, image, or volume anywhere in the output.
    """
    config_path = _write_openobserve_config(tmp_path, deployed_services=[])
    rendered = _render_project_compose(config_path, tmp_path, monkeypatch)

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


def test_find_service_config_resolves_flat_names_only() -> None:
    """Services resolve only by their flat short name under ``services:``. Dotted
    spellings (``osprey.<name>`` / ``applications.<app>.<name>``) are not a
    supported config shape — nothing populates a nested ``osprey:``/
    ``applications:`` services block — so they resolve to ``(None, None)``, which
    callers surface as a named "service not found" error.
    """
    from osprey.deployment.compose_generator import find_service_config

    config = {"services": {"openobserve": {"path": "./services/openobserve"}}}

    assert find_service_config(config, "openobserve")[0] is not None
    assert find_service_config(config, "osprey.openobserve") == (None, None)
    assert find_service_config(config, "applications.app.openobserve") == (None, None)


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


def test_postgres_image_follows_env_config_default_chain() -> None:
    """The postgres image is overridable like every other service image.

    Uniform env → config → default chain: OSPREY_POSTGRES_IMAGE wins, then
    services.postgresql.image, then the pinned pgvector default. A hard pin
    forces air-gapped/mirrored registries to fork the template.
    """
    svc = yaml.safe_load(_render_postgres_template("proj-a"))["services"]["postgresql"]
    assert svc["image"] == "${OSPREY_POSTGRES_IMAGE:-pgvector/pgvector:pg16}"

    from importlib import resources

    from jinja2 import Template

    tpl = resources.files("osprey").joinpath("templates/services/postgresql/docker-compose.yml.j2")
    rendered = Template(tpl.read_text(encoding="utf-8")).render(
        services={"postgresql": {"port_host": 5432, "image": "registry.local/pg:custom"}},
        deployment={},
        system={"timezone": "UTC"},
        osprey_labels={"project_name": "p", "project_root": "/r/p", "deployed_at": "now"},
        osprey_version="",
    )
    svc = yaml.safe_load(rendered)["services"]["postgresql"]
    assert svc["image"] == "${OSPREY_POSTGRES_IMAGE:-registry.local/pg:custom}"


def test_postgres_password_reads_minted_env_var() -> None:
    """POSTGRES_PASSWORD sources ARIEL_DB_PASSWORD from .env (minted by deploy
    up), falling back to the legacy config key, then the dev default — the
    same single-source convention as openobserve's ZO_ROOT_USER_PASSWORD."""
    svc = yaml.safe_load(_render_postgres_template("proj-a"))["services"]["postgresql"]
    assert svc["environment"]["POSTGRES_PASSWORD"] == "${ARIEL_DB_PASSWORD:-ariel}"


# ---------------------------------------------------------------------------
# mongodb: the archiver store
#
# Same shape as postgres — a pulled upstream image holding a persistent store —
# and the same three host-global hazards: a bare container_name collides across
# projects, in-network consumers (the recorder, the dispatch worker's archiver
# host override) need a name that survives that namespacing, and the credential
# has exactly one home (the minted MONGO_ROOT_PASSWORD in the project .env).
# The one thing postgres has no analogue for is the compression knob: block
# compression is set on the SERVER because the seeder creates the collection
# implicitly, so the knob has to reach mongod's own argv.
# ---------------------------------------------------------------------------
def _render_mongodb_template(project_name: str = "proj-a", **mongodb_config: object) -> str:
    from importlib import resources

    from jinja2 import Template

    tpl = resources.files("osprey").joinpath("templates/services/mongodb/docker-compose.yml.j2")
    template = Template(tpl.read_text(encoding="utf-8"))
    return template.render(
        services={"mongodb": mongodb_config},
        deployment={},
        system={"timezone": "UTC"},
        osprey_labels={
            "project_name": project_name,
            "project_root": f"/r/{project_name}",
            "deployed_at": "now",
        },
        osprey_version="",
    )


def _mongodb_service(project_name: str = "proj-a", **mongodb_config: object) -> dict:
    return yaml.safe_load(_render_mongodb_template(project_name, **mongodb_config))["services"][
        "mongodb"
    ]


def test_mongodb_container_name_is_project_namespaced() -> None:
    """Two projects render distinct, project-scoped mongo container names.

    `container_name` is host-global, so a bare `archiver-mongodb` would make two
    OSPREY projects deploying the archiver on one host collide and serialize.
    """
    name_a = _mongodb_service("proj-a")["container_name"]
    name_b = _mongodb_service("proj-b")["container_name"]

    assert name_a == "proj-a-archiver-mongodb", name_a
    assert name_b == "proj-b-archiver-mongodb", name_b


def test_mongodb_publishes_a_stable_in_network_alias() -> None:
    """`archiver-mongodb` stays resolvable in-network after the namespacing.

    The recorder writes to this host from inside the compose network (the
    host-published port is not reachable there), and the dispatch worker points
    the agent's archiver at the same name. Both break with "could not resolve
    host" if the alias goes away.
    """
    svc = _mongodb_service()
    aliases = svc["networks"]["osprey-network"]["aliases"]
    assert "archiver-mongodb" in aliases, aliases


def test_mongodb_image_follows_env_config_default_chain() -> None:
    """env → config → pinned default, like every other pulled service image.

    A hard pin would force air-gapped/mirrored registries to fork the template.
    """
    assert _mongodb_service()["image"] == "${OSPREY_MONGODB_IMAGE:-mongo:7}"
    assert (
        _mongodb_service(image="registry.local/mongo:custom")["image"]
        == "${OSPREY_MONGODB_IMAGE:-registry.local/mongo:custom}"
    )


def test_mongodb_root_password_reads_minted_env_var() -> None:
    """The root password sources MONGO_ROOT_PASSWORD from .env (minted by
    `osprey up`), the same single-source convention as postgres's
    ARIEL_DB_PASSWORD — one value for the container, the seeder, the recorder
    and the agent connector's `password_env`."""
    env = _mongodb_service()["environment"]
    assert env["MONGO_INITDB_ROOT_PASSWORD"] == "${MONGO_ROOT_PASSWORD:-osprey}"
    assert env["MONGO_INITDB_ROOT_USERNAME"] == "osprey"


def test_mongodb_block_compression_is_a_knob_on_mongod_argv() -> None:
    """Compression reaches mongod's own arguments, defaulting to zstd.

    Setting it server-side is what makes the seeder's implicitly created
    collection inherit it; a per-collection option would leave a window in
    which documents land under the default codec.
    """
    assert _mongodb_service()["command"] == ["--wiredTigerCollectionBlockCompressor", "zstd"]
    assert _mongodb_service(compression="snappy")["command"] == [
        "--wiredTigerCollectionBlockCompressor",
        "snappy",
    ]


def test_mongodb_port_publish_follows_bind_address_and_port_host() -> None:
    """The host publish honors `services.mongodb.port_host` and the deploy-wide
    bind address, defaulting to loopback:27017 — the address the host-side
    seeder and the agent connector both use."""
    assert _mongodb_service()["ports"] == ["127.0.0.1:27017:27017"]
    assert _mongodb_service(port_host=27117)["ports"] == ["127.0.0.1:27117:27017"]


def test_mongodb_healthcheck_pings_without_credentials() -> None:
    """The probe is an unauthenticated `ping`, so the minted password has
    exactly one spelling in this file. `osprey up` gates the base seed on this
    healthcheck, so it has to answer on a fresh volume too."""
    healthcheck = _mongodb_service()["healthcheck"]
    assert healthcheck["test"] == [
        "CMD-SHELL",
        "mongosh --quiet --eval 'db.adminCommand(\"ping\")'",
    ]
    assert healthcheck["start_period"] == "15s", (
        "a fresh volume creates the admin user and preallocates the journal "
        "before mongod answers; without a grace period those attempts burn the "
        "retry budget"
    )


def test_mongodb_store_is_a_named_volume() -> None:
    """Seeded + recorded history persists across `osprey down`: a base seed
    costs minutes, and a store that resets on restart would be younger than the
    machine it claims to describe."""
    rendered = yaml.safe_load(_render_mongodb_template())
    assert rendered["services"]["mongodb"]["volumes"] == ["archiver_mongodb_data:/data/db"]
    assert "archiver_mongodb_data" in rendered["volumes"], (
        "the store volume must be declared (named), never an anonymous mount"
    )


# ---------------------------------------------------------------------------
# archiver_recorder: the live-sampling half of the archiver
#
# A compose-template-only service — it runs the VIRTUAL ACCELERATOR's image
# with a different command, so it adds no image build to any deploy or CI lane.
# Three properties carry the design and each has a way of failing quietly:
#
# * it must not start before the store answers (a fresh mongo volume makes that
#   window seconds long),
# * everything cross-service is gated on ``deployed_services`` membership —
#   compose errors outright on a ``depends_on`` naming an undefined service,
#   and the CA/image settings the co-deployed branch derives have to be
#   supplied by the operator otherwise, or the recorder comes up routed
#   nowhere and times out looking like a slow machine, and
# * its Mongo address must be the in-network one: config.yml carries the HOST
#   view (localhost + the published port_host), which inside the network is
#   this container's own loopback.
# ---------------------------------------------------------------------------
_RECORDER_TEMPLATE = "archiver_recorder/docker-compose.yml.j2"


def _render_recorder_template(*, va_co_deployed: bool, project_name: str = "proj-a") -> str:
    deployed = ["mongodb", "archiver_recorder"]
    if va_co_deployed:
        deployed.append("virtual_accelerator")
    return _render_service_template(_RECORDER_TEMPLATE, project_name, deployed_services=deployed)


def _recorder_service(*, va_co_deployed: bool, project_name: str = "proj-a") -> dict:
    rendered = _render_recorder_template(va_co_deployed=va_co_deployed, project_name=project_name)
    return yaml.safe_load(rendered)["services"]["archiver-recorder"]


@pytest.mark.parametrize("va_co_deployed", [True, False])
def test_recorder_declares_no_build_context(va_co_deployed: bool) -> None:
    """The recorder never builds an image.

    It runs the VA's image with a different command, which is what keeps it out
    of the seven VA-stack CI lanes' build cost — and what stops two services
    racing to tag one image, the same rule the bluesky queueserver follows.
    """
    assert "build" not in _recorder_service(va_co_deployed=va_co_deployed)


def test_recorder_reuses_the_va_image_when_co_deployed() -> None:
    """Co-deployed: the recorder renders byte-identically to the VA's own image
    reference, so the two can never run different Channel Access stacks."""
    recorder = _recorder_service(va_co_deployed=True)["image"]
    va = yaml.safe_load(
        _render_service_template("virtual_accelerator/docker-compose.yml.j2", "proj-a")
    )["services"]["virtual-accelerator"]["image"]
    assert recorder == va == "${OSPREY_VA_IMAGE:-proj-a-va:local}"


def test_recorder_requires_an_explicit_image_without_a_co_deployed_va() -> None:
    """Without the VA, nothing in the deploy builds that tag.

    Compose would fail pulling ``<project>-va:local`` with "pull access denied"
    and never name the real problem, so the reference is required (``:?``)
    instead of defaulted.
    """
    image = _recorder_service(va_co_deployed=False)["image"]
    assert image.startswith("${OSPREY_VA_IMAGE:?"), image


def test_recorder_command_runs_the_recorder_module() -> None:
    """The command replaces the VA image's CMD outright (no ENTRYPOINT), so the
    same image serves the IOC in one container and the recorder in another."""
    assert _recorder_service(va_co_deployed=True)["command"] == [
        "python",
        "-u",
        "-m",
        "osprey.services.archiver_recorder",
    ]


@pytest.mark.parametrize("va_co_deployed", [True, False])
def test_recorder_always_waits_for_a_healthy_store(va_co_deployed: bool) -> None:
    """The store dependency is unconditional and health-gated.

    "Container started" is not "answering commands": on a fresh volume mongod
    creates the admin user and preallocates its journal first, and a recorder
    writing into that window fails its first inserts.
    """
    depends = _recorder_service(va_co_deployed=va_co_deployed)["depends_on"]
    assert depends["mongodb"] == {"condition": "service_healthy"}


def test_recorder_wires_va_ordering_and_ca_env_only_when_va_co_deployed() -> None:
    """Co-deployed: wait on the IOC's health and derive its CA address.

    The address is derived from ``services.virtual_accelerator.port`` (not
    hardcoded) so an operator moving the VA's port moves both services at once,
    and ``depends_on`` must be absent when the VA is external — compose errors
    on a dependency naming an undefined service.
    """
    svc = _recorder_service(va_co_deployed=True)
    assert svc["depends_on"]["virtual-accelerator"] == {"condition": "service_healthy"}
    assert svc["environment"]["EPICS_CA_NAME_SERVERS"] == "virtual-accelerator:5064"
    assert svc["environment"]["EPICS_CA_AUTO_ADDR_LIST"] == "NO"

    external = _recorder_service(va_co_deployed=False)
    assert "virtual-accelerator" not in external["depends_on"]


def test_recorder_requires_a_ca_address_when_the_va_is_external() -> None:
    """An unset bare ``${VAR}`` resolves to "", which would leave the recorder
    routed nowhere: every read times out, for minutes, looking exactly like a
    slow machine. ``:?`` makes it a startup abort naming the variable."""
    env = _recorder_service(va_co_deployed=False)["environment"]
    assert env["EPICS_CA_NAME_SERVERS"].startswith("${EPICS_CA_NAME_SERVERS:?"), env[
        "EPICS_CA_NAME_SERVERS"
    ]
    assert env["EPICS_CA_AUTO_ADDR_LIST"] == "${EPICS_CA_AUTO_ADDR_LIST:-NO}"


def test_recorder_addresses_the_store_in_network_not_on_the_host() -> None:
    """The archiver host/port overrides point at the store's network alias and
    its CONTAINER port.

    config.yml carries the HOST view (localhost + the published ``port_host``);
    used verbatim in-network that is this container's own loopback. The alias
    (not the compose service key, and never the container_name) is what stays
    resolvable after the per-project container_name namespacing.
    """
    env = _recorder_service(va_co_deployed=True)["environment"]
    assert env["OSPREY_ARCHIVER_MONGODB_HOST"] == "archiver-mongodb"
    assert env["OSPREY_ARCHIVER_MONGODB_PORT"] == "27017"


def test_recorder_and_store_agree_on_the_mongo_password_fallback() -> None:
    """Both halves must read the same variable AND fall back to the same value.

    They are two spellings of one credential: if the store defaults the root
    password and the recorder defaults something else, a deploy that never
    minted the secret comes up with a store the recorder cannot authenticate
    against — and the only symptom is an archive that stops growing.
    """
    recorder = _recorder_service(va_co_deployed=True)["environment"]["MONGO_ROOT_PASSWORD"]
    store = _mongodb_service()["environment"]["MONGO_INITDB_ROOT_PASSWORD"]
    assert recorder == store == "${MONGO_ROOT_PASSWORD:-osprey}"


def test_recorder_reads_the_channel_manifest_the_va_serves() -> None:
    """The channel list is the same build-derived variable the VA reads,
    resolved against the same mount path.

    It must be the manifest. ``channel_limits.json`` is a *write-safety
    projection* of that manifest, not a second copy of it: it carries one entry
    per address (read-only ones included — the DCCT current sits there as
    ``{"writable": false}``), plus top-level metadata keys ``_comment``,
    ``_version``, ``_description`` and ``defaults``. So its key set is not a
    channel list, and reading it as one would hand the recorder four names no
    IOC serves. The manifest is the single channel source the IOC and the
    recorder have to share; anything else is a second source free to drift.
    """
    svc = _recorder_service(va_co_deployed=True)
    assert svc["environment"]["VA_CHANNELS_FILE"] == "${VA_CHANNELS_FILE:-}"
    assert "./build/data/simulation:/data/simulation:ro" in svc["volumes"]
    assert not any("channel_limits" in mount for mount in svc["volumes"]), svc["volumes"]


def test_recorder_reads_config_yml_from_a_read_only_mount() -> None:
    """CONFIG_FILE points at a mounted file, not baked env: the enablement poll
    re-reads it, which is what lets the documented ``control_system.type`` flip
    take effect without a restart. CWD is the image WORKDIR, so without
    CONFIG_FILE every lookup errors "No config.yml found".

    The mount is the REPO ROOT, deliberately unlike the bluesky bridge's staged
    copy: a copy is rewritten only by ``osprey build``, so the poll would
    re-read a stale copy and the flip would silently need a rebuild. It is a
    directory rather than the single file because a single-file bind pins an
    inode at container start, and editors that save by rename leave a new one
    behind — see the template's mount comment.
    """
    svc = _recorder_service(va_co_deployed=True)
    assert svc["environment"]["CONFIG_FILE"] == "/app/project/build/config.yml"
    assert ".:/app/project:ro" in svc["volumes"]
    assert not any("archiver_recorder/config.yml" in mount for mount in svc["volumes"]), svc[
        "volumes"
    ]


def test_recorder_publishes_no_ports_declares_no_healthcheck_and_reads_no_bulk_env() -> None:
    """It opens no listening socket (nothing to publish, nothing to probe) and
    calls no LLM (so no bulk ``.env`` passthrough — everything it reads is
    interpolated explicitly). "Is history still arriving?" is answered by the
    ``archiver_freshness`` probe against the store, not by a container probe."""
    svc = _recorder_service(va_co_deployed=True)
    assert "ports" not in svc
    assert "healthcheck" not in svc
    assert "env_file" not in svc


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
            # Both bridge templates read their trigger with no fallback, so the
            # shared ctx must declare the blocks or every render through here
            # raises UndefinedError on `services.<bridge>`.
            "nextcloud_bridge": {"trigger": "t"},
            "gchat_bridge": {"trigger": "t"},
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
    # Reached by nobody in-network (outbound CA reads and Mongo writes only),
    # so it needs no alias either.
    ("archiver_recorder/docker-compose.yml.j2", "archiver-recorder", "archiver-recorder", {}),
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
# OSPREY_PROJECT_NAME (always) and OSPREY_DEV=1 (iff `osprey up --dev`,
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
    (
        "nextcloud_bridge/docker-compose.yml.j2",
        "nextcloud-bridge",
        "OSPREY_NEXTCLOUD_BRIDGE_IMAGE",
        "nextcloud-bridge",
    ),
    (
        "gchat_bridge/docker-compose.yml.j2",
        "gchat-bridge",
        "OSPREY_GCHAT_BRIDGE_IMAGE",
        "gchat-bridge",
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
# The worker runs the PROJECT image (<project>:local — built by `osprey up`
# from the project Dockerfile), never the dispatcher's image. Its rendered
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
_FIXTURE_WHEEL_EXPECTED_MANIFEST = 'aiohttp\nsoftioc>=4.5\ntomli>=2; python_version < "3.11"\n'


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

    from osprey.deployment import compose_generator

    calls: list = []
    real_run = subprocess_module.run

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(cmd, list) and cmd[1:3] == ["-m", "build"]:
            calls.append(list(cmd))
            outdir = cmd[cmd.index("--outdir") + 1]
            _write_fixture_wheel(Path(outdir, "osprey_framework-0.0.0-py3-none-any.whl"))
            return subprocess_module.CompletedProcess(cmd, 0, stdout="", stderr="")
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(compose_generator.subprocess, "run", _fake_run)
    return calls


def test_dev_wheel_builds_once_across_service_and_project_staging(
    tmp_path: Path, spy_wheel_build: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One process, many staging targets, exactly ONE ``python -m build``.

    Drives the real service-context staging (``prepare_compose_files`` with
    dev_mode) plus two further ``_copy_local_framework_for_override`` calls
    with distinct out_dirs — the same helper the project-image build
    (``_build_project_image``) and persona builds (``build_persona_images``)
    invoke against their own contexts. Every context must still receive its
    own wheel copy (the memo caches the BUILD, not the copy).
    """
    from osprey.deployment.compose_generator import _copy_local_framework_for_override

    config_path = _write_dispatch_stack_config(tmp_path, deployed=["event_dispatcher"])
    _copy_service_templates(tmp_path)

    project_image_ctx = tmp_path / "project-image-ctx"
    persona_ctx = tmp_path / "persona-ctx"
    project_image_ctx.mkdir()
    persona_ctx.mkdir()

    monkeypatch.chdir(tmp_path)
    prepare_compose_files(str(config_path), dev_mode=True)
    assert _copy_local_framework_for_override(str(project_image_ctx)) is True
    assert _copy_local_framework_for_override(str(persona_ctx)) is True

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
    tmp_path: Path, spy_wheel_build: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``rebuild_deployment`` runs ``prepare_compose_files`` twice (its own call
    plus the delegated ``deploy_up``'s) — the wheel build must still run once."""
    config_path = _write_dispatch_stack_config(tmp_path, deployed=["event_dispatcher"])
    _copy_service_templates(tmp_path)

    monkeypatch.chdir(tmp_path)
    prepare_compose_files(str(config_path), dev_mode=True)
    prepare_compose_files(str(config_path), dev_mode=True)

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

    monkeypatch.chdir(tmp_path)
    prepare_compose_files(str(config_path), dev_mode=True)

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
    return yaml.safe_load(compose_file.read_text())["services"]["event-dispatcher"]["build"]["args"]


def test_rendered_compose_carries_osprey_dev_when_wheel_staged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _rendered_dispatcher_build_args(tmp_path, monkeypatch, staging_result=True)
    assert args["OSPREY_DEV"] == "1"


def test_failed_wheel_staging_aborts_the_deploy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A --dev deploy whose staging fails must stop, not render a fallback.

    Rendering *without* OSPREY_DEV would keep the pinned install (fail-closed on
    the build arg) but still deploy successfully — containers up on released
    osprey under a flag that promises local code. The build-arg gate stays; the
    deploy no longer reaches it.
    """
    from osprey.deployment import compose_generator
    from osprey.deployment.errors import DevModeUnavailableError

    def _staging_fails(out_dir):  # type: ignore[no-untyped-def]
        raise DevModeUnavailableError("staging failed", "fix it")

    monkeypatch.setattr(compose_generator, "_copy_local_framework_for_override", _staging_fails)
    config_path = _write_dispatch_stack_config(tmp_path, deployed=["event_dispatcher"])
    _copy_service_templates(tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(DevModeUnavailableError):
        prepare_compose_files(str(config_path), dev_mode=True)


def test_dev_deploy_aborts_on_build_failure_and_stages_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive the REAL staging helper against a failing ``python -m build``:
    the deploy must abort, and no wheel may be left in the build context."""
    import subprocess as subprocess_module

    from osprey.deployment import compose_generator
    from osprey.deployment.errors import DevModeUnavailableError

    real_run = subprocess_module.run

    def _failing_build(cmd, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(cmd, list) and cmd[1:3] == ["-m", "build"]:
            return subprocess_module.CompletedProcess(
                cmd, 1, stdout="", stderr="No module named build"
            )
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(compose_generator.subprocess, "run", _failing_build)

    config_path = _write_dispatch_stack_config(tmp_path, deployed=["event_dispatcher"])
    _copy_service_templates(tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(DevModeUnavailableError):
        prepare_compose_files(str(config_path), dev_mode=True)

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
    from osprey.deployment import compose_generator, wheel_build
    from osprey.deployment.compose_generator import (
        _copy_local_framework_for_override,
        _reset_wheel_build_cache,
    )

    registered: list = []
    monkeypatch.setattr(wheel_build.atexit, "register", lambda fn: registered.append(fn))
    monkeypatch.setattr(wheel_build, "_wheel_cache_cleanup_registered", False)

    ctx = tmp_path / "ctx"
    ctx.mkdir()
    assert _copy_local_framework_for_override(str(ctx)) is True
    assert compose_generator._reset_wheel_build_cache in registered, (
        "creating the wheel cache dir must register the reset hook with atexit"
    )

    cache_dir = wheel_build._wheel_cache_dir
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
#
# Both builds necessarily read the LIVE source root — hatch-vcs needs the real
# git checkout to derive the version, so the tree cannot be snapshotted into
# tmp_path. That makes "identical source" a premise the test must verify rather
# than assume: any write under ``src/`` between the two builds (an editor save,
# a formatter, a concurrent process) makes the two wheels legitimately differ
# and the assertion would report build nondeterminism that does not exist. So
# each round is bracketed by a content snapshot of the packaged inputs, and the
# comparison is only made once a round provably saw the same tree twice.
#
# Exhausting the retries is a FAILURE, not a skip. Nothing writes under ``src/``
# during a CI test run, so a tree that churns through every attempt is a real
# anomaly worth a red rather than a condition to tolerate — and a skip here
# would exit 0 and read as success to any gate that does not assert on skip
# counts.
# ---------------------------------------------------------------------------


def _packaged_source_snapshot(source_root: Path) -> dict[str, str]:
    """Per-path content hashes of the source inputs hatchling packages.

    Returns a mapping rather than one combined digest so that a tree changing
    under the test can be reported as the specific paths that moved.

    A deliberate superset of the true include set (the whole ``src/`` tree plus
    the root metadata files): a superset is sound for the "did the tree change
    under us" guard, and avoids reimplementing hatchling's inclusion rules.

    ``src/osprey/_version.py`` is excluded — it is written *by* the hatch-vcs
    build hook on every build, so it is a build output, not an input. Excluding
    it makes the guard stricter, not weaker: a version stamp that varied between
    builds would still change the wheel bytes while the snapshot held steady,
    and the reproducibility assertion would (correctly) fail.
    """
    import hashlib

    snapshot: dict[str, str] = {}
    version_file = source_root / "src" / "osprey" / "_version.py"
    for path in sorted((source_root / "src").rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path == version_file:
            continue
        try:
            content = path.read_bytes()
        except OSError:
            # Mid-write or just-unlinked: record a marker rather than raising,
            # so the churn surfaces as an unstable snapshot and triggers a retry.
            content = b"<unreadable>"
        snapshot[str(path.relative_to(source_root))] = hashlib.sha256(content).hexdigest()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        meta = source_root / name
        if meta.is_file():
            snapshot[name] = hashlib.sha256(meta.read_bytes()).hexdigest()
    return snapshot


def _snapshot_churn(samples: list[dict[str, str]]) -> list[str]:
    """Paths whose content differed across any of ``samples``, with how."""
    baseline = samples[0]
    churn: dict[str, str] = {}
    for sample in samples[1:]:
        for path in sorted(set(baseline) | set(sample)):
            if baseline.get(path) == sample.get(path):
                continue
            if path not in baseline:
                churn[path] = "appeared"
            elif path not in sample:
                churn[path] = "removed"
            else:
                churn[path] = "content changed"
    return [f"{path} ({how})" for path, how in sorted(churn.items())]


def _wheel_difference_report(first: Path, second: Path) -> str:
    """Name the zip members that differ, so a real failure is diagnosable."""
    import zipfile

    with zipfile.ZipFile(first) as zf1, zipfile.ZipFile(second) as zf2:
        names1 = [info.filename for info in zf1.infolist()]
        names2 = [info.filename for info in zf2.infolist()]
        lines = []
        if only1 := sorted(set(names1) - set(names2)):
            lines.append(f"  only in first: {only1[:10]}")
        if only2 := sorted(set(names2) - set(names1)):
            lines.append(f"  only in second: {only2[:10]}")
        for name in sorted(set(names1) & set(names2)):
            info1, info2 = zf1.getinfo(name), zf2.getinfo(name)
            if zf1.read(name) != zf2.read(name):
                lines.append(f"  content differs: {name}")
            elif (info1.date_time, info1.external_attr, info1.compress_type) != (
                info2.date_time,
                info2.external_attr,
                info2.compress_type,
            ):
                lines.append(
                    f"  zip metadata differs: {name} "
                    f"{info1.date_time}/{info2.date_time} "
                    f"attr {info1.external_attr}/{info2.external_attr}"
                )
    return "\n".join(lines) or "  (archives differ only in container-level bytes)"


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

    attempts = 3
    all_samples: list[dict[str, str]] = []
    for attempt in range(attempts):
        wheels: list[Path] = []
        digests: list[str] = []
        # Four samples: before and after each of the two builds. Only when all
        # four agree did both builds provably read the same bytes.
        samples: list[dict[str, str]] = []
        for label in ("first", "second"):
            out_dir = tmp_path / f"attempt{attempt}-{label}"
            out_dir.mkdir()
            samples.append(_packaged_source_snapshot(source_root))
            # Reset the memo so the second round is a genuinely fresh build, not
            # a copy of the first round's cached wheel.
            _reset_wheel_build_cache()
            if not _copy_local_framework_for_override(str(out_dir)):
                pytest.skip("dev wheel build unavailable in this environment")
            [wheel] = out_dir.glob("*.whl")
            wheels.append(wheel)
            digests.append(hashlib.sha256(wheel.read_bytes()).hexdigest())
            samples.append(_packaged_source_snapshot(source_root))
        all_samples.extend(samples)
        if all(sample == samples[0] for sample in samples):
            break
    else:
        churn = _snapshot_churn(all_samples)
        shown = "\n".join(f"  {entry}" for entry in churn[:20])
        if len(churn) > 20:
            shown += f"\n  ... and {len(churn) - 20} more"
        pytest.fail(
            f"the packaged source tree changed during all {attempts} attempts, so "
            f"the two builds never read the same bytes and wheel reproducibility "
            f"could not be judged.\n\n"
            f"Paths that changed under the test:\n{shown}\n\n"
            f"Likely cause: a concurrent write under {source_root / 'src'}. That is "
            f"expected while several agents or an editor write to this checkout in "
            f"parallel — re-run on a quiet tree and it will pass. In CI it is a real "
            f"signal, not a flake: nothing writes to the source tree during a CI test "
            f"run, so a tree churning through every attempt means something genuinely "
            f"mutated the checkout mid-run."
        )

    assert digests[0] == digests[1], (
        "two wheel builds from identical source must be byte-identical — a "
        "nondeterministic wheel invalidates the Docker layer cache on every "
        "--dev rebuild. The source tree was verified unchanged across both "
        "builds, so this is the build itself:\n" + _wheel_difference_report(wheels[0], wheels[1])
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

    from osprey.deployment import compose_generator
    from osprey.deployment.compose_generator import _copy_local_framework_for_override
    from osprey.deployment.errors import DevModeUnavailableError

    real_run = subprocess_module.run

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(cmd, list) and cmd[1:3] == ["-m", "build"]:
            outdir = cmd[cmd.index("--outdir") + 1]
            Path(outdir, "osprey_framework-0.0.0-py3-none-any.whl").write_bytes(b"not a zip")
            return subprocess_module.CompletedProcess(cmd, 0, stdout="", stderr="")
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(compose_generator.subprocess, "run", _fake_run)

    ctx = tmp_path / "ctx"
    ctx.mkdir()
    with pytest.raises(DevModeUnavailableError):
        _copy_local_framework_for_override(str(ctx))
    assert not list(ctx.glob("*.whl")), "the half-staged wheel must be removed"
    assert not (ctx / "osprey-local-requirements.txt").exists()


def test_staging_fails_closed_when_manifest_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spy_wheel_build: list
) -> None:
    """Even with a valid wheel, a failed manifest WRITE must abort staging
    and remove the already-copied wheel."""
    from osprey.deployment import wheel_build
    from osprey.deployment.compose_generator import _copy_local_framework_for_override
    from osprey.deployment.errors import DevModeUnavailableError

    def _boom(cached_wheel, out_dir):  # type: ignore[no-untyped-def]
        raise OSError("disk full")

    monkeypatch.setattr(wheel_build, "_write_local_requirements_manifest", _boom)

    ctx = tmp_path / "ctx"
    ctx.mkdir()
    with pytest.raises(DevModeUnavailableError):
        _copy_local_framework_for_override(str(ctx))
    assert not list(ctx.glob("*.whl")), "the half-staged wheel must be removed"
    assert not (ctx / "osprey-local-requirements.txt").exists()


# ---------------------------------------------------------------------------
# Nextcloud Talk bridge service template
#
# The bridge is an outbound-only poller: no published port, no healthcheck, and
# a single named volume that IS its crash-safety ledger. Two properties of this
# template are load-bearing beyond "the YAML parses":
#
#   * every credential is a BARE ``${VAR}`` reference — a ``:-default`` on any of
#     them would let the container come up pointed at a guessable host or
#     authenticating with a placeholder instead of failing closed at boot, so
#     these tests assert the ABSENCE of a fallback, not just the presence of the
#     name;
#   * the rendered environment block is the bridge's whole configuration
#     surface, so it is fed through the real ``NextcloudBridgeConfig.from_env``
#     here rather than only string-matched — a renamed env var is dead config the
#     runtime silently ignores, which no substring assertion would catch;
#   * that block is also the ONLY way configuration reaches this service: it
#     declares no ``env_file:``, so the project .env's provider keys never enter
#     the container that faces the external Nextcloud instance.
# ---------------------------------------------------------------------------

_NEXTCLOUD_BRIDGE_TEMPLATE = "templates/services/nextcloud_bridge/docker-compose.yml.j2"

# Credentials and tokens that must render as bare ``${VAR}``. The three secrets
# (the Talk app password and both dispatch tokens) are the security-critical
# members; the base URL, bot account, and room list are here for the same reason
# — a default would silently point the bridge at the wrong instance or room.
_NEXTCLOUD_FAIL_CLOSED_VARS = [
    "NEXTCLOUD_BASE_URL",
    "NEXTCLOUD_BOT_ACCOUNT",
    "NEXTCLOUD_APP_PASSWORD",
    "NEXTCLOUD_ROOMS",
    "EVENT_DISPATCHER_TOKEN",
    "DISPATCH_WORKER_TOKEN",
]

# A single-variable compose substitution: ``${NAME}`` (bare, no fallback),
# ``${NAME:-default}``, or the REQUIRED form ``${NAME:?message}``, on which
# compose aborts instead of substituting. Anchored, so a value that merely
# CONTAINS a reference does not parse as one.
_COMPOSE_VAR_RE = re.compile(
    r"^\$\{([A-Z_][A-Z0-9_]*)(?::-(?P<default>.*)|:\?(?P<required>.*))?\}$"
)


class _ComposeRequiredVarUnset(RuntimeError):
    """Stands in for the compose CLI aborting on an unset ``${VAR:?message}``."""


def _render_nextcloud_bridge_template(
    *,
    env_present: bool = True,
    dispatcher_deployed: bool = True,
    worker_deployed: bool = True,
    services: dict | None = None,
    project_name: str = "p",
) -> str:
    """Render the packaged nextcloud-bridge compose template.

    Loads the packaged template directly (CWD-independent) and renders it with
    bare ``jinja2.Template``, the same default-Undefined mode
    ``compose_generator``'s Environment uses, so ``| default(...)`` chains behave
    exactly as in production.
    """
    from importlib import resources

    from jinja2 import Template

    tpl = resources.files("osprey").joinpath(_NEXTCLOUD_BRIDGE_TEMPLATE)
    template = Template(tpl.read_text(encoding="utf-8"))
    deployed = ["nextcloud_bridge"]
    if dispatcher_deployed:
        deployed.append("event_dispatcher")
    if worker_deployed:
        deployed.append("dispatch_worker")
    if services is None:
        services = {
            "nextcloud_bridge": {"trigger": "nextcloud-question"},
            "event_dispatcher": {},
            "dispatch_worker": {},
        }
    return template.render(
        services=services,
        deployment={},
        system={"timezone": "UTC"},
        deployed_services=deployed,
        osprey_labels={
            "project_name": project_name,
            "project_root": f"/r/{project_name}",
            "deployed_at": "now",
        },
        osprey_version="",
        osprey_env_present=env_present,
    )


def _nextcloud_bridge_service(**kwargs: object) -> dict:
    """Return the parsed ``nextcloud-bridge`` service block."""
    rendered = yaml.safe_load(_render_nextcloud_bridge_template(**kwargs))  # type: ignore[arg-type]
    return rendered["services"]["nextcloud-bridge"]


def _resolve_compose_env(
    rendered_env: dict, host_env: dict[str, str] | None = None
) -> dict[str, str]:
    """Resolve a rendered ``environment:`` block the way the compose CLI does.

    ``${VAR}`` resolves to the host value or the empty string; ``${VAR:-d}``
    resolves to the host value or ``d``; ``${VAR:?msg}`` raises when the host
    supplies nothing, exactly as compose refuses the deploy; anything else is a
    literal. This is what lets the tests below hand the template's own output to
    the real config parser instead of restating the values.
    """
    host_env = host_env or {}
    resolved: dict[str, str] = {}
    for name, raw in rendered_env.items():
        value = str(raw)
        match = _COMPOSE_VAR_RE.match(value)
        if match is None:
            resolved[name] = value
            continue
        from_host = host_env.get(match.group(1), "")
        if not from_host and match.group("required") is not None:
            raise _ComposeRequiredVarUnset(f"{match.group(1)}: {match.group('required')}")
        resolved[name] = from_host or (match.group("default") or "")
    return resolved


def test_nextcloud_bridge_image_follows_env_config_default_chain() -> None:
    """image = ${OSPREY_NEXTCLOUD_BRIDGE_IMAGE:-<project>-nextcloud-bridge:local}.

    Same three-level chain as every sibling service (env override wins, then a
    config-declared image, then the project-namespaced ``:local`` tag that
    ``osprey up`` builds). The local tag must carry the project name: it
    is a host-global docker tag, so a static default would make two projects
    fight over one image.
    """
    assert _nextcloud_bridge_service(project_name="proj-a")["image"] == (
        "${OSPREY_NEXTCLOUD_BRIDGE_IMAGE:-proj-a-nextcloud-bridge:local}"
    )
    assert _nextcloud_bridge_service(project_name="proj-b")["image"] == (
        "${OSPREY_NEXTCLOUD_BRIDGE_IMAGE:-proj-b-nextcloud-bridge:local}"
    )

    # A config-declared image displaces the local tag but stays under the env
    # override, so an operator can still repoint a published image at deploy
    # time without a rebuild.
    pinned = _nextcloud_bridge_service(
        services={
            "nextcloud_bridge": {
                "trigger": "nextcloud-question",
                "image": "ghcr.io/als-apg/osprey-nextcloud-bridge:1.2.3",
            },
            "event_dispatcher": {},
            "dispatch_worker": {},
        }
    )
    assert pinned["image"] == (
        "${OSPREY_NEXTCLOUD_BRIDGE_IMAGE:-ghcr.io/als-apg/osprey-nextcloud-bridge:1.2.3}"
    )


def test_nextcloud_bridge_build_context_is_project_dir_relative() -> None:
    """The image builds from ./nextcloud_bridge (compose-project-dir relative).

    With multiple ``-f`` compose files every relative path resolves against the
    FIRST file's dir (build/services/), not this file's own subdir, so a
    file-relative context ('.', '../nextcloud_bridge') breaks a fresh
    ``osprey up`` with "unable to prepare context: path ... not found".
    """
    build = _nextcloud_bridge_service()["build"]
    assert build["context"] == "./build/services/nextcloud_bridge"
    assert build["dockerfile"] == "Dockerfile"


def test_nextcloud_bridge_command_runs_the_bridge_module() -> None:
    """The service runs the bridge entrypoint as an exec-form ``python -m``.

    Exec form (a YAML list) and not a shell string: the container's PID 1 must be
    python itself so SIGTERM from ``osprey down`` reaches the poll loop's
    shutdown path instead of a shell that never forwards it.
    """
    assert _nextcloud_bridge_service()["command"] == [
        "python",
        "-m",
        "osprey.bridges.nextcloud_talk",
    ]


def test_nextcloud_bridge_state_volume_is_named_and_mounted_at_data() -> None:
    """/data is a NAMED volume — it holds the dedup ledger, history, and offsets.

    All three state files default to paths under /data (``DEDUP_PATH``,
    ``HISTORY_PATH``, ``OFFSETS_PATH``). Without a persisted volume a restart
    loses the in-flight dedup ledger (re-answering or dropping questions) and
    resets the poll offsets (replaying or skipping room history), so this is a
    correctness requirement, not a convenience. Named rather than a bind mount so
    it is namespaced per compose project and survives ``osprey down``.
    """
    rendered = _render_nextcloud_bridge_template()
    parsed = yaml.safe_load(rendered)
    service = parsed["services"]["nextcloud-bridge"]

    assert service["volumes"] == ["nextcloud_bridge_data:/data"]
    assert "nextcloud_bridge_data" in parsed["volumes"], (
        "the /data mount names nextcloud_bridge_data, so the compose file must "
        "declare it as a top-level named volume or `compose up` errors"
    )
    assert not any(str(v).startswith((".", "/", "$")) for v in service["volumes"]), (
        "bridge state must not be a host bind mount — a named volume is what "
        "keeps it project-namespaced and portable across runtimes"
    )


@pytest.mark.parametrize("var", _NEXTCLOUD_FAIL_CLOSED_VARS)
def test_nextcloud_bridge_credentials_render_without_a_default_fallback(var: str) -> None:
    """Credentials render as bare ``${VAR}`` — never ``${VAR:-something}``.

    This is the fail-closed contract. With a fallback, a deployment missing the
    Talk app password or a dispatch token would come up and authenticate with a
    guessable placeholder; bare, the value stays empty and
    ``NextcloudBridgeConfig.require_startup`` aborts at boot naming the missing
    variables. The absence of the fallback is the requirement, so both the parsed
    value and the raw text are checked — a substring test for the name alone
    would pass on the very regression this guards.
    """
    rendered = _render_nextcloud_bridge_template()
    environment = yaml.safe_load(rendered)["services"]["nextcloud-bridge"]["environment"]

    assert environment[var] == f"${{{var}}}", (
        f"{var} must be a bare ${{{var}}} reference (got {environment[var]!r}) so an "
        "unset value stays empty and the bridge fails closed at boot"
    )
    assert f"${{{var}:" not in rendered, (
        f"{var} carries a compose default — a missing secret would silently "
        "resolve to it instead of failing the boot"
    )


def test_nextcloud_bridge_neutral_tunables_keep_their_defaults_in_code() -> None:
    """Optional knobs pass through with an EMPTY ``:-`` default, not a restated value.

    An empty value makes ``CoreConfig.from_env`` fall back to its own default, so
    each tunable has exactly one definition (the dataclass) instead of drifting
    copies in the compose file. The empty default (rather than a bare reference)
    also keeps ``compose up`` quiet about unset optional vars on every deploy.
    ``POLL_BUDGET`` is the deliberate exception — its default is derived from the
    worker cap (see the poll-budget test below).
    """
    environment = _nextcloud_bridge_service()["environment"]
    for var in (
        "POLL_INTERVAL",
        "DRAIN_INTERVAL",
        "RETRY_MIN_AGE",
        "RETRY_GIVE_UP",
        "RETRY_LIFETIME_CAP",
        "BRIDGE_TRUST_ENV",
        "GITLAB_URL",
        "GITLAB_PROJECT",
        "GITLAB_ISSUES_TOKEN",
    ):
        assert environment[var] == f"${{{var}:-}}", (
            f"{var} must pass through with an empty default so CoreConfig owns "
            f"its default (got {environment[var]!r})"
        )


@pytest.mark.parametrize("env_present", [True, False])
def test_nextcloud_bridge_never_mounts_the_project_env_in_bulk(env_present: bool) -> None:
    """The bridge gets named variables only — never the whole project .env.

    The project .env holds the provider keys the dispatch worker needs
    (``CBORG_API_KEY``, ``OPENAI_API_KEY``, ...). The bridge never calls an LLM
    and it is the one component that talks to an external Nextcloud instance, so
    handing it the file would widen its blast radius for nothing: every value it
    actually reads arrives by interpolation into ``environment:``, resolved from
    that same .env because ``osprey up`` runs compose with ``--env-file .env``
    (and ``environment:`` outranks ``env_file:`` in compose regardless).
    Asserted for a .env both present and absent, so reintroducing the mount
    behind an ``osprey_env_present`` gate does not slip through.
    """
    rendered = _render_nextcloud_bridge_template(env_present=env_present)
    assert "env_file:" not in rendered and _ENV_FILE_LINE not in rendered
    assert "env_file" not in yaml.safe_load(rendered)["services"]["nextcloud-bridge"]


def test_nextcloud_bridge_state_paths_match_their_code_defaults() -> None:
    """The state paths' compose defaults equal the config dataclasses' defaults.

    These three are the only vars whose ``:-`` fallback restates a code default
    instead of passing through empty: their readers are plain
    ``e.get(NAME, default)`` calls, for which "" is an accepted value, so the
    empty-fallback trick the neutral tunables use would point the stores at the
    empty path. That duplication is the thing this test exists to bind — the
    defaults are read back OUT of the real config classes (built from an empty
    environment, so the dataclass defaults are what surface) rather than
    restated here, so changing either side alone fails.
    """
    from osprey.bridges.core import CoreConfig
    from osprey.bridges.nextcloud_talk.config import NextcloudBridgeConfig

    core_defaults = CoreConfig.from_env({})
    nextcloud_defaults = NextcloudBridgeConfig.from_env({})
    code_defaults = {
        "DEDUP_PATH": core_defaults.dedup_path,
        "HISTORY_PATH": core_defaults.history_path,
        "OFFSETS_PATH": nextcloud_defaults.offsets_path,
    }

    environment = _nextcloud_bridge_service()["environment"]
    for var, code_default in code_defaults.items():
        match = _COMPOSE_VAR_RE.match(str(environment[var]))
        assert match is not None and match.group("default") is not None, (
            f"{var} must render as ${{{var}:-<default>}} (got {environment[var]!r}) — "
            "the literal default is what keeps the path set when the var is unset, "
            "and the interpolation is what keeps it .env-overridable"
        )
        assert match.group("default") == code_default, (
            f"{var} renders {match.group('default')!r} but the config default is "
            f"{code_default!r}: the compose literal and the dataclass default must "
            "move together, or a deploy silently writes state somewhere else"
        )

    # Nothing on the host resolves to the code default, so the stores land on the
    # mounted volume; a value on the host wins, which is the override path the
    # removed bulk .env mount used to provide.
    resolved = _resolve_compose_env(environment)
    for var, code_default in code_defaults.items():
        assert resolved[var] == code_default

    overridden = _resolve_compose_env(
        environment, host_env={var: f"/srv/state/{var.lower()}.json" for var in code_defaults}
    )
    for var in code_defaults:
        assert overridden[var] == f"/srv/state/{var.lower()}.json"

    # And the override reaches the config objects under the names they read —
    # an empty value would NOT, which is why these three are not passed through
    # with an empty fallback.
    cfg = NextcloudBridgeConfig.from_env(overridden)
    assert cfg.core.dedup_path == "/srv/state/dedup_path.json"
    assert cfg.core.history_path == "/srv/state/history_path.json"
    assert cfg.offsets_path == "/srv/state/offsets_path.json"


def test_nextcloud_bridge_depends_on_dispatcher_only_when_co_deployed() -> None:
    """``depends_on: event-dispatcher`` renders IFF the dispatcher is co-deployed.

    The bridge reconciles in-flight runs against the dispatcher before accepting
    a message, so co-deployed it must start after the dispatcher's health probe.
    A bridge pointed at an EXTERNAL dispatcher must not emit the block at all:
    compose fails hard on a ``depends_on`` naming an undefined service.
    """
    with_dispatcher = _nextcloud_bridge_service(dispatcher_deployed=True)
    assert with_dispatcher["depends_on"] == {"event-dispatcher": {"condition": "service_healthy"}}

    external = _render_nextcloud_bridge_template(dispatcher_deployed=False)
    assert "depends_on:" not in external, (
        "a bridge deployed without the dispatcher must emit no depends_on — "
        "compose errors on a dependency naming an undefined service"
    )
    assert yaml.safe_load(external)["services"]["nextcloud-bridge"].get("depends_on") is None


def test_nextcloud_bridge_dispatch_urls_track_the_dispatch_templates_ports() -> None:
    """In-network URLs use the SAME ports the dispatch templates serve on.

    Derived from the sibling templates' own rendered output rather than restated
    here, so a port change in the dispatch pair cannot leave the bridge calling a
    closed port. Service keys (not container names) are the DNS names on
    osprey-network, and the worker is addressed directly because the bridge polls
    run status and fetches artifacts from it, not through the dispatcher.
    """
    for services, expected_dispatcher_port, expected_worker_port in (
        # Config-block defaults, and explicitly non-default ports.
        (
            {"nextcloud_bridge": {"trigger": "t"}, "event_dispatcher": {}, "dispatch_worker": {}},
            8020,
            9190,
        ),
        (
            {
                "nextcloud_bridge": {"trigger": "t"},
                "event_dispatcher": {"port": 8031},
                "dispatch_worker": {"worker_port_base": 9201},
            },
            8031,
            9201,
        ),
    ):
        dispatcher = yaml.safe_load(
            _render_service_template(
                "event_dispatcher/docker-compose.yml.j2", "p", services=services
            )
        )["services"]["event-dispatcher"]
        assert dispatcher["environment"]["FASTMCP_PORT"] == str(expected_dispatcher_port), (
            "test premise: the dispatcher template must serve the port this case expects"
        )

        environment = _nextcloud_bridge_service(services=services)["environment"]
        assert (
            environment["DISPATCHER_URL"] == f"http://event-dispatcher:{expected_dispatcher_port}"
        )
        assert environment["WORKER_URL"] == f"http://dispatch-worker-1:{expected_worker_port}"


def test_nextcloud_bridge_dispatch_urls_pass_through_when_external() -> None:
    """Without the dispatch pair co-deployed, both URLs are REQUIRED host vars.

    An external dispatcher/worker is reached by whatever address the deploy env
    supplies; hardcoding the in-network name would make the bridge call a
    nonexistent host, and the code's localhost default is wrong from inside a
    container either way. They render in compose's required form
    (``${VAR:?message}``) rather than as bare references, because compose
    resolves an unset bare reference to the EMPTY STRING and not to an absent
    key: ``CoreConfig.from_env``'s localhost default would never fire,
    ``require_startup`` does not cover these two, so the stack would boot and
    then POST every dispatch to a protocol-less URL. Since a raising
    ``handle_event`` deliberately leaves the persisted poll offset unadvanced,
    that turns one missing variable into a room re-fetching the same batch
    forever. ``:?`` makes it a startup abort instead.
    """
    environment = _nextcloud_bridge_service(
        dispatcher_deployed=False,
        worker_deployed=False,
        services={"nextcloud_bridge": {"trigger": "t"}},
    )["environment"]
    for var in ("DISPATCHER_URL", "WORKER_URL"):
        assert str(environment[var]).startswith(f"${{{var}:?"), (
            f"{var} must render as compose's required form ${{{var}:?...}} when the "
            f"dispatch pair is external (got {environment[var]!r}) — a bare reference "
            "resolves to an empty string and boots a bridge that can never dispatch"
        )

    # Nothing on the host: the deploy must be REFUSED, not resolved to "".
    with pytest.raises(_ComposeRequiredVarUnset, match="DISPATCHER_URL"):
        _resolve_compose_env(environment)

    # Supplied, they pass through verbatim — the point of the passthrough.
    resolved = _resolve_compose_env(
        environment,
        host_env={
            "DISPATCHER_URL": "https://dispatch.example.org",
            "WORKER_URL": "https://worker.example.org",
        },
    )
    assert resolved["DISPATCHER_URL"] == "https://dispatch.example.org"
    assert resolved["WORKER_URL"] == "https://worker.example.org"

    # The CO-DEPLOYED branch must NOT carry the guard: it renders the in-network
    # address itself and needs no host variable, so a `:?` there would abort a
    # perfectly valid single-stack deploy.
    co_deployed = _nextcloud_bridge_service()["environment"]
    for var in ("DISPATCHER_URL", "WORKER_URL"):
        assert ":?" not in str(co_deployed[var]), (
            f"{var} is rendered in-network when the dispatch pair is co-deployed; "
            "requiring a host variable there would break the common deploy"
        )


def test_nextcloud_bridge_trigger_comes_from_the_profile_and_has_no_template_default() -> None:
    """``DISPATCH_TRIGGER`` is rendered from the profile block, with no fallback here.

    ``NextcloudBridgeProfileConfig.trigger`` is the ONLY place the
    ``nextcloud-question`` default lives (the runtime's ``from_env`` applies none
    either), so a template-side default would be a second definition that fires
    some other facility's trigger when the config key goes missing.
    """
    from osprey.cli.build_profile import NextcloudBridgeProfileConfig

    profile_default = NextcloudBridgeProfileConfig().trigger
    rendered = _render_nextcloud_bridge_template(
        services={
            "nextcloud_bridge": {"trigger": profile_default},
            "event_dispatcher": {},
            "dispatch_worker": {},
        }
    )
    environment = yaml.safe_load(rendered)["services"]["nextcloud-bridge"]["environment"]
    assert environment["DISPATCH_TRIGGER"] == profile_default

    # A facility-chosen trigger must render verbatim, and the profile default
    # must not survive as a template-side fallback.
    custom = _render_nextcloud_bridge_template(
        services={
            "nextcloud_bridge": {"trigger": "als-talk-question"},
            "event_dispatcher": {},
            "dispatch_worker": {},
        }
    )
    custom_env = yaml.safe_load(custom)["services"]["nextcloud-bridge"]["environment"]
    assert custom_env["DISPATCH_TRIGGER"] == "als-talk-question"

    # A config that lost the key must render an EMPTY trigger, so the bridge
    # aborts at boot naming DISPATCH_TRIGGER. A template-side default would
    # instead fire whatever name the framework happened to pick.
    keyless = _render_nextcloud_bridge_template(
        services={"nextcloud_bridge": {}, "event_dispatcher": {}, "dispatch_worker": {}}
    )
    keyless_env = yaml.safe_load(keyless)["services"]["nextcloud-bridge"]["environment"]
    assert keyless_env["DISPATCH_TRIGGER"] == "", (
        f"a missing trigger key must render empty, not fall back to {profile_default!r} — "
        "the profile block is the single source of the trigger name"
    )


def test_nextcloud_bridge_rendered_env_parses_and_fails_closed_without_secrets() -> None:
    """The rendered env, resolved with nothing set on the host, refuses to boot.

    Feeds the template's own output through the real
    ``NextcloudBridgeConfig.from_env`` — so a renamed variable shows up as dead
    config here rather than as a bridge that silently ignores it — and asserts
    ``require_startup`` names exactly the fail-closed variables. The trigger is
    absent from that list precisely because the template renders it as a literal.
    """
    from osprey.bridges.nextcloud_talk.config import NextcloudBridgeConfig

    environment = _nextcloud_bridge_service()["environment"]
    cfg = NextcloudBridgeConfig.from_env(_resolve_compose_env(environment))

    with pytest.raises(ValueError) as excinfo:
        cfg.require_startup()
    missing = {name.strip() for name in str(excinfo.value).split(":", 1)[1].split(",")}
    assert missing == set(_NEXTCLOUD_FAIL_CLOSED_VARS), (
        "with nothing set on the host the bridge must abort naming exactly the "
        f"bare-reference variables; got {sorted(missing)}"
    )


def test_nextcloud_bridge_rendered_env_boots_with_host_secrets_supplied() -> None:
    """With the .env supplying the credentials, the same block yields a valid config.

    Proves the variable NAMES the template renders are the names the runtime
    reads: the Nextcloud settings, both tokens, the trigger, and the in-network
    dispatch endpoints all arrive on the config object, and the state paths stay
    on the /data volume.
    """
    from osprey.bridges.nextcloud_talk.config import NextcloudBridgeConfig

    environment = _nextcloud_bridge_service()["environment"]
    resolved = _resolve_compose_env(
        environment,
        host_env={
            "NEXTCLOUD_BASE_URL": "https://talk.example.org",
            "NEXTCLOUD_BOT_ACCOUNT": "osprey-bot",
            "NEXTCLOUD_APP_PASSWORD": "app-pw",
            "NEXTCLOUD_ROOMS": "abc123, def456",
            "EVENT_DISPATCHER_TOKEN": "dispatcher-token",
            "DISPATCH_WORKER_TOKEN": "worker-token",
        },
    )
    cfg = NextcloudBridgeConfig.from_env(resolved)
    cfg.require_startup()

    assert cfg.base_url == "https://talk.example.org"
    assert cfg.bot_account == "osprey-bot"
    assert cfg.app_password == "app-pw"
    assert cfg.rooms == ("abc123", "def456")
    assert cfg.core.event_dispatcher_token == "dispatcher-token"
    assert cfg.core.dispatch_worker_token == "worker-token"
    assert cfg.core.trigger == "nextcloud-question"
    assert cfg.core.dispatcher_url == "http://event-dispatcher:8020"
    assert cfg.core.worker_url == "http://dispatch-worker-1:9190"
    # The three state files must land on the mounted volume, not the image layer.
    for path in (cfg.offsets_path, cfg.core.dedup_path, cfg.core.history_path):
        assert path.startswith("/data/"), path


@pytest.mark.parametrize("worker_timeout", [None, 600])
def test_nextcloud_bridge_poll_budget_default_outlasts_the_worker_cap(
    worker_timeout: int | None,
) -> None:
    """The bridge's poll budget is derived from the worker's cap, not restated.

    ``CoreConfig.__post_init__`` rejects ``poll_budget < worker_timeout``, which
    would crash-loop the bridge, and both halves read the cap from the same
    ``services.dispatch_worker.timeout_sec`` key — so raising the cap in one
    place must raise the budget here too. Constructing the config from the
    resolved defaults is what proves the relation holds rather than asserting on
    the numbers alone.
    """
    from osprey.bridges.core import CoreConfig

    worker_config = {} if worker_timeout is None else {"timeout_sec": worker_timeout}
    environment = _nextcloud_bridge_service(
        services={
            "nextcloud_bridge": {"trigger": "t"},
            "event_dispatcher": {},
            "dispatch_worker": worker_config,
        }
    )["environment"]

    cfg = CoreConfig.from_env(_resolve_compose_env(environment))
    expected_timeout = float(worker_timeout if worker_timeout is not None else 300)
    assert cfg.worker_timeout == expected_timeout
    assert cfg.poll_budget == expected_timeout + 30, (
        "the poll budget default must exceed the worker cap it waits out"
    )


def test_nextcloud_bridge_config_lookups_survive_explicit_null_values() -> None:
    """A config key present but EMPTY must still render the documented default.

    Jinja's ``default`` filter substitutes only on *Undefined*, so a key written
    as ``timeout_sec:`` with no value — which YAML loads as ``None`` — sails past
    a plain ``| default(300)`` and renders the literal string "None".
    ``CoreConfig.from_env`` then dies on ``float("None")`` at boot, and
    ``None | int`` silently degrades ``POLL_BUDGET`` to 0. The boolean form
    (``default(300, true)``) substitutes on any falsy value, which is what keeps
    a half-written config booting on the defaults instead of crash-looping.
    """
    from osprey.bridges.core import CoreConfig

    environment = _nextcloud_bridge_service(
        services={
            "nextcloud_bridge": {"trigger": "t"},
            "event_dispatcher": {"port": None},
            "dispatch_worker": {"timeout_sec": None, "worker_port_base": None},
        }
    )["environment"]

    assert "None" not in str(environment["DISPATCH_TIMEOUT_SEC"])
    assert environment["DISPATCHER_URL"] == "http://event-dispatcher:8020"
    assert environment["WORKER_URL"] == "http://dispatch-worker-1:9190"

    cfg = CoreConfig.from_env(_resolve_compose_env(environment))
    assert cfg.worker_timeout == 300.0
    assert cfg.poll_budget == 330.0


def test_nextcloud_bridge_container_name_is_project_namespaced() -> None:
    """Two projects render distinct bridge container names.

    ``container_name`` is a HOST-GLOBAL docker identifier, so a static name stops
    two OSPREY projects from running a bridge on one host. Nothing reaches this
    service in-network (it only makes outbound calls), so no network alias is
    needed alongside the rename.
    """
    name_a = _nextcloud_bridge_service(project_name="proj-a")["container_name"]
    name_b = _nextcloud_bridge_service(project_name="proj-b")["container_name"]
    assert (name_a, name_b) == ("proj-a-nextcloud-bridge", "proj-b-nextcloud-bridge")


def test_nextcloud_bridge_publishes_no_ports_and_declares_no_healthcheck() -> None:
    """The bridge is a poller: no listening socket, so no ports and no probe.

    A published port would be dead surface, and a healthcheck against a service
    that opens no socket would mark a healthy bridge unhealthy — which
    ``depends_on: service_healthy`` elsewhere would then act on.
    """
    service = _nextcloud_bridge_service()
    assert "ports" not in service
    assert "healthcheck" not in service
    assert service["networks"] == ["osprey-network"]
    assert service["restart"] == "unless-stopped"


def test_nextcloud_bridge_template_is_bundled_into_a_declaring_project(tmp_path: Path) -> None:
    """``osprey build`` copies the packaged bridge template into the project tree.

    The whole service directory (compose template, Dockerfile, .dockerignore)
    must ship in the package and be discoverable under the ``nextcloud_bridge``
    service key, or ``osprey up`` has nothing to render and no build
    context to build.
    """
    _write_config(tmp_path, deployed_services=["nextcloud_bridge"])

    assert _copy_service_templates(tmp_path) == 1

    service_dir = tmp_path / "services" / "nextcloud_bridge"
    assert (service_dir / "docker-compose.yml.j2").is_file()
    assert (service_dir / "Dockerfile").is_file(), (
        "the build context needs its Dockerfile — the compose template declares "
        "build: ./nextcloud_bridge with dockerfile: Dockerfile"
    )
    assert (service_dir / ".dockerignore").is_file(), (
        ".dockerignore is the guaranteed COPY sibling the Dockerfile's optional "
        "wheel/requirements globs rely on, and it keeps a stale .env out of the image"
    )


# ---------------------------------------------------------------------------
# gchat-bridge service template
#
# The Google Chat bridge is the Nextcloud bridge's sibling: a subscriber, not a
# server, driving the same dispatch pair through the same `osprey.bridges.core`
# config. Its compose template therefore inherits the same contracts (no bulk
# .env, project-namespaced image/container name, derived poll budget) and the
# same three-way environment discipline, checked here the same way:
#
#   * bare `${VAR}` for everything a deployment must supply and must never
#     guess — an unset value stays empty so `require_startup` aborts at boot;
#   * `${VAR:?}` for the dispatch endpoints when they are NOT co-deployed,
#     because an unset bare reference would resolve to "" and boot a bridge that
#     redelivers the same Pub/Sub message forever;
#   * `${VAR:-}` for anything whose default lives in a config dataclass.
#
# Two things differ from the Nextcloud set and get their own tests: the
# service-account key is a FILE, mounted read-only, and there is no offsets
# store (the subscription's own ack state is the ingestion cursor).
# ---------------------------------------------------------------------------

_GCHAT_BRIDGE_TEMPLATE = "templates/services/gchat_bridge/docker-compose.yml.j2"

# Credentials and tokens that must render as bare ``${VAR}``. All five are
# security- or destination-critical: a default would authenticate the bridge as
# nobody, point it at another project's Pub/Sub queue, or let it dispatch with a
# guessable secret. ``DISPATCH_TRIGGER`` is required by ``require_startup`` too
# but is absent here because the template renders it as a profile-supplied
# literal, not as an interpolation.
_GCHAT_FAIL_CLOSED_VARS = [
    "GCHAT_SA_KEY",
    "GCHAT_SUBSCRIPTION",
    "GCHAT_APP_ID",
    "EVENT_DISPATCHER_TOKEN",
    "DISPATCH_WORKER_TOKEN",
]

# A fully-qualified pull subscription. Used wherever a test supplies one so the
# config's shape check (which warns on a bare subscription id) stays quiet.
_GCHAT_SUBSCRIPTION = "projects/als-apg/subscriptions/gchat-events"


def _render_gchat_bridge_template(
    *,
    env_present: bool = True,
    dispatcher_deployed: bool = True,
    worker_deployed: bool = True,
    services: dict | None = None,
    project_name: str = "p",
) -> str:
    """Render the packaged gchat-bridge compose template.

    Loads the packaged template directly (CWD-independent) and renders it with
    bare ``jinja2.Template``, the same default-Undefined mode
    ``compose_generator``'s Environment uses, so ``| default(...)`` chains behave
    exactly as in production.
    """
    from importlib import resources

    from jinja2 import Template

    tpl = resources.files("osprey").joinpath(_GCHAT_BRIDGE_TEMPLATE)
    template = Template(tpl.read_text(encoding="utf-8"))
    deployed = ["gchat_bridge"]
    if dispatcher_deployed:
        deployed.append("event_dispatcher")
    if worker_deployed:
        deployed.append("dispatch_worker")
    if services is None:
        services = {
            "gchat_bridge": {"trigger": "gchat-question"},
            "event_dispatcher": {},
            "dispatch_worker": {},
        }
    return template.render(
        services=services,
        deployment={},
        system={"timezone": "UTC"},
        deployed_services=deployed,
        osprey_labels={
            "project_name": project_name,
            "project_root": f"/r/{project_name}",
            "deployed_at": "now",
        },
        osprey_version="",
        osprey_env_present=env_present,
    )


def _gchat_bridge_service(**kwargs: object) -> dict:
    """Return the parsed ``gchat-bridge`` service block."""
    rendered = yaml.safe_load(_render_gchat_bridge_template(**kwargs))  # type: ignore[arg-type]
    return rendered["services"]["gchat-bridge"]


def _gchat_environment_text(rendered: str) -> str:
    """The raw text of the rendered ``environment:`` block.

    The fail-closed checks below assert on raw text as well as on parsed values
    (a parsed-value check alone would pass on ``${VAR:-guess}`` if the YAML
    parser were ever swapped), but they must be scoped to ``environment:``: the
    ``volumes:`` block legitimately carries ``${GCHAT_SA_KEY:-/dev/null}``, a
    mount sentinel that never reaches the container's environment (see
    ``test_gchat_bridge_sa_key_is_mounted_read_only_at_its_own_path``).
    """
    start = rendered.index("    environment:")
    return rendered[start : rendered.index("    volumes:", start)]


def test_gchat_bridge_image_follows_env_config_default_chain() -> None:
    """image = ${OSPREY_GCHAT_BRIDGE_IMAGE:-<project>-gchat-bridge:local}.

    Same three-level chain as every sibling service (env override wins, then a
    config-declared image, then the project-namespaced ``:local`` tag that
    ``osprey up`` builds). The local tag must carry the project name: it
    is a host-global docker tag, so a static default would make two projects
    fight over one image.
    """
    assert _gchat_bridge_service(project_name="proj-a")["image"] == (
        "${OSPREY_GCHAT_BRIDGE_IMAGE:-proj-a-gchat-bridge:local}"
    )
    assert _gchat_bridge_service(project_name="proj-b")["image"] == (
        "${OSPREY_GCHAT_BRIDGE_IMAGE:-proj-b-gchat-bridge:local}"
    )

    # A config-declared image displaces the local tag but stays under the env
    # override, so an operator can still repoint a published image at deploy
    # time without a rebuild.
    pinned = _gchat_bridge_service(
        services={
            "gchat_bridge": {
                "trigger": "gchat-question",
                "image": "ghcr.io/als-apg/osprey-gchat-bridge:1.2.3",
            },
            "event_dispatcher": {},
            "dispatch_worker": {},
        }
    )
    assert pinned["image"] == (
        "${OSPREY_GCHAT_BRIDGE_IMAGE:-ghcr.io/als-apg/osprey-gchat-bridge:1.2.3}"
    )


def test_gchat_bridge_build_context_is_project_dir_relative() -> None:
    """The image builds from ./gchat_bridge (compose-project-dir relative).

    With multiple ``-f`` compose files every relative path resolves against the
    FIRST file's dir (build/services/), not this file's own subdir, so a
    file-relative context ('.', '../gchat_bridge') breaks a fresh
    ``osprey up`` with "unable to prepare context: path ... not found".
    """
    build = _gchat_bridge_service()["build"]
    assert build["context"] == "./build/services/gchat_bridge"
    assert build["dockerfile"] == "Dockerfile"


def test_gchat_bridge_command_runs_the_bridge_module() -> None:
    """The service runs the bridge entrypoint as an exec-form ``python -m``.

    Exec form (a YAML list) and not a shell string: the container's PID 1 must be
    python itself so SIGTERM from ``osprey down`` reaches the subscriber's
    shutdown path (which cancels the streaming-pull future) instead of a shell
    that never forwards it.
    """
    assert _gchat_bridge_service()["command"] == [
        "python",
        "-m",
        "osprey.bridges.google_chat",
    ]


def test_gchat_bridge_state_volume_is_named_and_mounted_at_data() -> None:
    """/data is a NAMED volume — it holds the dedup ledger and the history store.

    Both state files default to paths under /data (``DEDUP_PATH``,
    ``HISTORY_PATH``). Without a persisted volume a restart loses the in-flight
    dedup ledger and the bridge re-answers or drops questions that were mid-run,
    so this is a correctness requirement, not a convenience. Named rather than a
    bind mount so it is namespaced per compose project and survives
    ``osprey down``. Unlike the Nextcloud bridge there is no offsets store: the
    Pub/Sub subscription's own ack state is the ingestion cursor.
    """
    rendered = _render_gchat_bridge_template()
    parsed = yaml.safe_load(rendered)
    service = parsed["services"]["gchat-bridge"]

    assert "gchat_bridge_data:/data" in service["volumes"]
    assert "gchat_bridge_data" in parsed["volumes"], (
        "the /data mount names gchat_bridge_data, so the compose file must "
        "declare it as a top-level named volume or `compose up` errors"
    )
    assert "OFFSETS_PATH" not in rendered, (
        "the Pub/Sub subscriber persists no poll offsets — an OFFSETS_PATH here "
        "would be dead config nothing reads"
    )


def test_gchat_bridge_sa_key_is_mounted_read_only_at_its_own_path() -> None:
    """The service-account key is bind-mounted READ-ONLY at the path it names.

    One variable, ``GCHAT_SA_KEY``, names the key on the host and in the
    container: a separate "container path" variable could drift from the mount,
    and a deployment that updated only one would authenticate against a path
    that does not exist. Read-only because nothing in the bridge ever writes the
    key, and it is the credential for every Google call the service makes.

    The mount is also the one place ``GCHAT_SA_KEY`` may carry a ``:-``
    fallback, and it must: compose rejects the empty spec ``::ro`` outright,
    with an error that never names the variable, whereas the ``/dev/null``
    sentinel is a harmless no-op that lets the container come up far enough for
    ``require_startup`` to name the missing variable itself (asserted by
    ``test_gchat_bridge_rendered_env_parses_and_fails_closed_without_secrets``).
    """
    volumes = _gchat_bridge_service()["volumes"]
    sa_mounts = [v for v in volumes if "GCHAT_SA_KEY" in str(v)]
    assert len(sa_mounts) == 1, f"expected exactly one SA-key mount, got {sa_mounts}"

    # Split on the `}:${` boundaries rather than on ":" — a compose default
    # (`:-`) puts colons inside the interpolation itself.
    spec = re.fullmatch(
        r"(?P<source>\$\{[^}]*\}):(?P<target>\$\{[^}]*\}):(?P<mode>[a-z]+)", str(sa_mounts[0])
    )
    assert spec is not None, (
        f"the SA-key mount must be <interpolated source>:<interpolated target>:<mode>, "
        f"got {sa_mounts[0]!r}"
    )
    source, target, mode = spec.group("source"), spec.group("target"), spec.group("mode")
    assert mode == "ro", f"the service-account key must be mounted read-only, got {mode!r}"
    assert source == target, (
        f"the key must be mounted at the path GCHAT_SA_KEY names ({source!r} -> {target!r}) — "
        "a distinct container path would be a second value that can drift from the env var"
    )

    # Unset, the sentinel keeps the deploy valid; set, both sides follow the
    # operator's path so the container opens the file the .env points at.
    assert source == "${GCHAT_SA_KEY:-/dev/null}"
    resolved_unset = _resolve_compose_env({"mount": source})
    assert resolved_unset["mount"] == "/dev/null"
    resolved_set = _resolve_compose_env(
        {"mount": source}, host_env={"GCHAT_SA_KEY": "/secrets/gchat-sa.json"}
    )
    assert resolved_set["mount"] == "/secrets/gchat-sa.json"


@pytest.mark.parametrize("var", _GCHAT_FAIL_CLOSED_VARS)
def test_gchat_bridge_credentials_render_without_a_default_fallback(var: str) -> None:
    """Credentials render as bare ``${VAR}`` — never ``${VAR:-something}``.

    This is the fail-closed contract. With a fallback, a deployment missing the
    service-account key or a dispatch token would come up authenticating with a
    guessable placeholder; bare, the value stays empty and
    ``GoogleChatBridgeConfig.require_startup`` aborts at boot naming the missing
    variables. The absence of the fallback is the requirement, so both the parsed
    value and the raw text of the ``environment:`` block are checked — a
    substring test for the name alone would pass on the very regression this
    guards.
    """
    rendered = _render_gchat_bridge_template()
    environment = yaml.safe_load(rendered)["services"]["gchat-bridge"]["environment"]

    assert environment[var] == f"${{{var}}}", (
        f"{var} must be a bare ${{{var}}} reference (got {environment[var]!r}) so an "
        "unset value stays empty and the bridge fails closed at boot"
    )
    assert f"${{{var}:" not in _gchat_environment_text(rendered), (
        f"{var} carries a compose default — a missing secret would silently "
        "resolve to it instead of failing the boot"
    )


def test_gchat_bridge_documents_the_single_subscriber_constraint() -> None:
    """The template warns, beside ``GCHAT_SUBSCRIPTION``, that ONE bridge may pull it.

    Pub/Sub load-balances a subscription across its consumers, so a second
    deployment pointed at the same subscription does not duplicate events — it
    silently splits them, and each half answers only the messages it happened to
    receive. Nothing in the config surface can detect that, which makes the
    comment the only place the constraint is stated at deploy time; this pins it
    so an edit cannot quietly drop it.
    """
    rendered = _render_gchat_bridge_template()
    subscription_line = next(
        i for i, line in enumerate(rendered.splitlines()) if "GCHAT_SUBSCRIPTION:" in line
    )
    preamble = "\n".join(rendered.splitlines()[max(0, subscription_line - 12) : subscription_line])

    assert "SINGLE SUBSCRIBER" in preamble, (
        "the single-subscriber constraint must be documented directly above "
        f"GCHAT_SUBSCRIPTION; preceding comment was:\n{preamble}"
    )
    assert "SPLIT" in preamble.upper(), (
        "the comment must say a second consumer SPLITS the events — an operator "
        "who expects duplicates would deploy a second bridge deliberately"
    )


def test_gchat_bridge_neutral_tunables_keep_their_defaults_in_code() -> None:
    """Optional knobs pass through with an EMPTY ``:-`` default, not a restated value.

    An empty value makes ``CoreConfig.from_env`` (and, for the ``GCS_*`` pair and
    the version tag, ``GoogleChatBridgeConfig.from_env``) fall back to its own
    default, so each tunable has exactly one definition — the dataclass —
    instead of drifting copies in the compose file. The empty default (rather
    than a bare reference) also keeps ``compose up`` quiet about unset optional
    vars on every deploy, which matters most for the ``GCS_*`` pair: publishing
    artifacts is opt-in, so unset is the normal case. ``POLL_BUDGET`` is the
    deliberate exception — its default is derived from the worker cap (see the
    poll-budget test below).
    """
    environment = _gchat_bridge_service()["environment"]
    for var in (
        "POLL_INTERVAL",
        "DRAIN_INTERVAL",
        "RETRY_MIN_AGE",
        "RETRY_GIVE_UP",
        "RETRY_LIFETIME_CAP",
        "BRIDGE_TRUST_ENV",
        "GITLAB_URL",
        "GITLAB_PROJECT",
        "GITLAB_ISSUES_TOKEN",
        "GCS_BUCKET",
        "GCS_PROJECT",
        "APP_VERSION_DISPLAY",
    ):
        assert environment[var] == f"${{{var}:-}}", (
            f"{var} must pass through with an empty default so the config dataclass "
            f"owns its default (got {environment[var]!r})"
        )

    # The optional Google settings are deliberately NOT fail-closed: an unset
    # bucket disables image delivery and the bridge still answers text-only, so
    # require_startup must keep ignoring them.
    from osprey.bridges.google_chat.config import GoogleChatBridgeConfig

    cfg = GoogleChatBridgeConfig.from_env(_resolve_compose_env(environment))
    assert (cfg.gcs_bucket, cfg.gcs_project, cfg.version_tag) == ("", "", "")


@pytest.mark.parametrize("env_present", [True, False])
def test_gchat_bridge_never_mounts_the_project_env_in_bulk(env_present: bool) -> None:
    """The bridge gets named variables only — never the whole project .env.

    The project .env holds the provider keys the dispatch worker needs
    (``CBORG_API_KEY``, ``OPENAI_API_KEY``, ...). The bridge never calls an LLM
    and it is the one component holding Google service-account credentials, so
    handing it the file would widen its blast radius for nothing: every value it
    actually reads arrives by interpolation into ``environment:``, resolved from
    that same .env because ``osprey up`` runs compose with ``--env-file .env``
    (and ``environment:`` outranks ``env_file:`` in compose regardless).
    Asserted for a .env both present and absent, so reintroducing the mount
    behind an ``osprey_env_present`` gate does not slip through.
    """
    rendered = _render_gchat_bridge_template(env_present=env_present)
    assert "env_file:" not in rendered and _ENV_FILE_LINE not in rendered
    assert "env_file" not in yaml.safe_load(rendered)["services"]["gchat-bridge"]


def test_gchat_bridge_state_paths_match_their_code_defaults() -> None:
    """The state paths' compose defaults equal the config dataclass's defaults.

    These two are the only environment vars whose ``:-`` fallback restates a code
    default instead of passing through empty: their reader is a plain
    ``e.get(NAME, default)`` call, for which "" is a value the code accepts, so
    the empty-fallback trick the neutral tunables use would point the stores at
    the empty path. That duplication is the thing this test exists to bind — the
    defaults are read back OUT of the real config class (built from an empty
    environment, so the dataclass defaults are what surface) rather than restated
    here, so changing either side alone fails.
    """
    from osprey.bridges.core import CoreConfig

    core_defaults = CoreConfig.from_env({})
    code_defaults = {
        "DEDUP_PATH": core_defaults.dedup_path,
        "HISTORY_PATH": core_defaults.history_path,
    }

    environment = _gchat_bridge_service()["environment"]
    for var, code_default in code_defaults.items():
        match = _COMPOSE_VAR_RE.match(str(environment[var]))
        assert match is not None and match.group("default") is not None, (
            f"{var} must render as ${{{var}:-<default>}} (got {environment[var]!r}) — "
            "the literal default is what keeps the path set when the var is unset, "
            "and the interpolation is what keeps it .env-overridable"
        )
        assert match.group("default") == code_default, (
            f"{var} renders {match.group('default')!r} but the config default is "
            f"{code_default!r}: the compose literal and the dataclass default must "
            "move together, or a deploy silently writes state somewhere else"
        )

    # Nothing on the host resolves to the code default, so the stores land on the
    # mounted volume; a value on the host wins, which is the override path the
    # absent bulk .env mount would otherwise have provided.
    resolved = _resolve_compose_env(environment)
    for var, code_default in code_defaults.items():
        assert resolved[var] == code_default

    overridden = _resolve_compose_env(
        environment, host_env={var: f"/srv/state/{var.lower()}.json" for var in code_defaults}
    )
    for var in code_defaults:
        assert overridden[var] == f"/srv/state/{var.lower()}.json"

    # And the override reaches the config object under the names it reads — an
    # empty value would NOT, which is why these two are not passed through with
    # an empty fallback.
    from osprey.bridges.google_chat.config import GoogleChatBridgeConfig

    cfg = GoogleChatBridgeConfig.from_env(overridden)
    assert cfg.core.dedup_path == "/srv/state/dedup_path.json"
    assert cfg.core.history_path == "/srv/state/history_path.json"


def test_gchat_bridge_depends_on_dispatcher_only_when_co_deployed() -> None:
    """``depends_on: event-dispatcher`` renders IFF the dispatcher is co-deployed.

    The bridge reconciles in-flight runs against the dispatcher before accepting
    a message, so co-deployed it must start after the dispatcher's health probe.
    A bridge pointed at an EXTERNAL dispatcher must not emit the block at all:
    compose fails hard on a ``depends_on`` naming an undefined service.
    """
    with_dispatcher = _gchat_bridge_service(dispatcher_deployed=True)
    assert with_dispatcher["depends_on"] == {"event-dispatcher": {"condition": "service_healthy"}}

    external = _render_gchat_bridge_template(dispatcher_deployed=False)
    assert "depends_on:" not in external, (
        "a bridge deployed without the dispatcher must emit no depends_on — "
        "compose errors on a dependency naming an undefined service"
    )
    assert yaml.safe_load(external)["services"]["gchat-bridge"].get("depends_on") is None


def test_gchat_bridge_dispatch_urls_track_the_dispatch_templates_ports() -> None:
    """In-network URLs use the SAME ports the dispatch templates serve on.

    Derived from the sibling templates' own rendered output rather than restated
    here, so a port change in the dispatch pair cannot leave the bridge calling a
    closed port. Service keys (not container names) are the DNS names on
    osprey-network, and the worker is addressed directly because the bridge polls
    run status and fetches artifacts from it, not through the dispatcher.
    """
    for services, expected_dispatcher_port, expected_worker_port in (
        # Config-block defaults, and explicitly non-default ports.
        (
            {"gchat_bridge": {"trigger": "t"}, "event_dispatcher": {}, "dispatch_worker": {}},
            8020,
            9190,
        ),
        (
            {
                "gchat_bridge": {"trigger": "t"},
                "event_dispatcher": {"port": 8031},
                "dispatch_worker": {"worker_port_base": 9201},
            },
            8031,
            9201,
        ),
    ):
        dispatcher = yaml.safe_load(
            _render_service_template(
                "event_dispatcher/docker-compose.yml.j2", "p", services=services
            )
        )["services"]["event-dispatcher"]
        assert dispatcher["environment"]["FASTMCP_PORT"] == str(expected_dispatcher_port), (
            "test premise: the dispatcher template must serve the port this case expects"
        )

        environment = _gchat_bridge_service(services=services)["environment"]
        assert (
            environment["DISPATCHER_URL"] == f"http://event-dispatcher:{expected_dispatcher_port}"
        )
        assert environment["WORKER_URL"] == f"http://dispatch-worker-1:{expected_worker_port}"


def test_gchat_bridge_dispatch_urls_pass_through_when_external() -> None:
    """Without the dispatch pair co-deployed, both URLs are REQUIRED host vars.

    An external dispatcher/worker is reached by whatever address the deploy env
    supplies; hardcoding the in-network name would make the bridge call a
    nonexistent host, and the code's localhost default is wrong from inside a
    container either way. They render in compose's required form
    (``${VAR:?message}``) rather than as bare references, because compose
    resolves an unset bare reference to the EMPTY STRING and not to an absent
    key: ``CoreConfig.from_env``'s localhost default would never fire,
    ``require_startup`` does not cover these two, so the stack would boot and
    then POST every dispatch to a protocol-less URL. Since a raising
    ``handle_event`` deliberately leaves the Pub/Sub message unacknowledged, that
    turns one missing variable into a subscription redelivering the same event
    forever. ``:?`` makes it a startup abort instead.
    """
    environment = _gchat_bridge_service(
        dispatcher_deployed=False,
        worker_deployed=False,
        services={"gchat_bridge": {"trigger": "t"}},
    )["environment"]
    for var in ("DISPATCHER_URL", "WORKER_URL"):
        assert str(environment[var]).startswith(f"${{{var}:?"), (
            f"{var} must render as compose's required form ${{{var}:?...}} when the "
            f"dispatch pair is external (got {environment[var]!r}) — a bare reference "
            "resolves to an empty string and boots a bridge that can never dispatch"
        )

    # Nothing on the host: the deploy must be REFUSED, not resolved to "".
    with pytest.raises(_ComposeRequiredVarUnset, match="DISPATCHER_URL"):
        _resolve_compose_env(environment)

    # Supplied, they pass through verbatim — the point of the passthrough.
    resolved = _resolve_compose_env(
        environment,
        host_env={
            "DISPATCHER_URL": "https://dispatch.example.org",
            "WORKER_URL": "https://worker.example.org",
        },
    )
    assert resolved["DISPATCHER_URL"] == "https://dispatch.example.org"
    assert resolved["WORKER_URL"] == "https://worker.example.org"

    # The CO-DEPLOYED branch must NOT carry the guard: it renders the in-network
    # address itself and needs no host variable, so a `:?` there would abort a
    # perfectly valid single-stack deploy.
    co_deployed = _gchat_bridge_service()["environment"]
    for var in ("DISPATCHER_URL", "WORKER_URL"):
        assert ":?" not in str(co_deployed[var]), (
            f"{var} is rendered in-network when the dispatch pair is co-deployed; "
            "requiring a host variable there would break the common deploy"
        )


def test_gchat_bridge_trigger_comes_from_the_profile_and_has_no_template_default() -> None:
    """``DISPATCH_TRIGGER`` is rendered from the profile block, with no fallback here.

    ``GChatBridgeProfileConfig.trigger`` is the ONLY place the ``gchat-question``
    default lives (the runtime's ``from_env`` applies none either), so a
    template-side default would be a second definition that fires some other
    facility's trigger when the config key goes missing.
    """
    from osprey.cli.build_profile import GChatBridgeProfileConfig

    profile_default = GChatBridgeProfileConfig().trigger
    rendered = _render_gchat_bridge_template(
        services={
            "gchat_bridge": {"trigger": profile_default},
            "event_dispatcher": {},
            "dispatch_worker": {},
        }
    )
    environment = yaml.safe_load(rendered)["services"]["gchat-bridge"]["environment"]
    assert environment["DISPATCH_TRIGGER"] == profile_default

    # A facility-chosen trigger must render verbatim, and the profile default
    # must not survive as a template-side fallback.
    custom = _render_gchat_bridge_template(
        services={
            "gchat_bridge": {"trigger": "als-chat-question"},
            "event_dispatcher": {},
            "dispatch_worker": {},
        }
    )
    custom_env = yaml.safe_load(custom)["services"]["gchat-bridge"]["environment"]
    assert custom_env["DISPATCH_TRIGGER"] == "als-chat-question"

    # A config that lost the key must render an EMPTY trigger, so the bridge
    # aborts at boot naming DISPATCH_TRIGGER. A template-side default would
    # instead fire whatever name the framework happened to pick.
    keyless = _render_gchat_bridge_template(
        services={"gchat_bridge": {}, "event_dispatcher": {}, "dispatch_worker": {}}
    )
    keyless_env = yaml.safe_load(keyless)["services"]["gchat-bridge"]["environment"]
    assert keyless_env["DISPATCH_TRIGGER"] == "", (
        f"a missing trigger key must render empty, not fall back to {profile_default!r} — "
        "the profile block is the single source of the trigger name"
    )


def test_gchat_bridge_rendered_env_parses_and_fails_closed_without_secrets() -> None:
    """The rendered env, resolved with nothing set on the host, refuses to boot.

    Feeds the template's own output through the real
    ``GoogleChatBridgeConfig.from_env`` — so a renamed variable shows up as dead
    config here rather than as a bridge that silently ignores it — and asserts
    ``require_startup`` names exactly the fail-closed variables. The trigger is
    absent from that list precisely because the template renders it as a literal.
    """
    from osprey.bridges.google_chat.config import GoogleChatBridgeConfig

    environment = _gchat_bridge_service()["environment"]
    cfg = GoogleChatBridgeConfig.from_env(_resolve_compose_env(environment))

    with pytest.raises(ValueError) as excinfo:
        cfg.require_startup()
    missing = {name.strip() for name in str(excinfo.value).split(":", 1)[1].split(",")}
    assert missing == set(_GCHAT_FAIL_CLOSED_VARS), (
        "with nothing set on the host the bridge must abort naming exactly the "
        f"bare-reference variables; got {sorted(missing)}"
    )


def test_gchat_bridge_rendered_env_boots_with_host_secrets_supplied() -> None:
    """With the .env supplying the credentials, the same block passes the boot gate.

    Proves the variable NAMES the template renders are the names the runtime
    reads: the Google settings, both tokens, the trigger, and the in-network
    dispatch endpoints all arrive on the config object, and the state paths stay
    on the /data volume. ``require_boot`` (not just ``require_startup``) is what
    is called, so the dispatcher/worker URLs the co-deployed branch renders are
    checked too.
    """
    from osprey.bridges.google_chat.config import GoogleChatBridgeConfig, require_boot

    environment = _gchat_bridge_service()["environment"]
    resolved = _resolve_compose_env(
        environment,
        host_env={
            "GCHAT_SA_KEY": "/secrets/gchat-sa.json",
            "GCHAT_SUBSCRIPTION": _GCHAT_SUBSCRIPTION,
            "GCHAT_APP_ID": "users/1234567890",
            "EVENT_DISPATCHER_TOKEN": "dispatcher-token",
            "DISPATCH_WORKER_TOKEN": "worker-token",
        },
    )
    cfg = GoogleChatBridgeConfig.from_env(resolved)
    require_boot(cfg)

    assert cfg.sa_key == "/secrets/gchat-sa.json"
    assert cfg.subscription == _GCHAT_SUBSCRIPTION
    assert cfg.app_id == "users/1234567890"
    assert cfg.core.event_dispatcher_token == "dispatcher-token"
    assert cfg.core.dispatch_worker_token == "worker-token"
    assert cfg.core.trigger == "gchat-question"
    assert cfg.core.dispatcher_url == "http://event-dispatcher:8020"
    assert cfg.core.worker_url == "http://dispatch-worker-1:9190"
    # Both state files must land on the mounted volume, not the image layer.
    for path in (cfg.core.dedup_path, cfg.core.history_path):
        assert path.startswith("/data/"), path


@pytest.mark.parametrize("worker_timeout", [None, 600])
def test_gchat_bridge_poll_budget_default_outlasts_the_worker_cap(
    worker_timeout: int | None,
) -> None:
    """The bridge's poll budget is derived from the worker's cap, not restated.

    ``CoreConfig.__post_init__`` rejects ``poll_budget < worker_timeout``, which
    would crash-loop the bridge, and both halves read the cap from the same
    ``services.dispatch_worker.timeout_sec`` key — so raising the cap in one
    place must raise the budget here too. Constructing the config from the
    resolved defaults is what proves the relation holds rather than asserting on
    the numbers alone.
    """
    from osprey.bridges.core import CoreConfig

    worker_config = {} if worker_timeout is None else {"timeout_sec": worker_timeout}
    environment = _gchat_bridge_service(
        services={
            "gchat_bridge": {"trigger": "t"},
            "event_dispatcher": {},
            "dispatch_worker": worker_config,
        }
    )["environment"]

    cfg = CoreConfig.from_env(_resolve_compose_env(environment))
    expected_timeout = float(worker_timeout if worker_timeout is not None else 300)
    assert cfg.worker_timeout == expected_timeout
    assert cfg.poll_budget == expected_timeout + 30, (
        "the poll budget default must exceed the worker cap it waits out"
    )


def test_gchat_bridge_config_lookups_survive_explicit_null_values() -> None:
    """A config key present but EMPTY must still render the documented default.

    Jinja's ``default`` filter substitutes only on *Undefined*, so a key written
    as ``timeout_sec:`` with no value — which YAML loads as ``None`` — sails past
    a plain ``| default(300)`` and renders the literal string "None".
    ``CoreConfig.from_env`` then dies on ``float("None")`` at boot, and
    ``None | int`` silently degrades ``POLL_BUDGET`` to 0. The boolean form
    (``default(300, true)``) substitutes on any falsy value, which is what keeps
    a half-written config booting on the defaults instead of crash-looping.
    """
    from osprey.bridges.core import CoreConfig

    environment = _gchat_bridge_service(
        services={
            "gchat_bridge": {"trigger": "t"},
            "event_dispatcher": {"port": None},
            "dispatch_worker": {"timeout_sec": None, "worker_port_base": None},
        }
    )["environment"]

    assert "None" not in str(environment["DISPATCH_TIMEOUT_SEC"])
    assert environment["DISPATCHER_URL"] == "http://event-dispatcher:8020"
    assert environment["WORKER_URL"] == "http://dispatch-worker-1:9190"

    cfg = CoreConfig.from_env(_resolve_compose_env(environment))
    assert cfg.worker_timeout == 300.0
    assert cfg.poll_budget == 330.0


def test_gchat_bridge_container_name_is_project_namespaced() -> None:
    """Two projects render distinct bridge container names.

    ``container_name`` is a HOST-GLOBAL docker identifier, so a static name stops
    two OSPREY projects from running a bridge on one host. Nothing reaches this
    service in-network (it only makes outbound calls), so no network alias is
    needed alongside the rename.
    """
    name_a = _gchat_bridge_service(project_name="proj-a")["container_name"]
    name_b = _gchat_bridge_service(project_name="proj-b")["container_name"]
    assert (name_a, name_b) == ("proj-a-gchat-bridge", "proj-b-gchat-bridge")


def test_gchat_bridge_publishes_no_ports_and_declares_no_healthcheck() -> None:
    """The bridge is a subscriber: no listening socket, so no ports and no probe.

    Google Chat reaches it through Pub/Sub, never over an inbound HTTP request,
    so a published port would be dead surface — and a healthcheck against a
    service that opens no socket would mark a healthy bridge unhealthy, which
    ``depends_on: service_healthy`` elsewhere would then act on.
    """
    service = _gchat_bridge_service()
    assert "ports" not in service
    assert "healthcheck" not in service
    assert service["networks"] == ["osprey-network"]
    assert service["restart"] == "unless-stopped"


def test_gchat_bridge_template_is_bundled_into_a_declaring_project(tmp_path: Path) -> None:
    """``osprey build`` copies the packaged bridge template into the project tree.

    The whole service directory (compose template, Dockerfile, .dockerignore)
    must ship in the package and be discoverable under the ``gchat_bridge``
    service key, or ``osprey up`` has nothing to render and no build
    context to build.
    """
    _write_config(tmp_path, deployed_services=["gchat_bridge"])

    assert _copy_service_templates(tmp_path) == 1

    service_dir = tmp_path / "services" / "gchat_bridge"
    assert (service_dir / "docker-compose.yml.j2").is_file()
    assert (service_dir / "Dockerfile").is_file(), (
        "the build context needs its Dockerfile — the compose template declares "
        "build: ./gchat_bridge with dockerfile: Dockerfile"
    )
    assert (service_dir / ".dockerignore").is_file(), (
        ".dockerignore is the guaranteed COPY sibling the Dockerfile's optional "
        "wheel/requirements globs rely on, and it keeps a stale .env out of the image"
    )


def test_gchat_bridge_image_installs_the_gchat_extra_on_both_install_lines() -> None:
    """Both framework installs carry ``[gchat]`` — the pinned one and the dev fallback.

    The Google client libraries (Pub/Sub, Chat, GCS) live behind the extra, so an
    install without it produces an image whose bridge dies on its first import.
    The dev fallback matters as much as the pin: ``osprey up --dev``
    against an unreleased version takes that branch, and it is the branch a
    plain copy of a sibling service's Dockerfile would leave unextra'd.
    """
    from importlib import resources

    dockerfile = (
        resources.files("osprey")
        .joinpath("templates/services/gchat_bridge/Dockerfile")
        .read_text(encoding="utf-8")
    )
    assert 'pip install --no-cache-dir "osprey-framework[gchat]==$OSPREY_VERSION"' in dockerfile
    assert 'pip install --no-cache-dir "osprey-framework[gchat]"' in dockerfile
    # A bare (extra-less) framework install anywhere would silently win or waste
    # a layer depending on order, so neither spelling may survive.
    assert '"osprey-framework==$OSPREY_VERSION"' not in dockerfile
    assert '"osprey-framework"' not in dockerfile


def test_find_existing_compose_files_answers_from_an_explicit_base(tmp_path, monkeypatch) -> None:
    """The lookup follows its ``base``, not the working directory.

    Regression guard: both ``build_dir`` and a service's declared ``path`` are
    relative, so from the wrong directory this function found nothing — and
    "no compose files" is what an empty deployment looks like too, so the wrong
    answer arrived with no error attached to it.
    """
    from osprey.deployment.compose_generator import find_existing_compose_files

    repo = tmp_path / "repo"
    (repo / "build" / "services" / "osprey" / "jupyter").mkdir(parents=True)
    (repo / "build" / "services" / "docker-compose.yml").write_text("services: {}\n")
    (repo / "build" / "services" / "osprey" / "jupyter" / "docker-compose.yml").write_text(
        "services: {}\n"
    )
    config = {
        "build_dir": "./build",
        "services": {"jupyter": {"path": "services/osprey/jupyter"}},
    }

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    found = find_existing_compose_files(config, ["jupyter"], quiet=True, base=repo)

    assert found == [
        "./build/services/docker-compose.yml",
        "./build/services/osprey/jupyter/docker-compose.yml",
    ]
    # The same call without the anchor, from the same foreign directory, is the
    # failure the base exists to prevent.
    assert find_existing_compose_files(config, ["jupyter"], quiet=True) == []


# ---------------------------------------------------------------------------
# The worker reaches the archive it can actually reach
# ---------------------------------------------------------------------------

_ARCHIVER_HOST_ENV = "OSPREY_ARCHIVER_MONGODB_HOST"
_ARCHIVER_PORT_ENV = "OSPREY_ARCHIVER_MONGODB_PORT"


def test_worker_is_pointed_at_the_store_this_project_deploys() -> None:
    """The worker's agent reads history like any other, but config.yml's
    connection block is written for the HOST side. Inside the network that names
    this container's own loopback, so the deploy hands it the store's alias.

    The alias, never the container name: `container_name` carries the project
    prefix, and the alias is pinned in the mongodb template precisely so this
    reference survives being deployed under any project name.
    """
    rendered = _render_worker_template(
        env_present=True, deployed_services=["dispatch_worker", "mongodb"]
    )

    assert f"{_ARCHIVER_HOST_ENV}: archiver-mongodb" in rendered
    assert f'{_ARCHIVER_PORT_ENV}: "27017"' in rendered
    assert f"{_ARCHIVER_HOST_ENV}: {_WORKER_PROJECT_NAME}-archiver-mongodb" not in rendered


def test_worker_is_not_pointed_at_a_store_this_project_does_not_deploy() -> None:
    """These two are a LITERAL address, not a fallback — so a project reading a
    facility's own MongoDB must not get them. Its configured block is already
    correct from anywhere, and `archiver-mongodb` would resolve to nothing.
    """
    rendered = _render_worker_template(env_present=True, deployed_services=["dispatch_worker"])

    assert _ARCHIVER_HOST_ENV not in rendered
    assert _ARCHIVER_PORT_ENV not in rendered


def test_the_worker_address_is_the_one_the_connector_would_dial() -> None:
    """The integration-level half, asserted at the address-computation level so
    it needs no Mongo and no Docker: feed the connector's own override reader
    exactly what this compose file exports, and it yields the alias and the
    container port — which is what `archiver_read` inside the worker connects
    to. The Docker-level proof of the same claim rides the archiver-world e2e.
    """
    from osprey.connectors.archiver.mongodb_archiver_connector import address_overrides

    rendered = _render_worker_template(
        env_present=True, deployed_services=["dispatch_worker", "mongodb"]
    )
    exported = dict(
        re.findall(r"^\s+(OSPREY_ARCHIVER_MONGODB_\w+):\s*\"?([^\"\n]+)\"?$", rendered, re.M)
    )
    assert set(exported) == {_ARCHIVER_HOST_ENV, _ARCHIVER_PORT_ENV}, exported

    with mock.patch.dict(os.environ, exported, clear=False):
        assert address_overrides() == ("archiver-mongodb", 27017)
