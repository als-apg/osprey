"""Framework-guard tests for projects with empty deployed_services.

The hello-world preset (and any future "agent-only" preset) declares no
deployed_services. Two failure modes have to stay fixed:

1. ``osprey build`` must still copy the root ``services/docker-compose.yml.j2``
   into the project, because the renderer references it unconditionally.
2. ``osprey deploy up`` must succeed (graceful no-op) instead of dying with
   ``TemplateNotFound`` mid-render.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ruamel.yaml import YAML

from osprey.cli.build_cmd import _copy_service_templates
from osprey.deployment.compose_generator import prepare_compose_files


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
        "deployed_services is empty (otherwise prepare_compose_files dies "
        "with TemplateNotFound at render time)"
    )


def test_prepare_compose_files_no_services_renders_root_only(tmp_path: Path) -> None:
    """With empty deployed_services, prepare_compose_files renders just the root."""
    config_path = _write_config(tmp_path, deployed_services=[])
    _copy_service_templates(tmp_path)

    monkey_cwd = Path.cwd()
    try:
        # render_template resolves SERVICES_DIR relative to cwd
        import os

        os.chdir(tmp_path)
        config, compose_files = prepare_compose_files(str(config_path))
    finally:
        import os

        os.chdir(monkey_cwd)

    assert len(compose_files) == 1, (
        f"expected exactly one rendered compose file (the root), got {compose_files}"
    )
    rendered = Path(compose_files[0])
    assert rendered.is_file()
    rendered_text = rendered.read_text()
    assert "services:" not in rendered_text, (
        f"empty-deployed-services preset rendered a compose with services block:\n{rendered_text}"
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
# Its startup hook (``inject_provider_env``) reads ``$OSPREY_PROJECT_DIR/.env``,
# so the worker compose service must mount the project ``.env`` read-only —
# otherwise dispatched runs cannot authenticate (status "error") in any real
# container deploy. Gated on ``.env`` existence to avoid docker auto-creating a
# stray empty ``.env`` directory when none is present.
# ---------------------------------------------------------------------------

_ENV_MOUNT = "../../.env:/app/project/.env:ro"


def _render_worker_template(*, env_present: bool) -> str:
    # Load the packaged template directly (CWD-independent — other tests in the
    # suite may chdir, so a FileSystemLoader(".") lookup is not safe here).
    from importlib import resources

    from jinja2 import Template

    tpl = resources.files("osprey").joinpath(
        "templates/services/dispatch_worker/docker-compose.yml.j2"
    )
    template = Template(tpl.read_text(encoding="utf-8"))
    return template.render(
        services={"dispatch_worker": {}},
        system={"timezone": "UTC"},
        osprey_labels={"project_name": "p", "project_root": "/r", "deployed_at": "now"},
        osprey_version="",
        osprey_env_present=env_present,
    )


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
    """The worker must NOT declare its own build for the shared image tag.

    event-dispatcher and dispatch-worker share osprey-dispatch:local. If both
    declared `build:`, `docker compose up` builds them concurrently and the two
    exports race to tag the image — one fails with
    ``ERROR: image "osprey-dispatch:local": already exists`` (deterministic once
    base layers are cached). event-dispatcher is the sole builder; the worker
    references its image and depends_on it.
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


def test_worker_template_mounts_env_when_present() -> None:
    rendered = _render_worker_template(env_present=True)
    assert _ENV_MOUNT in rendered, (
        "dispatch worker must mount the project .env so the agent can "
        "authenticate to the LLM provider"
    )


def test_worker_template_omits_env_mount_when_absent() -> None:
    rendered = _render_worker_template(env_present=False)
    assert _ENV_MOUNT not in rendered, (
        "no .env mount should be emitted when the project has no .env "
        "(avoids docker creating a stray empty .env directory)"
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
    assert "CONFIG_FILE: /app/project/config.yml" in rendered, (
        "dispatch worker must set CONFIG_FILE so the worker process (and the CLI "
        "subprocess it spawns) resolve config from the mounted project, not /app"
    )


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
