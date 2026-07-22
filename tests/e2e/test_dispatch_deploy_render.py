"""Render-only e2e for the dispatch stack's compose generation (no Docker, no token).

Sits between the unit tier (which renders the dispatch templates in isolation via
direct ``jinja2 template.render(**config)``, and mocks ``prepare_compose_files``
wholesale in the deploy CLI tests) and the full-Docker dispatch lanes
(``test_dispatch_deploy.py`` / ``test_dispatch_overlay_visibility.py``, gated on a
running daemon + ``ALS_APG_API_KEY``). Neither tier ever drives the REAL
``osprey deploy build`` render pipeline — config load, project-metadata injection,
per-service ``setup_build_dir`` + ``render_template`` — end to end on a
dispatch-enabled project.

This does exactly that, cheaply: a real ``osprey deploy build`` subprocess against a
minimal project whose only deployed services are the event-dispatcher and
dispatch-worker (staged with the same bundled templates ``osprey build`` copies in,
via ``_inject_dispatch``). ``build`` is pure rendering — it never touches the
container runtime or a provider token — so this runs in any CI lane.

It pins the render-time wiring the deployed stack depends on and that only the
Docker lanes otherwise exercise:

  * both service ``container_name``s are project-namespaced (two OSPREY projects
    can deploy on one host without colliding);
  * the worker's ``image`` defaults to the project image ``<project>:local``;
  * the worker's provider-auth ``env_file: ../../.env`` block renders ONLY when a
    project ``.env`` exists (``osprey_env_present``) — the load-bearing wiring that
    delivers the LLM key to the non-root worker;
  * both tokens fail closed (``${...}`` with no ``:-`` default);
  * a multi-worker / shared-workspace config renders one service + volume per
    worker with the documented names.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from importlib.resources import as_file, files
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from osprey.cli.build_cmd import _inject_dispatch
from osprey.cli.build_profile import DispatchConfig

PROJECT_NAME = "e2e-dispatch-render"

DEPLOY_BUILD_TIMEOUT_SEC = 120


def _find_osprey_console_script() -> Path:
    candidate = Path(sys.executable).parent / "osprey"
    if candidate.exists():
        return candidate
    found = shutil.which("osprey")
    if found:
        return Path(found)
    raise RuntimeError("Could not locate the 'osprey' console script.")


def _stage_dispatch_project(
    root: Path,
    *,
    with_env: bool = True,
    worker_count: int = 1,
    workspace_mode: str = "isolated",
) -> Path:
    """Build a minimal render-only project deploying only the dispatch services.

    Uses ``_inject_dispatch`` — the same build step ``osprey build`` runs — to copy
    the bundled event_dispatcher/dispatch_worker compose templates and register
    both in ``deployed_services``, then stages the top-level services template a
    real ``osprey build`` would scaffold. No container image is ever built.
    """
    project = root / "project"
    project.mkdir()

    yaml = YAML()
    config = {
        "project_name": PROJECT_NAME,
        "facility": {"name": "E2E Dispatch Render", "prefix": "e2edr", "timezone": "UTC"},
        "deployed_services": [],
        "services": {},
        "system": {"timezone": "UTC"},
    }
    with open(project / "config.yml", "w") as fh:
        yaml.dump(config, fh)

    dispatch = DispatchConfig(
        triggers="tutorial_triggers.yml",
        worker_count=worker_count,
        workspace_mode=workspace_mode,  # type: ignore[arg-type]
    )
    profile_dir = root / "profile"  # empty — forces bundled trigger resolution
    profile_dir.mkdir()
    _inject_dispatch(dispatch, profile_dir=profile_dir, project_path=project)

    # prepare_compose_files always renders the top-level services template via a
    # CWD-relative loader; a real `osprey build` scaffolds it, so a hand-built
    # project must stage it too (same rationale as test_deploy_lifecycle.py).
    top_template = files("osprey").joinpath("templates/services/docker-compose.yml.j2")
    with as_file(top_template) as template_path:
        shutil.copy(template_path, project / "services" / "docker-compose.yml.j2")

    if with_env:
        # Presence (not contents) is what flips osprey_env_present -> the worker's
        # env_file provider-auth block. The compose CLI reads the file at deploy
        # time; render only needs it to exist.
        (project / ".env").write_text(
            "EVENT_DISPATCHER_TOKEN=x\nDISPATCH_WORKER_TOKEN=y\n", encoding="utf-8"
        )

    return project


def _run_deploy_build(project: Path) -> subprocess.CompletedProcess:
    osprey_bin = _find_osprey_console_script()
    result = subprocess.run(
        [str(osprey_bin), "deploy", "build"],
        cwd=str(project),
        capture_output=True,
        text=True,
        timeout=DEPLOY_BUILD_TIMEOUT_SEC,
        env={**os.environ, "CLAUDECODE": ""},
    )
    assert result.returncode == 0, (
        f"osprey deploy build failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    return result


def _rendered(project: Path, service: str) -> str:
    path = project / "build" / "services" / service / "docker-compose.yml"
    assert path.is_file(), f"{service} compose file was not rendered at {path}"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def dispatch_project(tmp_path: Path) -> Callable[..., Path]:
    """Factory: build a render-only dispatch project with the given knobs."""

    def _make(**kwargs: object) -> Path:
        sub = tmp_path / f"proj-{len(list(tmp_path.iterdir()))}"
        sub.mkdir()
        return _stage_dispatch_project(sub, **kwargs)  # type: ignore[arg-type]

    return _make


def test_deploy_build_renders_dispatch_wiring(dispatch_project: Callable[..., Path]) -> None:
    """`osprey deploy build` renders both dispatch services with the deployed-stack
    wiring: project-namespaced container names, project-image worker default, the
    provider-auth env_file block, and fail-closed tokens.
    """
    project = dispatch_project(with_env=True)
    _run_deploy_build(project)

    worker = _rendered(project, "dispatch_worker")
    dispatcher = _rendered(project, "event_dispatcher")

    # container_name is host-global — both must be namespaced by project so two
    # deployments can coexist on one host.
    assert f"container_name: {PROJECT_NAME}-dispatch-worker-1" in worker
    assert f"container_name: {PROJECT_NAME}-event-dispatcher" in dispatcher

    # The worker runs the project image; its default resolves to <project>:local.
    assert f"${{OSPREY_WORKER_IMAGE:-{PROJECT_NAME}:local}}" in worker
    # The dispatcher builds its own project-prefixed dispatch image.
    assert f"${{OSPREY_DISPATCH_IMAGE:-{PROJECT_NAME}-dispatch:local}}" in dispatcher

    # Provider-auth wiring: a project .env exists, so the worker gets its env_file.
    assert "env_file:" in worker
    assert "../../.env" in worker

    # Both tokens fail closed — no ":-" default that would boot with a guessable
    # secret instead of refusing.
    assert "DISPATCH_WORKER_TOKEN: ${DISPATCH_WORKER_TOKEN}" in worker
    assert "EVENT_DISPATCHER_TOKEN: ${EVENT_DISPATCHER_TOKEN}" in dispatcher
    assert "${DISPATCH_WORKER_TOKEN:-" not in worker
    assert "${EVENT_DISPATCHER_TOKEN:-" not in dispatcher


def test_deploy_build_omits_env_file_without_dotenv(
    dispatch_project: Callable[..., Path],
) -> None:
    """Without a project ``.env`` the worker's provider-auth ``env_file`` block must
    NOT render — the block is conditional on ``osprey_env_present``, and emitting a
    non-existent ``env_file`` would make ``docker compose`` hard-fail at deploy.
    """
    project = dispatch_project(with_env=False)
    _run_deploy_build(project)

    worker = _rendered(project, "dispatch_worker")
    assert "env_file:" not in worker
    assert "../../.env" not in worker
    # The rest of the worker wiring is unaffected by the .env's absence.
    assert f"container_name: {PROJECT_NAME}-dispatch-worker-1" in worker


def test_deploy_build_multi_worker_shared_workspace(
    dispatch_project: Callable[..., Path],
) -> None:
    """A multi-worker, shared-workspace config renders one service per worker and a
    single shared workspace volume (isolated mode would render one volume each).
    """
    project = dispatch_project(with_env=True, worker_count=2, workspace_mode="shared")
    _run_deploy_build(project)

    worker = _rendered(project, "dispatch_worker")
    assert f"container_name: {PROJECT_NAME}-dispatch-worker-1" in worker
    assert f"container_name: {PROJECT_NAME}-dispatch-worker-2" in worker
    # Shared mode: one un-suffixed workspace volume both workers mount, not the
    # per-worker dispatch_workspace_1 / _2 of isolated mode.
    assert "dispatch_workspace:" in worker
    assert "dispatch_workspace_1:" not in worker
