"""Render-only e2e for the dispatch stack's compose generation (no Docker, no token).

Sits between the unit tier (which renders the dispatch templates in isolation via
direct ``jinja2 template.render(**config)``, and mocks ``prepare_compose_files``
wholesale in the deploy CLI tests) and the full-Docker dispatch lanes
(``test_dispatch_deploy.py`` / ``test_dispatch_overlay_visibility.py``, gated on a
running daemon + ``ALS_APG_API_KEY``). Neither tier ever drives the REAL
``osprey build`` render pipeline — profile resolution, config render,
project-metadata injection, per-service ``setup_build_dir`` + ``render_template``
— end to end on a dispatch-enabled deployment repo.

This does exactly that: a real ``osprey init`` + ``osprey build`` pair against a
deployment repo whose profile adds the ``dispatch:`` block, so the build's own
``_inject_dispatch`` step stages the bundled event_dispatcher/dispatch_worker
templates and registers both in ``deployed_services``. ``build`` renders files and
never containers, and needs no provider token, so this runs in any CI lane.

It pins the render-time wiring the deployed stack depends on and that only the
Docker lanes otherwise exercise:

  * both service ``container_name``s are project-namespaced (two OSPREY
    deployments can run on one host without colliding);
  * the worker's ``image`` defaults to the project image ``<project>:local``;
  * the worker's provider-auth ``env_file: ./.env`` block renders ONLY when the
    repo's ``.env`` exists (``osprey_env_present``) — the load-bearing wiring that
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
from pathlib import Path

import pytest

#: The deployment repo's directory name IS the deployment's name, and the name
#: the compose templates namespace every container by.
PROJECT_NAME = "e2e-dispatch-render"

INIT_TIMEOUT_SEC = 300
BUILD_TIMEOUT_SEC = 300


def _find_osprey_console_script() -> Path:
    candidate = Path(sys.executable).parent / "osprey"
    if candidate.exists():
        return candidate
    found = shutil.which("osprey")
    if found:
        return Path(found)
    raise RuntimeError("Could not locate the 'osprey' console script.")


def _run(cmd: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "CLAUDECODE": ""},
    )


def _init_dispatch_repo(
    root: Path,
    *,
    with_env: bool = True,
    worker_count: int = 1,
    workspace_mode: str = "isolated",
) -> Path:
    """Init + build a deployment repo that deploys the dispatch services.

    A deployment repo is one directory marked by ``profile.yml`` at its root, so
    the fixture material here is a profile: ``hello-world`` declares no dispatch
    of its own, and the override layers on the ``dispatch:`` block that makes the
    build's ``_inject_dispatch`` step copy the bundled
    event_dispatcher/dispatch_worker compose templates in and register both in
    ``deployed_services``. No container image is ever built.

    Returns the repo root; the render is at ``<repo>/build``.
    """
    osprey_bin = _find_osprey_console_script()
    repo = root / PROJECT_NAME

    override_path = root / "dispatch.yml"
    override_path.write_text(
        "dispatch:\n"
        "  triggers: tutorial_triggers.yml\n"
        f"  worker_count: {worker_count}\n"
        f"  workspace_mode: {workspace_mode}\n",
        encoding="utf-8",
    )

    init = _run(
        [
            str(osprey_bin),
            "init",
            str(repo),
            "--preset",
            "hello-world",
            "--no-git",
            "--override",
            str(override_path),
        ],
        cwd=root,
        timeout=INIT_TIMEOUT_SEC,
    )
    assert init.returncode == 0, (
        f"osprey init failed (rc={init.returncode}):\n"
        f"--- stdout ---\n{init.stdout}\n--- stderr ---\n{init.stderr}"
    )

    # Decided BEFORE the build, because the render probes the repo root for it:
    # `osprey_env_present` is what gates the worker's env_file block, and
    # `osprey init` seeds a .env only when the shell exports a provider key —
    # which would make this knob depend on the runner's environment.
    env_path = repo / ".env"
    if with_env:
        # Presence (not contents) is what flips osprey_env_present. The compose
        # CLI reads the file at deploy time; render only needs it to exist.
        env_path.write_text("EVENT_DISPATCHER_TOKEN=x\nDISPATCH_WORKER_TOKEN=y\n", encoding="utf-8")
    elif env_path.exists():
        env_path.unlink()

    build = _run(
        [str(osprey_bin), "build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle"],
        cwd=root,
        timeout=BUILD_TIMEOUT_SEC,
    )
    assert build.returncode == 0, (
        f"osprey build failed (rc={build.returncode}):\n"
        f"--- stdout ---\n{build.stdout}\n--- stderr ---\n{build.stderr}"
    )
    return repo


def _rendered(repo: Path, service: str) -> str:
    path = repo / "build" / "services" / service / "docker-compose.yml"
    assert path.is_file(), f"{service} compose file was not rendered at {path}"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def dispatch_repo(tmp_path: Path) -> Callable[..., Path]:
    """Factory: init + build a dispatch-enabled deployment repo with the given knobs."""

    def _make(**kwargs: object) -> Path:
        sub = tmp_path / f"case-{len(list(tmp_path.iterdir()))}"
        sub.mkdir()
        return _init_dispatch_repo(sub, **kwargs)  # type: ignore[arg-type]

    return _make


def test_build_renders_dispatch_wiring(dispatch_repo: Callable[..., Path]) -> None:
    """`osprey build` renders both dispatch services with the deployed-stack
    wiring: project-namespaced container names, project-image worker default, the
    provider-auth env_file block, and fail-closed tokens.
    """
    repo = dispatch_repo(with_env=True)

    worker = _rendered(repo, "dispatch_worker")
    dispatcher = _rendered(repo, "event_dispatcher")

    # container_name is host-global — both must be namespaced by project so two
    # deployments can coexist on one host.
    assert f"container_name: {PROJECT_NAME}-dispatch-worker-1" in worker
    assert f"container_name: {PROJECT_NAME}-event-dispatcher" in dispatcher

    # The worker runs the project image; its default resolves to <project>:local.
    assert f"${{OSPREY_WORKER_IMAGE:-{PROJECT_NAME}:local}}" in worker
    # The dispatcher builds its own project-prefixed dispatch image.
    assert f"${{OSPREY_DISPATCH_IMAGE:-{PROJECT_NAME}-dispatch:local}}" in dispatcher

    # Provider-auth wiring: both chain files exist, so the worker reads both —
    # in ASCENDING precedence, the local `.env` last, since compose lets a later
    # entry win. Compared as a list rather than by substring: `- ./.env.shared`
    # contains `- ./.env`, so an `in` check cannot tell the two apart and would
    # pass on a render that dropped the local file entirely.
    assert "env_file:" in worker
    chain = [line.strip() for line in worker.splitlines() if line.strip().startswith("- ./.env")]
    assert chain == ["- ./.env.shared", "- ./.env"], chain

    # Both tokens fail closed — no ":-" default that would boot with a guessable
    # secret instead of refusing.
    assert "DISPATCH_WORKER_TOKEN: ${DISPATCH_WORKER_TOKEN}" in worker
    assert "EVENT_DISPATCHER_TOKEN: ${EVENT_DISPATCHER_TOKEN}" in dispatcher
    assert "${DISPATCH_WORKER_TOKEN:-" not in worker
    assert "${EVENT_DISPATCHER_TOKEN:-" not in dispatcher


def test_build_omits_the_missing_local_env_from_the_chain(
    dispatch_repo: Callable[..., Path],
) -> None:
    """Without the repo's ``.env`` the chain must not name it.

    Membership is fixed at render time and ``docker compose`` hard-fails on an
    ``env_file:`` entry that is not there, so a missing file has to be left out.
    The block itself still renders: ``osprey init`` writes ``.env.shared``, so a
    repo with no ``.env`` still has a chain — a one-file one.

    Matched per line rather than by substring, because ``- ./.env.shared``
    contains ``- ./.env``: a substring check for the local file passes on the
    shared one and would call this green no matter what rendered.
    """
    repo = dispatch_repo(with_env=False)

    worker = _rendered(repo, "dispatch_worker")
    chain = [line.strip() for line in worker.splitlines() if line.strip().startswith("- ./.env")]
    assert chain == ["- ./.env.shared"], chain
    # The rest of the worker wiring is unaffected by the .env's absence.
    assert f"container_name: {PROJECT_NAME}-dispatch-worker-1" in worker


def test_build_multi_worker_shared_workspace(
    dispatch_repo: Callable[..., Path],
) -> None:
    """A multi-worker, shared-workspace config renders one service per worker and a
    single shared workspace volume (isolated mode would render one volume each).
    """
    repo = dispatch_repo(with_env=True, worker_count=2, workspace_mode="shared")

    worker = _rendered(repo, "dispatch_worker")
    assert f"container_name: {PROJECT_NAME}-dispatch-worker-1" in worker
    assert f"container_name: {PROJECT_NAME}-dispatch-worker-2" in worker
    # Shared mode: one un-suffixed workspace volume both workers mount, not the
    # per-worker dispatch_workspace_1 / _2 of isolated mode.
    assert "dispatch_workspace:" in worker
    assert "dispatch_workspace_1:" not in worker
