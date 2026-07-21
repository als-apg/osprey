"""Unit tests for the web-terminal deploy orchestration entrypoints.

Covers ``osprey.deployment.web_terminals.provision`` in isolation: the web
stack's own ``compose down`` and the post-``up`` image-drift reconcile. The
deploy_up-entry orchestration that wires the provisioning modules together
lives in ``tests/deployment/test_container_lifecycle.py``; the split-out
provisioning steps have their own modules and test files
(``test_persona_images.py``, ``test_env_production.py``,
``test_postup_hooks.py``).
"""

from __future__ import annotations

import os

from osprey.deployment.web_terminals import provision


class _FakeCompletedProcess:
    """Minimal stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# ---------------------------------------------------------------------------
# deploy_down_web_terminals -- the web stack's own `compose down`
# ---------------------------------------------------------------------------


def test_deploy_down_web_terminals_runs_compose_down_on_web_file(monkeypatch, tmp_path):
    """With a rendered docker-compose.web.yml at the project root, the web
    stack gets its own `compose -f docker-compose.web.yml down` under the
    pinned compose project — the mirror of deploy_up_web_terminals' second
    invocation. Without it the fixed-name `<prefix>-web-<user>`/`<prefix>-nginx`
    containers outlive every `osprey deploy down` and the next web-terminals
    deploy on the host dies at `up` with a container-name Conflict."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docker-compose.web.yml").write_text("services: {}\n", encoding="utf-8")

    recorded: dict = {}

    def _fake_run(cmd, **kwargs):
        recorded["cmd"] = cmd
        recorded["env"] = kwargs.get("env")
        return _FakeCompletedProcess()

    monkeypatch.setattr(provision.subprocess, "run", _fake_run)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config=None: ["docker", "compose"])

    provision.deploy_down_web_terminals(
        {"project_name": "myproj"}, dict(os.environ), ["--env-file", ".env"]
    )

    assert recorded["cmd"] == [
        "docker",
        "compose",
        "-f",
        "docker-compose.web.yml",
        "--env-file",
        ".env",
        "down",
    ]
    assert recorded["env"]["COMPOSE_PROJECT_NAME"] == "myproj"


def test_deploy_down_web_terminals_noop_without_web_file(monkeypatch, tmp_path):
    """No rendered web compose file (nothing was ever deployed from this root,
    or the render predates web terminals) → no compose invocation at all."""
    monkeypatch.chdir(tmp_path)

    def _unexpected_run(cmd, **kwargs):
        raise AssertionError(f"unexpected subprocess.run: {cmd}")

    monkeypatch.setattr(provision.subprocess, "run", _unexpected_run)

    provision.deploy_down_web_terminals({"project_name": "myproj"}, dict(os.environ), [])


# ---------------------------------------------------------------------------
# _reconcile_web_stack_image_drift -- post-`up` force-recreate on digest change
# ---------------------------------------------------------------------------

_WEB_COMPOSE = (
    "services:\n"
    "  nginx:\n"
    "    image: nginx:1.27-alpine\n"
    "    container_name: als-nginx\n"
    "  web-alice:\n"
    "    image: reg/web-terminal:latest\n"
    "    container_name: als-web-alice\n"
    "  web-bob:\n"
    "    image: reg/web-terminal-analysis:latest\n"
    "    container_name: als-web-bob\n"
)

_WEB_CMD = ["podman", "compose", "-f", "docker-compose.web.yml"]
_RUN_ENV = {"COMPOSE_PROJECT_NAME": "als"}


def _write_web_compose(tmp_path):
    (tmp_path / "docker-compose.web.yml").write_text(_WEB_COMPOSE, encoding="utf-8")


def _patch_ids(monkeypatch, image_ids, container_ids):
    monkeypatch.setattr(
        provision, "get_image_id", lambda runtime, image, env=None: image_ids[image]
    )
    monkeypatch.setattr(
        provision,
        "get_container_image_id",
        lambda runtime, container, env=None: container_ids[container],
    )


def test_reconcile_force_recreates_only_drifted_services(monkeypatch, tmp_path):
    """podman + a single service whose running image ID drifted from its tag →
    exactly one `up -d --force-recreate <that service>`, delta-scoped, under the
    same web_cmd/env as the preceding `up`."""
    monkeypatch.chdir(tmp_path)
    _write_web_compose(tmp_path)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config=None: ["podman", "compose"])
    _patch_ids(
        monkeypatch,
        image_ids={
            "nginx:1.27-alpine": "idnginx",
            "reg/web-terminal:latest": "idNEW",
            "reg/web-terminal-analysis:latest": "idbob",
        },
        container_ids={
            "als-nginx": "idnginx",
            "als-web-alice": "idOLD",  # stale: still on the pre-pull image
            "als-web-bob": "idbob",
        },
    )
    recorded = []

    def _fake_run(cmd, **kwargs):
        recorded.append((cmd, kwargs.get("env")))
        return _FakeCompletedProcess()

    monkeypatch.setattr(provision.subprocess, "run", _fake_run)

    provision._reconcile_web_stack_image_drift({}, _WEB_CMD, _RUN_ENV)

    assert len(recorded) == 1
    cmd, env = recorded[0]
    assert cmd == [
        "podman",
        "compose",
        "-f",
        "docker-compose.web.yml",
        "up",
        "-d",
        "--force-recreate",
        "web-alice",
    ]
    assert env == _RUN_ENV


def test_reconcile_noop_when_all_images_match(monkeypatch, tmp_path):
    """podman + every running image ID matches its tag → no compose command."""
    monkeypatch.chdir(tmp_path)
    _write_web_compose(tmp_path)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config=None: ["podman", "compose"])
    ids = {
        "nginx:1.27-alpine": "idnginx",
        "reg/web-terminal:latest": "idalice",
        "reg/web-terminal-analysis:latest": "idbob",
    }
    _patch_ids(
        monkeypatch,
        image_ids=ids,
        container_ids={"als-nginx": "idnginx", "als-web-alice": "idalice", "als-web-bob": "idbob"},
    )

    def _unexpected_run(cmd, **kwargs):
        raise AssertionError(f"unexpected subprocess.run: {cmd}")

    monkeypatch.setattr(provision.subprocess, "run", _unexpected_run)

    provision._reconcile_web_stack_image_drift({}, _WEB_CMD, _RUN_ENV)


def test_reconcile_skipped_entirely_on_docker(monkeypatch, tmp_path):
    """docker runtime → the reconcile is skipped before any inspect or compose
    call (docker compose already recreates after a re-pull)."""
    monkeypatch.chdir(tmp_path)
    _write_web_compose(tmp_path)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config=None: ["docker", "compose"])

    def _boom(*args, **kwargs):
        raise AssertionError("must not inspect or run compose on docker")

    monkeypatch.setattr(provision, "get_image_id", _boom)
    monkeypatch.setattr(provision, "get_container_image_id", _boom)
    monkeypatch.setattr(provision.subprocess, "run", _boom)

    provision._reconcile_web_stack_image_drift({}, _WEB_CMD, _RUN_ENV)


def test_reconcile_skips_service_on_inspect_error_without_raising(monkeypatch, tmp_path):
    """A service whose image or container can't be inspected (None) is skipped,
    never aborting the deploy; with the rest matching, no recreate runs."""
    monkeypatch.chdir(tmp_path)
    _write_web_compose(tmp_path)
    monkeypatch.setattr(provision, "get_runtime_command", lambda config=None: ["podman", "compose"])
    _patch_ids(
        monkeypatch,
        image_ids={
            "nginx:1.27-alpine": "idnginx",
            "reg/web-terminal:latest": None,  # image inspect failed / never pulled
            "reg/web-terminal-analysis:latest": "idbob",
        },
        container_ids={
            "als-nginx": "idnginx",
            "als-web-alice": "idOLD",  # would differ, but image side is None → skipped
            "als-web-bob": None,  # container not created yet → skipped
        },
    )

    def _unexpected_run(cmd, **kwargs):
        raise AssertionError(f"unexpected subprocess.run: {cmd}")

    monkeypatch.setattr(provision.subprocess, "run", _unexpected_run)

    # No raise, no compose invocation.
    provision._reconcile_web_stack_image_drift({}, _WEB_CMD, _RUN_ENV)


def test_reconcile_skipped_when_compose_file_unreadable(monkeypatch, tmp_path):
    """podman but no rendered docker-compose.web.yml at the root → advisory skip,
    no inspect, no compose command, no raise."""
    monkeypatch.chdir(tmp_path)  # no compose file written
    monkeypatch.setattr(provision, "get_runtime_command", lambda config=None: ["podman", "compose"])

    def _boom(*args, **kwargs):
        raise AssertionError("must not inspect or run compose when the file is unreadable")

    monkeypatch.setattr(provision, "get_image_id", _boom)
    monkeypatch.setattr(provision.subprocess, "run", _boom)

    provision._reconcile_web_stack_image_drift({}, _WEB_CMD, _RUN_ENV)
