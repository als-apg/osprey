"""Tests for runtime_helper.runtime_env's COMPOSE_PROJECT_NAME pin.

Docker/podman compose namespaces bare-declared named volumes (e.g. the web
terminal's per-user ``<user>-claude-config`` volume) with
``COMPOSE_PROJECT_NAME``, defaulting to ``basename(cwd)`` when unset. That
default is independent of ``compose_generator.resolve_project_name`` — the
project name baked into container labels — so the two can silently diverge.
``runtime_env`` pins ``COMPOSE_PROJECT_NAME`` to the label project name so
every runtime invocation agrees on one project namespace.
"""

from __future__ import annotations

import os

from osprey.deployment import runtime_helper
from osprey.deployment.runtime_helper import (
    get_container_image_id,
    get_image_id,
    runtime_env,
)


def test_runtime_env_pins_compose_project_name_from_config() -> None:
    """COMPOSE_PROJECT_NAME is set to resolve_project_name(config)."""
    config = {"project_name": "proj-a"}
    env = runtime_env(config, base_env={})
    assert env["COMPOSE_PROJECT_NAME"] == "proj-a"


def test_runtime_env_pins_compose_project_name_from_project_root() -> None:
    """Falls back to basename(project_root) just like resolve_project_name."""
    config = {"project_root": "/home/user/my-facility-project"}
    env = runtime_env(config, base_env={})
    assert env["COMPOSE_PROJECT_NAME"] == "my-facility-project"


def test_runtime_env_pins_default_project_name_for_falsy_config() -> None:
    """None or {} config resolves the same 'unnamed-project' default."""
    assert runtime_env(None, base_env={})["COMPOSE_PROJECT_NAME"] == "unnamed-project"
    assert runtime_env({}, base_env={})["COMPOSE_PROJECT_NAME"] == "unnamed-project"


def test_runtime_env_layers_onto_base_env_without_mutating_it() -> None:
    """The pin is added to a copy; the caller's base_env dict is untouched."""
    base_env = {"PATH": "/usr/bin", "SOME_VAR": "1"}
    original = dict(base_env)

    env = runtime_env({"project_name": "proj-a"}, base_env=base_env)

    assert base_env == original  # source dict left untouched
    assert env["PATH"] == "/usr/bin"
    assert env["SOME_VAR"] == "1"
    assert env["COMPOSE_PROJECT_NAME"] == "proj-a"
    assert env is not base_env


def test_runtime_env_defaults_to_os_environ_copy_without_mutating_it() -> None:
    """With no base_env given, os.environ is copied, not mutated in place."""
    sentinel_key = "OSPREY_RUNTIME_ENV_TEST_SENTINEL"
    assert sentinel_key not in os.environ

    env = runtime_env({"project_name": "proj-a"})

    assert env["COMPOSE_PROJECT_NAME"] == "proj-a"
    assert "COMPOSE_PROJECT_NAME" not in os.environ


class _FakeCompletedProcess:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_get_image_id_returns_normalized_id(monkeypatch) -> None:
    """A successful `image inspect` returns the ID with any sha256: prefix stripped."""
    recorded: dict = {}

    def _fake_run(cmd, **kwargs):
        recorded["cmd"] = cmd
        return _FakeCompletedProcess(returncode=0, stdout="sha256:abc123\n")

    monkeypatch.setattr(runtime_helper.subprocess, "run", _fake_run)

    assert get_image_id("podman", "reg/web-terminal:latest") == "abc123"
    assert recorded["cmd"] == [
        "podman",
        "image",
        "inspect",
        "--format",
        "{{.Id}}",
        "reg/web-terminal:latest",
    ]


def test_get_image_id_returns_none_for_missing_image(monkeypatch) -> None:
    """A non-zero exit (no such image) yields None rather than raising."""
    monkeypatch.setattr(
        runtime_helper.subprocess,
        "run",
        lambda cmd, **kwargs: _FakeCompletedProcess(returncode=125, stderr="no such image"),
    )
    assert get_image_id("podman", "reg/nope:latest") is None


def test_get_container_image_id_returns_normalized_id(monkeypatch) -> None:
    """A successful `container inspect` returns the created-from image ID, normalized."""
    recorded: dict = {}

    def _fake_run(cmd, **kwargs):
        recorded["cmd"] = cmd
        return _FakeCompletedProcess(returncode=0, stdout="abc123\n")

    monkeypatch.setattr(runtime_helper.subprocess, "run", _fake_run)

    assert get_container_image_id("podman", "als-web-alice") == "abc123"
    assert recorded["cmd"] == [
        "podman",
        "container",
        "inspect",
        "--format",
        "{{.Image}}",
        "als-web-alice",
    ]


def test_get_container_image_id_returns_none_for_missing_container(monkeypatch) -> None:
    """A missing container returns None so the caller can treat it as a no-op."""
    monkeypatch.setattr(
        runtime_helper.subprocess,
        "run",
        lambda cmd, **kwargs: _FakeCompletedProcess(returncode=1, stderr="no such container"),
    )
    assert get_container_image_id("podman", "als-web-ghost") is None


def test_inspect_id_returns_none_for_empty_output(monkeypatch) -> None:
    """A zero exit with empty stdout is still treated as absent (None)."""
    monkeypatch.setattr(
        runtime_helper.subprocess,
        "run",
        lambda cmd, **kwargs: _FakeCompletedProcess(returncode=0, stdout="\n"),
    )
    assert get_image_id("podman", "reg/web-terminal:latest") is None
