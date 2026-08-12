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


# ---------------------------------------------------------------------------
# COMPOSE_IGNORE_ORPHANS -- opt-in, and the default matters as much as the
# opt-in.
#
# A web-terminal deploy issues TWO compose invocations against ONE
# COMPOSE_PROJECT_NAME: the backend services (-f build/services/*.yml) and the
# web stack (-f docker-compose.web.yml). Each therefore sees the other's
# containers as orphans of the shared project and prints a warning naming every
# one of them, twice per deploy. The warning is not actionable there: its
# suggested remedy, `--remove-orphans`, would delete the OTHER stack, which is
# exactly why provision.py forbids that flag on both paths --
# ignore_orphans=True silences it at the source.
#
# The single-stack paths are the mirror image: plain `osprey up` reconciles
# with `up --remove-orphans`, `clean` with `down --remove-orphans`, and docker
# compose HARD-ERRORS on the combination ("cannot combine
# COMPOSE_IGNORE_ORPHANS and --remove-orphans"). A default-on env var breaks
# every one of those deploys outright, so the default-off pin below is a
# regression guard, not a formality.
# ---------------------------------------------------------------------------


def test_runtime_env_does_not_set_ignore_orphans_by_default() -> None:
    """Single-stack deploys pass --remove-orphans; the env var would hard-error."""
    env = runtime_env({"project_name": "proj-a"}, base_env={})
    assert "COMPOSE_IGNORE_ORPHANS" not in env


def test_runtime_env_suppresses_the_orphan_warning_on_request() -> None:
    """The two-invocations-one-project design makes orphans expected, not news."""
    env = runtime_env({"project_name": "proj-a"}, base_env={}, ignore_orphans=True)
    assert env["COMPOSE_IGNORE_ORPHANS"] == "1"


def test_runtime_env_never_mutates_the_base_env() -> None:
    """Both pins land on a copy -- the caller's env is untouched."""
    base: dict[str, str] = {}
    runtime_env({"project_name": "proj-a"}, base_env=base, ignore_orphans=True)
    assert base == {}


# ---------------------------------------------------------------------------
# with_plain_progress -- `--progress plain` for docker only.
#
# Compose's default `auto` progress picks its TTY renderer on an interactive
# terminal, which redraws frames with cursor-up and no erase-to-end-of-line.
# Long project/service names wrap, the renderer's line arithmetic goes wrong,
# and frames paint over each other into unreadable garbage. `plain` is
# append-only and cannot garble. Docker only: `--progress` is a docker compose
# v2 flag, and a podman deploy may resolve `podman compose` to a provider that
# rejects it -- a hard failure in exchange for cosmetics is a bad trade, and
# non-interactive runs (CI, systemd) already default to plain anyway.
#
# It takes a RESOLVED argv rather than a config on purpose: every deploy call
# site reaches its runtime through the `get_runtime_command` name bound in its
# own module, and that binding is the seam the deploy tests monkeypatch. A
# helper that resolved the runtime itself would bypass those patches and run
# live runtime detection inside unit tests.
# ---------------------------------------------------------------------------


def test_with_plain_progress_forces_plain_progress_on_docker() -> None:
    assert runtime_helper.with_plain_progress(["docker", "compose"]) == [
        "docker",
        "compose",
        "--progress",
        "plain",
    ]


def test_with_plain_progress_leaves_podman_untouched() -> None:
    assert runtime_helper.with_plain_progress(["podman", "compose"]) == ["podman", "compose"]


def test_with_plain_progress_tolerates_an_empty_argv() -> None:
    """Never index [0] blind -- a runtime stub may hand back an empty list."""
    assert runtime_helper.with_plain_progress([]) == []


def test_with_plain_progress_does_not_resolve_a_runtime_itself(monkeypatch) -> None:
    """The seam guard: it must never call get_runtime_command on its own."""

    def _boom(*args, **kwargs):
        raise AssertionError("with_plain_progress must not resolve the runtime")

    monkeypatch.setattr(runtime_helper, "get_runtime_command", _boom)
    assert runtime_helper.with_plain_progress(["docker", "compose"])[:2] == ["docker", "compose"]


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
