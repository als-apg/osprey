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

from osprey.deployment.runtime_helper import runtime_env


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
