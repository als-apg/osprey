"""Tests for the local-exec arming guard (task 2.11c).

The local python-executor path runs agent-authored code with cwd=project_root
and no filesystem/network sandboxing, so it can read BLUESKY_PROMOTE_TOKEN
straight out of .env/config.yml and POST the bridge's /runs/{id}/promote
directly — bypassing launch_scan's in-tool writes_enabled re-check entirely.
The container execution_method is fs/network isolated and doesn't have this
exposure. Per the user ruling (2026-07-06, "guard + document"),
container_lifecycle._ensure_service_tokens must refuse to mint
BLUESKY_PROMOTE_TOKEN — leaving the bridge unarmed (its own require_armed()
then 503s) — whenever control_system.writes_enabled and
execution.execution_method=local are both true, and must otherwise behave
exactly as task 2.10 shipped it.
"""

from __future__ import annotations

import logging

import pytest

from osprey.deployment import container_lifecycle


@pytest.fixture
def captured_argv(monkeypatch, tmp_path):
    """Patch deploy_up's collaborators for a project with only 'bluesky' deployed."""
    captured: dict = {}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )

    def _fake_run(cmd, env=None, check=False):
        captured["cmd"] = cmd

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)
    return captured


def _patch_prepare_compose_files(monkeypatch, config: dict) -> None:
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: (config, ["docker-compose.yml"]),
    )


@pytest.fixture
def _clean_token_env(monkeypatch):
    monkeypatch.delenv("BLUESKY_PROMOTE_TOKEN", raising=False)
    monkeypatch.delenv("EVENT_DISPATCHER_TOKEN", raising=False)
    monkeypatch.delenv("DISPATCH_WORKER_TOKEN", raising=False)


def _parse_env(tmp_path):
    from osprey.utils.dotenv import parse_dotenv_file

    p = tmp_path / ".env"
    return parse_dotenv_file(p) if p.is_file() else {}


def _config(**overrides) -> dict:
    base: dict = {"deployed_services": ["bluesky"]}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# _local_exec_arming_unsafe: pure function, easiest to pin down directly.
# ---------------------------------------------------------------------------


def test_arming_unsafe_true_when_writes_enabled_and_local():
    config = {
        "control_system": {"writes_enabled": True},
        "execution": {"execution_method": "local"},
    }
    assert container_lifecycle._local_exec_arming_unsafe(config) is True


def test_arming_safe_when_writes_enabled_and_container():
    config = {
        "control_system": {"writes_enabled": True},
        "execution": {"execution_method": "container"},
    }
    assert container_lifecycle._local_exec_arming_unsafe(config) is False


def test_arming_safe_when_writes_disabled_even_if_local():
    config = {
        "control_system": {"writes_enabled": False},
        "execution": {"execution_method": "local"},
    }
    assert container_lifecycle._local_exec_arming_unsafe(config) is False


def test_arming_safe_by_default_when_sections_absent():
    # writes_enabled defaults False, execution_method defaults "container".
    assert container_lifecycle._local_exec_arming_unsafe({}) is False


# ---------------------------------------------------------------------------
# End-to-end through deploy_up: the guard must actually suppress the mint.
# ---------------------------------------------------------------------------


def test_bluesky_token_not_minted_when_writes_enabled_and_local(
    captured_argv, _clean_token_env, monkeypatch, tmp_path, caplog
):
    config = _config(
        control_system={"writes_enabled": True},
        execution={"execution_method": "local"},
    )
    _patch_prepare_compose_files(monkeypatch, config)

    with caplog.at_level(logging.WARNING, logger="deployment.lifecycle"):
        container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert "BLUESKY_PROMOTE_TOKEN" not in env
    assert not (tmp_path / ".env").exists()

    # caplog's handler already calls record.getMessage() (formatting in %args),
    # so r.message here is the final rendered string — no further % needed.
    warnings = " ".join(r.message for r in caplog.records)
    assert "bluesky" in warnings
    assert "writes_enabled" in warnings
    assert "local" in warnings


def test_bluesky_token_minted_when_writes_enabled_and_container(
    captured_argv, _clean_token_env, monkeypatch, tmp_path
):
    config = _config(
        control_system={"writes_enabled": True},
        execution={"execution_method": "container"},
    )
    _patch_prepare_compose_files(monkeypatch, config)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert env.get("BLUESKY_PROMOTE_TOKEN")
    assert len(env["BLUESKY_PROMOTE_TOKEN"]) >= 40


def test_bluesky_token_minted_when_writes_enabled_and_execution_section_absent(
    captured_argv, _clean_token_env, monkeypatch, tmp_path
):
    """Default execution_method (no `execution:` section at all) is "container" — safe."""
    config = _config(control_system={"writes_enabled": True})
    _patch_prepare_compose_files(monkeypatch, config)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert env.get("BLUESKY_PROMOTE_TOKEN")


def test_bluesky_token_behavior_unchanged_when_writes_disabled(
    captured_argv, _clean_token_env, monkeypatch, tmp_path
):
    """Even with local exec, writes_enabled=False means no bypass risk — mint as usual."""
    config = _config(
        control_system={"writes_enabled": False},
        execution={"execution_method": "local"},
    )
    _patch_prepare_compose_files(monkeypatch, config)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert env.get("BLUESKY_PROMOTE_TOKEN")


def test_guard_does_not_affect_dispatch_tokens(
    captured_argv, _clean_token_env, monkeypatch, tmp_path
):
    """The guard is scoped to bluesky; dispatch tokens mint normally regardless."""
    config = _config(
        deployed_services=["bluesky", "event_dispatcher", "dispatch_worker"],
        control_system={"writes_enabled": True},
        execution={"execution_method": "local"},
    )
    _patch_prepare_compose_files(monkeypatch, config)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert "BLUESKY_PROMOTE_TOKEN" not in env
    assert env.get("EVENT_DISPATCHER_TOKEN")
    assert env.get("DISPATCH_WORKER_TOKEN")


def test_existing_manual_token_left_untouched_under_unsafe_config(
    captured_argv, monkeypatch, tmp_path
):
    """A user-set token is a deliberate override; the guard neither reads nor clobbers it."""
    (tmp_path / ".env").write_text("BLUESKY_PROMOTE_TOKEN=manually-set\n", encoding="utf-8")
    config = _config(
        control_system={"writes_enabled": True},
        execution={"execution_method": "local"},
    )
    _patch_prepare_compose_files(monkeypatch, config)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert env["BLUESKY_PROMOTE_TOKEN"] == "manually-set"
