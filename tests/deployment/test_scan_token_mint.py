"""Unit tests for the bluesky-bridge auth token in the generalized token-mint map.

Mirrors ``test_container_lifecycle.py``'s dispatch-token coverage, but for the
``bluesky`` deployed-service entry added to ``_SERVICE_TOKEN_VARS``
(container_lifecycle.py). Without this mint, the fail-closed bridge
(``osprey.services.bluesky_bridge.security.require_armed``) 503s on every
promote attempt after a fresh deploy.
"""

from __future__ import annotations

import pytest

from osprey.deployment import container_lifecycle


@pytest.fixture
def captured_argv(monkeypatch, tmp_path):
    """Patch deploy_up's collaborators for a project with only 'bluesky' deployed."""
    captured: dict = {}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: ({"deployed_services": ["bluesky"]}, ["docker-compose.yml"]),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )

    def _fake_run(cmd, env=None, check=False):
        captured["cmd"] = cmd

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)
    return captured


@pytest.fixture
def _clean_token_env(monkeypatch):
    """Ensure the bluesky promote token (and dispatch tokens) are unset in the process env."""
    monkeypatch.delenv("BLUESKY_PROMOTE_TOKEN", raising=False)
    monkeypatch.delenv("EVENT_DISPATCHER_TOKEN", raising=False)
    monkeypatch.delenv("DISPATCH_WORKER_TOKEN", raising=False)


def _parse_env(tmp_path):
    from osprey.utils.dotenv import parse_dotenv_file

    p = tmp_path / ".env"
    return parse_dotenv_file(p) if p.is_file() else {}


def test_bluesky_deploy_generates_promote_token(captured_argv, _clean_token_env, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True, dev_mode=False)

    env = _parse_env(tmp_path)
    assert env.get("BLUESKY_PROMOTE_TOKEN")
    # token_urlsafe(32) → ~43 url-safe chars
    assert len(env["BLUESKY_PROMOTE_TOKEN"]) >= 40
    # A deploy with only 'bluesky' deployed must not mint unrelated dispatch tokens.
    assert "EVENT_DISPATCHER_TOKEN" not in env
    assert "DISPATCH_WORKER_TOKEN" not in env


def test_bluesky_token_generation_is_idempotent(captured_argv, _clean_token_env, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    first = _parse_env(tmp_path)
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    second = _parse_env(tmp_path)

    assert first["BLUESKY_PROMOTE_TOKEN"] == second["BLUESKY_PROMOTE_TOKEN"]
    # No duplicate keys appended on the second run.
    text = (tmp_path / ".env").read_text()
    assert text.count("BLUESKY_PROMOTE_TOKEN=") == 1


def test_bluesky_existing_env_token_is_preserved(captured_argv, _clean_token_env, tmp_path):
    (tmp_path / ".env").write_text("BLUESKY_PROMOTE_TOKEN=my-real-token\n", encoding="utf-8")

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert env["BLUESKY_PROMOTE_TOKEN"] == "my-real-token"  # untouched


def test_bluesky_process_env_token_not_written_to_dotenv(captured_argv, monkeypatch, tmp_path):
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", "from-shell")

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    # A token resolvable from the process env is not duplicated into .env.
    assert "BLUESKY_PROMOTE_TOKEN" not in env


def test_bluesky_expose_refuses_empty_token(captured_argv, monkeypatch, tmp_path):
    # A token explicitly set empty must not be auto-overwritten, and --expose must
    # refuse rather than bind a fail-open server to 0.0.0.0.
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", "")

    with pytest.raises(RuntimeError, match="refusing to --expose"):
        container_lifecycle.deploy_up(
            str(tmp_path / "config.yml"), detached=True, expose_network=True
        )


def test_bluesky_alongside_dispatch_mints_both_independently(
    monkeypatch, _clean_token_env, tmp_path
):
    """Per-service-instance behavior: deploying bluesky AND dispatch mints all
    three vars (no cross-service leakage or accidental sharing)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: (
            {"deployed_services": ["event_dispatcher", "dispatch_worker", "bluesky"]},
            ["docker-compose.yml"],
        ),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda *a, **k: None)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert env.get("EVENT_DISPATCHER_TOKEN")
    assert env.get("DISPATCH_WORKER_TOKEN")
    assert env.get("BLUESKY_PROMOTE_TOKEN")
    # All three distinct values — no accidental sharing across services.
    assert (
        len(
            {
                env["EVENT_DISPATCHER_TOKEN"],
                env["DISPATCH_WORKER_TOKEN"],
                env["BLUESKY_PROMOTE_TOKEN"],
            }
        )
        == 3
    )


def test_service_token_vars_map_includes_bluesky():
    """Locks in the generalized map shape task 2.10 introduces."""
    assert container_lifecycle._SERVICE_TOKEN_VARS["bluesky"] == ("BLUESKY_PROMOTE_TOKEN",)
    assert "event_dispatcher" in container_lifecycle._SERVICE_TOKEN_VARS
    assert "dispatch_worker" in container_lifecycle._SERVICE_TOKEN_VARS
