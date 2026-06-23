"""Unit tests for container lifecycle argv construction.

These stub out the compose-file preparation, runtime checks, and the actual
subprocess invocation, then assert on the argv that ``deploy_up`` would run —
the cheapest way to lock in flag behavior (notably ``--build`` under ``--dev``)
without a container runtime.
"""

from __future__ import annotations

import pytest

from osprey.deployment import container_lifecycle


@pytest.fixture
def captured_argv(monkeypatch, tmp_path):
    """Patch deploy_up's collaborators and capture the compose argv.

    Runs in ``detached=True`` mode so the call lands on ``subprocess.run`` (which
    we capture) rather than ``os.execvpe`` (which would replace the process).
    """
    captured: dict = {}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: ({"deployed_services": ["event_dispatcher"]}, ["docker-compose.yml"]),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )

    def _fake_run(cmd, env=None, check=False):
        captured["cmd"] = cmd

    monkeypatch.setattr(container_lifecycle.subprocess, "run", _fake_run)
    return captured


def test_deploy_up_dev_mode_adds_build(captured_argv, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True, dev_mode=True)
    assert "--build" in captured_argv["cmd"]
    assert "up" in captured_argv["cmd"]


def test_deploy_up_non_dev_omits_build(captured_argv, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True, dev_mode=False)
    assert "--build" not in captured_argv["cmd"]
    assert "up" in captured_argv["cmd"]


# ---------------------------------------------------------------------------
# Dispatch token auto-generation (fail-closed auth)
# ---------------------------------------------------------------------------


@pytest.fixture
def _clean_token_env(monkeypatch):
    """Ensure the dispatch token vars are unset in the process env."""
    monkeypatch.delenv("EVENT_DISPATCHER_TOKEN", raising=False)
    monkeypatch.delenv("DISPATCH_WORKER_TOKEN", raising=False)


def _parse_env(tmp_path):
    from osprey.utils.dotenv import parse_dotenv_file

    p = tmp_path / ".env"
    return parse_dotenv_file(p) if p.is_file() else {}


def test_deploy_up_generates_tokens_when_unset(captured_argv, _clean_token_env, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True, dev_mode=False)

    env = _parse_env(tmp_path)
    assert env.get("EVENT_DISPATCHER_TOKEN") and env["EVENT_DISPATCHER_TOKEN"] != "dev-token"
    assert env.get("DISPATCH_WORKER_TOKEN") and env["DISPATCH_WORKER_TOKEN"] != "dev-token"
    # token_urlsafe(32) → ~43 url-safe chars
    assert len(env["EVENT_DISPATCHER_TOKEN"]) >= 40


def test_token_generation_is_idempotent(captured_argv, _clean_token_env, tmp_path):
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    first = _parse_env(tmp_path)
    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
    second = _parse_env(tmp_path)

    assert first["EVENT_DISPATCHER_TOKEN"] == second["EVENT_DISPATCHER_TOKEN"]
    assert first["DISPATCH_WORKER_TOKEN"] == second["DISPATCH_WORKER_TOKEN"]
    # No duplicate keys appended on the second run.
    text = (tmp_path / ".env").read_text()
    assert text.count("EVENT_DISPATCHER_TOKEN=") == 1
    assert text.count("DISPATCH_WORKER_TOKEN=") == 1


def test_existing_env_token_is_preserved(captured_argv, _clean_token_env, tmp_path):
    (tmp_path / ".env").write_text("EVENT_DISPATCHER_TOKEN=my-real-token\n", encoding="utf-8")

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    assert env["EVENT_DISPATCHER_TOKEN"] == "my-real-token"  # untouched
    assert env.get("DISPATCH_WORKER_TOKEN")  # the missing one was generated


def test_process_env_token_not_written_to_dotenv(captured_argv, monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "from-shell")
    monkeypatch.delenv("DISPATCH_WORKER_TOKEN", raising=False)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    env = _parse_env(tmp_path)
    # A token resolvable from the process env is not duplicated into .env.
    assert "EVENT_DISPATCHER_TOKEN" not in env
    assert env.get("DISPATCH_WORKER_TOKEN")


def test_non_dispatch_deploy_generates_no_tokens(monkeypatch, _clean_token_env, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        container_lifecycle,
        "prepare_compose_files",
        lambda *a, **k: ({"deployed_services": ["mock"]}, ["docker-compose.yml"]),
    )
    monkeypatch.setattr(container_lifecycle, "verify_runtime_is_running", lambda config: (True, ""))
    monkeypatch.setattr(
        container_lifecycle, "get_runtime_command", lambda config: ["docker", "compose"]
    )
    monkeypatch.setattr(container_lifecycle.subprocess, "run", lambda *a, **k: None)

    container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)

    assert not (tmp_path / ".env").exists()


def test_expose_refuses_empty_token(captured_argv, monkeypatch, tmp_path):
    # A token explicitly set empty must not be auto-overwritten, and --expose must
    # refuse rather than bind a fail-open server to 0.0.0.0.
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "")
    monkeypatch.delenv("DISPATCH_WORKER_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="refusing to --expose"):
        container_lifecycle.deploy_up(
            str(tmp_path / "config.yml"), detached=True, expose_network=True
        )
