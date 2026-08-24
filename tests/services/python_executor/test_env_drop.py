"""Unit tests for the executor-local web-terminal environment drop.

The python-executor sandbox child runs agent-authored code and has exactly one
callback surface — the ``save_artifact`` helper injected by the execution
wrapper — which writes to the filesystem. It never resolves a web-terminal URL
and never calls a web-terminal route, so the terminal's address family
(``OSPREY_WEB_PORT`` and ``OSPREY_TERMINAL_*``) only tells agent code where a
surface it must not reach is listening.

The drop is *executor-local*: the shared deny-list in
``osprey.utils.sensitive_env`` is the credential set the PTY child shares, and
the PTY child **is** the web terminal. These tests pin both halves — the
sandbox child loses the family, the PTY child keeps it.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from osprey.mcp_server.python_executor.executor import execute_code

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_all_config_caches(monkeypatch):
    """Reset ALL config caches before each test (see test_executor_adapter.py)."""
    from osprey.utils.workspace import reset_config_cache

    reset_config_cache()

    import osprey.utils.config as _cfg

    monkeypatch.setattr(_cfg, "_default_config", None)
    monkeypatch.setattr(_cfg, "_default_configurable", None)
    saved_cache = _cfg._config_cache.copy()
    _cfg._config_cache.clear()

    yield

    reset_config_cache()
    _cfg._config_cache.clear()
    _cfg._config_cache.update(saved_cache)


def _write_subprocess_config(tmp_path):
    config = {
        "control_system": {"type": "mock", "limits_checking": {"enabled": False}},
        "execution": {"execution_method": "subprocess"},
        "python_executor": {"execution_timeout_seconds": 300},
    }
    (tmp_path / "config.yml").write_text(yaml.dump(config))


async def _spawn_env(tmp_path) -> dict[str, str]:
    """Run the executor against a mocked spawn and return the child's ``env=``."""
    _write_subprocess_config(tmp_path)

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))
    mock_proc.returncode = 0
    mock_proc.kill = MagicMock()
    mock_proc.wait = MagicMock()  # not awaited on this path; avoids an unawaited-coroutine warning

    with patch(
        "osprey.mcp_server.python_executor.executor.asyncio.create_subprocess_exec",
        new_callable=AsyncMock,
        return_value=mock_proc,
    ) as mock_spawn:
        await execute_code("print(42)", "readonly", "test")

    assert mock_spawn.await_count == 1
    return mock_spawn.await_args.kwargs["env"]


# ---------------------------------------------------------------------------
# Sandbox child — the web-terminal env is gone
# ---------------------------------------------------------------------------


async def test_sandbox_env_excludes_web_port(tmp_path, monkeypatch):
    """``OSPREY_WEB_PORT`` never reaches the sandbox child."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OSPREY_WEB_PORT", "8080")

    passed_env = await _spawn_env(tmp_path)

    assert "OSPREY_WEB_PORT" not in passed_env


@pytest.mark.parametrize(
    "name",
    [
        "OSPREY_TERMINAL_USER",
        "OSPREY_TERMINAL_LANDING_URL",
        "OSPREY_TERMINAL_BIND_HOST",
        "OSPREY_TERMINAL_WEB_PORT",
        "OSPREY_TERMINAL_EXTERNAL_ORIGIN",
    ],
)
async def test_sandbox_env_excludes_terminal_family(tmp_path, monkeypatch, name):
    """Every ``OSPREY_TERMINAL_*`` name is dropped, matched by prefix."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(name, "value")

    passed_env = await _spawn_env(tmp_path)

    assert name not in passed_env


async def test_sandbox_env_excludes_whole_terminal_family_at_once(tmp_path, monkeypatch):
    """A child seeded with the full family loses all of it, not just the first name."""
    monkeypatch.chdir(tmp_path)
    seeded = {
        "OSPREY_WEB_PORT": "8080",
        "OSPREY_TERMINAL_USER": "alice",
        "OSPREY_TERMINAL_LANDING_URL": "https://terminal.example/",
        "OSPREY_TERMINAL_BIND_HOST": "0.0.0.0",
        "OSPREY_TERMINAL_SECRET_ALICE_B": "per-user-secret",
    }
    for key, value in seeded.items():
        monkeypatch.setenv(key, value)

    passed_env = await _spawn_env(tmp_path)

    assert not [key for key in passed_env if key in seeded]
    assert not [key for key in passed_env if key.startswith("OSPREY_TERMINAL_")]


async def test_sandbox_env_keeps_unrelated_osprey_vars(tmp_path, monkeypatch):
    """The drop is scoped to the terminal family, not to ``OSPREY_*`` at large."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OSPREY_WEB_PORT", "8080")
    monkeypatch.setenv("OSPREY_CONFIG", "/somewhere/config.yml")
    monkeypatch.setenv("OSPREY_WEBHOOK_SINK", "unrelated")

    passed_env = await _spawn_env(tmp_path)

    assert passed_env["OSPREY_CONFIG"] == "/somewhere/config.yml"
    assert passed_env["OSPREY_WEBHOOK_SINK"] == "unrelated"
    # The mode declaration the sandbox depends on is set after the drop.
    assert passed_env["OSPREY_EXECUTION_MODE"] == "readonly"


# ---------------------------------------------------------------------------
# PTY child — the web-terminal env is retained
# ---------------------------------------------------------------------------


def test_pty_env_retains_web_port(monkeypatch):
    """The terminal child keeps ``OSPREY_WEB_PORT``; the drop is executor-local.

    ``tests/hooks/conftest.py`` lists ``OSPREY_WEB_PORT`` among the hook-visible
    variables, which is the contract this asserts: hooks running inside a PTY
    agent session resolve the terminal's own address through it.
    """
    from osprey.interfaces.web_terminal.pty_manager import build_pty_env

    monkeypatch.setenv("OSPREY_WEB_PORT", "8080")

    env = build_pty_env()

    assert env["OSPREY_WEB_PORT"] == "8080"


def test_pty_env_retains_terminal_family(monkeypatch):
    """Non-credential ``OSPREY_TERMINAL_*`` names also survive the PTY path."""
    from osprey.interfaces.web_terminal.pty_manager import build_pty_env

    monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")
    monkeypatch.setenv("OSPREY_TERMINAL_LANDING_URL", "https://terminal.example/")

    env = build_pty_env()

    assert env["OSPREY_TERMINAL_USER"] == "alice"
    assert env["OSPREY_TERMINAL_LANDING_URL"] == "https://terminal.example/"
