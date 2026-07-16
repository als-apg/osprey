"""Unit tests for the python-executor sandbox environment scrub.

Covers ``scrub_sensitive_env`` (the pure filtering function) and the
``_execute_via_local`` subprocess-spawn seam that must use it. The scrub
prevents agent-generated code running in the local-execution sandbox from
reading write-arming secrets (e.g. ``BLUESKY_PROMOTE_TOKEN``) and calling a
write-gated endpoint directly, bypassing the ``writes_enabled`` re-check
inside ``launch_run``.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from osprey.mcp_server.python_executor.executor import execute_code
from osprey.mcp_server.sandbox_env import (
    _SENSITIVE_ENV_EXACT,
    _SENSITIVE_ENV_SUFFIXES,
    scrub_sensitive_env,
)


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


def _write_local_config(tmp_path):
    config = {
        "control_system": {"type": "mock", "limits_checking": {"enabled": False}},
        "execution": {"execution_method": "local"},
        "python_executor": {"execution_timeout_seconds": 300},
    }
    (tmp_path / "config.yml").write_text(yaml.dump(config))


# ---------------------------------------------------------------------------
# scrub_sensitive_env — pure function
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_scrub_removes_bluesky_promote_token():
    """BLUESKY_PROMOTE_TOKEN is dropped via the *_PROMOTE_TOKEN suffix rule."""
    env = {"BLUESKY_PROMOTE_TOKEN": "secret", "PATH": "/usr/bin"}
    scrubbed = scrub_sensitive_env(env)
    assert "BLUESKY_PROMOTE_TOKEN" not in scrubbed
    assert scrubbed["PATH"] == "/usr/bin"


@pytest.mark.unit
def test_scrub_removes_event_dispatcher_token():
    """EVENT_DISPATCHER_TOKEN is dropped via the exact-name rule."""
    env = {"EVENT_DISPATCHER_TOKEN": "secret", "PATH": "/usr/bin"}
    scrubbed = scrub_sensitive_env(env)
    assert "EVENT_DISPATCHER_TOKEN" not in scrubbed
    assert scrubbed["PATH"] == "/usr/bin"


@pytest.mark.unit
def test_scrub_generalizes_to_future_promote_tokens():
    """Any future *_PROMOTE_TOKEN name is scrubbed without a code change."""
    env = {"SOME_OTHER_BRIDGE_PROMOTE_TOKEN": "secret", "PATH": "/usr/bin"}
    scrubbed = scrub_sensitive_env(env)
    assert "SOME_OTHER_BRIDGE_PROMOTE_TOKEN" not in scrubbed


@pytest.mark.unit
def test_scrub_preserves_unrelated_env():
    """Ordinary env vars (including ones merely containing "TOKEN") pass through."""
    env = {
        "CONFIG_FILE": "/app/config.yml",
        "EPICS_CA_ADDR_LIST": "10.0.0.1",
        "HOME": "/home/user",
        # Contains "TOKEN" but is not a write-arming secret and does not match
        # either scrub rule — must survive.
        "TOKENIZER_CACHE_DIR": "/tmp/cache",
    }
    scrubbed = scrub_sensitive_env(env)
    assert scrubbed == env


@pytest.mark.unit
def test_scrub_does_not_mutate_input():
    """scrub_sensitive_env returns a copy; it must not mutate the caller's dict."""
    env = {"BLUESKY_PROMOTE_TOKEN": "secret", "PATH": "/usr/bin"}
    original = dict(env)
    scrub_sensitive_env(env)
    assert env == original


@pytest.mark.unit
def test_scrub_empty_env():
    assert scrub_sensitive_env({}) == {}


@pytest.mark.unit
def test_sensitive_env_constants_are_tuples():
    """Constants are tuples (immutable, module-level security constants — not config)."""
    assert isinstance(_SENSITIVE_ENV_EXACT, tuple)
    assert isinstance(_SENSITIVE_ENV_SUFFIXES, tuple)
    assert "EVENT_DISPATCHER_TOKEN" in _SENSITIVE_ENV_EXACT
    assert "_PROMOTE_TOKEN" in _SENSITIVE_ENV_SUFFIXES


# ---------------------------------------------------------------------------
# _execute_via_local — subprocess-spawn seam actually applies the scrub
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_local_subprocess_env_excludes_promote_token(tmp_path, monkeypatch):
    """The local-exec subprocess is spawned with an env that excludes the token."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", "super-secret-value")
    _write_local_config(tmp_path)

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
    passed_env = mock_spawn.await_args.kwargs["env"]
    assert "BLUESKY_PROMOTE_TOKEN" not in passed_env


@pytest.mark.unit
async def test_local_subprocess_env_excludes_event_dispatcher_token(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "super-secret-value")
    _write_local_config(tmp_path)

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
    passed_env = mock_spawn.await_args.kwargs["env"]
    assert "EVENT_DISPATCHER_TOKEN" not in passed_env


@pytest.mark.unit
async def test_local_subprocess_env_keeps_config_file(tmp_path, monkeypatch):
    """Non-sensitive vars the sandbox legitimately needs (e.g. CONFIG_FILE) survive."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BLUESKY_PROMOTE_TOKEN", "super-secret-value")
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "config.yml"))
    _write_local_config(tmp_path)

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
    passed_env = mock_spawn.await_args.kwargs["env"]
    assert passed_env.get("CONFIG_FILE") == str(tmp_path / "config.yml")
