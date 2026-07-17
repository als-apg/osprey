"""Unit tests for the workspace sandbox's environment scrub.

Mirrors ``test_executor_env_scrub.py`` for the sibling sandbox in
``osprey.mcp_server.workspace.execution.sandbox_executor`` (used by the
data-visualizer tools: ``create_static_plot``, ``create_interactive_plot``,
``create_dashboard``). This sandbox has its own local subprocess-spawn seam
(``execute_sandbox_code``'s ``asyncio.create_subprocess_exec`` call) that
previously had no ``env=`` kwarg at all and therefore inherited the full
parent environment — the same write-arming-token leak found and fixed in the
python-executor sandbox (task 2.11).
"""

import textwrap
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.mcp_server.sandbox_env import (
    _SENSITIVE_ENV_EXACT,
    _SENSITIVE_ENV_SUFFIXES,
    scrub_sensitive_env,
)
from osprey.mcp_server.workspace.execution.sandbox_executor import execute_sandbox_code

# ---------------------------------------------------------------------------
# scrub_sensitive_env — pure function
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_scrub_removes_bluesky_launch_token():
    """BLUESKY_LAUNCH_TOKEN is dropped via the *_LAUNCH_TOKEN suffix rule."""
    env = {"BLUESKY_LAUNCH_TOKEN": "secret", "PATH": "/usr/bin"}
    scrubbed = scrub_sensitive_env(env)
    assert "BLUESKY_LAUNCH_TOKEN" not in scrubbed
    assert scrubbed["PATH"] == "/usr/bin"


@pytest.mark.unit
def test_scrub_removes_event_dispatcher_token():
    """EVENT_DISPATCHER_TOKEN is dropped via the exact-name rule."""
    env = {"EVENT_DISPATCHER_TOKEN": "secret", "PATH": "/usr/bin"}
    scrubbed = scrub_sensitive_env(env)
    assert "EVENT_DISPATCHER_TOKEN" not in scrubbed
    assert scrubbed["PATH"] == "/usr/bin"


@pytest.mark.unit
def test_scrub_generalizes_to_future_launch_tokens():
    """Any future *_LAUNCH_TOKEN name is scrubbed without a code change."""
    env = {"SOME_OTHER_BRIDGE_LAUNCH_TOKEN": "secret", "PATH": "/usr/bin"}
    scrubbed = scrub_sensitive_env(env)
    assert "SOME_OTHER_BRIDGE_LAUNCH_TOKEN" not in scrubbed


@pytest.mark.unit
def test_scrub_preserves_unrelated_env():
    """Ordinary env vars (including ones merely containing "TOKEN") pass through."""
    env = {
        "HOME": "/home/user",
        "DISPLAY": ":0",
        # Contains "TOKEN" but is not a write-arming secret and does not match
        # either scrub rule — must survive.
        "TOKENIZER_CACHE_DIR": "/tmp/cache",
    }
    scrubbed = scrub_sensitive_env(env)
    assert scrubbed == env


@pytest.mark.unit
def test_scrub_does_not_mutate_input():
    """scrub_sensitive_env returns a copy; it must not mutate the caller's dict."""
    env = {"BLUESKY_LAUNCH_TOKEN": "secret", "PATH": "/usr/bin"}
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
    assert "_LAUNCH_TOKEN" in _SENSITIVE_ENV_SUFFIXES


@pytest.mark.unit
def test_scrub_shared_with_python_executor():
    """Both sandboxes import the SAME scrub from osprey.mcp_server.sandbox_env

    (not independent copies), so the deny-lists cannot drift by construction.
    """
    from osprey.mcp_server.python_executor import executor as python_executor_module

    assert python_executor_module.scrub_sensitive_env is scrub_sensitive_env


# ---------------------------------------------------------------------------
# execute_sandbox_code — subprocess-spawn seam actually applies the scrub
# ---------------------------------------------------------------------------


@pytest.fixture
def execution_folder(tmp_path):
    folder = tmp_path / "test_execution"
    folder.mkdir()
    return folder


@pytest.fixture
def workspace_root(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    (ws / "data").mkdir()
    return ws


@pytest.mark.unit
async def test_sandbox_subprocess_env_excludes_launch_token(
    execution_folder, workspace_root, monkeypatch
):
    """The sandbox subprocess is spawned with an env that excludes the token."""
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", "super-secret-value")

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))
    mock_proc.returncode = 0
    mock_proc.kill = MagicMock()
    mock_proc.wait = MagicMock()  # not awaited on this path; avoids an unawaited-coroutine warning

    with (
        patch(
            "osprey.utils.workspace.resolve_workspace_root",
            return_value=workspace_root,
        ),
        patch(
            "osprey.mcp_server.workspace.execution.sandbox_executor.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=mock_proc,
        ) as mock_spawn,
    ):
        await execute_sandbox_code(code="print(42)", execution_folder=execution_folder)

    assert mock_spawn.await_count == 1
    passed_env = mock_spawn.await_args.kwargs["env"]
    assert "BLUESKY_LAUNCH_TOKEN" not in passed_env


@pytest.mark.unit
async def test_sandbox_subprocess_env_excludes_event_dispatcher_token(
    execution_folder, workspace_root, monkeypatch
):
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "super-secret-value")

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))
    mock_proc.returncode = 0
    mock_proc.kill = MagicMock()
    mock_proc.wait = MagicMock()  # not awaited on this path; avoids an unawaited-coroutine warning

    with (
        patch(
            "osprey.utils.workspace.resolve_workspace_root",
            return_value=workspace_root,
        ),
        patch(
            "osprey.mcp_server.workspace.execution.sandbox_executor.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=mock_proc,
        ) as mock_spawn,
    ):
        await execute_sandbox_code(code="print(42)", execution_folder=execution_folder)

    assert mock_spawn.await_count == 1
    passed_env = mock_spawn.await_args.kwargs["env"]
    assert "EVENT_DISPATCHER_TOKEN" not in passed_env


@pytest.mark.unit
async def test_sandbox_subprocess_env_keeps_unrelated_vars(
    execution_folder, workspace_root, monkeypatch
):
    """Non-sensitive vars the sandbox legitimately needs (e.g. HOME) survive."""
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", "super-secret-value")
    monkeypatch.setenv("HOME", "/home/testuser")

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))
    mock_proc.returncode = 0
    mock_proc.kill = MagicMock()
    mock_proc.wait = MagicMock()  # not awaited on this path; avoids an unawaited-coroutine warning

    with (
        patch(
            "osprey.utils.workspace.resolve_workspace_root",
            return_value=workspace_root,
        ),
        patch(
            "osprey.mcp_server.workspace.execution.sandbox_executor.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=mock_proc,
        ) as mock_spawn,
    ):
        await execute_sandbox_code(code="print(42)", execution_folder=execution_folder)

    assert mock_spawn.await_count == 1
    passed_env = mock_spawn.await_args.kwargs["env"]
    assert passed_env.get("HOME") == "/home/testuser"


# ---------------------------------------------------------------------------
# Real end-to-end execution (pentest-style, mirrors test_executor_token_regression.py)
# ---------------------------------------------------------------------------


async def test_real_execution_cannot_see_launch_token(
    execution_folder, workspace_root, monkeypatch
):
    """Real (unmocked) subprocess: sandboxed code cannot read BLUESKY_LAUNCH_TOKEN."""
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", "super-secret-promote-value")

    code = textwrap.dedent("""\
        import os
        token = os.environ.get('BLUESKY_LAUNCH_TOKEN')
        print('TOKEN_VALUE:', repr(token))
    """)

    with patch(
        "osprey.utils.workspace.resolve_workspace_root",
        return_value=workspace_root,
    ):
        result = await execute_sandbox_code(code=code, execution_folder=execution_folder)

    assert result.success, f"sandbox execution failed: {result.error_message}\n{result.stderr}"
    assert "TOKEN_VALUE: None" in result.stdout
    assert "super-secret-promote-value" not in result.stdout
