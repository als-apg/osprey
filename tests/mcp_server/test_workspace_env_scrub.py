"""Unit tests for the workspace sandbox's environment scrub.

Mirrors ``test_executor_env_scrub.py`` for the sibling sandbox in
``osprey.mcp_server.workspace.execution.sandbox_executor`` (used by the
data-visualizer tools: ``create_static_plot``, ``create_interactive_plot``,
``create_dashboard``). This sandbox has its own local subprocess-spawn seam
(``execute_sandbox_code``'s ``asyncio.create_subprocess_exec`` call) which,
without an ``env=`` kwarg, would inherit the full parent environment — the
same write-arming-token leak the python-executor sandbox guards against
(task 2.11).
"""

import textwrap
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.mcp_server.sandbox_env import (
    PERIMETER_DENY_PORTS_ENV,
    PERIMETER_MARKER_ENV,
    SENSITIVE_ENV_EXACT,
    SENSITIVE_ENV_SUFFIXES,
    scrub_sandbox_child_env,
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
def test_scrub_removes_terminal_secret():
    """OSPREY_TERMINAL_SECRET is dropped: it authenticates a web-terminal session."""
    env = {"OSPREY_TERMINAL_SECRET": "secret", "PATH": "/usr/bin"}
    scrubbed = scrub_sensitive_env(env)
    assert "OSPREY_TERMINAL_SECRET" not in scrubbed
    assert scrubbed["PATH"] == "/usr/bin"


@pytest.mark.unit
def test_scrub_removes_panel_token():
    """OSPREY_PANEL_TOKEN is dropped: no sandbox has business calling panel routes."""
    env = {"OSPREY_PANEL_TOKEN": "secret", "PATH": "/usr/bin"}
    scrubbed = scrub_sensitive_env(env)
    assert "OSPREY_PANEL_TOKEN" not in scrubbed
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
    assert isinstance(SENSITIVE_ENV_EXACT, tuple)
    assert isinstance(SENSITIVE_ENV_SUFFIXES, tuple)
    assert "EVENT_DISPATCHER_TOKEN" in SENSITIVE_ENV_EXACT
    assert "OSPREY_TERMINAL_SECRET" in SENSITIVE_ENV_EXACT
    assert "OSPREY_PANEL_TOKEN" in SENSITIVE_ENV_EXACT
    assert "_LAUNCH_TOKEN" in SENSITIVE_ENV_SUFFIXES


@pytest.mark.unit
def test_sensitive_env_constants_are_reexports_of_canonical_module():
    """The names are re-exported from osprey.utils.sensitive_env, not re-typed here.

    Identity (not equality) is asserted: a second literal list that happened to
    agree today could silently drift from the canonical one tomorrow.
    """
    from osprey.utils import sensitive_env as canonical

    assert SENSITIVE_ENV_EXACT is canonical.SENSITIVE_ENV_EXACT
    assert SENSITIVE_ENV_SUFFIXES is canonical.SENSITIVE_ENV_SUFFIXES


@pytest.mark.unit
def test_scrub_shared_with_python_executor():
    """Both sandboxes build the child env with the SAME helper.

    ``scrub_sandbox_child_env`` from osprey.mcp_server.sandbox_env, not
    independent copies, so the credential scrub and the sandbox-only narrowing
    cannot drift between the two spawn paths by construction.
    """
    from osprey.mcp_server.python_executor import executor as python_executor_module
    from osprey.mcp_server.workspace.execution import sandbox_executor

    assert python_executor_module.scrub_sandbox_child_env is scrub_sandbox_child_env
    assert sandbox_executor.scrub_sandbox_child_env is scrub_sandbox_child_env


@pytest.mark.unit
def test_drop_lists_have_a_single_definition():
    """Neither spawn path holds a drop list of its own.

    Both import the shared module's names and call the shared helper, so there
    is no second literal tuple to drift: a name added for one spawn path is
    added for both, or neither sees it. A local ``*_TO_DROP`` reappearing in
    either module is exactly that drift starting.
    """
    from osprey.mcp_server.python_executor import executor as python_executor_module
    from osprey.mcp_server.workspace.execution import sandbox_executor

    for module in (python_executor_module, sandbox_executor):
        assert not [name for name in vars(module) if name.endswith("_TO_DROP")]


@pytest.mark.unit
def test_shared_helper_drops_the_web_terminal_address_book():
    """The pure helper drops OSPREY_WEB_PORT and the whole OSPREY_TERMINAL_ family."""
    env = {
        "OSPREY_WEB_PORT": "8080",
        "OSPREY_TERMINAL_LANDING_URL": "http://deploy:10000/",
        "OSPREY_TERMINAL_SECRET_ALICE": "secret",
        "PATH": "/usr/bin",
    }
    scrubbed = scrub_sandbox_child_env(env)
    assert scrubbed == {"PATH": "/usr/bin"}


@pytest.mark.unit
def test_shared_helper_drops_the_perimeter_stamp():
    """The pure helper drops both stamp names: the parent has already read them."""
    env = {
        PERIMETER_MARKER_ENV: "open",
        PERIMETER_DENY_PORTS_ENV: "10000,10100",
        "PATH": "/usr/bin",
    }
    scrubbed = scrub_sandbox_child_env(env)
    assert scrubbed == {"PATH": "/usr/bin"}


@pytest.mark.unit
def test_shared_helper_does_not_mutate_input():
    """os.environ is passed directly at both call sites, so this must be a copy."""
    env = {PERIMETER_MARKER_ENV: "open", "PATH": "/usr/bin"}
    original = dict(env)
    scrub_sandbox_child_env(env)
    assert env == original


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
# execute_sandbox_code — the web-terminal address book and the perimeter stamp
# ---------------------------------------------------------------------------
#
# Mirrors the perimeter-absence coverage in test_executor_env_scrub.py against
# the OTHER spawn seam. This sandbox renders visualizations; it has no more
# business resolving a terminal URL, or reading a deny-list it is not the one
# enforcing, than the general-purpose sandbox does — and it is the path that
# was silently inheriting both.


async def _sandbox_child_env(execution_folder, workspace_root) -> dict[str, str]:
    """Run one visualization execution against a stubbed spawn; return the child env."""
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
    return mock_spawn.await_args.kwargs["env"]


@pytest.mark.unit
async def test_sandbox_subprocess_env_excludes_the_perimeter_stamp(
    execution_folder, workspace_root, monkeypatch
):
    """The visualization child never sees either stamp name.

    Under `auth.method: none` the deployment stamps both onto the container.
    Leaving them in this child's environment would hand agent-authored plotting
    code the port map of every neighbouring terminal — and the very list it is
    not the one enforcing, which is precisely what "told, not derived" rules
    out.
    """
    monkeypatch.setenv("OSPREY_WEB_PERIMETER", "open")
    monkeypatch.setenv("OSPREY_WEB_PERIMETER_DENY_PORTS", "10000,10100")

    passed_env = await _sandbox_child_env(execution_folder, workspace_root)

    assert "OSPREY_WEB_PERIMETER" not in passed_env
    assert "OSPREY_WEB_PERIMETER_DENY_PORTS" not in passed_env


@pytest.mark.unit
async def test_sandbox_subprocess_env_excludes_the_web_terminal_address_book(
    execution_folder, workspace_root, monkeypatch
):
    """OSPREY_WEB_PORT and the OSPREY_TERMINAL_ family are dropped here too.

    These predate the perimeter stamp and were reaching this sandbox all along:
    the python executor dropped them, its sibling did not. One shared helper is
    what makes that a single fact rather than two.
    """
    monkeypatch.setenv("OSPREY_WEB_PORT", "8080")
    monkeypatch.setenv("OSPREY_TERMINAL_LANDING_URL", "http://deploy:10000/")
    monkeypatch.setenv("OSPREY_TERMINAL_SECRET_ALICE", "super-secret-value")

    passed_env = await _sandbox_child_env(execution_folder, workspace_root)

    assert "OSPREY_WEB_PORT" not in passed_env
    assert not [name for name in passed_env if name.startswith("OSPREY_TERMINAL_")]


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
