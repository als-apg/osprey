"""Unit tests for the python-executor sandbox environment scrub.

Covers ``scrub_sensitive_env`` (the pure filtering function) and the
``_execute_via_local`` subprocess-spawn seam that must use it. The scrub
prevents agent-generated code running in the local-execution sandbox from
reading write-arming secrets (e.g. ``BLUESKY_LAUNCH_TOKEN``) and calling a
write-gated endpoint directly, bypassing the ``writes_enabled`` re-check
inside the Bluesky queue's arming tools.

It also covers the navigation-only perimeter stamp the same seam reads: the
deny-list of the deployment's own web ports is consumed by THIS process and
handed to the sandbox as a constructor argument, never left in the child's
environment for executed code to read, widen or empty.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from osprey.mcp_server.python_executor.executor import _perimeter_denied_ports, execute_code
from osprey.mcp_server.sandbox_env import (
    SENSITIVE_ENV_EXACT,
    SENSITIVE_ENV_SUFFIXES,
    scrub_sensitive_env,
)
from osprey.services.python_executor.execution.wrapper import ExecutionWrapper


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


# ---------------------------------------------------------------------------
# _execute_via_local — subprocess-spawn seam actually applies the scrub
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_local_subprocess_env_excludes_launch_token(tmp_path, monkeypatch):
    """The local-exec subprocess is spawned with an env that excludes the token."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", "super-secret-value")
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
    passed_env = mock_spawn.await_args.kwargs["env"]
    assert "BLUESKY_LAUNCH_TOKEN" not in passed_env


@pytest.mark.unit
async def test_local_subprocess_env_excludes_event_dispatcher_token(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "super-secret-value")
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
    passed_env = mock_spawn.await_args.kwargs["env"]
    assert "EVENT_DISPATCHER_TOKEN" not in passed_env


@pytest.mark.unit
async def test_local_subprocess_env_keeps_config_file(tmp_path, monkeypatch):
    """Non-sensitive vars the sandbox legitimately needs (e.g. CONFIG_FILE) survive."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", "super-secret-value")
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "config.yml"))
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
    passed_env = mock_spawn.await_args.kwargs["env"]
    assert passed_env.get("CONFIG_FILE") == str(tmp_path / "config.yml")


# ---------------------------------------------------------------------------
# _execute_via_local — the navigation-only perimeter stamp
# ---------------------------------------------------------------------------
#
# Under `auth.method: none` the deployment renders OSPREY_WEB_PERIMETER=open and
# OSPREY_WEB_PERIMETER_DENY_PORTS onto every per-user container (the render half
# is pinned by tests/deployment/web_terminals/test_perimeter_stamp.py). This
# process reads both, hands the parsed ports to the ExecutionWrapper, and drops
# the two names from the sandbox child's environment: the sandbox is TOLD what
# it may not reach, and cannot re-derive, widen or empty the list.


async def _run_with_stubbed_subprocess() -> dict[str, str]:
    """Run one local execution against a stubbed spawn; return the child's env.

    Same stub shape as the scrub tests above — the sandbox process is never
    really started, so the assertions are about what the seam WOULD hand it.
    """
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


@pytest.mark.unit
def test_perimeter_ports_parse_when_the_marker_is_open():
    """Marker plus a well-formed list yields the ports, ascending and de-duplicated."""
    env = {"OSPREY_WEB_PERIMETER": "open", "OSPREY_WEB_PERIMETER_DENY_PORTS": "10101,10000,10100"}
    assert _perimeter_denied_ports(env) == (10000, 10100, 10101)


@pytest.mark.unit
def test_perimeter_ports_empty_without_the_marker():
    """A deny-list with no marker is inert.

    The marker names the posture that justifies the guard. A list rendered
    without it would mean a credentialed method, where denying these ports
    guards nothing and only breaks traffic entitled to them.
    """
    env = {"OSPREY_WEB_PERIMETER_DENY_PORTS": "10000,10100"}
    assert _perimeter_denied_ports(env) == ()


@pytest.mark.unit
def test_perimeter_ports_empty_when_the_marker_is_not_open():
    """Only the exact value "open" arms it — anything else stays inert."""
    env = {"OSPREY_WEB_PERIMETER": "token", "OSPREY_WEB_PERIMETER_DENY_PORTS": "10000"}
    assert _perimeter_denied_ports(env) == ()


@pytest.mark.unit
def test_perimeter_ports_empty_when_nothing_is_stamped():
    """A host with no deployment stamp at all gets the inert default."""
    assert _perimeter_denied_ports({}) == ()


@pytest.mark.unit
def test_perimeter_ports_ignore_junk_entries():
    """Unparseable and out-of-range entries are skipped, not fatal.

    This is deployment-rendered input read at execution time; one bad token
    must narrow the list, never raise inside the run the guard protects — and
    never discard the entries that did parse.
    """
    env = {
        "OSPREY_WEB_PERIMETER": "open",
        "OSPREY_WEB_PERIMETER_DENY_PORTS": "10000, ,not-a-port,,0,70000,-1,10100",
    }
    assert _perimeter_denied_ports(env) == (10000, 10100)


@pytest.mark.unit
def test_perimeter_ports_empty_when_every_entry_is_junk():
    """A list that parses to nothing is the inert value, not a partial guard."""
    env = {"OSPREY_WEB_PERIMETER": "open", "OSPREY_WEB_PERIMETER_DENY_PORTS": "abc,,-3"}
    assert _perimeter_denied_ports(env) == ()


@pytest.mark.unit
async def test_local_wrapper_receives_the_denied_ports(tmp_path, monkeypatch):
    """The stamped ports reach the ExecutionWrapper as a constructor argument.

    The wrapper is what renders the sandbox script, so this is the seam where
    "passed down explicitly" is either true or not.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OSPREY_WEB_PERIMETER", "open")
    monkeypatch.setenv("OSPREY_WEB_PERIMETER_DENY_PORTS", "10100,10000")
    _write_subprocess_config(tmp_path)

    with patch(
        "osprey.services.python_executor.execution.wrapper.ExecutionWrapper",
        wraps=ExecutionWrapper,
    ) as mock_wrapper:
        await _run_with_stubbed_subprocess()

    assert mock_wrapper.call_args.kwargs["perimeter_denied_ports"] == (10000, 10100)


@pytest.mark.unit
async def test_local_wrapper_receives_no_ports_without_a_stamp(tmp_path, monkeypatch):
    """No stamp, no guard: the wrapper is built with the inert empty tuple."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OSPREY_WEB_PERIMETER", raising=False)
    monkeypatch.delenv("OSPREY_WEB_PERIMETER_DENY_PORTS", raising=False)
    _write_subprocess_config(tmp_path)

    with patch(
        "osprey.services.python_executor.execution.wrapper.ExecutionWrapper",
        wraps=ExecutionWrapper,
    ) as mock_wrapper:
        await _run_with_stubbed_subprocess()

    assert mock_wrapper.call_args.kwargs["perimeter_denied_ports"] == ()


@pytest.mark.unit
async def test_local_subprocess_env_excludes_the_perimeter_stamp(tmp_path, monkeypatch):
    """The child never sees either name.

    This is the "never re-derived inside the sandbox" half of the contract: the
    sandbox receives the deny-list as a literal from its parent, so leaving the
    source variables in its environment would hand executed code both the list
    it must not widen and the port map of every neighbouring terminal.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OSPREY_WEB_PERIMETER", "open")
    monkeypatch.setenv("OSPREY_WEB_PERIMETER_DENY_PORTS", "10000,10100")
    _write_subprocess_config(tmp_path)

    passed_env = await _run_with_stubbed_subprocess()

    assert "OSPREY_WEB_PERIMETER" not in passed_env
    assert "OSPREY_WEB_PERIMETER_DENY_PORTS" not in passed_env
