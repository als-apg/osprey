"""Tests for the canonical agent-invisible credential list.

Covers three things: that the declared name set is the one the rest of the
safety work depends on, that :func:`strip_sensitive` removes exactly those
names and nothing else, and that the module stays a leaf — importable without
dragging in the ``osprey`` package tree, which is what lets ``agent_runner``
and ``mcp_server`` both depend on it without an import cycle.
"""

from __future__ import annotations

import ast
import os
import selectors
import subprocess
import sys
from pathlib import Path

import pytest

from osprey.utils.sensitive_env import (
    SENSITIVE_ENV_EXACT,
    SENSITIVE_ENV_SUFFIXES,
    is_sensitive,
    strip_sensitive,
)


class TestDeclaredNames:
    """The constants themselves — shape and membership."""

    def test_constants_are_tuples(self) -> None:
        """Immutable containers, so no consumer can append to the shared list."""
        assert isinstance(SENSITIVE_ENV_EXACT, tuple)
        assert isinstance(SENSITIVE_ENV_SUFFIXES, tuple)

    @pytest.mark.parametrize(
        "name",
        ["OSPREY_TERMINAL_SECRET", "OSPREY_PANEL_TOKEN", "EVENT_DISPATCHER_TOKEN"],
    )
    def test_required_exact_names_present(self, name: str) -> None:
        """Each credential the D2 scrub must cover is declared exactly."""
        assert name in SENSITIVE_ENV_EXACT

    def test_launch_token_suffix_present(self) -> None:
        """Bridge launch tokens are covered by convention, not per-name."""
        assert "_LAUNCH_TOKEN" in SENSITIVE_ENV_SUFFIXES

    def test_no_empty_entries(self) -> None:
        """An empty suffix would match every variable and strip the whole env."""
        assert all(SENSITIVE_ENV_EXACT)
        assert all(SENSITIVE_ENV_SUFFIXES)

    def test_no_duplicate_exact_names(self) -> None:
        """Duplicates would signal two people adding the same name blind."""
        assert len(set(SENSITIVE_ENV_EXACT)) == len(SENSITIVE_ENV_EXACT)


class TestIsSensitive:
    """The per-name predicate."""

    @pytest.mark.parametrize("name", SENSITIVE_ENV_EXACT)
    def test_exact_names_match(self, name: str) -> None:
        assert is_sensitive(name) is True

    @pytest.mark.parametrize(
        "name",
        ["BLUESKY_LAUNCH_TOKEN", "SOME_OTHER_BRIDGE_LAUNCH_TOKEN", "_LAUNCH_TOKEN"],
    )
    def test_suffix_names_match(self, name: str) -> None:
        """Any variable ending in a declared suffix matches, known or not."""
        assert is_sensitive(name) is True

    @pytest.mark.parametrize(
        "name",
        ["PATH", "HOME", "OSPREY_WEB_PORT", "OSPREY_CONFIG", "ANTHROPIC_API_KEY"],
    )
    def test_unrelated_names_do_not_match(self, name: str) -> None:
        assert is_sensitive(name) is False

    def test_matching_is_case_sensitive(self) -> None:
        """Lower-case spellings are different variables and are left alone."""
        assert is_sensitive("osprey_panel_token") is False
        assert is_sensitive("bluesky_launch_token") is False

    def test_prefix_match_is_not_enough(self) -> None:
        """A name merely starting with a sensitive name is not itself sensitive."""
        assert is_sensitive("OSPREY_PANEL_TOKEN_FILE") is False
        assert is_sensitive("EVENT_DISPATCHER_TOKEN_PATH") is False

    def test_suffix_must_be_at_the_end(self) -> None:
        assert is_sensitive("MY_LAUNCH_TOKEN_BACKUP") is False

    def test_empty_name_is_not_sensitive(self) -> None:
        assert is_sensitive("") is False


class TestStripSensitive:
    """The dict transform used at every spawn site."""

    def test_removes_exact_and_suffix_matches_keeps_the_rest(self) -> None:
        env = {
            "PATH": "/usr/bin",
            "OSPREY_TERMINAL_SECRET": "s1",
            "OSPREY_PANEL_TOKEN": "s2",
            "EVENT_DISPATCHER_TOKEN": "s3",
            "BLUESKY_LAUNCH_TOKEN": "s4",
            "OSPREY_WEB_PORT": "8000",
        }

        result = strip_sensitive(env)

        assert result == {"PATH": "/usr/bin", "OSPREY_WEB_PORT": "8000"}

    def test_does_not_mutate_the_input(self) -> None:
        """Callers may pass ``os.environ`` itself; their own env must survive."""
        env = {"PATH": "/usr/bin", "OSPREY_PANEL_TOKEN": "secret"}
        original = dict(env)

        strip_sensitive(env)

        assert env == original

    def test_returns_a_new_object(self) -> None:
        env = {"PATH": "/usr/bin"}

        result = strip_sensitive(env)

        assert result is not env
        assert result == env

    def test_preserves_values_verbatim(self) -> None:
        """Only keys are filtered — surviving values are untouched."""
        env = {"PATH": "/usr/bin:/bin", "EMPTY": ""}

        assert strip_sensitive(env) == env

    def test_empty_env(self) -> None:
        assert strip_sensitive({}) == {}

    def test_all_sensitive_env(self) -> None:
        env = dict.fromkeys(SENSITIVE_ENV_EXACT, "x")
        env["BLUESKY_LAUNCH_TOKEN"] = "x"

        assert strip_sensitive(env) == {}

    def test_accepts_any_mapping(self) -> None:
        """``os.environ`` is a Mapping, not a dict — the signature allows it."""
        from types import MappingProxyType

        env = MappingProxyType({"PATH": "/usr/bin", "OSPREY_PANEL_TOKEN": "s"})

        result = strip_sensitive(env)

        assert result == {"PATH": "/usr/bin"}
        assert isinstance(result, dict)

    def test_is_idempotent(self) -> None:
        env = {"PATH": "/usr/bin", "OSPREY_TERMINAL_SECRET": "s"}

        once = strip_sensitive(env)

        assert strip_sensitive(once) == once


class TestLeafModule:
    """The property that lets both agent_runner and mcp_server import this."""

    def test_source_imports_nothing_from_osprey(self) -> None:
        """Static check: no ``import osprey...`` of any spelling in the source."""
        import osprey.utils.sensitive_env as module

        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import, which resolves inside osprey.
                assert node.level == 0, "relative import breaks the leaf property"
                if node.module:
                    imported.append(node.module)

        assert not [name for name in imported if name.split(".")[0] == "osprey"], imported

    def test_loading_the_file_pulls_in_no_osprey_modules(self) -> None:
        """Behavioural check: executing the module imports no osprey package."""
        module_path = Path(__file__).resolve().parents[2] / "src/osprey/utils/sensitive_env.py"
        probe = f"""
import importlib.util, sys, json
spec = importlib.util.spec_from_file_location("_probe_sensitive_env", {str(module_path)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.SENSITIVE_ENV_EXACT, "module did not define its constants"
print(json.dumps(sorted(k for k in sys.modules if k == "osprey" or k.startswith("osprey."))))
"""
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
        )

        assert completed.stdout.strip() == "[]", completed.stdout


# --------------------------------------------------------------------------- #
# guard_process_memory — the holder's memory is unreadable to same-uid peers
# --------------------------------------------------------------------------- #

_LINUX = sys.platform.startswith("linux")

#: A child that holds a marker in its exec-time environment, optionally guards
#: itself, says so, and then waits to be released. The marker is what the
#: parent looks for in ``/proc/<pid>/environ``.
_CHILD_SOURCE = """
import sys
{guard}
print("ready", flush=True)
sys.stdin.readline()
"""

_GUARD_LINE = "from osprey.utils.sensitive_env import guard_process_memory; guard_process_memory()"

#: A child that guards itself and then spawns a grandchild carrying the same
#: environment. Its first line is the grandchild's pid; it holds both alive
#: until released, then releases the grandchild the same way.
_SPAWNING_SOURCE = """
import subprocess, sys
from osprey.utils.sensitive_env import guard_process_memory
guard_process_memory()
child = subprocess.Popen(
    [sys.executable, "-c", "import sys; print('ready', flush=True); sys.stdin.readline()"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
)
assert child.stdout.readline().strip() == "ready"
print(child.pid, flush=True)
sys.stdin.readline()
child.stdin.write("\\n")
child.stdin.close()
child.wait(timeout=10)
"""

#: How long a child gets to print its first line before the test gives up on it.
_READY_TIMEOUT_S = 30.0


def _start(source: str, *, marker: str) -> tuple[subprocess.Popen[str], str]:
    """Start a same-uid child carrying ``marker`` and return it with its first line.

    A child that exits or hangs before printing is killed and its pipes closed
    before the failure is raised, so a broken child never outlives the test.
    """
    proc = subprocess.Popen(
        [sys.executable, "-c", source],
        env={**os.environ, "SENSITIVE_ENV_TEST_MARKER": marker},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=_READY_TIMEOUT_S), "child printed nothing in time"
        first_line = proc.stdout.readline().strip()
        assert first_line, "child exited before printing"
    except BaseException:
        _stop(proc)
        raise
    return proc, first_line


def _spawn_holder(*, guarded: bool, marker: str) -> subprocess.Popen[str]:
    """Start a same-uid child whose exec-time environment carries ``marker``."""
    source = _CHILD_SOURCE.format(guard=_GUARD_LINE if guarded else "pass")
    proc, first_line = _start(source, marker=marker)
    try:
        assert first_line == "ready", first_line
    except BaseException:
        _stop(proc)
        raise
    return proc


def _stop(proc: subprocess.Popen[str]) -> None:
    """Kill a child outright and close both pipes."""
    proc.kill()
    proc.wait(timeout=10)
    _close_pipes(proc)


def _release(proc: subprocess.Popen[str]) -> None:
    """Let a waiting child finish on its own, then close both pipes."""
    assert proc.stdin is not None
    try:
        proc.stdin.write("\n")
        proc.stdin.close()
        proc.wait(timeout=10)
    except BaseException:
        proc.kill()
        proc.wait(timeout=10)
        raise
    finally:
        _close_pipes(proc)


def _close_pipes(proc: subprocess.Popen[str]) -> None:
    for pipe in (proc.stdin, proc.stdout):
        if pipe is not None and not pipe.closed:
            pipe.close()


def _exec_time_environment(pid: int) -> bytes:
    return Path(f"/proc/{pid}/environ").read_bytes()


@pytest.mark.skipif(not _LINUX, reason="/proc/<pid>/environ and PR_SET_DUMPABLE are Linux")
class TestGuardProcessMemory:
    """A process that guards itself cannot be read by a peer running as the same uid."""

    def test_guarded_exec_environment_is_unreadable_to_a_same_uid_peer(self) -> None:
        """The gap, then its closure, in one process pair.

        First an unguarded child: a same-uid peer (this test) reads its exec-time
        environment block and finds the marker — the reproduction. Then a guarded
        child: the same read is refused.
        """
        marker = "sensitive-env-marker-" + os.urandom(8).hex()

        unguarded = _spawn_holder(guarded=False, marker=marker)
        try:
            try:
                baseline = _exec_time_environment(unguarded.pid)
            except PermissionError:
                pytest.skip("this kernel already refuses same-uid /proc reads; nothing to prove")
            assert marker.encode() in baseline, (
                "precondition failed: a same-uid peer could not read an unguarded "
                "process's exec-time environment, so the guard has nothing to close here"
            )
        finally:
            _release(unguarded)

        guarded = _spawn_holder(guarded=True, marker=marker)
        try:
            with pytest.raises(PermissionError):
                _exec_time_environment(guarded.pid)
        finally:
            _release(guarded)

    def test_a_child_of_a_guarded_process_starts_readable(self) -> None:
        """``execve`` resets the flag: what a guarded holder spawns is open as before.

        The guard closes the holder, not its subtree — the agent CLI, its MCP
        servers and a sandboxed run must stay debuggable. A guarded child
        spawns a grandchild; this same-uid peer reads the grandchild's exec-time
        environment while the guarded child's own stays refused.
        """
        marker = "sensitive-env-marker-" + os.urandom(8).hex()

        holder, grandchild_pid = _start(_SPAWNING_SOURCE, marker=marker)
        try:
            try:
                inherited = _exec_time_environment(int(grandchild_pid))
            except PermissionError:
                pytest.skip("this kernel already refuses same-uid /proc reads; nothing to prove")
            assert marker.encode() in inherited
            with pytest.raises(PermissionError):
                _exec_time_environment(holder.pid)
        finally:
            _release(holder)

    def test_guard_reports_the_flag_it_set(self) -> None:
        """On Linux the guard sets the flag and says so; the kernel agrees."""
        from osprey.utils import sensitive_env

        assert sensitive_env.guard_process_memory() is True
        assert sensitive_env._prctl(sensitive_env.PR_GET_DUMPABLE, 0) == 0

    def test_guard_fails_closed_when_the_kernel_refuses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A guard that could not be established is an error, never a silent no-op."""
        from osprey.utils import sensitive_env

        monkeypatch.setattr(sensitive_env, "_prctl", lambda option, arg: -1)
        with pytest.raises(RuntimeError, match="PR_SET_DUMPABLE"):
            sensitive_env.guard_process_memory()


def test_guard_is_a_no_op_where_there_is_no_proc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Off Linux there is no exec-time environment file to close, so nothing is called."""
    from osprey.utils import sensitive_env

    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        sensitive_env, "_prctl", lambda option, arg: calls.append((option, arg)) or 0
    )

    assert sensitive_env.guard_process_memory() is False
    assert calls == []
