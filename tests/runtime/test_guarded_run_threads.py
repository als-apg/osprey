"""One run lock per process: a thread the run starts runs under it.

The lock a process holds is the process's, kept in one place behind one lock
(:mod:`osprey.runtime._run_state`). A thread that asks again is busy, a caller
that joins runs under the hold, and a fork is ordered against the lock's entry
and release. The pins below hold the run-state lock to system calls only, so a
fork may wait for it. Threads are ordered by ``threading.Event`` and joined
with a timeout.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import osprey.runtime._run_state as _run_state
import osprey.runtime.guarded_run as guarded_run
from osprey.runtime import ENV_CONTROL_TARGET
from osprey.runtime.guarded_run import LOCK_FILE_NAME, OspreyRunBusy, lock
from tests import _guarded_run as fixtures
from tests._guarded_run import (
    fresh_run_state,  # noqa: F401 - used through pytestmark
    write_config,
)

pytestmark = pytest.mark.usefixtures("fresh_run_state")

#: How long a thread may take to reach a point or finish; a hang guard, never
#: a statement about speed.
TIMEOUT_S = 60.0


@pytest.fixture(autouse=True)
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A host deployment repo on a live baseline that asks nobody, entered and stamped ``live``."""
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    (root / "profile.yml").write_text("name: probe\n", encoding="utf-8")
    write_config(root, {"enabled": False})
    monkeypatch.chdir(root)
    monkeypatch.setenv(ENV_CONTROL_TARGET, "live")
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    return root


class _Thread:
    """A thread running ``target``; what escapes it is kept in ``error``."""

    def __init__(self, target: Callable[[], None]) -> None:
        self.error: BaseException | None = None

        def run() -> None:
            try:
                target()
            except BaseException as exc:  # kept for the test to assert on
                self.error = exc

        self._thread = threading.Thread(target=run)
        self._thread.start()

    def join(self) -> BaseException | None:
        self._thread.join(TIMEOUT_S)
        assert not self._thread.is_alive(), "the thread did not finish"
        return self.error


def _holder(directory: Path) -> dict[str, Any] | None:
    text = (directory / LOCK_FILE_NAME).read_text(encoding="utf-8")
    return json.loads(text) if text else None


def test_a_run_on_the_main_thread_joins_a_hold_a_helper_thread_took() -> None:
    taken = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with lock("live"):
            taken.set()
            assert release.wait(TIMEOUT_S)

    thread = _Thread(holder)
    try:
        assert taken.wait(TIMEOUT_S)
        hold = _run_state.STATE.hold
        with lock(None, join=True) as directory:
            assert _run_state.STATE.hold is hold
            with pytest.raises(OspreyRunBusy):
                with lock(None):
                    pytest.fail("a second run without join is busy")
        assert _run_state.STATE.hold is hold, "the join released nothing"
        assert _holder(directory) is not None
    finally:
        release.set()
    assert thread.join() is None
    assert _run_state.STATE.hold is None


def test_a_thread_asking_for_the_lock_is_busy_like_any_second_run() -> None:
    with lock("live") as directory:
        error = _Thread(lambda: lock("live").__enter__()).join()

    assert isinstance(error, OspreyRunBusy)
    assert error.root == str(directory)
    assert error.pid == os.getpid()


def test_a_lock_on_another_target_is_refused_while_one_is_held(repo: Path) -> None:
    refused = "this process holds the run lock on live; it cannot take one on va"
    with lock("live"):
        with pytest.raises(RuntimeError, match=refused):
            with lock("va"):
                pytest.fail("a second target's lock is never taken")

    assert not (repo / "var" / "guarded_run" / "va").exists(), "nothing was opened for va"
    with lock("va"):
        pass


def _tree(module: Any) -> ast.Module:
    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))


def _context_vars(tree: ast.Module) -> list[str]:
    names = []
    for node in tree.body:
        value = node.value if isinstance(node, ast.Assign | ast.AnnAssign) else None
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "ContextVar"
        ):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names += [t.id for t in targets if isinstance(t, ast.Name)]
    return names


def _bound_by_import(tree: ast.Module) -> list[str]:
    return [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name in {"STATE", "LOCK"}
    ]


def test_the_run_state_is_the_hold_alone_and_no_module_binds_it() -> None:
    assert [f.name for f in _run_state.RunState.__dataclass_fields__.values()] == ["hold"]
    assert _context_vars(_tree(guarded_run)) == []
    assert _bound_by_import(_tree(guarded_run)) == []
    assert _bound_by_import(ast.parse("from osprey.runtime._run_state import LOCK")) == ["LOCK"]


def test_the_pyaml_run_is_one_module_state_and_no_context_variable_decides() -> None:
    def tree(name: str) -> ast.Module:
        spec = importlib.util.find_spec(name)
        assert spec is not None and spec.origin is not None
        return ast.parse(Path(spec.origin).read_text(encoding="utf-8"))

    run_tool, run = tree("pyaml_cs_osprey.run_tool"), tree("pyaml_cs_osprey._run")
    assigned = {
        target.id
        for node in run.body
        if isinstance(node, ast.Assign | ast.AnnAssign)
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }

    assert _context_vars(run_tool) == ["_INTERRUPT"]
    assert _context_vars(run) == []
    assert {"_RUN", "_GUARD"} <= assigned


#: The callees a ``with _run_state.LOCK:`` body may call: system calls and the
#: module's own helpers that make only system calls.
_UNDER_LOCK = frozenset(
    {
        "contextlib.suppress",
        "json.dumps",
        "json.loads",
        "time.strftime",
        "time.gmtime",
        "_open_run_dir",
        "_open_levels",
        "_open_dir",
        "_open_regular",
        "_read_holder",
        "_write_holder",
        "_utc_now",
        "_run_state.Hold",
        "OspreyRunBusy",
        "GuardedRunDirError",
        "RuntimeError",
        "str",
    }
)


def _callee(node: ast.Call) -> str:
    return ast.unparse(node.func)


def _is_state_lock(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "LOCK"
        and isinstance(node.value, ast.Name)
        and node.value.id == "_run_state"
    )


def _calls_under_the_lock(tree: ast.Module) -> list[str]:
    """Every callee of a call inside a ``with _run_state.LOCK:`` body outside :data:`_UNDER_LOCK`."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.With) and any(_is_state_lock(i.context_expr) for i in node.items):
            for statement in node.body:
                for call in ast.walk(statement):
                    if not isinstance(call, ast.Call):
                        continue
                    name = _callee(call)
                    if name.startswith(("os.", "fcntl.")) or name in _UNDER_LOCK:
                        continue
                    found.append(name)
    return found


def test_only_system_calls_run_under_the_run_state_lock() -> None:
    assert _calls_under_the_lock(_tree(guarded_run)) == []


def test_a_deployment_lookup_under_the_run_state_lock_is_seen() -> None:
    synthetic = ast.parse("with _run_state.LOCK:\n    root = _repo_root()\n    os.close(3)\n")

    assert _calls_under_the_lock(synthetic) == ["_repo_root"]


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_a_release_unlocks_before_it_forgets_the_hold_and_closes_after() -> None:
    release = _function(_tree(guarded_run), "_release")
    unlock = next(
        n.lineno
        for n in ast.walk(release)
        if isinstance(n, ast.Call)
        and _callee(n) == "fcntl.flock"
        and any(ast.unparse(a) == "fcntl.LOCK_UN" for a in n.args)
    )
    forget = next(
        n.lineno
        for n in ast.walk(release)
        if isinstance(n, ast.Assign)
        and ast.unparse(n.targets[0]) == "state.hold"
        and ast.unparse(n.value) == "None"
    )
    close = next(
        n.lineno
        for n in ast.walk(release)
        if isinstance(n, ast.Call)
        and _callee(n) == "os.close"
        and ast.unparse(n.args[0]) == "hold.fd"
    )

    assert unlock < forget < close


def test_the_fork_hooks_spell_the_lock_at_use_and_the_child_never_unlocks() -> None:
    tree = _tree(guarded_run)
    (register,) = [
        n for n in ast.walk(tree) if isinstance(n, ast.Call) and _callee(n) == "os.register_at_fork"
    ]
    hooks = {k.arg: ast.unparse(k.value) for k in register.keywords}

    assert set(hooks) == {"before", "after_in_parent", "after_in_child"}
    assert hooks["after_in_child"] == "_drop_held_locks_in_child"
    for name in hooks.values():
        function = _function(tree, name)
        names = [n.id for n in ast.walk(function) if isinstance(n, ast.Name)]
        assert "LOCK" not in names, f"{name} binds the run-state lock"
    for name in (hooks["before"], hooks["after_in_parent"]):
        assert any(_is_state_lock(n) for n in ast.walk(_function(tree, name))), name
    child = _function(tree, hooks["after_in_child"])
    assert not [n for n in ast.walk(child) if isinstance(n, ast.Call) and "flock" in _callee(n)]


def test_a_fresh_run_state_replaces_the_lock_and_the_state() -> None:
    reset = _function(
        ast.parse(Path(fixtures.__file__).read_text(encoding="utf-8")), "reset_run_state"
    )
    replaced = {
        ast.literal_eval(n.args[1])
        for n in ast.walk(reset)
        if isinstance(n, ast.Call)
        and _callee(n) == "monkeypatch.setattr"
        and ast.unparse(n.args[0]) == "_run_state"
    }

    assert replaced == {"LOCK", "STATE"}
