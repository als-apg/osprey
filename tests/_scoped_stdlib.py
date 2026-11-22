"""Fake ``subprocess`` / ``shutil`` for one module without touching the stdlib module.

A patch target that names the module under test is *not* automatically scoped to
it. After a module does ``import subprocess``, ``<module>.subprocess`` **is** the
stdlib module object, so all of these mutate ``sys.modules["subprocess"]``:

    patch("subprocess.run")                                  # obviously global
    patch("osprey.cli.chat_cmd.subprocess.run")              # equally global, looks scoped
    monkeypatch.setattr(chat_cmd.subprocess, "run", fake)    # equally global

While any of them is active the fake is visible to every thread, background
server and module in the worker. What does scope the fake is rebinding the
*importing module's own* ``subprocess`` name to a stand-in module, which is what
every helper here does. The stand-in answers the names it overrides and hands
every other attribute to the module behind it, so ``except
subprocess.TimeoutExpired`` and ``subprocess.PIPE`` in the code under test keep
meaning what they mean. ``tests/infrastructure/test_no_wide_stdlib_patch.py``
refuses the global spellings.

This requires the module under test to import ``subprocess`` / ``shutil`` at
module level and look the function up where it calls it; the same test file
holds ``src/`` to that.

Usage — with ``monkeypatch``, returning the stand-in::

    scoped(monkeypatch, app_cmd, "shutil", which=lambda name: None)
    scoped(monkeypatch, "osprey.deployment.container_lifecycle", "subprocess", run=fake_run)

Two calls on the same module and name layer: the second stand-in sits over the
first, and ``monkeypatch`` takes both off in turn.

With ``unittest.mock``, as a decorator (the injected argument is the stand-in)
or a context manager::

    @scoped_patch("osprey.mcp_server.session", "shutil", which=MagicMock(return_value=None))
    def test_no_git(self, fake_shutil): ...

    @patch_subprocess("osprey.cli.chat_cmd", return_value=Mock(returncode=0))
    def test_launch(self, fake_subprocess, ...):
        ...
        fake_subprocess.run.assert_called_once()

``patch_subprocess`` makes ``run`` a fresh ``MagicMock`` built from its keyword
arguments and leaves the real ``Popen`` in place unless ``popen=`` is passed:
``popen=True`` for a bare ``MagicMock``, any other value as the replacement::

    with patch_subprocess("osprey.cli.web_cmd", popen=True) as fake:
        fake.Popen.return_value.pid = 4321
        ...
        assert fake.Popen.call_args.args[0] == [...]

A test that states "nothing here starts a child process" names no module to
fake. :func:`no_child_process` refuses and records every child the code it
wraps starts, through the interpreter's audit events, and only in that code's
context (the test's thread, the tasks it starts, the request handlers a
``TestClient`` runs for it)::

    with no_child_process() as started:
        process_table.snapshot()
    assert started == []
"""

from __future__ import annotations

import contextlib
import contextvars
import importlib
import sys
from collections.abc import Iterator
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

#: The audit events every way of starting a child process raises.
_CHILD_EVENTS = frozenset(
    {
        "subprocess.Popen",
        "os.fork",
        "os.forkpty",
        "os.posix_spawn",
        "os.spawn",
        "os.exec",
        "os.system",
    }
)

_refusing: contextvars.ContextVar[list[tuple[str, tuple[Any, ...]]] | None] = (
    contextvars.ContextVar("_refusing", default=None)
)

_audit_hook_added = False


class _StandIn(ModuleType):
    """A module that answers its own attributes and hands the rest to the module behind it."""

    def __init__(self, behind: ModuleType, overrides: dict[str, Any]) -> None:
        super().__init__(behind.__name__)
        self.__dict__["_behind"] = behind
        self.__dict__.update(overrides)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["_behind"], name)


def stand_in(behind: ModuleType, **overrides: Any) -> ModuleType:
    """A stand-in for *behind* that answers *overrides* and delegates every other name."""
    return _StandIn(behind, overrides)


def _module(target: ModuleType | str) -> ModuleType:
    return importlib.import_module(target) if isinstance(target, str) else target


def scoped(
    monkeypatch: pytest.MonkeyPatch, target: ModuleType | str, name: str, **overrides: Any
) -> ModuleType:
    """Rebind *target*'s own *name* (``"subprocess"`` / ``"shutil"``) to a stand-in.

    Args:
        monkeypatch: The test's ``monkeypatch``; it restores the name afterwards.
        target: The module under test, or its dotted path. It is the module
            that imports the stdlib module, not the stdlib module itself.
        name: The name the target imported, ``"subprocess"`` or ``"shutil"``.
        **overrides: The attributes the target sees replaced, e.g. ``run=fake``.

    Returns:
        The stand-in, for assertions against the fakes it carries.
    """
    module = _module(target)
    fake = stand_in(getattr(module, name), **overrides)
    monkeypatch.setattr(module, name, fake)
    return fake


def scoped_patch(target_module: str, name: str, **overrides: Any):
    """:func:`scoped` as a ``unittest.mock`` patcher, usable as decorator or context manager.

    The stand-in stands over the stdlib module the name refers to (see
    :func:`scoped`); the patcher supplies it as the injected argument or the
    ``as`` target.
    """

    def _new() -> ModuleType:
        return stand_in(importlib.import_module(name), **overrides)

    return patch(f"{target_module}.{name}", new_callable=_new)


def patch_subprocess(target_module: str, *, popen: Any = None, **run_kwargs: Any):
    """Replace ``subprocess`` as seen by ``target_module`` with a stand-in.

    Args:
        target_module: Dotted path of the module under test, e.g.
            ``"osprey.cli.chat_cmd"``. Note this is the module that imports
            ``subprocess``, not the module that defines it.
        popen: How to scope ``Popen``. ``None`` (the default) leaves the real
            stdlib ``Popen`` in place, so unrelated spawning — notably the
            ``git init`` behind the exemplar repo fixture — still works. ``True``
            substitutes a bare ``MagicMock``. Any other value is used as the
            replacement directly, which is how a test supplies its own recorder.
        **run_kwargs: Passed to the ``run`` mock (``return_value``,
            ``side_effect``, ...).

    Returns:
        A patcher usable as either a decorator or a context manager. It supplies
        the stand-in; assert against its ``run`` (and, when scoped, ``Popen``).
    """

    def _new() -> ModuleType:
        overrides: dict[str, Any] = {"run": MagicMock(**run_kwargs)}
        if popen is True:
            overrides["Popen"] = MagicMock()
        elif popen is not None:
            overrides["Popen"] = popen
        return stand_in(importlib.import_module("subprocess"), **overrides)

    return patch(f"{target_module}.subprocess", new_callable=_new)


class ChildProcessRefused(OSError):
    """Raised where a child process would start inside :func:`no_child_process`."""


def _refuse_children(event: str, args: tuple[Any, ...]) -> None:
    started = _refusing.get()
    if started is not None and event in _CHILD_EVENTS:
        started.append((event, args))
        raise ChildProcessRefused(f"{event} refused: this code starts no child process")


@contextlib.contextmanager
def no_child_process() -> Iterator[list[tuple[str, tuple[Any, ...]]]]:
    """Refuse every child process started in this context, and record each attempt.

    Yields:
        The attempts, as ``(audit event, its arguments)``. A refusal raises
        :class:`ChildProcessRefused` where the child would start; code that
        swallows it still leaves its attempt here, so assert the list is empty.
    """
    global _audit_hook_added
    if not _audit_hook_added:
        # An audit hook cannot be removed; it refuses nothing outside a context
        # this function opened.
        sys.addaudithook(_refuse_children)
        _audit_hook_added = True
    started: list[tuple[str, tuple[Any, ...]]] = []
    token = _refusing.set(started)
    try:
        yield started
    finally:
        _refusing.reset(token)
