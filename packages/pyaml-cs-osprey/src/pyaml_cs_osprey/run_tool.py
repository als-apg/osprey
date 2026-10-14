"""Run a pyAML tool under the control target's run lock, recording what it moves.

A pyAML tuning tool (a response-matrix measurement, a tune or orbit correction)
writes many setpoints in sequence. :func:`run_tool` calls it under the target's
run lock (:func:`osprey.runtime.guarded_run.lock`, joining the hold the process
already has): a second run on the target is refused busy before the tool starts,
and the tool's writes are the only pyAML writes the package lets through
(:mod:`pyaml_cs_osprey._run`). Before the first write of each address its value
is read once and recorded in memory, in first-write order.

Nothing is written back. A run that stops, fails or is interrupted leaves the
machine where it stopped, and the call lists every setpoint it moved with its
value before the call: :attr:`ToolRun.moved` on return, ``moved_setpoints`` on
an escaping exception. A run that reaches the executor's timeout is killed like
any other; it gets no reserve and no early stop.

Each ``run_tool`` call reports its own outcome: a stop in a nested call does not
stop the outer one. A ``run_tool`` on a helper thread runs under the process's
hold and records into the process's one run.

A Ctrl-C (``SIGINT``) stops every tool of the process's run. The outermost
``run_tool`` of a context, when it runs on the main thread, installs a
``SIGINT`` handler for the duration of the call and puts the previous one back
on exit; a handler installed outside Python is left alone. The handler sets the
run's abort flag, which every ``run_tool`` callback, on any thread, reads first,
so a tool that takes a ``callback`` stops through pyAML's own abort path. For a
main-thread tool that takes none the handler raises ``KeyboardInterrupt`` at
once. A helper-thread tool that takes no callback is not stopped by the flag:
it ends with the process when the main thread's ``KeyboardInterrupt`` ends the
script. After a Ctrl-C ``run_tool`` raises ``KeyboardInterrupt`` carrying
``moved_setpoints``, so the script ends and no later statement writes.
"""

from __future__ import annotations

import contextlib
import inspect
import signal
import threading
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from pyaml.tuning_tools.measurement_tool import MeasurementTool

from osprey.runtime.guarded_run import lock
from pyaml_cs_osprey import _run

__all__ = ["ToolRun", "run_tool"]


@dataclass(frozen=True)
class ToolRun:
    """How one ``run_tool`` call ended.

    Attributes:
        completed: The tool ran to its end; ``False`` when it was stopped.
        moved: Each address the process's run wrote, with its value before
            its first write, in first-write order. Nothing was written back.
    """

    completed: bool
    moved: dict[str, Any]


@dataclass
class _Level:
    """One ``run_tool`` call on this context's stack.

    Attributes:
        accepts_callback: The tool method has a ``callback`` parameter, so a Ctrl-C
            stops it through that callback rather than by raising.
        in_tool: The tool method is running (not yet returned or raised).
        stopped: The caller's callback asked the tool to stop, or the tool
            returned ``False``.
    """

    accepts_callback: bool
    in_tool: bool = False
    stopped: bool = False


@dataclass
class _Interrupt:
    """The ``run_tool`` calls in progress in one context, outermost first.

    It attributes a Ctrl-C to the call it lands in and decides nothing: the
    stop request is the process's run's (:func:`pyaml_cs_osprey._run.abort_requested`).
    """

    levels: list[_Level] = field(default_factory=list)


#: The ``run_tool`` calls in progress in this context.
_INTERRUPT: ContextVar[_Interrupt | None] = ContextVar("pyaml_cs_osprey_interrupt", default=None)


def _sigint_handler(state: _Interrupt) -> Callable[[int, Any], None]:
    """The ``SIGINT`` handler of the context whose call stack is ``state``.

    It asks the process's run to stop, and raises ``KeyboardInterrupt`` when no
    run is open or the innermost tool method is running and takes no callback
    that could stop it.
    """

    def handler(_signum: int, _frame: Any) -> None:
        if not _run.request_abort():
            raise KeyboardInterrupt
        level = state.levels[-1] if state.levels else None
        if level is not None and level.in_tool and not level.accepts_callback:
            raise KeyboardInterrupt

    return handler


@contextlib.contextmanager
def _interrupt_scope() -> Iterator[_Interrupt]:
    """This context's ``run_tool`` call stack, created by the outermost call."""
    state = _INTERRUPT.get()
    if state is not None:
        yield state
        return
    state = _Interrupt()
    token = _INTERRUPT.set(state)
    try:
        yield state
    finally:
        _INTERRUPT.reset(token)


@contextlib.contextmanager
def _sigint_installed(state: _Interrupt, outermost: bool) -> Iterator[None]:
    """Install the ``SIGINT`` handler for the body when this is the context's outermost call.

    Only on the main thread, and only when the handler in place was installed
    by Python (``signal.getsignal`` answers ``None`` otherwise, and it could not
    be put back). The previous handler is put back on exit.
    """
    if not outermost or threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGINT)
    if previous is None:
        yield
        return
    signal.signal(signal.SIGINT, _sigint_handler(state))
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def _clear_owner_callback(tool_method: Callable[..., Any]) -> None:
    """Unregister the callback from the tool that owns ``tool_method``.

    pyAML only ever overwrites a measurement tool's callback slot, and the
    chromaticity monitor sends its first callback before registering the new
    one, so a slot left set would fire inside a later unguarded measurement.
    A tool with a ``chromaticity_monitor`` (a chromaticity response matrix)
    hands the callback on to that monitor, so the monitor's slot is cleared too.
    """
    owner = getattr(tool_method, "__self__", None)
    if not isinstance(owner, MeasurementTool):
        return
    owner._register_callback(None)
    if not hasattr(type(owner), "chromaticity_monitor"):
        return
    try:
        monitor = owner.chromaticity_monitor
    except Exception:  # a monitor that cannot be resolved holds no callback
        return
    if isinstance(monitor, MeasurementTool):
        monitor._register_callback(None)


def _accepts_callback(tool_method: Callable[..., Any]) -> bool:
    """Whether ``tool_method``'s signature has a ``callback`` parameter."""
    try:
        return "callback" in inspect.signature(tool_method).parameters
    except (TypeError, ValueError):
        return False


def run_tool(
    tool_method: Callable[..., Any],
    /,
    *args: Any,
    callback: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> ToolRun:
    """Call a pyAML tool method under the target's run lock, recording what it moves.

    Outcomes:

    * The method returns anything but ``False``: the run completed; the machine
      stays as the tool left it and ``completed`` is ``True``.
    * The method returns ``False`` (how a pyAML ``measure()`` reports an
      interrupted run), the caller's callback returns a falsy value, or the
      method raises ``KeyboardInterrupt`` after such a callback (how pyAML's
      callback machinery aborts): the tool stopped, ``completed`` is ``False``,
      and the machine is left where it stopped.
    * A Ctrl-C (``SIGINT``) while the tool runs, in this or any other
      ``run_tool`` of the process: the tool is stopped (through its callback
      when it takes one) and ``KeyboardInterrupt`` is raised carrying
      ``moved_setpoints``, whatever the method returned.
    * The method raises any other exception, ``KeyboardInterrupt``,
      ``SystemExit`` and ``GeneratorExit`` included: it propagates carrying
      ``moved_setpoints``.

    In every case the tool's own callback slot is cleared afterwards, and
    nothing is written back.

    The call holds the run lock of the process's stamped control target
    (:func:`osprey.runtime.guarded_run.lock`), joining the hold the process
    already has, from this thread or another, and taking it otherwise. A
    method that takes a ``callback`` always gets one, which returns ``False``
    once a Ctrl-C asked the run to stop and otherwise passes the caller's
    callback's result through.

    Args:
        tool_method: A bound pyAML tool method, e.g. ``sr.live.trm.measure``.
        *args: Positional arguments for the method.
        callback: Passed to the method as ``callback=`` when given (wrapped, so
            its results are observed and a Ctrl-C stops the tool).
        **kwargs: Keyword arguments for the method.

    Returns:
        Whether the tool completed, and every setpoint the process's run moved
        with its value before its first write.

    Raises:
        osprey.runtime.guarded_run.OspreyRunBusy: Another run holds the
            target's run lock; the method was not called.
        osprey.runtime.guarded_run.GuardedRunDirError: The target's guarded-run
            directory cannot be resolved or written; the method was not called.
        osprey.runtime.guarded_run.OspreyWriteRefused: This is a readonly run;
            the method was not called.
        KeyboardInterrupt: The run got a ``SIGINT``, or the method was
            interrupted without a callback asking it to stop; carries
            ``moved_setpoints``.
        Exception: Whatever the method raised, carrying ``moved_setpoints``.
    """
    with _interrupt_scope() as state:
        level = _Level(accepts_callback=_accepts_callback(tool_method))
        outermost = not state.levels
        state.levels.append(level)
        try:
            with lock(None, join=True) as directory, _run.opened(directory.name):
                with _sigint_installed(state, outermost):
                    try:
                        return _run_level(tool_method, args, kwargs, callback, level)
                    except BaseException as exc:
                        aborted = _run.abort_requested()
                        moved = _run.moved()
                        if isinstance(exc, KeyboardInterrupt) and level.stopped and not aborted:
                            return ToolRun(completed=False, moved=moved)
                        with contextlib.suppress(AttributeError, TypeError):
                            exc.moved_setpoints = moved  # type: ignore[attr-defined]
                        raise
        finally:
            state.levels.remove(level)


def _run_level(
    tool_method: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    callback: Callable[..., Any] | None,
    level: _Level,
) -> ToolRun:
    """:func:`run_tool`'s body, run inside the process's pyAML run.

    ``level`` is this call's entry on its context's stack. A Ctrl-C the run
    was asked for raises ``KeyboardInterrupt``; a ``False`` return is a stop.
    """
    if callback is not None or level.accepts_callback:

        def chained(action: Any, data: Any) -> Any:
            """Stop on a Ctrl-C, else pass the caller's result through, noting a stop."""
            if _run.abort_requested():
                return False
            if callback is None:
                return True
            result = callback(action, data)
            if not result:
                level.stopped = True
            return result

        kwargs["callback"] = chained

    try:
        if _run.abort_requested():
            raise KeyboardInterrupt
        level.in_tool = True
        try:
            result = tool_method(*args, **kwargs)
        finally:
            level.in_tool = False
    finally:
        _clear_owner_callback(tool_method)
    if _run.abort_requested():
        raise KeyboardInterrupt
    if result is False:
        level.stopped = True
    return ToolRun(completed=not level.stopped, moved=_run.moved())
