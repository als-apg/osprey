"""Run a pyAML tool inside a journaled guarded run, so an interrupted run is put back.

A pyAML tuning tool (a response-matrix measurement, a tune or orbit correction)
writes many setpoints in sequence. :func:`run_tool` calls it inside
:func:`osprey.runtime.guarded_run.journaled_run`: the target's run lock is held for
the whole call, every device write is journaled (durably, before the address is
first written) and the tool's own writes are the only ones the package lets
through. A run killed outright leaves its durable journal for the next guarded
run on the target to restore.

``run_tool`` stops the tool and guards the deadline; the runtime restores. Every
way a tool stops part-way - the caller's callback says stop, pyAML raises
``KeyboardInterrupt``, the tool returns ``False``, a write is refused - leaves the
guarded run with an exception, and ``journaled_run`` writes back what the run
moved as it leaves, hangs the report on the exception and prints the run's one
tagged line, ``OSPREY_GUARDED_RUN_RESTORE <json>``, on stderr. A run that
completes prints the same line with ``aborted: false``. A ``run_tool`` nested in
another runs inside the outer guarded run, so a stop in it ends the outer run
too, restored and reported once. From a helper thread it runs under the
process's run: the guarded run belongs to the process, not to a thread.

Inside the python_executor sandbox the executor exports the absolute time at which
it kills the script (``OSPREY_EXECUTION_DEADLINE``, Unix seconds). When the tool
method accepts a ``callback``, ``run_tool`` chains a deadline check behind the
caller's callback, so the tool is stopped through pyAML's own abort path, and
restored, while there is still time to write everything back.

A Ctrl-C (``SIGINT``) aborts the whole guarded run. The outermost ``run_tool``,
when it runs on the main thread, installs a ``SIGINT`` handler for the duration of
the call and puts the previous one back on exit. The handler sets one abort flag
that every nested ``run_tool`` shares. A tool method that accepts a ``callback``
always receives one that returns ``False`` once the flag is set, deadline or not,
so the tool stops through pyAML's own abort path; for a method that takes none the
handler raises ``KeyboardInterrupt`` at once. Either way the guarded run is
restored and ``run_tool`` raises ``KeyboardInterrupt`` carrying the report, so the
script ends and no later statement writes. The runtime's exit path ignores
``SIGINT`` while it restores and reports.
"""

from __future__ import annotations

import contextlib
import inspect
import signal
import threading
import time
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from pyaml.tuning_tools.measurement_tool import MeasurementTool

import osprey.runtime
from osprey.runtime.guarded_run import (
    RESTORE_REPORT_TAG,
    RestoreReport,
    journaled_run,
    restore_writes,
)
from osprey.runtime.journal import Journal, pop_journal, push_journal, read_map

__all__ = [
    "DEFAULT_WRITE_LATENCY_S",
    "EXIT_RESERVE_S",
    "REPORT_TAG",
    "RestoreReport",
    "run_tool",
]

#: Prefix of the one tagged line each guarded run prints; the python executor
#: files every line carrying it.
REPORT_TAG = RESTORE_REPORT_TAG

#: Write latency, in seconds, the deadline guard budgets before a write is timed.
DEFAULT_WRITE_LATENCY_S = 1.0

#: Seconds kept free after the restore for the script's tail and interpreter exit.
EXIT_RESERVE_S = 5.0

_Clock = Callable[[], float]

#: Wall clock the deadline guard measures against; the deadline is Unix seconds.
_clock: _Clock = time.time


@dataclass
class _Level:
    """One ``run_tool`` call on the stack of a guarded run.

    Attributes:
        accepts_callback: The tool method has a ``callback`` parameter, so a Ctrl-C
            stops it through that callback rather than by raising.
        in_tool: The tool method is running (not yet returned or raised).
    """

    accepts_callback: bool
    in_tool: bool = False


@dataclass
class _Interrupt:
    """The Ctrl-C state one outermost ``run_tool`` shares with every nested one.

    Attributes:
        requested: The ``SIGINT`` handler asked the run to abort.
        stopped: A level's callback asked its tool to stop, or a tool returned
            ``False``.
        levels: The ``run_tool`` calls in progress, outermost first.
    """

    requested: bool = False
    stopped: bool = False
    levels: list[_Level] = field(default_factory=list)


#: The Ctrl-C state of the ``run_tool`` calls in progress in this context.
_INTERRUPT: ContextVar[_Interrupt | None] = ContextVar("pyaml_cs_osprey_interrupt", default=None)


def _sigint_handler(state: _Interrupt) -> Callable[[int, Any], None]:
    """The ``SIGINT`` handler of the guarded run whose state is ``state``.

    It sets the abort flag, and raises ``KeyboardInterrupt`` when the innermost
    tool method is running and takes no callback that could stop it.
    """

    def handler(_signum: int, _frame: Any) -> None:
        state.requested = True
        level = state.levels[-1] if state.levels else None
        if level is not None and level.in_tool and not level.accepts_callback:
            raise KeyboardInterrupt

    return handler


@contextlib.contextmanager
def _interrupt_scope() -> Iterator[_Interrupt]:
    """The Ctrl-C state for one ``run_tool`` call, installing the handler when outermost.

    The outermost call is the one that finds no state in its context. Only that
    call, on the main thread, installs a ``SIGINT`` handler, and it puts the
    previous handler back on exit. A handler installed outside Python
    (``signal.getsignal`` answers ``None``) cannot be put back, so none is
    installed then. A nested call shares the outer call's state.
    """
    state = _INTERRUPT.get()
    if state is not None:
        yield state
        return
    state = _Interrupt()
    token = _INTERRUPT.set(state)
    previous: Any = None
    installed = False
    if threading.current_thread() is threading.main_thread():
        previous = signal.getsignal(signal.SIGINT)
        if previous is not None:
            signal.signal(signal.SIGINT, _sigint_handler(state))
            installed = True
    try:
        yield state
    finally:
        if installed:
            signal.signal(signal.SIGINT, previous)
        _INTERRUPT.reset(token)


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


def _seed_interval(tool_method: Callable[..., Any], kwargs: dict[str, Any]) -> float:
    """The callback interval implied by the run's sleeps.

    A per-call ``sleep_between_step`` / ``sleep_between_meas`` keyword overrides the
    tool's attribute of the same name; a missing or ``None`` value counts as zero.
    """
    owner = getattr(tool_method, "__self__", None)
    total = 0.0
    for name in ("sleep_between_step", "sleep_between_meas"):
        value = kwargs.get(name)
        if value is None:
            value = getattr(owner, name, None)
        try:
            total += max(0.0, float(value)) if value is not None else 0.0
        except (TypeError, ValueError):
            continue
    return total


def _deadline_callback(
    caller: Callable[..., Any] | None,
    journal: Journal,
    *,
    deadline: float,
    seed_interval: float,
) -> tuple[Callable[[Any, Any], bool], Callable[[], None]]:
    """A callback that stops the tool when the caller says so or the deadline is near.

    The caller's callback runs first; its result is judged by pyAML's own rule, so
    any falsy value (``None`` included) stops the tool. Otherwise the time left
    before ``deadline`` is compared with ``margin + EXIT_RESERVE_S``, where::

        margin = 2 * (interval + latency * (writes + 1) + read_latency)

    ``interval`` is the longest gap between consecutive invocations, never less
    than ``seed_interval``; ``latency`` is the longest journaled write latency
    (``DEFAULT_WRITE_LATENCY_S`` before the first one); ``writes`` is
    :func:`osprey.runtime.guarded_run.restore_writes`, the writes a ``max_step``
    walk-back needs over the journaled addresses from their current values,
    read once per invocation; ``read_latency`` is the
    longest such read.

    Args:
        caller: The caller's callback, or ``None``.
        journal: The journal of the guarded run; invocations are noted in it.
        deadline: Unix time at which the sandbox is killed.
        seed_interval: The interval assumed before two invocations are seen.

    Returns:
        ``(callback, disarm)``: the chained callback, which returns a strict
        ``bool``, and a function that makes it inert (it then returns ``True``
        and does nothing).
    """
    armed = True
    read_latency = 0.0
    max_steps: dict[str, float | None] = {}

    def disarm() -> None:
        nonlocal armed
        armed = False

    def callback(action: Any, data: Any) -> bool:
        nonlocal read_latency
        if not armed:
            return True
        if caller is not None and not caller(action, data):
            return False
        journal.note_callback(_clock())
        gap = journal.max_callback_gap
        interval = seed_interval if gap is None else max(seed_interval, gap)
        latency = journal.max_write_latency
        if latency is None:
            latency = DEFAULT_WRITE_LATENCY_S
        addresses = journal.addresses
        current: dict[str, Any] | None = None
        if addresses:
            start = _clock()
            try:
                current = read_map(addresses)
            except Exception:  # unknown values budget one write each
                current = None
            read_latency = max(read_latency, _clock() - start)
        writes = restore_writes(journal, current, max_steps)
        margin = 2.0 * (interval + latency * (writes + 1) + read_latency)
        return bool(deadline - _clock() >= margin + EXIT_RESERVE_S)

    return callback, disarm


def run_tool(
    tool_method: Callable[..., Any],
    /,
    *args: Any,
    callback: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> RestoreReport:
    """Call a pyAML tool method in a journaled guarded run; restore it if it does not complete.

    Outcomes:

    * The method returns anything but ``False``: the run completed; the machine
      stays as the tool left it and the report is empty with ``aborted=False``.
    * The method returns ``False`` (how a pyAML ``measure()`` reports an
      interrupted run), or raises ``KeyboardInterrupt`` after a callback
      returned a falsy value (how pyAML's callback machinery aborts): the
      guarded run is restored and its report is returned with ``aborted=True``.
    * The method raises ``KeyboardInterrupt`` that no callback asked for (a
      Ctrl-C): the guarded run is restored, the report is attached as
      ``exc.restore_report``, and the interrupt propagates.
    * The method raises any other exception, ``SystemExit`` and
      ``GeneratorExit`` included: the guarded run is restored, the report is
      attached as ``exc.restore_report``, and the exception propagates.
    * A Ctrl-C (``SIGINT``) while the tool runs, in this or any enclosing
      ``run_tool``: the tool is stopped (through its callback when it takes one),
      the guarded run is restored, and ``KeyboardInterrupt`` is raised carrying
      the report as ``restore_report``, whatever the method returned.

    In every case the tool's own callback slot is cleared afterwards.

    The run is :func:`osprey.runtime.guarded_run.journaled_run` on the process's
    stamped control target: it holds the target's run lock for the whole call,
    restores a killed run's journal before the method starts, journals each
    address durably before its first write, and is the one place the run is
    restored and reported: its exit path ignores ``SIGINT`` and prints the run's
    one tagged line on stderr, for a completed run as for a stopped one. A
    ``run_tool`` nested in another, or in a journaled run opened by its caller,
    runs under the outer run's lock and journal and prints nothing; a stop in
    it, a ``False`` return included, ends the outer run, which restores and
    reports once. From a helper thread it runs under the process's run: it
    journals into the run held or opened elsewhere in the process, or opens
    one under the lock the process holds.

    When ``OSPREY_EXECUTION_DEADLINE`` is set and the method has a ``callback``
    parameter, the method receives a chained callback that also stops the tool
    once the time left no longer covers a restore (see ``_deadline_callback``);
    the report then carries ``deadline_guard=True``. The chain is made inert on
    return.

    Args:
        tool_method: A bound pyAML tool method, e.g. ``sr.live.trm.measure``.
        *args: Positional arguments for the method.
        callback: Passed to the method as ``callback=`` when given (wrapped, so
            its results are observed), behind the deadline check when the run
            is guarded.
        **kwargs: Keyword arguments for the method.

    Returns:
        The run's report.

    Raises:
        osprey.runtime.guarded_run.OspreyRunBusy: Another guarded run holds the
            target's run lock; the method was not called.
        osprey.runtime.guarded_run.GuardedRunDirError: The target's guarded-run
            directory cannot be resolved or written; the method was not called.
        osprey.runtime.journal.OspreyStaleJournal: A killed run left a journal
            for another control target or generation; the method was not called
            and the journal is left in place.
        osprey.runtime.journal.OspreyRestoreIncomplete: A killed run's journal
            could not be fully restored; the method was not called.
        osprey.runtime.journal.OspreyWriteRefused: This is a readonly run; the
            method was not called.
        Exception: Whatever the method raised, carrying the report as
            ``restore_report`` once the outermost guarded run has restored.
        KeyboardInterrupt: The method was interrupted without a callback
            asking it to stop, the run got a ``SIGINT``, or a nested run was
            stopped; carries the report as ``restore_report`` once the
            outermost guarded run has restored.
    """
    with _interrupt_scope() as state:
        level = _Level(accepts_callback=_accepts_callback(tool_method))
        state.levels.append(level)
        try:
            deadline = osprey.runtime.execution_deadline()
            guarded = deadline is not None and level.accepts_callback
            try:
                with journaled_run(None, deadline_guard=guarded):
                    return _run_guarded(tool_method, args, kwargs, callback, state, level, deadline)
            except KeyboardInterrupt as stop:
                report = getattr(stop, "restore_report", None)
                if not isinstance(report, RestoreReport) or state.requested or not state.stopped:
                    raise
                return report
        finally:
            state.levels.remove(level)


def _run_guarded(
    tool_method: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    callback: Callable[..., Any] | None,
    state: _Interrupt,
    level: _Level,
    deadline: float | None,
) -> RestoreReport:
    """:func:`run_tool`'s body, run inside the journaled guarded run.

    ``state`` is the Ctrl-C state shared with every enclosing ``run_tool`` and
    ``level`` is this call's entry on its stack; ``deadline`` is the execution
    deadline, ``None`` when there is none.

    A ``False`` return is a stop: it is noted in ``state`` and raised as
    ``KeyboardInterrupt``, so the guarded run restores it like any other. An
    exception escaping here is restored by the guarded run; one escaping a
    deadline-guarded call carries ``deadline_guard = True`` so its report says so.
    """
    journal = Journal()
    disarm: Callable[[], None] | None = None
    chained: Callable[..., Any] | None = callback
    if deadline is not None and level.accepts_callback:
        chained, disarm = _deadline_callback(
            callback,
            journal,
            deadline=deadline,
            seed_interval=_seed_interval(tool_method, kwargs),
        )
    guarded = disarm is not None
    if chained is not None or level.accepts_callback:
        inner = chained

        def recording(action: Any, data: Any) -> Any:
            """Stop on a Ctrl-C, else pass through to the chain, noting a stop it asks for."""
            if state.requested:
                return False
            if inner is None:
                return True
            result = inner(action, data)
            if not result:
                state.stopped = True
            return result

        kwargs["callback"] = recording

    push_journal(journal)
    try:
        try:
            if state.requested:
                raise KeyboardInterrupt
            level.in_tool = True
            result = tool_method(*args, **kwargs)
        finally:
            level.in_tool = False
            pop_journal(journal)
        if state.requested:
            raise KeyboardInterrupt
        if result is False:
            state.stopped = True
            raise KeyboardInterrupt
    except BaseException as exc:
        if guarded:
            with contextlib.suppress(AttributeError, TypeError):
                exc.deadline_guard = True  # type: ignore[attr-defined]
        raise
    finally:
        if disarm is not None:
            disarm()
        _clear_owner_callback(tool_method)
    return RestoreReport(aborted=False, deadline_guard=guarded)
