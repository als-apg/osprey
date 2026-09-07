"""The notebook kernel's identity, and its launcher.

A notebook kernel runs as its own process, started by the Jupyter server, with
no handle on the web terminal the operator is actually working in. It needs
none: the deployment has one control context, recorded under the shared
agent-data root, and every client reads that one record. What the kernel
supplies is its own name — ``kernel:<kernel_id>``, taken from the connection
file Jupyter wrote for it — so that the records it files are attributable to a
kernel an operator can find and interrupt.

A kernel process must not import the web terminal (it would pull FastAPI and
uvicorn into every notebook), so every MODULE-LEVEL import here is standard
library. :func:`compute_stamps` and :func:`main` do reach into OSPREY — they
stamp the names the executor stamps, from the same modules — but every one of
those imports sits inside the function body, so importing this module costs
the terminal nothing but the standard library. ``ipykernel`` is imported the
same way, inside :func:`main`, so this module stays importable in a process
that has no kernel stack at all.

A cell's refusals are the launcher's other job. A write the connector
refuses raises in the cell rather than in an ``execute()`` call, so nothing on
the executor's path files the record or explains the refusal;
:func:`install_refusal_handler` puts both on the kernel's own exception hook.
What the cell shows is the refusal and one action line, and nothing else: the
process's log records are routed to the terminal log by
:func:`_route_logs_to_process_stderr` rather than left to resolve stderr while
``ipykernel`` is publishing it into the cell.

The kernel starts sandboxed: the launcher stamps the literal ``*=sandbox``
launch pin and no control target at all. What takes it out of that is the cell
hook — :func:`pre_run_cell` rewrites the target, the generation and the launch
pin from the deployment's record before every cell, so a kernel follows the
record for its whole life instead of holding whatever was true when it started.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable, MutableMapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

__all__ = [
    "ENV_CONTROL_TARGET_REFUSAL",
    "ENV_IN_CELL",
    "JUPYTER_SHARED_SUBTREE",
    "KERNEL_SESSION_PREFIX",
    "compute_stamps",
    "install_cell_hooks",
    "install_refusal_handler",
    "install_shell_stream_rearm",
    "kernel_id_from_argv",
    "main",
    "post_run_cell",
    "pre_run_cell",
]

logger = logging.getLogger(__name__)

#: The shared-root subtree holding the notebook sidecar's own state. Named
#: here so the sidecar and the kernel launcher derive it from one constant.
JUPYTER_SHARED_SUBTREE = "jupyter"

#: How a kernel spells itself as an audit session. The prefix is what tells a
#: reader of a record that the session is a notebook kernel and not a terminal.
KERNEL_SESSION_PREFIX = "kernel:"

#: The kernelspec argument naming the connection file the Jupyter server writes
#: per kernel: ``-f {connection_file}``. It is the only handle this process has
#: on its own identity before ``ipykernel`` exists.
CONNECTION_FILE_FLAG = "-f"

#: Jupyter names a connection file ``kernel-<kernel_id>.json``.
CONNECTION_FILE_PREFIX = "kernel-"


def _connection_file(argv: Sequence[str] | None) -> str | None:
    """The path ``-f`` names in *argv*, or ``None`` when it names none.

    Both spellings ``traitlets`` accepts are read. The kernelspec writes the
    two-token form; nothing stops an operator from starting a kernel with the
    joined one.
    """
    if not argv:
        return None
    joined = f"{CONNECTION_FILE_FLAG}="
    for index, argument in enumerate(argv):
        if argument == CONNECTION_FILE_FLAG:
            return (argv[index + 1] if index + 1 < len(argv) else "") or None
        if argument.startswith(joined):
            return argument[len(joined) :] or None
    return None


def kernel_id_from_argv(argv: Sequence[str] | None) -> str | None:
    """This kernel's id, read from the connection file named in *argv*.

    Jupyter writes one connection file per kernel and names it
    ``kernel-<kernel_id>.json``, so the id is in the file name and reading it
    costs no kernel stack. A name in another shape is used whole rather than
    refused: a kernel that cannot say which one it is is worse than one whose
    id spells something unexpected, and no name is worth failing a start over.

    Args:
        argv: The kernel's arguments, without the program name.

    Returns:
        The id, or ``None`` when *argv* names no connection file.
    """
    connection_file = _connection_file(argv)
    if connection_file is None:
        return None
    return Path(connection_file).stem.removeprefix(CONNECTION_FILE_PREFIX).strip() or None


#: The Jupyter server's own token, re-issued per launch by the sidecar. A cell
#: must not be able to read it, and the empty value the kernelspec carries is
#: not enough on its own: the provisioner merges the kernelspec env OVER
#: ``os.environ``, so an empty value keeps the KEY. :func:`main` pops the name
#: outright, and a cell's environment then holds no ``JUPYTER_TOKEN`` at all.
JUPYTER_TOKEN_ENV_VAR = "JUPYTER_TOKEN"

#: The config path the sidecar passes down, and the one the config loader
#: reads. They are different names: the web terminal resolves its deployment
#: from ``OSPREY_CONFIG``, while :mod:`osprey_connectors.config` looks only at
#: ``CONFIG_FILE`` or a ``config.yml`` in the working directory. A kernel's
#: working directory is the notebooks folder, so without the second name every
#: runtime call in a cell would fall back to defaults — a ``mock`` control
#: system with no connector types registered. :func:`_prepare_environment`
#: publishes the second from the first, the way the connector host does for
#: its own child.
OSPREY_CONFIG_ENV_VAR = "OSPREY_CONFIG"
CONFIG_FILE_ENV_VAR = "CONFIG_FILE"


def compute_stamps(
    kernel_id: str | None,
    env: MutableMapping[str, str],
) -> dict[str, str]:
    """Stamp this kernel's identity into *env* and pin it sandboxed.

    Identity first, because everything resolved afterwards is looked up UNDER
    it: the audit records this kernel files, and the records it is attributed
    by. The launch pin follows, and it is the literal ``*=sandbox`` rather
    than whatever the store would answer — nothing here selects a control
    target, so reads route to the deployment baseline and every write is
    refused by the connector's launch pin.

    The three target names are REMOVED rather than left inherited, and they are
    named by reusing ``python_executor.executor``'s constants rather than by
    restating them: a stamp passed down from whatever started the sidecar would
    route cells at a target nobody selected here.

    Args:
        kernel_id: This kernel's id, as :func:`kernel_id_from_argv` derives it,
            or ``None`` when the kernel was started without a connection file.
            An unnamed kernel is stamped with no session at all and any
            inherited one is dropped — that name belongs to another process,
            and wearing it would file this kernel's records under it.
        env: This process's own environment — ``os.environ``, and nothing else.
            The resolvers a cell calls read the PROCESS environment, so the
            identity has to be visible to them. Stamped in place, and the three
            target names are REMOVED from it.

    Returns:
        The names this call stamped, mapped to their values. Removals are not
        in it: they are absences, which is how every reader of the stamp reads
        "no target".
    """
    from osprey.audit import posture
    from osprey.mcp_server.python_executor import executor
    from osprey_connectors import posture_store

    stamps: dict[str, str] = {}
    if kernel_id:
        stamps[posture.POSTURE_SESSION_ENV_VAR] = f"{KERNEL_SESSION_PREFIX}{kernel_id}"
    else:
        env.pop(posture.POSTURE_SESSION_ENV_VAR, None)
    for name in executor._STAMP_ENV_NAMES:
        env.pop(name, None)
    stamps[executor.ENV_LAUNCH_POSTURE] = posture_store.launch_posture_stamp(
        None, posture_store.POSTURE_SANDBOX
    )

    env.update(stamps)
    return stamps


def _prepare_environment(argv: Sequence[str] | None = None) -> dict[str, str]:
    """Take the server's token away and stamp this kernel's identity.

    ``OSPREY_AGENT_DATA_ROOT`` is left exactly as the kernelspec carries it:
    that stamp is the sidecar's shared root, and the control context and the
    audit trail are both read and written under it.

    The config path is published first, under the name the loader reads, so
    that every runtime call a cell makes sees the deployment rather than the
    defaults. A ``CONFIG_FILE`` that already names a path is left alone:
    whoever set it chose it on purpose. A blank one is not a choice — the
    loader treats it as unset — so it is filled like an absent name rather
    than preserved, which ``setdefault`` would have done.

    Args:
        argv: The kernel's arguments, without the program name. The connection
            file among them is what names this kernel.

    Returns:
        The stamps :func:`compute_stamps` applied, for a caller that wants to
        report them. Nothing here raises on a kernel that could not be named:
        it runs sandboxed, which is how every kernel starts.
    """
    os.environ.pop(JUPYTER_TOKEN_ENV_VAR, None)
    config_path = os.environ.get(OSPREY_CONFIG_ENV_VAR)
    if config_path and not os.environ.get(CONFIG_FILE_ENV_VAR):
        os.environ[CONFIG_FILE_ENV_VAR] = config_path

    return compute_stamps(kernel_id_from_argv(argv), os.environ)


#: Why this cell may not act on the record, when it may not. Written by
#: :func:`pre_run_cell` and read by :func:`osprey.runtime._get_connector`, which
#: raises it on the cell's first control-system call: ``IPython``'s event
#: trigger swallows a callback's exception, so the hook cannot refuse a cell by
#: raising and leaves the refusal where the cell will actually meet it.
#:
#: The value is ``switch_in_progress:<comma-separated pids>`` — the token
#: :func:`~osprey.mcp_server.python_executor.executor.switch_in_progress_message`
#: opens with, so one string identifies the condition on every surface that
#: refuses for it. Absence is the normal state and means nothing is in the way.
ENV_CONTROL_TARGET_REFUSAL = "OSPREY_CONTROL_TARGET_REFUSAL"

#: Whether a cell is running in this process right now. It is what makes the
#: in-flight marker a CELL's claim rather than the kernel's: a background thread
#: reaching the control system between cells writes no marker, because there is
#: no cell for a switch to wait on — its writes are held by the generation pin
#: instead. Set by :func:`pre_run_cell`, removed by :func:`post_run_cell`, and
#: read by :func:`osprey.runtime._get_connector`, which writes the marker.
ENV_IN_CELL = "OSPREY_NOTEBOOK_IN_CELL"

#: The one value :data:`ENV_IN_CELL` is ever set to. Readers test for presence.
IN_CELL = "1"


def _sandbox_pin() -> str:
    """The launch pin for a cell that reached no target: sandboxed everywhere."""
    from osprey_connectors import posture_store

    return posture_store.launch_posture_stamp(None, posture_store.POSTURE_SANDBOX)


def _switch_in_progress_refusal(pids: tuple[int, ...]) -> str:
    """The :data:`ENV_CONTROL_TARGET_REFUSAL` value naming *pids*.

    The token is the executor's own ``failure_kind`` for the same condition,
    reused rather than re-spelled: a switch in flight refuses a cell and an
    ``execute()`` call for one reason, and one string is what lets an operator
    reading either recognise it as the same wait.
    """
    from osprey.mcp_server.python_executor import executor

    return f"{executor.FAILURE_KIND_SWITCH_IN_PROGRESS}:{','.join(str(pid) for pid in pids)}"


def _stamp_from_record(env: MutableMapping[str, str]) -> None:
    """Rewrite every routing name in *env* from the deployment's record.

    A total function of the record: every name this contract occupies is given
    a value or an absence on every call, so a cell is routed
    by what the record says now and never by what a previous cell left behind.
    That is the whole difference between a kernel and an executor sandbox — a
    sandbox is stamped once because it dies at the end of the run, while a
    kernel outlives every switch an operator makes under it.

    The order is fail-closed: every name is cleared and the sandbox pin is laid
    down BEFORE the record is read, and only the branch that reaches a target
    lifts it. A pin that is merely absent reads as "permitted", so the pin must
    exist from the first statement rather than be assigned at the end.

    The three outcomes, in the order they are decided:

    * a live controls server is mid-switch — no target, sandboxed, and
      :data:`ENV_CONTROL_TARGET_REFUSAL` names the servers to wait for;
    * no readable record, or one naming a target this deployment cannot build —
      no target, sandboxed, no refusal: the cell reads the deployment baseline,
      which is the same fail-closed outcome the kernel starts in;
    * otherwise the record's target, its generation, and the launch pin the
      record's posture answers for that target.

    The gate and the launch pin are the executor's own, called rather than
    restated, so the notebook and the ``execute()`` tool cannot disagree about
    what "settled" means or about which posture a run starts under.

    Args:
        env: This process's environment — ``os.environ``, and nothing else.
            The runtime resolves a cell's connector from the PROCESS
            environment, so that is where the stamp has to land.
    """
    from osprey.mcp_server.python_executor import executor
    from osprey_connectors import posture_store

    for name in (*executor._STAMP_ENV_NAMES, ENV_CONTROL_TARGET_REFUSAL):
        env.pop(name, None)
    env[executor.ENV_LAUNCH_POSTURE] = _sandbox_pin()

    record = executor._deployment_record()
    if record is None:
        return
    if blocking := executor._blocking_pids(record):
        env[ENV_CONTROL_TARGET_REFUSAL] = _switch_in_progress_refusal(blocking)
        return
    if not executor._target_is_resolvable(record.target):
        return

    permitted = record.posture.get(record.target) != posture_store.POSTURE_SANDBOX
    env[executor.ENV_CONTROL_TARGET] = record.target
    env[executor.ENV_CONTROL_TARGET_GENERATION] = str(record.generation)
    env[executor.ENV_LAUNCH_POSTURE] = posture_store.launch_posture_stamp(
        record.target,
        posture_store.POSTURE_WRITES if permitted else posture_store.POSTURE_SANDBOX,
    )


def _cell_kernel_id() -> str | None:
    """This kernel's id, read back off the session stamp it wrote at start.

    The stamp is the one place the id is kept — deriving it here rather than
    holding a module-level copy is what keeps the records this kernel files and
    the markers it removes named by the same string.

    Returns:
        The id, or ``None`` for a kernel that could not name itself and
        therefore stamped no session.
    """
    from osprey.audit import posture

    session = posture.posture_session() or ""
    if not session.startswith(KERNEL_SESSION_PREFIX):
        return None
    return session[len(KERNEL_SESSION_PREFIX) :] or None


def _remove_cell_markers() -> None:
    """Drop every in-flight marker this kernel's cells left behind.

    The marker is written lazily, by the first control-system call in a cell,
    and there is no handle on it afterwards — so the end of the cell removes
    them by name instead: every marker carrying THIS kernel's id, and no other
    process's. Leaving one behind would refuse every later target switch with
    nothing an operator could stop.

    Never raises: this runs in ``run_cell``'s ``finally``, where an exception
    would replace whatever the cell was actually doing.
    """
    kernel_id = _cell_kernel_id()
    if kernel_id is None:
        return
    try:
        from osprey.mcp_server.control_system import target_state

        for entry in target_state.state_dir().glob(target_state.INFLIGHT_FILE_GLOB):
            marker = target_state.read_file(entry)
            if marker is not None and marker.get("kernel_id") == kernel_id:
                entry.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 - a marker left behind is not worth a cell
        logger.warning(
            "Could not remove this kernel's in-flight markers; a target switch may be "
            "refused until they are swept",
            exc_info=True,
        )


def pre_run_cell(info: Any = None) -> None:
    """Route this cell at the deployment's control target, then open the cell.

    Nothing here refuses a cell: ``IPython`` swallows a callback's exception,
    so a raise would be lost and the cell would run anyway. What a refusal
    leaves behind is :data:`ENV_CONTROL_TARGET_REFUSAL`, which the cell's first
    control-system call raises on.

    Args:
        info: ``IPython``'s ``ExecutionInfo``, unused. The routing is read from
            the record, not from the code about to run.
    """
    _stamp_from_record(os.environ)
    os.environ[ENV_IN_CELL] = IN_CELL


def post_run_cell(result: Any = None) -> None:
    """Close the cell: no cell is running, and its markers are gone.

    ``IPython`` fires this from ``run_cell``'s ``finally``, so it runs for a
    cell that raised and for one that was interrupted as well as for one that
    finished — which is what makes the marker removal reliable enough for a
    switch to wait on it.

    Args:
        result: ``IPython``'s ``ExecutionResult``, unused. A cell that failed
            still held the target while it ran.
    """
    os.environ.pop(ENV_IN_CELL, None)
    _remove_cell_markers()


def install_cell_hooks(shell: Any) -> None:
    """Put the per-cell routing hooks on *shell*.

    Args:
        shell: The kernel's ``InteractiveShell``.
    """
    shell.events.register("pre_run_cell", pre_run_cell)
    shell.events.register("post_run_cell", post_run_cell)


#: The audit surface a notebook cell's refusals file under. Named here rather
#: than beside :data:`~osprey.audit.envelope.SURFACE_EXECUTOR` because that is
#: where the executor's surface sits only for the ``source`` exemption it is
#: granted; every other surface in the tree is declared by the module that
#: emits it, and this one has no exemption to ask for.
SURFACE_NOTEBOOK_KERNEL = "notebook_kernel"

#: Audit ``subject`` for every record written here. A refusal that reached the
#: kernel's hook was raised by a cell, and a cell has no name to give — the
#: channel the write named goes in ``detail`` instead.
REFUSAL_SUBJECT = "notebook_cell"

#: ``reason`` codes, one per refusal class, keyed by class name so the mapping
#: costs no import at module scope. A class that is not in it is a subclass of
#: one that is, and files under the fallback rather than under a guess.
_REFUSAL_REASONS = {
    "ChannelWriteBlockedError": "channel_write_blocked",
    "ChannelLimitsViolationError": "channel_limits_violation",
    "ControlTargetChangedError": "control_target_changed",
}

#: The target moved while the cell was running. A kernel re-routes itself from
#: the record before every cell, so the cell is what picks the new target up.
HINT_TARGET_CHANGED = "The control target changed while this cell ran. Re-run the cell."

#: The launch pin refused: the cell was opened with writes off, for its target
#: or for every target. The chip is where that is turned on, and the next cell
#: is stamped from it.
HINT_WRITES_OFF = (
    "Writes are off for this cell. Turn writes on from the chip, then re-run the cell."
)


def _refusal_classes() -> tuple[type[BaseException], ...]:
    """The exception classes the kernel's hook answers for.

    Imported inside the function for the module's one-way dependency rule: a
    process that only reads the binding must not pull the connectors package
    in behind it.
    """
    from osprey.runtime import ControlTargetChangedError
    from osprey_connectors.errors import ChannelLimitsViolationError, ChannelWriteBlockedError

    return (ChannelWriteBlockedError, ChannelLimitsViolationError, ControlTargetChangedError)


def _hint_for(value: BaseException) -> str | None:
    """The one action line for *value*, or ``None`` when there is no action.

    Every branch is read from the stamp this cell was given, so the hint and
    the connector's own refusal text answer from one state rather than from a
    flag the kernel would have to keep in step. A live-store refusal, a limits
    violation and a switch in flight get no line: their own message already
    names what to do, and a second line would either repeat it or send the
    operator somewhere the message did not.

    Args:
        value: The refusal that reached the hook.

    Returns:
        The line to print before the traceback, or ``None`` to print none.
    """
    from osprey.mcp_server.python_executor import executor
    from osprey.runtime import ControlTargetChangedError, SwitchInProgressError
    from osprey_connectors import posture_store
    from osprey_connectors.errors import ChannelWriteBlockedError

    if isinstance(value, SwitchInProgressError):
        return None
    if isinstance(value, ControlTargetChangedError):
        return HINT_TARGET_CHANGED
    if not isinstance(value, ChannelWriteBlockedError):
        return None
    target = (os.environ.get(executor.ENV_CONTROL_TARGET) or "").strip() or None
    if posture_store.launch_permits(target):
        return None
    return HINT_WRITES_OFF


def _record_refusal(value: BaseException) -> None:
    """File one audit record for *value* under :data:`SURFACE_NOTEBOOK_KERNEL`.

    The fields are the executor's, resolved the same way. The writer swallows
    its own failures, so the lazy imports are the only thing left that could
    raise and they are guarded too: a refusal that could not be recorded is
    still a refusal, and must still reach the cell.
    """
    try:
        from osprey.audit import posture
        from osprey.audit.envelope import DECISION_REFUSED
        from osprey.audit.writer import record

        channel = getattr(value, "channel_address", None)
        record(
            decision=DECISION_REFUSED,
            reason=_REFUSAL_REASONS.get(type(value).__name__, "refused"),
            surface=SURFACE_NOTEBOOK_KERNEL,
            posture=posture.posture(),
            posture_source=posture.posture_source(),
            session=posture.posture_session(),
            subject=REFUSAL_SUBJECT,
            detail=f"channel={channel}" if isinstance(channel, str) and channel else None,
        )
    except Exception:  # noqa: BLE001 - the audit trail degrades; the refusal does not
        logger.warning("Could not record the notebook refusal for audit", exc_info=True)


def _refusal_handler(
    shell: Any,
    etype: type[BaseException],
    value: BaseException,
    tb: TracebackType | None,
    tb_offset: int | None = None,
) -> None:
    """Record a refusal, say what to do about it, then show it as usual.

    ``IPython`` binds this as a method of the shell, which is why *shell* is
    the first argument rather than a closure.

    Args:
        shell: The ``InteractiveShell`` the hook was installed on.
        etype: The refusal's class.
        value: The refusal.
        tb: Its traceback.
        tb_offset: Frames ``IPython`` wants skipped, passed straight through.

    Returns:
        ``None`` — the traceback is displayed by ``showtraceback`` here rather
        than by handing a structured one back for ``IPython`` to display.
    """
    _record_refusal(value)
    hint = _hint_for(value)
    if hint is not None:
        print(hint)
    shell.showtraceback((etype, value, tb), tb_offset=tb_offset)


def install_refusal_handler(shell: Any) -> None:
    """Put the refusal hook on *shell*.

    A cell's write goes to the connector directly, so a refusal surfaces as a
    raised exception and nothing else — no audit record, and no line saying
    what the operator does about it. This hook supplies both, and leaves every
    other exception alone.

    Args:
        shell: The kernel's ``InteractiveShell``.
    """
    shell.set_custom_exc(_refusal_classes(), _refusal_handler)


def _initialize_registry() -> None:
    """Load the deployment's registry, the way the executor sandbox does.

    A cell's ``read_channel`` builds its connector through the registry, and a
    fresh process has nothing registered: the executor sandbox loads the
    registry before user code runs, and a kernel is the same kind of process
    without that preamble. The call, its arguments and its guard are the
    sandbox's: a registry that fails to load leaves the kernel usable for
    everything that does not need one, and the failure is in the log.
    """
    try:
        from osprey.registry import initialize_registry

        initialize_registry(auto_export=False, config_path=os.environ.get(CONFIG_FILE_ENV_VAR))
    except Exception:  # noqa: BLE001 - the kernel still starts; the cause is logged
        logger.warning("Registry initialization failed", exc_info=True)


#: Set on ``SubshellManager`` once :func:`install_shell_stream_rearm` has wrapped
#: its reply send, so a second call is a no-op rather than a second wrapper.
_SHELL_REARM_MARK = "_osprey_shell_stream_rearm"


def install_shell_stream_rearm(shell_stream: Callable[[], Any]) -> None:
    """Re-arm the kernel's shell stream after every reply it sends behind the stream's back.

    ``ipykernel`` 7 reads shell requests through a ``ZMQStream`` on the ROUTER
    socket, but writes every reply to that socket directly, from the shell
    channel thread (``SubshellManager._send_on_shell_channel``). A direct send
    makes libzmq process the socket's pending commands, and that consumes the
    edge-triggered wake-up the stream's event loop is waiting on. ``pyzmq``
    re-reads ``ZMQ_EVENTS`` after each of its own operations and never after
    somebody else's, so a request that arrived in that window sits in the
    socket until some later command wakes the stream — in practice the next
    client's connection.

    The first message on a fresh channels socket is the one that lands there.
    ``jupyter_server`` nudges a new connection with a ``kernel_info_request`` on
    a transient shell channel and closes that channel as soon as the control
    channel answers, so the kernel's shell reply is always sent to a peer that
    is gone: the ROUTER drops it, and a reply that went nowhere produces no
    follow-up command. A client whose first request arrived while that reply
    was being sent waits forever. ``ipykernel`` 6 sent replies through the
    stream itself and never had this window.

    The wrapper re-reads the stream's events after each direct send, which is
    what ``pyzmq`` does after its own sends. It runs on the shell channel
    thread, where both the send and the stream live. Idempotent, and a no-op
    on an ``ipykernel`` without ``SubshellManager``.

    Args:
        shell_stream: Returns the ``ZMQStream`` wrapping the shell socket. Read
            lazily, because the stream exists only once the kernel does.
    """
    try:
        from ipykernel.subshell_manager import SubshellManager
    except ImportError:  # ipykernel < 7 replies through the stream; nothing to re-arm
        return
    if getattr(SubshellManager, _SHELL_REARM_MARK, False):
        return
    send_on_shell_channel = SubshellManager._send_on_shell_channel

    def send_and_rearm(self: Any, msg: Any) -> None:
        send_on_shell_channel(self, msg)
        stream = shell_stream()
        rebuild = getattr(stream, "_rebuild_io_state", None)
        if rebuild is not None:
            rebuild()

    SubshellManager._send_on_shell_channel = send_and_rearm  # type: ignore[method-assign]
    setattr(SubshellManager, _SHELL_REARM_MARK, True)


#: Format for this process's log records on the terminal log. Level and logger
#: name are what make a kernel's line findable among the sidecar's own.
LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"

#: The file descriptor a process inherits its standard error on.
_STDERR_FD = 2


def _writes_to_process_stderr(handler: logging.Handler) -> bool:
    """Whether *handler* writes where ``ipykernel`` will put the cell."""
    stream = getattr(handler, "stream", None)
    if stream is None:
        return False
    if stream is sys.stderr or stream is sys.__stderr__:
        return True
    try:
        return bool(stream.fileno() == _STDERR_FD)
    except (AttributeError, OSError, ValueError):
        return False


def _route_logs_to_process_stderr() -> None:
    """Send this process's log records to the stderr the kernel was started on.

    A kernel's log records are for whoever reads the terminal log, and every
    one of them was landing in the cell instead: ``ipykernel`` replaces
    ``sys.stderr`` with a stream that publishes to the notebook, so a handler
    that resolves stderr when it emits — the root logger's last-resort handler
    included — writes a cell's refusal audit above the one line the operator is
    meant to read.

    Naming the descriptor is not enough either: with ``capture_fd_output`` on,
    which is ``IPKernelApp``'s default, descriptor 2 itself is replaced by a
    pipe that publishes to the cell AND echoes to the terminal, so a handler
    holding descriptor 2 would print in both places. A duplicate taken before
    the kernel exists refers to the original file, and the replacement does not
    reach it.

    Called once, first thing in :func:`main`, so that everything the
    preparation and the registry log is already routed. No level is set: the
    root logger's own decides, as it did before.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if _writes_to_process_stderr(handler):
            root.removeHandler(handler)
    handler = logging.StreamHandler(os.fdopen(os.dup(_STDERR_FD), "w", buffering=1))
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(handler)


def main(argv: list[str] | None = None) -> None:
    """Run a notebook kernel under this deployment's control context.

    ``ipykernel`` is imported here rather than at module scope so that
    importing this module costs a process no kernel stack.

    Logging is routed before anything else runs, so that the preparation's and
    the registry's own records go where the rest of this process's records go.

    The environment is prepared before the registry is loaded, because the
    registry is created from the config path the preparation publishes; and
    both happen before the kernel exists, so no cell can run ahead of them.

    The kernel statements are separate rather than one chained call because
    things go between them: the shell stream re-arm is installed before
    ``initialize``, which is what builds the kernel and its shell channel; the
    hooks go after it, because ``initialize`` is also what builds the shell;
    and ``start`` does not return until the kernel stops.

    Args:
        argv: Kernel arguments. ``None`` takes them from the command line, the
            way ``ipykernel``'s own entry point does — and the preparation is
            handed the same list, because the connection file among them is
            what names this kernel.
    """
    _route_logs_to_process_stderr()
    _prepare_environment(argv if argv is not None else sys.argv[1:])
    _initialize_registry()

    from ipykernel.kernelapp import IPKernelApp

    app = IPKernelApp.instance()
    install_shell_stream_rearm(lambda: app.kernel.shell_stream)
    app.initialize(argv)
    install_refusal_handler(app.shell)
    install_cell_hooks(app.shell)
    app.start()


if __name__ == "__main__":  # pragma: no cover - the kernelspec's entry point
    main()
