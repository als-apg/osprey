"""The notebook kernel's side of the session binding, and its launcher.

A notebook kernel runs as its own process, started by the Jupyter server, with
no handle on the web terminal that the operator is actually working in. The
session binding is how it finds one: a single JSON document under the shared
agent-data root naming the PTY session most recently attached in the browser,
which the kernel reads at start-up to join that session's posture and control
target instead of guessing at its own.

Both ends of that document live here, and only the path constant and the
reader are used by the web terminal — :mod:`osprey.interfaces.web_terminal`
imports :data:`BINDING_RELPATH` so the path has exactly one producer. The
dependency never runs the other way: a kernel process must not import the web
terminal (it would pull FastAPI and uvicorn into every notebook), so every
MODULE-LEVEL import here is standard library, and :func:`read_binding` repeats
the tolerant-read contract of
:func:`osprey.interfaces.web_terminal._json_store.read_json_object` rather than
importing it. :func:`compute_stamps` and :func:`main` do reach into OSPREY —
they resolve the same records the executor resolves, from the same modules —
but every one of those imports sits inside the function body, so importing
this module costs the terminal nothing but the standard library. ``ipykernel``
is imported the same way, inside :func:`main`, so the binding reader stays
importable in a process that has no kernel stack at all.

A cell's refusals are the launcher's other job. A write the connector
refuses raises in the cell rather than in an ``execute()`` call, so nothing on
the executor's path files the record or explains the refusal;
:func:`install_refusal_handler` puts both on the kernel's own exception hook.
What the cell shows is the refusal and one action line, and nothing else: the
process's log records are routed to the terminal log by
:func:`_route_logs_to_process_stderr` rather than left to resolve stderr while
``ipykernel`` is publishing it into the cell.

A missing or damaged binding is a normal state, not an error — the terminal may
never have attached a session, or the document may be mid-rewrite on a
filesystem without atomic replace. :func:`read_binding` answers ``None`` for
every such case, and the launcher treats ``None`` as "join nothing, run
sandboxed".
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from types import TracebackType
from typing import Any

__all__ = [
    "BINDING_RELPATH",
    "binding_path",
    "compute_stamps",
    "install_refusal_handler",
    "main",
    "read_binding",
]

logger = logging.getLogger(__name__)

#: Location of the session-binding document, relative to the shared
#: agent-data root. The writer and every reader derive their path from this
#: one constant.
BINDING_RELPATH = "jupyter/session-binding.json"


def binding_path(shared_root: str | os.PathLike[str]) -> Path:
    """Return the session-binding document's path under *shared_root*.

    Args:
        shared_root: The shared agent-data root — the one that spans sessions,
            never a session-scoped directory.

    Returns:
        The absolute path of the binding document. The file itself, and the
        directory holding it, may not exist yet.
    """
    return Path(shared_root) / BINDING_RELPATH


def read_binding(shared_root: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Return the session binding under *shared_root*, or ``None``.

    ``None`` covers every degraded outcome — no document, an unreadable one,
    one that is not valid JSON, and one that parses to something other than an
    object — so a caller has one branch to write and never a ``try``.

    Args:
        shared_root: The shared agent-data root to read the binding from.

    Returns:
        The binding as a mapping, or ``None`` when there isn't a readable one.
    """
    try:
        document = json.loads(binding_path(shared_root).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict):
        return None
    return document


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


def _bound_identity(binding: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    """The session key and agent-data root *binding* names, or ``(None, None)``.

    Both halves or neither: a binding that names a session but no root cannot
    say which store that session is recorded in, and one that names a root but
    no session addresses nothing in it. Answering ``(None, None)`` for either
    puts such a document on the same fail-closed path as no document at all.
    """
    if not isinstance(binding, Mapping):
        return None, None
    session_id = binding.get("session_id")
    agent_data_root = binding.get("agent_data_root")
    if not isinstance(session_id, str) or not session_id.strip():
        return None, None
    if not isinstance(agent_data_root, str) or not agent_data_root.strip():
        return None, None
    return session_id.strip(), agent_data_root.strip()


def compute_stamps(
    binding: Mapping[str, Any] | None,
    env: MutableMapping[str, str],
) -> dict[str, str]:
    """Stamp the bound session's identity, posture and target into *env*.

    The kernel is joining a session it is not a child of, so it stamps itself
    with what that session's own children carry. The order is the contract:
    the agent-data root and the posture session go first because everything
    resolved afterwards is looked up UNDER them — the control-target state
    directory follows the root stamp, and so does the posture store — then the
    store's parsed copy is dropped, and only then is the live record resolved
    and the target stamped from it.

    The three target names are stamped exactly as
    ``python_executor.executor._apply_target_stamp`` stamps them, by reusing
    that module's constants and helpers rather than restating them: this is a
    second producer of one routing contract, and a paraphrase is how two
    producers come to disagree. The record is the only thing resolved
    differently — the executor matches on its own parent, and a kernel has no
    parent in the session at all, so it matches on the bound PTY's pid.

    Fail-closed on one point of substance the executor does not share. Where
    the executor's unstamped run means "a session whose target could not be
    named", the kernel's means "no session", so the pin is the literal
    ``*=sandbox`` rather than whatever the store would answer for a session key
    that is not there: reads route to the deployment baseline and every write
    is refused by the connector's launch pin.

    Args:
        binding: The session binding as :func:`read_binding` returns it, or
            ``None`` when there is no readable one.
        env: This process's own environment — ``os.environ``, and nothing else.
            The resolvers called here read the PROCESS environment, so the
            identity half has to be visible to them by the time the target half
            is resolved. Stamped in place, and the three target names are
            REMOVED from it on the fail-closed path, because an inherited stamp
            passed through would route cells at a target nobody selected.

    Returns:
        The names this call stamped, mapped to their values. Removals are not
        in it: they are absences, which is how every reader of the stamp reads
        "no target".
    """
    from osprey.audit import posture
    from osprey.mcp_server.control_system import target_banner
    from osprey.mcp_server.python_executor import executor
    from osprey_connectors import session_store

    stamps: dict[str, str] = {}
    session_id, agent_data_root = _bound_identity(binding)
    if session_id is not None and agent_data_root is not None:
        stamps[session_store.AGENT_DATA_ROOT_ENV_VAR] = agent_data_root
        stamps[posture.POSTURE_SESSION_ENV_VAR] = session_id
        env.update(stamps)
    # The store may already be parsed from whatever root this process inherited.
    session_store.invalidate_cache()

    record = None
    if session_id is not None and binding is not None:
        record = target_banner.session_record_for_pid(binding.get("pty_pid"))

    target = None
    if record is not None:
        # A record without an int generation cannot pin a run, which is the
        # same bar `target_state.session_record(require_generation=True)` sets
        # for the executor; `session_record_for_pid` does not apply it.
        generation = record.get("generation")
        candidate = str(record.get("target"))
        if (
            isinstance(generation, int)
            and not isinstance(generation, bool)
            and executor._target_is_resolvable(candidate)
        ):
            target = candidate

    if record is None or target is None:
        for name in executor._STAMP_ENV_NAMES:
            env.pop(name, None)
        stamps[executor.ENV_LAUNCH_POSTURE] = session_store.launch_posture_stamp(
            None, session_store.POSTURE_SANDBOX
        )
    else:
        stamps[executor.ENV_CONTROL_TARGET] = target
        stamps[executor.ENV_CONTROL_TARGET_GENERATION] = str(int(record["generation"]))
        # The record's identity, so a cell pins against this server's file and
        # not against whatever else is in the state directory.
        stamps[executor.ENV_CONTROL_TARGET_STATE_PID] = str(int(record["server_pid"]))
        stamps[executor.ENV_LAUNCH_POSTURE] = executor._launch_posture(target)

    env.update(stamps)
    return stamps


def _prepare_environment() -> dict[str, str]:
    """Take the server's token away and join the bound session.

    The root the binding lives under is the one
    :func:`osprey_connectors.session_store.agent_data_root` answers: the
    ``OSPREY_AGENT_DATA_ROOT`` stamp the kernelspec carries, else the config
    derivation. That is the same rule the posture store and the control-target
    state file already read by, so the kernel cannot look for the binding in
    one directory and its session's records in another.

    The config path is published first, under the name the loader reads, so
    that the target resolution inside :func:`compute_stamps` — and every
    runtime call a cell makes afterwards — sees the deployment rather than the
    defaults. A ``CONFIG_FILE`` that already names a path is left alone:
    whoever set it chose it on purpose. A blank one is not a choice — the
    loader treats it as unset — so it is filled like an absent name rather
    than preserved, which ``setdefault`` would have done.

    Returns:
        The stamps :func:`compute_stamps` applied, for a caller that wants to
        report them. Nothing here raises on a missing binding: that is the
        ordinary state of a kernel started before any terminal attached.
    """
    os.environ.pop(JUPYTER_TOKEN_ENV_VAR, None)
    config_path = os.environ.get(OSPREY_CONFIG_ENV_VAR)
    if config_path and not os.environ.get(CONFIG_FILE_ENV_VAR):
        os.environ[CONFIG_FILE_ENV_VAR] = config_path

    from osprey_connectors import session_store

    root = session_store.agent_data_root()
    binding = None if root is None else read_binding(root)
    return compute_stamps(binding, os.environ)


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

#: The target moved after this kernel stamped itself. The kernel holds one
#: stamp for its whole life, so following the new target takes a restart.
HINT_TARGET_CHANGED = "The session's control target changed. Restart the kernel to follow it."

#: The launch pin refused, and it was pinned everywhere: no session was bound
#: when the kernel started, so there is nothing to turn on yet.
HINT_NO_SESSION = (
    "No chat session was open when this kernel started. Open one, then restart the kernel."
)

#: The launch pin refused for a named target: the session had writes off for it
#: at launch, and the pin does not re-read the store.
HINT_WRITES_OFF = (
    "This kernel started with writes off. Turn writes on from the chip, then restart the kernel."
)

#: The session this kernel joined is gone rather than re-pointed — "+ New" in
#: the session tile ends one session and attaches another. The pin refuses
#: exactly as it does for a target switch, so without this line the operator is
#: told the control target changed when it did not.
HINT_SESSION_ENDED = "The chat session this kernel followed has ended. Restart the kernel."

#: The connector's remedy for a launch-pin refusal, rewritten for this surface.
#: A script picks up a write state set since it launched by being re-run; a
#: kernel holds its launch stamp for its whole life, so only a restart does.
#:
#: The keys are pinned copies of the sentences ``_writes_disabled_result`` ends
#: its two launch-pin messages with, in
#: ``packages/osprey-connectors/src/osprey_connectors/control_system/base.py``.
#: A wording change there would stop matching here and leave the script advice
#: in the cell, so the tests assert these two sentences against the connector's
#: own output rather than against this mapping alone.
LAUNCH_PIN_REMEDIES = {
    "Re-run the script to pick it up.": "Restart the kernel to pick it up.",
    "Re-run the script to pick up the current write state.": (
        "Restart the kernel to pick up the current write state."
    ),
}


def _refusal_classes() -> tuple[type[BaseException], ...]:
    """The exception classes the kernel's hook answers for.

    Imported inside the function for the module's one-way dependency rule: a
    process that only reads the binding must not pull the connectors package
    in behind it.
    """
    from osprey.runtime import ControlTargetChangedError
    from osprey_connectors.errors import ChannelLimitsViolationError, ChannelWriteBlockedError

    return (ChannelWriteBlockedError, ChannelLimitsViolationError, ControlTargetChangedError)


def _stamped_session_ended() -> bool:
    """Whether the state record this kernel stamped itself from is gone.

    The pin that raises ``ControlTargetChangedError`` cannot tell a switch from
    an ending: it compares the stamp against what the controls server publishes
    NOW, and a session that ended publishes nothing, so "the record moved" and
    "the record is gone" arrive as the same refusal. The record is what
    separates them, and :func:`osprey.runtime._current_target_record` is the
    resolver that already reads it — the same one the pin itself consults, off
    the ``OSPREY_CONTROL_TARGET_STATE_PID`` stamp
    :func:`compute_stamps` wrote. Asking it a second time here rather than
    resolving the record another way is what keeps the two answers from
    disagreeing.

    An unstamped kernel is not pinned and never reaches this, but the stamp is
    checked anyway: without it, ``None`` would mean "no session was ever
    joined" and be reported as one that ended.

    Returns:
        ``True`` only when this kernel joined a session whose record is no
        longer readable. Every failure answers ``False``, which leaves the
        target-changed line — both lines ask for the same restart, so the
        fail-safe direction is the one that claims less.
    """
    try:
        from osprey import runtime
        from osprey.mcp_server.python_executor import executor

        if not (os.environ.get(executor.ENV_CONTROL_TARGET_STATE_PID) or "").strip():
            return False
        return runtime._current_target_record() is None
    except Exception:  # noqa: BLE001 - a hint is not worth failing a refusal over
        logger.debug("Could not check the bound session's record", exc_info=True)
        return False


def _hint_for(value: BaseException) -> str | None:
    """The one action line for *value*, or ``None`` when there is no action.

    Every branch is read from the stamp this process already carries, so the
    hint and the connector's own refusal text answer from one state rather
    than from a flag the kernel would have to keep in step. A live-store
    refusal and a limits violation get no line: the connector's message
    already names what to do, and a second line would either repeat it or send
    the operator somewhere the message did not.

    The target-changed refusal covers two states and names one, so it forks on
    :func:`_stamped_session_ended` before it is reported as a switch.

    Args:
        value: The refusal that reached the hook.

    Returns:
        The line to print before the traceback, or ``None`` to print none.
    """
    from osprey.mcp_server.python_executor import executor
    from osprey.runtime import ControlTargetChangedError
    from osprey_connectors import session_store
    from osprey_connectors.errors import ChannelWriteBlockedError

    if isinstance(value, ControlTargetChangedError):
        return HINT_SESSION_ENDED if _stamped_session_ended() else HINT_TARGET_CHANGED
    if not isinstance(value, ChannelWriteBlockedError):
        return None
    target = (os.environ.get(executor.ENV_CONTROL_TARGET) or "").strip() or None
    if session_store.launch_permits(target):
        return None
    if session_store.launch_narrowed_target() == session_store.LAUNCH_POSTURE_ALL_TARGETS:
        return HINT_NO_SESSION
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


def _rewrite_launch_pin_remedy(value: BaseException) -> None:
    """Point a launch-pin refusal's own message at a kernel restart.

    The connector writes one message for every surface, and its remedy —
    re-run the script — is the wrong one here: a cell re-run reuses the kernel,
    which still carries the launch stamp that refused. The hint above the
    traceback says so, but the exception under it contradicts the hint, and the
    exception is the line an operator acts on.

    Rewritten by rebuilding ``args`` rather than by subclassing the connector's
    error or patching its message builder: the class stays the connector's, and
    only the surface that displays it is changed. A message that does not end
    in a pinned sentence is left exactly as it is.

    Args:
        value: The refusal, mutated in place when its message matches.
    """
    args = getattr(value, "args", ())
    if not args or not isinstance(args[0], str):
        return
    message = args[0]
    for connector_remedy, kernel_remedy in LAUNCH_PIN_REMEDIES.items():
        if message.endswith(connector_remedy):
            value.args = (message[: -len(connector_remedy)] + kernel_remedy, *args[1:])
            return


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

    The refusal itself is rewritten before it is shown where its remedy names
    the wrong surface — see :func:`_rewrite_launch_pin_remedy`.

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
    # The two launch-pin hints ARE the launch-pin predicate, already evaluated:
    # deriving the rewrite from them rather than re-asking the store is what
    # keeps one refusal from getting a hint and a remedy that disagree.
    if hint in (HINT_NO_SESSION, HINT_WRITES_OFF):
        _rewrite_launch_pin_remedy(value)
    if hint is not None:
        print(hint)
    shell.showtraceback((etype, value, tb), tb_offset=tb_offset)


def install_refusal_handler(shell: Any) -> None:
    """Put the refusal hook on *shell*.

    A cell's write goes to the connector directly, so a refusal surfaces as a
    raised exception and nothing else — no audit record, and a traceback whose
    remedy is a kernel restart the message has no way to ask for. This hook
    supplies both, and leaves every other exception alone.

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
    """Run a notebook kernel joined to the bound terminal session.

    ``ipykernel`` is imported here rather than at module scope so that reading
    the binding costs a process no kernel stack.

    Logging is routed before anything else runs, so that the preparation's and
    the registry's own records go where the rest of this process's records go.

    The environment is prepared before the registry is loaded, because the
    registry is created from the config path the preparation publishes; and
    both happen before the kernel exists, so no cell can run ahead of them.

    The last three statements are three rather than one chained call because
    the hook goes between the second and the third: ``initialize`` is what
    builds the shell, and ``start`` does not return until the kernel stops.

    Args:
        argv: Kernel arguments. ``None`` takes them from the command line, the
            way ``ipykernel``'s own entry point does.
    """
    _route_logs_to_process_stderr()
    _prepare_environment()
    _initialize_registry()

    from ipykernel.kernelapp import IPKernelApp

    app = IPKernelApp.instance()
    app.initialize(argv)
    install_refusal_handler(app.shell)
    app.start()


if __name__ == "__main__":  # pragma: no cover - the kernelspec's entry point
    main()
