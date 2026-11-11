"""A long-lived process's one binding to the deployment's control target.

:mod:`osprey.runtime` serves a script: it is stamped with a target by whoever
launched it, builds a connector on first use, and may rebuild it between
notebook cells. A long-lived server process — a panel backend that streams
channel values for hours — needs the opposite shape, and this module is it:

* The process reads the deployment's control-context record **once**, at
  :meth:`ControlSession.open`, and builds one connector for the target that
  record names. That pairing is the :class:`Binding`.
* The connector is never rebuilt in process. When the binding moves — the
  record names another ``(target, generation)`` — :meth:`ControlSession.moved`
  says so and the process exits with :data:`EXIT_BINDING_MOVED`; its
  supervisor starts a new one, which opens against the new record. A connector
  chooses its gateway at ``connect()``, so re-pointing one in place is not
  something this module could do honestly: a change of the bound target's write
  state that asks for the other gateway (``read_only`` and ``write_access``) is
  a move too. Any other change of write state is not: every write is judged on
  the verdict read at its admission, and :attr:`ControlSession.writes` follows
  the record and the hold in place.
* The same holds for a connector whose transport went stale — the control
  system serves a channel again, but the client library will not notice. The
  connector raises that as a lost connector
  (:class:`~osprey_connectors.errors.ConnectorUnavailableError`, the root a
  pooled child is replaced on); :meth:`ControlSession.wait_stale` returns on
  it and the process exits with :data:`EXIT_TRANSPORT_STALE`, which its
  supervisor treats exactly like a move. The connector counts only channels
  that connected in its own process, so in a successor a channel that never
  comes back cannot drive a respawn loop.
* Nothing here raises for a control system that is absent or misconfigured.
  No readable record is the ``fallback`` state (the deployment's own baseline,
  read-only); a target the config cannot serve, or a connector that will not
  build or connect, is the ``unservable`` state (no connector at all). Both are
  bindings a caller can report, and both still watch the record.

The one thing :meth:`ControlSession.open` does raise for is an environment the
process must not run in: :class:`SessionEnvironmentError`, which a caller turns
into :data:`EXIT_START_REFUSED`.

Writes follow one fixed order (:meth:`ControlSession.write`). The call itself
decides only two things, synchronously: writes stopped, and a write to the
same channel still in flight. Everything else runs in one owned task that is
never cancelled — the in-flight marker, the pin, the binding state, the
execution verdict, the switch gate, the ``attempt`` record, the checked write
and the outcome record — so a caller that stops waiting never cuts a write in
half. Whether this login may write here is the execution verdict
(:func:`~osprey_connectors.posture_store.execution_verdict`): one order shared
with every other write path, the session's hold included, and every refusal
it decides names that verdict's reason in that verdict's words
(:func:`~osprey_connectors.refusals.refusal_phrase`). At the ``asked``
position a person is asked before every write: a write that does not carry
that person's confirmation is refused ``approval_missing``. The session adds
no other reason for "can this login write"; it adds only its own states — no
record, no connector, a binding that moved, a write still in flight, an audit
ledger that is down.

The hold is asked about for the sessions this process writes for: the
environment's session key by default, or the keys a ``session_keys``
provider names at each read — a process that serves a whole terminal is held
while any of its live sessions is.
The task's done-callback releases the marker and the write entry when the
connector call has returned. The caller waits until the write's admission plus
the bound connector's bound ``D_b`` and the margin ``M``; past that it answers
"outcome unknown" if nothing was recorded, and the write, still running, is
logged when it ends.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey.runtime.budget import WRITE_MARGIN_S, WriteBudget, bound_budget, write_budget
from osprey.runtime.session_env import (
    SessionEnvironmentError,
    check_environment,
    scrub_epics_environment,
)
from osprey.runtime.session_outcomes import (
    DECISION_ALLOWED,
    DECISION_REFUSED,
    MESSAGE_AUDIT_UNAVAILABLE,
    MESSAGE_CONFIRMATION_MISSING,
    MESSAGE_NO_RECORD,
    MESSAGE_UNCONFIRMED,
    MESSAGE_UNKNOWN,
    MESSAGE_UNSERVABLE,
    MESSAGE_WRITE_IN_PROGRESS,
    MESSAGE_WRITTEN,
    OUTCOME_CONFIRMED,
    OUTCOME_MISMATCH,
    OUTCOME_REFUSED,
    OUTCOME_SWITCH_IN_PROGRESS,
    OUTCOME_TARGET_CHANGED,
    OUTCOME_UNCONFIRMED,
    OUTCOME_UNKNOWN,
    REASON_APPROVAL_MISSING,
    REASON_AUDIT_UNAVAILABLE,
    REASON_CONTEXT_UNAVAILABLE,
    REASON_LIMITS,
    REASON_RECORD_UNAVAILABLE,
    REASON_SWITCHING,
    REASON_WRITE_ATTEMPT,
    REASON_WRITE_IN_PROGRESS,
    REASON_WRITE_LANDED,
    REASON_WRITE_UNCONFIRMED,
    REFUSAL_OUTCOMES,
    SessionWriteResult,
    detail_tokens,
    failed_result,
    refusal_message,
)
from osprey.runtime.write_steps import PinVerdict, checked_write, pin_verdict
from osprey_connectors.logger import get_logger
from osprey_connectors.safety_file import SHARED_FILE_MODE, atomic_replace, ensure_shared_dir

if TYPE_CHECKING:
    from osprey_connectors.control_system import (
        ChannelMetadata,
        ChannelValue,
        ControlSystemConnector,
    )
    from osprey_connectors.posture_store import WriteVerdict

logger = get_logger("control_session")

__all__ = [
    "EXIT_BINDING_MOVED",
    "EXIT_START_REFUSED",
    "EXIT_TRANSPORT_STALE",
    "MOVE_POLL_S",
    "OUTCOME_CONFIRMED",
    "OUTCOME_MISMATCH",
    "OUTCOME_REFUSED",
    "OUTCOME_SWITCH_IN_PROGRESS",
    "OUTCOME_TARGET_CHANGED",
    "OUTCOME_UNCONFIRMED",
    "OUTCOME_UNKNOWN",
    "SESSION_KEY_ENV_VAR",
    "STATE_BOUND",
    "STATE_FALLBACK",
    "STATE_UNSERVABLE",
    "WARM_BOUND_S",
    "Binding",
    "ControlSession",
    "SessionEnvironmentError",
    "SessionUnservableError",
    "SessionWriteResult",
    "Subscription",
]

#: The process exits with this when its binding moved. A supervisor respawns at
#: once: nothing is wrong, the next process simply opens against the new record.
#: (``EX_TEMPFAIL`` in sysexits terms.)
EXIT_BINDING_MOVED = 75

#: The process exits with this when its connector's transport went stale — the
#: connector raised the lost-connector root and :meth:`ControlSession.wait_stale`
#: returned. A supervisor treats it exactly like :data:`EXIT_BINDING_MOVED` and
#: respawns at once: a new process gets a new client transport. (``EX_PROTOCOL``
#: in sysexits terms.)
EXIT_TRANSPORT_STALE = 76

#: The process exits with this when :meth:`ControlSession.open` refused its
#: environment. A supervisor never respawns it: the same environment would be
#: refused again. (``EX_CONFIG`` in sysexits terms.)
EXIT_START_REFUSED = 78

#: How often :meth:`ControlSession.wait_moved` looks at the record's file
#: signature.
MOVE_POLL_S = 0.25

#: The most :meth:`ControlSession.open` waits on the warm step.
WARM_BOUND_S = 1.0

#: The three states a binding can be in.
STATE_BOUND = "bound"
STATE_FALLBACK = "fallback"
STATE_UNSERVABLE = "unservable"

#: The margin M a write's waiter adds to ``D_b`` (``osprey.runtime.budget``).
#: A module name so a test can shrink it.
_MARGIN_S = WRITE_MARGIN_S

#: The environment variable naming the web terminal's session this process
#: writes for, the key its hold is kept under. The verdict reads the same
#: variable (``osprey_connectors.posture_store``); the session only watches the
#: hold record it names, so a hold raised or answered shows without a restart.
SESSION_KEY_ENV_VAR = "OSPREY_SESSION_ID"


class SessionUnservableError(ConnectionError):
    """A read or metadata call on a session that holds no connector.

    The session is ``unservable`` (or closed): there is no control system
    behind it to ask. Subscriptions do not raise this — they report the channel
    as not connected and deliver nothing.
    """


@dataclass(frozen=True)
class Binding:
    """What a session is bound to, fixed for the life of the process.

    Attributes:
        target: The control target — the record's when one was read, otherwise
            the deployment's baseline.
        generation: The record's generation, or ``None`` when no record was
            read.
        connector_type: The connector type the target resolved to, or ``None``
            when it did not resolve.
        display_name: What a person calls the target.
        state: ``bound``, ``fallback`` or ``unservable``.
    """

    target: str
    generation: int | None
    connector_type: str | None
    display_name: str
    state: str

    @property
    def id(self) -> str:
        """``<target>@<generation>`` when bound, otherwise the state's name.

        A client quotes this back with a write so that a write composed against
        one binding is never executed by a process holding another. The two
        degraded states have no generation to quote, and no write to execute.
        """
        if self.state == STATE_BOUND:
            return f"{self.target}@{self.generation}"
        return self.state


@dataclass
class _Write:
    """One admitted write, from admission until its task's done-callback.

    Attributes:
        task: The owned task running the write; never cancelled by this module.
        admitted_at: ``time.monotonic()`` at admission.
        recorded: Whether the write's outcome record was filed — by the task,
            or by the waiter as ``unknown`` once the deadline passed.
        marker: The in-flight marker, once written.
        result: The outcome, once the task decided it.
        verdict: The execution verdict at admission — the read this write was
            judged under, kept so a refusal that follows names its kind.
        confirmed: Whether a person confirmed the write, as the caller said.
    """

    task: asyncio.Task[SessionWriteResult] | None
    admitted_at: float
    recorded: bool = False
    marker: Path | None = None
    result: SessionWriteResult | None = field(default=None, repr=False)
    confirmed: bool = False
    verdict: WriteVerdict | None = None


@dataclass(frozen=True)
class Subscription:
    """The handle :meth:`ControlSession.subscribe` returns. Opaque to callers."""

    address: str
    token: int


def _control_system_section() -> dict[str, Any]:
    """The deployment's ``control_system:`` section, or an empty one."""
    from osprey_connectors.config import get_config_value

    try:
        section = get_config_value("control_system", {})
    except Exception:
        logger.warning("The control_system config could not be read", exc_info=True)
        return {}
    return section if isinstance(section, dict) else {}


def _display_name(target: str) -> str:
    """What a person calls *target*, as every other surface of the deployment names it.

    Rendered by ``target_display_metadata`` over the full config in the file
    ``OSPREY_CONFIG`` names (the environment check has already required it), so
    a panel and the web terminal cannot call one machine two things. Anything
    that goes wrong on the way — an unreadable file, the describer itself — is
    the target's own name.
    """
    try:
        import yaml

        from osprey.mcp_server.control_system.connector_host_manager import (
            target_display_metadata,
        )

        path = Path(os.environ["OSPREY_CONFIG"])
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        name = target_display_metadata(config)[target]["display_name"]
    except Exception:
        logger.warning("No display name for control target %r", target, exc_info=True)
        return str(target)
    return name if isinstance(name, str) and name else str(target)


def _read_record() -> Any:
    """The deployment's control context now, or ``None`` for every way of not having one."""
    from osprey_connectors import control_context

    try:
        return control_context.read_record()
    except Exception:
        logger.debug("Control context unreadable", exc_info=True)
        return None


def _record_key(record: Any) -> tuple[str, int] | None:
    return None if record is None else (record.target, record.generation)


def _fallback_target(section: dict[str, Any]) -> str:
    """The target a session serves when there is no record to name one.

    The deployment's own baseline, except that a session never picks the
    stand-in by its own choice: a stand-in baseline becomes the virtual
    accelerator when the deployment configures one, and the live machine
    otherwise.
    """
    from osprey_connectors import types

    target = types.baseline_target(section)
    if target != types.TARGET_STANDIN:
        return target
    connectors = section.get("connector")
    if isinstance(connectors, dict) and types.VIRTUAL_ACCELERATOR in connectors:
        return types.TARGET_VA
    return types.TARGET_LIVE


#: What names the sessions a process writes for: a zero-argument callable read
#: at each verdict, or ``None`` for the environment's session key.
SessionKeys = Callable[[], Sequence[str]] | None


def _keys_now(session_keys: SessionKeys) -> list[str] | None:
    """The keys *session_keys* names now; ``None`` for the environment's.

    Raises whatever the provider raises: a caller that cannot tell which
    sessions it writes for must not answer as if it wrote for none.
    """
    if session_keys is None:
        return None
    return [str(key) for key in session_keys() if key]


def _verdict(target: str, session_keys: SessionKeys = None) -> WriteVerdict:
    """The execution verdict on a write by this login to *target* now.

    :func:`~osprey_connectors.posture_store.execution_verdict`: the write
    verdict's terms plus the hold of the sessions *session_keys* names (the
    environment's when ``None``). A config or a key list that cannot be read
    refuses as ``record_unavailable``. Never raises.
    """
    from osprey_connectors.config import get_config_value
    from osprey_connectors.posture_store import WriteVerdict, execution_verdict

    try:
        section = get_config_value("control_system", {})
        keys = _keys_now(session_keys)
    except Exception:
        logger.warning(
            "The control_system config or the session keys could not be read", exc_info=True
        )
        return WriteVerdict(allowed=False, reason="record_unavailable", target=target, kind=None)
    return execution_verdict(section, target, session_keys=keys)


def _record_signature() -> Any:
    """The record file's signature now; stats the file and reads nothing."""
    from osprey_connectors import control_context

    return control_context.file_signature(control_context.record_path())


def _hold_signature(session_keys: SessionKeys = None) -> Any:
    """The hold records' signatures now, or ``None`` when this process has no hold store.

    One record per session, ``<hold root>/<session key>/record.json``
    (:mod:`osprey_connectors.session_hold`); a flag raised or answered replaces
    it whole, so its signature moves exactly when the hold does. The keys are
    *session_keys*' (the environment's when ``None``), so a session that comes
    or goes moves the signature too. Stats the files and reads nothing else.
    """
    from osprey_connectors import control_context, session_hold

    root = session_hold.hold_root()
    if root is None:
        return None
    try:
        keys = _keys_now(session_keys)
    except Exception:
        # Unreadable now: a signature of its own, so a later read moves it.
        return ("unreadable",)
    if keys is None:
        keys = [os.environ.get(SESSION_KEY_ENV_VAR, "")]
    return tuple(
        (key, control_context.file_signature(root / key / session_hold.RECORD_NAME))
        for key in sorted(set(keys))
        if key and session_hold.is_safe_session_key(key)
    )


def _bounded_config(
    section: dict[str, Any], connector_type: str, budget: WriteBudget
) -> tuple[dict[str, Any], float, float]:
    """The section a session's connector is built from, and its ``(T_b, D_b)``.

    A copy of *section* with ``type`` set to *connector_type* and that type's
    block copied with ``timeout_s`` = T_b and the step-read ceiling = S_b; the
    deployment's own mapping is never edited.

    Raises:
        ValueError: The block's call bound cannot be read; its connector would
            refuse to connect alike.
    """
    from osprey_connectors.control_system import STEP_READ_TIMEOUT_KEY

    blocks = section.get("connector")
    blocks = dict(blocks) if isinstance(blocks, Mapping) else {}
    block = blocks.get(connector_type)
    block = dict(block) if isinstance(block, Mapping) else {}
    call_bound_s, step_bound_s, write_bound_s = bound_budget(block, budget, connector_type)
    block["timeout_s"] = call_bound_s
    block[STEP_READ_TIMEOUT_KEY] = step_bound_s
    blocks[connector_type] = block
    config = dict(section)
    config["type"] = connector_type
    config["connector"] = blocks
    return config, call_bound_s, write_bound_s


async def _warm(connector: Any) -> None:
    """Start the connector's client context, bounded and never fatal.

    The first Channel Access call a process makes loads libca and creates its
    context, which costs far more than any call after it. Paying that here
    keeps it off the first subscriber. No channel is read, and no IOC has to be
    there: with a dead control system the session still opens and every channel
    simply reports not connected. A connector without a ``warm()`` has nothing
    to start.
    """
    try:
        hook = getattr(connector, "warm", None)
        if callable(hook):
            result = hook()
            if asyncio.iscoroutine(result):
                await asyncio.wait_for(result, WARM_BOUND_S)
    except TimeoutError:
        logger.warning("Control session warm step did not finish within %.1f s", WARM_BOUND_S)
    except Exception:
        logger.warning("Control session warm step failed", exc_info=True)


class ControlSession:
    """One process's connector, bound to one ``(target, generation)`` for life.

    Build one with :meth:`open`. There is no per-call target, no identity
    argument and no way to rebuild: a session whose binding moved is finished,
    and its process exits.
    """

    def __init__(
        self,
        *,
        surface: str,
        subject: str | None = None,
        client_label: str | None = None,
        binding: Binding,
        connector: ControlSystemConnector | None,
        record_key: tuple[str, int] | None,
        verdict: WriteVerdict,
        session_keys: SessionKeys = None,
        record_signature: Any = None,
        hold_signature: Any = None,
        call_bound_s: float | None = None,
        write_bound_s: float | None = None,
    ) -> None:
        self._surface = surface
        self._subject = subject or f"{surface}_write"
        self._client_label = client_label or f"a {surface} write"
        self._binding = binding
        self._connector = connector
        self._record_key = record_key
        #: The sessions whose hold this process's writes are held with; ``None``
        #: is the environment's session key.
        self._session_keys = session_keys
        #: The execution verdict for this session's own target, as last read: at
        #: open, at every write's admission, and whenever :meth:`wait_moved`
        #: sees the record or the session's hold change. ``writes`` reads only
        #: this.
        self._verdict: WriteVerdict = verdict
        #: The record's and the hold record's file signatures that verdict was read under.
        self._seen_signature = record_signature
        self._seen_hold_signature = hold_signature
        #: Whether :meth:`wait_moved` saw the record name another
        #: ``(target, generation)``, or failed to watch it.
        self._target_moving = False
        if call_bound_s is None or write_bound_s is None:
            default = write_budget({})
            call_bound_s, write_bound_s = default.T, default.D
        #: T_b: the bound connector's call bound; a caller timeout is clamped to it.
        self._call_bound_s = call_bound_s
        #: D_b: the bound connector's whole write bound (S_b + 2 T_b).
        self._write_bound_s = write_bound_s
        self._subscriptions: dict[int, str | None] = {}
        self._next_token = 0
        self._writes_stopped = False
        self._writes: dict[str, _Write] = {}
        self._latest_admission: float | None = None
        self._close_task: asyncio.Task[None] | None = None

    # -- opening ------------------------------------------------------------

    @classmethod
    async def open(
        cls,
        *,
        surface: str,
        subject: str | None = None,
        client_label: str | None = None,
        budget: WriteBudget | None = None,
        session_keys: SessionKeys = None,
    ) -> ControlSession:
        """Bind this process to the deployment's control target.

        Checks the environment, marks the process as a long-lived session,
        drops inherited ``EPICS_*`` settings, reads the record once, builds and
        connects the connector for its target, and warms it.

        The connector is built from a copy of its block with ``timeout_s`` =
        T_b and the step-read ceiling = S_b (``osprey.runtime.budget``), so
        one write's connector call stays within ``D_b = S_b + 2 T_b`` without
        any connector growing a parameter.

        Args:
            surface: The kind of client this session serves, as the audit and
                the in-flight markers name it.
            subject: The audit ``subject`` of every write record;
                ``<surface>_write`` when omitted.
            client_label: How a refused target switch names this session's
                in-flight write, a noun phrase that fits after "belongs to";
                ``a <surface> write`` when omitted.
            budget: The process's write budget; computed from the deployment's
                ``control_system`` section when omitted.
            session_keys: What names the sessions this process writes for,
                read at every verdict: a process serving a whole terminal is
                held while any of them is. Omitted, the environment's session
                key (``OSPREY_SESSION_ID``).

        Raises:
            SessionEnvironmentError: The environment is one a session must not
                run in. Nothing else raises: a missing record is the
                ``fallback`` state and an unbuildable connector the
                ``unservable`` one.
        """
        from osprey_connectors import types
        from osprey_connectors.control_system import mark_long_lived_session
        from osprey_connectors.factory import ConnectorFactory, register_builtin_connectors

        check_environment(os.environ)
        mark_long_lived_session()
        scrub_epics_environment()
        register_builtin_connectors()

        section = _control_system_section()
        if budget is None:
            budget = write_budget({"control_system": section})
        call_bound_s, _step_bound_s, write_bound_s = bound_budget(None, budget)
        record_signature = _record_signature()
        record = _read_record()
        record_key = _record_key(record)
        if record is not None:
            target, generation, state = record.target, record.generation, STATE_BOUND
        else:
            target, generation, state = _fallback_target(section), None, STATE_FALLBACK

        connector: ControlSystemConnector | None = None
        connector_type: str | None = None
        try:
            connector_type = types.resolve_target(section, target)
            config, call_bound_s, write_bound_s = _bounded_config(section, connector_type, budget)
            connector = await ConnectorFactory.create_control_system_connector(
                config=config, control_target=target
            )
        except Exception as exc:
            logger.warning(
                "Control target %r cannot be served here (%s: %s); every channel "
                "reports not connected",
                target,
                type(exc).__name__,
                exc,
            )
            connector = None
            state = STATE_UNSERVABLE

        binding = Binding(
            target=target,
            generation=generation,
            connector_type=connector_type,
            display_name=_display_name(target),
            state=state,
        )
        hold_signature = _hold_signature(session_keys)
        execution = _verdict(target, session_keys)
        session = cls(
            surface=surface,
            subject=subject,
            client_label=client_label,
            binding=binding,
            connector=connector,
            record_key=record_key,
            verdict=execution,
            session_keys=session_keys,
            record_signature=record_signature,
            hold_signature=hold_signature,
            call_bound_s=call_bound_s,
            write_bound_s=write_bound_s,
        )
        if connector is not None:
            await _warm(connector)
        logger.info("Control session opened for %s: %s", surface, binding.id)
        return session

    @property
    def binding(self) -> Binding:
        """What this session is bound to. Never changes."""
        return self._binding

    @property
    def surface(self) -> str:
        """The kind of client this session serves."""
        return self._surface

    @property
    def connector(self) -> ControlSystemConnector | None:
        """The connector this session reads and writes through; ``None`` for a fallback."""
        return self._connector

    @property
    def verdict(self) -> WriteVerdict:
        """The execution verdict for this session's target, as last read. Reads no file."""
        return self._verdict

    @property
    def writes(self) -> dict[str, Any]:
        """What this session's writes are, on the execution verdict's own terms. Reads no file.

        :func:`~osprey_connectors.control_context.write_statement` of the cached
        verdict: ``{allowed, position, reason, kind}``. ``position`` says whether
        a person is asked before a write (``asked``) or not (``allowed``);
        ``reason`` names the refusal when there is one, ``session_held`` for a
        hold on this session. A reader words it with
        :func:`~osprey_connectors.refusals.refusal_phrase`. The session's own
        states — :attr:`binding` (``fallback``, ``unservable``) and
        :attr:`target_moving` — are read beside it, not folded into it.
        """
        from osprey_connectors.control_context import write_statement

        return write_statement(self._verdict)

    @property
    def target_moving(self) -> bool:
        """Whether :meth:`wait_moved` saw the record name another binding, or failed."""
        return self._target_moving

    # -- reading ------------------------------------------------------------

    def _require_connector(self) -> ControlSystemConnector:
        if self._connector is None:
            raise SessionUnservableError(
                f"No control system is reachable from this session ({self._binding.id})"
            )
        return self._connector

    async def read(self, address: str) -> ChannelValue:
        """Read one channel once.

        Raises:
            SessionUnservableError: The session holds no connector.
        """
        return await self._require_connector().read_channel(address)

    async def metadata(self, address: str) -> ChannelMetadata:
        """The channel's static metadata.

        Raises:
            SessionUnservableError: The session holds no connector.
        """
        return await self._require_connector().get_metadata(address)

    async def subscribe(
        self,
        address: str,
        on_value: Callable[[ChannelValue], None],
        *,
        on_connection: Callable[[bool], None] | None = None,
    ) -> Subscription:
        """Deliver every value of *address* to *on_value* until unsubscribed.

        ``on_connection`` is told whether the channel is connected, and only
        when that changes: ``False`` before this returns, ``True`` when the
        connector reports the channel up and before every value that follows a
        ``False``, and ``False`` again when the connector reports a drop. A
        channel that never connects therefore reports ``False`` once and
        nothing further, and a session with no connector reports every channel
        that way. Each subscription keeps its own state.
        """
        self._next_token += 1
        handle = Subscription(address=address, token=self._next_token)
        reported: bool | None = None

        def report(up: bool) -> None:
            nonlocal reported
            if up == reported:
                return
            reported = up
            if on_connection is not None:
                on_connection(up)

        def deliver(value: ChannelValue) -> None:
            report(True)
            on_value(value)

        # The state exists before the connector is asked: a connector may
        # report the channel up and deliver its first value inside subscribe().
        report(False)
        if self._connector is None:
            self._subscriptions[handle.token] = None
            return handle

        self._subscriptions[handle.token] = await self._connector.subscribe(
            address, deliver, on_connection=report
        )
        return handle

    async def unsubscribe(self, handle: Subscription) -> None:
        """Stop a subscription. Unknown and already-stopped handles are ignored."""
        subscription_id = self._subscriptions.pop(handle.token, None)
        if subscription_id is not None and self._connector is not None:
            await self._connector.unsubscribe(subscription_id)

    # -- writing ------------------------------------------------------------

    async def write(
        self,
        address: str,
        value: Any,
        *,
        origin: Any = None,
        timeout: float | None = None,
        confirmed: bool = False,
    ) -> SessionWriteResult:
        """Write *value* to *address* in the session's fixed order. Never raises for an outcome.

        In this call, synchronously: writes stopped → ``target_changed``; a
        write to the same channel (:meth:`_channel_key`) still admitted → ``refused`` /
        ``write_in_progress`` at once (no queue). Otherwise the write is
        admitted and one owned task runs the rest (:meth:`_run_write`). This
        call waits for it until admission + ``D_b`` + M; past that, if nothing
        was recorded, it files and answers ``unknown`` while the task
        runs on, never cancelled.

        Args:
            address: The channel, as the client named it.
            value: The value to write.
            origin: The client's ``{screen, rev}`` (a mapping or an object with
                those attributes), for the audit only.
            timeout: A caller's call bound, clamped to the bound connector's
                ``T_b``.
            confirmed: Whether a person confirmed this write. At the ``asked``
                position a write without it is refused ``approval_missing``;
                at ``allowed`` it is not read.
        """
        if self._writes_stopped:
            return self._refuse_now(
                address,
                value,
                origin,
                OUTCOME_TARGET_CHANGED,
                REASON_SWITCHING,
                self._phrase(),
                confirmed=confirmed,
            )
        key = self._channel_key(address)
        if key in self._writes:
            return self._refuse_now(
                address,
                value,
                origin,
                OUTCOME_REFUSED,
                REASON_WRITE_IN_PROGRESS,
                MESSAGE_WRITE_IN_PROGRESS,
                confirmed=confirmed,
            )

        entry = _Write(task=None, admitted_at=time.monotonic(), confirmed=confirmed)
        self._writes[key] = entry
        self._latest_admission = entry.admitted_at
        call_bound = self._call_bound_s
        if timeout is not None and 0 < timeout < call_bound:
            call_bound = timeout
        task = asyncio.get_running_loop().create_task(
            self._run_write(entry, address, value, origin, call_bound),
            name=f"control-session-write:{key}",
        )
        entry.task = task
        task.add_done_callback(lambda done: self._release(key, entry, done))

        deadline = entry.admitted_at + self._write_bound_s + _MARGIN_S
        await asyncio.wait({task}, timeout=max(0.0, deadline - time.monotonic()))
        if task.done() and not task.cancelled():
            return task.result()
        if entry.recorded and entry.result is not None:
            return entry.result
        # Outcome unknown: the write is still running past its bound. It is
        # not cancelled; this is the one record of it, and its completion is
        # logged, not re-audited.
        entry.recorded = True
        result = SessionWriteResult(
            outcome=OUTCOME_UNKNOWN, reason=REASON_WRITE_UNCONFIRMED, message=MESSAGE_UNKNOWN
        )
        logger.warning(
            "Write to %s still running %.1f s after admission; answered as unknown",
            address,
            time.monotonic() - entry.admitted_at,
        )
        self._audit(
            DECISION_ALLOWED, REASON_WRITE_UNCONFIRMED, address, value, origin, confirmed=confirmed
        )
        return result

    def refuse(
        self,
        address: str,
        value: Any,
        reason: str,
        *,
        origin: Any = None,
        confirmed: bool = False,
    ) -> SessionWriteResult:
        """Audit and answer a refusal the caller decided on. No connector call is made.

        *reason* is a member of :data:`~osprey_connectors.refusals.REASONS` and
        is answered in its phrase, for this session's kind of machine; a caller
        has no words of its own. ``switching`` — the caller holds a stale
        binding id — answers ``target_changed``, every other reason ``refused``.

        Raises:
            ValueError: *reason* is not a refusal reason.
        """
        from osprey_connectors.refusals import REASONS

        if reason not in REASONS:
            raise ValueError(f"not a refusal reason: {reason!r}")
        outcome = OUTCOME_TARGET_CHANGED if reason == REASON_SWITCHING else OUTCOME_REFUSED
        return self._refuse_now(
            address, value, origin, outcome, reason, self._phrase(reason), confirmed=confirmed
        )

    def stop_writes(self) -> None:
        """Admit no further write. Irreversible for this session."""
        self._writes_stopped = True

    async def drain(self, timeout: float) -> bool:
        """Wait for admitted writes to end. ``True`` when none is left.

        Waits until the earlier of now + *timeout* and the latest admission +
        ``D_b`` + M: no admitted write can be waited on for longer than its own
        bound. Nothing is cancelled; a write still running at the end is left
        running and ``False`` is returned.
        """
        end = time.monotonic() + max(0.0, timeout)
        while self._writes:
            latest = self._latest_admission
            bound = end if latest is None else min(end, latest + self._write_bound_s + _MARGIN_S)
            remaining = bound - time.monotonic()
            if remaining <= 0:
                break
            tasks = {entry.task for entry in self._writes.values() if entry.task is not None}
            if not tasks:
                break
            await asyncio.wait(tasks, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
        return not self._writes

    # -- the write task -------------------------------------------------------

    async def _run_write(
        self, entry: _Write, address: str, value: Any, origin: Any, call_bound: float
    ) -> SessionWriteResult:
        """Everything after admission. Synchronous up to the ``attempt`` record."""
        try:
            result = self._admit(entry, address, value, origin)
        except Exception:  # a defect before the attempt record: nothing was written
            logger.exception("Write to %s failed before its attempt record", address)
            result = SessionWriteResult(
                outcome=OUTCOME_REFUSED,
                reason=REASON_RECORD_UNAVAILABLE,
                message=self._phrase(REASON_RECORD_UNAVAILABLE),
            )
            self._settle(entry, address, value, origin, result)
            return result
        if result is not None:
            return result
        try:
            result = await self._connector_write(address, value, call_bound, verdict=entry.verdict)
            self._settle(entry, address, value, origin, result)
            return result
        except Exception:  # never let a defect here leave a write unanswered
            logger.exception("Write to %s failed inside the session", address)
            result = SessionWriteResult(
                outcome=OUTCOME_UNKNOWN, reason=REASON_WRITE_UNCONFIRMED, message=MESSAGE_UNKNOWN
            )
            self._settle(entry, address, value, origin, result)
            return result

    def _admit(
        self, entry: _Write, address: str, value: Any, origin: Any
    ) -> SessionWriteResult | None:
        """The session's own checks, marker first, ending with the ``attempt`` record.

        Returns the refusal that ended the write (already recorded), or ``None``
        when the ``attempt`` record stands and the connector is to be called.
        """
        from osprey_connectors import control_context

        def refused(outcome: str, reason: str, message: str) -> SessionWriteResult:
            result = SessionWriteResult(outcome=outcome, reason=reason, message=message)
            self._settle(entry, address, value, origin, result)
            return result

        try:
            entry.marker = self._write_marker(address)
        except Exception:
            logger.warning("Could not write the in-flight marker for %s", address, exc_info=True)
            return refused(
                OUTCOME_REFUSED, REASON_RECORD_UNAVAILABLE, self._phrase(REASON_RECORD_UNAVAILABLE)
            )

        record = _read_record()
        pin = pin_verdict(record, self._record_key)
        if pin is PinVerdict.MOVED:
            return refused(OUTCOME_TARGET_CHANGED, REASON_SWITCHING, self._phrase())
        if pin is PinVerdict.UNAVAILABLE or self._binding.state == STATE_FALLBACK:
            return refused(OUTCOME_REFUSED, REASON_CONTEXT_UNAVAILABLE, MESSAGE_NO_RECORD)
        if self._binding.state == STATE_UNSERVABLE or self._connector is None:
            return refused(OUTCOME_REFUSED, REASON_CONTEXT_UNAVAILABLE, MESSAGE_UNSERVABLE)

        # The execution verdict, read once for this write: the switch's terms
        # and the sessions' hold, in the order every write path shares. A
        # refusal is answered in the verdict's own words. At ``asked`` a person
        # is asked before every write, so one that carries no confirmation is
        # refused, whatever position the caller last saw.
        execution = _verdict(self._binding.target, self._session_keys)
        self._verdict = entry.verdict = execution
        if not execution.allowed:
            return refused(OUTCOME_REFUSED, str(execution.reason), refusal_message(execution))
        if execution.position == "asked" and not entry.confirmed:
            return refused(OUTCOME_REFUSED, REASON_APPROVAL_MISSING, MESSAGE_CONFIRMATION_MISSING)

        try:
            blocking = control_context.blocking_pids(record, control_context.live_reports(), None)
        except Exception:
            logger.warning("The switch gate could not be read for %s", address, exc_info=True)
            return refused(
                OUTCOME_REFUSED, REASON_RECORD_UNAVAILABLE, self._phrase(REASON_RECORD_UNAVAILABLE)
            )
        if blocking:
            return refused(OUTCOME_SWITCH_IN_PROGRESS, REASON_SWITCHING, self._phrase())

        if not self._audit(
            DECISION_ALLOWED,
            REASON_WRITE_ATTEMPT,
            address,
            value,
            origin,
            confirmed=entry.confirmed,
        ):
            return refused(OUTCOME_REFUSED, REASON_AUDIT_UNAVAILABLE, MESSAGE_AUDIT_UNAVAILABLE)
        return None

    def _phrase(self, reason: str = REASON_SWITCHING) -> str:
        """The phrase for *reason*, naming this session's kind of machine."""
        from osprey_connectors.refusals import refusal_phrase

        return refusal_phrase(reason, self._verdict.kind)

    async def _connector_write(
        self,
        address: str,
        value: Any,
        call_bound: float,
        *,
        verdict: WriteVerdict | None = None,
    ) -> SessionWriteResult:
        """The checked write on the bound connector, mapped to the write contract's outcomes."""
        from osprey_connectors.errors import (
            ChannelLimitsViolationError,
            ChannelWriteBlockedError,
            ChannelWriteFailedError,
        )

        connector = self._connector
        if connector is None:  # closed between admission and here
            return SessionWriteResult(
                outcome=OUTCOME_REFUSED,
                reason=REASON_CONTEXT_UNAVAILABLE,
                message=MESSAGE_UNSERVABLE,
            )
        try:
            result = await checked_write(
                connector,
                address,
                value,
                net=None,
                on_attempt=None,
                timeout=call_bound,
            )
        except ChannelWriteBlockedError as exc:
            # The connector refused: its reason is a RefusalReason member and
            # its text the operator's words, trailer included. A limits refusal
            # chains the violation as its cause, whose sentence is the one a
            # person reads, without the wrapper. The monitor's own posture
            # refusal means the verdict moved between admission and the door;
            # it is logged beside the verdict the write was admitted under.
            if exc.reason == REASON_LIMITS:
                violation = exc.__cause__
                text = (
                    violation.violation_reason or ""
                    if isinstance(violation, ChannelLimitsViolationError)
                    else str(exc)
                )
                return SessionWriteResult(
                    outcome=OUTCOME_REFUSED, reason=REASON_LIMITS, message=text.strip()
                )
            if exc.reason == "WRITES_DISABLED":
                logger.info(
                    "Write to %s refused by the reference monitor (admitted under %s): %s",
                    address,
                    verdict,
                    exc,
                )
            return SessionWriteResult(
                outcome=OUTCOME_REFUSED,
                reason=str(exc.reason or "VALIDATION_ERROR"),
                message=str(exc).strip(),
            )
        except ChannelWriteFailedError as exc:
            return failed_result(exc)
        except Exception:
            logger.warning("Write to %s raised in the connector", address, exc_info=True)
            return SessionWriteResult(
                outcome=OUTCOME_UNKNOWN, reason=REASON_WRITE_UNCONFIRMED, message=MESSAGE_UNKNOWN
            )
        if result.outcome == "confirmed":
            return SessionWriteResult(
                outcome=OUTCOME_CONFIRMED,
                reason=REASON_WRITE_LANDED,
                message=MESSAGE_WRITTEN,
                observed_value=result.observed_value,
            )
        # ``unrequested``: the connector's confirm policy asked for no re-read.
        return SessionWriteResult(
            outcome=OUTCOME_UNCONFIRMED,
            reason=REASON_WRITE_UNCONFIRMED,
            message=MESSAGE_UNCONFIRMED,
            observed_value=result.observed_value,
        )

    def _settle(
        self,
        entry: _Write,
        address: str,
        value: Any,
        origin: Any,
        result: SessionWriteResult,
    ) -> None:
        """File the write's one outcome record, unless the waiter already filed ``unknown``."""
        entry.result = result
        if entry.recorded:
            logger.warning(
                "Write to %s ended after its deadline was answered: %s (%s); not re-audited",
                address,
                result.outcome,
                result.reason,
            )
            return
        entry.recorded = True
        decision = DECISION_REFUSED if result.outcome in REFUSAL_OUTCOMES else DECISION_ALLOWED
        self._audit(decision, result.reason, address, value, origin, confirmed=entry.confirmed)

    def _channel_key(self, address: str) -> str:
        """The key one channel's writes are admitted under: the connector's own.

        Whitespace stripped when the session holds no connector; such a write
        is still admitted, then refused in :meth:`_run_write`. Used only as
        this key: the write itself goes to the address as given.
        """
        if self._connector is None:
            return address.strip()
        return self._connector.channel_key(address)

    def _release(self, key: str, entry: _Write, task: asyncio.Task[Any]) -> None:
        """The task's done-callback: drop the entry and, unless cancelled, the marker."""
        if self._writes.get(key) is entry:
            del self._writes[key]
        if task.cancelled():
            # Process teardown. The pid-named marker stays and is swept once
            # this process is gone: the connector call may still be running.
            logger.warning("Write task for %s was cancelled; its marker stays", key)
            return
        if entry.marker is not None:
            try:
                entry.marker.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove the in-flight marker %s", entry.marker)

    def _refuse_now(
        self,
        address: str,
        value: Any,
        origin: Any,
        outcome: str,
        reason: str,
        message: str,
        *,
        confirmed: bool = False,
    ) -> SessionWriteResult:
        """A refusal decided without admitting the write: one record, no marker."""
        self._audit(DECISION_REFUSED, reason, address, value, origin, confirmed=confirmed)
        return SessionWriteResult(outcome=outcome, reason=reason, message=message)

    def _write_marker(self, address: str) -> Path:
        """Write this write's in-flight marker; the switch gate waits for it."""
        from osprey.mcp_server.control_system import target_eligibility, target_state

        directory = target_state.state_dir()
        ensure_shared_dir(directory)
        path = directory / (
            f"{target_state.INFLIGHT_FILE_PREFIX}{os.getpid()}_{uuid.uuid4().hex}"
            f"{target_state.INFLIGHT_FILE_SUFFIX}"
        )
        marker = {
            "pid": os.getpid(),
            "session": self._audit_session(),
            "surface": self._surface,
            target_eligibility.MARKER_CLIENT_FIELD: self._client_label,
            "target": self._binding.target,
            "channel": address.strip(),
            "started_at": datetime.now().astimezone().isoformat(),
        }
        # A reader must never meet half a marker, so it is replaced whole
        # through atomic_replace, as the notebook kernel's cell marker is.
        atomic_replace(
            path, json.dumps(marker).encode("utf-8"), mode=SHARED_FILE_MODE, sync="best_effort"
        )
        return path

    # -- audit ----------------------------------------------------------------

    def _audit_keys(self) -> list[str]:
        """The session keys a write is judged under, for the audit. Never raises."""
        try:
            keys = _keys_now(self._session_keys)
        except Exception:
            keys = []
        if keys is None:
            key = os.environ.get(SESSION_KEY_ENV_VAR, "")
            keys = [key] if key else []
        return sorted(set(keys))

    def _audit_session(self) -> str:
        return f"{self._surface}:{os.getpid()}"

    def _audit(
        self,
        decision: str,
        reason: str,
        address: str,
        value: Any,
        origin: Any,
        *,
        confirmed: bool = False,
    ) -> bool:
        """File one audit record. ``True`` when it was stored. Never raises.

        The detail names the position and the sessions the write was judged
        under (the latest verdict read) and whether a person confirmed it.
        """
        try:
            from osprey.audit import posture
            from osprey.audit.writer import record

            stored = record(
                surface=self._surface,
                posture=posture.posture(),
                posture_source=posture.posture_source(),
                session=self._audit_session(),
                subject=self._subject,
                decision=decision,
                reason=reason,
                detail=detail_tokens(
                    address,
                    value,
                    origin,
                    position=self._verdict.position,
                    confirmed=confirmed,
                    sessions=self._audit_keys(),
                ),
            )
        except Exception:
            logger.warning("Could not record a %s write for audit", self._surface, exc_info=True)
            return False
        return stored is not None

    # -- the binding moving ---------------------------------------------------

    def moved(self) -> bool:
        """Whether the binding this session was opened on still stands.

        A move is a record that now names another ``(target, generation)`` —
        including one that appeared where there was none, or went away — or a
        switch that now asks for the gateway the connector did not select
        (:meth:`_gateway_lags`). Any other switch change is not a move, and
        neither is a hold: each is judged per write and shown in :attr:`writes`.

        The record comparison is :func:`~osprey.runtime.write_steps.pin_verdict`,
        the same one a write's pin uses. Its ``unavailable`` is a move only for a
        session bound to a record: one opened without a record (a fallback) has
        not moved while there is still none to read.
        """
        self._verdict = _verdict(self._binding.target, self._session_keys)
        return self._record_moved() or self._gateway_lags()

    def _gateway_lags(self) -> bool:
        """Whether the switch now asks for a gateway this session's connector did not select.

        The connector chose at ``connect()`` and keeps its choice, refusing an
        allowed write ``switching`` while it sits on the read gateway. Only a
        block with both gateways chooses by the switch; one with a write gateway
        alone stays on it, and its writes are refused in software. The choice
        follows :func:`~osprey_connectors.posture_store.write_verdict`, which has
        no hold term, so a hold never asks for another gateway. There is no
        public accessor for the choice, so this reads the attributes the
        connector's ``_role_refusal`` reads, and any other connector never lags.
        """
        connector = self._connector
        selected = getattr(connector, "_selected_role", None)
        if selected is None or not getattr(connector, "_write_gateway_configured", False):
            return False
        from osprey_connectors.posture_store import write_verdict

        section = _control_system_section()
        blocks = section.get("connector")
        block = blocks.get(self._binding.connector_type) if isinstance(blocks, Mapping) else None
        gateways = block.get("gateways") if isinstance(block, Mapping) else None
        if not isinstance(gateways, Mapping) or not gateways.get("read_only"):
            return False
        allowed = write_verdict(section, self._binding.target).allowed
        return bool(allowed != (selected == "write_access"))

    def _record_moved(self) -> bool:
        """Whether the record now names another ``(target, generation)``, as :meth:`moved` reads it."""
        verdict = pin_verdict(_read_record(), self._record_key)
        if verdict is PinVerdict.MOVED:
            return True
        return verdict is PinVerdict.UNAVAILABLE and self._record_key is not None

    async def wait_moved(self) -> None:
        """Return once the binding has moved.

        Polls the record's file signature, and the session's hold record's,
        every :data:`MOVE_POLL_S`, starting from the ones the cached verdict was
        read under. Unchanged signatures read nothing. A changed record reads
        the record and the verdict once; that one read refreshes the cached
        verdict, records whether the record now names another ``(target,
        generation)``, and decides the move as :meth:`moved` would. So when this
        returns, :attr:`writes` and :attr:`target_moving` already say what the
        move is, and a switch change that is no move (:meth:`moved`) is in
        :attr:`writes` without returning. A changed hold record refreshes the cached verdict
        only, so :attr:`writes` says ``session_held`` while the hold stands and
        nothing once it is answered; a hold is never a move.

        Never returns otherwise; a caller that is shutting down cancels it. A
        watch that fails marks the target moving before the error propagates.
        """
        try:
            while True:
                now = _record_signature()
                if now != self._seen_signature:
                    self._seen_signature = now
                    self._seen_hold_signature = _hold_signature(self._session_keys)
                    if self._refresh():
                        return
                else:
                    hold = _hold_signature(self._session_keys)
                    if hold != self._seen_hold_signature:
                        self._seen_hold_signature = hold
                        self._verdict = _verdict(self._binding.target, self._session_keys)
                await asyncio.sleep(MOVE_POLL_S)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._target_moving = True
            raise

    def _refresh(self) -> bool:
        """Read the record and the verdicts once; cache both facts; ``True`` on a move."""
        record_moved = self._record_moved() or self._gateway_lags()
        self._verdict = _verdict(self._binding.target, self._session_keys)
        if record_moved:
            self._target_moving = True
        return record_moved

    async def wait_stale(self) -> None:
        """Return once the connector's transport has gone stale.

        Passes through to the connector's own ``wait_stale``, which raises the
        stale transport as a lost connector
        (:class:`~osprey_connectors.errors.ConnectorUnavailableError`): this
        returns on it, and the caller exits :data:`EXIT_TRANSPORT_STALE` so its
        supervisor starts a process with a fresh transport. A session with no
        connector (``unservable``), or one whose connector cannot tell, never
        returns; a caller that is shutting down cancels it. Anything else the
        connector raises propagates.
        """
        from osprey_connectors.errors import ConnectorUnavailableError

        hook = getattr(self._connector, "wait_stale", None)
        if not callable(hook):
            await asyncio.get_running_loop().create_future()
            return  # unreachable: nothing ever resolves that future
        try:
            await hook()
        except ConnectorUnavailableError as exc:
            logger.warning("Control session %s lost its connector: %s", self._binding.id, exc)

    # -- closing ------------------------------------------------------------

    async def close(self) -> None:
        """Stop writes, drain, drop every subscription and disconnect. Idempotent.

        The drain runs to the latest admission's bound (``D_b`` + M). The steps
        run once, in a stored task that every call — the first included —
        awaits shielded, so a caller cancelled while closing cannot cut the
        close in half. ``connector.disconnect()`` is skipped when the drain
        failed: a write still in the connector must not have its transport
        pulled from under it.
        """
        if self._close_task is None:
            self._close_task = asyncio.get_running_loop().create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        self.stop_writes()
        drained = await self.drain(math.inf)
        connector, self._connector = self._connector, None
        subscriptions, self._subscriptions = self._subscriptions, {}
        if connector is None:
            return
        for subscription_id in subscriptions.values():
            if subscription_id is None:
                continue
            try:
                await connector.unsubscribe(subscription_id)
            except Exception:
                logger.debug("Unsubscribe failed while closing", exc_info=True)
        if not drained:
            logger.warning(
                "Writes still in flight at close; the connector is left connected for them"
            )
            return
        try:
            await connector.disconnect()
        except Exception:
            logger.warning("Connector disconnect failed while closing", exc_info=True)
