"""Stateless adapter over bluesky-queueserver's RE manager.

The bridge is the sole facade in front of the queue server: panels, MCP tools,
and the agent never see a qserver API. This module is the one place that speaks
it. Everything here is a thin, typed translation of
``bluesky-queueserver-api``'s ``REManagerAPI`` — the manager (and the Redis it
persists to) is the single source of truth for queue and history, so this class
deliberately caches nothing about them. The only state it owns is the client
handle, the resolved connector type, and the environment-lifecycle policy.

Two responsibilities beyond pass-through:

**Environment ownership.** Queueserver's worker environment is opened and closed
by the bridge and nobody else — it is never exposed to panels or MCP. The bridge
opens it at startup (with bounded retry; connecting devices can take tens of
seconds) when the resolved ``control_system.type`` is EPICS-like, and never on a
mock deployment, where a closed environment is the correct, healthy steady
state. :meth:`QueueBackend.ensure_environment` is both the startup call and the
re-open path when something else closed the environment underneath a deployment
that is supposed to be able to execute.

**Capability derivation.** :meth:`QueueBackend.capability` answers "can plans
actually execute in this deployment, and if not, why not" as a machine-readable
record. It is fail-closed in every direction: an unreadable project config, an
unconfigured manager, and an unreachable manager all yield
``can_execute: False`` with distinct reason codes, so no deployment can ever
advertise an execution ability it does not have. The reason codes are also the
vocabulary every consumer branches on, which is why one of them
(:data:`REASON_SESSION_TARGET_MISMATCH`) is defined here even though only the
host-side MCP server can mint it — see the comment on that constant.

**Emergency abort.** :meth:`QueueBackend.abort` is the one place where this
module composes several manager calls into a single operator-facing operation,
because upstream's Run Engine will not abort a plan that is still running — it
must be paused first. Everything that composition needs to get right (which
pause option, how long to wait, what "already paused" and "it finished while we
were pausing" mean) is stated in that method's docstring.

All failures leave this module as one of the :class:`QueueBackendError`
subclasses below, each carrying a stable ``reason`` code; the route layer maps
the type to an HTTP status and puts the code on the wire.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from bluesky_queueserver_api.comm_base import RequestFailedError, RequestTimeoutError

logger = logging.getLogger("osprey.services.bluesky_bridge.queue_backend")

# Connector types that can drive real Channel Access — a virtual accelerator
# soft-IOC or live hardware. Everything else browses only.
_EPICS_LIKE_CONNECTOR_TYPES = ("virtual_accelerator", "epics")

# Read by `bluesky_queueserver_api` itself when `REManagerAPI` is constructed
# with no explicit address (see `comm_base`), so the compose spec is the only
# place these are set. Named here because `from_env` gates on the control
# address: no address means no manager was deployed at all, which is a
# different — and differently actionable — failure than a deployed manager that
# will not answer.
QSERVER_CONTROL_ADDRESS_ENV = "QSERVER_ZMQ_CONTROL_ADDRESS"
QSERVER_PUBLIC_KEY_ENV = "QSERVER_ZMQ_PUBLIC_KEY"

# The item-metadata key carrying OSPREY's own run id through queueserver and
# into the RunEngine's start documents, which is how live rows and Tiled results
# are matched back to the run the operator enqueued.
RUN_ID_META_KEY = "osprey_run_id"

# The item-metadata key carrying the plan's own identity — its name and the
# kwargs it was enqueued with — along the same path. Start documents are the
# only place a completed run's plan identity survives into Tiled, so this stamp
# is what lets results be rendered as the plan the operator actually asked for.
PLAN_META_KEY = "osprey_plan"

# Manager states in which a plan is under way (or about to be). Enqueuing during
# one of these is an armed operation — the item joins a queue that is already
# draining toward hardware — so the route layer gates it behind the launch
# token. Transitional environment states are deliberately absent: opening or
# closing the environment moves nothing.
QUEUE_ACTIVE_MANAGER_STATES = frozenset(
    {"starting_queue", "executing_queue", "executing_task", "paused"}
)

# The manager state in which the Run Engine is paused and therefore abortable.
# Upstream's `re_abort`/`re_stop`/`re_halt` all require it (see
# `QueueBackend.abort`), so the abort composition targets this state by name.
PAUSED_MANAGER_STATE = "paused"

# The manager state meaning nothing is under way. Reaching it while an abort is
# being prepared means the plan ended on its own — there is nothing left to
# abort, which is a different (and honest) answer from "the abort failed".
IDLE_MANAGER_STATE = "idle"

# Pause option for the abort composition. "deferred" (upstream's default) runs
# the plan on to its next checkpoint, and past the LAST checkpoint it runs the
# plan to completion — the exact opposite of what someone asking for an
# emergency abort wants. "immediate" rolls back to the previous checkpoint and
# pauses now.
ABORT_PAUSE_OPTION = "immediate"

# The polling budget for one worker-namespace function call. It is checked
# between result polls, so the call answers within this budget plus at most one
# in-flight round trip — the client bounds that one itself (`timeout_recv`), and
# a deployment that raises the client's timeout raises this ceiling with it.
# Short by design — the only caller is the read-only pre-flight, whose answer a
# human is waiting on before approving a run, and an answer that never comes is
# worse for them than "no summary available".
FUNCTION_TIMEOUT_S = 10.0

# How often the result of a submitted function is polled for. Well below the
# budget above, so a fast function (the common case: a plan walked in
# milliseconds) is reported back promptly rather than at a coarse tick.
FUNCTION_POLL_INTERVAL_S = 0.1

# Capability reason codes. `can_execute: True` carries `executable`; every other
# code is a distinct, machine-readable "no" that panels and MCP tools branch on
# and that the operator-facing detail text explains.
REASON_EXECUTABLE = "executable"
REASON_BROWSE_ONLY_CONNECTOR = "browse_only_connector"
REASON_UNSUPPORTED_CONNECTOR = "unsupported_connector"
REASON_CONFIG_UNREADABLE = "config_unreadable"
REASON_MANAGER_NOT_CONFIGURED = "manager_not_configured"
REASON_MANAGER_UNREACHABLE = "manager_unreachable"

# The one reason in this vocabulary that this module never mints itself.
#
# A bridge serves exactly one plan lane, wired at render time to one control
# target, and it does not know — must not know — which target the agent session
# is currently pointed at: that lives in a state file written by the controls
# MCP server on the host, which a bridge container cannot see. So the host-side
# bluesky MCP server is what compares the session target against the target this
# deployment's lane serves, and what refuses `queue_add`/`queue_start` while the
# two differ on a single-lane deployment (every deployment, until a second lane
# is opted into).
#
# The code lives here anyway, beside the reasons it stands with, because this
# module is where the capability vocabulary is defined: panels, the JS queue
# client and the MCP tools all branch on these strings, and a code minted in one
# layer but unknown to the others is as useless to them as no code at all. The
# host-side tool module spells the string again rather than importing it (this
# module pulls in `bluesky-queueserver-api`, which that server refuses to
# depend on) and a test pins the two spellings equal.
REASON_SESSION_TARGET_MISMATCH = "session_target_mismatch"

# The env var naming which plan LANE this bridge process is — the service key
# of its own config block (`bluesky`, `bluesky_va`, `bluesky_live`), written
# into the lane's compose environment at render time.
#
# It exists because a two-lane deployment mounts ONE config.yml in both
# bridges: `control_system.type` is the same string in both containers, so the
# config alone cannot tell a bridge which of the two lanes it is. The container
# knows; this is how it says so. Unset means the single-lane deployment every
# project has rendered so far, whose block has always been `services.bluesky`.
LANE_ENV = "OSPREY_BLUESKY_LANE"

# The lane every deployment has until a second one is opted into.
DEFAULT_LANE = "bluesky"

# The one-line flip that turns a browse-only deployment into an executing one.
# Carried in the capability detail so the refusal tells the operator exactly
# what to do rather than only what went wrong.
FLIP_COMMAND = "osprey set connector=virtual_accelerator"

# Same exception set `_resolve_control_system_type` treats as "no readable
# config" — see `_resolve_connector_type` for why this module probes for it
# separately instead of accepting that helper's fail-safe "mock".
_CONFIG_READ_ERRORS = (FileNotFoundError, KeyError, RuntimeError)


def _resolve_control_system_type() -> str:
    """Read ``control_system.type`` from the bridge's mounted project config.

    Single source of truth (Connector = the single control-system interface):
    one config line flips the whole Bluesky stack between the mock connector and
    real Channel Access (virtual accelerator or live hardware) — see the
    ``control-assistant`` preset's ``config.control_system.type`` comment.

    Fail-SAFE default: ``"mock"`` whenever the config can't be read at all (no
    project config context — most unit-test environments — or a transient
    lookup failure), never ``"virtual_accelerator"``/``"epics"`` — the mock
    connector never touches Channel Access, so an unreadable config can never
    silently be reported as able to move hardware.
    """
    from osprey.utils.config import get_config_value

    try:
        control_system_type = get_config_value("control_system.type", "mock")
    except _CONFIG_READ_ERRORS:
        return "mock"

    if not control_system_type or not isinstance(control_system_type, str):
        return "mock"
    return control_system_type


def _baseline_lane_target() -> str:
    """The target a lane with no declared one serves: the deployment baseline.

    ``va`` for a virtual accelerator, ``live`` for everything else — including
    the mock, which serves no virtual accelerator and whose deployment can
    never be switched in practice. That is not this module's rule to invent: it
    is exactly what ``target_banner.resolve_baseline_target`` answers host-side,
    and the host is what refuses ``queue_add`` when the session target differs
    from the lane's. Two rules would let a deployment be refused over a
    mismatch neither side really has. The rule is respelled here rather than
    imported because that module lives in the controls MCP server, which the
    bridge image does not carry; a test pins the two answers equal.
    """
    from osprey_connectors import types as connector_types

    if _resolve_control_system_type() == connector_types.VIRTUAL_ACCELERATOR:
        return connector_types.TARGET_VA
    return connector_types.TARGET_LIVE


def _control_system_section() -> dict[str, Any]:
    """The mounted config's ``control_system:`` mapping, or an empty one.

    Empty is the same answer :func:`_resolve_control_system_type` fails safe to:
    a deployment whose config cannot be read has declared nothing, and every
    resolver that takes this section already reads "nothing declared" as the
    fail-closed side.
    """
    from osprey.utils.config import get_config_value

    try:
        section = get_config_value("control_system", {})
    except _CONFIG_READ_ERRORS:
        return {}
    return section if isinstance(section, dict) else {}


def _declared_lane_target(lane: str) -> str | None:
    """The target ``services.<lane>.target`` declares, or ``None`` when it says nothing.

    ``None`` covers both an unreadable config and the single-lane deployment
    every project has rendered so far, whose block has never carried the key —
    the two want the same answer, which is "this lane declares no target".
    """
    from osprey.utils.config import get_config_value

    try:
        declared = get_config_value(f"services.{lane}.target", None)
    except _CONFIG_READ_ERRORS:
        return None
    return declared if isinstance(declared, str) and declared else None


def resolve_lane_connector_type() -> tuple[str, str | None]:
    """The connector type this lane's queueserver worker builds, and how it degraded.

    A resolution ladder, top rung wins, because a lane's declared target and the
    deployment's own ``control_system.type`` are two different facts and a
    two-lane deployment is exactly where they disagree:

    1. The lane declares NO target — every single-lane deployment — so
       ``control_system.type`` is the whole answer, unchanged from before the
       lane axis existed.
    2. The lane declares a target that RESOLVES on this deployment: that type.
       It is also the key of the block the connector's settings and its write
       posture come from, so a VA lane beside a live baseline is armed by
       ``control_system.connector.virtual_accelerator`` alone.
    3. The lane declares a target this deployment cannot resolve. The LIVE lane
       of a virtual-accelerator baseline is the shipped case: it declares
       ``live`` and the config carries no live connector block, because its
       gateway is a facility address no build can verify and
       ``${EPICS_CA_NAME_SERVERS:?}`` is what supplies it at ``osprey up``. The
       answer stays ``control_system.type`` — what that lane has always built —
       so it keeps working, with the write posture that type inherits from the
       deployment-wide key. The mismatch is *reported* rather than absorbed:
       the lane addresses one machine with a type the config tied to another.
    4. Nothing usable at all: ``mock``, which never touches Channel Access.

    Never raises, for the same reason :func:`resolve_lane_identity` does not.

    Returns:
        ``(connector_type, lane_degraded)``. ``lane_degraded`` is ``None``
        except on rung 3, where it is the operator-facing sentence naming the
        target the lane declared against the type it actually built.
    """
    from osprey_connectors.types import EPICS, resolve_target

    lane = os.environ.get(LANE_ENV) or DEFAULT_LANE
    baseline_type = _resolve_control_system_type()

    declared = _declared_lane_target(lane)
    if declared is None:
        return baseline_type, None

    try:
        return resolve_target(_control_system_section(), declared), None
    except ValueError:
        return baseline_type, (
            f"Lane {lane!r} declares the {declared!r} target, which this deployment "
            f"cannot resolve to a control system, so its worker is built as "
            f"{baseline_type!r} — the deployment baseline — and inherits the "
            f"deployment-wide write posture. Configure the connector block for the "
            f"machine this lane addresses (control_system.connector.{EPICS}, say) to "
            f"make the lane concrete."
        )


def resolve_lane_identity() -> tuple[str, str]:
    """This bridge's lane and the control target that lane serves.

    STATIC, and deliberately so. Both halves come from things fixed when the
    deployment was rendered — the container's own env var and its own
    ``services.<lane>.target`` block — and NEITHER may come from session state.
    The bridge cannot read that state even if this function wanted to: the
    session target lives in a state file the controls MCP server writes on the
    HOST, outside every bridge container's filesystem. Which is the whole
    producer split: each lane publishes what it is, and the host composes the
    active/inactive view by comparing that against the session target it alone
    can see. A record that moved when a session switched would make the two
    layers disagree about a fact only one of them is entitled to.

    Never raises: a lane identity is part of a fail-closed record, so an
    unreadable config yields the same answer an unswitchable deployment gets
    rather than a half-built capability.

    Returns:
        ``(lane, lane_target)`` — the lane's service key, and ``live``/``va``.
    """
    lane = os.environ.get(LANE_ENV) or DEFAULT_LANE
    declared = _declared_lane_target(lane)
    if declared is not None:
        return lane, declared
    # No declared target is the single-lane deployment: it has never had a
    # reason to name a target, because it serves the only one the deployment
    # has.
    return lane, _baseline_lane_target()


class QueueBackendError(Exception):
    """Base class for every failure this adapter reports.

    ``reason`` is the stable, machine-readable code the route layer puts on the
    wire; the message is the human half.
    """

    reason = "queue_backend_error"


class QueueUnavailableError(QueueBackendError):
    """The RE manager could not be reached at all (unconfigured or not answering)."""

    reason = REASON_MANAGER_UNREACHABLE


class QueueRequestRejectedError(QueueBackendError):
    """The manager was reached and refused the request (bad state, unknown uid, ...)."""

    reason = "queue_request_rejected"


class EnvironmentUnavailableError(QueueBackendError):
    """The worker environment is not open and could not be opened."""

    reason = "environment_unavailable"


class NothingRunningError(QueueBackendError):
    """An abort was asked for while no plan was under way.

    Not a failure of the halt path: the manager was reached and answered, and
    the honest answer is that there is nothing to stop. Distinct from
    :class:`QueueRequestRejectedError` so callers can say "nothing is running"
    rather than relaying a generic manager refusal.
    """

    reason = "nothing_running"


class AbortPauseTimeoutError(QueueBackendError):
    """The Run Engine never reached the paused state an abort requires.

    The plan may well still be running. This is the one abort outcome that must
    never be reported as a halt: whoever asked is entitled to know immediately
    that the machine did not stop, so they can reach for their facility's own
    means.
    """

    reason = "abort_pause_timeout"


class FunctionTimeoutError(QueueBackendError):
    """A worker-namespace function did not finish inside its budget.

    The manager accepted the call and may well still be running it; what timed
    out is this process's willingness to wait. Distinct from
    :class:`QueueUnavailableError` because the manager is answering perfectly
    well — the caller learns "no result yet", not "no queue server".
    """

    reason = "function_timeout"


class FunctionFailedError(QueueBackendError):
    """The manager ran a worker-namespace function and the task itself failed.

    The manager was reached, accepted the call, ran it, and reported back that
    it did not succeed — a function that raised, one whose return value could
    not be serialized, or a result the manager no longer holds. The message
    carries whatever the manager said about it.
    """

    reason = "function_failed"


class ExecutionUnavailableError(QueueBackendError):
    """This deployment cannot execute plans; carries the capability's reason code."""

    reason = "execution_unavailable"

    def __init__(self, capability: Capability) -> None:
        super().__init__(capability.detail)
        self.capability = capability
        self.reason = capability.reason


@dataclass(frozen=True)
class Capability:
    """Whether plans can execute in this deployment, and why not when they cannot.

    Attributes:
        can_execute: True only when a reachable manager is backed by a connector
            that can drive real Channel Access.
        reason: Machine-readable code — one of the ``REASON_*`` constants.
        detail: Operator-facing explanation. For a browse-only mock deployment
            it names the exact command that flips it.
        lane: The plan lane this bridge is — its own service key
            (``bluesky``/``bluesky_va``/``bluesky_live``).
        lane_target: The control target that lane serves, ``live`` or ``va``.
            Fixed at render time; see :func:`resolve_lane_identity` for why it
            can never be the SESSION's target.
        lane_degraded: ``None`` on a lane whose declared target resolves to a
            connector type this deployment configured, which is every lane that
            is fully described. Otherwise the sentence naming what the lane
            declared against what its worker was actually built as — see rung 3
            of :func:`resolve_lane_connector_type`. Independent of
            ``can_execute``: such a lane executes plans perfectly well, against
            a machine the config never tied to its target.
    """

    can_execute: bool
    reason: str
    detail: str
    lane: str
    lane_target: str
    lane_degraded: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The capability as the JSON object the status surface publishes."""
        return {
            "can_execute": self.can_execute,
            "reason": self.reason,
            "detail": self.detail,
            "lane": self.lane,
            "lane_target": self.lane_target,
            "lane_degraded": self.lane_degraded,
        }


def _resolve_connector_type() -> str | None:
    """The deployment's ``control_system.type``, or ``None`` when it can't be read.

    Delegates the actual lookup to :func:`_resolve_control_system_type`, which
    fails safe to ``"mock"`` when the project config is unreadable. That is the right *safety*
    answer — an unreadable config must never connect to hardware — but it
    collapses two situations the capability record needs to keep apart:
    a deployment that is deliberately mock (flip the connector to fix it) and
    one whose config the bridge simply could not read (fix the mount). So read
    the config once to learn whether it is readable at all, then let the shared
    helper produce the value.
    """
    from osprey.utils.config import get_config_value

    try:
        get_config_value("control_system.type", "mock")
    except _CONFIG_READ_ERRORS:
        return None
    return _resolve_control_system_type()


def is_environment_open(status: dict[str, Any]) -> bool:
    """True if the manager reports a live worker environment."""
    return bool(status.get("worker_environment_exists"))


def is_queue_active(status: dict[str, Any]) -> bool:
    """True if the manager is running a plan or on its way to running one."""
    return status.get("manager_state") in QUEUE_ACTIVE_MANAGER_STATES


def run_id_of(item: dict[str, Any]) -> str | None:
    """The OSPREY run id carried in a queue or history item's metadata, if any.

    Queueserver's ``meta`` is whatever the enqueuer put there, so this tolerates
    items that predate — or simply lack — the key.
    """
    meta = item.get("meta")
    if isinstance(meta, dict):
        run_id = meta.get(RUN_ID_META_KEY)
        if isinstance(run_id, str):
            return run_id
    return None


class QueueBackend:
    """Typed, stateless adapter over one queueserver RE manager.

    Args:
        manager: A connected ``bluesky_queueserver_api`` ``REManagerAPI`` (the
            ``zmq.aio`` flavour in production), or ``None`` when no manager is
            deployed. Injected rather than constructed so tests can drive a
            mocked client; use :meth:`from_env` for the production path.
        env_open_attempts: How many times to ask the manager to open the
            environment before giving up.
        env_open_polls: Status polls to wait per attempt for the environment to
            come up. Connecting devices can take tens of seconds, so the default
            window is generous.
        env_poll_interval: Seconds between those polls.
        abort_pause_polls: Status polls to wait for the Run Engine to reach the
            paused state during an abort. Multiplied by ``abort_poll_interval``
            this is the abort's SLEEP budget only (10s at the defaults) — it
            excludes the manager round trips, so it is not the whole abort's
            latency. The true worst case adds every call the composition makes,
            each bounded by the queueserver client's own per-request timeout
            (2.0s by default): 1 initial status + 1 initial ``re_pause`` + one
            status and one ``re_pause`` re-issue per poll + ``re_abort`` + a
            final status, i.e. ~43 round trips, for ~96s in total at the
            defaults. That figure is what the layered clients budget against
            (the bluesky-web sidecar's ``queue_relay._ABORT_TIMEOUT`` and the MCP
            ``stop_run`` tool both allow 120s); change these defaults and those
            two must move with them. The sleep budget stays deliberately short
            regardless — an operator waiting on a halt must get an answer, one
            way or the other, in seconds.
        abort_poll_interval: Seconds between those polls.
        function_timeout: Polling budget for :meth:`function_execute`, in
            seconds. Unlike the abort budget above it counts the whole call
            rather than only the sleeps — but it is checked between polls, so
            the ceiling is this budget plus one in-flight round trip, which the
            client bounds on its own (``timeout_recv``).
        function_poll_interval: Seconds between those result polls.
    """

    def __init__(
        self,
        manager: Any | None,
        *,
        env_open_attempts: int = 3,
        env_open_polls: int = 30,
        env_poll_interval: float = 1.0,
        abort_pause_polls: int = 20,
        abort_poll_interval: float = 0.5,
        function_timeout: float = FUNCTION_TIMEOUT_S,
        function_poll_interval: float = FUNCTION_POLL_INTERVAL_S,
    ) -> None:
        self._manager = manager
        self._env_open_attempts = env_open_attempts
        self._env_open_polls = env_open_polls
        self._env_poll_interval = env_poll_interval
        self._abort_pause_polls = abort_pause_polls
        self._abort_poll_interval = abort_poll_interval
        self._function_timeout = function_timeout
        self._function_poll_interval = function_poll_interval

    @classmethod
    def from_env(cls, **kwargs: Any) -> QueueBackend:
        """Build a backend against the manager the compose spec points at.

        Returns a backend with no manager — every call fail-closed, capability
        ``manager_not_configured`` — when ``QSERVER_ZMQ_CONTROL_ADDRESS`` is
        unset, which is the normal shape of a browse-only deployment that ships
        no queue server at all. The address and CurveZMQ public key are read by
        ``REManagerAPI`` itself from the environment.
        """
        if not os.environ.get(QSERVER_CONTROL_ADDRESS_ENV):
            logger.info(
                "%s unset - queue backend running without a manager (browse-only)",
                QSERVER_CONTROL_ADDRESS_ENV,
            )
            return cls(None, **kwargs)

        from bluesky_queueserver_api.zmq.aio import REManagerAPI

        return cls(REManagerAPI(), **kwargs)

    # ---------------------------------------------------------------- plumbing

    def _require_manager(self) -> Any:
        if self._manager is None:
            raise QueueUnavailableError(
                "No queue server is configured for this deployment "
                f"({QSERVER_CONTROL_ADDRESS_ENV} is unset)."
            )
        return self._manager

    async def _call(self, method: str, /, **kwargs: Any) -> dict[str, Any]:
        """Invoke one manager method, translating its errors into ours.

        A timeout means the manager is unreachable (down, wrong address, or the
        CurveZMQ handshake failed) and is retryable; an explicit failure means
        the manager answered and said no, which is not.
        """
        manager = self._require_manager()
        try:
            result = await getattr(manager, method)(**kwargs)
        except RequestTimeoutError as exc:
            raise QueueUnavailableError(f"Queue server did not answer {method!r}: {exc}") from exc
        except RequestFailedError as exc:
            raise QueueRequestRejectedError(f"Queue server refused {method!r}: {exc}") from exc
        return dict(result) if result is not None else {}

    async def close(self) -> None:
        """Release the client's transport. Safe to call with no manager."""
        if self._manager is not None:
            await self._manager.close()

    # ------------------------------------------------------------- queue reads

    async def status(self, *, reload: bool = False) -> dict[str, Any]:
        """The manager's status document.

        Args:
            reload: Bypass the client's short-lived status cache. Required
                wherever a stale read would be a safety question rather than a
                cosmetic one — the enqueue/start arming checks and the
                environment-open polling loop.
        """
        return await self._call("status", reload=reload)

    async def items(self) -> dict[str, Any]:
        """The queue as the manager holds it: pending ``items`` plus ``running_item``."""
        return await self._call("queue_get")

    async def history(self) -> dict[str, Any]:
        """Recent completed items, newest last, as the manager holds them."""
        return await self._call("history_get")

    # ---------------------------------------------------------- queue mutation

    async def add_item(
        self,
        item: Any,
        *,
        run_id: str | None = None,
        pos: Any = None,
        before_uid: str | None = None,
        after_uid: str | None = None,
    ) -> dict[str, Any]:
        """Append (or insert) one item into the queue.

        When a run id is given, the item also carries the plan's identity — its
        name and kwargs — under :data:`PLAN_META_KEY`, so a finished run can be
        rendered as the plan it was without consulting the queue.

        Args:
            item: A queueserver item — a plain dict, or a ``BItem``/``BPlan``.
            run_id: OSPREY's run id for this item. Threaded through the item's
                metadata so the RunEngine's start document carries it and live
                rows and Tiled results can be matched back to the enqueued run.
            pos: Queue position, per queueserver (``"front"``, ``"back"``, or an
                index). Defaults to the manager's own default (back).
            before_uid: Insert ahead of this item.
            after_uid: Insert behind this item.

        Returns:
            The manager's response, including the created ``item`` with its
            server-assigned ``item_uid``.
        """
        payload = self._as_item_dict(item)
        if run_id is not None:
            meta = dict(payload.get("meta") or {})
            meta[RUN_ID_META_KEY] = run_id
            meta[PLAN_META_KEY] = {
                "name": payload.get("name"),
                "kwargs": dict(payload.get("kwargs") or {}),
            }
            payload["meta"] = meta

        kwargs: dict[str, Any] = {"item": payload}
        if pos is not None:
            kwargs["pos"] = pos
        if before_uid is not None:
            kwargs["before_uid"] = before_uid
        if after_uid is not None:
            kwargs["after_uid"] = after_uid
        return await self._call("item_add", **kwargs)

    async def reorder(
        self,
        uid: str,
        *,
        pos_dest: Any = None,
        before_uid: str | None = None,
        after_uid: str | None = None,
    ) -> dict[str, Any]:
        """Move one queued item.

        Exactly one destination must be given: ``pos_dest`` (``"front"``,
        ``"back"``, or an index), ``before_uid``, or ``after_uid``.
        """
        destinations = [pos_dest, before_uid, after_uid]
        if sum(dest is not None for dest in destinations) != 1:
            raise QueueRequestRejectedError(
                "reorder needs exactly one of pos_dest, before_uid, after_uid."
            )

        kwargs: dict[str, Any] = {"uid": uid}
        if pos_dest is not None:
            kwargs["pos_dest"] = pos_dest
        if before_uid is not None:
            kwargs["before_uid"] = before_uid
        if after_uid is not None:
            kwargs["after_uid"] = after_uid
        return await self._call("item_move", **kwargs)

    async def remove(self, uid: str) -> dict[str, Any]:
        """Drop one queued item. The running item is not removable — stop it instead."""
        return await self._call("item_remove", uid=uid)

    async def start(self) -> dict[str, Any]:
        """Start draining the queue.

        The environment must already be open; callers on the armed path go
        through :meth:`ensure_environment` first. Autostart stays disabled on
        the manager, so this is the only way execution ever begins.
        """
        return await self._call("queue_start")

    async def stop(self, *, cancel: bool = False) -> dict[str, Any]:
        """Stop the queue after the running item finishes, or cancel a pending stop.

        Args:
            cancel: Withdraw a stop request that has not taken effect yet,
                leaving the queue draining.
        """
        return await self._call("queue_stop_cancel" if cancel else "queue_stop")

    async def abort(self) -> dict[str, Any]:
        """Abort the plan that is running RIGHT NOW, leaving the queue stopped.

        This is the emergency halt, and the only operation in this module that
        composes several manager calls. Upstream will not abort a plan that is
        still running: ``re_abort`` (like ``re_stop``/``re_halt``) acts only on
        a Run Engine that is already **paused**, so the sequence is

        1. ``status(reload=True)`` — fresh, never the client's ~0.5 s cache. The
           whole decision below turns on the manager's live state, and acting on
           a half-second-old read is exactly the class of bug that makes a halt
           lie.
        2. If the Run Engine is already paused, skip straight to step 4 (a human
           may have paused it from qserver's own console).
        3. Otherwise, if the queue is active, ``re_pause(option="immediate")``
           and then poll ``status(reload=True)`` every ``abort_poll_interval``
           for up to ``abort_pause_polls`` polls, re-issuing the pause on each
           poll that still shows an active-but-unpaused manager. Re-issuing
           covers the ``starting_queue`` window, where the first pause is
           refused because the plan has not begun yet; an immediate pause that
           already landed is not re-applied, because the loop exits the moment
           the state reads ``paused``.
        4. ``re_abort`` — the plan's remaining points are discarded and the
           queue is left stopped.
        5. One final ``status(reload=True)`` purely to report the state. A
           failure here is swallowed: the abort has already been accepted, and
           turning a reporting failure into a 503 would tell the operator the
           machine did not stop when it did.

        Not active at all (idle, or an environment-lifecycle state) raises
        :class:`NothingRunningError` rather than issuing anything — an honest
        "there is nothing to stop", not a failure. If the plan ENDS while the
        pause is being chased, the poll sees ``idle`` and raises the same error,
        which is again the truth: nothing is running now.

        Returns:
            ``{"aborted": True, "abort_pending", "paused_first", "manager_state",
            "msg"}``. ``abort_pending`` is True when the manager had not yet
            settled out of an active state at the moment of the final read — the
            abort is accepted and unwinding, not finished. ``msg`` is the
            manager's own sentence, relayed rather than rewritten.

        Raises:
            NothingRunningError: No plan was under way (409 at the route layer).
            AbortPauseTimeoutError: The Run Engine never paused, so nothing was
                aborted and the plan may still be running.
            QueueUnavailableError: The manager stopped answering.
            QueueRequestRejectedError: The manager reached the paused state but
                refused the abort itself.
        """
        status = await self.status(reload=True)
        state = status.get("manager_state")

        paused_first = False
        if state != PAUSED_MANAGER_STATE:
            if not is_queue_active(status):
                raise NothingRunningError(
                    f"No plan is running, so there is nothing to abort (manager state {state!r})."
                )
            paused_first = True
            state = await self._pause_for_abort()
            if state is None:
                raise AbortPauseTimeoutError(
                    "The Run Engine did not reach the paused state an abort requires "
                    f"within {self._abort_pause_polls * self._abort_poll_interval:g}s, so "
                    "NOTHING WAS ABORTED and the plan may still be running."
                )
            if state == IDLE_MANAGER_STATE:
                raise NothingRunningError(
                    "The plan ended while the abort was being prepared, so there was "
                    "nothing left to abort; the queue is idle."
                )

        result = await self._call("re_abort")

        try:
            final_state = (await self.status(reload=True)).get("manager_state")
        except QueueBackendError as exc:  # pragma: no cover - narrow race
            # The abort is already accepted. Reporting is best-effort; a raise
            # here would report a delivered halt as a failure.
            logger.warning("abort accepted but the follow-up status read failed: %s", exc)
            final_state = None

        return {
            "aborted": True,
            "abort_pending": final_state is None or final_state != IDLE_MANAGER_STATE,
            "paused_first": paused_first,
            "manager_state": final_state,
            "msg": str(result.get("msg") or ""),
        }

    async def _pause_for_abort(self) -> str | None:
        """Drive the Run Engine to ``paused`` (or observe it go ``idle``).

        Returns the settling manager state, or ``None`` when the bounded window
        closed with the manager still active and unpaused. A refused pause is
        logged, never raised: the manager refuses one whenever no plan is
        running *yet*, and the poll below is what settles whether that was a
        transient window or a genuine nothing-to-abort.
        """
        await self._request_pause()
        for _ in range(self._abort_pause_polls):
            await asyncio.sleep(self._abort_poll_interval)
            status = await self.status(reload=True)
            state = status.get("manager_state")
            if state == PAUSED_MANAGER_STATE:
                return PAUSED_MANAGER_STATE
            if not is_queue_active(status):
                # Includes `idle`: the plan finished on its own. Anything else
                # non-active means the queue is no longer draining either.
                return IDLE_MANAGER_STATE
            await self._request_pause()
        return None

    async def _request_pause(self) -> None:
        """Ask for an immediate pause, tolerating the manager's refusal."""
        try:
            await self._call("re_pause", option=ABORT_PAUSE_OPTION)
        except QueueRequestRejectedError as exc:
            logger.debug("re_pause refused while preparing an abort: %s", exc)

    # -------------------------------------------------------- worker namespace

    async def upload_script(self, script: str, *, update_lists: bool = True) -> dict[str, Any]:
        """Execute a script in the worker namespace, defining the plans it declares.

        How validated session plans reach the worker: the namespace is rebuilt
        from the startup script on every environment cycle, so the bridge
        re-uploads its validated set after each open. The manager runs the
        script as a background task and answers immediately with a
        ``task_uid`` — poll :meth:`task_result` for the outcome.

        Args:
            script: Python source to execute in the worker's namespace.
            update_lists: Refresh the existing/allowed plan and device lists
                afterwards, so a newly defined plan becomes enqueueable.
        """
        return await self._call("script_upload", script=script, update_lists=update_lists)

    async def task_result(self, task_uid: str) -> dict[str, Any]:
        """The status and result of a background manager task, e.g. a script upload."""
        return await self._call("task_result", task_uid=task_uid)

    async def function_execute(self, name: str, *args: Any) -> Any:
        """Run one function that already exists in the worker namespace, and
        return what it returned.

        The manager only *starts* a function: it answers with a task id and the
        return value is collected afterwards. This method hides that two-step
        behind one await, polling until ``function_timeout`` closes, so no
        caller can be left waiting on a worker that never finishes. The budget
        is checked between polls, which puts the real ceiling at the budget plus
        one in-flight round trip; the client bounds that one itself.

        Submitted with ``run_in_background=True``, which is what makes the call
        possible at all while a plan is running: a foreground task is accepted
        only by an idle manager, and a read-only summary an operator asks for
        mid-queue is exactly the case that would otherwise be refused. The
        function itself must therefore be one that is safe to run beside a plan;
        upstream runs background tasks on their own thread and guarantees
        nothing about them.

        Which functions may be called at all is the manager's permissions to
        decide, not this module's — an unpermitted name is refused by the
        manager and arrives here as a rejection like any other.

        Args:
            name: The function's name in the worker namespace.
            *args: Positional arguments for it. Must be JSON-serializable, as
                must its return value; the manager fails the task otherwise.

        Returns:
            The function's own return value, as the manager deserialized it.

        Raises:
            FunctionTimeoutError: The budget closed with the task unfinished.
            FunctionFailedError: The manager ran it and reported a failure, no
                longer holds its result, or accepted it without naming a task —
                all three are the call going wrong after it was taken, not a
                refusal to take it.
            QueueUnavailableError: The manager stopped answering.
            QueueRequestRejectedError: The manager refused the call — an
                unpermitted function name, or a worker that cannot take it
                right now.
        """
        deadline = time.monotonic() + self._function_timeout
        item = {"item_type": "function", "name": name, "args": list(args), "kwargs": {}}
        reply = await self._call("function_execute", item=item, run_in_background=True)

        task_uid = reply.get("task_uid")
        if not isinstance(task_uid, str) or not task_uid:
            # An acceptance with nowhere to read the result from is a protocol
            # anomaly, not a refusal: the manager said yes. Classified as a
            # failure so nobody reading the reason goes looking for a
            # permission that was never denied.
            raise FunctionFailedError(
                f"The queue server accepted {name!r} without naming a task to read its "
                "result from, so there is no result to wait for."
            )

        while True:
            reply = await self.task_result(task_uid)
            state = reply.get("status")
            result = reply.get("result")
            result = result if isinstance(result, dict) else {}

            if state == "completed":
                if result.get("success"):
                    return result.get("return_value")
                raise FunctionFailedError(
                    f"The worker ran {name!r} and it failed: "
                    f"{result.get('msg') or result.get('return_value') or 'no detail reported'}"
                )
            if state == "not_found":
                # Results expire after a retention window far longer than this
                # method's budget, so reaching this is a manager that never held
                # the task — not a result this call waited too long for.
                raise FunctionFailedError(
                    f"The queue server holds no task {task_uid!r} for {name!r}, so its "
                    "result cannot be read."
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise FunctionTimeoutError(
                    f"{name!r} had not finished after {self._function_timeout:g}s, so no "
                    "result was read; the worker may still be running it."
                )
            await asyncio.sleep(min(self._function_poll_interval, remaining))

    async def plans_allowed(self) -> dict[str, Any]:
        """The plans the worker namespace currently exposes.

        The authority on whether an enqueued plan name still exists — an
        environment cycle that dropped a session plan shows up here before it
        shows up as a failed run.
        """
        return await self._call("plans_allowed")

    async def devices_allowed(self) -> dict[str, Any]:
        """The devices the worker namespace currently exposes.

        The authority on which device *names* a plan's parameters can carry:
        every plan resolves its device fields by string name against the
        worker's namespace, so a name absent here fails the run on its first
        iteration. Reading it is how a caller learns the set instead of
        guessing at it.
        """
        return await self._call("devices_allowed")

    # ---------------------------------------------- environment lifecycle

    async def open_environment(self) -> dict[str, Any]:
        """Ask the manager to spawn the RE worker. Returns as soon as it accepts.

        Callers that need the environment to actually be up want
        :meth:`ensure_environment`, which waits.
        """
        return await self._call("environment_open")

    async def close_environment(self) -> dict[str, Any]:
        """Tear down the RE worker.

        Refused while the queue is active: closing the environment mid-plan
        would abort work the operator is watching. Stop the queue first.
        """
        status = await self.status(reload=True)
        if is_queue_active(status):
            raise QueueRequestRejectedError(
                "Cannot close the worker environment while the queue is active "
                f"(manager state {status.get('manager_state')!r}). Stop the queue first."
            )
        return await self._call("environment_close")

    async def ensure_environment(self) -> bool:
        """Make sure the worker environment is open, opening it if it is not.

        This is both the bridge's startup call and its self-healing path: any
        time the bridge finds the environment closed on a deployment whose
        capability says it should be able to execute, it comes back through
        here. On a browse-only deployment it opens nothing and returns False —
        a closed environment is that deployment's healthy steady state.

        Returns:
            True if the environment is open when this returns.

        Raises:
            EnvironmentUnavailableError: The manager accepted the open requests
                but the environment never came up within the bounded window.
            QueueUnavailableError: The manager stopped answering.
        """
        capability = await self.capability()
        if not capability.can_execute:
            logger.debug(
                "not opening the worker environment: %s (%s)",
                capability.reason,
                capability.detail,
            )
            return False

        for attempt in range(1, self._env_open_attempts + 1):
            status = await self.status(reload=True)
            if is_environment_open(status):
                return True

            if status.get("manager_state") != "creating_environment":
                try:
                    await self.open_environment()
                except QueueRequestRejectedError as exc:
                    # A concurrent opener won the race, or the manager is busy
                    # with something else; the poll below settles which.
                    logger.debug("environment_open refused on attempt %d: %s", attempt, exc)

            if await self._await_environment():
                return True

            logger.warning(
                "worker environment did not come up on attempt %d/%d",
                attempt,
                self._env_open_attempts,
            )

        raise EnvironmentUnavailableError(
            f"The worker environment did not open after {self._env_open_attempts} attempts."
        )

    async def ensure_environment_for_execute(self) -> None:
        """Guard the armed path: refuse unless plans can actually run right now.

        Raises:
            ExecutionUnavailableError: This deployment cannot execute plans at
                all; the exception carries the capability record.
            EnvironmentUnavailableError: It should be able to, but the worker
                environment will not come up.
        """
        capability = await self.capability()
        if not capability.can_execute:
            raise ExecutionUnavailableError(capability)
        if not await self.ensure_environment():
            raise EnvironmentUnavailableError("The worker environment is not open.")

    async def _await_environment(self) -> bool:
        """Poll the manager until the environment exists or the window closes."""
        for _ in range(self._env_open_polls):
            await asyncio.sleep(self._env_poll_interval)
            if is_environment_open(await self.status(reload=True)):
                return True
        return False

    # ------------------------------------------------------------- capability

    async def capability(self) -> Capability:
        """Whether this deployment can execute plans, and why not when it cannot.

        Fail-closed at every step, and ordered so the operator gets the most
        actionable answer: the connector is checked before the manager, so a
        mock deployment is told to flip the connector rather than that some
        queue server it was never meant to have is unreachable.

        The lane identity rides on every branch, including the refusals: a
        consumer looking at a "no" needs to know WHICH lane said it. It is
        resolved once, up here, from render-time facts alone — nothing below
        can make it depend on what the manager answered.

        The connector type below is the LANE's, not the deployment's: the
        queueserver container beside this bridge builds its worker by
        :func:`resolve_lane_connector_type`, and a record naming a different
        connector than the one plans will actually run against would be a
        capability describing some other lane.
        """
        lane, lane_target = resolve_lane_identity()
        if _resolve_connector_type() is None:
            return Capability(
                can_execute=False,
                reason=REASON_CONFIG_UNREADABLE,
                lane=lane,
                lane_target=lane_target,
                detail=(
                    "The bridge cannot read the project's control_system.type, so it will "
                    "not claim plans can execute. Check the config mount on the bridge "
                    "container."
                ),
            )
        connector_type, lane_degraded = resolve_lane_connector_type()

        if connector_type == "mock":
            return Capability(
                can_execute=False,
                reason=REASON_BROWSE_ONLY_CONNECTOR,
                lane=lane,
                lane_target=lane_target,
                lane_degraded=lane_degraded,
                detail=(
                    "This deployment uses the mock connector, which cannot move hardware, "
                    "so plans can be composed and validated but not executed. To "
                    f"execute plans, run `{FLIP_COMMAND}` and redeploy."
                ),
            )

        if connector_type not in _EPICS_LIKE_CONNECTOR_TYPES:
            return Capability(
                can_execute=False,
                reason=REASON_UNSUPPORTED_CONNECTOR,
                lane=lane,
                lane_target=lane_target,
                lane_degraded=lane_degraded,
                detail=(
                    f"The {connector_type!r} connector is not one the plan stack can execute "
                    f"plans against. To execute plans, run `{FLIP_COMMAND}` and redeploy."
                ),
            )

        if self._manager is None:
            return Capability(
                can_execute=False,
                reason=REASON_MANAGER_NOT_CONFIGURED,
                lane=lane,
                lane_target=lane_target,
                lane_degraded=lane_degraded,
                detail=(
                    "No queue server is configured for this deployment "
                    f"({QSERVER_CONTROL_ADDRESS_ENV} is unset), so plans cannot be executed."
                ),
            )

        try:
            await self.status()
        except QueueBackendError as exc:
            return Capability(
                can_execute=False,
                reason=REASON_MANAGER_UNREACHABLE,
                lane=lane,
                lane_target=lane_target,
                lane_degraded=lane_degraded,
                detail=f"The queue server is not answering, so plans cannot be executed: {exc}",
            )

        return Capability(
            can_execute=True,
            reason=REASON_EXECUTABLE,
            lane=lane,
            lane_target=lane_target,
            lane_degraded=lane_degraded,
            detail=f"Plans execute against the {connector_type!r} connector.",
        )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _as_item_dict(item: Any) -> dict[str, Any]:
        """Normalize a ``BItem``/``BPlan`` or plain mapping to the wire dict.

        Working in dicts keeps the metadata injection in :meth:`add_item` honest
        about the shape that actually reaches the manager, and copies so a
        caller's item is never mutated.
        """
        to_dict = getattr(item, "to_dict", None)
        if callable(to_dict):
            return dict(to_dict())
        if isinstance(item, dict):
            return dict(item)
        raise QueueRequestRejectedError(f"Not a queue item: {type(item).__name__}")
