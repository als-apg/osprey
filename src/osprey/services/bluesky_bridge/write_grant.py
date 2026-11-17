"""Under which write authorization a queued plan's writes go out.

A plan the queueserver worker runs writes through the worker's one in-process
connector, and that connector judges a write against a program's bounded write
authorization (:mod:`osprey_connectors.write_authorization`) whenever the
process names one (``OSPREY_WRITE_AUTHORIZATION``, read on every write). This
module gives each plan run its own, the way an optimizer run holds one:

1. **Survey.** The plan is built a second time and stepped beside the real
   one (:func:`survey_writes`): every ``set`` message it yields is one write
   call, so the survey counts the steps and the range of values per channel.
   Its ``read`` and ``locate`` messages go to the RunEngine and get the real
   answer, which moves nothing, so a plan that writes relative to where a
   device sits (an ORM sweep about each working point) is surveyed at the
   values it will write. Every other message is answered ``None`` and never
   carried out. A plan that reads may still branch on readings that change
   once it moves, so its survey carries headroom.
2. **Acquire.** At ``allowed`` the worker issues the grant itself
   (``self_issue``). At ``asked`` it files a request — the web terminal's chip
   and ``osprey write-authorization approve`` list it — and waits, yielding
   ``sleep`` messages so the RunEngine still pauses and aborts. At ``off`` it
   asks nobody and the run goes on exactly as before: the connector refuses
   its writes. A plan that writes nothing asks nobody.
3. **Extend.** Before each ``set`` the worker checks the grant it holds against
   that write (:func:`_shortfall`), without asking the verdict: the connector
   still judges every write. When the survey fell short — no step left, a value outside its range, a
   channel it never saw, or the grant expired — the worker files a
   *replacement* request: the steps still unspent plus more, and every range
   widened to the value in hand. A person approves the whole scope going
   forward, never a bare increment. The new grant is named for the next write;
   the old one is no longer renewed, so its lease lapses. At most
   :data:`MAX_EXTENSIONS` replacements per run; past that the write goes to the
   connector and is refused.

A denial, an expiry of the wait, or a run aborted while it waits ends the
plan with :class:`WriteAuthorizationError` and leaves the machine where it
is: nothing is restored on the run's behalf.

The flow is on only where the worker has a ledger directory
(``OSPREY_WRITE_LEDGER_DIR``), which the compose template binds; without one
every plan runs as before.
"""

from __future__ import annotations

import logging
import math
import numbers
import os
import threading
import time
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from osprey_connectors.write_authorization import Grant

logger = logging.getLogger("osprey.services.bluesky_bridge.write_grant")

#: How long a run waits for a person to answer one request.
WAIT_S_ENV = "BLUESKY_WRITE_AUTHORIZATION_WAIT_S"
DEFAULT_WAIT_S = 900.0
#: How often the wait asks whether the request was approved.
POLL_S = 2.0
#: One renewal of the lease; the heartbeat renews at a third of it.
LEASE_S = 60.0
#: How long a grant stands from its issue. An extension issues a fresh one.
EXPIRES_S = 2 * 3600.0
#: Replacement requests one run may file.
MAX_EXTENSIONS = 3
#: The fewest steps an extension for an exhausted budget asks for.
EXTRA_STEPS_FLOOR = 10
#: Messages the survey steps through before it stops counting.
SURVEY_MESSAGE_CAP = 200_000

#: Messages whose answer a plan may compute its next move from. The survey
#: answers them ``None``, so a plan that asks one may go elsewhere for real.
_ANSWERED = frozenset({"read", "locate"})

#: The holder a bridge reports requests for: its own lane's queueserver.
WORKER_HOLDER_ENV = "BLUESKY_WORKER_HOLDER"

#: The refusals an extension can answer: the grant is short, not wrong.
EXTENDABLE = frozenset({"auth_exhausted", "auth_range", "auth_channel", "auth_expired"})


class WriteAuthorizationError(RuntimeError):
    """The run has no authorization for its next write, and will not get one."""

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = reason
        super().__init__(f"{detail} [osprey-refusal: {reason}]")


@dataclass
class WriteSurvey:
    """What a plan writes, as far as stepping it without a RunEngine shows.

    Attributes:
        ranges: ``{address: [min, max]}`` of the values each channel is set to.
        steps: The ``set`` messages seen, one write call each.
        complete: The plan ran to its end; ``False`` when it stopped early.
        target: The control target of the connector the writes go to.
    """

    ranges: dict[str, list[float]] = field(default_factory=dict)
    steps: int = 0
    complete: bool = True
    target: str | None = None

    def add(self, address: str, value: float) -> None:
        bounds = self.ranges.setdefault(address, [value, value])
        bounds[0] = min(bounds[0], value)
        bounds[1] = max(bounds[1], value)


def enabled() -> bool:
    """Whether this worker has a ledger directory to hold write authorizations in."""
    from osprey_connectors.write_authorization import LEDGER_DIR_ENV_VAR

    return bool((os.environ.get(LEDGER_DIR_ENV_VAR) or "").strip())


def _write_of(msg: Any) -> tuple[str, Any, str | None] | None:
    """``(address, value, target)`` of a ``set`` message to a connector device, else ``None``."""
    if getattr(msg, "command", None) != "set" or not msg.args:
        return None
    address = getattr(msg.obj, "_setpoint_pv", None)
    if not isinstance(address, str):
        return None
    connector = getattr(msg.obj, "_osprey_connector", None)
    target = getattr(connector, "_control_target", None)
    return address, msg.args[0], target if isinstance(target, str) else None


def _number(address: str, value: Any) -> float:
    """*value* as a finite float, or the error a write authorization would refuse it with."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise WriteAuthorizationError(
            "auth_range",
            f"The plan sets {address} to {value!r}; a write authorization covers "
            "finite numbers only.",
        )
    return float(value)


def survey_writes(build: Callable[[], Iterable[Any]]) -> Generator[Any, Any, WriteSurvey]:
    """Step a freshly built plan and count what it writes; ``yield from`` it in a plan.

    Only the plan's ``read`` and ``locate`` messages are yielded on, to the
    RunEngine running the caller, and their real answers sent back: neither
    moves anything, and a write computed from where a device sits is then the
    write the run will make. Every other message is answered ``None`` and
    never carried out — no ``set``, ``trigger`` or ``sleep`` of the surveyed
    copy reaches the RunEngine. The survey is marked incomplete when the plan
    reads (it may branch on readings that change once it moves), when it
    raises or ends early on a ``None`` it could not use, and past
    :data:`SURVEY_MESSAGE_CAP` messages. What it saw up to there is kept.
    """
    survey = WriteSurvey()
    try:
        plan = iter(build())
    except Exception:  # a plan that cannot be built twice is surveyed as nothing
        logger.debug("write survey: the plan could not be built again", exc_info=True)
        survey.complete = False
        return survey
    reply = None
    try:
        for _ in range(SURVEY_MESSAGE_CAP):
            try:
                msg = plan.send(reply) if hasattr(plan, "send") else next(plan)
            except StopIteration:
                return survey
            except WriteAuthorizationError:
                raise
            except Exception:  # the plan needed an answer the survey cannot give
                logger.debug("write survey stopped early", exc_info=True)
                survey.complete = False
                return survey
            reply = None
            if getattr(msg, "command", None) in _ANSWERED:
                survey.complete = False
                # The RunEngine's answer. What it throws in here instead (an
                # abort, a read that failed) is the run's, and goes on up.
                reply = yield msg
                continue
            write = _write_of(msg)
            if write is None:
                continue
            address, value, target = write
            survey.add(address, _number(address, value))
            survey.steps += 1
            survey.target = survey.target or target
        survey.complete = False
        return survey
    finally:
        close = getattr(plan, "close", None)
        if close is not None:
            close()


def _wait_s() -> float:
    raw = os.environ.get(WAIT_S_ENV)
    try:
        return float(raw) if raw else DEFAULT_WAIT_S
    except ValueError:
        return DEFAULT_WAIT_S


def _section() -> Any:
    from osprey_connectors import config as connectors_config

    try:
        return connectors_config.get_config_value("control_system", {})
    except (FileNotFoundError, KeyError, RuntimeError):
        return {}


def _shortfall(grant: Grant, spent: int, address: str, value: float) -> str | None:
    """Which of the grant's own terms *one more write* of *value* to *address* falls short of.

    The terms a replacement can widen (:data:`EXTENDABLE`), judged from the
    grant the worker holds and the steps it has spent under it. Everything
    else (a stop, the switch moving, a lapsed lease) is the connector's to
    refuse.
    """
    if time.time_ns() >= grant.expires_at_ns:
        return "auth_expired"
    bound = grant.channels.get(address)
    if bound is None:
        return "auth_channel"
    if not bound.holds(value):
        return "auth_range"
    if spent >= grant.steps:
        return "auth_exhausted"
    return None


class PlanAuthorization:
    """One plan run's write authorization: acquired, extended, released."""

    def __init__(self, plan_name: str, owner: str) -> None:
        self.plan_name = plan_name
        self.owner = owner
        self.target: str | None = None
        self.grant: Any = None
        self.spent = 0
        self.extensions = 0
        #: ``True`` once the run was found to need no authorization (switch off).
        self.ungoverned = False
        self._pending: str | None = None
        self._previous_env: str | None = None
        self._stop = threading.Event()
        self._heartbeat: threading.Thread | None = None

    # -- acquiring ------------------------------------------------------------

    def acquire(
        self, ranges: dict[str, list[float]], steps: int, *, label: str
    ) -> Generator[Any, Any, None]:
        """Hold a grant for *ranges* and *steps*: self-issued at ``allowed``, asked for below.

        At ``off`` nothing is issued or asked and :attr:`ungoverned` is set.
        """
        from osprey_connectors import write_authorization as wa

        if self.target is None:
            raise WriteAuthorizationError(
                "target_unknown", f"Plan {self.plan_name!r} writes to no stamped control target."
            )
        if steps > wa.MAX_STEPS:
            raise WriteAuthorizationError(
                "auth_exhausted",
                f"Plan {self.plan_name!r} needs {steps} writes; one authorization allows at most "
                f"{wa.MAX_STEPS}.",
            )
        channels = {address: (low, high) for address, (low, high) in ranges.items()}
        expires_at_ns = time.time_ns() + int(min(EXPIRES_S, wa.MAX_EXPIRES_S - 60) * 1e9)
        try:
            grant = wa.self_issue(
                _section(),
                self.target,
                channels,
                steps=steps,
                expires_at_ns=expires_at_ns,
                lease_s=LEASE_S,
                label=label,
                owner=self.owner,
            )
        except wa.AuthorizationRefused as refused:
            if refused.reason == "switch_off":
                self.ungoverned = True
                return
            if refused.reason != "auth_revoked":
                raise WriteAuthorizationError(refused.reason, str(refused)) from refused
            grant = yield from self._ask(channels, steps, expires_at_ns, label)
        self._hold(grant)

    def _ask(
        self,
        channels: dict[str, tuple[float, float]],
        steps: int,
        expires_at_ns: int,
        label: str,
    ) -> Generator[Any, Any, Any]:
        """File a request and wait for its grant, sleeping through the RunEngine."""
        from bluesky import plan_stubs as bps

        from osprey_connectors import write_authorization as wa

        assert self.target is not None  # acquire() refuses an unstamped target first
        try:
            request = wa.write_request(
                self.target,
                channels,
                steps=steps,
                expires_at_ns=expires_at_ns,
                lease_s=LEASE_S,
                label=label,
                owner=self.owner,
            )
        except wa.AuthorizationRefused as refused:
            raise WriteAuthorizationError(refused.reason, str(refused)) from refused
        self._pending = request.id
        logger.warning(
            "Plan %s awaits approval of write request %s for %s: %d write(s) on %d channel(s) (%s)",
            self.plan_name,
            request.id,
            self.owner,
            steps,
            len(channels),
            label,
        )
        wait_s = _wait_s()
        deadline = time.monotonic() + wait_s
        try:
            while True:
                try:
                    grant = wa.approved_grant(request.id, owner=self.owner)
                except wa.AuthorizationRefused as refused:
                    raise WriteAuthorizationError(refused.reason, str(refused)) from refused
                if grant is not None:
                    return grant
                if time.monotonic() >= deadline:
                    raise WriteAuthorizationError(
                        "auth_timeout",
                        f"Nobody approved write request {request.id} for plan "
                        f"{self.plan_name!r} within {wait_s:.0f} s.",
                    )
                yield from bps.sleep(POLL_S)
        finally:
            self._withdraw()

    def _withdraw(self) -> None:
        from osprey_connectors import write_authorization as wa

        request, self._pending = self._pending, None
        if request is None:
            return
        try:
            wa.withdraw_request(request)
        except Exception:  # a request left behind expires; it never authorizes anything
            logger.warning("Could not withdraw write request %s", request, exc_info=True)

    def _hold(self, grant: Any) -> None:
        """Open the lease, name the grant to this process's writes, keep the lease renewed."""
        from osprey_connectors import write_authorization as wa

        try:
            wa.start_lease(grant.id, owner=self.owner)
        except wa.AuthorizationRefused as refused:
            raise WriteAuthorizationError(refused.reason, str(refused)) from refused
        if self.grant is None:
            self._previous_env = os.environ.get(wa.AUTHORIZATION_ENV_VAR)
        self.grant = grant
        self.spent = 0
        os.environ[wa.AUTHORIZATION_ENV_VAR] = grant.id
        logger.info(
            "Plan %s holds write authorization %s (approved from %s), %d step(s)",
            self.plan_name,
            grant.id,
            grant.approver.surface,
            grant.steps,
        )
        if self._heartbeat is None:
            self._heartbeat = threading.Thread(
                target=self._renew_loop, name=f"write-grant-{self.plan_name}", daemon=True
            )
            self._heartbeat.start()

    def _renew_loop(self) -> None:
        from osprey_connectors import write_authorization as wa

        while not self._stop.wait(LEASE_S / 3):
            grant = self.grant
            if grant is not None and not wa.renew_lease(grant.id, owner=self.owner):
                logger.warning("The lease of write authorization %s could not be renewed", grant.id)

    # -- each write ---------------------------------------------------------

    def before_write(
        self, address: str, value: Any, target: str | None
    ) -> Generator[Any, Any, None]:
        """Make sure the grant covers this write, extending it when it falls short."""
        if self.ungoverned:
            return
        number = _number(address, value)
        if self.grant is None:
            # The survey saw no write: the first one opens the grant.
            self.target = self.target or target
            yield from self.acquire(
                {address: [number, number]},
                EXTRA_STEPS_FLOOR,
                label=f"bluesky {self.plan_name}",
            )
            if self.ungoverned:
                return
        reason = _shortfall(self.grant, self.spent, address, number)
        if reason is not None:
            if self.extensions >= MAX_EXTENSIONS:
                logger.warning(
                    "Plan %s: write authorization %s refuses %s (%s) and the run has used its "
                    "%d extensions",
                    self.plan_name,
                    self.grant.id,
                    address,
                    reason,
                    MAX_EXTENSIONS,
                )
            else:
                yield from self._extend(address, number, reason)
        self.spent += 1

    def _extend(self, address: str, value: float, reason: str) -> Generator[Any, Any, None]:
        """Ask for a grant replacing the current one, wide enough for this write."""
        old = self.grant
        ranges = {name: [bound.min, bound.max] for name, bound in old.channels.items()}
        low, high = ranges.get(address, [value, value])
        ranges[address] = [min(low, value), max(high, value)]
        # The plan has gone where the survey did not: whatever the reason, the
        # rest of it is unknown, so the replacement carries headroom as well as
        # the steps still unspent.
        unspent = max(old.steps - self.spent, 0)
        steps = unspent + max(EXTRA_STEPS_FLOOR, old.steps // 2)
        self.extensions += 1
        label = (
            f"bluesky {self.plan_name}: extends {old.id[:8]} ({reason}; {self.spent} of "
            f"{old.steps} spent; {address} = {value:g})"
        )
        yield from self.acquire(ranges, steps, label=label)

    # -- the end --------------------------------------------------------------

    def release(self) -> None:
        """Stop renewing, withdraw a pending request, and name no grant any more."""
        from osprey_connectors import write_authorization as wa

        self._stop.set()
        self._withdraw()
        if self.grant is not None:
            if self._previous_env is None:
                os.environ.pop(wa.AUTHORIZATION_ENV_VAR, None)
            else:
                os.environ[wa.AUTHORIZATION_ENV_VAR] = self._previous_env


def awaiting_approval() -> dict[str, Any] | None:
    """The write request this lane's worker is waiting on, for the queue summary.

    Read on the bridge's side: the requests directory the deployment binds
    read-only (``OSPREY_WRITE_REQUESTS_DIR``), narrowed to the requests filed
    by this lane's queueserver (:data:`WORKER_HOLDER_ENV`). A worker runs one
    plan at a time and withdraws its request once it is answered or the run
    ends, so a live request is the running plan's. ``None`` when there is none,
    or when the bridge has no directory to read.
    """
    from datetime import UTC, datetime
    from pathlib import Path

    from osprey_connectors import write_authorization as wa

    holder = (os.environ.get(WORKER_HOLDER_ENV) or "").strip()
    directory = (os.environ.get(wa.REQUESTS_DIR_ENV_VAR) or "").strip()
    if not holder or not directory:
        return None
    try:
        requests, _unreadable = wa.list_requests(Path(directory))
    except OSError:
        return None
    now = time.time_ns()
    mine = [r for r in requests if r.holder == holder and r.expires_at_ns > now]
    if not mine:
        return None
    request = min(mine, key=lambda r: r.requested_at_ns)

    def iso(ns: int) -> str:
        return datetime.fromtimestamp(ns / 1e9, tz=UTC).isoformat(timespec="seconds")

    return {
        "id": request.id,
        "owner": request.owner,
        "target": request.target,
        "steps": request.steps,
        "channels": sorted(request.channels),
        "label": request.label,
        "requested_at": iso(request.requested_at_ns),
        "expires_at": iso(request.expires_at_ns),
    }


def run_under_grant(
    plan_name: str, build: Callable[[], Iterable[Any]], plan: Iterable[Any]
) -> Generator[Any, Any, Any]:
    """Run *plan* with every write under a write authorization for this run.

    *build* makes a fresh copy of the same plan for the survey; *plan* is the
    one the RunEngine runs. Must be entered inside the wrapper's
    ``bind_owner``, so the owner is the queue item's.
    """
    from bluesky.preprocessors import plan_mutator, single_gen

    from osprey_connectors.posture_store import NO_OWNER, current_owner

    owner = current_owner()
    if not enabled() or owner is NO_OWNER or not isinstance(owner, str):
        return (yield from plan)
    survey = yield from survey_writes(build)
    if survey.complete and survey.steps == 0:
        return (yield from plan)
    authorization = PlanAuthorization(plan_name, owner)
    authorization.target = survey.target
    try:
        if survey.steps:
            headroom = 0 if survey.complete else EXTRA_STEPS_FLOOR
            yield from authorization.acquire(
                survey.ranges,
                survey.steps + headroom,
                label=f"bluesky {plan_name}: {survey.steps} write(s) seen"
                + (f" + {headroom} headroom" if headroom else ""),
            )

        def check(msg: Any) -> tuple[Any, None]:
            write = _write_of(msg)
            if write is None or authorization.ungoverned:
                return None, None

            def head() -> Generator[Any, Any, Any]:
                yield from authorization.before_write(*write)
                return (yield from single_gen(msg))

            return head(), None

        return (yield from plan_mutator(plan, check))
    finally:
        authorization.release()
