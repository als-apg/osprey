"""Apply: write the pending step, or put back settings an earlier action moved.

An apply body carries exactly one of two shapes, anything else is 400
``bad_request``:

- ``{"step_digest": str}`` writes the pending step that digest names: each
  corrector with ``δ ≠ 0`` is set to its value read just before plus ``δ``.
- ``{"model": str, "values": [{"name": str, "before": value}, ...]}`` is a
  put-back: each named setting is set back to its ``before``. A name is one
  of :func:`putback_names` — an RM column of the service's tool or an address
  of a declared setup setting; ``model`` must be the service's own (409
  ``model_mismatch``). A put-back naming a setting address voids the current
  setup before its first write: the setup no longer holds what it set.
  Nothing about a put-back is stored by the service: its values arrive in the
  call.

The apply runs as a holder call on :class:`~.sessions.Sessions`. Its worker
precondition, composed after the holder check, does only pure in-memory
checks. Any apply is refused while a continuous run is in progress (409
``continuous_running``), before anything else, so a put-back then voids
nothing. For a step: a pending step whose digest is the body's (409
``digest_mismatch``, also when no step is pending), whose shot is the latest
shot (409 ``stale_shot``), not yet consumed (409 ``step_consumed``) and not
all zero (409 ``zero_step``, the detail is the tool's status). A put-back
voids the pending step as its first statement, then checks the model and the
names.

The action, run under the run lock, then refuses the whole apply before the
first write when:

- for a step, the machine moved since its shot (409 ``machine_moved``): the
  action re-reads the shot's setpoint snapshot — the full RM-column corrector
  vector — together with the setup re-check's readbacks in one batch, and a
  setpoint that differs from the snapshot by more than
  :data:`~.acquisition.SNAPSHOT_TOLERANCE` × its ``max_step`` (exactly, for a
  corrector stating none), a shot with no snapshot, or a setup identity other
  than the shot's ``setup_sha`` refuses it. A recompute on the same shot stays
  appliable while the machine is unmoved and the step unconsumed; once a step
  is applied, the correctors it moved refuse every later step of that shot;
- a value to write breaks the limits validator the turn resolved for the
  service target — its ``[min, max]``, its ``max_step`` measured against the
  values just read (for a step, the snapshot batch) — (409
  ``limits_violation``);
- a Bluesky plan runs on a bridge serving the target (409 ``plan_running``),
  or a bridge deployed for the target does not answer (503 ``plan_running``);
- the action's stop event is set (409 ``action_stopped``);
- for a step, the pending step is no longer the exact step object the
  precondition matched (replaced, voided, or consumed by another apply).

Only then is the step marked consumed, and the settings are written one at a
time through :func:`~.calls.guarded_call` with the checked write, the stop
event and the deadline (``action_timeout_s`` from the action's start) checked
between writes. Every write that was not refused is listed in ``moved`` in
put-back shape, ``before`` being the value read just before it and ``after``
the value written, also a write that raised (it may have landed). A stop answers a
:class:`~.state.LastAction` with ``stopped`` set; a write that fails or is
refused, and a deadline passed, raise :class:`ApplyFailed` carrying ``moved``.
A write blocking past the deadline is awaited to its end with the run lock
held, and is listed.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from osprey.services.trajectory.acquisition import SetupCheck, moved_setpoints
from osprey.services.trajectory.actions import ACTION_STOPPED
from osprey.services.trajectory.admit import Admission
from osprey.services.trajectory.calls import guarded_call, read_value
from osprey.services.trajectory.refusal import Refused
from osprey.services.trajectory.state import LastAction, Moved, SettingValue, ShotResult, Step
from osprey.services.trajectory.step import TrajectoryTool, max_steps, model_of

if TYPE_CHECKING:
    from osprey.services.trajectory.sessions import Sessions
    from osprey.services.trajectory.setup import DeclaredSetting, SetupKeeper
    from osprey.services.trajectory.startup import TrajectoryStartup

__all__ = [
    "APPLY",
    "BAD_REQUEST",
    "CONTINUOUS_RUNNING",
    "DIGEST_MISMATCH",
    "LIMITS_VIOLATION",
    "MACHINE_MOVED",
    "MODEL_MISMATCH",
    "PLAN_RUNNING",
    "PUT_BACK",
    "STALE_SHOT",
    "STEP_CONSUMED",
    "ZERO_STEP",
    "ApplyFailed",
    "ApplyRefused",
    "PlanProbe",
    "PutBack",
    "StepApply",
    "apply",
    "bridge_plan_running",
    "parse_body",
    "putback_names",
    "settings_addresses",
]

#: The name an apply carries in the service state.
APPLY = "apply"
#: The name a put-back carries in the service state.
PUT_BACK = "put_back"

BAD_REQUEST = "bad_request"
MODEL_MISMATCH = "model_mismatch"
DIGEST_MISMATCH = "digest_mismatch"
STALE_SHOT = "stale_shot"
STEP_CONSUMED = "step_consumed"
ZERO_STEP = "zero_step"
LIMITS_VIOLATION = "limits_violation"
PLAN_RUNNING = "plan_running"
CONTINUOUS_RUNNING = "continuous_running"
MACHINE_MOVED = "machine_moved"

#: The HTTP status each refusal raised here answers with; a bridge that does
#: not answer turns ``plan_running`` into a 503 (:meth:`ApplyRefused.bridge_silent`).
_STATUS = {
    BAD_REQUEST: 400,
    MODEL_MISMATCH: 409,
    DIGEST_MISMATCH: 409,
    STALE_SHOT: 409,
    STEP_CONSUMED: 409,
    ZERO_STEP: 409,
    LIMITS_VIOLATION: 409,
    PLAN_RUNNING: 409,
    ACTION_STOPPED: 409,
    CONTINUOUS_RUNNING: 409,
    MACHINE_MOVED: 409,
}

#: Seconds one bridge gets to answer whether a plan runs.
_BRIDGE_TIMEOUT_S = 2.0

#: Answers whether a Bluesky plan runs on a bridge serving the target: ``True``
#: or ``False``, ``None`` when a bridge deployed for the target does not answer.
PlanProbe = Callable[[str], Awaitable[bool | None]]


class ApplyRefused(Refused):
    """An apply refused whole; nothing was written."""

    statuses = _STATUS

    @classmethod
    def bridge_silent(cls, detail: str) -> ApplyRefused:
        """``plan_running`` answered 503: a bridge deployed for the target did not answer."""
        refused = cls(PLAN_RUNNING, detail)
        refused.status = 503
        return refused


class ApplyFailed(Exception):
    """An apply stopped part-way by a failed or refused write, or by its deadline.

    Attributes:
        moved: Every write that was not refused, in order, for a put-back.
        reason: What ended the apply.
    """

    def __init__(self, reason: str, moved: list[Moved]) -> None:
        super().__init__(f"apply failed: {reason}")
        self.reason = reason
        self.moved = moved


# -- the body ------------------------------------------------------------------------


@dataclass(frozen=True)
class StepApply:
    """An apply of the pending step its digest names."""

    digest: str


@dataclass(frozen=True)
class PutBack:
    """A put-back: each name set back to its ``before``, in the order given."""

    model: str
    values: tuple[tuple[str, SettingValue], ...]


_STEP_KEYS = frozenset({"step_digest"})
_PUTBACK_KEYS = frozenset({"model", "values"})
_VALUE_KEYS = frozenset({"name", "before"})


def _bad(detail: str) -> ApplyRefused:
    return ApplyRefused(BAD_REQUEST, detail)


def _setting(value: Any) -> bool:
    if isinstance(value, bool | int | str):
        return True
    return isinstance(value, float) and math.isfinite(value)


def parse_body(body: Any) -> StepApply | PutBack:
    """The apply a body asks for.

    Raises:
        ApplyRefused: ``bad_request`` (400) unless the body is exactly
            ``{"step_digest": str}`` or ``{"model": str, "values": [...]}``
            with one ``{"name", "before"}`` per distinct name.
    """
    if not isinstance(body, Mapping):
        raise _bad("an apply body is an object")
    keys = frozenset(body)
    if keys == _STEP_KEYS:
        digest = body["step_digest"]
        if not isinstance(digest, str) or not digest:
            raise _bad("step_digest is the digest of the pending step, a non-empty string")
        return StepApply(digest=digest)
    if keys == _PUTBACK_KEYS:
        model = body["model"]
        values = body["values"]
        if not isinstance(model, str) or not model:
            raise _bad("model is the service's model, a non-empty string")
        if not isinstance(values, list) or not values:
            raise _bad("values lists at least one {name, before} to put back")
        pairs: list[tuple[str, SettingValue]] = []
        seen: set[str] = set()
        for entry in values:
            if not isinstance(entry, Mapping) or frozenset(entry) != _VALUE_KEYS:
                raise _bad("each value is exactly {name, before}")
            name, before = entry["name"], entry["before"]
            if not isinstance(name, str) or not name:
                raise _bad("a value's name is a non-empty string")
            if not _setting(before):
                raise _bad(f"{name}: before is a finite number, a string or a boolean")
            if name in seen:
                raise _bad(f"{name} is listed twice")
            seen.add(name)
            pairs.append((name, before))
        return PutBack(model=model, values=tuple(pairs))
    raise _bad(
        "an apply body carries exactly one of {step_digest} or {model, values}; "
        f"got {sorted(str(key) for key in keys)}"
    )


# -- what a put-back may name -----------------------------------------------------


def putback_names(
    startup: TrajectoryStartup, declared: Sequence[DeclaredSetting] = ()
) -> frozenset[str]:
    """Every name a put-back may set: the tool's RM columns and every setting address.

    Args:
        startup: The started service.
        declared: The model's declared setup settings.
    """
    names: set[str] = set()
    tool = startup.tool
    if isinstance(tool, TrajectoryTool):
        names.update(tool.layout.hcorr_names)
        names.update(tool.layout.vcorr_names)
    return frozenset(names | settings_addresses(declared))


def settings_addresses(declared: Sequence[DeclaredSetting]) -> frozenset[str]:
    """The address each declared setting writes: a device setting's over the whole
    device table, a global setting's once."""
    addresses: set[str] = set()
    for setting in declared:
        if isinstance(setting.setting, Mapping):
            addresses.update(setting.setting.values())
        else:
            addresses.add(setting.setting)
    return frozenset(addresses)


# -- whether a plan runs ---------------------------------------------------------


async def bridge_plan_running(target: str) -> bool | None:
    """Whether a Bluesky plan runs on a bridge deployed for *target*.

    A bridge is deployed for *target* when the config carries its lane's
    ``services.<lane>`` block and the lane serves *target* (a lane block that
    declares no target serves the deployment's only one). With none deployed no
    plan can run. Each deployed bridge is asked ``GET /queue``; a manager in a
    running state is a running plan.

    Returns:
        ``True`` or ``False``; ``None`` when a deployed bridge does not answer.
    """
    import httpx

    from osprey.bluesky_bridge_connection import (
        LANE_KEYS,
        UnknownBlueskyLaneError,
        lane_declared_target,
        resolve_bridge_url,
    )
    from osprey.services.bluesky_bridge.queue_backend import is_queue_active
    from osprey_connectors.workspace import load_osprey_config

    try:
        config = load_osprey_config()
    except Exception:  # an unreadable config names no bridge to ask
        return None
    services = config.get("services") if isinstance(config, dict) else None
    lanes = [
        lane
        for lane in LANE_KEYS
        if isinstance(services, dict)
        and isinstance(services.get(lane), dict)
        and lane_declared_target(lane, config) in (None, target)
    ]
    running = False
    async with httpx.AsyncClient(timeout=_BRIDGE_TIMEOUT_S) as client:
        for lane in lanes:
            try:
                response = await client.get(f"{resolve_bridge_url(lane)}/queue")
                response.raise_for_status()
                status = response.json().get("status")
            except (httpx.HTTPError, UnknownBlueskyLaneError, ValueError, AttributeError):
                return None
            if not isinstance(status, dict) or status.get("available") is False:
                return None
            running = running or is_queue_active(status)
    return running


# -- the apply ---------------------------------------------------------------------


@dataclass(frozen=True)
class _Write:
    name: str
    value: Any
    before: Any


async def apply(
    sessions: Sessions,
    owner: Any,
    kind: str,
    client_id: str | None,
    body: Any,
    *,
    plan_running: PlanProbe = bridge_plan_running,
    setup_check: SetupCheck | None = None,
    setup: SetupKeeper | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> LastAction:
    """Apply the pending step or put settings back, as the holder's worker call.

    Args:
        sessions: The started service's sessions.
        owner: The admitted caller.
        kind: The door the call came through.
        client_id: The caller's session token.
        body: The apply body.
        plan_running: Asks whether a Bluesky plan runs for the target.
        setup_check: The current setup's re-check; its readbacks join the
            snapshot batch and the identity it answers must be the shot's
            ``setup_sha``. Without one the current setup identity is ``None``.
        setup: The current setup's keeper; its declared settings join what a
            put-back may name, and a put-back naming one voids its setup.
        clock: Seconds, monotonic; bounds the writes by ``action_timeout_s``.

    Returns:
        A :class:`~.state.LastAction` named ``apply`` or ``put_back`` listing what moved;
        ``stopped`` when the stop event ended it between writes.

    Raises:
        ApplyRefused: Refused whole; nothing was written.
        ApplyFailed: A write failed or was refused, or the deadline passed,
            after the writes in its ``moved``.
        SessionRefused: The caller does not hold the target.
        StepRefused: ``tool_unavailable`` when the service has no tool to
            name its model.
        AdmitRefused, ActionRefused: As any worker action.
    """
    asked = parse_body(body)
    name = APPLY if isinstance(asked, StepApply) else PUT_BACK
    startup = sessions.layer.startup
    declared = setup.declared if setup is not None else ()
    matched: list[tuple[Step, ShotResult | None]] = []

    def check_step(asked: StepApply) -> None:
        step = sessions.step.step
        if step is None:
            raise ApplyRefused(DIGEST_MISMATCH, "no step is pending; take a shot to compute one")
        if step.digest != asked.digest:
            raise ApplyRefused(
                DIGEST_MISMATCH,
                "the digest is not the pending step's; read the step again before applying",
            )
        latest = sessions.latest_shot_id
        if latest is not None and step.shot_id != latest:
            raise ApplyRefused(
                STALE_SHOT,
                f"the pending step is of shot {step.shot_id!r}, and shot {latest!r} is newer",
            )
        if sessions.step.consumed:
            raise ApplyRefused(STEP_CONSUMED, "the pending step was already applied")
        if step.all_zero:
            raise ApplyRefused(ZERO_STEP, step.status or str(step.reason))
        matched.append((step, sessions.step.shot))

    def check_putback(asked: PutBack) -> None:
        sessions.step.void()
        model = model_of(startup)
        if asked.model != model:
            raise ApplyRefused(
                MODEL_MISMATCH,
                f"the put-back is for model {asked.model!r}; this service drives {model!r}",
            )
        allowed = putback_names(startup, declared)
        unknown = [name for name, _ in asked.values if name not in allowed]
        if unknown:
            raise ApplyRefused(
                BAD_REQUEST,
                f"a put-back sets RM columns and setup settings only; not {', '.join(unknown)}",
            )

    async def action(_admission: Admission, stop: asyncio.Event) -> LastAction:
        deadline = clock() + startup.settings.action_timeout_s
        connector = startup.connector
        if isinstance(asked, StepApply):
            step, shot = matched[-1]
            planned = [(name, delta) for name, delta in step.deltas.items() if delta != 0]
            current = await _unmoved(
                connector,
                shot,
                [name for name, _ in planned],
                setup_check,
                call_timeout_s=sessions.call_timeout_s,
            )
            writes = [
                _Write(name, _increment(name, current[name], delta), current[name])
                for name, delta in planned
            ]
        else:
            step = None
            current = {name: await read_value(connector, name) for name, _ in asked.values}
            writes = [_Write(name, value, current[name]) for name, value in asked.values]
        _check_limits(getattr(connector, "_limits_validator", None), writes, current)
        await _check_plans(plan_running, startup.target)
        if stop.is_set():
            raise ApplyRefused(ACTION_STOPPED, "the apply was stopped before its first write")
        if step is not None and not sessions.step.consume(step):
            if sessions.step.step is step:
                raise ApplyRefused(STEP_CONSUMED, "the pending step was already applied")
            raise ApplyRefused(
                DIGEST_MISMATCH, "the step was replaced or voided before the apply ran"
            )
        if setup is not None:
            held = settings_addresses(declared)
            touched = [write.name for write in writes if write.name in held]
            if touched:
                await setup.void(
                    f"The setup was voided: a put-back set {', '.join(touched[:5])}"
                    f"{' and more' if len(touched) > 5 else ''}."
                )
        return await _write_all(connector, writes, stop, clock, deadline, name)

    def precondition(_admission: Admission) -> None:
        if sessions.continuous_running:
            raise ApplyRefused(
                CONTINUOUS_RUNNING,
                "a continuous run is in progress; stop it before applying",
            )
        if isinstance(asked, StepApply):
            check_step(asked)
        else:
            check_putback(asked)

    return await sessions.call(owner, kind, client_id, action, precondition=precondition, name=name)


async def _unmoved(
    connector: Any,
    shot: ShotResult | None,
    planned: list[str],
    setup_check: SetupCheck | None,
    *,
    call_timeout_s: float,
) -> dict[str, Any]:
    """Re-read *shot*'s setpoint snapshot in one batch and refuse when the machine moved.

    Returns:
        The value just read of every snapshot and planned name.

    Raises:
        ApplyRefused: ``machine_moved`` when the shot carries no snapshot, a
            setpoint no longer matches it, or the setup identity changed.
    """
    snapshot = shot.setpoints if shot is not None else None
    if shot is None or snapshot is None:
        raise ApplyRefused(
            MACHINE_MOVED,
            "the step's shot carries no setpoint snapshot to check the machine against; "
            "take a new shot",
        )
    setup = setup_check.addresses() if setup_check is not None else []
    addresses = list(dict.fromkeys([*snapshot, *planned, *setup]))
    values: Mapping[str, Any] = await guarded_call(
        lambda: connector.read_multiple_channels(addresses, timeout=call_timeout_s),
        record=lambda _: None,
    )
    current = {
        name: getattr(values[name], "value", None) if name in values else None
        for name in [*snapshot, *planned]
    }
    bounds = max_steps(getattr(connector, "_limits_validator", None))
    moved = moved_setpoints(snapshot, current, bounds)
    if moved:
        raise ApplyRefused(
            MACHINE_MOVED,
            f"nothing was written: {', '.join(moved)} moved since shot {shot.shot_id!r}; "
            "take a new shot",
        )
    identity = await setup_check.check(values) if setup_check is not None else None
    if identity != shot.setup_sha:
        raise ApplyRefused(
            MACHINE_MOVED,
            f"nothing was written: the setup changed since shot {shot.shot_id!r}; take a new shot",
        )
    return current


def _increment(name: str, before: Any, delta: float) -> float:
    if isinstance(before, bool) or not isinstance(before, int | float) or not math.isfinite(before):
        raise ApplyRefused(
            LIMITS_VIOLATION, f"{name} reads {before!r}; a step adds to a finite number"
        )
    return float(before) + delta


def _check_limits(validator: Any, writes: list[_Write], current: Mapping[str, Any]) -> None:
    """Refuse the whole apply when any value breaks the turn's limits validator."""
    if validator is None:  # the turn ran with limits stated off
        return
    from osprey_connectors.errors import ChannelLimitsViolationError

    for write in writes:
        try:
            validator.validate(write.name, write.value, read_current=lambda a: current.get(a))
        except ChannelLimitsViolationError as exc:
            raise ApplyRefused(
                LIMITS_VIOLATION, f"nothing was written: {write.name} = {write.value!r}: {exc}"
            ) from exc


async def _check_plans(plan_running: PlanProbe, target: str) -> None:
    running = await plan_running(target)
    if running is None:
        raise ApplyRefused.bridge_silent(
            f"the Bluesky bridge deployed for target {target!r} does not answer, so a "
            "running plan cannot be ruled out"
        )
    if running:
        raise ApplyRefused(
            PLAN_RUNNING, f"a Bluesky plan runs on target {target!r}; apply after it ends"
        )


async def _write_all(
    connector: Any,
    writes: list[_Write],
    stop: asyncio.Event,
    clock: Callable[[], float],
    deadline: float,
    name: str = APPLY,
) -> LastAction:
    moved: list[Moved] = []
    for index, write in enumerate(writes):
        if index:
            if stop.is_set():
                return LastAction(name=name, stopped=True, moved=moved)
            if clock() >= deadline:
                raise ApplyFailed(
                    f"action_timeout_s passed after {index} of {len(writes)} writes", moved
                )

        try:
            await _write_one(connector, write, moved)
        except Exception as exc:
            raise ApplyFailed(f"{write.name}: {exc}", moved) from exc
    return LastAction(name=name, stopped=False, moved=moved)


async def _write_one(connector: Any, write: _Write, moved: list[Moved]) -> None:
    """Write one setting with the checked write; list it in *moved* unless it was refused."""
    from osprey_connectors.errors import ChannelWriteBlockedError

    def record(outcome: Any) -> None:
        if isinstance(outcome, ChannelWriteBlockedError):
            return
        if isinstance(outcome, BaseException):
            # The write may have landed; it is listed so it can be put back.
            word = getattr(outcome, "outcome", None) or "failed"
        else:
            word = outcome.outcome
        moved.append(
            Moved(name=write.name, before=write.before, after=write.value, outcome=str(word))
        )

    await guarded_call(
        lambda: connector.write_channel_checked(write.name, write.value), record=record
    )
