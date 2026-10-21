"""The trajectory service's one gated app: a panel bundle and two doors.

One FastAPI app behind the shared web gate (:func:`configure_interface_app`:
``WebAuthMiddleware`` admits on the terminal secret, a roster secret or the
panel token, and stamps the owner the credential names; ``HttpAuditMiddleware``
sits innermost):

- ``/panel/`` — the panel bundle;
- ``/panel/api/*`` — the **panel door**, kind ``panel``;
- ``/api/*`` — the **agent door**, kind ``agent``, which also refuses a caller
  without this service's launch token (``TRAJECTORY_LAUNCH_TOKEN``);
- ``GET /health`` — exempt from the gate; asks the target live.

Both doors serve ``POST session``, ``POST takeover``, ``DELETE session``,
``GET state``, ``POST stop``, ``POST shot``, ``GET step`` and ``POST apply``.
The panel door alone also serves ``POST continuous`` and
``POST continuous/heartbeat``, the references (``GET``/``POST references``,
``GET references/{name}``, ``POST references/select``), ``POST solve``,
``GET shots``, ``GET shots/{shot_id}/raw``, ``GET``/``POST setup`` and the state
stream ``GET events``. The door is
the route, never the request; the
owner is the account the gate minted from the credential that admitted the
request, read through :func:`owner_from_header` on both doors — a name it
refuses is no owner, and an action with no owner is refused by ``admit``. The
panel door never checks the launch token.

A holder's call names its session in the ``X-Trajectory-Client`` header. A
shot runs the declared chain (:mod:`~.acquisition`): the panel door's body is a
:class:`~.state.ChainBody` naming the ticked steps, the readout mode, turns,
shots and averaging, refused 409 ``setup_required`` without a current setup,
and answered with a :class:`~.state.ChainResult`; the agent door's body
chooses only ``n_shots`` and ``average``, runs the declared default chain, is
never refused for setup, and is answered with the chain's shot. A continuous
run starts with the panel door's chain body. Shots go through
:class:`Sessions`, which binds the post-shot hook that computes the step.
The header's machine readings (the declaration's RF-on and stored-current
readbacks) are read in one batch outside the run slot after a chain's last
step, on open, take-over and heartbeat, and periodically while a state stream
is open. ``GET step`` reads the pending step with only the door owner: no session,
never queued behind the worker. Stop answers 202 at once with the stop budget.

``POST solve`` is the holder's SolveConfig update: its body is a
:class:`~.state.SolveConfigPartial` (``keep_live``, never a discard count; an
empty body is Calculate RM), written through the one writer
(:mod:`~.solve_config`) inside a holder call, and answered with the published
solve and the pending step.

``POST setup`` is the holder's setup (:mod:`~.setup`): its body is a
:class:`~.state.SetupBody`, a partial update of the declared settings plus the
BPM exclusions it hands the SolveConfig writer, run as one holder action
(refused 409 ``continuous_running`` during a run) and answered with a
:class:`~.state.SetupResult`; the last action it publishes is ``setup`` with
what it moved. ``GET setup`` answers the declared settings as read back, a
:class:`~.state.SetupReadbacks`. The current setup is persisted per target and
re-checked at start and before every inject and readout turn of a chain on
both doors; a divergence voids the setup and the step with a notice.

``GET shots/{shot_id}/raw`` answers the shot's raw buffer file
(:mod:`~.raw`, ``application/octet-stream``); a derived shot answers its root
shot's. The store keeps the snapshot's ``retention`` root shots with their
buffers, so a shot pruned past it is refused 410 ``raw_evicted``. The snapshot's ``solve`` is that published solve on
both doors, and its ``notices`` name the overrides the load dropped.

A refusal is ``HTTPException(detail={code, detail})``. Each writing action is
recorded once, by the route, on :data:`SURFACE_TRAJECTORY_SERVICE` with the
owner as actor: a control-path action (take-over, release, Stop, continuous
start) as it is decided at admission, a worker action (open, shot,
apply, reference select, solve, setup) after the worker answered, with what it moved. A
release is never admitted, so its record carries only the owner, the target
and the ``client_id``.

``GET state`` answers a :class:`~.state.Snapshot`; ``GET events`` streams
:mod:`~osprey.services.trajectory.state` events, a snapshot first and a fresh
snapshot after every change. A stream ends only on the service-shutdown signal
(``app.state.shutdown_signal``, set once by :meth:`Sessions.shutdown`), never on
the holder stop event, so Stop and a take-over reach every open stream as events.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from osprey.interfaces._app_setup import configure_interface_app
from osprey.interfaces.common_middleware import OWNER_STATE_KEY
from osprey.services.trajectory import reach
from osprey.services.trajectory.acquisition import (
    CHAIN_FAILED,
    SETUP_REQUIRED,
    AcquisitionRefused,
    ChainDeclaration,
    ChainFailed,
    ChainRun,
    ShotRows,
    ShotStopped,
    chain_action,
    chain_declaration,
    chain_run,
    default_run,
    makes_step,
    n_shots_max,
)
from osprey.services.trajectory.actions import ACTION_STOPPED, ActionLayer
from osprey.services.trajectory.admit import (
    NO_OPERATOR_IDENTITY,
    Admission,
    AdmitGate,
    AdmitRefused,
)
from osprey.services.trajectory.apply import ApplyFailed, apply
from osprey.services.trajectory.calls import guarded_call
from osprey.services.trajectory.constants import (
    AGENT_DOOR,
    CLIENT_HEADER,
    PANEL_DOOR,
    connector_call_timeout_s,
)
from osprey.services.trajectory.extraction import AVERAGE_MODES
from osprey.services.trajectory.raw import RAW_EVICTED, RAW_MEDIA_TYPE, RawEvicted, encode_raw
from osprey.services.trajectory.reach import REACHABLE, TRAJECTORY_UNAVAILABLE
from osprey.services.trajectory.refusal import Refused
from osprey.services.trajectory.sessions import (
    BAD_REQUEST,
    CONTINUOUS_RUNNING,
    OWNER_IN_BODY,
    SessionRefused,
    Sessions,
)
from osprey.services.trajectory.setup import (
    SETUP,
    SETUP_FAILED,
    SetupFailed,
    SetupKeeper,
    check_body,
    declared_settings,
    setup_action,
)
from osprey.services.trajectory.startup import StartupRefused, TrajectoryStartup
from osprey.services.trajectory.state import (
    ApplyEvent,
    Catalog,
    ChainBody,
    ChainResult,
    ContinuousEndedEvent,
    HolderView,
    LastAction,
    LimitsPosture,
    Machine,
    Mode,
    Readout,
    ReferenceImport,
    ReferenceLoad,
    SetupBody,
    SetupState,
    ShotEvent,
    ShotResult,
    ShotSummary,
    Snapshot,
    SnapshotEvent,
    SolveConfigPartial,
    Step,
    StepEvent,
    StoppedEvent,
    TakenOver,
    TakenOverEvent,
)
from osprey.services.trajectory.step import TOOL_UNAVAILABLE, StepRefused, require_tool
from osprey.services.trajectory.store import (
    ForeignReference,
    InvalidName,
    NoMatchingBpms,
    ReferenceNotFound,
    StoreUnconfigured,
    TrajectoryStore,
)
from osprey.utils.launch_token import LAUNCH_TOKEN_HEADER, verify_launch_token
from osprey.utils.owner_header import owner_from_header

if TYPE_CHECKING:
    from osprey.services.trajectory.acquisition import PostShotHook
    from osprey.services.trajectory.actions import Action
    from osprey.services.trajectory.solve_config import SolveConfigStore

logger = logging.getLogger(__name__)

__all__ = [
    "AGENT_DOOR",
    "CLIENT_HEADER",
    "LAUNCH_TOKEN_ENV",
    "PANEL_DOOR",
    "create_app",
    "shot_rows",
]

#: The environment variable that holds the agent door's launch token.
LAUNCH_TOKEN_ENV = "TRAJECTORY_LAUNCH_TOKEN"

#: The panel bundle, served under ``/panel/``.
_PANEL_DIR = Path(__file__).parent / "panel"

#: The posture an admitted action acts under; a refusal is filed under the
#: posture that held it back.
_POSTURE_WRITES = "writes"
_POSTURE_SANDBOX = "sandbox"
_POSTURE_READ_ONLY = "read_only"
_SANDBOX_CODES = frozenset({"narrowing", "writes_disabled"})

#: Seconds between two keep-alive comments on an idle state stream.
_HEARTBEAT_S = 15.0
#: Seconds between two machine reads while a state stream is open.
_MACHINE_REFRESH_S = 5.0
#: Frames a slow stream may fall behind by before frames are dropped for it.
_STREAM_DEPTH = 256
_KEEPALIVE = ": keep-alive\n\n"

# The codes the routes themselves refuse with.
NO_STEP = "no_step"
NO_SHOT = "no_shot"
STORE_UNAVAILABLE = "store_unavailable"
INVALID_NAME = "invalid_name"
REFERENCE_NOT_FOUND = "reference_not_found"
SHOT_NOT_FOUND = "shot_not_found"
NO_MATCHING_BPMS = "no_matching_bpms"
APPLY_FAILED = "apply_failed"
SHOT_NOT_FRESH = "shot_not_fresh"
READOUT_MISMATCH = "readout_mismatch"
REFERENCE_FOREIGN = "reference_foreign"
UNKNOWN_BPM = "unknown_bpm"
READING_NOT_FINITE = "reading_not_finite"
REFERENCE_TOO_LARGE = "reference_too_large"

#: The largest reference import body accepted, in bytes.
MAX_REFERENCE_IMPORT_BYTES = 1 << 20
#: The readout mode a reference saved from a shot is taken in: only a fresh
#: TbT shot becomes a reference.
_SAVED_READOUT: Readout = "tbt"

StartFn = Callable[[], Awaitable[TrajectoryStartup]]


class RouteRefused(Refused):
    """A request a route refuses itself, or an action that failed part-way."""

    statuses = {
        NO_STEP: 404,
        NO_SHOT: 409,
        STORE_UNAVAILABLE: 503,
        INVALID_NAME: 400,
        REFERENCE_NOT_FOUND: 404,
        SHOT_NOT_FOUND: 404,
        RAW_EVICTED: 410,
        NO_MATCHING_BPMS: 409,
        APPLY_FAILED: 502,
        SHOT_NOT_FRESH: 409,
        READOUT_MISMATCH: 409,
        REFERENCE_FOREIGN: 409,
        UNKNOWN_BPM: 400,
        READING_NOT_FINITE: 400,
        REFERENCE_TOO_LARGE: 413,
        CHAIN_FAILED: 502,
        ACTION_STOPPED: 409,
        SETUP_FAILED: 502,
    }


# -- dependencies --------------------------------------------------------------


async def door_owner(request: Request) -> str | None:
    """The owner the gate minted for this request, or ``None``.

    Read from the name ``WebAuthMiddleware`` stamped from the credential that
    admitted the request; a name :func:`owner_from_header` refuses is no owner.
    """
    return owner_from_header(getattr(request.state, OWNER_STATE_KEY, None))


async def _launch_token(request: Request) -> None:
    """Refuse an agent-door request that lacks this service's launch token."""
    verify_launch_token(request.headers.get(LAUNCH_TOKEN_HEADER), LAUNCH_TOKEN_ENV)


def _sessions(request: Request) -> Sessions:
    sessions: Sessions = request.app.state.sessions
    return sessions


def _feed(request: Request) -> _Feed:
    feed: _Feed = request.app.state.feed
    return feed


def _store(request: Request) -> TrajectoryStore:
    store: TrajectoryStore | None = request.app.state.store
    if store is None:
        raise RouteRefused(
            STORE_UNAVAILABLE,
            "the service has no state directory (OSPREY_TRAJECTORY_STATE_DIR), so it keeps "
            "no shot files and no references",
        )
    return store


def _client_id(request: Request) -> str:
    """The caller's session token, from :data:`CLIENT_HEADER`."""
    client_id = request.headers.get(CLIENT_HEADER, "").strip()
    if not client_id:
        raise SessionRefused(BAD_REQUEST, f"the call names no session; send {CLIENT_HEADER}")
    return client_id


async def _body(request: Request) -> Mapping[str, Any] | None:
    raw = await request.body()
    if not raw.strip():
        return None
    try:
        body: Mapping[str, Any] | None = json.loads(raw)
    except ValueError as exc:
        raise SessionRefused(BAD_REQUEST, "the request body is not JSON") from exc
    return body


def _fields(body: Any, allowed: frozenset[str]) -> Mapping[str, Any]:
    """*body*'s fields, refusing an owner or any field outside *allowed*."""
    if body is None:
        return {}
    if not isinstance(body, Mapping):
        raise SessionRefused(BAD_REQUEST, "the request body is a JSON object")
    if "owner" in body:
        raise SessionRefused(
            OWNER_IN_BODY, "the owner is the identity the request arrived with, never a body field"
        )
    extra = sorted(set(body) - allowed)
    if extra:
        raise SessionRefused(
            BAD_REQUEST,
            f"unknown fields {', '.join(map(repr, extra))}; this call takes "
            f"{', '.join(sorted(allowed)) or 'none'}",
        )
    return body


def _count(fields: Mapping[str, Any], key: str, default: int | None) -> int | None:
    value = fields.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SessionRefused(BAD_REQUEST, f"{key} is a whole number of at least 1, not {value!r}")
    return int(value)


def _shot_args(fields: Mapping[str, Any]) -> tuple[int, str]:
    n_shots = _count(fields, "n_shots", 1)
    assert n_shots is not None
    average = fields.get("average", "mean")
    if average not in AVERAGE_MODES:
        raise SessionRefused(
            BAD_REQUEST, f"average is one of {', '.join(AVERAGE_MODES)}, not {average!r}"
        )
    return n_shots, average


def _require_owner(owner: str | None) -> str:
    if owner is None:
        raise AdmitRefused(NO_OPERATOR_IDENTITY, "the request names no operator")
    return owner


def _model(startup: TrajectoryStartup) -> str:
    commands = startup.commands
    if commands is None:
        raise AcquisitionRefused(
            TRAJECTORY_UNAVAILABLE, "the build has no trajectory model, so it has no shots"
        )
    return commands.model


def shot_rows(startup: TrajectoryStartup) -> ShotRows:
    """The row layout a shot reads: the tool's monitors, rowed by the device table.

    The response matrix's x rows and y rows read the same monitors in the same
    order (the build refuses any other); each monitor's x row names the device
    of the declaration's table whose y and sum rows the shot reads with it. A
    monitor is named by its x row, the name the tool's layout gives it.

    Raises:
        AcquisitionRefused: ``trajectory_unavailable`` without a trajectory model.
        StepRefused: ``tool_unavailable`` without a tool, or with a monitor the
            device table states no device for.
    """
    _model(startup)
    commands = startup.commands
    assert commands is not None  # _model refuses a service without one
    tool = require_tool(startup.tool)
    by_x = {device.x_row: device for device in commands.devices}
    x = tuple(tool.layout.bpm_names_h)
    unknown = [row for row in x if row not in by_x]
    if unknown:
        raise StepRefused(
            TOOL_UNAVAILABLE,
            f"model {commands.model!r}: the response matrix reads {', '.join(unknown)}, "
            "which no device of the trajectory declaration carries",
        )
    devices = [by_x[row] for row in x]
    return ShotRows(
        bpms=x,
        x=x,
        y=tuple(device.y_row for device in devices),
        sum=tuple(device.sum_row for device in devices),
    )


# -- audit ---------------------------------------------------------------------


def _record(
    *,
    owner: str | None,
    subject: str,
    refusal: Refused | None,
    detail: Mapping[str, Any],
) -> None:
    """File the one record of a writing action; the action never fails on it."""
    from osprey.audit.dedup import record_and_mark
    from osprey.audit.envelope import (
        DECISION_ALLOWED,
        DECISION_REFUSED,
        POSTURE_SOURCE_APP,
        SURFACE_TRAJECTORY_SERVICE,
    )

    code = refusal.code if refusal is not None else None
    fields: dict[str, Any] = {
        "surface": SURFACE_TRAJECTORY_SERVICE,
        "posture": _POSTURE_SANDBOX if code in _SANDBOX_CODES else _POSTURE_WRITES,
        "posture_source": POSTURE_SOURCE_APP,
        "session": None,
        "subject": subject,
        "detail": " ".join(f"{key}={value}" for key, value in detail.items()),
    }
    if owner:
        fields["actor"] = owner
    try:
        record_and_mark(
            decision=DECISION_ALLOWED if refusal is None else DECISION_REFUSED,
            reason=subject if code is None else code,
            **fields,
        )
    except Exception:  # the audit trail degrades; the action does not
        logger.warning("Could not record the trajectory %s for audit", subject, exc_info=True)


def _moved(action: LastAction | list[Any]) -> str:
    """What an action moved, as one compact JSON list for its record."""
    moved = action.moved if isinstance(action, LastAction) else action
    return json.dumps([item.model_dump(mode="json") for item in moved], separators=(",", ":"))


@contextmanager
def _refusals_recorded(
    owner: str | None, subject: str, presented: Mapping[str, Any]
) -> Iterator[None]:
    """Record a refusal raised in the body under *presented*, and answer it as the door does."""
    try:
        yield
    except Refused as exc:
        _record(owner=owner, subject=subject, refusal=exc, detail=presented)
        raise HTTPException(status_code=exc.status, detail=exc.as_detail()) from exc


@contextmanager
def _answered() -> Iterator[None]:
    """Answer a refusal of a reading route as the door does; nothing is recorded."""
    try:
        yield
    except Refused as exc:
        raise HTTPException(status_code=exc.status, detail=exc.as_detail()) from exc


@contextmanager
def _store_refusals() -> Iterator[None]:
    """The store's errors as the refusals a door answers with."""
    try:
        yield
    except StoreUnconfigured as exc:
        raise RouteRefused(STORE_UNAVAILABLE, str(exc)) from exc
    except InvalidName as exc:
        raise RouteRefused(INVALID_NAME, str(exc)) from exc
    except NoMatchingBpms as exc:
        raise RouteRefused(NO_MATCHING_BPMS, str(exc)) from exc
    except ForeignReference as exc:
        raise RouteRefused(REFERENCE_FOREIGN, str(exc)) from exc
    except ReferenceNotFound as exc:
        raise RouteRefused(REFERENCE_NOT_FOUND, str(exc)) from exc
    except RawEvicted as exc:
        raise RouteRefused(RAW_EVICTED, str(exc)) from exc
    except LookupError as exc:
        raise RouteRefused(SHOT_NOT_FOUND, str(exc)) from exc
    except ValueError as exc:
        raise SessionRefused(BAD_REQUEST, str(exc)) from exc


def _failed(code: str, reason: str, moved: list[Any]) -> HTTPException:
    return HTTPException(
        status_code=RouteRefused.statuses[code],
        detail={
            "code": code,
            "detail": reason,
            "moved": [item.model_dump(mode="json") for item in moved],
        },
    )


# -- the state view and its stream ------------------------------------------------


class _Subscriber:
    """One open state stream: whose it is and the frames waiting for it."""

    def __init__(self, owner: str | None) -> None:
        self.owner = owner
        self.frames: asyncio.Queue[str] = asyncio.Queue(maxsize=_STREAM_DEPTH)


def _frame(event: BaseModel) -> str:
    """One SSE frame: the event's ``type`` names the frame, its JSON is the data."""
    name = event.model_dump(include={"type"})["type"]
    return f"event: {name}\ndata: {event.model_dump_json()}\n\n"


class _Feed:
    """The service's state view and its fan-out to the open state streams.

    Holds what the sessions do not: the declared chain, the build's catalog,
    the latest take-over, the last shot, the last finished writing action, the
    current reference's match, the machine readings and the current setup. A
    stream ends only when :attr:`shutdown` is set. While a stream is open the
    machine readings are re-read every :data:`_MACHINE_REFRESH_S`.

    Attributes:
        sessions: The started service's sessions.
        store: Its shot files and references, ``None`` without a state dir.
        declaration: The model's declared chain and readbacks.
        catalog: The devices the build's response matrix covers, ``None``
            when the build wrote no catalog for the model.
        machine: The latest machine readings, ``None`` until read.
        setup_keeper: Holds, persists and re-checks the current setup.
        shutdown: The service-shutdown signal, the sessions' own; set once, it
            ends every stream.
    """

    def __init__(
        self,
        sessions: Sessions,
        store: TrajectoryStore | None,
        declaration: ChainDeclaration | None = None,
        setup_keeper: SetupKeeper | None = None,
    ) -> None:
        from osprey.services.trajectory.acquisition import DEFAULT_DECLARATION

        self.sessions = sessions
        self.store = store
        self.declaration = declaration or DEFAULT_DECLARATION
        self.catalog = _catalog(sessions.layer.startup)
        self.machine: Machine | None = None
        self.setup_keeper = setup_keeper or SetupKeeper()
        self.setup_keeper.on_void = self._setup_voided
        self.shutdown = sessions.shutdown_signal
        self.taken_over: TakenOver | None = None
        self.last_shot: ShotSummary | None = None
        self.last_action: LastAction | None = None
        self.reference: ReferenceLoad | None = None
        self._subscribers: set[_Subscriber] = set()
        self._machine_loop: asyncio.Future[None] | None = None

    # -- the setup -------------------------------------------------------------

    @property
    def setup(self) -> SetupState | None:
        """The current setup, ``None`` when none was applied."""
        return self.setup_keeper.state

    @setup.setter
    def setup(self, state: SetupState | None) -> None:
        self.setup_keeper.state = state

    def _setup_voided(self) -> None:
        """A re-check voided the setup: the step it was made under goes with it."""
        self.sessions.step.void()
        self.publish(StepEvent(step=None))

    def notices(self) -> list[str]:
        """What the service voided or dropped, for the operator."""
        solve = self.sessions.solve
        return [*(solve.notices if solve is not None else ()), *self.setup_keeper.notices]

    # -- the snapshot ----------------------------------------------------------

    def snapshot(self, caller: str | None) -> Snapshot:
        """Every fact the panel draws its states from, as *caller* sees them."""
        sessions = self.sessions
        layer = sessions.layer
        startup = layer.startup
        state = sessions.state()
        holder = sessions.holder
        action = sessions.running_action()
        run = sessions.continuous
        writes_enabled = _writes_enabled(layer.gate)
        mode: Mode = "idle"
        if sessions.continuous_running:
            mode = "continuous"
        elif action is not None:
            mode = "action"
        return Snapshot(
            target=sessions.target,
            caller=caller,
            writes_enabled=writes_enabled,
            posture=_posture(layer.gate, caller, writes_enabled),
            limits=_limits(layer.gate),
            writing_available=startup.writing_available,
            model=startup.commands.model if startup.commands is not None else None,
            model_kind="single_pass" if startup.commands is not None else "none",
            holder=HolderView(owner=holder.owner, kind=holder.kind)  # type: ignore[arg-type]
            if holder is not None
            else None,
            taken_over=self.taken_over,
            mode=mode,
            action=action,
            queue_wait_s=_queue_wait_s(layer, action is not None, state["action_budget_s"]),
            action_timeout_s=state["action_timeout_s"],
            action_budget_s=state["action_budget_s"],
            stop_budget_s=state["stop_budget_s"],
            last_shot=self.last_shot,
            step=sessions.step.step,
            reference=self.reference,
            last_action=self.last_action,
            continuous=run.status() if run is not None and run.running else None,
            chain=list(self.declaration.steps),
            catalog=self.catalog,
            machine=self.machine,
            setup=self.setup,
            solve=sessions.solve.solve if sessions.solve is not None else None,
            retention=startup.settings.raw_retention,
            n_shots_max=self.n_shots_max(),
            notices=self.notices(),
        )

    def n_shots_max(self) -> int | None:
        """The most default-chain injections one action admits; ``None`` when none can run."""
        startup = self.sessions.layer.startup
        if startup.commands is None or startup.sequence is None:
            return None
        return n_shots_max(
            self.declaration,
            startup.settings,
            max_pause_s=startup.sequence.metadata.max_pause_s,
            call_timeout_s=self.sessions.call_timeout_s,
        )

    def require_setup(self) -> None:
        """Refuse a panel chain while no current setup holds.

        Raises:
            AcquisitionRefused: ``setup_required`` when no setup was applied or
                the current one is void.
        """
        setup = self.setup
        if setup is None or setup.void:
            state = "no setup was applied" if setup is None else "the setup is void"
            raise AcquisitionRefused(
                SETUP_REQUIRED, f"{state} on this target; apply the setup before a chain"
            )

    async def refresh_machine(self) -> bool:
        """Read the declared machine readbacks in one batched read.

        Reads outside the run slot; a read that fails keeps the last readings.

        Returns:
            Whether the readings changed.
        """
        addresses = self.declaration.machine
        if not addresses:
            return False
        startup = self.sessions.layer.startup
        try:
            values = await guarded_call(
                lambda: startup.connector.read_multiple_channels(
                    list(addresses.values()), timeout=self.sessions.call_timeout_s
                ),
                record=lambda _: None,
            )
        except Exception:  # the header degrades; nothing else depends on it
            logger.warning("trajectory service: the machine readings were not read", exc_info=True)
            return False
        read: dict[str, Any] = {}
        for name, address in addresses.items():
            reading = values.get(address)
            value = getattr(reading, "value", None)
            item = getattr(value, "item", None)
            read[name] = item() if callable(item) else value
        current = read.get("stored_current")
        try:
            machine = Machine(
                rf_on=read.get("rf_on"),
                stored_current=float(current) if current is not None else None,
            )
        except (TypeError, ValueError):
            logger.warning("trajectory service: the machine readings are not readable values")
            return False
        changed = machine != self.machine
        self.machine = machine
        return changed

    async def _refresh_while_watched(self) -> None:
        """Re-read the machine every :data:`_MACHINE_REFRESH_S` while a stream is open."""
        while self._subscribers and not self.shutdown.is_set():
            if await self.refresh_machine():
                self.publish()
            try:
                await asyncio.wait_for(self.shutdown.wait(), timeout=_MACHINE_REFRESH_S)
            except TimeoutError:
                continue

    async def refresh_reference(self) -> None:
        """Match the current reference over the service's BPMs, as the snapshot shows it."""
        store = self.store
        startup = self.sessions.layer.startup
        if store is None or startup.commands is None or startup.tool is None:
            self.reference = None
            return
        try:
            model = startup.commands.model
            name = await store.current(model)
            if name is None:
                self.reference = None
                return
            loaded = await store.load_reference(model, name, self.selection())
        except (Refused, ValueError, LookupError, OSError):
            self.reference = None
            return
        self.reference = loaded.load

    def selection(self) -> list[str]:
        """The BPMs a reference is matched over: the pending step's shot's, else the tool's.

        Raises:
            AcquisitionRefused, StepRefused: Without a pending shot, the
                service has no trajectory model or no tool.
        """
        pending = self.sessions.step.shot
        if pending is not None:
            return list(pending.bpm_order)
        return list(shot_rows(self.sessions.layer.startup).bpms)

    # -- the streams -----------------------------------------------------------

    def subscribe(self, owner: str | None) -> _Subscriber:
        """A new stream for *owner*, its first frame the current snapshot."""
        subscriber = _Subscriber(owner)
        subscriber.frames.put_nowait(_frame(SnapshotEvent(snapshot=self.snapshot(owner))))
        self._subscribers.add(subscriber)
        if self.declaration.machine and (self._machine_loop is None or self._machine_loop.done()):
            self._machine_loop = asyncio.ensure_future(self._refresh_while_watched())
        return subscriber

    def unsubscribe(self, subscriber: _Subscriber) -> None:
        self._subscribers.discard(subscriber)
        if not self._subscribers and self._machine_loop is not None:
            self._machine_loop.cancel()
            self._machine_loop = None

    def publish(self, *events: BaseModel) -> None:
        """Send *events*, then a fresh snapshot, to every open stream."""
        frames = [_frame(event) for event in events]
        for subscriber in list(self._subscribers):
            try:
                snapshot = _frame(SnapshotEvent(snapshot=self.snapshot(subscriber.owner)))
            except Exception:  # a stream degrades; the action it reports does not
                logger.warning("trajectory service: no snapshot for a stream", exc_info=True)
                snapshot = None
            for frame in [*frames, *([snapshot] if snapshot is not None else [])]:
                try:
                    subscriber.frames.put_nowait(frame)
                except asyncio.QueueFull:
                    logger.warning("trajectory service: a state stream fell behind; frame dropped")

    async def next_frame(self, subscriber: _Subscriber) -> str | None:
        """The stream's next frame, a keep-alive when idle, ``None`` once shut down."""
        if self.shutdown.is_set():
            return None
        if not subscriber.frames.empty():
            return subscriber.frames.get_nowait()
        getter = asyncio.ensure_future(subscriber.frames.get())
        ender = asyncio.ensure_future(self.shutdown.wait())
        try:
            done, _ = await asyncio.wait(
                {getter, ender}, timeout=_HEARTBEAT_S, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            for task in (getter, ender):
                if not task.done():
                    task.cancel()
        if ender in done:
            return None
        if getter in done:
            return getter.result()
        return _KEEPALIVE

    async def stream(self, owner: str | None) -> AsyncIterator[str]:
        """The frames of one state stream, until the shutdown signal."""
        subscriber = self.subscribe(owner)
        try:
            while (frame := await self.next_frame(subscriber)) is not None:
                yield frame
        finally:
            self.unsubscribe(subscriber)

    # -- shots -----------------------------------------------------------------

    def chain_build(
        self, store: TrajectoryStore, run: ChainRun
    ) -> Callable[[PostShotHook], Action[ChainResult]]:
        """The chain-action factory :class:`Sessions` binds its post-shot hook into.

        The hook it builds also publishes the shot and the step it produced.

        Raises:
            AcquisitionRefused, StepRefused: The service cannot run *run*.
        """
        startup = self.sessions.layer.startup
        rows = shot_rows(startup)
        call_timeout_s = self.sessions.call_timeout_s
        declaration = self.declaration
        # Built once here so a chain the service cannot run is refused now,
        # not by the first iteration of a continuous run already answered.
        chain_action(
            startup,
            store,
            rows=rows,
            run=run,
            declaration=declaration,
            call_timeout_s=call_timeout_s,
            **self.shot_selection(),
        )

        def build(hook: PostShotHook) -> Action[ChainResult]:
            async def published(shot: ShotResult) -> Step | None:
                answer = hook(shot)
                if inspect.isawaitable(answer):
                    answer = await answer
                step = answer if isinstance(answer, Step) else None
                self.last_shot = ShotSummary.of(shot)
                self.publish(ShotEvent(shot=shot), StepEvent(step=step))
                return step

            async def action(admission: Any, stop: asyncio.Event) -> ChainResult:
                # Built inside the turn, so the shot takes the exclusions in
                # force when it runs, not those of when it was queued.
                chained = chain_action(
                    startup,
                    store,
                    rows=rows,
                    run=run,
                    declaration=declaration,
                    post_shot=published,
                    call_timeout_s=call_timeout_s,
                    setup_check=self.setup_keeper,
                    **self.shot_selection(),
                )
                result = await chained(admission, stop)
                if result.shot is not None and not makes_step(result.shot):
                    self.shot_unhooked(result.shot)
                return result

            return action

        return build

    def shot_selection(self) -> dict[str, Any]:
        """The BPM exclusions and loss cutoff a shot is taken with: the SolveConfig's in force.

        Every shot path reads them here, so a shot, the step solved from it,
        the published solve and the Apply guard see one BPM set. Without a
        SolveConfig nothing is excluded and the configured cutoff applies.
        """
        solve = self.sessions.solve
        if solve is None:
            return {"excluded_bpms": (), "loss_cutoff": None}
        values = solve.values
        cutoff = values.get("loss_cutoff")
        return {
            "excluded_bpms": tuple(values["excluded_bpms"]),
            "loss_cutoff": None if cutoff is None else float(cutoff),
        }

    def shot_unhooked(self, shot: ShotResult) -> None:
        """Publish a stored shot that made no step: not fresh, not TbT, or nothing included."""
        self.last_shot = ShotSummary.of(shot)
        self.publish(ShotEvent(shot=shot), StepEvent(step=None))


def _catalog(startup: TrajectoryStartup) -> Catalog | None:
    """The devices the build's response matrix covers, in ring order; ``None`` without a catalog."""
    return startup.catalog


def _writes_enabled(gate: AdmitGate) -> bool:
    from osprey_connectors.types import target_writes_enabled

    return bool(target_writes_enabled(gate.section, gate.target))


def _posture(gate: AdmitGate, caller: str | None, writes_enabled: bool) -> str:
    """The posture *caller* acts under here: writes, read_only, sandbox or a refusal code."""
    if not writes_enabled:
        return _POSTURE_READ_ONLY
    if caller is None:
        return _POSTURE_WRITES
    try:
        gate.admit(caller, PANEL_DOOR)
    except AdmitRefused as exc:
        return _POSTURE_SANDBOX if exc.code == "narrowing" else exc.code
    return _POSTURE_WRITES


def _limits(gate: AdmitGate) -> LimitsPosture:
    """The limits posture actions on the service target run under."""
    from osprey_connectors.types import target_limits_posture

    try:
        posture = target_limits_posture(gate.section, gate.target)
    except Exception:
        return "unavailable"
    if not posture.incomplete and posture.enabled is False:
        return "off"
    if posture.enabled and posture.mode == "optional":
        return "optional"
    if posture.enabled and posture.mode == "exclusive":
        return "exclusive"
    return "unavailable"


def _queue_wait_s(layer: ActionLayer, running: bool, action_budget_s: float) -> float:
    """The longest a call submitted now waits to start: one action budget per action ahead."""
    return (layer.queued + (1 if running else 0)) * action_budget_s


# -- the doors -----------------------------------------------------------------


def _door(kind: str, *, prefix: str, dependencies: list[Any]) -> APIRouter:
    """The routes both doors serve; every writing route is ``async def``,
    so its record reaches ``HttpAuditMiddleware`` on the same task."""
    router = APIRouter(prefix=prefix, dependencies=dependencies)

    @router.post("/session")
    async def open_session(
        request: Request, owner: str | None = Depends(door_owner)
    ) -> dict[str, str]:
        sessions = _sessions(request)
        detail: dict[str, Any] = {"owner": owner, "kind": kind, "target": sessions.target}
        with _refusals_recorded(owner, "session_open", detail):
            grant = await sessions.open(owner, kind, await _body(request))
        _record(owner=owner, subject="session_open", refusal=None, detail=detail)
        feed = _feed(request)
        await feed.refresh_machine()
        feed.publish()
        return grant.as_body()

    @router.post("/takeover")
    async def take_over(
        request: Request, owner: str | None = Depends(door_owner)
    ) -> dict[str, str]:
        sessions = _sessions(request)
        feed = _feed(request)
        detail: dict[str, Any] = {"owner": owner, "kind": kind, "target": sessions.target}
        previous = sessions.holder
        with _refusals_recorded(owner, "session_takeover", detail):
            grant = sessions.take_over(owner, kind, await _body(request))
        _record(owner=owner, subject="session_takeover", refusal=None, detail=detail)
        events: list[BaseModel] = [StepEvent(step=None)]
        if previous is not None:
            taken = TakenOver(
                previous=HolderView(owner=previous.owner, kind=previous.kind),  # type: ignore[arg-type]
                by=HolderView(owner=grant.owner, kind=kind),  # type: ignore[arg-type]
            )
            feed.taken_over = taken
            events.insert(0, TakenOverEvent(previous=taken.previous, by=taken.by))
        await feed.refresh_machine()
        feed.publish(*events)
        return grant.as_body()

    @router.delete("/session")
    async def release(request: Request, owner: str | None = Depends(door_owner)) -> dict[str, str]:
        sessions = _sessions(request)
        client_id = request.headers.get(CLIENT_HEADER, "").strip() or None
        presented = {"owner": owner, "target": sessions.target, "client_id": client_id}
        with _refusals_recorded(owner, "session_release", presented):
            if client_id is None:
                raise SessionRefused(
                    BAD_REQUEST, f"the call names no session; send {CLIENT_HEADER}"
                )
            done = sessions.release(owner, kind, client_id, await _body(request))
        _record(owner=owner, subject="session_release", refusal=None, detail=done.as_record())
        _feed(request).publish(StepEvent(step=None))
        return done.as_record()

    @router.get("/state")
    async def state(request: Request, owner: str | None = Depends(door_owner)) -> dict[str, Any]:
        return _feed(request).snapshot(owner).model_dump(mode="json")

    @router.post("/stop", status_code=202)
    async def stop(request: Request, owner: str | None = Depends(door_owner)) -> dict[str, Any]:
        sessions = _sessions(request)
        detail: dict[str, Any] = {"owner": owner, "kind": kind, "target": sessions.target}
        with _refusals_recorded(owner, "stop", detail):
            done = sessions.stop(owner, kind, await _body(request))
        _record(owner=owner, subject="stop", refusal=None, detail=done.as_record())
        _feed(request).publish()
        return {**done.as_record(), "stop_budget_s": sessions.budgets().stop_budget_s}

    @router.post("/shot")
    async def shot(request: Request, owner: str | None = Depends(door_owner)) -> dict[str, Any]:
        sessions = _sessions(request)
        feed = _feed(request)
        detail: dict[str, Any] = {"owner": owner, "kind": kind, "target": sessions.target}
        stopped: ShotStopped | None = None
        failed: ChainFailed | None = None
        with _refusals_recorded(owner, "shot", detail):
            client_id = _client_id(request)
            run = _door_run(kind, feed, await _body(request))
            detail.update(
                steps=",".join(run.steps),
                readout=run.readout,
                n_shots=run.n_shots,
                average=run.average,
            )
            store = _store(request)
            build = feed.chain_build(store, run)
            try:
                result = await sessions.shot(owner, kind, client_id, store, build)
            except ShotStopped as exc:
                stopped = exc
            except ChainFailed as exc:
                failed = exc
        if stopped is not None:
            stopper = sessions.stopper
            fired = ",".join(stopped.commands_fired)
            _record(
                owner=owner,
                subject="shot",
                refusal=RouteRefused(ACTION_STOPPED, f"stopped by {stopper}"),
                detail={**detail, "stopper": stopper, "commands_fired": fired},
            )
            feed.last_action = LastAction(name="shot", stopped=True, moved=[])
            feed.publish(StoppedEvent(action="shot", stopper=stopper, moved=[]))
            raise HTTPException(
                status_code=RouteRefused.statuses[ACTION_STOPPED],
                detail={
                    "code": ACTION_STOPPED,
                    "detail": f"the shot was stopped by {stopper!r}",
                    "commands_fired": stopped.commands_fired,
                },
            )
        if failed is not None:
            _record(
                owner=owner,
                subject="shot",
                refusal=RouteRefused(CHAIN_FAILED, failed.reason),
                detail={**detail, "commands_fired": ",".join(failed.commands_fired)},
            )
            await feed.refresh_machine()
            feed.publish()
            raise HTTPException(
                status_code=RouteRefused.statuses[CHAIN_FAILED],
                detail={
                    "code": CHAIN_FAILED,
                    "detail": failed.reason,
                    "steps_run": failed.steps_run,
                    "commands_fired": failed.commands_fired,
                },
            )
        shot = result.shot
        detail.update(
            shot_id=shot.shot_id if shot is not None else None,
            commands_fired=",".join(result.commands_fired),
        )
        _record(owner=owner, subject="shot", refusal=None, detail=detail)
        # The machine is read after the chain's last step; the hook published the
        # shot while the action ran, and this snapshot shows the action ended.
        await feed.refresh_machine()
        feed.publish()
        if kind == AGENT_DOOR:
            assert shot is not None  # the agent door runs the default chain, which reads out
            return shot.model_dump(mode="json")
        return result.model_dump(mode="json")

    @router.get("/step")
    async def step(request: Request, owner: str | None = Depends(door_owner)) -> dict[str, Any]:
        with _answered():
            _require_owner(owner)
            pending = _sessions(request).step.step
            if pending is None:
                raise RouteRefused(NO_STEP, "no step is pending; take a shot first")
        return pending.model_dump(mode="json")

    @router.post("/apply")
    async def apply_step(
        request: Request, owner: str | None = Depends(door_owner)
    ) -> dict[str, Any]:
        sessions = _sessions(request)
        feed = _feed(request)
        body = await _body(request)
        put_back = isinstance(body, Mapping) and "values" in body
        subject = "put_back" if put_back else "apply"
        detail: dict[str, Any] = {"owner": owner, "kind": kind, "target": sessions.target}
        if isinstance(body, Mapping):
            asked = body.get("model") if put_back else body.get("step_digest")
            detail["model" if put_back else "step_digest"] = asked
        pending = sessions.step.step
        failed: ApplyFailed | None = None
        with _refusals_recorded(owner, subject, detail):
            client_id = _client_id(request)
            try:
                done = await apply(
                    sessions,
                    owner,
                    kind,
                    client_id,
                    body,
                    setup_check=feed.setup_keeper,
                    setup=feed.setup_keeper,
                )
            except ApplyFailed as exc:
                failed = exc
        if failed is not None:
            _record(
                owner=owner,
                subject=subject,
                refusal=RouteRefused(APPLY_FAILED, failed.reason),
                detail={**detail, "moved": _moved(failed.moved)},
            )
            feed.last_action = LastAction(name=subject, stopped=False, moved=failed.moved)
            feed.publish()
            raise _failed(APPLY_FAILED, failed.reason, failed.moved)
        _record(
            owner=owner,
            subject=subject,
            refusal=None,
            detail={**detail, "stopped": done.stopped, "moved": _moved(done)},
        )
        feed.last_action = LastAction(name=subject, stopped=done.stopped, moved=done.moved)
        if done.stopped:
            feed.publish(StoppedEvent(action=subject, stopper=sessions.stopper, moved=done.moved))
        else:
            digest = None if put_back else detail.get("step_digest")
            shot_id = pending.shot_id if pending is not None and not put_back else None
            feed.publish(
                ApplyEvent(step_digest=digest, shot_id=shot_id, moved=done.moved),
                StepEvent(step=sessions.step.step),
            )
        return done.model_dump(mode="json")

    return router


#: The fields an agent-door shot body may carry.
_SHOT_FIELDS = frozenset({"n_shots", "average"})

#: The fields a panel chain body may carry: a shot or a continuous start.
_CHAIN_FIELDS = frozenset(ChainBody.model_fields)


def _chain_body(body: Mapping[str, Any] | None) -> ChainBody:
    """A panel chain body, refused as the door refuses a bad body."""
    fields = _fields(body, _CHAIN_FIELDS)
    try:
        return ChainBody.model_validate(dict(fields))
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(map(str, error['loc'])) or 'body'}: {error['msg']}"
            for error in exc.errors()
        )
        raise SessionRefused(BAD_REQUEST, f"the chain body is refused: {problems}") from exc


def _door_run(kind: str, feed: _Feed, body: Mapping[str, Any] | None) -> ChainRun:
    """The run a door's shot body asks for.

    The panel door runs the ticked chain and needs a current setup; the agent
    door runs the declared default chain, choosing only shots and averaging,
    and is never refused for setup.
    """
    if kind == AGENT_DOOR:
        n_shots, average = _shot_args(_fields(body, _SHOT_FIELDS))
        return default_run(feed.declaration, n_shots=n_shots, average=average)
    chain = _chain_body(body)
    feed.require_setup()
    return chain_run(feed.declaration, chain)


def _panel_only() -> APIRouter:
    """The routes only the panel door serves: continuous, references, shots
    and the state stream."""
    kind = PANEL_DOOR
    router = APIRouter(prefix="/panel/api")

    @router.post("/continuous")
    async def continuous(
        request: Request, owner: str | None = Depends(door_owner)
    ) -> dict[str, Any]:
        sessions = _sessions(request)
        feed = _feed(request)
        detail: dict[str, Any] = {"owner": owner, "kind": kind, "target": sessions.target}
        with _refusals_recorded(owner, "continuous_start", detail):
            client_id = _client_id(request)
            chain = _door_run(kind, feed, await _body(request))
            detail.update(
                steps=",".join(chain.steps),
                readout=chain.readout,
                n_shots=chain.n_shots,
                average=chain.average,
            )
            store = _store(request)
            build = feed.chain_build(store, chain)
            run = sessions.start_continuous(
                owner,
                kind,
                client_id,
                store,
                build,
                on_end=_published_end(feed),
            )
        _record(owner=owner, subject="continuous_start", refusal=None, detail=detail)
        feed.publish()
        return run.status().model_dump(mode="json")

    @router.post("/continuous/heartbeat")
    async def heartbeat(
        request: Request, owner: str | None = Depends(door_owner)
    ) -> dict[str, Any]:
        sessions = _sessions(request)
        with _answered():
            status = sessions.heartbeat(owner, kind, _client_id(request))
        feed = _feed(request)
        if await feed.refresh_machine():
            feed.publish()
        return {"continuous": status.model_dump(mode="json") if status is not None else None}

    @router.get("/references")
    async def references(
        request: Request, owner: str | None = Depends(door_owner)
    ) -> dict[str, Any]:
        with _answered(), _store_refusals():
            _require_owner(owner)
            store = _store(request)
            model = _model(_sessions(request).layer.startup)
            names = await store.list_references(model)
            current = await store.current(model)
        return {"model": model, "names": names, "current": current}

    @router.post("/references")
    async def save_reference(
        request: Request, owner: str | None = Depends(door_owner)
    ) -> dict[str, Any]:
        sessions = _sessions(request)
        feed = _feed(request)
        detail: dict[str, Any] = {"owner": owner, "kind": kind, "target": sessions.target}
        with _refusals_recorded(owner, "reference_save", detail), _store_refusals():
            fields = _fields(await _body(request), frozenset({"name", "shot_id"}))
            name = fields.get("name")
            detail["name"] = name
            if not isinstance(name, str):
                raise RouteRefused(INVALID_NAME, f"a reference is saved under a name, not {name!r}")
            sessions.layer.gate.admit(owner, kind)
            store = _store(request)
            model = _model(sessions.layer.startup)
            shot = await _reference_shot(sessions, store, model, fields.get("shot_id"))
            detail["shot_id"] = shot.shot_id
            _require_reference_shot(shot)
            sha256 = await store.save_reference(
                model,
                name,
                _readings(shot),
                readout=shot.readout,
                source_target=store.target,
                source_model=model,
            )
        detail["sha256"] = sha256
        _record(owner=owner, subject="reference_save", refusal=None, detail=detail)
        await feed.refresh_reference()
        feed.publish()
        return {"model": model, "name": name, "sha256": sha256, "shot_id": shot.shot_id}

    @router.get("/references/{name}")
    async def load_reference(
        name: str, request: Request, owner: str | None = Depends(door_owner)
    ) -> dict[str, Any]:
        sessions = _sessions(request)
        with _answered(), _store_refusals():
            _require_owner(owner)
            store = _store(request)
            startup = sessions.layer.startup
            model = _model(startup)
            loaded = await store.load_reference(model, name, _feed(request).selection())
        return loaded.load.model_dump(mode="json", exclude={"readings"})

    @router.get("/references/{name}/readings")
    async def reference_readings(
        name: str, request: Request, owner: str | None = Depends(door_owner)
    ) -> dict[str, Any]:
        with _answered(), _store_refusals():
            _require_owner(owner)
            store = _store(request)
            model = _model(_sessions(request).layer.startup)
            stored = await store.reference_file(model, name)
        return stored.model_dump(mode="json")

    @router.post("/references/import")
    async def import_reference(
        request: Request, owner: str | None = Depends(door_owner)
    ) -> dict[str, Any]:
        sessions = _sessions(request)
        feed = _feed(request)
        detail: dict[str, Any] = {"owner": owner, "kind": kind, "target": sessions.target}
        step: Step | None = None
        with _refusals_recorded(owner, "reference_import", detail), _store_refusals():
            raw = await request.body()
            if len(raw) > MAX_REFERENCE_IMPORT_BYTES:
                raise RouteRefused(
                    REFERENCE_TOO_LARGE,
                    f"the reference is {len(raw)} bytes; an import takes at most "
                    f"{MAX_REFERENCE_IMPORT_BYTES}",
                )
            body = await _body(request)
            fields = _fields(body, _IMPORT_FIELDS)
            detail["name"] = fields.get("name")
            imported = _reference_import(fields)
            detail.update(readout=imported.readout, select=imported.select)
            store = _store(request)
            startup = sessions.layer.startup
            model = _model(startup)
            _check_import(imported, shot_rows(startup).bpms)
            if imported.select:
                await _require_readout(sessions, store, model, imported.readout)
            readings = {bpm: (reading.x, reading.y) for bpm, reading in imported.readings.items()}

            async def save(_admission: Admission | None = None, _stop: Any = None) -> str:
                return await store.save_reference(
                    model,
                    imported.name,
                    readings,
                    readout=imported.readout,
                    source_target=imported.source_target,
                    source_model=imported.source_model,
                )

            if imported.select:
                # Selecting needs the holder: the save runs as the holder's call,
                # so a non-holder's import stores nothing.
                client_id = _client_id(request)
                sha256 = await sessions.call(owner, kind, client_id, save, name="reference")
                step = await sessions.select_reference(owner, kind, client_id, store, imported.name)
            else:
                sessions.layer.gate.admit(owner, kind)
                sha256 = await save()
        detail["sha256"] = sha256
        _record(owner=owner, subject="reference_import", refusal=None, detail=detail)
        await feed.refresh_reference()
        if imported.select:
            feed.publish(StepEvent(step=step))
        else:
            feed.publish()
        return {
            "model": model,
            "name": imported.name,
            "sha256": sha256,
            "reference": feed.reference.model_dump(mode="json")
            if imported.select and feed.reference
            else None,
            "step": step.model_dump(mode="json") if step is not None else None,
        }

    @router.post("/references/select")
    async def select_reference(
        request: Request, owner: str | None = Depends(door_owner)
    ) -> dict[str, Any]:
        sessions = _sessions(request)
        feed = _feed(request)
        detail: dict[str, Any] = {"owner": owner, "kind": kind, "target": sessions.target}
        with _refusals_recorded(owner, "reference_select", detail), _store_refusals():
            client_id = _client_id(request)
            fields = _fields(await _body(request), frozenset({"name"}))
            if "name" not in fields:
                raise SessionRefused(BAD_REQUEST, "name the reference to select, or null for none")
            name = fields["name"]
            detail["name"] = name
            store = _store(request)
            if name is not None:
                model = _model(sessions.layer.startup)
                stored = await store.reference_file(model, name)
                await _require_readout(sessions, store, model, stored.readout)
            step = await sessions.select_reference(owner, kind, client_id, store, name)
        _record(owner=owner, subject="reference_select", refusal=None, detail=detail)
        await feed.refresh_reference()
        feed.publish(StepEvent(step=step))
        return {
            "reference": feed.reference.model_dump(mode="json") if feed.reference else None,
            "step": step.model_dump(mode="json") if step is not None else None,
        }

    @router.post("/solve")
    async def solve(request: Request, owner: str | None = Depends(door_owner)) -> dict[str, Any]:
        sessions = _sessions(request)
        feed = _feed(request)
        detail: dict[str, Any] = {"owner": owner, "kind": kind, "target": sessions.target}
        with _refusals_recorded(owner, "solve", detail):
            client_id = _client_id(request)
            fields = _fields(await _body(request), _SOLVE_FIELDS)
            try:
                partial = SolveConfigPartial.model_validate(dict(fields))
            except ValidationError as exc:
                raise SessionRefused(
                    BAD_REQUEST, f"the SolveConfig update does not validate: {exc}"
                ) from exc
            detail["keys"] = ",".join(sorted(fields)) or "recompute"
            store = _store(request)
            update, step = await sessions.solve_config(owner, kind, client_id, store, partial)
        detail["changed"] = ",".join(sorted(update.changed)) or "none"
        detail["solve_sha"] = update.solve.sha
        _record(owner=owner, subject="solve", refusal=None, detail=detail)
        feed.publish(StepEvent(step=step))
        return {
            "solve": update.solve.model_dump(mode="json"),
            "step": step.model_dump(mode="json") if step is not None else None,
        }

    @router.post("/setup")
    async def setup(request: Request, owner: str | None = Depends(door_owner)) -> dict[str, Any]:
        sessions = _sessions(request)
        feed = _feed(request)
        detail: dict[str, Any] = {"owner": owner, "kind": kind, "target": sessions.target}
        failed: SetupFailed | None = None
        with _refusals_recorded(owner, SETUP, detail):
            client_id = _client_id(request)
            body = _setup_body(await _body(request))
            detail["settings"] = ",".join(sorted(body.settings)) or "none"
            if body.excluded_bpms is not None:
                detail["excluded_bpms"] = ",".join(body.excluded_bpms) or "none"
            keeper = feed.setup_keeper
            check_body(keeper, body)
            store = _store(request)
            _refuse_while_running(sessions)

            def precondition(_admission: Admission) -> None:
                _refuse_while_running(sessions)

            try:
                done = await sessions.call(
                    owner,
                    kind,
                    client_id,
                    setup_action(keeper, sessions, store, body, owner=owner, client_id=client_id),
                    precondition=precondition,
                    name=SETUP,
                )
            except SetupFailed as exc:
                failed = exc
        if failed is not None:
            _record(
                owner=owner,
                subject=SETUP,
                refusal=RouteRefused(SETUP_FAILED, failed.reason),
                detail={**detail, "moved": _moved(failed.moved)},
            )
            feed.last_action = LastAction(name=SETUP, stopped=False, moved=failed.moved)
            feed.publish(StepEvent(step=sessions.step.step))
            raise _failed(SETUP_FAILED, failed.reason, failed.moved)
        result = done.result
        detail.update(
            stopped=done.stopped,
            moved=_moved(result.moved),
            unsettled=",".join(result.unsettled) or "none",
            settings_sha=result.setup.settings_sha if result.setup is not None else None,
        )
        _record(owner=owner, subject=SETUP, refusal=None, detail=detail)
        feed.last_action = LastAction(name=SETUP, stopped=done.stopped, moved=result.moved)
        if done.stopped:
            feed.publish(
                StoppedEvent(action=SETUP, stopper=sessions.stopper, moved=result.moved),
                StepEvent(step=sessions.step.step),
            )
        else:
            feed.publish(StepEvent(step=sessions.step.step))
        return result.model_dump(mode="json")

    @router.get("/setup")
    async def setup_readbacks(
        request: Request, owner: str | None = Depends(door_owner)
    ) -> dict[str, Any]:
        sessions = _sessions(request)
        with _answered():
            _require_owner(owner)
            read = await _feed(request).setup_keeper.readbacks(
                sessions.layer.startup.connector, sessions.call_timeout_s
            )
        return read.model_dump(mode="json")

    @router.get("/shots")
    async def shots(request: Request, owner: str | None = Depends(door_owner)) -> dict[str, Any]:
        with _answered(), _store_refusals():
            _require_owner(owner)
            store = _store(request)
            model = _model(_sessions(request).layer.startup)
            listed = await store.list_shots(model)
        return {"model": model, "shots": [shot.model_dump(mode="json") for shot in listed]}

    @router.get("/shots/{shot_id}/raw", response_class=Response)
    async def shot_raw(
        shot_id: str, request: Request, owner: str | None = Depends(door_owner)
    ) -> Response:
        with _answered(), _store_refusals():
            _require_owner(owner)
            store = _store(request)
            model = _model(_sessions(request).layer.startup)
            _root, buffer = await store.load_raw(model, shot_id)
        return Response(content=encode_raw(buffer), media_type=RAW_MEDIA_TYPE)

    @router.get("/events")
    async def events(
        request: Request, owner: str | None = Depends(door_owner)
    ) -> StreamingResponse:
        return StreamingResponse(
            _feed(request).stream(owner),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return router


#: The keys a SolveConfig update may carry.
_SOLVE_FIELDS = frozenset(SolveConfigPartial.model_fields)

#: The keys a setup body may carry.
_SETUP_FIELDS = frozenset(SetupBody.model_fields)


def _setup_body(body: Mapping[str, Any] | None) -> SetupBody:
    """A setup body, refused as the door refuses a bad body."""
    fields = _fields(body, _SETUP_FIELDS)
    try:
        return SetupBody.model_validate(dict(fields))
    except ValidationError as exc:
        raise SessionRefused(BAD_REQUEST, f"the setup body does not validate: {exc}") from exc


def _refuse_while_running(sessions: Sessions) -> None:
    if sessions.continuous_running:
        raise SessionRefused(
            CONTINUOUS_RUNNING, "a continuous run is in progress; stop it before the setup"
        )


def _published_end(feed: _Feed) -> Callable[[ContinuousEndedEvent], None]:
    def ended(event: ContinuousEndedEvent) -> None:
        feed.publish(event)

    return ended


async def _reference_shot(
    sessions: Sessions, store: TrajectoryStore, model: str, shot_id: Any
) -> ShotResult:
    """The shot a reference is saved from: the one named, else the step's, else the newest."""
    if shot_id is not None:
        if not isinstance(shot_id, str):
            raise SessionRefused(BAD_REQUEST, f"shot_id is a string, not {shot_id!r}")
        return await store.load_shot(model, shot_id)
    pending = sessions.step.shot
    if pending is not None:
        return pending
    listed = await store.list_shots(model)
    if not listed:
        raise RouteRefused(NO_SHOT, f"model {model!r} has no stored shot to save a reference from")
    return listed[0]


def _require_reference_shot(shot: ShotResult) -> None:
    """Refuse a shot that may not become a reference: only a fresh TbT shot does."""
    if not shot.fresh:
        raise RouteRefused(
            SHOT_NOT_FRESH,
            f"shot {shot.shot_id!r} is not fresh (its action did not arm and inject with the "
            "machine held still), so it cannot become a reference",
        )
    if shot.readout != _SAVED_READOUT:
        raise RouteRefused(
            READOUT_MISMATCH,
            f"shot {shot.shot_id!r} was read as {shot.readout}; a reference is saved from a "
            f"{_SAVED_READOUT} shot",
        )


async def _require_readout(
    sessions: Sessions, store: TrajectoryStore, model: str, readout: Readout
) -> None:
    """Refuse a reference of *readout* where the shown shot was read in another mode.

    The shown shot is the pending step's (always TbT), else the newest stored
    one; with neither, any mode is accepted.
    """
    shot = sessions.step.shot
    if shot is None:
        listed = await store.list_shots(model)
        shot = listed[0] if listed else None
    if shot is not None and shot.readout != readout:
        raise RouteRefused(
            READOUT_MISMATCH,
            f"the reference was taken as {readout} and shot {shot.shot_id!r} was read as "
            f"{shot.readout}; a reference is never used across readout modes",
        )


#: The fields a reference import takes.
_IMPORT_FIELDS = frozenset(ReferenceImport.model_fields)


def _reference_import(fields: Mapping[str, Any]) -> ReferenceImport:
    """*fields* as a reference import, refusing a body of the wrong shape."""
    name = fields.get("name")
    if not isinstance(name, str):
        raise RouteRefused(INVALID_NAME, f"a reference is imported under a name, not {name!r}")
    try:
        return ReferenceImport.model_validate(dict(fields))
    except ValidationError as exc:
        raise SessionRefused(BAD_REQUEST, f"the reference import does not validate: {exc}") from exc


def _check_import(imported: ReferenceImport, bpms: Iterable[str]) -> None:
    """Refuse an import naming no BPM, a BPM outside *bpms*, or a non-finite reading."""
    known = set(bpms)
    if not imported.readings:
        raise SessionRefused(BAD_REQUEST, f"reference {imported.name!r} has no BPM readings")
    unknown = [bpm for bpm in imported.readings if bpm not in known]
    if unknown:
        raise RouteRefused(
            UNKNOWN_BPM,
            f"the reference names {', '.join(unknown)}, which the tool does not read",
        )
    bad = [
        f"{plane} at {bpm}"
        for bpm, reading in imported.readings.items()
        for plane, value in (("x", reading.x), ("y", reading.y))
        if value is not None and not math.isfinite(value)
    ]
    if bad:
        raise RouteRefused(
            READING_NOT_FINITE,
            f"readings {', '.join(bad)} are not finite; a reading is a finite number or null",
        )


def _readings(shot: ShotResult) -> dict[str, tuple[float | None, float | None]]:
    """Each BPM's ``(x, y)`` reading of *shot*, keyed by BPM in its order."""
    return {
        bpm: (shot.x[x_row], shot.y[y_row])
        for bpm, x_row, y_row in zip(shot.bpm_order, shot.x, shot.y, strict=True)
    }


async def _default_start() -> TrajectoryStartup:
    from osprey.services.trajectory.startup import start

    return await start()


def _trajectory_slots(config: Mapping[str, Any], model: str) -> Mapping[str, Any] | None:
    """The model's ``trajectory_slots`` entry of the build's facts, ``None`` without one."""
    from osprey.services.trajectory.startup import _facts

    slots = _facts(config).get("trajectory_slots")
    entry = slots.get(model) if isinstance(slots, Mapping) else None
    return entry if isinstance(entry, Mapping) else None


def _declaration(config: Mapping[str, Any], startup: TrajectoryStartup) -> ChainDeclaration:
    """The selected model's declared chain; the default chain without a model or a block.

    Raises:
        StartupRefused: The model's declared chain is malformed.
    """
    from osprey.services.trajectory.acquisition import DEFAULT_DECLARATION

    commands = startup.commands
    if commands is None:
        return DEFAULT_DECLARATION
    try:
        return chain_declaration(_trajectory_slots(config, commands.model))
    except ValueError as exc:
        raise StartupRefused(f"model {commands.model!r}: {exc}") from exc


def _load_config() -> Mapping[str, Any]:
    from osprey_connectors.workspace import load_osprey_config

    return load_osprey_config()


def _open_store(startup: TrajectoryStartup) -> TrajectoryStore | None:
    """The state directory's store, keeping the raw retention's root shots."""
    try:
        return TrajectoryStore.from_env(shots_per_model=startup.settings.raw_retention)
    except StoreUnconfigured:
        logger.warning(
            "trajectory service: no state directory; shots and references are unavailable"
        )
        return None


async def _solve_config(
    startup: TrajectoryStartup, store: TrajectoryStore | None
) -> SolveConfigStore | None:
    """The service's SolveConfig, its overrides read from the store.

    ``None`` without a tool, or when the tool cannot publish a solve at all:
    the service starts either way, and every step then solves with the tool's
    configured values.
    """
    from osprey.services.trajectory.solve_config import SolveConfigStore
    from osprey.services.trajectory.step import TrajectoryTool

    tool = startup.tool
    if not isinstance(tool, TrajectoryTool):
        return None
    try:
        return await SolveConfigStore.load(
            tool,
            loss_cutoff=startup.settings.loss_sum_cutoff,
            directory=store.directory if store is not None else None,
        )
    except Exception:  # the service starts without a published solve
        logger.warning("trajectory service: no SolveConfig could be loaded", exc_info=True)
        return None


def _setup_keeper(
    config: Mapping[str, Any],
    startup: TrajectoryStartup,
    sessions: Sessions,
    store: TrajectoryStore | None,
) -> SetupKeeper:
    """The model's setup keeper, holding the setup persisted beside the SolveConfig."""
    commands = startup.commands
    if commands is None:
        return SetupKeeper()

    def excluded() -> list[str]:
        solve = sessions.solve
        return list(solve.values["excluded_bpms"]) if solve is not None else []

    keeper = SetupKeeper(
        declared_settings(_trajectory_slots(config, commands.model)),
        model=commands.model,
        directory=store.directory if store is not None else None,
        devices={device.x_row: device.name for device in commands.devices},
        excluded=excluded,
    )
    keeper.load()
    return keeper


def create_app(*, start: StartFn | None = None) -> FastAPI:
    """Build the service's app.

    Args:
        start: Builds the started service; the deployment checks of
            :func:`osprey.services.trajectory.startup.start` by default.

    Returns:
        The gated app. Its lifespan starts the service, the admission gate,
        the action worker, the sessions, the store and the state view. On the
        way out it calls :meth:`Sessions.shutdown` (a no-op when the server
        already called it) and then disconnects the connector.
    """
    start_fn = start or _default_start

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        startup = await start_fn()
        config = _load_config()
        declaration = _declaration(config, startup)
        gate = AdmitGate.from_config(config, startup.target)
        layer = ActionLayer(startup, gate)
        layer.start()
        sessions = Sessions(layer, call_timeout_s=connector_call_timeout_s(config, startup.target))
        store = _open_store(startup)
        sessions.solve = await _solve_config(startup, store)
        keeper = _setup_keeper(config, startup, sessions, store)
        await keeper.recheck(startup.connector, sessions.call_timeout_s)
        feed = _Feed(sessions, store, declaration, keeper)
        await feed.refresh_reference()
        app.state.startup = startup
        app.state.sessions = sessions
        app.state.store = store
        app.state.feed = feed
        app.state.shutdown_signal = feed.shutdown
        try:
            yield
        finally:
            await sessions.shutdown()
            disconnect = getattr(startup.connector, "disconnect", None)
            if disconnect is not None:
                try:
                    await disconnect()
                except Exception:
                    logger.warning("trajectory service: connector did not close", exc_info=True)

    app = FastAPI(title="OSPREY Trajectory", lifespan=lifespan)

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        startup: TrajectoryStartup = request.app.state.startup
        code = await reach.probe(startup)
        return {"status": "ok", "target": code or REACHABLE, "control_target": startup.target}

    app.include_router(_door(PANEL_DOOR, prefix="/panel/api", dependencies=[]))
    app.include_router(_panel_only())
    app.include_router(_door(AGENT_DOOR, prefix="/api", dependencies=[Depends(_launch_token)]))
    # After the door routes: Starlette matches in declaration order, so
    # /panel/api/* reaches its routes and every other /panel/ path the bundle.
    app.mount("/panel", StaticFiles(directory=_PANEL_DIR, html=True), name="panel")
    configure_interface_app(app, static_dir=_PANEL_DIR / "static")
    return app
