"""One narrowing, two users, one deployment — from the chip's record to the run record.

The pieces this joins are each pinned on their own elsewhere, and none of those
files can show what an operator actually meets. ``tests/connectors`` pins the
store's verdicts against a tree; ``tests/services/bluesky_bridge`` pins the
wrapper binding the owner and the single line a refusal earns, both against
connector doubles that raise refusals the test itself wrote; ``tests/services/
bluesky_bridge/test_runs*.py`` pins the run-record projection against literal
manager documents. Here the whole chain runs at once and nobody stands in for
the decision: a real plan wrapper binds the owner a queue item carried, a real
``RunEngine`` schedules a real device's ``set``, the REAL reference monitor in
``osprey_connectors.control_system.base`` asks the REAL store, and the store
reads a real provisioned tree. What the monitor answers is then carried out to
the two surfaces a person reads — the run record's ``error`` and the worker's
log — rather than asserted at the seam it was decided in.

That is the point of running it whole: every one of those seams agrees with its
neighbour in isolation, and a deployment where a lane's queueserver holds the
whole per-user tree is where they have to agree with each other.

**Two users, one deployment.** Alice's narrowing is recorded WHILE a plan is
running, which is what the feature is for: a target flipped to read-only from
the header chip refuses the very next write of a plan already in flight, with
no respawn. The plan's own first write is the clock that says it is running —
the connector plants the record from inside it, standing in for the other
process the chip really writes from. The same arrangement, with only the name
on the item changed, is what shows the narrowing is Alice's alone: Bob's plan
runs through it and completes.

**Which surface the run-record rows assert.** A run record is a projection over
the manager's own history item, and there is no manager in this process: the
plan runs under a bare ``RunEngine``, as the queueserver worker runs it. So the
rows compose the history item upstream would have filed for the run that just
failed — from the exception the ``RunEngine`` really raised — and assert on what
:mod:`osprey.services.bluesky_bridge.runs` publishes for it. What is NOT pinned
here is that a live queueserver files that item; ``tests/e2e`` owns that.
:func:`_manager_history_item` names the upstream code the composition is read
from, and it is the only place in this file where anything is assumed rather
than run.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("bluesky")
pytest.importorskip("ophyd_async")

from bluesky import RunEngine  # noqa: E402
from bluesky import plan_stubs as bps  # noqa: E402
from bluesky.utils import FailedStatus  # noqa: E402
from pydantic import BaseModel, ConfigDict  # noqa: E402

from osprey.services.bluesky_bridge import qserver_startup, runs, session_upload  # noqa: E402
from osprey.services.bluesky_bridge.devices.connector import ConnectorSettable  # noqa: E402
from osprey.services.bluesky_bridge.plan_fields import MovableChannel  # noqa: E402
from osprey.services.bluesky_bridge.queue_backend import RUN_ID_META_KEY  # noqa: E402
from osprey.services.bluesky_bridge.session_upload import install_session_plan  # noqa: E402
from osprey_connectors import control_context, posture_store  # noqa: E402
from osprey_connectors.control_system.mock_connector import MockConnector  # noqa: E402
from osprey_connectors.factory import (  # noqa: E402
    ConnectorFactory,
    isolated_connector_registries,
)
from osprey_connectors.posture_store import (  # noqa: E402
    NO_OWNER,
    RESERVED_OWNER_KWARG,
    StoreVerdict,
)

#: The person whose chip narrows the target in every row that has a narrowing.
NARROWING_OWNER = "alice"
#: The other person on the same deployment, who narrowed nothing.
OTHER_OWNER = "bob"

_TARGET = "live"
_CHANNEL = "SR:QUAD:STRENGTH"
_DEVICE_NAME = "quadrupole"
_PLAN_NAME = "two_step_write_plan"
#: The plan writes twice. The first write is the one that says the plan is
#: running; the second is the "next write" a narrowing has to reach.
_FIRST_SETPOINT = 1.5
_SECOND_SETPOINT = 3.5

#: The connector type the deployment's write posture is keyed on. ``mock`` is a
#: real deployment's spelling for the simulated control system, so the posture
#: these rows arm is one a deployment can actually set.
_CONNECTOR_TYPE = "mock"
#: The config key a deployment sets to arm writes for that connector type, and
#: the one a refusal by the ceiling names — the type-keyed spelling, because the
#: connector below carries a type stamp.
_TYPE_WRITES_ENABLED_KEY = f"control_system.connector.{_CONNECTOR_TYPE}.writes_enabled"

#: The machine-readable half of every refusal here: the monitor declined to
#: attempt the write. Which of its three reasons decided it is in the message,
#: never in this code, so the code alone never tells a narrowing from a ceiling.
_REFUSAL_CODE = "WRITES_DISABLED"

_RUN_ID = "run-0001"
_ITEM_UID = "item-0001"


class _RecordingMockConnector(MockConnector):
    """The simulated control system, with what reached it recorded.

    Subclassed rather than doubled: every gate these rows are about lives on
    ``ControlSystemConnector`` and runs before ``MockConnector.write_channel``
    is entered at all, so a hand-written double would be asserting against a
    refusal the test wrote rather than the one the monitor decides.

    ``_put`` is the simulated machine's own transport seam — the line past which
    a value has been sent — which makes ``puts`` the answer to "did anything
    actually move", and an empty ``puts`` the proof that a refusal was decided
    before the value was sent rather than after.

    ``after_first_put`` runs once the plan has written something, which is this
    file's stand-in for the moment a chip is flipped on a plan already in
    flight. It is deliberately not "before the run": a record read at enqueue
    time would satisfy every other row here and none of the ones that matter.
    """

    def __init__(self) -> None:
        super().__init__()
        self.puts: list[tuple[str, Any]] = []
        self.after_first_put: Callable[[], None] | None = None

    def _put(self, channel_address: str, value: Any) -> None:
        super()._put(channel_address, value)
        self.puts.append((channel_address, value))
        if len(self.puts) == 1 and self.after_first_put is not None:
            self.after_first_put()


class _Params(BaseModel):
    """The plan's schema, declaring the one channel it drives.

    ``extra="forbid"`` keeps the reserved owner kwarg's removal an assertion
    rather than an assumption: a wrapper that passed it on would fail
    validation here instead of handing it to the plan as a parameter.
    """

    model_config = ConfigDict(extra="forbid")

    motor: MovableChannel = _DEVICE_NAME
    first: float = _FIRST_SETPOINT
    second: float = _SECOND_SETPOINT


def _two_step_write_plan(devices: Any, params: Any) -> Iterator[Any]:
    """Drive the declared movable twice."""
    yield from bps.mv(devices[params.motor], params.first)
    yield from bps.mv(devices[params.motor], params.second)


_SESSION_SOURCE = f'''PLAN_METADATA = {{"name": {_PLAN_NAME!r}}}

from bluesky import plan_stubs as bps
from pydantic import BaseModel, ConfigDict

from osprey.services.bluesky_bridge.plan_fields import MovableChannel


class PARAMS(BaseModel):
    model_config = ConfigDict(extra="forbid")

    motor: MovableChannel = {_DEVICE_NAME!r}
    first: float = {_FIRST_SETPOINT!r}
    second: float = {_SECOND_SETPOINT!r}


def build_plan(devices, params):
    """Drive the declared movable twice."""
    yield from bps.mv(devices[params.motor], params.first)
    yield from bps.mv(devices[params.motor], params.second)
'''


# --- the deployment these rows run in --------------------------------------


def arm_deployment_writes(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> None:
    """Set the deployment's write ceiling, the half config owns.

    The monitor ANDs this with the live half a record can narrow, so it has to
    be armed for a narrowing to be the thing that refuses — and disarmed for the
    ceiling to be shown to be the thing that refuses an owner-less run. Both
    key shapes are answered because the monitor asks the deployment-wide key for
    an unstamped connector and the ``control_system`` block for a stamped one.
    """

    def config_value(key: str, default: Any = None) -> Any:
        if key == "control_system.writes_enabled":
            return enabled
        if key == "control_system":
            return {"writes_enabled": enabled}
        return default

    monkeypatch.setattr("osprey_connectors.config.get_config_value", config_value)


@pytest.fixture
def control_context_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A provisioned control-context tree, bound as a lane's queueserver has it.

    The stamped-owner rung and the launch pin are cleared so a developer's
    environment cannot decide these rows, and the tree bind is what makes
    ``NO_OWNER`` — rather than the account this process runs as — the ladder's
    answer for a plan that arrived without a name. The bound owner is reset on
    the way out as well as in: a run that failed mid-plan would otherwise hand
    its owner to the next test.
    """
    tree = tmp_path / posture_store.STATE_DIR_NAME
    tree.mkdir(parents=True)
    (tree / posture_store.CONTROL_TREE_MARKER_NAME).write_text("provisioned\n", encoding="utf-8")
    monkeypatch.delenv(posture_store.CONTROL_OWNER_ENV_VAR, raising=False)
    monkeypatch.delenv(posture_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(tree))
    posture_store.invalidate_cache()
    posture_store._owner_var.set(NO_OWNER)
    yield tree
    posture_store._owner_var.set(NO_OWNER)
    posture_store.invalidate_cache()


@pytest.fixture
def armed_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The deployment ceiling open, so the record is what decides a write."""
    arm_deployment_writes(monkeypatch, enabled=True)


@pytest.fixture
def connector(armed_deployment: None) -> Iterator[_RecordingMockConnector]:  # noqa: ARG001 - the deployment's write ceiling is open before the connector is built
    """The connector a lane builds, through the factory that builds a lane's.

    Built rather than constructed, because the monitor reads two stamps that
    only the factory sets between construction and ``connect()``: the type,
    which selects the deployment's connector block, and the control target,
    which indexes the per-user store. A connector stamped by hand here would be
    this file agreeing with itself about the shape a lane's connector has.

    The recording subclass is registered under the deployment's own ``mock``
    spelling inside :func:`~osprey_connectors.factory.isolated_connector_registries`,
    the sanctioned bracket for mutating the factory registries, so the
    registration is restored on the way out and no later test builds this
    class.
    """
    with isolated_connector_registries():
        ConnectorFactory.register_control_system(_CONNECTOR_TYPE, _RecordingMockConnector)
        instance = asyncio.run(
            ConnectorFactory.create_control_system_connector(
                {
                    "type": _CONNECTOR_TYPE,
                    "connector": {_CONNECTOR_TYPE: {"response_delay_ms": 0, "noise_level": 0.0}},
                },
                control_target=_TARGET,
            )
        )
        assert isinstance(instance, _RecordingMockConnector)
        yield instance


def narrow_for(tree: Path, owner: str, target: str = _TARGET) -> Path:
    """Record *owner*'s narrowing of *target*, the way that owner's chip writes it."""
    path = tree / owner / control_context.RECORD_FILENAME
    control_context.write_record(
        control_context.ControlContext(
            target=target,
            generation=3,
            posture={target: posture_store.POSTURE_SANDBOX},
        ),
        path=path,
    )
    posture_store.invalidate_cache()
    return path


# --- the two plan wrappers a worker runs -----------------------------------


def _catalog_wrapper(device: ConnectorSettable) -> Callable[..., Iterator[Any]]:
    """The plan as ``qserver_startup`` hands it to the manager."""
    spec = type(
        "_Spec",
        (),
        {
            "name": _PLAN_NAME,
            "schema": _Params,
            "plan": staticmethod(_two_step_write_plan),
            "description": "A stub plan that drives one channel twice.",
        },
    )()
    return qserver_startup._make_plan_function(spec, devices={_DEVICE_NAME: device})


def _session_wrapper(device: ConnectorSettable) -> Callable[..., Iterator[Any]]:
    """The plan as ``session_upload`` installs it in the worker namespace."""
    namespace: dict[str, Any] = {"__name__": "__main__", _DEVICE_NAME: device}
    return install_session_plan(namespace, _PLAN_NAME, _SESSION_SOURCE)


@pytest.fixture(params=["catalog", "session"])
def wrapped_plan(
    request: pytest.FixtureRequest, connector: _RecordingMockConnector
) -> tuple[Callable[..., Iterator[Any]], str]:
    """Both wrappers around the plan, with the logger each one warns on.

    Both run every row. A session plan is authored by an agent at the terminal
    and installed into the worker namespace at upload; one that reached the
    ``RunEngine`` without its owner bound, or that swallowed the line a refusal
    earns, would be the quiet way past a gate the catalog path enforces.
    """
    device = ConnectorSettable(connector, _CHANNEL, name=_DEVICE_NAME)
    if request.param == "catalog":
        return _catalog_wrapper(device), qserver_startup.logger.name
    return _session_wrapper(device), session_upload.logger.name


def _run(plan_function: Callable[..., Iterator[Any]], owner: str | None) -> None:
    """Execute *plan_function* for *owner*, as a worker's RunEngine does.

    ``None`` is an item that names nobody: the reserved kwarg is absent rather
    than empty, which is the shape an out-of-band enqueue leaves.
    """
    kwargs = {} if owner is None else {RESERVED_OWNER_KWARG: owner}
    RunEngine(context_managers=[])(plan_function(**kwargs))


def wrapper_warnings(caplog: pytest.LogCaptureFixture, logger_name: str) -> list[logging.LogRecord]:
    """The wrapper's own warnings, with every neighbour's out of the count.

    Scoped by logger name because the store warns too when a record cannot be
    read, on its own channel: counting every warning in the process would make
    "exactly one line per refusal" a claim about how many modules happened to
    log, rather than about the wrapper writing the line once.
    """
    return [
        record
        for record in caplog.records
        if record.name == logger_name and record.levelno == logging.WARNING
    ]


def store_answered(verdict: StoreVerdict) -> str:
    """The clause a refusal decided by the store CLOSES on, for *verdict*.

    The bare verdict word does not separate the two store arms and must never
    be asserted on alone: the ``control_context_unavailable`` message says "This
    is not a narrowing anybody set", so ``"narrowing" in message`` is true for
    both arms and a row written that way passes while the store answered the
    other thing. The closing clause is what differs, so it is what every row
    below asserts, composed from the verdict rather than typed out — a renamed
    member fails these rows instead of leaving them agreeing with a spelling
    nothing answers any more.
    """
    return f"answered {verdict}"


def assert_names_the_refusal(line: str, owner: str | object, closing_clause: str) -> None:
    """The one line a refusal earns, by what a person has to find in it.

    *closing_clause* is the end of the monitor's own message — for a store
    refusal, :func:`store_answered`; for one the deployment ceiling made, the
    config key it tells the operator to set. The bracketed reason code is
    asserted for every refusal alike: a log reader groups on it without parsing
    prose, and all three refusals here are the monitor declining to attempt a
    write.
    """
    assert _PLAN_NAME in line
    assert str(owner) in line
    assert _CHANNEL in line
    assert _REFUSAL_CODE in line
    assert closing_clause in line


# --- the run record an operator reads --------------------------------------


def _manager_history_item(owner: str | None, failure: BaseException | None) -> dict[str, Any]:
    """The history item the queueserver files for the run that just happened.

    Composed here, from the exception the ``RunEngine`` really raised, because
    this process runs the plan without a manager to file it. Two upstream
    spellings are reproduced, both read from bluesky-queueserver 0.0.25, the
    version this environment resolved — the project pins
    ``bluesky-queueserver-api`` alone, so nothing holds the manager package at
    that version and both spellings are worth re-reading when it moves:

    * ``manager/worker.py``, on a plan that raised: ``plan_state`` is
      ``"failed"`` and ``err_msg`` is ``f"Plan failed: {ex}"``, so the text an
      operator reads is the exception's own ``str`` behind a fixed prefix;
    * ``manager/plan_queue_ops.py::_set_processed_item_as_completed``, which
      files that ``err_msg`` on the item as ``result["msg"]`` beside the
      ``exit_status``.

    Everything after this function is OSPREY's own: the projection decides the
    record's ``status`` from the exit status and lifts the message into
    ``error``.
    """
    kwargs: dict[str, Any] = {
        "motor": _DEVICE_NAME,
        "first": _FIRST_SETPOINT,
        "second": _SECOND_SETPOINT,
    }
    if owner is not None:
        kwargs[RESERVED_OWNER_KWARG] = owner
    if failure is None:
        result = {"exit_status": "completed", "run_uids": [], "msg": ""}
    else:
        result = {"exit_status": "failed", "run_uids": [], "msg": f"Plan failed: {failure}"}
    return {
        "name": _PLAN_NAME,
        "item_type": "plan",
        "kwargs": kwargs,
        "item_uid": _ITEM_UID,
        "meta": {RUN_ID_META_KEY: _RUN_ID},
        "result": result,
    }


def run_record(owner: str | None, failure: BaseException | None = None) -> dict[str, Any]:
    """What ``GET /runs`` publishes for this run, through the real projection."""
    [record] = runs.list_records(
        running_item=None,
        queue_items=[],
        history_items=[_manager_history_item(owner, failure)],
    )
    return record


# --- a narrowing that lands while the plan is running ----------------------


def test_a_narrowing_recorded_while_the_plan_runs_refuses_its_next_write(
    wrapped_plan: tuple[Callable[..., Iterator[Any]], str],
    connector: _RecordingMockConnector,
    control_context_tree: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The whole feature in one row: the chip reaches a plan already in flight.

    The plan's first write lands, the owner's record appears while the run is
    between writes, and the second write is refused — no respawn, no config
    edit, and nothing re-read at enqueue time. The refusal reaches the operator
    twice over and says the same thing both times: as the run record's error,
    which is what a person reads, and as one line in the worker's log, which is
    the only place whose narrowing decided it is recorded.
    """
    plan_function, logger_name = wrapped_plan
    connector.after_first_put = lambda: narrow_for(control_context_tree, NARROWING_OWNER)

    with caplog.at_level(logging.WARNING, logger=logger_name):
        with pytest.raises(FailedStatus) as failure:
            _run(plan_function, NARROWING_OWNER)

    assert connector.puts == [(_CHANNEL, _FIRST_SETPOINT)]

    record = run_record(NARROWING_OWNER, failure.value)
    assert record["status"] == runs.STATUS_ERROR
    # The clause, not the word: see `store_answered`. A row asserting the bare
    # verdict word here passes for a record that could not be read at all, and
    # would report the store refusing a decision nobody made as this person's
    # own narrowing.
    assert store_answered(StoreVerdict.NARROWING) in record["error"]
    assert store_answered(StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE) not in record["error"]
    assert record["owner"] == NARROWING_OWNER
    # The attribution is a channel of its own: the reserved kwarg is never a
    # plan argument, and a record that replayed it as one would invite a
    # consumer to re-enqueue it as one.
    assert RESERVED_OWNER_KWARG not in record["plan_args"]

    warnings = wrapper_warnings(caplog, logger_name)
    assert len(warnings) == 1
    assert_names_the_refusal(
        warnings[0].getMessage(), NARROWING_OWNER, store_answered(StoreVerdict.NARROWING)
    )


def test_two_users_one_deployment_the_unnarrowed_owners_plan_completes(
    wrapped_plan: tuple[Callable[..., Iterator[Any]], str],
    connector: _RecordingMockConnector,
    control_context_tree: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One person's narrowing is theirs alone, on the deployment they share.

    The arrangement is the row above, unchanged down to the moment the record
    appears — only the name on the queue item differs. Without this, a refusal
    there would be explained just as well by a store that refuses everybody
    once any record exists, and the owner would be deciding nothing.
    """
    plan_function, logger_name = wrapped_plan
    connector.after_first_put = lambda: narrow_for(control_context_tree, NARROWING_OWNER)

    with caplog.at_level(logging.WARNING, logger=logger_name):
        _run(plan_function, OTHER_OWNER)

    assert connector.puts == [(_CHANNEL, _FIRST_SETPOINT), (_CHANNEL, _SECOND_SETPOINT)]

    record = run_record(OTHER_OWNER)
    assert record["status"] == runs.STATUS_COMPLETED
    assert record["owner"] == OTHER_OWNER
    assert "error" not in record

    assert wrapper_warnings(caplog, logger_name) == []


# --- a tree the container cannot read --------------------------------------


@pytest.mark.parametrize("owner", [NARROWING_OWNER, OTHER_OWNER])
def test_an_unmounted_tree_refuses_every_owners_write_as_control_context_unavailable(
    owner: str,
    wrapped_plan: tuple[Callable[..., Iterator[Any]], str],
    connector: _RecordingMockConnector,
    control_context_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A lane whose tree volume never arrived refuses, rather than running open.

    The reader fails closed, and it has to: a container that cannot see the
    records would otherwise run everybody's owned plan at the deployment
    ceiling, which is precisely the state the owner exists to prevent. Both
    names are driven because this is not anyone's narrowing — the person who
    narrowed nothing is refused exactly as the person who did.
    """
    plan_function, logger_name = wrapped_plan
    monkeypatch.setenv(
        posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(control_context_tree.parent / "unmounted")
    )
    posture_store.invalidate_cache()

    with caplog.at_level(logging.WARNING, logger=logger_name):
        with pytest.raises(FailedStatus) as failure:
            _run(plan_function, owner)

    assert connector.puts == []

    record = run_record(owner, failure.value)
    assert record["status"] == runs.STATUS_ERROR
    # The closing clause is what tells this apart from a narrowing somebody
    # set. An operator sent to the chip to lift this would be undoing a
    # decision they never made, and the message says so in prose the other arm
    # also uses — so the claim rests on the clause, and the narrowing clause is
    # asserted absent rather than the word.
    assert store_answered(StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE) in record["error"]
    assert store_answered(StoreVerdict.NARROWING) not in record["error"]

    warnings = wrapper_warnings(caplog, logger_name)
    assert len(warnings) == 1
    assert_names_the_refusal(
        warnings[0].getMessage(), owner, store_answered(StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE)
    )


@pytest.mark.parametrize("owner", [NARROWING_OWNER, OTHER_OWNER])
def test_a_tree_without_its_marker_refuses_every_owners_write_as_control_context_unavailable(
    owner: str,
    wrapped_plan: tuple[Callable[..., Iterator[Any]], str],
    connector: _RecordingMockConnector,
    control_context_tree: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A directory at the bind is not a provisioned tree.

    An empty directory is what a mount points at before the deployment has
    built anything, and it reads identically to a tree in which nobody has
    narrowed anything. The marker is what tells those two apart, so a bind
    without one is unreadable rather than permissive.
    """
    plan_function, logger_name = wrapped_plan
    (control_context_tree / posture_store.CONTROL_TREE_MARKER_NAME).unlink()
    posture_store.invalidate_cache()

    with caplog.at_level(logging.WARNING, logger=logger_name):
        with pytest.raises(FailedStatus) as failure:
            _run(plan_function, owner)

    assert connector.puts == []

    record = run_record(owner, failure.value)
    assert record["status"] == runs.STATUS_ERROR
    assert store_answered(StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE) in record["error"]
    assert store_answered(StoreVerdict.NARROWING) not in record["error"]
    assert posture_store.CONTROL_TREE_MARKER_NAME in record["error"]

    warnings = wrapper_warnings(caplog, logger_name)
    assert len(warnings) == 1
    assert_names_the_refusal(
        warnings[0].getMessage(), owner, store_answered(StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE)
    )


def plant_unreadable_record(tree: Path, owner: str) -> Path:
    """Leave *owner* a record file whose bytes are not a record.

    One of the shapes where a record really exists and still cannot be read.
    The tree reader refuses a present record it cannot parse rather than
    reading it as an owner who narrowed nothing, which is what makes this the
    control for the narrowing rows: same owner, same moment, same refusal —
    a different verdict.
    """
    path = tree / owner / control_context.RECORD_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("this is not a control-context record\n", encoding="utf-8")
    posture_store.invalidate_cache()
    return path


def test_an_unreadable_record_is_not_reported_as_that_owners_narrowing(
    wrapped_plan: tuple[Callable[..., Iterator[Any]], str],
    connector: _RecordingMockConnector,
    control_context_tree: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The control for the headline row: refused, but not by anyone's decision.

    The arrangement is that row's exactly, down to the record landing between
    the plan's two writes — only the bytes in it differ. Both runs are refused,
    and what the operator is told has to differ: a narrowing is lifted from the
    person's own chip, while a record the lane cannot read is a deployment to
    fix and a chip that will not touch it. Written because the two messages
    share prose: this row fails for any assertion that reads the narrowing
    arm's verdict word out of the other arm's message.
    """
    plan_function, logger_name = wrapped_plan
    connector.after_first_put = lambda: plant_unreadable_record(
        control_context_tree, NARROWING_OWNER
    )

    with caplog.at_level(logging.WARNING, logger=logger_name):
        with pytest.raises(FailedStatus) as failure:
            _run(plan_function, NARROWING_OWNER)

    assert connector.puts == [(_CHANNEL, _FIRST_SETPOINT)]

    record = run_record(NARROWING_OWNER, failure.value)
    assert record["status"] == runs.STATUS_ERROR
    assert store_answered(StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE) in record["error"]
    assert store_answered(StoreVerdict.NARROWING) not in record["error"]

    warnings = wrapper_warnings(caplog, logger_name)
    assert len(warnings) == 1
    assert_names_the_refusal(
        warnings[0].getMessage(),
        NARROWING_OWNER,
        store_answered(StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE),
    )


# --- an item that names nobody ---------------------------------------------


def test_an_owner_less_item_writes_under_the_deployment_ceiling_alone(
    wrapped_plan: tuple[Callable[..., Iterator[Any]], str],
    connector: _RecordingMockConnector,
    control_context_tree: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Work that belongs to nobody reads nobody's record.

    An item enqueued out-of-band names no owner, and the ladder answers
    ``NO_OWNER`` rather than the account the worker runs as. That is an answer,
    not a failure: the run is governed by the deployment ceiling alone, and a
    narrowing landing mid-run — Alice's, here — governs her plans and not this
    one.
    """
    plan_function, logger_name = wrapped_plan
    connector.after_first_put = lambda: narrow_for(control_context_tree, NARROWING_OWNER)

    with caplog.at_level(logging.WARNING, logger=logger_name):
        _run(plan_function, None)

    assert connector.puts == [(_CHANNEL, _FIRST_SETPOINT), (_CHANNEL, _SECOND_SETPOINT)]

    record = run_record(None)
    assert record["status"] == runs.STATUS_COMPLETED
    # Omitted, not empty: a consumer can tell an unattributed run from one
    # attributed to a nameless person.
    assert "owner" not in record

    assert wrapper_warnings(caplog, logger_name) == []


@pytest.mark.usefixtures("control_context_tree")
def test_an_owner_less_item_is_refused_when_the_deployment_arms_no_writes(
    wrapped_plan: tuple[Callable[..., Iterator[Any]], str],
    connector: _RecordingMockConnector,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ceiling ONLY: nobody's record can widen what the deployment did not arm.

    The row above shows an owner-less run writing, which on its own would also
    be what a store that had stopped being consulted looks like. This is the
    other half — the same run, the same tree, one config key — and its refusal
    has to name the deployment rather than a narrowing, or it sends the person
    who hit it to a chip that cannot lift it.
    """
    plan_function, logger_name = wrapped_plan
    arm_deployment_writes(monkeypatch, enabled=False)

    with caplog.at_level(logging.WARNING, logger=logger_name):
        with pytest.raises(FailedStatus) as failure:
            _run(plan_function, None)

    assert connector.puts == []

    record = run_record(None, failure.value)
    assert record["status"] == runs.STATUS_ERROR
    assert _TYPE_WRITES_ENABLED_KEY in record["error"]
    # Neither store arm: the deployment refused this before any record was
    # consulted, and a message closing on a verdict would send the reader to
    # the chip instead of to the build profile.
    assert store_answered(StoreVerdict.NARROWING) not in record["error"]
    assert store_answered(StoreVerdict.CONTROL_CONTEXT_UNAVAILABLE) not in record["error"]

    warnings = wrapper_warnings(caplog, logger_name)
    assert len(warnings) == 1
    assert_names_the_refusal(warnings[0].getMessage(), NO_OWNER, _TYPE_WRITES_ENABLED_KEY)
