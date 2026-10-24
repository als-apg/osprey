"""One switch, two users, one deployment — from the chip's record to the run record.

The pieces this joins are each pinned on their own elsewhere, and none of those
files can show what an operator actually meets. ``packages/osprey-connectors``
pins the write verdict against a tree; ``tests/services/bluesky_bridge`` pins the
wrapper binding the owner and the single line a refusal earns, both against
connector doubles that raise refusals the test itself wrote; ``tests/services/
bluesky_bridge/test_runs*.py`` pins the run-record projection against literal
manager documents. Here the whole chain runs at once and nobody stands in for
the decision: a real plan wrapper binds the owner a queue item carried, a real
``RunEngine`` schedules a real device's ``set``, the REAL reference monitor in
``osprey_connectors.control_system._base`` asks the REAL write verdict, and the
verdict reads a real provisioned tree. What the monitor answers is then carried out to
the two surfaces a person reads — the run record's ``error`` and the worker's
log — rather than asserted at the seam it was decided in.

That is the point of running it whole: every one of those seams agrees with its
neighbour in isolation, and a deployment where a lane's queueserver holds the
whole per-user tree is where they have to agree with each other.

**Two users, one deployment.** Alice turns her switch off WHILE a plan is
running, which is what the switch is for: a machine switched off from the
header chip refuses the very next write of a plan already in flight, with no
respawn. The plan's own first write is the clock that says it is running — the
connector rewrites the record from inside it, standing in for the other process
the chip really writes from. The same arrangement, with only the name on the
item changed, is what shows the switch is Alice's alone: Bob's plan runs
through it and completes.

**The deployment.** Every row runs on a deployment whose write tier allows
writes, whose target resolves (the connector is stamped ``va``, the simulator),
and whose ``login_tiers`` map names both users, with each user's switch on for
that target before the plan starts. What a row changes is then the one thing it
is about.

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

from bluesky import RunEngine
from bluesky import plan_stubs as bps
from bluesky.utils import FailedStatus
from pydantic import BaseModel, ConfigDict

from osprey.services.bluesky_bridge import qserver_startup, runs, session_upload
from osprey.services.bluesky_bridge.devices.connector import ConnectorSettable
from osprey.services.bluesky_bridge.plan_fields import MovableChannel
from osprey.services.bluesky_bridge.queue_backend import RUN_ID_META_KEY
from osprey.services.bluesky_bridge.session_upload import install_session_plan
from osprey_connectors import control_context, posture_store
from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector
from osprey_connectors.factory import (
    ConnectorFactory,
    isolated_connector_registries,
)
from osprey_connectors.posture_store import (
    NO_OWNER,
    RESERVED_OWNER_KWARG,
)
from osprey_connectors.refusals import STORE_PATH_MARKERS, refusal_phrase
from osprey_connectors_tests._harness import write_payload
from tests.facility.served_tree import in_process_config, served_tree

#: The person whose chip turns the switch off in every row that turns one off.
SWITCHING_OWNER = "alice"
#: The other person on the same deployment, who leaves their switch on.
OTHER_OWNER = "bob"
#: A login the deployment's ``login_tiers`` map does not name.
UNMAPPED_OWNER = "carol"

#: The simulator: a target every deployment resolves, so the verdict's target
#: term passes and the rows reach the owner and the switch.
_TARGET = "va"
#: The machine kind :func:`refusal_phrase` names for :data:`_TARGET`.
_KIND = "simulator"
_CHANNEL = "SR:QUAD:STRENGTH"
_DEVICE_NAME = "quadrupole"
_PLAN_NAME = "two_step_write_plan"
#: The plan writes twice. The first write is the one that says the plan is
#: running; the second is the "next write" a switch turned off has to reach.
_FIRST_SETPOINT = 1.5
_SECOND_SETPOINT = 3.5

#: The connector type the target resolves to, and the deployment's own type.
_CONNECTOR_TYPE = "virtual_accelerator"

#: Both users' tiers, as the resolved render maps them for a shared container.
_LOGIN_TIERS = {SWITCHING_OWNER: "allowed", OTHER_OWNER: "allowed"}

#: The machine-readable half of every refusal here: the monitor declined to
#: attempt the write. Which verdict term decided it is in the message, never in
#: this code, so the code alone never tells one reason from another.
_REFUSAL_CODE = "WRITES_DISABLED"

_RUN_ID = "run-0001"
_ITEM_UID = "item-0001"


class _RecordingInProcessConnector(VAInProcessConnector):
    """The simulated control system, with what reached it recorded.

    Subclassed rather than doubled: every gate these rows are about lives on
    ``ControlSystemConnector`` and runs before ``VAInProcessConnector.write_channel``
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


def arm_deployment_writes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tier: str = "allowed",
    login_tiers: dict[str, str] | None = None,
) -> None:
    """Set the half of the verdict config owns: the tier and the login map.

    *tier* is the deployment's ``writes`` leaf, which the verdict asks first;
    *login_tiers* is the per-login map a container holding the tree reads each
    owner's tier from, :data:`_LOGIN_TIERS` unless a row names another.
    """
    section = {
        "type": _CONNECTOR_TYPE,
        "writes": tier,
        "login_tiers": dict(_LOGIN_TIERS if login_tiers is None else login_tiers),
    }

    def config_value(key: str, default: Any = None) -> Any:
        if key == "control_system":
            return section
        return default

    monkeypatch.setattr("osprey_connectors.config.get_config_value", config_value)


def switch_for(tree: Path, owner: str, state: str, target: str = _TARGET) -> Path:
    """Write *owner*'s record with the switch *state* for *target*, as the chip leaves it.

    The payload is written whole rather than through
    :func:`control_context.write_record`, which keeps the switch already on disk:
    turning the switch is the one change that writer exists not to make.
    """
    payload = control_context.ControlContext(
        target=target, generation=3, switch_epoch=1
    ).to_payload()
    payload["switch"] = {target: state}
    path = write_payload(tree / owner / control_context.RECORD_FILENAME, payload)
    posture_store.invalidate_cache()
    return path


@pytest.fixture
def control_context_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A provisioned control-context tree, bound as a lane's queueserver has it.

    Every user starts with their switch on for the target. The stamped-owner
    rung, the dispatch marker, the read-only mode and the launch pin are
    cleared so a developer's environment cannot decide these rows, and the tree
    bind is what makes ``NO_OWNER`` — rather than the account this process runs
    as — the ladder's answer for a plan that arrived without a name. The bound
    owner is reset on the way out as well as in: a run that failed mid-plan
    would otherwise hand its owner to the next test.
    """
    tree = tmp_path / posture_store.STATE_DIR_NAME
    tree.mkdir(parents=True)
    (tree / posture_store.CONTROL_TREE_MARKER_NAME).write_text("provisioned\n", encoding="utf-8")
    for name in (
        posture_store.CONTROL_OWNER_ENV_VAR,
        posture_store.DISPATCH_RUN_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        "OSPREY_EXECUTION_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(tree))
    for owner in (SWITCHING_OWNER, OTHER_OWNER, UNMAPPED_OWNER):
        switch_for(tree, owner, "on")
    posture_store._owner_var.set(NO_OWNER)
    yield tree
    posture_store._owner_var.set(NO_OWNER)
    posture_store.invalidate_cache()


@pytest.fixture
def armed_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tier that allows writes and a login map naming both users."""
    arm_deployment_writes(monkeypatch)


@pytest.fixture
def connector(armed_deployment: None, tmp_path: Path) -> Iterator[_RecordingInProcessConnector]:  # noqa: ARG001 - the deployment's tier is set before the connector is built
    """The connector a lane builds, through the factory that builds a lane's.

    Built rather than constructed, because the monitor reads the control-target
    stamp that only the factory sets between construction and ``connect()``,
    and the verdict resolves that target. A connector stamped by hand here
    would be this file agreeing with itself about the shape a lane's connector
    has.

    The recording subclass is registered under the deployment's own
    ``virtual_accelerator`` spelling inside
    :func:`~osprey_connectors.factory.isolated_connector_registries`, the
    sanctioned bracket for mutating the factory registries, so the registration
    is restored on the way out and no later test builds this class. Its block
    names no ``serving`` venue, so the factory takes the registered class
    rather than choosing the in-process one itself.
    """
    with isolated_connector_registries():
        ConnectorFactory.register_control_system(_CONNECTOR_TYPE, _RecordingInProcessConnector)
        instance = asyncio.run(
            ConnectorFactory.create_control_system_connector(
                {
                    "type": _CONNECTOR_TYPE,
                    "connector": {
                        _CONNECTOR_TYPE: _without_venue(
                            in_process_config(
                                served_tree(tmp_path / "served", [_CHANNEL]),
                                response_delay_ms=0,
                            )
                        )
                    },
                },
                control_target=_TARGET,
            )
        )
        assert isinstance(instance, _RecordingInProcessConnector)
        yield instance


def _without_venue(block: dict) -> dict:
    """*block* with no ``serving`` leaf, so the registry answers for the type."""
    return {key: value for key, value in block.items() if key != "serving"}


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
    request: pytest.FixtureRequest, connector: _RecordingInProcessConnector
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


def refused_as(reason: str) -> str:
    """The phrase a refusal for *reason* carries, for this deployment's target.

    Composed from :func:`refusal_phrase` rather than typed out, so a reworded
    phrase fails these rows instead of leaving them agreeing with a spelling
    nothing answers any more. Every phrase is distinct, so asserting one in a
    message and another absent tells the two reasons apart.
    """
    return refusal_phrase(reason, _KIND)


def assert_names_the_refusal(line: str, owner: str | object, reason: str) -> None:
    """The one line a refusal earns, by what a person has to find in it.

    The bracketed reason code is asserted for every refusal alike: a log reader
    groups on it without parsing prose, and every refusal here is the monitor
    declining to attempt a write. The phrase for *reason* is the operator's
    words for which verdict term decided it.
    """
    assert _PLAN_NAME in line
    assert str(owner) in line
    assert _CHANNEL in line
    assert _REFUSAL_CODE in line
    assert refused_as(reason) in line


def assert_names_no_store_path(message: str) -> None:
    """A refusal sends nobody to a file: no store path, no tree marker, no key."""
    for marker in (*STORE_PATH_MARKERS, posture_store.CONTROL_TREE_MARKER_NAME, "writes_enabled"):
        assert marker not in message


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


# --- a switch turned off while the plan is running ------------------------


def test_turning_the_switch_off_while_the_plan_runs_refuses_its_next_write(
    wrapped_plan: tuple[Callable[..., Iterator[Any]], str],
    connector: _RecordingInProcessConnector,
    control_context_tree: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The whole feature in one row: the chip reaches a plan already in flight.

    The plan's first write lands, the owner's switch goes off while the run is
    between writes, and the second write is refused — no respawn, no config
    edit, and nothing re-read at enqueue time. The refusal reaches the operator
    twice over and says the same thing both times: as the run record's error,
    which is what a person reads, and as one line in the worker's log, which is
    the only place whose switch decided it is recorded.
    """
    plan_function, logger_name = wrapped_plan
    connector.after_first_put = lambda: switch_for(control_context_tree, SWITCHING_OWNER, "off")

    with caplog.at_level(logging.WARNING, logger=logger_name):
        with pytest.raises(FailedStatus) as failure:
            _run(plan_function, SWITCHING_OWNER)

    assert connector.puts == [(_CHANNEL, _FIRST_SETPOINT)]

    record = run_record(SWITCHING_OWNER, failure.value)
    assert record["status"] == runs.STATUS_ERROR
    assert refused_as("switch_off") in record["error"]
    assert refused_as("record_unavailable") not in record["error"]
    assert_names_no_store_path(record["error"])
    assert record["owner"] == SWITCHING_OWNER
    # The attribution is a channel of its own: the reserved kwarg is never a
    # plan argument, and a record that replayed it as one would invite a
    # consumer to re-enqueue it as one.
    assert RESERVED_OWNER_KWARG not in record["plan_args"]

    warnings = wrapper_warnings(caplog, logger_name)
    assert len(warnings) == 1
    assert_names_the_refusal(warnings[0].getMessage(), SWITCHING_OWNER, "switch_off")


def test_two_users_one_deployment_the_other_owners_plan_completes(
    wrapped_plan: tuple[Callable[..., Iterator[Any]], str],
    connector: _RecordingInProcessConnector,
    control_context_tree: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One person's switch is theirs alone, on the deployment they share.

    The arrangement is the row above, unchanged down to the moment Alice's
    switch goes off — only the name on the queue item differs. Without this, a
    refusal there would be explained just as well by a verdict that refuses
    everybody once any switch is off, and the owner would be deciding nothing.
    """
    plan_function, logger_name = wrapped_plan
    connector.after_first_put = lambda: switch_for(control_context_tree, SWITCHING_OWNER, "off")

    with caplog.at_level(logging.WARNING, logger=logger_name):
        _run(plan_function, OTHER_OWNER)

    assert connector.puts == [(_CHANNEL, _FIRST_SETPOINT), (_CHANNEL, _SECOND_SETPOINT)]

    record = run_record(OTHER_OWNER)
    assert record["status"] == runs.STATUS_COMPLETED
    assert record["owner"] == OTHER_OWNER
    assert "error" not in record

    assert wrapper_warnings(caplog, logger_name) == []


# --- a tree the container cannot read --------------------------------------


@pytest.mark.parametrize("owner", [SWITCHING_OWNER, OTHER_OWNER])
def test_an_unmounted_tree_refuses_every_owners_write_as_record_unavailable(
    owner: str,
    wrapped_plan: tuple[Callable[..., Iterator[Any]], str],
    connector: _RecordingInProcessConnector,
    control_context_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A lane whose tree volume never arrived refuses, rather than running open.

    The reader fails closed, and it has to: a container that cannot see the
    records would otherwise have to guess everybody's switch. Both names are
    driven because this is nobody's switch — the person whose switch is on is
    refused exactly as anyone else would be.
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
    # An operator sent to the chip to fix this would be flipping a switch that
    # was never the problem, so the phrase is the unreadable-record one and not
    # the switch-off one.
    assert refused_as("record_unavailable") in record["error"]
    assert refused_as("switch_off") not in record["error"]
    assert_names_no_store_path(record["error"])

    warnings = wrapper_warnings(caplog, logger_name)
    assert len(warnings) == 1
    assert_names_the_refusal(warnings[0].getMessage(), owner, "record_unavailable")


@pytest.mark.parametrize("owner", [SWITCHING_OWNER, OTHER_OWNER])
def test_a_tree_without_its_marker_refuses_every_owners_write_as_record_unavailable(
    owner: str,
    wrapped_plan: tuple[Callable[..., Iterator[Any]], str],
    connector: _RecordingInProcessConnector,
    control_context_tree: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A directory at the bind is not a provisioned tree.

    An empty directory is what a mount points at before the deployment has
    built anything. The marker is what tells that apart from a tree, so a bind
    without one is unreadable rather than permissive — even with each owner's
    switched-on record sitting in it. The refusal names no marker file: the
    phrase sends the reader to the deployment admin, not to a path.
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
    assert refused_as("record_unavailable") in record["error"]
    assert refused_as("switch_off") not in record["error"]
    assert_names_no_store_path(record["error"])

    warnings = wrapper_warnings(caplog, logger_name)
    assert len(warnings) == 1
    assert_names_the_refusal(warnings[0].getMessage(), owner, "record_unavailable")


def plant_unreadable_record(tree: Path, owner: str) -> Path:
    """Leave *owner* a record file whose bytes are not a record.

    One of the shapes where a record really exists and still cannot be read.
    The tree reader refuses a present record it cannot parse rather than
    reading it as a switch that is off, which is what makes this the control
    for the switch-off row: same owner, same moment, same refusal — a different
    reason.
    """
    path = tree / owner / control_context.RECORD_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("this is not a control-context record\n", encoding="utf-8")
    posture_store.invalidate_cache()
    return path


def test_an_unreadable_record_is_not_reported_as_that_owners_switch_off(
    wrapped_plan: tuple[Callable[..., Iterator[Any]], str],
    connector: _RecordingInProcessConnector,
    control_context_tree: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The control for the headline row: refused, but not by anyone's switch.

    The arrangement is that row's exactly, down to the record changing between
    the plan's two writes — only the bytes in it differ. Both runs are refused,
    and what the operator is told has to differ: a switch is turned back on
    from the person's own chip, while a record the lane cannot read is a
    deployment to fix and a chip that will not touch it.
    """
    plan_function, logger_name = wrapped_plan
    connector.after_first_put = lambda: plant_unreadable_record(
        control_context_tree, SWITCHING_OWNER
    )

    with caplog.at_level(logging.WARNING, logger=logger_name):
        with pytest.raises(FailedStatus) as failure:
            _run(plan_function, SWITCHING_OWNER)

    assert connector.puts == [(_CHANNEL, _FIRST_SETPOINT)]

    record = run_record(SWITCHING_OWNER, failure.value)
    assert record["status"] == runs.STATUS_ERROR
    assert refused_as("record_unavailable") in record["error"]
    assert refused_as("switch_off") not in record["error"]

    warnings = wrapper_warnings(caplog, logger_name)
    assert len(warnings) == 1
    assert_names_the_refusal(warnings[0].getMessage(), SWITCHING_OWNER, "record_unavailable")


# --- whose tier governs the write ------------------------------------------


@pytest.mark.usefixtures("control_context_tree")
def test_an_owner_missing_from_the_login_map_is_refused(
    wrapped_plan: tuple[Callable[..., Iterator[Any]], str],
    connector: _RecordingInProcessConnector,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A container cannot use its own tier for another login's write.

    Carol's record is in the tree with her switch on, but the deployment's
    ``login_tiers`` map does not name her, so her tier is not known and the
    verdict refuses before her record is read.
    """
    plan_function, logger_name = wrapped_plan

    with caplog.at_level(logging.WARNING, logger=logger_name):
        with pytest.raises(FailedStatus) as failure:
            _run(plan_function, UNMAPPED_OWNER)

    assert connector.puts == []

    record = run_record(UNMAPPED_OWNER, failure.value)
    assert record["status"] == runs.STATUS_ERROR
    assert refused_as("owner_unknown") in record["error"]
    assert refused_as("switch_off") not in record["error"]

    warnings = wrapper_warnings(caplog, logger_name)
    assert len(warnings) == 1
    assert_names_the_refusal(warnings[0].getMessage(), UNMAPPED_OWNER, "owner_unknown")


@pytest.mark.usefixtures("control_context_tree")
def test_a_demoted_owners_stale_switch_on_is_refused(
    wrapped_plan: tuple[Callable[..., Iterator[Any]], str],
    connector: _RecordingInProcessConnector,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The tier is read before the record, so a demoted login's "on" stays inert.

    Alice's record still says "on" from before her login was mapped ``never``.
    Her plan is refused for her tier, not her switch, and Bob — still allowed,
    on the same deployment — writes through the same arrangement.
    """
    plan_function, logger_name = wrapped_plan
    arm_deployment_writes(
        monkeypatch, login_tiers={SWITCHING_OWNER: "never", OTHER_OWNER: "allowed"}
    )

    with caplog.at_level(logging.WARNING, logger=logger_name):
        with pytest.raises(FailedStatus) as failure:
            _run(plan_function, SWITCHING_OWNER)

    assert connector.puts == []

    record = run_record(SWITCHING_OWNER, failure.value)
    assert record["status"] == runs.STATUS_ERROR
    assert refused_as("tier_never") in record["error"]
    assert refused_as("switch_off") not in record["error"]

    warnings = wrapper_warnings(caplog, logger_name)
    assert len(warnings) == 1
    assert_names_the_refusal(warnings[0].getMessage(), SWITCHING_OWNER, "tier_never")

    _run(plan_function, OTHER_OWNER)
    assert connector.puts == [(_CHANNEL, _FIRST_SETPOINT), (_CHANNEL, _SECOND_SETPOINT)]


# --- an item that names nobody ---------------------------------------------


@pytest.mark.usefixtures("control_context_tree")
def test_an_owner_less_item_is_refused_as_owner_unknown(
    wrapped_plan: tuple[Callable[..., Iterator[Any]], str],
    connector: _RecordingInProcessConnector,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Work that belongs to nobody has no switch to follow, so it does not write.

    An item enqueued out-of-band names no owner, and the ladder answers
    ``NO_OWNER`` rather than the account the worker runs as. Nothing is
    governed by the tier alone: the first write is refused, with every user's
    switch on.
    """
    plan_function, logger_name = wrapped_plan

    with caplog.at_level(logging.WARNING, logger=logger_name):
        with pytest.raises(FailedStatus) as failure:
            _run(plan_function, None)

    assert connector.puts == []

    record = run_record(None, failure.value)
    assert record["status"] == runs.STATUS_ERROR
    assert refused_as("owner_unknown") in record["error"]
    # Omitted, not empty: a consumer can tell an unattributed run from one
    # attributed to a nameless person.
    assert "owner" not in record

    warnings = wrapper_warnings(caplog, logger_name)
    assert len(warnings) == 1
    assert_names_the_refusal(warnings[0].getMessage(), NO_OWNER, "owner_unknown")


@pytest.mark.usefixtures("control_context_tree")
def test_an_owner_less_item_on_a_tier_of_never_is_refused_by_the_tier(
    wrapped_plan: tuple[Callable[..., Iterator[Any]], str],
    connector: _RecordingInProcessConnector,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The tier is asked first, so its refusal is the one a person reads.

    The same run, the same tree, one config value. The refusal has to name the
    tier rather than the owner or a switch, or it sends the person who hit it to
    a chip that cannot lift it.
    """
    plan_function, logger_name = wrapped_plan
    arm_deployment_writes(monkeypatch, tier="never")

    with caplog.at_level(logging.WARNING, logger=logger_name):
        with pytest.raises(FailedStatus) as failure:
            _run(plan_function, None)

    assert connector.puts == []

    record = run_record(None, failure.value)
    assert record["status"] == runs.STATUS_ERROR
    assert refused_as("tier_never") in record["error"]
    assert refused_as("owner_unknown") not in record["error"]
    assert refused_as("switch_off") not in record["error"]
    assert_names_no_store_path(record["error"])

    warnings = wrapper_warnings(caplog, logger_name)
    assert len(warnings) == 1
    assert_names_the_refusal(warnings[0].getMessage(), NO_OWNER, "tier_never")
