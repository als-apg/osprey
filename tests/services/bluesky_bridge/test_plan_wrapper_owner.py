"""The owner a queued plan carries, through both worker plan wrappers.

An owner reaches a queueserver worker as one reserved kwarg on the queue item,
and the wrapper is where it stops being an argument and becomes context: the
wrapper's whole body runs inside
:func:`~osprey_connectors.posture_store.bind_owner`, so every write the plan
makes is gated against the chip of the person who queued it, and nothing the
plan's own schema validates ever sees the reserved key.

There are exactly two such wrappers — ``qserver_startup._make_plan_function``
for catalog plans and the one ``session_upload.install_session_plan`` installs
for session plans — and a session plan that skipped this binding would be the
quiet way past a gate the catalog path enforces. So every row here runs
against both, from one stub plan written twice: once as a Python generator for
the catalog wrapper's ``spec.plan``, once as plan-file source text for the
session wrapper to ``exec``.

What this file owes, and what it deliberately leaves alone. Most of it is the
properties only a *generator* wrapper has: the owner is visible at every yield,
not merely at the call, and the binding is gone the moment the generator closes
— whether it ran to completion or was closed mid-plan, which is what an aborted
run does. Those rows drive the wrappers as the plain generators they are.
``tests/connectors/test_owner_context.py`` owns the ladder and the sentinel and
is not restated.

The rows about the *line a refusal earns* cannot be driven that way, because a
refusal wears two shapes. Raised in the plan's own frame it reaches the wrapper
as itself; raised where a plan really writes — inside a device's ``set()``, on
the task the RunEngine scheduled it onto — the wrapper never sees it at all,
only ``bluesky.utils.FailedStatus`` thrown back into the plan with the refusal
inside. A wrapper that recognized one shape and not the other would write the
line for a refusal a test wrote and stay silent for the ones an operator meets,
so the second shape is driven through a real ``RunEngine``.
``test_bind_owner_runengine.py`` owns what that arrangement pins about the
*binding*; what is pinned here is what the wrapper says.

Every test stamps the control-context tree, because that is the deployment
shape these wrappers run in: a lane's queueserver holds the whole per-user
tree and no chip of its own, so "no owner" is the ladder's final answer there
and ``NO_OWNER`` is what an owner-less plan actually runs under.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from osprey.audit.posture import POSTURE_ENV_VAR
from osprey.services.bluesky_bridge import qserver_startup, session_upload
from osprey.services.bluesky_bridge.session_upload import install_session_plan
from osprey_connectors import posture_store
from osprey_connectors.errors import ChannelWriteBlockedError, ChannelWriteFailedError
from osprey_connectors.posture_store import (
    NO_OWNER,
    RESERVED_OWNER_KWARG,
    StoreVerdict,
    current_owner,
)

pytest.importorskip("bluesky")
pytest.importorskip("ophyd_async")

from bluesky import RunEngine
from bluesky import plan_stubs as bps
from bluesky.utils import FailedStatus
from ophyd.status import StatusBase

from osprey.services.bluesky_bridge.devices.connector import ConnectorSettable
from osprey.services.bluesky_bridge.plan_fields import MovableChannel
from tests.services.bluesky_bridge.test_connector_devices import FakeConnector

_OWNER = "alice"
_PLAN_NAME = "stub_plan"
_CHANNEL = "SR:BEND:SETPOINT"
_DEVICE_NAME = "motor"
_SETPOINT = 3.5

_REFUSAL_TARGET = "vacuum"
_CONNECTOR_TYPE = "epics"


def refuse_a_narrowed_write() -> None:
    """Raise the refusal the reference monitor raises for a narrowed target.

    Composed by the monitor's own result builder and raised through its own
    denial contract, rather than typed out here: the reason code and the
    operator-facing sentence are what these rows look for in the wrapper's
    warning, and a message written by the test would agree with that line
    whatever either of them said.

    Every caller runs under :func:`monitor_refusal_inputs`, which settles the
    three environment answers that fork this refusal onto other wordings.

    Public because the session wrapper's plan source imports it: both wrappers
    have to meet the same refusal, and a plan file is exec'd in a namespace of
    its own that sees nothing of this module.
    """
    from osprey_connectors.control_system import base as connector_base

    connector_base.raise_for_write_result(
        connector_base._writes_disabled_result(
            _CHANNEL,
            _SETPOINT,
            _CONNECTOR_TYPE,
            _REFUSAL_TARGET,
            store_verdict=StoreVerdict.NARROWING,
        )
    )
    raise AssertionError("a refused write result must raise")


def _monitor_refusal() -> ChannelWriteBlockedError:
    """That same refusal as an object, for a caller that hands it on."""
    try:
        refuse_a_narrowed_write()
    except ChannelWriteBlockedError as refusal:
        return refusal
    raise AssertionError("unreachable")


class _Params(BaseModel):
    """A plan schema that forbids what it does not declare.

    ``extra="forbid"`` is the point: it is what makes "the reserved kwarg never
    reaches the schema" an assertion rather than an assumption — an unpopped
    key would fail validation here instead of quietly arriving as a plan
    parameter.
    """

    model_config = ConfigDict(extra="forbid")

    steps: int = 2


def _owner_probe_plan(_devices: Any, params: Any) -> Any:
    """Yield who the run belongs to, once per step."""
    for _ in range(params.steps):
        yield current_owner()


def _refusing_plan(_devices: Any, _params: Any) -> Any:
    """Run one message, then meet a gated write that refuses."""
    yield current_owner()
    refuse_a_narrowed_write()


_PROBE_SOURCE = '''PLAN_METADATA = {"name": "stub_plan"}

from pydantic import BaseModel, ConfigDict

from osprey_connectors.posture_store import current_owner


class PARAMS(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: int = 2


def build_plan(devices, params):
    """Yield who the run belongs to, once per step."""
    for _ in range(params.steps):
        yield current_owner()
'''

_REFUSAL_SOURCE = '''PLAN_METADATA = {"name": "stub_plan"}

from pydantic import BaseModel, ConfigDict

from osprey_connectors.posture_store import current_owner

from tests.services.bluesky_bridge.test_plan_wrapper_owner import refuse_a_narrowed_write


class PARAMS(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: int = 2


def build_plan(devices, params):
    """Run one message, then meet a gated write that refuses."""
    yield current_owner()
    refuse_a_narrowed_write()
'''

#: What the channel reported back when the value was sent and did not stick.
_FAILURE_OUTCOME = "MISMATCH"


def _failing_plan(_devices: Any, _params: Any) -> Any:
    """Run one message, then meet a write the channel did not confirm."""
    yield current_owner()
    raise ChannelWriteFailedError(_CHANNEL, _FAILURE_OUTCOME)


_FAILURE_SOURCE = f'''PLAN_METADATA = {{"name": "stub_plan"}}

from pydantic import BaseModel, ConfigDict

from osprey_connectors.errors import ChannelWriteFailedError
from osprey_connectors.posture_store import current_owner


class PARAMS(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: int = 2


def build_plan(devices, params):
    """Run one message, then meet a write the channel did not confirm."""
    yield current_owner()
    raise ChannelWriteFailedError({_CHANNEL!r}, {_FAILURE_OUTCOME!r})
'''


@pytest.fixture(autouse=True)
def clean_owner_environment(monkeypatch: pytest.MonkeyPatch):
    """The lane worker's shape, and no owner left bound by a neighbour.

    The stamped-owner rung is cleared so a deployment that exports one cannot
    decide these tests, and the tree bind is stamped so the ladder's last rung
    is nobody rather than this test process's account. The variable is reset on
    the way out as well as the way in: a test that binds an owner and then
    fails mid-plan would otherwise hand its owner to the next test.
    """
    monkeypatch.delenv(posture_store.CONTROL_OWNER_ENV_VAR, raising=False)
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, "/var/osprey/control")
    posture_store._owner_var.set(NO_OWNER)
    yield
    posture_store._owner_var.set(NO_OWNER)


@pytest.fixture(autouse=True)
def monitor_refusal_inputs(monkeypatch: pytest.MonkeyPatch):
    """Settle the three answers that decide which refusal wording is raised.

    Each forks :func:`refuse_a_narrowed_write` onto a different message, and a
    test process answers all three the way no deployment does: a build profile
    it cannot read reads as writes-disabled deployment-wide, and an inherited
    readonly mode or launch pin refuses before the target is consulted at all.
    Settled here rather than at the raise, so what a row meets is the
    narrowed-target arm — the one a queued plan meets — on any machine.
    """

    def armed(key: str, default: Any = None) -> Any:
        return {"writes_enabled": True} if key == "control_system" else default

    monkeypatch.setattr("osprey_connectors.config.get_config_value", armed)
    monkeypatch.delenv(posture_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
    monkeypatch.delenv(POSTURE_ENV_VAR, raising=False)


@pytest.fixture
def monitor_refusal() -> ChannelWriteBlockedError:
    """The refusal these rows are written against, as the monitor raises it."""
    return _monitor_refusal()


def _catalog_wrapper(plan: Any) -> Any:
    """A catalog plan as ``qserver_startup`` hands it to the manager."""
    spec = SimpleNamespace(
        name=_PLAN_NAME,
        schema=_Params,
        plan=plan,
        description="A stub plan that reports its owner.",
    )
    return qserver_startup._make_plan_function(spec, devices={})


def _session_wrapper(source: str) -> Any:
    """A session plan as ``session_upload`` installs it in the worker namespace."""
    namespace: dict[str, Any] = {"RE": object(), "__name__": "__main__"}
    return install_session_plan(namespace, _PLAN_NAME, source)


@pytest.fixture(params=["catalog", "session"])
def probe_wrapper(request: pytest.FixtureRequest) -> Any:
    """Both wrappers, each around the plan that yields ``current_owner()``."""
    if request.param == "catalog":
        return _catalog_wrapper(_owner_probe_plan)
    return _session_wrapper(_PROBE_SOURCE)


@pytest.fixture(params=["catalog", "session"])
def refusing_wrapper(request: pytest.FixtureRequest) -> tuple[Any, str]:
    """Both wrappers around the refusing plan, with the logger each one warns on."""
    if request.param == "catalog":
        return _catalog_wrapper(_refusing_plan), qserver_startup.logger.name
    return _session_wrapper(_REFUSAL_SOURCE), session_upload.logger.name


@pytest.fixture(params=["catalog", "session"])
def failing_wrapper(request: pytest.FixtureRequest) -> tuple[Any, str]:
    """Both wrappers around the plan whose write failed rather than was refused."""
    if request.param == "catalog":
        return _catalog_wrapper(_failing_plan), qserver_startup.logger.name
    return _session_wrapper(_FAILURE_SOURCE), session_upload.logger.name


def _warnings_from(caplog: pytest.LogCaptureFixture, logger_name: str) -> list[logging.LogRecord]:
    """The wrapper's own warnings, with any neighbour's out of the count."""
    return [
        record
        for record in caplog.records
        if record.name == logger_name and record.levelno == logging.WARNING
    ]


# --- the binding, over the generator's whole life ---------------------------


def test_the_owner_is_bound_at_every_yield(probe_wrapper: Any) -> None:
    """Not just at the call: a plan's writes happen between its yields.

    The wrapper is a generator function, so its body does not start running
    until the RunEngine pulls the first message and is re-entered at every
    message after it. A binding that only held while the call returned would
    be gone by the time the plan wrote anything.
    """
    owners = list(probe_wrapper(**{RESERVED_OWNER_KWARG: _OWNER, "steps": 3}))

    assert owners == [_OWNER, _OWNER, _OWNER]


def test_an_owner_less_call_runs_under_no_owner(probe_wrapper: Any) -> None:
    """A plan that reached the queue without a name is governed by the ceiling."""
    owners = list(probe_wrapper(steps=2))

    assert owners == [NO_OWNER, NO_OWNER]
    assert all(owner is NO_OWNER for owner in owners)


def test_the_binding_is_gone_when_the_generator_finishes(probe_wrapper: Any) -> None:
    """The next run in this worker must not be gated against this run's owner."""
    assert list(probe_wrapper(**{RESERVED_OWNER_KWARG: _OWNER})) == [_OWNER, _OWNER]

    assert posture_store._owner_var.get() is NO_OWNER
    assert current_owner() is NO_OWNER


def test_the_binding_is_gone_when_the_generator_is_closed_mid_plan(probe_wrapper: Any) -> None:
    """An aborted run is the case that leaks: the plan never reaches its end.

    Queueserver closes a plan's generator when a run is aborted or the
    environment is destroyed, which raises ``GeneratorExit`` inside the body
    — so the reset has to be in a ``finally``-shaped block, not after the last
    statement.
    """
    plan = probe_wrapper(**{RESERVED_OWNER_KWARG: _OWNER})
    assert next(plan) == _OWNER

    plan.close()

    assert posture_store._owner_var.get() is NO_OWNER


# --- the reserved kwarg, against the plan's own schema ----------------------


def test_the_reserved_kwarg_never_reaches_the_plan_schema(probe_wrapper: Any) -> None:
    """It is popped before ``PARAMS`` validation, so no plan declares it."""
    assert list(probe_wrapper(**{RESERVED_OWNER_KWARG: _OWNER, "steps": 1})) == [_OWNER]


def test_an_undeclared_kwarg_is_still_refused(probe_wrapper: Any) -> None:
    """The control for the row above: this schema really does forbid extras.

    Without this, "the reserved key validated fine" would be true of every key
    and would pin nothing.
    """
    with pytest.raises(ValidationError):
        list(probe_wrapper(**{RESERVED_OWNER_KWARG: _OWNER, "bogus": 1}))


# --- the one warning a refusal earns ---------------------------------------


def test_a_refused_write_logs_exactly_one_warning_naming_the_owner(
    refusing_wrapper: tuple[Any, str],
    monitor_refusal: ChannelWriteBlockedError,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One line, naming the plan, the owner, the channel and the reason.

    A refusal mid-plan reaches the operator as the run record's error, which
    says nothing about *whose* narrowing refused it. This line is where that
    is recorded — once per refused run, not once per retry — and it is the
    whole of the wrapper's response: no audit-ledger entry, and the exception
    travels on untouched.
    """
    wrapper, logger_name = refusing_wrapper

    with caplog.at_level(logging.WARNING, logger=logger_name):
        with pytest.raises(ChannelWriteBlockedError):
            list(wrapper(**{RESERVED_OWNER_KWARG: _OWNER}))

    warnings = _warnings_from(caplog, logger_name)
    assert len(warnings) == 1
    line = warnings[0].getMessage()
    # The refused write is the event the line is about, and the plan and its
    # owner are a parenthetical to it: a head clause naming the plan as the
    # actor would read as though the plan, or the person it belongs to, did the
    # refusing.
    assert line.startswith(f"a write to {_CHANNEL} was refused")
    assert _PLAN_NAME in line
    assert _OWNER in line
    # Both halves of the reason: the refusal's own code, which a log reader can
    # group on without parsing prose, and the verdict word the operator-facing
    # message closes with, carried in from the refusal rather than re-derived.
    assert monitor_refusal.reason in line
    assert StoreVerdict.NARROWING in line

    assert posture_store._owner_var.get() is NO_OWNER


def test_the_refusal_travels_on_unchanged(
    refusing_wrapper: tuple[Any, str],
    monitor_refusal: ChannelWriteBlockedError,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The worker reports ``Plan failed: {ex}``, so the text must survive.

    Logging a refusal is not handling it: swallowing it, or re-wrapping it in
    the wrapper's own exception, would leave the run's error saying something
    other than what the reference monitor said.
    """
    wrapper, logger_name = refusing_wrapper

    with caplog.at_level(logging.WARNING, logger=logger_name):
        with pytest.raises(ChannelWriteBlockedError) as refusal:
            list(wrapper(**{RESERVED_OWNER_KWARG: _OWNER}))

    assert type(refusal.value) is ChannelWriteBlockedError
    assert str(refusal.value) == str(monitor_refusal)
    assert refusal.value.channel_address == _CHANNEL
    assert refusal.value.reason == "WRITES_DISABLED"


def test_a_write_that_failed_in_the_plans_own_frame_earns_no_line(
    failing_wrapper: tuple[Any, str], caplog: pytest.LogCaptureFixture
) -> None:
    """A channel that disagreed is nobody's narrowing, so nobody is named.

    The handler is offered every exception the plan raises, and this one is a
    write that was attempted: naming an owner for it would attribute to a
    person a disagreement between a setpoint and a readback. The sibling row
    over a real RunEngine pins the same property for the other shape a failed
    write arrives in.
    """
    wrapper, logger_name = failing_wrapper

    with caplog.at_level(logging.WARNING, logger=logger_name):
        with pytest.raises(ChannelWriteFailedError) as failure:
            list(wrapper(**{RESERVED_OWNER_KWARG: _OWNER}))

    assert _warnings_from(caplog, logger_name) == []
    assert failure.value.channel_address == _CHANNEL
    assert posture_store._owner_var.get() is NO_OWNER


def test_an_owner_less_refusal_names_the_sentinel(
    refusing_wrapper: tuple[Any, str], caplog: pytest.LogCaptureFixture
) -> None:
    """The line that says a plan ran at the ceiling still has to be written.

    The owner in it is ``NO_OWNER``, which prints and nothing more — a
    sentinel that raised on formatting would cost exactly this line.
    """
    wrapper, logger_name = refusing_wrapper

    with caplog.at_level(logging.WARNING, logger=logger_name):
        with pytest.raises(ChannelWriteBlockedError):
            list(wrapper())

    warnings = _warnings_from(caplog, logger_name)
    assert len(warnings) == 1
    assert str(NO_OWNER) in warnings[0].getMessage()


# --- the same refusal, where a plan actually writes -------------------------


class _DeviceParams(BaseModel):
    """A plan schema declaring the one channel the plan drives.

    The movable role is what makes the device reach the plan at all: a wrapper
    hands a plan the channels its params declare and nothing else.
    """

    model_config = ConfigDict(extra="forbid")

    motor: MovableChannel = _DEVICE_NAME
    setpoint: float = _SETPOINT


def _write_plan(devices: Any, params: Any) -> Any:
    """Drive the declared movable once."""
    yield from bps.mv(devices[params.motor], params.setpoint)


_WRITE_SOURCE = f'''PLAN_METADATA = {{"name": {_PLAN_NAME!r}}}

from bluesky import plan_stubs as bps
from pydantic import BaseModel, ConfigDict

from osprey.services.bluesky_bridge.plan_fields import MovableChannel


class PARAMS(BaseModel):
    model_config = ConfigDict(extra="forbid")

    motor: MovableChannel = {_DEVICE_NAME!r}
    setpoint: float = {_SETPOINT!r}


def build_plan(devices, params):
    """Drive the declared movable once."""
    yield from bps.mv(devices[params.motor], params.setpoint)
'''


class _RefusingConnector(FakeConnector):
    """A connector whose write the reference monitor refuses.

    Stands in for the transport alone, and refuses where a real refusal is
    decided: inside the write call, which the device has by then scheduled onto
    a task of the RunEngine's own. Nothing is sent — a refusal means the
    channel was left alone — so the base class's call log stays empty and the
    readback never moves.
    """

    def __init__(self) -> None:
        super().__init__(readbacks={_CHANNEL: 0.0})

    async def write_channel_checked(self, channel_address: str, value: Any, **kwargs: Any):  # noqa: ARG002 - the connector write signature
        refuse_a_narrowed_write()


def _device(connector: Any) -> ConnectorSettable:
    """The movable the plan drives, writing through *connector*.

    Left aliased (no separate readback) and confirmed, so one ``set`` is
    exactly one connector call with no settle poll behind it and the run
    reaches its refusal without waiting on anything.
    """
    return ConnectorSettable(connector, _CHANNEL, name=_DEVICE_NAME)


def _catalog_write_wrapper(device: ConnectorSettable) -> Any:
    """The device-driving plan as ``qserver_startup`` hands it to the manager."""
    spec = SimpleNamespace(
        name=_PLAN_NAME,
        schema=_DeviceParams,
        plan=_write_plan,
        description="A stub plan that drives one channel.",
    )
    return qserver_startup._make_plan_function(spec, devices={_DEVICE_NAME: device})


def _session_write_wrapper(device: ConnectorSettable) -> Any:
    """The device-driving plan as ``session_upload`` installs it in the worker."""
    namespace: dict[str, Any] = {"__name__": "__main__", _DEVICE_NAME: device}
    return install_session_plan(namespace, _PLAN_NAME, _WRITE_SOURCE)


@pytest.fixture(params=["catalog", "session"])
def writing_wrapper(request: pytest.FixtureRequest) -> tuple[Any, str]:
    """Both wrappers around the plan that writes, with the logger each warns on."""
    if request.param == "catalog":
        return _catalog_write_wrapper, qserver_startup.logger.name
    return _session_write_wrapper, session_upload.logger.name


def test_a_refusal_inside_a_device_write_earns_the_same_one_line(
    writing_wrapper: tuple[Any, str],
    monitor_refusal: ChannelWriteBlockedError,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The shape every refusal an operator meets actually has.

    A plan writes through a device, and ``ConnectorSettable.set`` is
    ``AsyncStatus``-wrapped: the refusal is raised in the task the RunEngine
    scheduled, which reports it by throwing its own ``FailedStatus`` into the
    suspended plan. Recognizing only a refusal raised in the plan's own frame
    would leave this — the one that happens on a machine — unrecorded, with
    nothing anywhere saying whose narrowing the run was judged against.
    """
    make_plan, logger_name = writing_wrapper
    connector = _RefusingConnector()
    plan_function = make_plan(_device(connector))

    with caplog.at_level(logging.WARNING, logger=logger_name):
        with pytest.raises(FailedStatus) as failure:
            RunEngine(context_managers=[])(plan_function(**{RESERVED_OWNER_KWARG: _OWNER}))

    warnings = _warnings_from(caplog, logger_name)
    assert len(warnings) == 1
    line = warnings[0].getMessage()
    assert _PLAN_NAME in line
    assert _OWNER in line
    assert _CHANNEL in line
    assert monitor_refusal.reason in line
    assert StoreVerdict.NARROWING in line

    # The run still fails on the monitor's own wording, which is what the item
    # reports: the line is written beside the refusal, never instead of it.
    assert isinstance(failure.value.__cause__, ChannelWriteBlockedError)
    assert str(failure.value.__cause__) == str(monitor_refusal)
    assert StoreVerdict.NARROWING in str(failure.value)
    assert connector.readbacks[_CHANNEL] == 0.0


def test_an_owner_less_run_refused_at_a_device_write_still_earns_its_line(
    writing_wrapper: tuple[Any, str], caplog: pytest.LogCaptureFixture
) -> None:
    """A plan that reached the queue without a name is governed by the ceiling.

    The line is what says so. Its owner is ``NO_OWNER``, and a sentinel that
    raised on formatting — inside an exception handler, where a second failure
    would replace the first — would cost exactly this line.
    """
    make_plan, logger_name = writing_wrapper
    plan_function = make_plan(_device(_RefusingConnector()))

    with caplog.at_level(logging.WARNING, logger=logger_name):
        with pytest.raises(FailedStatus):
            RunEngine(context_managers=[])(plan_function())

    warnings = _warnings_from(caplog, logger_name)
    assert len(warnings) == 1
    assert str(NO_OWNER) in warnings[0].getMessage()


def test_a_write_that_failed_rather_than_was_refused_earns_no_line(
    writing_wrapper: tuple[Any, str], caplog: pytest.LogCaptureFixture
) -> None:
    """Nobody's narrowing decided a mismatch, so no owner is named for one.

    The handler is offered every exception a run raises, which is what makes
    this the row that matters: a failed write aborts the plan exactly as a
    refusal does, and a line naming an owner for it would be an invented
    attribution. The write was attempted and the channel disagreed.
    """
    make_plan, logger_name = writing_wrapper
    connector = FakeConnector(readbacks={_CHANNEL: 0.0})
    connector.write_side_effect = ChannelWriteFailedError(_CHANNEL, "MISMATCH")
    plan_function = make_plan(_device(connector))

    with caplog.at_level(logging.WARNING, logger=logger_name):
        with pytest.raises(FailedStatus) as failure:
            RunEngine(context_managers=[])(plan_function(**{RESERVED_OWNER_KWARG: _OWNER}))

    assert _warnings_from(caplog, logger_name) == []
    assert isinstance(failure.value.__cause__, ChannelWriteFailedError)
    assert connector.write_calls != []


# --- what a plan's own failure may be carrying ------------------------------
#
# A wrapper hands every exception its run raises to the seam that decides
# whether it is a refusal, so the object in `args[0]` is whatever a plan put
# there — and a session plan is arbitrary worker-side source. The rows below
# are the shapes that answer the question "what failed?" badly: two that never
# answer it at all, and one whose answer is a cancellation. None of them is a
# refusal, and none of them may be allowed to stop the wrapper from re-raising
# what actually failed.

# What "promptly" means here. Generous, because the assertion is about orders
# of magnitude: reading a status-like takes microseconds, and the failure this
# guards against does not take longer — it never returns. The row carries a
# test timeout as well, so a regression fails the run instead of hanging it.
_PROMPT_S = 5.0


class _CancellingStatus:
    """A status-like whose report of what failed is itself a cancellation."""

    def exception(self, timeout: float | None = None) -> BaseException:  # noqa: ARG002 - the status protocol signature
        """Never answers, and raises the one failure that is not an ``Exception``."""
        raise asyncio.CancelledError


class _LegacyStatus:
    """A status-like that spells the bound on its answer under its own name.

    Nothing in the pinned stack asks for a bound this way. It is here because
    the ask falls back to a positional bound for a signature that refuses the
    keyword, and a fallback nothing drives is a claim rather than a rung.
    """

    def __init__(self, error: BaseException) -> None:
        self._error = error

    def exception(self, deadline: float | None = None) -> BaseException:
        """What failed — and never without a bound, which is the whole point."""
        if deadline is None:
            raise AssertionError("a status-like must never be asked unbounded")
        return self._error


# The two real ones are the stack's own: a sync `ophyd` status, which `bluesky`
# installs, and a plain future. Both wait for a result that is never coming
# when they are asked without a bound.
_STALLING_SHAPES: dict[str, type] = {
    "unfinished sync status": StatusBase,
    "pending future": Future,
    "cancelling status": _CancellingStatus,
}

_LEGACY_SHAPE = "status-like with a bound of its own"


def make_status_like(kind: str) -> Any:
    """The object a plan's failure carries as its first argument.

    Public because the session wrapper's plan source imports it: both wrappers
    have to meet the same object, and a plan file is exec'd in a namespace of
    its own that sees nothing of this module.
    """
    if kind == _LEGACY_SHAPE:
        return _LegacyStatus(_monitor_refusal())
    return _STALLING_SHAPES[kind]()


def _status_like_plan(kind: str) -> Any:
    """A catalog plan that fails carrying the named status-like."""

    def plan(_devices: Any, _params: Any) -> Any:
        """Run one message, then fail with something that is not a refusal."""
        yield current_owner()
        raise RuntimeError(make_status_like(kind))

    return plan


def _status_like_source(kind: str) -> str:
    """The same plan as plan-file source, for the session wrapper to ``exec``."""
    return f'''PLAN_METADATA = {{"name": {_PLAN_NAME!r}}}

from pydantic import BaseModel, ConfigDict

from osprey_connectors.posture_store import current_owner
from tests.services.bluesky_bridge.test_plan_wrapper_owner import make_status_like


class PARAMS(BaseModel):
    model_config = ConfigDict(extra="forbid")


def build_plan(devices, params):
    """Run one message, then fail with something that is not a refusal."""
    yield current_owner()
    raise RuntimeError(make_status_like({kind!r}))
'''


def _catalog_status_like(kind: str) -> Any:
    """The failing plan as ``qserver_startup`` hands it to the manager."""
    return _catalog_wrapper(_status_like_plan(kind))


def _session_status_like(kind: str) -> Any:
    """The failing plan as ``session_upload`` installs it in the worker."""
    return _session_wrapper(_status_like_source(kind))


@pytest.fixture(params=["catalog", "session"])
def status_like_wrapper(request: pytest.FixtureRequest) -> tuple[Any, str]:
    """Both wrappers, each building a plan whose failure carries a status-like."""
    if request.param == "catalog":
        return _catalog_status_like, qserver_startup.logger.name
    return _session_status_like, session_upload.logger.name


@pytest.mark.timeout(30)
@pytest.mark.parametrize("kind", sorted(_STALLING_SHAPES))
def test_a_failure_carrying_a_status_like_is_read_without_waiting_on_it(
    status_like_wrapper: tuple[Any, str], kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The failure travels on, whatever the thing it is carrying does.

    Asking a status-like what failed is the seam's last resort, and asking it
    unbounded is how a lane wedges: the plan is suspended inside the handler
    that still has to re-raise, so the item never fails, no error is reported
    anywhere, and the RunEngine thread stops. A cancellation escaping the same
    call is the quieter version of the same fault — the run would report being
    cancelled instead of reporting what actually went wrong.
    """
    build, logger_name = status_like_wrapper
    wrapper = build(kind)

    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger=logger_name):
        with pytest.raises(RuntimeError) as failure:
            list(wrapper(**{RESERVED_OWNER_KWARG: _OWNER}))
    elapsed = time.monotonic() - started

    assert elapsed < _PROMPT_S
    assert _warnings_from(caplog, logger_name) == []
    # The same failure, still carrying the same object: nothing about it was
    # read as a refusal, and nothing replaced it.
    assert type(failure.value) is RuntimeError
    assert isinstance(failure.value.args[0], _STALLING_SHAPES[kind])
    assert posture_store._owner_var.get() is NO_OWNER


def test_a_refusal_reported_under_another_bound_still_earns_its_line(
    status_like_wrapper: tuple[Any, str], caplog: pytest.LogCaptureFixture
) -> None:
    """The control for the rows above: a bounded ask is still an ask.

    Without this they would be equally well satisfied by a seam that gave up on
    every status-like it met, and the line a refusal earns would be pinned only
    where the refusal was easy to find.
    """
    build, logger_name = status_like_wrapper
    wrapper = build(_LEGACY_SHAPE)

    with caplog.at_level(logging.WARNING, logger=logger_name):
        with pytest.raises(RuntimeError):
            list(wrapper(**{RESERVED_OWNER_KWARG: _OWNER}))

    warnings = _warnings_from(caplog, logger_name)
    assert len(warnings) == 1
    line = warnings[0].getMessage()
    assert _CHANNEL in line
    assert _OWNER in line
    assert StoreVerdict.NARROWING in line
