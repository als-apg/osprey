"""The owner a queued plan carries, from the wrapper into the gated write.

A plan wrapper binds the queue item's owner around its own body, and the
write that owner governs does not happen there. It happens in the task
``ConnectorSettable.set``'s ``AsyncStatus`` wraps a coroutine into, created
by the RunEngine while it processes the plan's ``set`` message — a different
frame, on a different thread, at a point the generator is suspended. So the
binding has to survive as *context* rather than as a local: this file drives
a real ``RunEngine`` and reads the owner back from inside the connector call
that the write monitor's store lookup is made for.

That is the whole of what is pinned here, and it is what neither neighbour
can pin. ``test_plan_wrapper_owner.py`` drives both wrappers as the plain
generators they are, so it sees the owner at every yield and never sees a
write; ``test_connector_devices.py`` drives the device without a wrapper, so
it sees the write and never an owner. ``tests/connectors/test_owner_context.py``
owns the ladder and the sentinel, and none of the three are restated here.

Both wrappers run every row — ``qserver_startup._make_plan_function`` for a
catalog plan and the one ``session_upload.install_session_plan`` installs for
a session plan — because a session plan that reached the RunEngine without
its owner in context would be the quiet way past a gate the catalog path
enforces.

The store the write asks is the real one, reading a real control-context
tree: the record is planted for a named owner, so a refusal here is the
store answering narrowing for *that person* rather than a stub agreeing to
refuse. Every test stamps the tree bind, the deployment shape a lane's
queueserver runs in, where the ladder's last answer is ``NO_OWNER`` rather
than the account the process runs as.
"""

from __future__ import annotations

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

from osprey.services.bluesky_bridge import qserver_startup  # noqa: E402
from osprey.services.bluesky_bridge.devices.connector import ConnectorSettable  # noqa: E402
from osprey.services.bluesky_bridge.plan_fields import MovableChannel  # noqa: E402
from osprey.services.bluesky_bridge.session_upload import install_session_plan  # noqa: E402
from osprey_connectors import control_context, posture_store  # noqa: E402
from osprey_connectors.errors import ChannelWriteBlockedError  # noqa: E402
from osprey_connectors.posture_store import (  # noqa: E402
    NO_OWNER,
    RESERVED_OWNER_KWARG,
    StoreVerdict,
    current_owner,
    store_verdict,
)
from tests.services.bluesky_bridge.test_connector_devices import FakeConnector  # noqa: E402

_OWNER = "alice"
_UNNARROWED_OWNER = "bob"
_PLAN_NAME = "owner_probe_plan"
_DEVICE_NAME = "motor"
_CHANNEL = "SR:BEND:SETPOINT"
_TARGET = "live"
_SETPOINT = 3.5


class _GatedConnector(FakeConnector):
    """A connector that asks the store who the write belongs to, and records it.

    Stands in for the transport alone. The question it asks is the reference
    monitor's own — the recorded write state for this control target, for
    whoever the ladder says this write belongs to — and it asks it where a
    real connector asks it: inside the write call, which the device has by
    then scheduled onto its own task.

    The write is left aliased (no ``readback_pv``) and confirmed, so one
    ``set`` is exactly one connector call and no settle poll: the owner log
    below has one entry per write the plan made, with nothing to disentangle.
    """

    def __init__(self) -> None:
        super().__init__(readbacks={_CHANNEL: 0.0})
        self.write_owners: list[Any] = []

    async def write_channel_checked(self, channel_address: str, value: Any, **kwargs: Any):
        self.write_owners.append(current_owner())
        verdict = store_verdict(_TARGET)
        if verdict is not StoreVerdict.PERMITTED:
            # The reason word is the verdict's own spelling, not a second one:
            # ``StoreVerdict`` is a ``StrEnum``, so what the store answered is
            # what the operator reads.
            raise ChannelWriteBlockedError(
                channel_address,
                "WRITES_DISABLED",
                message=(
                    f"Write to '{channel_address}' blocked: writes are off for the "
                    f"'{_TARGET}' control target — turn them back on from the "
                    f"control-target chip if the write is intended. "
                    f"The store answered {verdict}."
                ),
            )
        return await super().write_channel_checked(channel_address, value, **kwargs)


class _Params(BaseModel):
    """A plan schema declaring the one channel the plan drives.

    ``extra="forbid"`` keeps the reserved owner kwarg's removal an assertion
    rather than an assumption, and the movable role is what makes the device
    reach the plan at all: a wrapper hands a plan the channels its params
    declare and nothing else.
    """

    model_config = ConfigDict(extra="forbid")

    motor: MovableChannel = _DEVICE_NAME
    setpoint: float = _SETPOINT


def _write_probe_plan(devices: Any, params: Any) -> Iterator[Any]:
    """Drive the declared movable once."""
    yield from bps.mv(devices[params.motor], params.setpoint)


_SESSION_SOURCE = f'''PLAN_METADATA = {{"name": {_PLAN_NAME!r}}}

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


@pytest.fixture(autouse=True)
def control_context_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A provisioned control-context tree, bound as a lane's queueserver has it.

    The stamped-owner rung and the launch pin are cleared so that a
    deployment's environment cannot decide these tests, and the tree bind is
    what makes ``NO_OWNER`` — rather than this test process's account — the
    ladder's answer for a plan that arrived without a name. The bound owner
    is reset on the way out as well as in: a run that failed mid-plan would
    otherwise hand its owner to the next test.
    """
    tree = tmp_path / posture_store.STATE_DIR_NAME
    tree.mkdir(parents=True)
    (tree / posture_store.CONTROL_TREE_MARKER_NAME).write_text("provisioned\n", encoding="utf-8")
    monkeypatch.delenv(posture_store.CONTROL_OWNER_ENV_VAR, raising=False)
    monkeypatch.delenv(posture_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_TREE_ENV_VAR, str(tree))
    posture_store._owner_var.set(NO_OWNER)
    yield tree
    posture_store._owner_var.set(NO_OWNER)


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


def _catalog_wrapper(device: ConnectorSettable) -> Callable[..., Iterator[Any]]:
    """The plan as ``qserver_startup`` hands it to the manager."""
    spec = type(
        "_Spec",
        (),
        {
            "name": _PLAN_NAME,
            "schema": _Params,
            "plan": staticmethod(_write_probe_plan),
            "description": "A stub plan that drives one channel.",
        },
    )()
    return qserver_startup._make_plan_function(spec, devices={_DEVICE_NAME: device})


def _session_wrapper(device: ConnectorSettable) -> Callable[..., Iterator[Any]]:
    """The plan as ``session_upload`` installs it in the worker namespace."""
    namespace: dict[str, Any] = {"__name__": "__main__", _DEVICE_NAME: device}
    return install_session_plan(namespace, _PLAN_NAME, _SESSION_SOURCE)


@pytest.fixture(params=["catalog", "session"])
def wrapped_plan(request: pytest.FixtureRequest) -> Callable[[ConnectorSettable], Any]:
    """Both wrappers, each around the plan that drives one gated channel."""
    if request.param == "catalog":
        return _catalog_wrapper
    return _session_wrapper


def _device(connector: _GatedConnector) -> ConnectorSettable:
    """The movable the plan drives, writing through *connector*."""
    return ConnectorSettable(connector, _CHANNEL, name=_DEVICE_NAME)


# --- the binding, across the RunEngine's own scheduling ---------------------


def test_the_owner_reaches_the_write_the_run_engine_schedules(
    wrapped_plan: Callable[[ConnectorSettable], Any],
) -> None:
    """Not the generator's frame: the task the device's write is wrapped into.

    ``ConnectorSettable.set`` is ``AsyncStatus``-wrapped, so the RunEngine
    turns one ``set`` message into a coroutine it schedules and the plan is
    suspended while that coroutine runs. A binding held anywhere but in
    context would be gone by the time the connector is asked, and the write
    would be gated against the worker account instead of the person who
    queued it.
    """
    connector = _GatedConnector()
    plan_function = wrapped_plan(_device(connector))

    RunEngine(context_managers=[])(plan_function(**{RESERVED_OWNER_KWARG: _OWNER}))

    assert connector.write_owners == [_OWNER]
    assert connector.readbacks[_CHANNEL] == _SETPOINT


def test_the_binding_does_not_outlive_the_run(
    wrapped_plan: Callable[[ConnectorSettable], Any],
) -> None:
    """Whoever asks after the run gets the ceiling, not the last run's owner."""
    connector = _GatedConnector()
    plan_function = wrapped_plan(_device(connector))

    RunEngine(context_managers=[])(plan_function(**{RESERVED_OWNER_KWARG: _OWNER}))

    assert current_owner() is NO_OWNER
    assert posture_store._owner_var.get() is NO_OWNER


def test_an_owner_less_run_after_an_owned_one_writes_under_no_owner(
    wrapped_plan: Callable[[ConnectorSettable], Any],
) -> None:
    """The leak that matters is between runs, not after the last one.

    A worker's RunEngine outlives every item it executes and runs them on one
    thread, so an owner left in the context a run was advanced from would gate
    the *next* person's plan against the previous person's narrowing. Both
    runs go through one RunEngine and one connector, which is the arrangement
    where that would show.
    """
    connector = _GatedConnector()
    plan_function = wrapped_plan(_device(connector))
    run_engine = RunEngine(context_managers=[])

    run_engine(plan_function(**{RESERVED_OWNER_KWARG: _OWNER}))
    run_engine(plan_function())

    assert connector.write_owners == [_OWNER, NO_OWNER]
    assert connector.write_owners[1] is NO_OWNER


# --- the store's answer, for the owner the run carries ----------------------


def test_a_narrowing_recorded_for_the_owner_refuses_the_write_and_fails_the_run(
    wrapped_plan: Callable[[ConnectorSettable], Any], control_context_tree: Path
) -> None:
    """The refusal reaches the operator saying what the store answered.

    The RunEngine reports a failed device write as a ``FailedStatus`` carrying
    the underlying exception's message, which is how the monitor's own wording
    — and with it the reason the write was refused — becomes the run's error
    rather than a generic failure. Nothing is written: a refusal is decided
    before the value is sent.
    """
    narrow_for(control_context_tree, _OWNER)
    connector = _GatedConnector()
    plan_function = wrapped_plan(_device(connector))

    with pytest.raises(FailedStatus, match="narrowing") as failure:
        RunEngine(context_managers=[])(plan_function(**{RESERVED_OWNER_KWARG: _OWNER}))

    assert isinstance(failure.value.__cause__, ChannelWriteBlockedError)
    assert failure.value.__cause__.channel_address == _CHANNEL
    assert connector.write_calls == []
    assert connector.readbacks[_CHANNEL] == 0.0


def test_the_same_plan_is_permitted_for_an_owner_the_store_does_not_narrow(
    wrapped_plan: Callable[[ConnectorSettable], Any], control_context_tree: Path
) -> None:
    """The control for the row above: one person's narrowing is theirs alone.

    Same tree, same target, same plan — only the name the item carries
    differs. Without this, a refusal would equally well be explained by a
    store that refuses everyone, and the owner would be pinning nothing.
    """
    narrow_for(control_context_tree, _OWNER)
    connector = _GatedConnector()
    plan_function = wrapped_plan(_device(connector))

    RunEngine(context_managers=[])(plan_function(**{RESERVED_OWNER_KWARG: _UNNARROWED_OWNER}))

    assert connector.write_owners == [_UNNARROWED_OWNER]
    assert connector.readbacks[_CHANNEL] == _SETPOINT
