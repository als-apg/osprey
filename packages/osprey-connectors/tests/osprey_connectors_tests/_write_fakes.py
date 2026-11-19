"""Shared fakes for the write-path and write-posture tests.

One recording connector, one served config, one mocked ``EPICSConnector``
builder and one way to arm writes, so the files that pin the write guard do not
each carry their own byte-equivalent copy.

Arming a write takes what the reference monitor's verdict asks for
(:func:`osprey_connectors.write_switch.write_verdict`): a ``control_system``
section whose tier allows writes, a connector stamped with the machine it
writes to, and the login's switch turned on for that machine in its record.
:data:`ARMED_SECTION` is the first, :func:`arm_switch` the other two.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from osprey_connectors import control_context, testing
from osprey_connectors.config import LoadedConfig
from osprey_connectors.connector_types import TARGET_VA
from osprey_connectors.control_system import ConnectorContext
from osprey_connectors.control_system._base import (
    ChannelValue,
    ControlSystemConnector,
    SendReport,
)
from osprey_connectors.control_system._epics_connector import EPICSConnector
from osprey_connectors_tests._config_trees import serve_control_system
from osprey_connectors_tests._epics_fakes import FakeAioca, FakeChannel, ca_connector
from osprey_connectors_tests._harness._control_context_fixtures import (
    isolate_verdict_environment,
    write_control_context,
)

#: A ``control_system`` section whose write tier allows writes. Its type is
#: not a real control system, so ``live`` does not resolve on it; ``va`` does
#: on every section, which is why :data:`ARMED_TARGET` is the simulator.
ARMED_SECTION: dict[str, Any] = {"writes": "allowed"}

#: The machine :func:`arm_switch` turns on and stamps when told nothing else.
ARMED_TARGET = TARGET_VA


def serve_writes_enabled() -> LoadedConfig:
    """Serve :data:`ARMED_SECTION` as the process config's ``control_system`` section.

    For the rest of the test (see :func:`osprey_connectors_tests._config_trees.serve_config`).
    The tier alone arms nothing: the write also needs its machine's switch on
    (:func:`arm_switch`).
    """
    return serve_control_system(ARMED_SECTION)


def armed_context(**fields: Any) -> ConnectorContext:
    """The context a test connects a connector with, naming the stamped machine.

    The target is the one the class default context names: the machine
    :func:`arm_switch` (or a test's own patch of that default) stamped, or
    ``None`` when nothing is stamped. *fields* are the context's other fields.
    """
    return ConnectorContext(control_target=ControlSystemConnector._context.control_target, **fields)


def arm_switch(
    monkeypatch: Any,
    root: Path,
    target: str | None = ARMED_TARGET,
    *,
    stamp: bool = True,
) -> Path | None:
    """Turn the login's switch on for *target* and stamp connectors with it.

    Points the host record at *root*, pins the acting identity, clears every
    variable that would change the verdict
    (:func:`~osprey_connectors_tests._harness._control_context_fixtures.isolate_verdict_environment`) and writes an
    ownerless record whose switch is ``{target: "asked"}`` through the person's
    gesture. ``None`` writes no record at all: every switch off.

    *stamp* sets the class default context to one naming *target*, so a
    connector that was never connected writes to that machine, and one a test
    connects with :func:`armed_context` does too. A test about the unstamped
    connector passes ``stamp=False``.

    The tier is not set here: serve a section that allows writes as the process
    config (:func:`serve_writes_enabled`, which serves :data:`ARMED_SECTION`).

    Returns:
        The record's path, or ``None`` when no record was written.
    """
    isolate_verdict_environment(monkeypatch, root)
    if stamp:
        monkeypatch.setattr(
            ControlSystemConnector, "_context", ConnectorContext(control_target=target)
        )
    if target is None:
        return None
    return write_control_context(root, target=target, owned_by=None, switch={target: "asked"})


def turn_switch(target: str, state: str) -> None:
    """The person's switch gesture for *target*, against the record the environment names.

    *state* is a switch position. A move to an armed position names the record's
    epoch, as the chip does; an epoch-less record is stamped by the refused
    gesture, and the retry with that epoch succeeds.
    """
    if state == "off":
        control_context.set_switch([target], "off")
    else:
        record = control_context.read_record()
        try:
            control_context.set_switch([target], state, epoch=record.switch_epoch)
        except control_context.StoppedSinceError:
            testing.invalidate_cache()
            record = control_context.read_record()
            control_context.set_switch([target], state, epoch=record.switch_epoch)
    testing.invalidate_cache()


class RecordingConnector(ControlSystemConnector):
    """A connector that records the writes the base-class guard let through.

    Every write that reaches its send is recorded and holds the value sent, so
    it comes back ``CONFIRMED``.
    """

    def __init__(self) -> None:
        self.writes: list[tuple[str, Any]] = []
        self._held: dict[str, Any] = {}

    async def _open(self, config: dict[str, Any]) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def read_channel(self, address: str, *, timeout_s: float | None = None):
        raise NotImplementedError

    async def _send(self, address: str, value: Any, *, timeout_s: float, wait: bool) -> SendReport:  # noqa: ARG002 - the send hook fixes this signature
        self.writes.append((address, value))
        self._held[address] = value
        return SendReport()

    async def _confirming_read(self, address: str, sent: SendReport, *, timeout_s: float):  # noqa: ARG002 - the read-back hook fixes this signature
        return ChannelValue(
            address=address, value=self._held[address], sampled_at=datetime.now(UTC)
        )

    async def subscribe(self, address, callback, *, on_connection=None):
        raise NotImplementedError

    async def unsubscribe(self, address):
        raise NotImplementedError


def make_mock_epics_connector(
    *,
    validate_side_effect: Any = None,
    put: int | None = None,
) -> EPICSConnector:
    """An ``EPICSConnector`` wired with a fake aioca and a mock limits validator.

    Bypasses ``connect()`` by setting the attributes the write path depends
    on directly. Every channel name connects; ``put`` is the code every
    ``caput`` answers with — ``None`` for success, else an ECA code
    (``ECA_NOWTACCESS`` for an access-security denial, ``ECA_DISCONN`` for a
    circuit lost mid-put). The fake is at ``connector._aioca``.
    """
    connector = ca_connector(aioca=FakeAioca(default=FakeChannel(put=put)))
    connector._limits_validator = MagicMock()
    connector._limits_validator.validate = MagicMock(side_effect=validate_side_effect)
    return connector
