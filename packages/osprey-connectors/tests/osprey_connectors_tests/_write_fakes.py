"""Shared fakes for the write-path and write-posture tests.

One recording connector, one config-reader shape, one mocked ``EPICSConnector``
builder and one way to arm writes, so the files that pin the write guard do not
each carry their own byte-equivalent copy.

Arming a write takes what the reference monitor's verdict asks for
(:func:`osprey_connectors.posture_store.write_verdict`): a ``control_system``
section whose tier allows writes, a connector stamped with the machine it
writes to, and the login's switch turned on for that machine in its record.
:data:`ARMED_SECTION` is the first, :func:`arm_switch` the other two.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from osprey_connectors import control_context
from osprey_connectors.control_system._base import (
    ChannelWriteResult,
    ControlSystemConnector,
    WriteOutcome,
)
from osprey_connectors.control_system._epics_connector import EPICSConnector
from osprey_connectors.types import TARGET_VA
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


def writes_enabled_config(key: str, default: Any = None) -> Any:
    """``get_config_value`` stand-in: a write tier that allows, all else defaulted.

    Serves :data:`ARMED_SECTION` as the ``control_system`` section. The tier
    alone arms nothing: the write also needs its machine's switch on
    (:func:`arm_switch`).
    """
    if key == "control_system":
        return dict(ARMED_SECTION)
    return default


def config_reader(section: dict[str, Any]) -> Callable[..., Any]:
    """A ``get_config_value`` stand-in serving one ``control_system:`` section.

    Answers the whole section, which the write verdict reads its tier from.
    """

    def _get(key: str, default: Any = None) -> Any:
        if key == "control_system":
            return section
        return default

    return _get


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

    *stamp* sets the class default of ``_control_target`` to *target*, so a
    connector built directly rather than through the factory writes to that
    machine. A test about the unstamped connector passes ``stamp=False`` or
    sets ``connector._control_target = None`` itself.

    The tier is not set here: serve a section that allows writes
    (:data:`ARMED_SECTION`, :func:`writes_enabled_config`) through
    ``get_config_value``.

    Returns:
        The record's path, or ``None`` when no record was written.
    """
    isolate_verdict_environment(monkeypatch, root)
    if stamp:
        monkeypatch.setattr(ControlSystemConnector, "_control_target", target)
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
        except control_context.StoppedSince:
            control_context.invalidate_cache()
            record = control_context.read_record()
            control_context.set_switch([target], state, epoch=record.switch_epoch)
    control_context.invalidate_cache()


class RecordingConnector(ControlSystemConnector):
    """A connector that records the writes the base-class guard let through.

    The abstract signature verbatim (``confirm`` included): a stand-in whose
    signature has drifted from the base class is wrapped by the same guard but
    exercises a call shape no real connector has. Every write it receives
    comes back ``CONFIRMED``.
    """

    def __init__(self) -> None:
        self.writes: list[tuple[str, Any]] = []

    async def connect(self, config: dict[str, Any]) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def read_channel(self, channel_address: str, timeout: float | None = None):
        raise NotImplementedError

    async def read_multiple_channels(self, channel_addresses, timeout=None):
        raise NotImplementedError

    async def write_channel(
        self,
        channel_address: str,
        value: Any,
        timeout: float | None = None,  # noqa: ARG002 - the control-system connector interface fixes this signature
        confirm: bool | None = None,  # noqa: ARG002 - the control-system connector interface fixes this signature
    ) -> ChannelWriteResult:
        self.writes.append((channel_address, value))
        return ChannelWriteResult(
            channel_address=channel_address,
            value_written=value,
            outcome=WriteOutcome.CONFIRMED,
        )

    async def write_multiple_channels(self, operations, timeout=None):  # noqa: ARG002 - the control-system connector interface fixes this signature
        return [await self.write_channel(addr, val) for addr, val in operations]

    async def subscribe(self, channel_address, callback):
        raise NotImplementedError

    async def unsubscribe(self, channel_address):
        raise NotImplementedError

    async def get_metadata(self, channel_address):
        raise NotImplementedError

    async def validate_channel(self, channel_address) -> bool:  # noqa: ARG002 - the control-system connector interface fixes this signature
        return True


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
