"""pyAML writes only inside a ``run_tool`` call; everywhere else they are refused.

``execute`` holds the target's run lock and nothing more, so a pyAML set inside it
is refused before anything is read or written. The same set inside ``run_tool``
has its prior value read, then is written, and is recorded only once the runtime
sent or landed it. With no guarded-run directory the run stops before the tool
is called.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pyaml.accelerator import Accelerator

import osprey.runtime
from osprey.errors import (
    ChannelLimitsViolationError,
    ChannelWriteBlockedError,
)
from osprey.runtime.guarded_run import GuardedRunDirError, lock
from osprey_connectors.control_system.base import ChannelWriteResult, WriteOutcome
from osprey_connectors.control_system.limits_validator import (
    ChannelLimitsConfig,
    LimitsValidator,
)
from osprey_connectors.types import LIMITS_MODE_EXCLUSIVE
from pyaml_cs_osprey import _run
from pyaml_cs_osprey.catalog import parse_reference
from pyaml_cs_osprey.device import OspreyDevice
from pyaml_cs_osprey.devices import OspreyDeviceList
from pyaml_cs_osprey.errors import (
    OspreyReadFailed,
    OspreyWriteFailed,
    OspreyWriteRefused,
)
from pyaml_cs_osprey.run_tool import run_tool
from tests.pyaml_cs_osprey.conftest import DictConnector, RunRecord

MAGNET = "QF_001"
SETPOINT = "QF_001:Cm:set"
READBACK = "QF_001:Cm:rdbk"

CONFIG: dict[str, Any] = {
    "type": "pyaml.accelerator",
    "facility": "Test",
    "machine": "sr",
    "energy": 1.0e9,
    "controls": [{"type": "pyaml_cs_osprey.controlsystem", "name": "live"}],
    "devices": [
        {
            "type": "pyaml.magnet.quadrupole",
            "name": MAGNET,
            "model": {
                "type": "pyaml.magnet.identity_model",
                "physics": f"({READBACK}, {SETPOINT})[1/m]",
                "unit": "1/m",
            },
        }
    ],
}


class _CountingConnector(DictConnector):
    """A dict connector that records every read and every put it serves."""

    def __init__(self, values: dict[str, Any]) -> None:
        super().__init__(values)
        self.reads: list[str] = []
        self.puts: list[tuple[str, Any]] = []
        #: Every read and put in the order served, as ``("read", address)`` or
        #: ``("put", address)``.
        self.served: list[tuple[str, str]] = []
        self.read_error: Exception | None = None

    async def read_channel(self, channel_address: str, timeout: float | None = None):
        self.reads.append(channel_address)
        self.served.append(("read", channel_address))
        if self.read_error is not None:
            raise self.read_error
        return await super().read_channel(channel_address, timeout)

    def _put(self, channel_address: str, value: Any) -> None:
        self.puts.append((channel_address, value))
        self.served.append(("put", channel_address))
        super()._put(channel_address, value)


@pytest.fixture(scope="module")
def sr() -> Accelerator:
    return Accelerator.from_dict(copy.deepcopy(CONFIG))


@pytest.fixture
def connector(
    monkeypatch: pytest.MonkeyPatch,
    guarded_repo: Path,  # noqa: ARG001 - the run needs its repo
) -> Iterator[_CountingConnector]:
    monkeypatch.setattr(osprey.runtime, "_limits_validator", None)
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readwrite")
    machine = _CountingConnector({SETPOINT: 1.0, READBACK: 1.0})
    with patch("osprey.runtime._get_connector", new_callable=AsyncMock) as get:
        get.return_value = machine
        yield machine


def _strength(sr: Accelerator) -> Any:
    return sr.live.magnet.get(name=MAGNET).strength


class TestExecuteIsReadOnly:
    """Under the run lock alone, as ``execute`` holds it, a pyAML set is refused."""

    def test_a_set_under_the_run_lock_is_refused_with_no_put(
        self, sr: Accelerator, connector: _CountingConnector
    ) -> None:
        with lock("live"), pytest.raises(OspreyWriteRefused) as info:
            _strength(sr).set(1.5)

        assert info.value.reason == "pyAML writes run only inside pyaml_measure"
        assert info.value.channel_address == SETPOINT
        assert connector.puts == []
        assert connector.reads == []
        assert connector._state[SETPOINT] == 1.0

    def test_a_set_with_no_run_at_all_is_refused_the_same_way(
        self, sr: Accelerator, connector: _CountingConnector
    ) -> None:
        with pytest.raises(OspreyWriteRefused, match="only inside pyaml_measure"):
            _strength(sr).set(1.5)
        assert connector.puts == []


class TestRunToolWrites:
    """Inside ``run_tool`` the same set has its prior value read, then is written."""

    def test_the_sets_first_value_is_read_before_it_is_written(
        self, sr: Accelerator, connector: _CountingConnector
    ) -> None:
        run = run_tool(lambda: _strength(sr).set(1.5))

        assert run.completed is True
        first_put = connector.served.index(("put", SETPOINT))
        assert ("read", SETPOINT) in connector.served[:first_put], "read before the put"
        assert run.moved == {SETPOINT: 1.0}
        assert connector.puts == [(SETPOINT, 1.5)]

    def test_no_guarded_run_directory_stops_the_run_before_the_tool(
        self,
        sr: Accelerator,
        connector: _CountingConnector,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bare = tmp_path / "bare"
        bare.mkdir()
        monkeypatch.chdir(bare)
        called: list[str] = []

        def tool() -> None:
            called.append("ran")
            _strength(sr).set(1.5)

        with pytest.raises(GuardedRunDirError, match="guarded runs need var/guarded_run"):
            run_tool(tool)

        assert called == []
        assert connector.puts == []

    def test_a_failed_pre_write_read_is_a_read_failure_with_no_put(
        self, sr: Accelerator, connector: _CountingConnector
    ) -> None:
        connector.read_error = TimeoutError("timed out")

        with lock("live"), _run.opened("live"), pytest.raises(OspreyReadFailed) as info:
            _strength(sr).set(1.5)

        assert SETPOINT in str(info.value)
        assert connector.puts == []


def _setpoint() -> OspreyDevice:
    """The magnet's setpoint as an OSPREY device, below pyAML's own range check."""
    return OspreyDevice(parse_reference(f"({READBACK}, {SETPOINT})[1/m]"))


def _limits_net(ceiling: float) -> LimitsValidator:
    """An injected limits net that refuses every setpoint value above ``ceiling``."""
    return LimitsValidator(
        {
            SETPOINT: ChannelLimitsConfig(
                channel_address=SETPOINT, min_value=0.0, max_value=ceiling, writable=True
            )
        },
        {"mode": LIMITS_MODE_EXCLUSIVE},
    )


#: Three setpoints written as one device-list batch.
BATCH = ["QF_001:Cm:set", "QF_002:Cm:set", "QF_003:Cm:set"]
BATCH_HOME = {"QF_001:Cm:set": 1.0, "QF_002:Cm:set": 2.0, "QF_003:Cm:set": 3.0}


def _batch() -> OspreyDeviceList:
    devices = OspreyDeviceList()
    devices.add_devices(
        [OspreyDevice(parse_reference(f"({address}, {address})[1/m]")) for address in BATCH]
    )
    return devices


class _BatchConnector(DictConnector):
    """The three batch setpoints; ``answer`` decides one address's write.

    ``answer(address, value)`` returns a result to report for that write, raises
    to abort the connector's batch there, or returns ``None`` to write the value.
    """

    def __init__(self, answer: Any) -> None:
        super().__init__(BATCH_HOME)
        self.answer = answer

    async def write_channel(self, channel_address: str, value: Any, **kwargs: Any):
        result = self.answer(channel_address, value)
        if result is not None:
            return result
        return await super().write_channel(channel_address, value, **kwargs)


def _refused(address: str, value: Any, reason: str) -> ChannelWriteResult:
    return ChannelWriteResult(
        channel_address=address,
        value_written=value,
        outcome=WriteOutcome.REFUSED,
        refusal_reason=reason,
        error_message=f"Write to '{address}' refused ({reason})",
    )


@pytest.fixture
def serve(
    monkeypatch: pytest.MonkeyPatch,
    guarded_repo: Path,  # noqa: ARG001 - the run needs its repo
) -> Iterator[Any]:
    """Serve a connector to ``osprey.runtime`` for the test, with no injected limits net."""
    monkeypatch.setattr(osprey.runtime, "_limits_validator", None)
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readwrite")
    with patch("osprey.runtime._get_connector", new_callable=AsyncMock) as get:
        yield lambda machine: setattr(get, "return_value", machine) or machine


class TestMovedListsOnlyWhatMoved:
    """``moved`` holds an address only once the runtime sent or landed its write.

    A refused write records nothing and its refusal propagates unchanged. A
    write whose call raised something other than a refusal of that address,
    after the runtime attempted it but before it reported an outcome, is
    listed: it may have gone out.
    """

    def test_a_limits_refused_set_records_nothing_and_the_refusal_propagates(
        self,
        connector: _CountingConnector,
        guarded: RunRecord,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(osprey.runtime, "_limits_validator", _limits_net(10.0))

        with pytest.raises(OspreyWriteRefused) as info:
            _setpoint().set(99.0)

        assert info.value.channel_address == SETPOINT
        assert isinstance(info.value.__cause__, ChannelLimitsViolationError)
        assert guarded.moved == {}
        assert connector.puts == []

    def test_a_batch_with_one_refused_channel_lists_exactly_the_sent_ones(
        self, serve: Any, guarded: RunRecord
    ) -> None:
        middle = BATCH[1]
        machine = serve(
            _BatchConnector(
                lambda a, v: _refused(a, v, "CONTROL_SYSTEM_REFUSED") if a == middle else None
            )
        )

        with pytest.raises(OspreyWriteFailed) as info:
            _batch().set([10.0, 20.0, 30.0])

        assert info.value.channel_address == middle
        assert isinstance(info.value.__cause__, ChannelWriteBlockedError)
        assert guarded.moved == {BATCH[0]: 1.0, BATCH[2]: 3.0}
        assert list(guarded.moved) == [BATCH[0], BATCH[2]], "in the batch's order"
        assert machine._state == {BATCH[0]: 10.0, middle: 2.0, BATCH[2]: 30.0}

    def test_a_batch_refused_whole_lists_nothing(self, serve: Any, guarded: RunRecord) -> None:
        serve(_BatchConnector(lambda a, v: _refused(a, v, "WRITES_DISABLED")))

        with pytest.raises(OspreyWriteRefused):
            _batch().set([10.0, 20.0, 30.0])

        assert guarded.moved == {}

    def test_a_repeat_write_keeps_the_first_before_value(
        self, sr: Accelerator, connector: _CountingConnector, guarded: RunRecord
    ) -> None:
        _strength(sr).set(1.5)
        _strength(sr).set(2.5)

        assert guarded.moved == {SETPOINT: 1.0}
        assert connector.reads.count(SETPOINT) == 1, "a recorded address is not read again"
        assert connector.puts == [(SETPOINT, 1.5), (SETPOINT, 2.5)]

    def test_a_refused_then_sent_write_records_the_value_before_the_sent_one(
        self,
        connector: _CountingConnector,
        guarded: RunRecord,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(osprey.runtime, "_limits_validator", _limits_net(10.0))
        with pytest.raises(OspreyWriteRefused):
            _setpoint().set(99.0)
        assert guarded.moved == {}

        connector._state[SETPOINT] = 1.25
        _setpoint().set(1.5)

        assert guarded.moved == {SETPOINT: 1.25}
        assert connector.puts == [(SETPOINT, 1.5)]

    def test_a_batch_the_connector_aborts_midway_lists_what_may_have_gone_out(
        self, serve: Any, guarded: RunRecord
    ) -> None:
        """A transport error partway through: the channels before it, and the rest, are listed."""
        middle = BATCH[1]

        def answer(address: str, _value: Any) -> None:
            if address == middle:
                raise ConnectionError("link dropped")

        machine = serve(_BatchConnector(answer))

        with pytest.raises(OspreyWriteFailed) as info:
            _batch().set([10.0, 20.0, 30.0])

        assert isinstance(info.value.__cause__, ConnectionError)
        assert machine._state[BATCH[0]] == 10.0, "the first channel went out"
        assert guarded.moved == BATCH_HOME

    def test_a_connector_limits_refusal_midway_lists_every_channel_but_the_refused_one(
        self, serve: Any, guarded: RunRecord
    ) -> None:
        middle = BATCH[1]

        def answer(address: str, value: Any) -> None:
            if address == middle:
                raise ChannelLimitsViolationError(address, value, "MAX_EXCEEDED", "above max")

        machine = serve(_BatchConnector(answer))

        with pytest.raises(OspreyWriteRefused) as info:
            _batch().set([10.0, 20.0, 30.0])

        assert info.value.channel_address == middle
        assert machine._state[BATCH[0]] == 10.0, "the first channel went out"
        assert guarded.moved == {BATCH[0]: 1.0, BATCH[2]: 3.0}

    def test_a_single_write_lost_to_a_transport_error_is_listed(
        self, serve: Any, guarded: RunRecord
    ) -> None:
        def answer(_address: str, _value: Any) -> None:
            raise TimeoutError("no reply")

        serve(_BatchConnector(answer))
        device = OspreyDevice(parse_reference(f"({BATCH[0]}, {BATCH[0]})[1/m]"))

        with pytest.raises(OspreyWriteFailed):
            device.set(10.0)

        assert guarded.moved == {BATCH[0]: 1.0}

    def test_an_interrupt_escaping_the_write_lists_every_pending_address(
        self, serve: Any, guarded: RunRecord
    ) -> None:
        """A ``KeyboardInterrupt`` before any attempt: the write may still go out."""
        machine = serve(_BatchConnector(lambda _a, _v: None))
        interrupt = KeyboardInterrupt()

        def write(**_kwargs: Any) -> None:
            raise interrupt

        def unmapped(exc: BaseException, *_args: Any) -> BaseException:
            raise AssertionError(f"an interrupt is never mapped: {exc!r}")

        with pytest.raises(KeyboardInterrupt) as info:
            _run.write(BATCH, write, unmapped, map_read=unmapped, confirm=None)

        assert info.value is interrupt, "the interrupt propagates unchanged"
        assert guarded.moved == BATCH_HOME
        assert machine._state == BATCH_HOME, "nothing was written"
