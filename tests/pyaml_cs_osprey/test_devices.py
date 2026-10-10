"""An OSPREY device list reads and writes its devices as one ``osprey.runtime`` call each."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from pyaml.common.exception import PyAMLException
from pyaml.control.deviceaccesslist import DeviceAccessList

import osprey.runtime
from osprey.errors import (
    ChannelLimitsViolationError,
    ChannelWriteBlockedError,
    ChannelWriteFailedError,
)
from osprey.runtime.journal import pop_journal, push_journal
from osprey_connectors.control_system.base import ChannelWriteResult, WriteOutcome
from osprey_connectors.control_system.limits_validator import (
    ChannelLimitsConfig,
    LimitsValidator,
)
from osprey_connectors.types import LIMITS_MODE_EXCLUSIVE
from pyaml_cs_osprey.catalog import parse_reference
from pyaml_cs_osprey.device import OspreyDevice
from pyaml_cs_osprey.devices import OspreyDeviceList
from pyaml_cs_osprey.errors import OspreyReadFailed, OspreyWriteFailed, OspreyWriteRefused
from tests.pyaml_cs_osprey.conftest import DictConnector
from tests.pyaml_cs_osprey.conftest import FakeRuntime as _Runtime

REFS = [
    "(QF_001:Cm:rdbk, QF_001:Cm:set)[1/m]",
    "(QF_002:Cm:rdbk, QF_002:Cm:set)[1/m]",
    "(QF_003:Cm:rdbk, QF_003:Cm:set)[1/m]",
]
SETS = ["QF_001:Cm:set", "QF_002:Cm:set", "QF_003:Cm:set"]
RDBKS = ["QF_001:Cm:rdbk", "QF_002:Cm:rdbk", "QF_003:Cm:rdbk"]


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> _Runtime:
    values: dict[str, Any] = {}
    for i, (s, r) in enumerate(zip(SETS, RDBKS, strict=True)):
        values[s] = 1.0 + i
        values[r] = 1.5 + i
    return _Runtime(values, batch_only=True).install(monkeypatch)


def _list(refs: Sequence[str] = REFS) -> OspreyDeviceList:
    dl = OspreyDeviceList()
    dl.add_devices([OspreyDevice(parse_reference(r)) for r in refs])
    return dl


# --- collection --------------------------------------------------------------


def test_it_is_a_pyaml_device_access_list_and_starts_empty() -> None:
    dl = OspreyDeviceList()
    assert isinstance(dl, DeviceAccessList)
    assert dl.len() == 0
    assert len(dl) == 0
    assert list(dl) == []


def test_add_devices_takes_one_or_a_list_in_order() -> None:
    devices = [OspreyDevice(parse_reference(r)) for r in REFS]
    dl = OspreyDeviceList()
    dl.add_devices(devices[0])
    dl.add_devices(devices[1:])
    assert dl.len() == 3
    assert [dl.get_device_at(i) for i in range(3)] == devices
    assert dl[1] is devices[1]
    assert list(dl) == devices
    assert list(dl) == devices


# --- reads -------------------------------------------------------------------


def test_get_is_one_batch_read_in_reference_order(runtime: _Runtime) -> None:
    values = _list().get()
    assert runtime.calls == ["read_channels"]
    assert runtime.batch_reads == [SETS]
    assert isinstance(values, np.ndarray)
    assert values.dtype == np.float64
    np.testing.assert_array_equal(values, [1.0, 2.0, 3.0])


def test_readback_reads_the_readback_addresses(runtime: _Runtime) -> None:
    values = _list().readback()
    assert runtime.batch_reads == [RDBKS]
    np.testing.assert_array_equal(values, [1.5, 2.5, 3.5])


def test_repeated_addresses_are_one_read_placed_at_every_position(
    runtime: _Runtime,
) -> None:
    """``read_channels`` itself reads a repeated address once and returns it per position."""
    dl = _list([REFS[1], REFS[0], REFS[1]])
    values = dl.get()
    assert runtime.batch_reads == [[SETS[1], SETS[0], SETS[1]]]
    np.testing.assert_array_equal(values, [2.0, 1.0, 2.0])


def test_read_failure_maps_to_osprey_read_failed(runtime: _Runtime) -> None:
    runtime.values[SETS[2]] = None
    with pytest.raises(OspreyReadFailed) as info:
        _list().get()
    assert info.value.addresses == [SETS[2]]
    assert runtime.calls == ["read_channels"]


def test_empty_list_reads_nothing(runtime: _Runtime) -> None:
    values = OspreyDeviceList().get()
    assert values.shape == (0,)
    assert runtime.calls == []


def test_check_device_availability_is_one_read(runtime: _Runtime) -> None:
    assert _list().check_device_availability() is True
    assert runtime.calls == ["read_channels"]
    runtime.values[SETS[0]] = None
    assert _list().check_device_availability() is False
    assert runtime.calls == ["read_channels", "read_channels"]


# --- writes ------------------------------------------------------------------


@pytest.mark.usefixtures("guarded")
def test_set_is_one_batch_write_in_reference_order(runtime: _Runtime) -> None:
    _list().set(np.array([10.0, 20.0, 30.0]))
    assert runtime.calls == ["read_channels", "write_channels"]
    [(batch, kwargs)] = runtime.batch_writes
    assert list(batch.items()) == [(SETS[0], 10.0), (SETS[1], 20.0), (SETS[2], 30.0)]
    assert kwargs == {}


@pytest.mark.usefixtures("guarded")
def test_set_and_wait_confirms(runtime: _Runtime) -> None:
    _list().set_and_wait([10.0, 20.0, 30.0])
    [(_, kwargs)] = runtime.batch_writes
    assert kwargs == {"confirm": True}


def test_set_refuses_a_wrong_length_before_any_call(runtime: _Runtime) -> None:
    with pytest.raises(ValueError):
        _list().set([1.0, 2.0])
    assert runtime.calls == []


def test_guarded_set_journals_every_address_before_the_write(runtime: _Runtime, guarded) -> None:
    _list().set([10.0, 20.0, 30.0])
    assert runtime.calls == ["read_channels", "write_channels"]
    assert runtime.batch_reads == [SETS]
    assert guarded.values == {SETS[0]: 1.0, SETS[1]: 2.0, SETS[2]: 3.0}


def test_every_active_journal_records_every_address(runtime: _Runtime, guarded) -> None:
    inner = push_journal()
    try:
        _list().set([10.0, 20.0, 30.0])
    finally:
        pop_journal(inner)
    assert set(guarded.addresses) == set(SETS)
    assert set(inner.addresses) == set(SETS)
    assert runtime.calls == ["read_channels", "write_channels"]


@pytest.mark.usefixtures("guarded")
def test_guarded_pre_read_failure_writes_nothing(runtime: _Runtime) -> None:
    runtime.values[SETS[1]] = None
    with pytest.raises(OspreyReadFailed):
        _list().set([10.0, 20.0, 30.0])
    assert runtime.calls == ["read_channels"]


@pytest.mark.usefixtures("guarded")
def test_limits_violation_maps_to_refused(runtime: _Runtime) -> None:
    runtime.write_error = ChannelLimitsViolationError(
        channel_address=SETS[1],
        value=99.0,
        violation_type="MAX_EXCEEDED",
        violation_reason="99.0 above max 5.0",
    )
    with pytest.raises(OspreyWriteRefused) as info:
        _list().set([1.0, 99.0, 3.0])
    assert info.value.channel_address == SETS[1]


@pytest.mark.usefixtures("guarded")
def test_write_failure_names_failing_and_sent_addresses(runtime: _Runtime) -> None:
    runtime.write_error = ChannelWriteFailedError(SETS[1], "FAILED")
    with pytest.raises(OspreyWriteFailed) as info:
        _list().set([10.0, 20.0, 30.0])
    assert info.value.channel_address == SETS[1]
    for address in SETS:
        assert address in str(info.value)
    assert runtime.calls == ["read_channels", "write_channels"]


# --- metadata ----------------------------------------------------------------


def test_unit_is_the_first_devices_unit_or_empty() -> None:
    assert _list().unit() == "1/m"
    assert OspreyDeviceList().unit() == ""


@pytest.mark.usefixtures("guarded")
def test_device_list_values_are_si(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each member scales by its own suffix; one batch read and one batch write per call."""
    refs = ["BPM1:x[mm]", "(BPM2:x:rb, BPM2:x:sp)[um]", "BPM3:x[m]"]
    runtime = _Runtime(
        {"BPM1:x": 2.0, "BPM2:x:sp": 3.0, "BPM2:x:rb": 5.0, "BPM3:x": 0.25},
        batch_only=True,
    ).install(monkeypatch)
    dl = _list(refs)

    np.testing.assert_allclose(dl.get(), [2e-3, 3e-6, 0.25], rtol=1e-12)
    np.testing.assert_allclose(dl.readback(), [2e-3, 5e-6, 0.25], rtol=1e-12)
    assert runtime.batch_reads == [
        ["BPM1:x", "BPM2:x:sp", "BPM3:x"],
        ["BPM1:x", "BPM2:x:rb", "BPM3:x"],
    ]
    assert dl.unit() == "m"

    dl.set([4e-3, 7e-6, 0.5])
    assert runtime.calls == ["read_channels", "read_channels", "read_channels", "write_channels"]
    [(batch, _)] = runtime.batch_writes
    assert list(batch) == ["BPM1:x", "BPM2:x:sp", "BPM3:x"]
    np.testing.assert_allclose(list(batch.values()), [4.0, 7.0, 0.5], rtol=1e-12)


def test_device_list_mixed_si_words_refused() -> None:
    """A list mixing m and rad has no one unit; an empty list's unit is ''."""
    with pytest.raises(PyAMLException) as info:
        _list(["BPM1:x[mm]", "HCM1:kick[urad]"]).unit()
    assert "'m'" in str(info.value) and "'rad'" in str(info.value)
    assert _list(["BPM1:x[mm]", "BPM2:x[um]"]).unit() == "m"
    assert OspreyDeviceList().unit() == ""


def test_get_range_is_flat(runtime: _Runtime) -> None:
    runtime.limits[SETS[0]] = ChannelLimitsConfig(
        channel_address=SETS[0], min_value=-1.0, max_value=1.0, writable=True
    )
    runtime.limits[SETS[2]] = ChannelLimitsConfig(
        channel_address=SETS[2], min_value=0.0, max_value=5.0, writable=True
    )
    assert _list().get_range() == [-1.0, 1.0, None, None, 0.0, 5.0]


# --- against a mock connector through the real runtime -----------------------


class _FailingConnector(DictConnector):
    """A dict-backed connector whose put fails for one address."""

    def __init__(self, failing: str) -> None:
        super().__init__(dict.fromkeys(SETS, 0.0))
        self._failing = failing

    def _put(self, channel_address: str, value: Any) -> None:
        if channel_address == self._failing:
            raise RuntimeError("simulated put failure")
        super()._put(channel_address, value)


@pytest.fixture
def unvalidated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(osprey.runtime, "_limits_validator", None)


@pytest.mark.usefixtures("unvalidated")
@pytest.mark.usefixtures("guarded")
def test_mid_batch_failure_on_a_mock_connector() -> None:
    connector = _FailingConnector(SETS[1])
    with patch("osprey.runtime._get_connector", new_callable=AsyncMock) as get:
        get.return_value = connector
        with pytest.raises(OspreyWriteFailed) as info:
            _list().set([10.0, 20.0, 30.0], confirm=False)
    assert info.value.channel_address == SETS[1]
    message = str(info.value)
    for address in SETS:
        assert address in message
    assert connector._state[SETS[0]] == 10.0
    assert connector._state[SETS[2]] == 30.0
    assert connector._state.get(SETS[1]) != 20.0


@pytest.mark.usefixtures("guarded")
def test_limits_violation_refuses_the_whole_batch_before_sending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = LimitsValidator(
        {
            a: ChannelLimitsConfig(channel_address=a, min_value=0.0, max_value=50.0, writable=True)
            for a in SETS
        },
        {"mode": LIMITS_MODE_EXCLUSIVE},
    )
    monkeypatch.setattr(osprey.runtime, "_limits_validator", validator)
    connector = _FailingConnector("NONE")
    sent: list[str] = []
    original = connector.write_channel

    async def spy(address: str, value: Any, **kwargs: Any):
        sent.append(address)
        return await original(address, value, **kwargs)

    connector.write_channel = spy  # type: ignore[method-assign]
    with patch("osprey.runtime._get_connector", new_callable=AsyncMock) as get:
        get.return_value = connector
        with pytest.raises(OspreyWriteRefused) as info:
            _list().set([10.0, 20.0, 99.0], confirm=False)
    assert info.value.channel_address == SETS[2]
    assert sent == []
    assert connector._state.get(SETS[0]) != 10.0


# --- read-failure mapping ------------------------------------------------------


def test_a_non_numeric_value_is_a_failed_read(runtime: _Runtime) -> None:
    runtime.values[SETS[2]] = object()
    with pytest.raises(OspreyReadFailed):
        _list().get()
    assert _list().check_device_availability() is False


@pytest.mark.usefixtures("guarded")
def test_any_journal_read_error_maps_to_read_failed(
    runtime: _Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    boom = RuntimeError("connector could not be acquired")

    def read_channels(_addresses: Sequence[str], **_kwargs: Any) -> list[Any]:
        raise boom

    monkeypatch.setattr(osprey.runtime, "read_channels", read_channels)
    with pytest.raises(OspreyReadFailed) as info:
        _list().set([10.0, 20.0, 30.0])
    assert info.value.addresses == SETS
    assert info.value.__cause__ is boom
    assert runtime.batch_writes == []


# --- duplicate addresses ---------------------------------------------------------


def test_one_address_given_two_values_is_refused_before_any_call(runtime: _Runtime) -> None:
    with pytest.raises(ValueError, match=SETS[0]):
        _list([REFS[0], REFS[1], REFS[0]]).set([1.0, 2.0, 3.0])
    assert runtime.calls == []


@pytest.mark.usefixtures("guarded")
def test_one_address_given_the_same_value_twice_is_written_once(runtime: _Runtime) -> None:
    _list([REFS[0], REFS[1], REFS[0]]).set([1.0, 2.0, 1.0])
    assert runtime.batch_writes == [({SETS[0]: 1.0, SETS[1]: 2.0}, {})]


# --- reference mode --------------------------------------------------------------


@pytest.mark.usefixtures("guarded")
def test_a_bare_device_is_written_at_its_one_address(runtime: _Runtime) -> None:
    runtime.values["TUNE:x"] = 0.25
    _list([REFS[0], "TUNE:x"]).set([1.0, 0.3])
    assert runtime.batch_writes == [({SETS[0]: 1.0, "TUNE:x": 0.3}, {})]


def test_a_write_only_device_refuses_the_read(runtime: _Runtime) -> None:
    with pytest.raises(PyAMLException, match="write-only"):
        _list([REFS[0], "(HCOR_001:Cm:set)"]).get()
    assert runtime.calls == []


# --- refusals in a batch ---------------------------------------------------------


@pytest.mark.parametrize("reason", ["WRITES_DISABLED", "LIMITS"])
@pytest.mark.usefixtures("guarded")
def test_a_pre_send_refusal_of_a_batch_stays_refused(runtime: _Runtime, reason: str) -> None:
    runtime.write_error = ChannelWriteBlockedError(SETS[0], reason)
    with pytest.raises(OspreyWriteRefused):
        _list().set([10.0, 20.0, 30.0])


@pytest.mark.usefixtures("guarded")
def test_a_single_channel_control_system_refusal_stays_refused(runtime: _Runtime) -> None:
    runtime.write_error = ChannelWriteBlockedError(SETS[0], "CONTROL_SYSTEM_REFUSED")
    with pytest.raises(OspreyWriteRefused):
        _list(REFS[:1]).set([10.0])


class _RefusingConnector(_FailingConnector):
    """A connector that refuses one address with ``reason`` and writes the rest."""

    def __init__(self, refused: str, reason: str) -> None:
        super().__init__("NONE")
        self._refused = refused
        self._reason = reason

    async def write_channel(self, channel_address: str, value: Any, **kwargs: Any):
        if channel_address == self._refused:
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.REFUSED,
                refusal_reason=self._reason,
                error_message=f"Write to '{channel_address}' refused ({self._reason})",
            )
        return await super().write_channel(channel_address, value, **kwargs)


@pytest.mark.usefixtures("unvalidated")
@pytest.mark.parametrize("reason", ["CONTROL_SYSTEM_REFUSED", "VALIDATION_ERROR"])
@pytest.mark.usefixtures("guarded")
def test_a_refusal_in_the_middle_of_a_batch_names_the_channels_sent(reason: str) -> None:
    connector = _RefusingConnector(SETS[1], reason)
    with patch("osprey.runtime._get_connector", new_callable=AsyncMock) as get:
        get.return_value = connector
        with pytest.raises(OspreyWriteFailed) as info:
            _list().set([10.0, 20.0, 30.0], confirm=False)
    assert info.value.channel_address == SETS[1]
    assert f"also sent in this batch: {SETS[0]}, {SETS[2]}" in str(info.value)
    assert isinstance(info.value.__cause__, ChannelWriteBlockedError)
    assert connector._state[SETS[0]] == 10.0
    assert connector._state[SETS[2]] == 30.0


# --- indexed members ---------------------------------------------------------


def test_indexed_list_reads_base_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Indexed members sharing one waveform cost one read of it, each taking its element."""
    runtime = _Runtime(
        {"BPM:x": np.array([1.0, 2.0, 3.0]), "Q:k": 0.5, "TUNE": [0.1]},
        batch_only=True,
    ).install(monkeypatch)
    dl = _list(["BPM:x@2", "Q:k", "BPM:x@0[mm]", "BPM:x@1"])

    np.testing.assert_allclose(dl.get(), [3.0, 0.5, 1e-3, 2.0], rtol=1e-12)
    np.testing.assert_allclose(dl.readback(), [3.0, 0.5, 1e-3, 2.0], rtol=1e-12)
    assert runtime.batch_reads == [["BPM:x", "Q:k"], ["BPM:x", "Q:k"]]
    assert dl.check_device_availability() is True

    with pytest.raises(OspreyReadFailed) as info:
        _list(["BPM:x@0", "BPM:x@3"]).get()
    assert info.value.addresses == ["BPM:x"]
    assert "'BPM:x@3'" in str(info.value)
    assert "length 3" in str(info.value)

    with pytest.raises(OspreyReadFailed) as info:
        _list(["BPM:x@0", "Q:k@0"]).readback()
    assert info.value.addresses == ["Q:k"]
    assert "'Q:k@0'" in str(info.value)
    assert "scalar" in str(info.value)

    assert _list(["TUNE@1"]).check_device_availability() is False


def test_indexed_list_set_refused(runtime: _Runtime, guarded) -> None:
    """A list set holding any indexed member writes and journals nothing."""
    runtime.values["BPM:x"] = [1.0, 2.0]
    dl = _list([REFS[0], "BPM:x@1", REFS[1]])
    runtime.calls.clear()

    for write in (lambda: dl.set([1.0, 2.0, 3.0]), lambda: dl.set_and_wait([1.0, 2.0, 3.0])):
        with pytest.raises(PyAMLException, match=r"'BPM:x@1'.*read-only"):
            write()
    # A wrongly sized value is still refused for the indexed member, not the size.
    with pytest.raises(PyAMLException, match=r"'BPM:x@1'"):
        dl.set([1.0])

    assert runtime.calls == []
    assert runtime.batch_writes == []
    assert guarded.values == {}
