"""An OSPREY device reads and writes one channel reference through ``osprey.runtime``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from pyaml.common.exception import PyAMLException
from pyaml.control.deviceaccess import DeviceAccess

import osprey.runtime
from osprey.errors import (
    ChannelLimitsViolationError,
    ChannelWriteFailedError,
)
from osprey.runtime.journal import Journal, pop_journal, push_journal
from osprey.runtime.journal import OspreyWriteRefused as RuntimeOspreyWriteRefused
from osprey_connectors.control_system.limits_validator import ChannelLimitsConfig
from pyaml_cs_osprey.catalog import parse_reference
from pyaml_cs_osprey.device import OspreyDevice
from pyaml_cs_osprey.errors import OspreyReadFailed, OspreyWriteFailed, OspreyWriteRefused
from tests.pyaml_cs_osprey.conftest import FakeRuntime as _Runtime

RW = "(QF_001:Cm:rdbk, QF_001:Cm:set)[1/m]"


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> _Runtime:
    fake = _Runtime({"QF_001:Cm:set": 1.5, "QF_001:Cm:rdbk": 1.49, "TUNE:x": 0.25})
    return fake.install(
        monkeypatch, ("read_channel", "read_channels", "write_channel", "channel_limits")
    )


def _device(text: str) -> OspreyDevice:
    return OspreyDevice(parse_reference(text))


# --- names ------------------------------------------------------------------


def test_it_is_a_pyaml_device_access() -> None:
    """Every abstract method is implemented, so the class instantiates."""
    assert isinstance(_device(RW), DeviceAccess)


def test_read_write_pair_names() -> None:
    """name() is the setpoint, measure_name() the readback, repr the reference text."""
    dev = _device(RW)
    assert dev.name() == "QF_001:Cm:set"
    assert dev.measure_name() == "QF_001:Cm:rdbk"
    assert repr(dev) == RW
    assert str(dev) == RW


def test_bare_reference_names_its_one_address() -> None:
    """A bare reference names its one address for both."""
    dev = _device("TUNE:x[]")
    assert dev.name() == "TUNE:x"
    assert dev.measure_name() == "TUNE:x"


def test_write_only_names_use_the_setpoint_address() -> None:
    """A write-only reference measures on its setpoint address."""
    dev = _device("(HCOR_001:Cm:set)[rad]")
    assert dev.name() == "HCOR_001:Cm:set"
    assert dev.measure_name() == "HCOR_001:Cm:set"


def test_unit_is_the_si_word_of_the_reference_suffix() -> None:
    """unit() is the SI word of the bracketed suffix, empty when absent."""
    assert _device(RW).unit() == "1/m"
    assert _device("TUNE:x").unit() == ""
    assert _device("RF:freq[MHz]").unit() == "Hz"


# --- units ------------------------------------------------------------------

BPM = "(BPM1:x:rdbk, BPM1:x:set)[mm]"


@pytest.mark.usefixtures("guarded")
def test_device_values_are_si(runtime: _Runtime, guarded: Journal) -> None:
    """A SPEAR3-style BPM in mm reads, writes and ranges in metres; the journal stays native."""
    runtime.values.update({"BPM1:x:set": 2.0, "BPM1:x:rdbk": 1.5})
    runtime.limits["BPM1:x:set"] = ChannelLimitsConfig(
        channel_address="BPM1:x:set", min_value=-4.0, max_value=None, writable=True
    )
    dev = _device(BPM)
    assert dev.unit() == "m"
    assert dev.get() == pytest.approx(2.0e-3)
    assert dev.readback() == pytest.approx(1.5e-3)
    assert dev.get_range() == [pytest.approx(-4.0e-3), None]
    dev.set(3.0e-3)
    assert runtime.writes == [("BPM1:x:set", pytest.approx(3.0), {})]
    assert guarded.values == {"BPM1:x:set": 2.0}


def test_an_unknown_unit_suffix_is_a_pyaml_exception(runtime: _Runtime) -> None:
    """A suffix with no SI factor surfaces as a ``PyAMLException`` and writes nothing."""
    runtime.values["X:y"] = 1.0
    dev = _device("X:y[furlong]")
    for call in (dev.unit, dev.get, lambda: dev.set(1.0)):
        with pytest.raises(PyAMLException, match="furlong"):
            call()
    assert runtime.writes == []


# --- reads ------------------------------------------------------------------


def test_get_reads_the_setpoint_as_float(runtime: _Runtime) -> None:
    """get() reads the setpoint address and returns a float."""
    runtime.values["QF_001:Cm:set"] = 2
    value = _device(RW).get()
    assert value == 2.0 and isinstance(value, float)
    assert runtime.reads == ["QF_001:Cm:set"]


def test_get_on_a_bare_reference_reads_its_address(runtime: _Runtime) -> None:
    """A bare reference's get() reads its one address."""
    assert _device("TUNE:x").get() == 0.25
    assert runtime.reads == ["TUNE:x"]


def test_readback_reads_the_readback_address(runtime: _Runtime) -> None:
    """readback() reads the RB half of a read-write pair."""
    assert _device(RW).readback() == pytest.approx(1.49)
    assert runtime.reads == ["QF_001:Cm:rdbk"]


def test_none_value_is_a_failed_read(runtime: _Runtime) -> None:
    """A timeout that yields None raises OspreyReadFailed naming the address."""
    runtime.values["QF_001:Cm:set"] = None
    with pytest.raises(OspreyReadFailed) as info:
        _device(RW).get()
    assert info.value.addresses == ["QF_001:Cm:set"]


def test_read_exception_maps_to_read_failed(runtime: _Runtime) -> None:
    """A raising read is re-raised as OspreyReadFailed with the cause kept."""
    boom = RuntimeError("channel disconnected")
    runtime.read_error = boom
    with pytest.raises(OspreyReadFailed) as info:
        _device(RW).readback()
    assert info.value.addresses == ["QF_001:Cm:rdbk"]
    assert info.value.__cause__ is boom


def test_availability_is_a_successful_read(runtime: _Runtime) -> None:
    """check_device_availability() is True when the read succeeds, else False."""
    dev = _device(RW)
    assert dev.check_device_availability() is True
    runtime.values["QF_001:Cm:set"] = None
    assert dev.check_device_availability() is False
    runtime.read_error = RuntimeError("down")
    assert dev.check_device_availability() is False


# --- writes -----------------------------------------------------------------


@pytest.mark.usefixtures("guarded")
def test_set_writes_the_setpoint(runtime: _Runtime) -> None:
    """set() writes the setpoint address, leaving confirm to the channel default."""
    _device(RW).set(3.0)
    assert runtime.writes == [("QF_001:Cm:set", 3.0, {})]


def test_set_outside_a_journaled_run_is_refused(runtime: _Runtime) -> None:
    """Outside a journaled guarded run a write is refused, reading and writing nothing."""
    with pytest.raises(RuntimeOspreyWriteRefused, match="only inside pyaml_measure"):
        _device(RW).set(3.0)
    assert runtime.calls == []


def test_set_under_a_pushed_journal_alone_is_refused(runtime: _Runtime) -> None:
    """A journal pushed without the run lock does not open the guarded run."""
    journal = push_journal()
    try:
        with pytest.raises(RuntimeOspreyWriteRefused, match="only inside pyaml_measure"):
            _device(RW).set(3.0)
    finally:
        pop_journal(journal)
    assert runtime.calls == []
    assert journal.values == {}


@pytest.mark.usefixtures("guarded")
def test_set_and_wait_confirms(runtime: _Runtime) -> None:
    """set_and_wait() is a confirmed write."""
    _device(RW).set_and_wait(4.0)
    assert runtime.writes == [("QF_001:Cm:set", 4.0, {"confirm": True})]


def test_set_journals_the_prior_setpoint(runtime: _Runtime, guarded: Journal) -> None:
    """Inside a guard the setpoint before the first write is journaled."""
    dev = _device(RW)
    dev.set(3.0)
    dev.set(5.0)
    assert guarded.values == {"QF_001:Cm:set": 1.5}
    assert runtime.batch_reads == [["QF_001:Cm:set"]]
    assert guarded.max_write_latency is not None


@pytest.mark.usefixtures("runtime")
def test_set_records_into_every_active_journal(guarded: Journal) -> None:
    """A nested guard journals the same prior setpoint at both levels."""
    inner = push_journal()
    try:
        _device(RW).set(3.0)
    finally:
        pop_journal(inner)
    assert guarded.values == inner.values == {"QF_001:Cm:set": 1.5}


@pytest.mark.usefixtures("guarded")
def test_failed_journal_read_writes_nothing(runtime: _Runtime) -> None:
    """If the prior setpoint cannot be read the write never happens."""
    runtime.values["QF_001:Cm:set"] = None
    with pytest.raises(OspreyReadFailed) as info:
        _device(RW).set(3.0)
    assert info.value.addresses == ["QF_001:Cm:set"]
    assert runtime.writes == []


@pytest.mark.usefixtures("guarded")
def test_limits_refusal_maps_to_write_refused(runtime: _Runtime) -> None:
    """A limits violation becomes OspreyWriteRefused naming the channel."""
    err = ChannelLimitsViolationError(
        channel_address="QF_001:Cm:set",
        value=99.0,
        violation_type="MAX_EXCEEDED",
        violation_reason="above maximum 10.0",
    )
    runtime.write_error = err
    with pytest.raises(OspreyWriteRefused) as info:
        _device(RW).set(99.0)
    assert info.value.channel_address == "QF_001:Cm:set"
    assert info.value.__cause__ is err


def test_write_failure_maps_to_write_failed(runtime: _Runtime, guarded: Journal) -> None:
    """A failed write becomes OspreyWriteFailed; its latency is still journaled."""
    err = ChannelWriteFailedError("QF_001:Cm:set", "FAILED")
    runtime.write_error = err
    with pytest.raises(OspreyWriteFailed) as info:
        _device(RW).set(3.0)
    assert info.value.__cause__ is err
    assert guarded.values == {"QF_001:Cm:set": 1.5}
    assert guarded.max_write_latency is not None


@pytest.mark.usefixtures("guarded")
def test_mapped_errors_are_pyaml_exceptions(runtime: _Runtime) -> None:
    """Any write exception surfaces as a PyAMLException."""
    runtime.write_error = RuntimeError("socket closed")
    with pytest.raises(PyAMLException):
        _device(RW).set(1.0)


# --- range ------------------------------------------------------------------


def test_range_comes_from_channel_limits(runtime: _Runtime) -> None:
    """get_range() is [min, max] from the channel's limits entry."""
    runtime.limits["QF_001:Cm:set"] = ChannelLimitsConfig(
        channel_address="QF_001:Cm:set", min_value=-2.0, max_value=2.0, writable=True
    )
    assert _device(RW).get_range() == [-2.0, 2.0]


@pytest.mark.usefixtures("runtime")
def test_range_is_unbounded_without_limits() -> None:
    """With no limits entry both bounds are None."""
    assert _device(RW).get_range() == [None, None]


# --- read-failure mapping -----------------------------------------------------


def test_a_non_numeric_value_is_a_failed_read(runtime: _Runtime) -> None:
    """A value that is not a number fails the read instead of raising ``TypeError``."""
    runtime.values["QF_001:Cm:set"] = object()
    with pytest.raises(OspreyReadFailed) as info:
        _device(RW).get()
    assert info.value.addresses == ["QF_001:Cm:set"]
    assert _device(RW).check_device_availability() is False


@pytest.mark.usefixtures("guarded")
def test_any_journal_read_error_maps_to_read_failed(
    runtime: _Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-read that raises anything (e.g. no connector) writes nothing and is mapped."""
    boom = RuntimeError("connector could not be acquired")

    def read_channels(_addresses: Sequence[str], **_kwargs: Any) -> list[Any]:
        raise boom

    monkeypatch.setattr(osprey.runtime, "read_channels", read_channels)
    with pytest.raises(OspreyReadFailed) as info:
        _device(RW).set(3.0)
    assert info.value.addresses == ["QF_001:Cm:set"]
    assert info.value.__cause__ is boom
    assert runtime.writes == []


# --- reference mode -----------------------------------------------------------


@pytest.mark.usefixtures("guarded")
def test_a_bare_reference_reads_and_writes_one_address(runtime: _Runtime) -> None:
    """A bare reference's set() writes its one address and get() reads it back."""
    dev = _device("master_clock:freq[Hz]")
    runtime.values["master_clock:freq"] = 5e8
    dev.set(5.0001e8)
    assert runtime.writes == [("master_clock:freq", 5.0001e8, {})]
    assert dev.get() == pytest.approx(5.0001e8)
    assert dev.readback() == pytest.approx(5.0001e8)


@pytest.mark.parametrize("method", ["get", "readback"])
def test_reads_on_a_write_only_reference_are_refused(runtime: _Runtime, method: str) -> None:
    """A write-only reference is never read; the refusal is a ``PyAMLException``."""
    runtime.values["HCOR_001:Cm:set"] = 0.001
    dev = _device("(HCOR_001:Cm:set)[rad]")
    with pytest.raises(PyAMLException, match="write-only"):
        getattr(dev, method)()
    assert runtime.reads == []
    assert dev.check_device_availability() is False


# --- indexed references -----------------------------------------------------


def test_indexed_device_reads_and_refuses_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """``A@1`` reads element 1 of ``A``; scalars, short arrays and every write are refused."""
    fake = _Runtime({"A": [0.1, 0.2, 0.3], "S": 4.0, "M": (1.0, 2.0)})
    fake.limits["A"] = ChannelLimitsConfig(
        channel_address="A", min_value=-1.0, max_value=1.0, writable=True
    )
    fake.install(monkeypatch)
    dev = OspreyDevice(parse_reference("A@1"))

    assert dev.get() == pytest.approx(0.2)
    assert dev.readback() == pytest.approx(0.2)
    assert dev.check_device_availability() is True
    assert fake.reads == ["A", "A", "A"]

    scaled = OspreyDevice(parse_reference("M@1[mm]"))
    assert scaled.get() == pytest.approx(2.0e-3)

    with pytest.raises(OspreyReadFailed) as scalar:
        OspreyDevice(parse_reference("S@0")).get()
    assert "S@0" in str(scalar.value) and scalar.value.addresses == ["S"]

    with pytest.raises(OspreyReadFailed) as short:
        OspreyDevice(parse_reference("M@2")).readback()
    assert "M@2" in str(short.value) and "length 2" in str(short.value)
    assert OspreyDevice(parse_reference("M@2")).check_device_availability() is False

    journal = push_journal()
    try:
        for write in (lambda: dev.set(0.5), lambda: dev.set_and_wait(0.5)):
            with pytest.raises(PyAMLException, match="A@1"):
                write()
        assert journal.values == {}
    finally:
        pop_journal(journal)

    assert dev.get_range() == [None, None]
    assert fake.writes == []
    assert "write_channel" not in fake.calls
    assert fake.values["A"] == [0.1, 0.2, 0.3]
