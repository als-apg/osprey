"""Connector exceptions map onto the pyAML exceptions of ``pyaml_cs_osprey.errors``."""

from __future__ import annotations

import pytest
from pyaml.common.exception import PyAMLException

from osprey.errors import (
    ChannelLimitsViolationError,
    ChannelReadFailedError,
    ChannelWriteBlockedError,
    ChannelWriteFailedError,
)
from osprey.runtime import ControlTargetChangedError, SwitchInProgressError
from pyaml_cs_osprey.errors import (
    OspreyReadFailed,
    OspreyWriteFailed,
    OspreyWriteRefused,
    map_read_error,
    map_write_error,
)

ADDR = "QD_015:Cm:set"


def _limits_violation() -> ChannelLimitsViolationError:
    return ChannelLimitsViolationError(
        ADDR,
        1.5,
        "MAX_STEP",
        "Step 0.5 exceeds max_step 0.1",
        max_step=0.1,
        current_value=1.0,
    )


@pytest.mark.parametrize("cls", [OspreyWriteRefused, OspreyWriteFailed, OspreyReadFailed])
def test_mapped_classes_are_pyaml_exceptions(cls: type) -> None:
    assert issubclass(cls, PyAMLException)


def test_direct_limits_violation_is_refused_with_violation_reason() -> None:
    original = _limits_violation()
    mapped = map_write_error(original)
    assert isinstance(mapped, OspreyWriteRefused)
    assert mapped.__cause__ is original
    assert mapped.reason == "Step 0.5 exceeds max_step 0.1"
    assert "Step 0.5 exceeds max_step 0.1" in str(mapped)
    assert ADDR in str(mapped)
    # A restore-report reason stays one line, not the multi-line banner.
    assert "\n" not in mapped.reason


def test_blocked_limits_with_banner_is_refused_with_violation_line() -> None:
    banner = str(_limits_violation())
    original = ChannelWriteBlockedError(ADDR, "LIMITS", message=banner)
    mapped = map_write_error(original)
    assert isinstance(mapped, OspreyWriteRefused)
    assert mapped.__cause__ is original
    assert "Step 0.5 exceeds max_step 0.1" in mapped.reason
    assert "\n" not in mapped.reason
    assert "Step 0.5 exceeds max_step 0.1" in str(mapped)
    assert mapped.channel_address == ADDR


def test_blocked_limits_takes_the_violation_from_its_cause() -> None:
    """The connector's LIMITS refusal is raised from the violation; its reason wins."""
    violation = _limits_violation()
    original = ChannelWriteBlockedError(ADDR, "LIMITS", message="refused by limits")
    original.__cause__ = violation
    mapped = map_write_error(original)
    assert isinstance(mapped, OspreyWriteRefused)
    assert mapped.reason == "LIMITS: Step 0.5 exceeds max_step 0.1"


@pytest.mark.parametrize(
    ("reason", "message", "expected"),
    [
        ("WRITES_DISABLED", None, "WRITES_DISABLED"),
        ("VALIDATION_ERROR", "writable: false for QD_016:Cm:set", "writable: false"),
        ("CONTROL_SYSTEM_REFUSED", None, "CONTROL_SYSTEM_REFUSED"),
        ("LIMITS", None, "LIMITS"),
    ],
)
def test_other_blocked_writes_are_refused(reason: str, message: str | None, expected: str) -> None:
    original = ChannelWriteBlockedError(ADDR, reason, message=message)
    mapped = map_write_error(original)
    assert isinstance(mapped, OspreyWriteRefused)
    assert mapped.__cause__ is original
    assert expected in mapped.reason
    assert reason in mapped.reason
    assert expected in str(mapped)


def test_control_target_changed_is_refused() -> None:
    original = ControlTargetChangedError("control target moved to 'production' (generation 4)")
    mapped = map_write_error(original, address=ADDR)
    assert isinstance(mapped, OspreyWriteRefused)
    assert mapped.__cause__ is original
    assert "control target moved" in mapped.reason
    assert mapped.channel_address == ADDR


def test_switch_in_progress_is_refused() -> None:
    original = SwitchInProgressError("switch_in_progress:1234")
    mapped = map_write_error(original, address=ADDR)
    assert isinstance(mapped, OspreyWriteRefused)
    assert mapped.__cause__ is original
    assert "switch_in_progress:1234" in mapped.reason


@pytest.mark.parametrize("reason", ["FAILED", "MISMATCH", "UNCONFIRMED"])
def test_write_failed_maps_to_write_failed(reason: str) -> None:
    original = ChannelWriteFailedError(ADDR, reason, value_written=1.0, observed_value=0.9)
    mapped = map_write_error(original)
    assert isinstance(mapped, OspreyWriteFailed)
    assert not isinstance(mapped, OspreyWriteRefused)
    assert mapped.__cause__ is original
    assert reason in mapped.reason
    assert reason in str(mapped)
    assert ADDR in str(mapped)


def test_unknown_write_exception_is_a_failure_not_a_refusal() -> None:
    original = TimeoutError("gateway did not answer")
    mapped = map_write_error(original, address=ADDR)
    assert isinstance(mapped, OspreyWriteFailed)
    assert mapped.__cause__ is original
    assert "gateway did not answer" in mapped.reason


def test_mapped_write_error_raises_with_cause() -> None:
    original = _limits_violation()
    with pytest.raises(OspreyWriteRefused) as info:
        raise map_write_error(original) from original
    assert info.value.__cause__ is original


def test_read_failed_names_every_address() -> None:
    addresses = ["BPM_001:x", "BPM_002:x", "BPM_003:y"]
    original = ChannelReadFailedError(addresses)
    mapped = map_read_error(original)
    assert isinstance(mapped, OspreyReadFailed)
    assert mapped.__cause__ is original
    assert mapped.addresses == addresses
    for address in addresses:
        assert address in str(mapped)


def test_other_read_exception_names_the_addresses_asked() -> None:
    original = ConnectionError("channel access disconnected")
    mapped = map_read_error(original, ["BPM_001:x", "BPM_002:x"])
    assert isinstance(mapped, OspreyReadFailed)
    assert mapped.__cause__ is original
    assert "channel access disconnected" in mapped.reason
    assert "BPM_001:x" in str(mapped)
    assert "BPM_002:x" in str(mapped)


def test_other_read_exception_names_a_single_address() -> None:
    original = TimeoutError("read timed out")
    mapped = map_read_error(original, ["BPM_001:x"])
    assert mapped.addresses == ["BPM_001:x"]
    assert "read timed out" in str(mapped)
