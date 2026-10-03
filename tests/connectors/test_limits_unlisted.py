"""A channel with no limits record, on hello-world's rendered limits database.

The limits database a hello-world render carries holds three records: two
bounded quadrupole setpoints and the beam current, which is not writable. A
channel with no record follows ``control_system.limits_checking.mode`` whether
or not the facility serves its address: ``optional`` writes it with no limits,
``exclusive`` refuses it. The three records mean the same under both modes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.connectors.control_system.limits_validator import LimitsValidator
from osprey.errors import ChannelLimitsViolationError
from osprey_connectors.types import LIMITS_MODES
from tests.facility._limits_render import render_limits, validator_under

QF = "SR:MAG:QF:01:CURRENT:SP"
QD = "SR:MAG:QD:01:CURRENT:SP"
BEAM = "SR:BEAM:CURRENT"

#: A readback the facility serves and ``limits.yaml`` does not list.
SERVED_READBACK = "SR:MAG:QF:01:CURRENT:RB"
#: An address the facility file does not hold.
OUTSIDE = "LAB:NOT:SERVED:SP"

RECORDLESS = [SERVED_READBACK, OUTSIDE]


@pytest.fixture
def validator(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str) -> LimitsValidator:
    """Hello-world's rendered limits database, loaded under ``mode``."""
    return validator_under(monkeypatch, render_limits(tmp_path, "hello_world"), mode)


def _violation(validator: LimitsValidator, channel: str, value: float) -> str:
    with pytest.raises(ChannelLimitsViolationError) as refusal:
        validator.validate(channel, value)
    return str(refusal.value.violation_type)


def test_the_recordless_channels_have_no_entry(tmp_path: Path) -> None:
    import json

    document = json.loads(render_limits(tmp_path, "hello_world").read_bytes())

    assert sorted(document) == sorted(["_version", BEAM, QD, QF])


@pytest.mark.parametrize("mode", ["optional"])
@pytest.mark.parametrize("channel", RECORDLESS)
@pytest.mark.parametrize("value", [0.0, 150.0, -1.0e12, 1.0e12])
def test_optional_writes_a_recordless_channel_with_no_limits(
    validator: LimitsValidator, channel: str, value: float
) -> None:
    validator.validate(channel, value)


@pytest.mark.parametrize("mode", ["exclusive"])
@pytest.mark.parametrize("channel", RECORDLESS)
def test_exclusive_refuses_a_recordless_channel(validator: LimitsValidator, channel: str) -> None:
    assert _violation(validator, channel, 0.0) == "UNLISTED_CHANNEL"


@pytest.mark.parametrize("mode", LIMITS_MODES)
class TestTheThreeRecordsUnderEitherMode:
    @pytest.mark.parametrize(("channel", "value"), [(QF, 0.0), (QF, 300.0), (QD, 0.0), (QD, 250.0)])
    def test_a_bounded_setpoint_takes_a_value_inside_its_band(
        self, validator: LimitsValidator, channel: str, value: float
    ) -> None:
        validator.validate(channel, value)

    @pytest.mark.parametrize(
        ("channel", "value", "violation"),
        [
            (QF, 300.5, "MAX_EXCEEDED"),
            (QF, -0.5, "MIN_EXCEEDED"),
            (QD, 250.5, "MAX_EXCEEDED"),
            (QD, -0.5, "MIN_EXCEEDED"),
        ],
    )
    def test_a_bounded_setpoint_is_held_to_its_band(
        self, validator: LimitsValidator, channel: str, value: float, violation: str
    ) -> None:
        assert _violation(validator, channel, value) == violation

    def test_the_beam_current_is_refused(self, validator: LimitsValidator) -> None:
        assert _violation(validator, BEAM, 1.0) == "READ_ONLY_CHANNEL"
