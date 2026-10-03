"""The two limits modes, on one limits file.

``control_system.limits_checking.mode`` is ``exclusive`` or ``optional``.
Under ``exclusive`` the limits file is the complete list of writable channels:
a channel with no record is refused. Under ``optional`` a channel with a record
is checked against it and a channel with no record is written with no limits.
A record means the same thing in both modes.

The file here holds one bounded setpoint record and one ``writable: false``
record, loaded through ``LimitsValidator.from_config`` so the mode is read from
the config the way a deployment states it. The build half pins the two
refusals a config earns: a ``mode`` that is not one of the two values, and a
leaf the limits block does not define.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_profile_deploy import limits_block_errors
from osprey.connectors.control_system.limits_validator import LimitsValidator
from osprey.errors import ChannelLimitsViolationError
from osprey_connectors.types import (
    LIMITS_MODES,
    LimitsPosture,
    incomplete_limits_blocks,
    type_limits_posture,
)

BOUNDED = "DEMO:MAG:HCM:01:CURRENT:SP"
LOCKED = "DEMO:VAC:PUMP:01:VOLTAGE:SP"
UNLISTED = "DEMO:RF:CAVITY:01:FREQUENCY:SP"

LIMITS_FILE = {
    BOUNDED: {"min_value": -12.0, "max_value": 12.0},
    LOCKED: {"writable": False},
}

MODE_KEY = "control_system.limits_checking.mode"
VA = "virtual_accelerator"
VA_MODE_KEY = f"control_system.connector.{VA}.limits_checking.mode"

#: A leaf a limits block does not define, spelled the way a config written for
#: the boolean this key replaced would still carry it.
LEFTOVER_LEAF = "allow_unlisted_channels"


def _validator(monkeypatch, tmp_path: Path, mode: Any) -> LimitsValidator:
    """Load the limits file under *mode*, stated deployment-wide."""
    db_file = tmp_path / "channel_limits.json"
    db_file.write_text(json.dumps(LIMITS_FILE), encoding="utf-8")
    values = {
        "control_system": {"limits_checking": {"enabled": True, "mode": mode}},
        "control_system.limits_checking.database_path": str(db_file),
        "project_root": None,
    }
    monkeypatch.setattr(
        "osprey.utils.config.get_config_value", lambda key, default=None: values.get(key, default)
    )
    monkeypatch.setattr("osprey.utils.config.default_config_path", lambda: None)
    validator = LimitsValidator.from_config()
    assert isinstance(validator, LimitsValidator)
    return validator


def _refusal(validator: LimitsValidator, channel: str, value: float) -> ChannelLimitsViolationError:
    with pytest.raises(ChannelLimitsViolationError) as exc:
        validator.validate(channel, value)
    return exc.value


class TestTheModeDecidesAChannelWithNoRecord:
    def test_the_two_modes_are_exclusive_and_optional(self) -> None:
        assert LIMITS_MODES == ("exclusive", "optional")

    def test_exclusive_refuses_a_channel_with_no_record(self, monkeypatch, tmp_path) -> None:
        validator = _validator(monkeypatch, tmp_path, "exclusive")

        refusal = _refusal(validator, UNLISTED, 499.5)

        assert refusal.violation_type == "UNLISTED_CHANNEL"
        assert MODE_KEY in refusal.violation_reason

    @pytest.mark.parametrize("value", [499.5, -1.0e12, 1.0e12])
    def test_optional_writes_a_channel_with_no_record_with_no_bounds(
        self, monkeypatch, tmp_path, value: float
    ) -> None:
        validator = _validator(monkeypatch, tmp_path, "optional")

        validator.validate(UNLISTED, value)

    def test_an_unstated_mode_refuses_a_channel_with_no_record(self, monkeypatch, tmp_path) -> None:
        db_file = tmp_path / "channel_limits.json"
        db_file.write_text(json.dumps(LIMITS_FILE), encoding="utf-8")
        values = {
            "control_system": {"limits_checking": {"enabled": True}},
            "control_system.limits_checking.database_path": str(db_file),
        }
        monkeypatch.setattr(
            "osprey.utils.config.get_config_value",
            lambda key, default=None: values.get(key, default),
        )
        monkeypatch.setattr("osprey.utils.config.default_config_path", lambda: None)

        refusal = _refusal(LimitsValidator.from_config(), UNLISTED, 499.5)

        assert refusal.violation_type == "UNLISTED_CHANNEL"


@pytest.mark.parametrize("mode", LIMITS_MODES)
class TestARecordMeansTheSameInBothModes:
    @pytest.mark.parametrize("value", [-12.0, 0.0, 12.0])
    def test_the_bounded_record_takes_a_value_inside_its_bounds(
        self, monkeypatch, tmp_path, mode: str, value: float
    ) -> None:
        _validator(monkeypatch, tmp_path, mode).validate(BOUNDED, value)

    @pytest.mark.parametrize(
        ("value", "violation"), [(12.5, "MAX_EXCEEDED"), (-12.5, "MIN_EXCEEDED")]
    )
    def test_the_bounded_record_is_enforced(
        self, monkeypatch, tmp_path, mode: str, value: float, violation: str
    ) -> None:
        refusal = _refusal(_validator(monkeypatch, tmp_path, mode), BOUNDED, value)

        assert refusal.violation_type == violation

    def test_the_writable_false_record_is_refused(self, monkeypatch, tmp_path, mode: str) -> None:
        refusal = _refusal(_validator(monkeypatch, tmp_path, mode), LOCKED, 0.0)

        assert refusal.violation_type == "READ_ONLY_CHANNEL"


class TestTheModeIsPerConnectorType:
    SECTION = {
        "limits_checking": {"enabled": True, "mode": "exclusive"},
        "connector": {VA: {"limits_checking": {"enabled": True, "mode": "optional"}}},
    }

    def test_a_per_type_block_answers_for_its_type(self) -> None:
        posture = type_limits_posture(self.SECTION, VA)

        assert posture == LimitsPosture(True, "optional", VA)
        assert posture.key("mode") == VA_MODE_KEY

    def test_a_type_with_no_block_reads_the_deployment_wide_mode(self) -> None:
        posture = type_limits_posture(self.SECTION, "epics")

        assert posture == LimitsPosture(True, "exclusive", None)
        assert posture.key("mode") == MODE_KEY

    def test_strict_is_enabled_and_exclusive(self) -> None:
        assert type_limits_posture(self.SECTION, "epics").strict is True
        assert type_limits_posture(self.SECTION, VA).strict is False


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------


def _render_errors(tmp_path: Path, control_system: dict[str, Any]) -> list[str]:
    """The build's render-side limits check on a rendered ``config.yml``."""
    from osprey.cli.build_cmd import _incomplete_limits_errors

    render_dir = tmp_path / "render"
    render_dir.mkdir()
    (render_dir / "config.yml").write_text(
        yaml.safe_dump({"project_name": "demo", "control_system": control_system}),
        encoding="utf-8",
    )
    return _incomplete_limits_errors(render_dir)


def _build(tmp_path: Path, config: dict[str, Any]):
    """Run ``osprey build`` against a repo whose profile states *config*."""
    from osprey.cli.build_cmd import build

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "data").mkdir()
    (repo / "profile.yml").write_text(
        yaml.safe_dump(
            {"name": "Demo Facility", "data": "data", "config": config}, sort_keys=False
        ),
        encoding="utf-8",
    )
    return CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])


def _names_the_mode(message: str) -> bool:
    return "limits_checking.mode" in message and "exclusive | optional" in message


BAD_MODES = ["strict", "Optional", True, False, 1, None, "${OSPREY_LIMITS_MODE}"]


class TestABadModeFailsTheBuild:
    @pytest.mark.parametrize("mode", BAD_MODES)
    def test_the_render_check_names_the_two_values(self, tmp_path, mode: Any) -> None:
        errors = _render_errors(tmp_path, {"limits_checking": {"enabled": True, "mode": mode}})

        assert errors == [
            f"{MODE_KEY} is {mode!r}, not exclusive or optional; a limits leaf that "
            "cannot be read states no posture and blocks every write as a failsafe"
        ]

    @pytest.mark.parametrize("mode", BAD_MODES)
    def test_a_per_type_bad_mode_names_the_per_type_key(self, mode: Any) -> None:
        section = {"connector": {VA: {"limits_checking": {"enabled": True, "mode": mode}}}}

        (error,) = incomplete_limits_blocks(section)

        assert error.startswith(f"{VA_MODE_KEY} is {mode!r}, not exclusive or optional")

    @pytest.mark.parametrize("key", [MODE_KEY, VA_MODE_KEY])
    def test_the_profile_lint_names_the_two_values(self, key: str) -> None:
        config = {key.rsplit(".", 1)[0] + ".enabled": True, key: "strict"}

        (error,) = limits_block_errors(config)

        assert f"`{key}` as 'strict'" in error
        assert _names_the_mode(error)

    def test_osprey_build_refuses_it(self, tmp_path) -> None:
        result = _build(tmp_path, {MODE_KEY: "strict"})

        assert result.exit_code != 0, result.output
        assert "Profile validation failed" in result.output
        assert _names_the_mode(result.output)

    @pytest.mark.parametrize("mode", LIMITS_MODES)
    def test_the_two_values_pass(self, tmp_path, mode: str) -> None:
        assert _render_errors(tmp_path, {"limits_checking": {"enabled": True, "mode": mode}}) == []
        assert limits_block_errors({MODE_KEY: mode}) == []


class TestALeafTheBlockDoesNotDefineFailsTheBuild:
    def test_the_render_check_names_the_leaf_and_the_mode(self, tmp_path) -> None:
        errors = _render_errors(
            tmp_path, {"limits_checking": {"enabled": True, LEFTOVER_LEAF: True}}
        )

        assert errors == [
            f"control_system.limits_checking.{LEFTOVER_LEAF} is not a limits leaf; "
            "a limits block states enabled and limits_checking.mode: exclusive | optional"
        ]

    def test_a_per_type_block_is_checked_too(self) -> None:
        block = {"enabled": True, "mode": "optional", LEFTOVER_LEAF: True}

        errors = incomplete_limits_blocks({"connector": {VA: {"limits_checking": block}}})

        assert errors == [
            f"control_system.connector.{VA}.limits_checking.{LEFTOVER_LEAF} is not a limits "
            "leaf; a limits block states enabled and limits_checking.mode: exclusive | optional"
        ]

    def test_database_path_is_a_leaf_the_block_defines(self, tmp_path) -> None:
        block = {"enabled": True, "mode": "exclusive", "database_path": "data/channel_limits.json"}

        assert _render_errors(tmp_path, {"limits_checking": block}) == []

    @pytest.mark.parametrize(
        "key",
        [
            f"control_system.limits_checking.{LEFTOVER_LEAF}",
            f"control_system.connector.{VA}.limits_checking.{LEFTOVER_LEAF}",
        ],
    )
    def test_the_profile_lint_names_the_entry_and_the_mode(self, key: str) -> None:
        block = key.rsplit(".", 1)[0]
        config = {f"{block}.enabled": True, f"{block}.mode": "optional", key: True}

        (error,) = limits_block_errors(config)

        assert f"`{key}`" in error
        assert "`enabled`" in error
        assert _names_the_mode(error)

    def test_osprey_build_refuses_it(self, tmp_path) -> None:
        result = _build(tmp_path, {f"control_system.limits_checking.{LEFTOVER_LEAF}": True})

        assert result.exit_code != 0, result.output
        assert "Profile validation failed" in result.output
        assert LEFTOVER_LEAF in result.output
        assert "enabled" in result.output
        assert _names_the_mode(result.output)
