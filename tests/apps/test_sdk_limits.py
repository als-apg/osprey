"""``prewrite_verdict`` relays the target's connector limits judgement before a write.

Each test writes a temporary limits file and a config, then asks for a verdict:
a declared record, an unlisted address under each mode and under no mode, a
``max_step`` breach with and without the present value, a ``writable: false``
record, limits off, a fail-safe validator, and a per-type block overriding the
deployment-wide mode.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osprey.apps.sdk import WriteVerdictView, _limits, prewrite_verdict

BOUNDED = "SR:HCM:01:CURRENT:SP"
STEPPED = "SR:HCM:02:CURRENT:SP"
LOCKED = "SR:VAC:01:VOLTAGE:SP"
UNLISTED = "SR:RF:01:FREQUENCY:SP"

LIMITS_FILE = {
    BOUNDED: {"writable": True, "min_value": -12.0, "max_value": 12.0},
    STEPPED: {"writable": True, "min_value": -10.0, "max_value": 10.0, "max_step": 0.5},
    LOCKED: {"writable": False},
}

VA = "virtual_accelerator"


@pytest.fixture(autouse=True)
def _fresh_cache():
    _limits._validator.cache_clear()
    yield
    _limits._validator.cache_clear()


def _configure(
    monkeypatch,
    tmp_path: Path,
    limits_block: dict[str, Any],
    *,
    connector: dict[str, Any] | None = None,
    database: bool = True,
) -> None:
    """Point the connector config at a temporary limits file under *limits_block*."""
    db_file = tmp_path / "channel_limits.json"
    if database:
        db_file.write_text(json.dumps(LIMITS_FILE), encoding="utf-8")
    section: dict[str, Any] = {"type": "epics", "limits_checking": limits_block}
    if connector is not None:
        section["connector"] = connector
    values = {
        "control_system": section,
        "control_system.limits_checking.database_path": str(db_file),
        "project_root": None,
    }
    monkeypatch.setattr(
        "osprey_connectors.config.get_config_value",
        lambda key, default=None: values.get(key, default),
    )
    monkeypatch.setattr("osprey_connectors.config.default_config_path", lambda: None)


def _on(mode: str | None) -> dict[str, Any]:
    block: dict[str, Any] = {"enabled": True}
    if mode is not None:
        block["mode"] = mode
    return block


class TestADeclaredRecord:
    @pytest.mark.parametrize("mode", ["exclusive", "optional"])
    def test_a_value_inside_the_bounds_is_allowed_and_the_record_shown(
        self, monkeypatch, tmp_path, mode
    ):
        _configure(monkeypatch, tmp_path, _on(mode))

        verdict = prewrite_verdict(BOUNDED, 3.0, target="live", current=2.0)

        assert verdict == WriteVerdictView(
            allowed=True,
            reason=None,
            declared=True,
            enforced=True,
            mode=mode,
            limits={"min": -12.0, "max": 12.0, "max_step": None},
        )

    def test_a_value_past_the_max_is_refused_with_the_validators_reason(
        self, monkeypatch, tmp_path
    ):
        _configure(monkeypatch, tmp_path, _on("exclusive"))

        verdict = prewrite_verdict(BOUNDED, 12.5, target="live", current=0.0)

        assert verdict.allowed is False
        assert verdict.declared is True
        assert verdict.reason.startswith("MAX_EXCEEDED: ")

    def test_writable_false_is_refused(self, monkeypatch, tmp_path):
        _configure(monkeypatch, tmp_path, _on("optional"))

        verdict = prewrite_verdict(LOCKED, 0.0, target="live", current=0.0)

        assert verdict.allowed is False
        assert verdict.declared is True
        assert verdict.reason.startswith("READ_ONLY_CHANNEL: ")


class TestAnUnlistedAddress:
    def test_exclusive_refuses_it(self, monkeypatch, tmp_path):
        _configure(monkeypatch, tmp_path, _on("exclusive"))

        verdict = prewrite_verdict(UNLISTED, 499.5, target="live", current=499.0)

        assert verdict.allowed is False
        assert verdict.declared is False
        assert verdict.limits is None
        assert verdict.mode == "exclusive"
        assert verdict.reason.startswith("UNLISTED_CHANNEL: ")

    def test_optional_allows_it_with_no_limits(self, monkeypatch, tmp_path):
        _configure(monkeypatch, tmp_path, _on("optional"))

        verdict = prewrite_verdict(UNLISTED, 1.0e12, target="live", current=None)

        assert verdict == WriteVerdictView(
            allowed=True, reason=None, declared=False, enforced=True, mode="optional", limits=None
        )

    def test_an_unstated_mode_refuses_it(self, monkeypatch, tmp_path):
        _configure(monkeypatch, tmp_path, _on(None))

        verdict = prewrite_verdict(UNLISTED, 499.5, target="live", current=499.0)

        assert verdict.allowed is False
        assert verdict.enforced is True
        assert verdict.mode is None
        assert verdict.reason.startswith("UNLISTED_CHANNEL: ")


class TestMaxStep:
    def test_a_step_within_max_step_of_current_is_allowed(self, monkeypatch, tmp_path):
        _configure(monkeypatch, tmp_path, _on("exclusive"))

        verdict = prewrite_verdict(STEPPED, 1.4, target="live", current=1.0)

        assert verdict.allowed is True
        assert verdict.limits == {"min": -10.0, "max": 10.0, "max_step": 0.5}

    def test_a_step_past_max_step_of_current_is_refused(self, monkeypatch, tmp_path):
        _configure(monkeypatch, tmp_path, _on("exclusive"))

        verdict = prewrite_verdict(STEPPED, 2.0, target="live", current=1.0)

        assert verdict.allowed is False
        assert verdict.reason.startswith("MAX_STEP_EXCEEDED: ")

    def test_without_current_a_max_step_record_refuses_because_the_step_cannot_be_measured(
        self, monkeypatch, tmp_path
    ):
        _configure(monkeypatch, tmp_path, _on("exclusive"))

        for value in (1.2, 2.0):
            verdict = prewrite_verdict(STEPPED, value, target="live", current=None)

            assert verdict.allowed is False
            assert verdict.declared is True
            assert verdict.reason.startswith("STEP_CHECK_FAILED: ")


class TestLimitsOffAndFailSafe:
    def test_limits_off_is_not_enforced_and_allows(self, monkeypatch, tmp_path):
        _configure(monkeypatch, tmp_path, {"enabled": False, "mode": "exclusive"})

        verdict = prewrite_verdict(UNLISTED, 1.0e12, target="live", current=None)

        assert verdict == WriteVerdictView(
            allowed=True, reason=None, declared=False, enforced=False, mode=None, limits=None
        )

    def test_an_unavailable_database_refuses_every_write_with_the_fail_safe_reason(
        self, monkeypatch, tmp_path
    ):
        _configure(monkeypatch, tmp_path, _on("exclusive"), database=False)

        verdict = prewrite_verdict(BOUNDED, 0.0, target="live", current=0.0)

        assert verdict.allowed is False
        assert verdict.enforced is True
        assert verdict.declared is False
        assert verdict.reason.startswith("LIMITS_DATABASE_UNAVAILABLE: ")


class TestThePostureIsTheTargets:
    CONNECTOR = {VA: {"limits_checking": {"enabled": True, "mode": "optional"}}}

    def test_a_per_type_block_overrides_the_deployment_wide_mode(self, monkeypatch, tmp_path):
        _configure(monkeypatch, tmp_path, _on("exclusive"), connector=self.CONNECTOR)

        va = prewrite_verdict(UNLISTED, 499.5, target="va", current=None)
        live = prewrite_verdict(UNLISTED, 499.5, target="live", current=None)

        assert (va.allowed, va.mode) == (True, "optional")
        assert (live.allowed, live.mode) == (False, "exclusive")

    def test_the_validator_is_built_once_per_target(self, monkeypatch, tmp_path):
        _configure(monkeypatch, tmp_path, _on("exclusive"))

        prewrite_verdict(BOUNDED, 0.0, target="live", current=0.0)
        prewrite_verdict(BOUNDED, 1.0, target="live", current=0.0)
        prewrite_verdict(BOUNDED, 1.0, target="va", current=0.0)

        assert _limits._validator.cache_info().misses == 2
