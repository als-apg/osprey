"""Unit tests for :mod:`osprey.services.python_executor.execution.control`.

This module is part of the write-safety surface: ``ExecutionControlConfig``
carries the posture that decides whether generated code runs in a read-only or
write-enabled sandbox, so its field set and its factory are pinned strictly here
rather than sampled.

``control_system_writes_enabled`` is the single authoritative write-gating field.
Its default is ``False`` (fail-safe: writes disabled unless explicitly enabled),
and write access is granted only when that field is ``True`` *and* the analysed
code actually contains write operations — the gate itself reads the field
directly. The legacy write-gating field has been removed entirely; the exact
field set is pinned below so a reintroduction is a deliberate, reviewed change.
"""

from __future__ import annotations

import dataclasses

import pytest

import osprey.services.python_executor.execution.control as control_mod
from osprey.services.python_executor.execution.control import (
    ExecutionControlConfig,
    ExecutionMode,
    get_execution_control_config,
)


class TestExecutionModeEnum:
    def test_mode_values(self):
        assert ExecutionMode.READ_ONLY.value == "read_only"
        assert ExecutionMode.WRITE_ACCESS.value == "write_access"


class TestConfigFields:
    def test_defaults_are_safe(self):
        cfg = ExecutionControlConfig()
        # Fail-safe default: writes disabled unless explicitly enabled.
        assert cfg.control_system_writes_enabled is False
        # Fail-closed default: an unspecified type is the simulator in process,
        # never a live system.
        assert cfg.control_system_type == "virtual_accelerator"

    def test_modern_field_is_honoured(self):
        cfg = ExecutionControlConfig(control_system_writes_enabled=True)
        assert cfg.control_system_writes_enabled is True

    def test_field_set_is_exactly_the_modern_contract(self):
        # The legacy write-gating field was removed with zero back-compat.
        # Pinning the exact field set means reintroducing it (or any unexpected
        # constructor kwarg) fails naturally as a TypeError and trips this test.
        names = {f.name for f in dataclasses.fields(ExecutionControlConfig)}
        assert names == {
            "control_system_writes_enabled",
            "control_system_type",
            "active_target",
            "writes_enabled_key",
        }

    def test_target_identity_defaults_to_unknown(self):
        # A config built without reading a target names none, and the key it
        # reports is the login tier's ceiling — the one key that governs it.
        cfg = ExecutionControlConfig()
        assert cfg.active_target is None
        assert cfg.writes_enabled_key == "control_system.writes"


class TestValidate:
    def test_no_warning_when_writes_disabled(self):
        assert ExecutionControlConfig(control_system_writes_enabled=False).validate() == []

    def test_warning_when_writes_enabled(self):
        warnings = ExecutionControlConfig(control_system_writes_enabled=True).validate()
        assert len(warnings) == 1
        assert "control_system.writes=allowed" in warnings[0]


class TestGetExecutionControlConfigFactory:
    def test_reads_writes_enabled_true(self, monkeypatch):
        def fake_get_config_value(path, _default=None, _config_path=None):
            assert path == "control_system"
            return {"writes_enabled": True, "type": "epics"}

        monkeypatch.setattr("osprey_connectors.config.get_config_value", fake_get_config_value)
        cfg = get_execution_control_config()
        assert cfg.control_system_writes_enabled is True
        assert cfg.control_system_type == "epics"

    def test_defaults_when_writes_key_missing(self, monkeypatch):
        monkeypatch.setattr(
            "osprey_connectors.config.get_config_value",
            lambda path, default=None, config_path=None: {},
        )
        cfg = get_execution_control_config()
        assert cfg.control_system_writes_enabled is False
        # Type falls back to the simulator in process, not to a live control system.
        assert cfg.control_system_type == control_mod.VIRTUAL_ACCELERATOR

    def test_exception_falls_back_to_safe_defaults(self, monkeypatch):
        def boom(_path, _default=None, _config_path=None):
            raise RuntimeError("config unavailable")

        monkeypatch.setattr("osprey_connectors.config.get_config_value", boom)
        cfg = get_execution_control_config()
        # On any failure the factory must return a write-disabled config on a
        # non-live control system.
        assert cfg.control_system_writes_enabled is False
        assert cfg.control_system_type == control_mod.VIRTUAL_ACCELERATOR


class TestTierPosture:
    """The factory reports the login tier's ceiling, ``control_system.writes``.

    Target-blind: the tier allows or never allows writes for every machine, and
    which machine is written is the session's switch to decide. The sections
    below are normalized — they state ``writes`` and carry no retired
    ``writes_enabled`` leaf — except where a test names the retired shape.
    """

    ALLOWED = {
        "type": "epics",
        "writes": "allowed",
        "connector": {"epics": {}, "virtual_accelerator": {}},
    }
    NEVER = {"type": "epics", "writes": "never", "connector": {"epics": {}}}

    @staticmethod
    def _config(monkeypatch, section):
        monkeypatch.setattr(
            "osprey_connectors.config.get_config_value",
            lambda path, default=None, config_path=None: section,
        )

    @pytest.mark.parametrize("target", ["live", "va", None, "somewhere-else"])
    def test_an_allowed_render_reports_writes_for_every_target(self, monkeypatch, target):
        # Arrange
        self._config(monkeypatch, self.ALLOWED)

        # Act
        cfg = get_execution_control_config(target=target)

        # Assert
        assert cfg.control_system_writes_enabled is True
        assert cfg.writes_enabled_key == "control_system.writes"

    @pytest.mark.parametrize("target", ["live", "va", None])
    def test_a_never_render_reports_no_writes(self, monkeypatch, target):
        # Arrange
        self._config(monkeypatch, self.NEVER)

        # Act
        cfg = get_execution_control_config(target=target)

        # Assert
        assert cfg.control_system_writes_enabled is False
        assert cfg.writes_enabled_key == "control_system.writes"

    def test_the_target_is_carried_and_none_answers_the_baseline(self, monkeypatch):
        # An unstamped run builds the baseline connector, so the baseline
        # target is the one the config is about.
        self._config(monkeypatch, self.ALLOWED)

        assert get_execution_control_config(target="va").active_target == "va"
        assert get_execution_control_config().active_target == "live"

    def test_the_tier_key_wins_over_a_retired_leaf(self, monkeypatch):
        # A stray retired leaf beside the tier key decides nothing.
        self._config(monkeypatch, {"type": "epics", "writes": "never", "writes_enabled": True})

        cfg = get_execution_control_config(target="live")

        assert cfg.control_system_writes_enabled is False

    def test_an_unnormalized_section_is_read_the_way_the_normalizer_rewrites_it(self, monkeypatch):
        # Any armed retired leaf is ``allowed``, for every target.
        self._config(
            monkeypatch,
            {
                "type": "epics",
                "writes_enabled": False,
                "connector": {"epics": {}, "virtual_accelerator": {"writes_enabled": True}},
            },
        )

        cfg = get_execution_control_config(target="live")

        assert cfg.control_system_writes_enabled is True
        assert cfg.writes_enabled_key == "control_system.writes"

    def test_the_warning_names_the_tier_key(self, monkeypatch):
        self._config(monkeypatch, self.ALLOWED)

        warnings = get_execution_control_config(target="va").validate()

        assert len(warnings) == 1
        assert "control_system.writes=allowed" in warnings[0]
