"""Unit tests for :mod:`osprey.services.python_executor.execution.control`.

This module is part of the write-safety surface: ``ExecutionControlConfig``
decides whether generated code runs in a read-only or write-enabled sandbox, so
its gating logic is pinned strictly here rather than sampled.

``control_system_writes_enabled`` is the single authoritative write-gating field.
Its default is ``False`` (fail-safe: writes disabled unless explicitly enabled),
and :meth:`ExecutionControlConfig.get_execution_mode` grants write access only
when that field is ``True`` *and* the analysed code actually contains write
operations. The legacy write-gating field has been removed entirely; the exact
field set is pinned below so a reintroduction is a deliberate, reviewed change.
"""

from __future__ import annotations

import dataclasses

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
        assert cfg.control_system_type == "epics"

    def test_modern_field_is_honoured(self):
        cfg = ExecutionControlConfig(control_system_writes_enabled=True)
        assert cfg.control_system_writes_enabled is True

    def test_field_set_is_exactly_the_modern_contract(self):
        # The legacy write-gating field was removed with zero back-compat.
        # Pinning the exact field set means reintroducing it (or any unexpected
        # constructor kwarg) fails naturally as a TypeError and trips this test.
        names = {f.name for f in dataclasses.fields(ExecutionControlConfig)}
        assert names == {"control_system_writes_enabled", "control_system_type"}


class TestGetExecutionMode:
    def test_write_granted_only_when_detected_and_enabled(self):
        cfg = ExecutionControlConfig(control_system_writes_enabled=True)
        mode = cfg.get_execution_mode(has_epics_writes=True, has_epics_reads=True)
        assert mode is ExecutionMode.WRITE_ACCESS

    def test_write_detected_but_policy_disabled_stays_read_only(self):
        cfg = ExecutionControlConfig(control_system_writes_enabled=False)
        mode = cfg.get_execution_mode(has_epics_writes=True, has_epics_reads=True)
        assert mode is ExecutionMode.READ_ONLY

    def test_default_policy_stays_read_only(self):
        # Fail-safe default (writes disabled) blocks write access.
        cfg = ExecutionControlConfig()
        mode = cfg.get_execution_mode(has_epics_writes=True, has_epics_reads=True)
        assert mode is ExecutionMode.READ_ONLY

    def test_writes_enabled_but_no_write_ops_stays_read_only(self):
        cfg = ExecutionControlConfig(control_system_writes_enabled=True)
        mode = cfg.get_execution_mode(has_epics_writes=False, has_epics_reads=True)
        assert mode is ExecutionMode.READ_ONLY

    def test_reads_flag_does_not_affect_mode(self):
        # has_epics_reads is documented as inert for mode selection.
        cfg = ExecutionControlConfig(control_system_writes_enabled=True)
        with_reads = cfg.get_execution_mode(has_epics_writes=True, has_epics_reads=True)
        without_reads = cfg.get_execution_mode(has_epics_writes=True, has_epics_reads=False)
        assert with_reads is without_reads is ExecutionMode.WRITE_ACCESS

    def test_no_ops_detected_stays_read_only(self):
        cfg = ExecutionControlConfig(control_system_writes_enabled=True)
        mode = cfg.get_execution_mode(has_epics_writes=False, has_epics_reads=False)
        assert mode is ExecutionMode.READ_ONLY


class TestModernFieldAuthoritative:
    """The modern field alone governs get_execution_mode; there is no other gate."""

    def test_modern_field_true_grants_write(self):
        cfg = ExecutionControlConfig(control_system_writes_enabled=True)
        mode = cfg.get_execution_mode(has_epics_writes=True, has_epics_reads=True)
        assert mode is ExecutionMode.WRITE_ACCESS

    def test_modern_field_false_blocks_write(self):
        cfg = ExecutionControlConfig(control_system_writes_enabled=False)
        mode = cfg.get_execution_mode(has_epics_writes=True, has_epics_reads=True)
        assert mode is ExecutionMode.READ_ONLY


class TestValidate:
    def test_no_warning_when_writes_disabled(self):
        assert ExecutionControlConfig(control_system_writes_enabled=False).validate() == []

    def test_warning_when_writes_enabled(self):
        warnings = ExecutionControlConfig(control_system_writes_enabled=True).validate()
        assert len(warnings) == 1
        assert "writes_enabled=true" in warnings[0]


class TestGetExecutionControlConfigFactory:
    def test_reads_writes_enabled_true(self, monkeypatch):
        def fake_get_config_value(path, default=None, config_path=None):
            assert path == "control_system"
            return {"writes_enabled": True, "type": "mock"}

        monkeypatch.setattr("osprey.utils.config.get_config_value", fake_get_config_value)
        cfg = get_execution_control_config()
        assert cfg.control_system_writes_enabled is True
        assert cfg.control_system_type == "mock"
        assert (
            cfg.get_execution_mode(has_epics_writes=True, has_epics_reads=False)
            is ExecutionMode.WRITE_ACCESS
        )

    def test_defaults_when_writes_key_missing(self, monkeypatch):
        monkeypatch.setattr(
            "osprey.utils.config.get_config_value",
            lambda path, default=None, config_path=None: {},
        )
        cfg = get_execution_control_config()
        assert cfg.control_system_writes_enabled is False
        # Type falls back to the EPICS default.
        assert cfg.control_system_type == control_mod.EPICS

    def test_exception_falls_back_to_safe_defaults(self, monkeypatch):
        def boom(path, default=None, config_path=None):
            raise RuntimeError("config unavailable")

        monkeypatch.setattr("osprey.utils.config.get_config_value", boom)
        cfg = get_execution_control_config()
        # On any failure the factory must return a write-disabled config.
        assert cfg.control_system_writes_enabled is False
        assert (
            cfg.get_execution_mode(has_epics_writes=True, has_epics_reads=True)
            is ExecutionMode.READ_ONLY
        )
