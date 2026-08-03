"""Unit tests for :mod:`osprey.services.python_executor.config`.

The Python executor is part of the sandboxed-execution safety surface, so the
resource-limit default (execution timeout) and the config-override
path are asserted strictly. The ``limits_validator`` property's lazy-load and
caching behaviour is covered with the underlying validator monkeypatched.
"""

from __future__ import annotations

from osprey.services.python_executor.config import PythonExecutorConfig


class TestDefaults:
    def test_defaults_when_no_config(self):
        cfg = PythonExecutorConfig()
        # 10-minute execution ceiling for the sandbox.
        assert cfg.execution_timeout_seconds == 600

    def test_none_config_uses_defaults(self):
        cfg = PythonExecutorConfig(None)
        assert cfg.execution_timeout_seconds == 600

    def test_empty_python_executor_block_uses_defaults(self):
        cfg = PythonExecutorConfig({"unrelated": {"foo": 1}})
        assert cfg.execution_timeout_seconds == 600


class TestOverrides:
    def test_override_replaces_default(self):
        cfg = PythonExecutorConfig({"python_executor": {"execution_timeout_seconds": 30}})
        assert cfg.execution_timeout_seconds == 30


class TestLimitsValidator:
    def test_lazy_load_and_cache(self, monkeypatch):
        from osprey.connectors.control_system import limits_validator as lv_mod

        calls = {"count": 0}
        sentinel = object()

        def _from_config():
            calls["count"] += 1
            return sentinel

        monkeypatch.setattr(lv_mod.LimitsValidator, "from_config", staticmethod(_from_config))

        cfg = PythonExecutorConfig()
        assert cfg._limits_validator is None
        first = cfg.limits_validator
        second = cfg.limits_validator
        assert first is sentinel
        assert second is sentinel
        # Cached after the first access -- from_config runs exactly once.
        assert calls["count"] == 1

    def test_disabled_validator_returns_none(self, monkeypatch):
        from osprey.connectors.control_system import limits_validator as lv_mod

        monkeypatch.setattr(lv_mod.LimitsValidator, "from_config", staticmethod(lambda: None))
        cfg = PythonExecutorConfig()
        assert cfg.limits_validator is None
