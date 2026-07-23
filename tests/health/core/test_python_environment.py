"""Tests for the core ``python_environment`` health category (task 4.3).

Patches interpreter attributes (``sys.version_info``, venv markers) and the
dependency-import probe (``importlib.util.find_spec``) to exercise each row's
ok/warning/error branch without altering the real environment.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import namedtuple

from osprey.health.core.python_environment import (
    CORE_DEPENDENCIES,
    python_environment,
)
from osprey.health.models import CheckResult, Status

_VersionInfo = namedtuple("_VersionInfo", "major minor micro releaselevel serial")


def _run(config=None, context=None) -> dict[str, CheckResult]:
    """Invoke the category and index its results by name."""
    results = python_environment(config, context)()
    assert isinstance(results, list)
    return {r.name: r for r in results}


def test_returns_three_named_rows() -> None:
    by_name = _run()
    assert set(by_name) == {"python_version", "virtual_environment", "core_dependencies"}
    assert all(r.category == "python_environment" for r in by_name.values())


def test_factory_ignores_config_and_context() -> None:
    # Passing arbitrary config/context must not change the interpreter checks.
    by_name = _run(config={"anything": 1}, context=object())
    assert set(by_name) == {"python_version", "virtual_environment", "core_dependencies"}


def test_python_version_ok_on_supported(monkeypatch) -> None:
    monkeypatch.setattr(sys, "version_info", _VersionInfo(3, 12, 4, "final", 0))
    row = _run()["python_version"]
    assert row.status is Status.OK
    assert "3.12.4" in row.message


def test_python_version_warns_below_311(monkeypatch) -> None:
    monkeypatch.setattr(sys, "version_info", _VersionInfo(3, 10, 9, "final", 0))
    row = _run()["python_version"]
    assert row.status is Status.WARNING
    assert "3.10.9" in row.message
    assert "3.11+" in row.message


def test_virtual_environment_active_via_base_prefix(monkeypatch) -> None:
    monkeypatch.delattr(sys, "real_prefix", raising=False)
    monkeypatch.setattr(sys, "base_prefix", "/usr/local")
    monkeypatch.setattr(sys, "prefix", "/venv")
    row = _run()["virtual_environment"]
    assert row.status is Status.OK
    assert "active" in row.message


def test_virtual_environment_active_via_real_prefix(monkeypatch) -> None:
    # Legacy virtualenv marker: real_prefix present, base_prefix == prefix.
    monkeypatch.setattr(sys, "real_prefix", "/usr/local", raising=False)
    monkeypatch.setattr(sys, "base_prefix", "/venv")
    monkeypatch.setattr(sys, "prefix", "/venv")
    row = _run()["virtual_environment"]
    assert row.status is Status.OK


def test_virtual_environment_warns_when_absent(monkeypatch) -> None:
    monkeypatch.delattr(sys, "real_prefix", raising=False)
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    monkeypatch.setattr(sys, "prefix", "/usr")
    row = _run()["virtual_environment"]
    assert row.status is Status.WARNING
    assert "Not in a virtual environment" in row.message


def test_core_dependencies_ok_when_all_present() -> None:
    row = _run()["core_dependencies"]
    assert row.status is Status.OK
    assert str(len(CORE_DEPENDENCIES)) in row.message


def test_core_dependencies_error_lists_missing(monkeypatch) -> None:
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if name in {"jinja2", "litellm"}:
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    row = _run()["core_dependencies"]
    assert row.status is Status.ERROR
    assert "jinja2" in row.message
    assert "litellm" in row.message
    assert "click" not in row.message
