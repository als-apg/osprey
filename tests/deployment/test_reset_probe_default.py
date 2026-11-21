"""A ``RuntimeProbe`` built without ``run=`` runs the ``subprocess.run`` its module sees at the call.

A probe that took the function when ``osprey.deployment.reset`` was imported
would keep whatever stood there then: a fake active at that moment would answer
every later probe in the process, long after the test that installed it ended.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

# Imported at module scope so every module the copy below imports is already
# loaded: the copy is then the only module that binds the stand-in at import.
from osprey.deployment import reset
from tests._scoped_stdlib import scoped, stand_in


def _copy_of_reset_imported_under(
    monkeypatch: pytest.MonkeyPatch, run_at_import: MagicMock
) -> ModuleType:
    """A fresh copy of ``osprey.deployment.reset``, imported while ``subprocess.run`` is *run_at_import*."""
    spec = importlib.util.spec_from_file_location("_reset_imported_under_a_fake", reset.__file__)
    assert spec is not None and spec.loader is not None
    copy = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, copy)
    with monkeypatch.context() as during_import:
        during_import.setitem(sys.modules, "subprocess", stand_in(subprocess, run=run_at_import))
        spec.loader.exec_module(copy)
    return copy


def test_a_probe_runs_the_function_its_module_sees_when_it_runs(monkeypatch):
    run_at_import = MagicMock(name="run at import")
    copy = _copy_of_reset_imported_under(monkeypatch, run_at_import)
    run_at_call = MagicMock(return_value=subprocess.CompletedProcess([], 0, stdout="", stderr=""))
    scoped(monkeypatch, copy, "subprocess", run=run_at_call)

    copy.RuntimeProbe("docker")._capture(["ps"])

    run_at_import.assert_not_called()
    assert run_at_call.call_args.args[0] == ["docker", "ps"]
