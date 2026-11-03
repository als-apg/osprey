"""Executing the Sphinx ``conf.py`` the way a docs build does, with a pinned version."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

import osprey.version

CONF_PY = Path(__file__).resolve().parents[2] / "docs" / "source" / "conf.py"


def run_conf(
    monkeypatch: pytest.MonkeyPatch, *, released: bool = True, running: str = "9.9.9"
) -> dict[str, Any]:
    """Execute ``conf.py`` with a pinned version source and return its namespace.

    ``conf.py`` prepends its extension directories to ``sys.path``; the test
    gets a copy of the path to mutate, so the entries leave with it.
    """
    monkeypatch.setattr(osprey.version, "is_release", lambda: released)
    monkeypatch.setattr(osprey.version, "get_running_version", lambda: running)
    monkeypatch.chdir(CONF_PY.parent)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return runpy.run_path(str(CONF_PY))
