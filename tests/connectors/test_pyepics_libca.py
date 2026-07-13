"""Tests for the connector's pyepics libca resolution helper.

pyepics's bundled ``clibs`` are x86_64-only (its ``find_libca()`` picks
``clibs/linux64`` for any 64-bit OS), so on an arm64 host it loads a
mismatched-architecture ``libca.so`` and CA init fails. The connector points
``PYEPICS_LIBCA`` at ``epicscorelibs``' correct per-architecture libca before
importing pyepics. These tests exercise that resolution helper directly (no CA,
no container).
"""

from __future__ import annotations

import sys

import pytest

from osprey.connectors.control_system.epics_connector import _configure_pyepics_libca


def test_sets_pyepics_libca_from_epicscorelibs_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    from epicscorelibs.path import get_lib

    monkeypatch.delenv("PYEPICS_LIBCA", raising=False)
    _configure_pyepics_libca()
    import os

    assert os.environ.get("PYEPICS_LIBCA") == get_lib("ca")


def test_respects_existing_pyepics_libca_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator-provided PYEPICS_LIBCA must win — the helper never clobbers it."""
    monkeypatch.setenv("PYEPICS_LIBCA", "/operator/chosen/libca.so")
    _configure_pyepics_libca()
    import os

    assert os.environ["PYEPICS_LIBCA"] == "/operator/chosen/libca.so"


def test_noop_when_epicscorelibs_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """With epicscorelibs unimportable, the helper leaves PYEPICS_LIBCA unset and
    does not raise — pyepics falls back to its own libca resolution."""
    monkeypatch.delenv("PYEPICS_LIBCA", raising=False)
    # Force `from epicscorelibs.path import get_lib` to fail.
    monkeypatch.setitem(sys.modules, "epicscorelibs.path", None)
    _configure_pyepics_libca()
    import os

    assert "PYEPICS_LIBCA" not in os.environ
