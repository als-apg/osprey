"""Shared fixtures for the ``osprey`` CLI test suite.

CLI tests drive Click commands in-process through ``CliRunner``, so whatever a
command does to process-global state happens inside the pytest process itself.
The guards here keep that blast radius contained.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# The gold-standard three-zone deployment repo, re-exported so every lifecycle
# test under tests/cli/ can ask for it by name. Defined in tests/fixtures/ with
# the exemplar content it materializes.
from tests.fixtures.lifecycle_repo import (  # noqa: F401
    lifecycle_repo,
    lifecycle_repo_factory,
)


@pytest.fixture(autouse=True)
def _guard_os_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn an in-process ``os._exit`` into an ordinary test failure.

    Leak guarded: ``osprey health`` falls back to ``os._exit`` when a sync check
    was abandoned after a timeout (``src/osprey/cli/health_cmd.py:320-326``),
    because a normal ``sys.exit`` can wedge on interpreter teardown with a
    daemon thread still running. Under an in-process ``CliRunner`` invocation
    that call terminates pytest itself; under xdist it kills the whole worker
    and every test still queued on it, surfacing as a crashed node rather than
    a failing test. Raising instead keeps the failure attributable.

    The real daemon-thread/``os._exit`` guarantee is pinned by the
    real-subprocess no-hang test in ``test_health_cmd.py``, which is unaffected:
    this patch lives in the parent process, not the child's.
    """

    def _raise(code: int) -> None:
        raise AssertionError(f"os._exit({code}) called in-process")

    monkeypatch.setattr(os, "_exit", _raise)


@pytest.fixture(autouse=True)
def _neutral_color_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip color-forcing variables from the environment CLI tests run under.

    Leak guarded: Rich and Click honor ``FORCE_COLOR``/``CLICOLOR_FORCE`` as a
    documented opt-in, so a developer whose terminal exports either gets ANSI
    escapes inside ``CliRunner``'s captured output — and a false-red lane on
    pristine main (seven tests: exact-output asserts, plus two whose captured
    YAML stops parsing on ``\\x1b``). CI is green only because its runners
    never export them; this makes the suite hermetic instead of lucky.
    ``NO_COLOR`` is left alone: unset it is the same default CI runs under,
    and pinning it would hide a command that fails to honor the opt-out.
    """
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("CLICOLOR_FORCE", raising=False)


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``Path.home()`` and ``$HOME`` to a tmp directory.

    Leak guarded: CLI code that writes under the home directory (skills install,
    template scaffolding, hook registration) would otherwise touch the
    developer's real ``~``. Patching the method on ``Path`` is required because
    callers use ``Path.home()`` directly; ``$HOME`` is patched alongside it for
    subprocess-style code paths that read the environment instead.

    Returns:
        The tmp directory standing in for ``$HOME``.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: home)
    return home
