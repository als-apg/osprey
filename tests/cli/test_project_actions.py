"""Tests for the interactive-menu action handlers in ``osprey.cli.project_actions``.

Focused on :func:`handle_health_action`, which drives the menu's "Health Check"
entry. The command was rebuilt onto the ``osprey.health`` framework; the handler
must invoke the real ``osprey.cli.health_cmd.health`` Click command
programmatically (in ``--full`` mode) and translate its exit code into a menu
message — never fall through the blanket ``except Exception`` error path.

These tests never run a real health check (no network, no container runtime):
they stub the ``health`` command's ``.main`` to record its arguments and raise
``SystemExit`` with a chosen code, exactly as the real command would.
"""

from __future__ import annotations

import pytest

import osprey.cli.health_cmd as health_cmd
import osprey.cli.project_actions as project_actions


@pytest.fixture(autouse=True)
def _no_blocking_input(monkeypatch):
    """Neutralize the trailing ``input('Press ENTER...')`` so tests never block."""
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")


@pytest.fixture
def recorded_prints(monkeypatch):
    """Capture every ``console.print`` call made by the handler.

    Returns a list of the (markup) strings printed, which retain the
    ``Messages.*`` payload text (e.g. "Health check completed successfully").
    """
    calls: list[str] = []

    def _record(msg="", *args, **kwargs):
        calls.append(str(msg))

    monkeypatch.setattr(project_actions.console, "print", _record)
    return calls


def _stub_health_main(monkeypatch, exit_code):
    """Replace ``health.main`` with a recorder that raises ``SystemExit(code)``.

    Patches the ``main`` attribute on the *real* ``health`` command object, so a
    passing test proves the handler imported and invoked that object.
    """
    seen: dict[str, object] = {}

    def _fake_main(*args, **kwargs):
        seen["args"] = kwargs.get("args", args[0] if args else None)
        seen["prog_name"] = kwargs.get("prog_name")
        seen["standalone_mode"] = kwargs.get("standalone_mode")
        raise SystemExit(exit_code)

    monkeypatch.setattr(health_cmd.health, "main", _fake_main)
    return seen


def test_invokes_real_health_command_in_full_mode(monkeypatch, recorded_prints):
    """The handler calls the real ``health`` command with ``--full`` and reports success on exit 0."""
    seen = _stub_health_main(monkeypatch, exit_code=0)

    project_actions.handle_health_action()

    # The real command object was invoked (not a fully-mocked module).
    assert seen["args"] == ["--full"]
    assert seen["standalone_mode"] is False
    # Success message emitted; no blanket-exception error path was hit.
    joined = "\n".join(recorded_prints)
    assert "Health check completed successfully" in joined
    assert "✗" not in joined  # error marker from Messages.error must be absent


@pytest.mark.parametrize(
    ("exit_code", "expected_fragment"),
    [
        (0, "Health check completed successfully"),
        (1, "Health check completed with warnings"),
        (2, "Health check reported errors"),
        (3, "Health check reported errors"),
        (130, "Health check interrupted"),
    ],
)
def test_exit_code_maps_to_message(monkeypatch, recorded_prints, exit_code, expected_fragment):
    """Each health exit code maps to its menu message."""
    _stub_health_main(monkeypatch, exit_code=exit_code)

    project_actions.handle_health_action()

    joined = "\n".join(recorded_prints)
    assert expected_fragment in joined


def test_none_exit_code_treated_as_success(monkeypatch, recorded_prints):
    """A ``SystemExit`` with ``code=None`` (Click's clean exit) is treated as 0."""
    _stub_health_main(monkeypatch, exit_code=None)

    project_actions.handle_health_action()

    assert "Health check completed successfully" in "\n".join(recorded_prints)


def test_health_symbol_importable_from_health_cmd():
    """Guard: the ``health`` command object the handler imports actually exists.

    This is the regression's root cause — the handler previously imported a
    since-deleted ``HealthChecker`` symbol, which the blanket ``except`` masked.
    """
    from osprey.cli.health_cmd import health as imported_health

    assert imported_health is health_cmd.health
    assert hasattr(imported_health, "main")
