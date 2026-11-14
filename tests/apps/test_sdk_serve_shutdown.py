"""``serve_tools(register, on_shutdown=...)``: an application's hook for when its server stops.

The hook runs exactly once after the server has stopped, whether serving returned
or raised, so an application can release a session it holds. The transport loop
is left out: ``run_mcp_server`` is replaced by a stand-in that returns or raises.
"""

from __future__ import annotations

import inspect

import pytest

from osprey.apps import sdk
from osprey.mcp_server import startup


def _register(_server) -> None:  # pragma: no cover - never called by the stand-ins
    raise AssertionError("the stand-in server registers nothing")


@pytest.fixture
def stops(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The order of events: the server's stop, then every hook call."""
    events: list[str] = []
    monkeypatch.setattr(startup, "run_mcp_server", lambda module: events.append("stopped"))
    return events


def test_the_hook_runs_once_after_the_server_returns(stops):
    sdk.serve_tools(_register, on_shutdown=lambda: stops.append("hook"))
    assert stops == ["stopped", "hook"]


def test_the_hook_runs_once_when_the_server_raises(monkeypatch):
    events: list[str] = []

    def failing(_module: str) -> None:
        events.append("stopped")
        raise RuntimeError("transport lost")

    monkeypatch.setattr(startup, "run_mcp_server", failing)
    with pytest.raises(RuntimeError, match="transport lost"):
        sdk.serve_tools(_register, on_shutdown=lambda: events.append("hook"))
    assert events == ["stopped", "hook"]


def test_without_a_hook_serving_is_unchanged(stops):
    sdk.serve_tools(_register)
    assert stops == ["stopped"]


def test_register_stays_the_one_positional_argument():
    params = inspect.signature(sdk.serve_tools).parameters
    assert [p.kind for p in params.values()] == [
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ]
    assert params["on_shutdown"].default is None
