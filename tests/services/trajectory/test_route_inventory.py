"""The trajectory service's HTTP/SSE surface, pinned route by route.

The panel builds on these routes, the agent's trajectory tools call the agent
door's, and the terminal's panel tier lists that door's: a route added,
dropped, renamed or moved between doors fails here first. Each door is held
to an exact set of ``(method, path)`` pairs read from the app's OpenAPI
document, so the inventory is the app's own, not a second hand-written list
that could agree with itself.

The surface is the one the threading loop hands the panel: setup, shot
(``n_shots``, ``average``), ``GET step``, apply (a step or a put-back), the
continuous run (start and heartbeat), Stop, the references (save, load,
select), the shots listing and the state stream — beside the session routes
both doors serve and the gate-exempt ``GET /health``.
"""

from __future__ import annotations

from osprey.services.trajectory.app import create_app

#: What both doors serve.
_BOTH_DOORS = {
    ("POST", "session"),
    ("POST", "takeover"),
    ("DELETE", "session"),
    ("GET", "state"),
    ("POST", "stop"),
    ("POST", "shot"),
    ("GET", "step"),
    ("POST", "apply"),
}

#: What only the panel door serves.
_PANEL_ONLY = {
    ("POST", "setup"),
    ("POST", "continuous"),
    ("POST", "continuous/heartbeat"),
    ("GET", "references"),
    ("POST", "references"),
    ("GET", "references/{name}"),
    ("POST", "references/select"),
    ("GET", "shots"),
    ("GET", "events"),
}


def _routes() -> set[tuple[str, str]]:
    paths = create_app().openapi()["paths"]
    return {(method.upper(), path) for path, operations in paths.items() for method in operations}


def _door(prefix: str, routes: set[tuple[str, str]]) -> set[tuple[str, str]]:
    return {(method, f"{prefix}/{name}") for method, name in routes}


def test_the_agent_door_serves_exactly_the_session_stop_shot_step_and_apply() -> None:
    agent = {(method, path) for method, path in _routes() if path.startswith("/api/")}
    assert agent == _door("/api", _BOTH_DOORS)


def test_the_panel_door_serves_the_agent_doors_routes_and_its_own() -> None:
    panel = {(method, path) for method, path in _routes() if path.startswith("/panel/")}
    assert panel == _door("/panel/api", _BOTH_DOORS | _PANEL_ONLY)


def test_nothing_else_is_served_but_health() -> None:
    rest = {
        (method, path) for method, path in _routes() if not path.startswith(("/api/", "/panel/"))
    }
    assert rest == {("GET", "/health")}


def test_stop_answers_202() -> None:
    paths = create_app().openapi()["paths"]
    for door in ("/api", "/panel/api"):
        assert "202" in paths[f"{door}/stop"]["post"]["responses"]
