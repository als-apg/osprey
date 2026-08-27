"""Route-table completeness for the two server-side tier gates.

``web.config_panel.enabled: false`` and ``web.scaffold_gallery.write_enabled:
false`` are enforced by hand-wired call sites — ``_require_config_panel`` in six
handlers, ``_require_scaffold_writes`` in six more. The suites that pin them
(``test_config_panel_gate.py``, ``test_scaffold_tier_gate.py``) drive
hand-maintained literal tables of requests, which is the right shape for
asserting *how* each refusal behaves but cannot answer the question a reviewer
actually has: is the table still the whole surface?

It already has not been, once. ``POST /api/scaffold/untracked/register`` was not
among the routes the plan named, and was added to the gate and to the table by
hand afterwards. Nothing failed in between. A seventh write route can ship today
with both suites green, because a route nobody listed is a route nobody tested.

So this module does not carry a list. It walks ``app.openapi()["paths"]`` — the
same public, version-stable enumeration
``test_scaffold_routes_registration.py`` uses, and the only view of the app that
grows automatically when a handler is added — and requires that:

* every mutating verb under ``/api/scaffold`` refuses with 403 when the gallery
  is disabled (read verbs are deliberately open: seeing what the agent runs is
  not authoring it), and
* every verb under ``/api/config``, ``/api/claude-setup`` and ``/api/hooks``
  refuses with 403 when the Config panel is disabled — except the routes named
  in :data:`DELIBERATELY_UNGATED`, which must be *exactly* the ones that do not.

That last clause is the part that makes this a gate test rather than a headcount.
Adding a route without gating it fails here; deciding a route should stay
ungated is possible, but only by writing the decision and its reason down.

The requests are synthesized from each operation's own OpenAPI schema, so a new
route needs no fixture here — and a body that is schema-valid keeps a 422 from
masquerading as a passed gate. The gate runs first in every handler, so a valid
body is never actually read.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import register_scaffold_conflict_handlers
from osprey.interfaces.web_terminal.routes import router
from osprey.interfaces.web_terminal.routes.agent_activity import ACTIVITY_RING_MAX

#: Verbs that author. Everything else under ``/api/scaffold`` is a read, which
#: the gallery gate deliberately leaves open.
MUTATING_VERBS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

#: Prefixes whose *mutating* verbs must be gated. The gallery's reads stay open.
MUTATING_ONLY_PREFIXES = ("/api/scaffold",)

#: Prefixes where EVERY verb must be gated, reads included. The Config panel's
#: reads expose ``config.yml`` and the ``.claude/`` tree — the agent's rendered
#: permission surface — so a tier that may not use the panel may not read
#: through it either. ``/api/hooks`` is walked with them because it ships from
#: the same module and belongs to the same panel; see DELIBERATELY_UNGATED.
#: ``/api/audit`` is walked here for the same reason and is reads-only today:
#: the ledger names every safety decision the deployment made, its subjects and
#: its sessions, so it sits behind the same tier the panel does.
ALL_VERB_PREFIXES = ("/api/config", "/api/claude-setup", "/api/hooks", "/api/audit")

#: Routes inside the walked surface that are intentionally NOT gated, each with
#: the reason it is safe for them to answer a disabled tier. Asserted to be
#: exactly the set of non-403 routes: an ungated route that is not written here
#: fails, and so does an entry here that has since been gated.
DELIBERATELY_UNGATED: dict[tuple[str, str], str] = {
    ("GET", "/api/hooks/debug-status"): (
        "reports whether hook debug logging is on; a boolean about the "
        "deployment's own diagnostics, authoring nothing and naming no path"
    ),
    ("GET", "/api/hooks/debug-log"): (
        "serves hook_debug.jsonl entries — diagnostic output the deployment "
        "wrote about itself; reading it changes no agent-facing instruction"
    ),
}

#: Floor on how many routes the walk must find, so an empty or truncated
#: OpenAPI document cannot green this file by finding nothing to check.
#: Thirteen gated routes today: six scaffold write verbs, three on
#: ``/api/config``, three on ``/api/claude-setup``, and ``GET
#: /api/audit/recent``. BUMP THIS DELIBERATELY when a route is added, as a
#: second, human acknowledgement that the surface grew.
MIN_GATED_ROUTES = 13

#: Stand-in for any ``{param}`` in a path. The gate runs ahead of the handler
#: body, so nothing ever resolves it.
_DUMMY_PARAM = "x"

_PARAM = re.compile(r"\{[^}]+\}")

BASE_CONFIG = {
    "project_name": "tier-gate-completeness",
    "control_system": {"writes_enabled": False},
    "claude_code": {"default_model": "sonnet"},
}


# ---- app + walk ---- #


@pytest.fixture
def project_dir(tmp_path):
    """A project directory carrying a parseable ``config.yml``."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "config.yml").write_text(yaml.safe_dump(BASE_CONFIG, sort_keys=False))
    return project


def _app(project_dir) -> FastAPI:
    """The full route surface with BOTH tiers disabled.

    One app, not two: the walk has to see every route the web terminal serves,
    and the two gates are independent, so a single disabled-everything app
    answers both questions at once.
    """
    app = FastAPI()
    app.include_router(router)
    register_scaffold_conflict_handlers(app)
    app.state.config_path = project_dir / "config.yml"
    app.state.project_cwd = str(project_dir)
    app.state.agent_activity_ring = deque(maxlen=ACTIVITY_RING_MAX)
    app.state.config_panel_enabled = False
    app.state.scaffold_write_enabled = False
    return app


@pytest.fixture
def disabled_client(project_dir):
    """Client for the both-tiers-disabled app.

    ``raise_server_exceptions=False`` so an *ungated* route that then blows up
    on the junk request is reported as its status code rather than exploding
    the walk. A crash is not a refusal, and this file's job is to say so.
    """
    with TestClient(_app(project_dir), raise_server_exceptions=False) as client:
        yield client


def _resolve(ref: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve a local ``#/components/schemas/Name`` pointer."""
    node: Any = schema
    for part in ref.lstrip("#/").split("/"):
        node = node[part]
    return node


def _dummy(node: dict[str, Any], schema: dict[str, Any], depth: int = 0) -> Any:
    """A minimal schema-valid value for *node*.

    Enough to clear FastAPI's request validation so the response reflects the
    gate rather than the body. Depth-capped because a recursive schema would
    otherwise recur forever.
    """
    if depth > 4:
        return None
    if "$ref" in node:
        return _dummy(_resolve(node["$ref"], schema), schema, depth + 1)
    for combinator in ("anyOf", "oneOf", "allOf"):
        options = [opt for opt in node.get(combinator, []) if opt.get("type") != "null"]
        if options:
            return _dummy(options[0], schema, depth + 1)
    kind = node.get("type", "object")
    if kind == "object":
        properties = node.get("properties", {})
        return {
            name: _dummy(sub, schema, depth + 1)
            for name, sub in properties.items()
            if name in node.get("required", properties)
        }
    return {
        "string": "x",
        "integer": 1,
        "number": 1.0,
        "boolean": False,
        "array": [],
        "null": None,
    }.get(kind, "x")


def _body(operation: dict[str, Any], schema: dict[str, Any]) -> Any | None:
    """The JSON body to send for *operation*, or ``None`` when it takes none."""
    content = operation.get("requestBody", {}).get("content", {})
    json_schema = content.get("application/json", {}).get("schema")
    return None if json_schema is None else _dummy(json_schema, schema)


def _walk(client: TestClient) -> list[tuple[str, str, Any]]:
    """Every route this file governs, as ``(METHOD, path, body)``.

    Derived from the served OpenAPI document, never from a literal list — that
    is the entire point of the module.
    """
    schema = client.app.openapi()
    found: list[tuple[str, str, Any]] = []
    for path, operations in sorted(schema["paths"].items()):
        for verb, operation in operations.items():
            method = verb.upper()
            if path.startswith(ALL_VERB_PREFIXES) or (
                path.startswith(MUTATING_ONLY_PREFIXES) and method in MUTATING_VERBS
            ):
                found.append((method, path, _body(operation, schema)))
    return found


def _statuses(client: TestClient) -> dict[tuple[str, str], int]:
    """Status code each walked route answers a disabled tier with."""
    results: dict[tuple[str, str], int] = {}
    for method, path, body in _walk(client):
        url = _PARAM.sub(_DUMMY_PARAM, path)
        response = client.request(method, url, json=body)
        results[method, path] = response.status_code
    return results


# ---- the assertions ---- #


class TestWalkIsLoadBearing:
    """Guards against the walk itself going quiet."""

    def test_walk_finds_at_least_the_known_route_floor(self, disabled_client):
        gated = [route for route in _statuses(disabled_client) if route not in DELIBERATELY_UNGATED]
        assert len(gated) >= MIN_GATED_ROUTES, (
            f"walk found only {len(gated)} gated routes, expected at least "
            f"{MIN_GATED_ROUTES}: {sorted(gated)}"
        )

    def test_every_route_in_the_allowlist_still_exists(self, disabled_client):
        walked = {(method, path) for method, path, _ in _walk(disabled_client)}
        stale = sorted(set(DELIBERATELY_UNGATED) - walked)
        assert stale == [], f"DELIBERATELY_UNGATED names routes that no longer exist: {stale}"

    def test_allowlist_entries_carry_a_reason(self):
        blank = sorted(route for route, why in DELIBERATELY_UNGATED.items() if not why.strip())
        assert blank == [], f"allowlisted routes with no reason: {blank}"


class TestEveryWalkedRouteIsGated:
    """The completeness claim itself."""

    def test_gated_routes_all_refuse_with_403(self, disabled_client):
        statuses = _statuses(disabled_client)
        open_routes = {
            route: status
            for route, status in statuses.items()
            if status != 403 and route not in DELIBERATELY_UNGATED
        }
        assert open_routes == {}, (
            "routes answered a disabled tier without a 403 — either gate them, or "
            f"add them to DELIBERATELY_UNGATED with a reason: {open_routes}"
        )

    def test_ungated_routes_are_exactly_the_allowlist(self, disabled_client):
        statuses = _statuses(disabled_client)
        ungated = {route for route, status in statuses.items() if status != 403}
        assert ungated == set(DELIBERATELY_UNGATED), (
            "the set of routes outside the gate has drifted from the recorded "
            f"decision: found {sorted(ungated)}, recorded {sorted(DELIBERATELY_UNGATED)}"
        )


class TestWalkCatchesADroppedCallSite:
    """The mutation twins.

    A completeness test that cannot fail is worth nothing, and the failure it
    has to catch is precise: ONE handler losing its gate call while every other
    handler keeps it. Both twins simulate exactly that, by swapping the gate for
    one that refuses everything except a single chosen method-and-path.
    """

    @staticmethod
    def _gate_except(method: str, path: str, detail: str):
        """A gate that skips its refusal for one route and refuses all others."""

        def gate(request: Request) -> None:
            if request.method == method and request.url.path == path:
                return
            raise HTTPException(status_code=403, detail=detail)

        return gate

    def test_a_dropped_config_panel_call_site_is_reported(self, disabled_client):
        target = ("PATCH", "/api/config")
        with patch(
            "osprey.interfaces.web_terminal.routes.config._require_config_panel",
            self._gate_except(*target, "panel disabled"),
        ):
            statuses = _statuses(disabled_client)

        ungated = {route for route, status in statuses.items() if status != 403}
        assert target in ungated, "walk failed to notice an ungated /api/config route"
        assert ungated == set(DELIBERATELY_UNGATED) | {target}

    def test_a_dropped_scaffold_call_site_is_reported(self, disabled_client):
        target = ("POST", "/api/scaffold/untracked/register")
        with patch(
            "osprey.interfaces.web_terminal.routes.scaffold._require_scaffold_writes",
            self._gate_except(target[0], target[1], "gallery writes disabled"),
        ):
            statuses = _statuses(disabled_client)

        ungated = {route for route, status in statuses.items() if status != 403}
        assert target in ungated, "walk failed to notice an ungated /api/scaffold write route"
        assert ungated == set(DELIBERATELY_UNGATED) | {target}
