"""Verify the FastAPI app registers the new /api/scaffold/* routes and dropped /api/prompts/*."""

from __future__ import annotations

from fastapi import FastAPI

from osprey.interfaces.web_terminal.routes import router


def _registered_paths() -> dict[str, set[str]]:
    """Return ``{path: {METHOD, ...}}`` from the app's public OpenAPI schema.

    Enumerating the OpenAPI schema rather than ``router.routes`` keeps this test
    robust to Starlette's internal route representation: since Starlette 1.0,
    ``include_router`` stores opaque wrapper objects on the parent router instead
    of flattening the child routes, so ``router.routes`` no longer exposes a
    per-route ``.path``. The OpenAPI schema is the public, version-stable contract
    for what the app actually serves.
    """
    app = FastAPI()
    app.include_router(router)
    schema = app.openapi()
    return {
        path: {method.upper() for method in operations}
        for path, operations in schema["paths"].items()
    }


# Scaffold routes as they appear in the OpenAPI schema. Path converters such as
# ``{name:path}`` are normalized to ``{name}``, and the override path carries both
# PUT and DELETE, so the 11 route handlers collapse to 10 distinct paths.
EXPECTED_SCAFFOLD_ROUTES = {
    "/api/scaffold": {"GET"},
    "/api/scaffold/create": {"POST"},
    "/api/scaffold/untracked": {"GET"},
    "/api/scaffold/untracked/register": {"POST"},
    "/api/scaffold/untracked/{name}": {"DELETE"},
    "/api/scaffold/{name}": {"GET"},
    "/api/scaffold/{name}/claim": {"POST"},
    "/api/scaffold/{name}/diff": {"GET"},
    "/api/scaffold/{name}/framework": {"GET"},
    "/api/scaffold/{name}/override": {"PUT", "DELETE"},
}


def test_scaffold_routes_registered_no_legacy_prompts() -> None:
    paths = _registered_paths()

    scaffold = {p: m for p, m in paths.items() if p.startswith("/api/scaffold")}
    assert scaffold == EXPECTED_SCAFFOLD_ROUTES

    legacy = sorted(p for p in paths if p.startswith("/api/prompts"))
    assert legacy == [], f"unexpected legacy routes: {legacy}"


def test_claim_route_uses_claim_verb_not_scaffold() -> None:
    paths = _registered_paths()

    assert "/api/scaffold/{name}/claim" in paths
    assert "POST" in paths["/api/scaffold/{name}/claim"]
    assert "/api/scaffold/{name}/scaffold" not in paths
