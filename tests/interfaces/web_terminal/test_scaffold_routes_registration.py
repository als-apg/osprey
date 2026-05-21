"""Verify the FastAPI app registers the new /api/scaffold/* routes and dropped /api/prompts/*."""

from __future__ import annotations

from osprey.interfaces.web_terminal.routes import router


def _route_pairs() -> list[tuple[str, frozenset[str]]]:
    pairs: list[tuple[str, frozenset[str]]] = []
    for r in router.routes:
        path = getattr(r, "path", "")
        methods = frozenset(getattr(r, "methods", []) or [])
        pairs.append((path, methods))
    return pairs


def test_scaffold_routes_registered_no_legacy_prompts() -> None:
    pairs = _route_pairs()
    scaffold = [(p, m) for p, m in pairs if p.startswith("/api/scaffold")]
    legacy = [(p, m) for p, m in pairs if p.startswith("/api/prompts")]

    assert len(scaffold) == 11, f"expected 11 scaffold routes, got {sorted(scaffold)}"
    assert legacy == [], f"unexpected legacy routes: {sorted(legacy)}"


def test_claim_route_uses_claim_verb_not_scaffold() -> None:
    paths = {p for p, _ in _route_pairs()}
    assert "/api/scaffold/{name:path}/claim" in paths
    assert "/api/scaffold/{name:path}/scaffold" not in paths
