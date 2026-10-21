"""The dashboard's HTTP contract is pinned as an OpenAPI document.

``openapi.json`` beside this file is ``create_app(...).openapi()`` as sorted-key
JSON. A route, a response model or a field that changes shows up as a named
difference here, so the page, the agent tools and the server move together.

Regenerating: ``LATTICE_OPENAPI_REGEN=1`` rewrites the snapshot instead of
comparing; the rewrite is reviewed and committed with the change it records.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from osprey.interfaces.lattice_dashboard.app import create_app

SNAPSHOT = Path(__file__).with_name("openapi.json")


def _document(tmp_path: Path) -> dict[str, Any]:
    app = create_app(workspace_root=tmp_path / "ws", render_root=tmp_path / "render")
    return json.loads(json.dumps(app.openapi(), sort_keys=True))


def _render(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def differences(expected: Any, actual: Any, path: str = "") -> list[str]:
    """Return one line per path where *actual* departs from *expected*."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        lines: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            where = f"{path}/{key}"
            if key not in actual:
                lines.append(f"{where}: missing")
            elif key not in expected:
                lines.append(f"{where}: added")
            else:
                lines.extend(differences(expected[key], actual[key], where))
        return lines
    if expected != actual:
        return [f"{path or '/'}: {expected!r} -> {actual!r}"]
    return []


def test_openapi_matches_the_snapshot(tmp_path):
    document = _document(tmp_path)
    if os.environ.get("LATTICE_OPENAPI_REGEN") == "1":
        SNAPSHOT.write_text(_render(document))
    expected = json.loads(SNAPSHOT.read_text())

    assert differences(expected, document) == []


def test_a_dropped_field_is_named(tmp_path):
    document = _document(tmp_path)
    mutated = copy.deepcopy(document)
    del mutated["components"]["schemas"]["FigureNotCurrent"]["properties"]["key"]

    lines = differences(document, mutated)

    assert lines == ["/components/schemas/FigureNotCurrent/properties/key: missing"]
