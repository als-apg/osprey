"""The lattice app's backend, as the ``lattice_panel`` recipe serves it.

The bundled lattice app is deployment source, not an importable OSPREY module,
so the recipe cannot name its ``create_app`` directly. :func:`create_app` puts
the app's source tree on the path and builds the backend over the example
facility's storage-ring deck, with every result computed before the app is
returned: the capture then shows a ready analysis however long pyAT took.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
APP_DIR = REPO_ROOT / "src/osprey/profiles/presets/apps/lattice"
DECK = REPO_ROOT / "src/osprey/templates/facilities/example/decks/SR.json"

#: The analysis the capture opens on, computed ahead of it.
ANALYSIS = "optics"


def create_app() -> object:
    """Build the lattice backend over the example storage ring, its optics ready."""
    from fastapi.testclient import TestClient

    from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER

    # The backend imports the app's package and its workers run it by module name.
    sys.path.insert(0, str(APP_DIR))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(APP_DIR), os.environ.get("PYTHONPATH")) if p
    )
    from lattice_app.config import load_settings
    from lattice_app.server import create_app as create_backend

    settings = load_settings(
        {"models": [{"name": "sr", "label": "Storage ring", "deck": str(DECK)}]}
    )
    cache = Path(tempfile.mkdtemp(prefix="osprey-lattice-shot-"))

    warm = create_backend(settings, cache=cache)
    with TestClient(warm) as client:
        client.headers[OPERATOR_SECRET_HEADER] = warm.state.web_credentials.operator_secret
        while True:
            result = client.get("/api/result", params={"analysis": ANALYSIS, "wait_s": 45}).json()
            if result["status"] == "ready":
                break
            if result["status"] == "failed":
                raise RuntimeError(f"lattice {ANALYSIS} failed: {result.get('error')}")

    return create_backend(settings, cache=cache)
