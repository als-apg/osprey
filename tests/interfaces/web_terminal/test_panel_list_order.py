"""The panel lists the server hands the browser come out in catalog order.

``/api/panels`` ``enabled``/``visible`` seed the launcher rail's order, and the
enabled built-ins are held as a set. Listing a set of strings gives an order
that depends on the process's hash seed, so every restart of the terminal
(every redeploy) used to reshuffle the rail. These tests pin the order to the
built-in catalog, and prove it holds across processes with different seeds.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from osprey.interfaces.web_terminal import app as web_terminal_app
from osprey.interfaces.web_terminal.routes import panels as panels_module
from osprey.profiles.web_panels import BUILTIN_PANEL_ORDER, BUILTIN_PANELS, SIDECAR_PANELS
from osprey.registry.web import FRAMEWORK_WEB_SERVERS

#: Companion servers in registry order, then the sidecars in registry order.
CATALOG = [
    *dict.fromkeys(d.panel_id for d in FRAMEWORK_WEB_SERVERS.values()),
    *SIDECAR_PANELS,
]


def _request(**state):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(**state)))


def _no_config():
    return patch(
        "osprey_connectors.workspace.load_osprey_config", side_effect=RuntimeError("no config")
    )


def test_catalog_order_is_every_builtin_once():
    assert list(BUILTIN_PANEL_ORDER) == CATALOG
    assert set(BUILTIN_PANEL_ORDER) == BUILTIN_PANELS


def test_api_panels_lists_enabled_and_visible_in_catalog_order():
    enabled = set(BUILTIN_PANELS)
    with _no_config():
        visible = web_terminal_app._load_panel_runtime_config(enabled, []).visible_panels
    body = asyncio.run(
        panels_module.get_panels(_request(enabled_panels=enabled, visible_panels=visible))
    )
    assert body["enabled"] == CATALOG
    assert body["visible"] == CATALOG


def test_visible_defaults_to_enabled_in_catalog_order():
    body = asyncio.run(panels_module.get_panels(_request(enabled_panels=set(BUILTIN_PANELS))))
    assert body["visible"] == CATALOG


def test_runtime_visible_puts_custom_panels_after_builtins_in_declared_order():
    enabled = set(BUILTIN_PANELS)
    custom = [{"id": "zeta"}, {"id": "alpha"}]
    with _no_config():
        visible = web_terminal_app._load_panel_runtime_config(enabled, custom).visible_panels
    assert visible == [*CATALOG, "zeta", "alpha"]


def test_rail_order_move_starts_from_catalog_order():
    """With no stored rail, a move reorders the catalog list, not a set's order."""
    broadcasts = []
    state = SimpleNamespace(
        enabled_panels=set(BUILTIN_PANELS),
        broadcaster=SimpleNamespace(broadcast=broadcasts.append),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    moved = CATALOG[-1]
    body = asyncio.run(
        panels_module.set_rail_order(
            panels_module.RailOrderRequest(panel=moved, before=CATALOG[0]), request
        )
    )
    assert body["order"] == [moved, *CATALOG[:-1]]


_PROBE = """
import asyncio, json
from types import SimpleNamespace
from unittest.mock import patch
from osprey.interfaces.web_terminal import app as a
from osprey.interfaces.web_terminal.routes import panels as p
from osprey.profiles.web_panels import BUILTIN_PANELS
enabled = set(BUILTIN_PANELS)
with patch("osprey_connectors.workspace.load_osprey_config", side_effect=RuntimeError):
    visible = a._load_panel_runtime_config(enabled, []).visible_panels
req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(enabled_panels=enabled)))
body = asyncio.run(p.get_panels(req))
print(json.dumps({"runtime_visible": visible, "enabled": body["enabled"],
                  "visible": body["visible"]}))
"""


@pytest.mark.parametrize("seed", ["1", "2", "3"])
def test_order_is_the_same_in_every_process(seed):
    """A restart is a new process with a new hash seed; the lists must not move."""
    env = {**os.environ, "PYTHONHASHSEED": seed}
    out = subprocess.run(
        [sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, check=True
    )
    lists = json.loads(out.stdout.strip().splitlines()[-1])
    assert lists == {"runtime_visible": CATALOG, "enabled": CATALOG, "visible": CATALOG}
