"""App-side wiring tests for the native ``system-health`` builtin web panel.

These guard the app-side half of that wiring:

- the proxy state-attr map, the ``/api/system-health-server`` config endpoint,
  and the panel-catalog.js descriptor (with an EXPLICIT ``healthEndpoint`` so
  the sidecar is actually polled) are all present;
- the control-assistant preset lists the tab without disturbing the separate
  Bluesky ``health`` panel.

Launch behaviour — auto-launch gating, the published URL agreeing with the port
``ServerLauncher`` binds, and ``require_section`` — is no longer panel-specific:
one launcher serves all six, and ``tests/interfaces/test_panel_launch_gating.py``
pins it parametrized over the whole registry.

The registry-level registration invariants (builtin set ↔ registry key ↔
state-attr map three-way consistency, env-var derivation) live in
``tests/registry/test_system_health_panel_registration.py``.
"""

from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

import yaml

from osprey.interfaces.web_terminal import app as web_terminal_app
from osprey.interfaces.web_terminal.routes import proxy as proxy_module


def test_proxy_state_map_wires_system_health():
    assert proxy_module._PANEL_STATE_MAP["system-health"] == "system_health_server_url"


def test_panels_route_exposes_system_health_server_endpoint():
    """The frontend's configEndpoint /api/system-health-server must exist."""
    from osprey.interfaces.web_terminal.routes import panels as panels_module

    paths = {getattr(r, "path", None) for r in panels_module.router.routes}
    assert "/api/system-health-server" in paths


async def test_system_health_server_config_endpoint_returns_proxy_path():
    """The endpoint returns the /panel/system-health proxy path when available."""
    from osprey.interfaces.web_terminal.routes import panels as panels_module

    available = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(system_health_server_url="http://127.0.0.1:8094"))
    )
    result = await panels_module.system_health_server_config(available)
    assert result == {"url": "/panel/system-health", "available": True}

    unavailable = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    result = await panels_module.system_health_server_config(unavailable)
    assert result == {"url": None, "available": False}


def test_panel_catalog_registers_system_health_tab_with_explicit_health_endpoint():
    """The shipped panel catalog must register SYSTEM with an explicit health poll.

    An omitted/null ``healthEndpoint`` SKIPS polling and pins the panel healthy,
    which would leave the SYSTEM rail entry permanently enabled even with its
    sidecar down — so the descriptor MUST carry ``healthEndpoint: '/health'``.
    (The rail draws no per-entry LED; the poll feeds only the coarse
    ``.disabled`` state, and the detailed readout is the ``web_panels``
    health category. The explicit endpoint matters either way.)

    Reads ``panel-catalog.js``, which owns the shipped ``PANELS`` array —
    ``panel-manager.js`` imports it and holds only the state machine.
    """
    pm_path = os.path.join(
        os.path.dirname(inspect.getfile(web_terminal_app)),
        "static",
        "js",
        "panel-catalog.js",
    )
    with open(pm_path, encoding="utf-8") as fh:
        js = fh.read()
    assert "id: 'system-health'" in js
    assert "/api/system-health-server" in js
    assert "'SYSTEM'" in js
    assert "healthEndpoint: '/health'" in js


def test_control_assistant_preset_lists_system_health():
    """The preset lists the SYSTEM tab among its web panels."""
    preset_path = os.path.join(
        os.path.dirname(inspect.getfile(web_terminal_app)),
        "..",
        "..",
        "profiles",
        "presets",
        "control-assistant.yml",
    )
    with open(os.path.abspath(preset_path), encoding="utf-8") as fh:
        preset = yaml.safe_load(fh)
    web_panels = preset["web_panels"]
    assert "system-health" in web_panels
