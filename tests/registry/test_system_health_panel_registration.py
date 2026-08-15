"""Registration + wiring tests for the native ``system-health`` builtin web panel.

Mirrors ``tests/registry/test_okf_panel_registration.py``: prove the panel is a
builtin, its ``WebServerDefinition`` constructs with the required fields, and the
wiring is consistent across every site. Two facts are specific to this panel:

* the panel **id** ``system-health`` (hyphen) differs from the registry **key**
  ``system_health`` (underscore); the definition's own ``panel_id`` field is the
  one place that relation is stated (see
  ``tests/registry/test_web_panel_namespaces.py``);
* the id ``system-health`` was chosen to avoid the already-occupied ``health``
  id (the Bluesky scan-stack tab). The collision guard asserts no ``_inject_*``
  build step registers a ``system-health`` panel, and that the Bluesky ``health``
  entry is left untouched.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from osprey.cli import web_cmd
from osprey.infrastructure import server_launcher
from osprey.interfaces.web_terminal import app as web_terminal_app
from osprey.interfaces.web_terminal.routes import proxy as proxy_module
from osprey.profiles.web_panels import BUILTIN_PANEL_LABELS, BUILTIN_PANELS
from osprey.registry.web import (
    FRAMEWORK_WEB_SERVERS,
    PANEL_ID_TO_REGISTRY_KEY,
    WebServerDefinition,
)

PANEL_ID = "system-health"
REGISTRY_KEY = "system_health"


def _fresh_source(obj) -> str:
    """Read *obj*'s source file directly from disk (deterministic; avoids the
    ``inspect.getsource`` line-slicing flake under ``.py``/``.pyc`` skew)."""
    path = inspect.getsourcefile(obj) or inspect.getfile(obj)
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _fresh_build_package_source() -> str:
    """Concatenate every ``build_*.py`` concern module's source.

    The build pipeline is split across sibling modules (``build_cmd`` orchestrates;
    injectors, environment, lifecycle, persistence, and profile live alongside it).
    A panel-id guard must cover wherever the injectors *currently* live, so this
    scans the whole family rather than naming one module — otherwise moving an
    injector between modules silently voids the guard while everything still
    imports and passes.
    """
    from osprey.cli import build_cmd

    build_dir = Path(inspect.getsourcefile(build_cmd) or inspect.getfile(build_cmd)).parent
    sources = sorted(build_dir.glob("build_*.py"))
    assert sources, f"no build_*.py concern modules found under {build_dir}"
    return "\n".join(p.read_text(encoding="utf-8") for p in sources)


# -- builtin registration + definition -----------------------------------------


def test_system_health_is_a_builtin_panel_with_system_label():
    assert PANEL_ID in BUILTIN_PANELS
    assert BUILTIN_PANEL_LABELS[PANEL_ID] == "SYSTEM"


def test_web_server_definition_fields():
    defn = FRAMEWORK_WEB_SERVERS[REGISTRY_KEY]
    assert isinstance(defn, WebServerDefinition)
    assert defn.config_key == "health"
    assert defn.factory_path == "osprey.interfaces.health.app:create_app"
    assert defn.config_web_subkey == "web"
    assert defn.port_default == 8094
    assert defn.require_section is False  # health ships a usable default → always launchable
    assert defn.multi_user_base_port == 9791
    assert defn.factory_config_kwargs == {}  # create_app takes no config-derived kwargs


def test_port_env_override_key_is_health():
    defn = FRAMEWORK_WEB_SERVERS[REGISTRY_KEY]
    assert f"OSPREY_{defn.config_key.upper()}_PORT" == "OSPREY_HEALTH_PORT"


# -- launcher alias + proxy state ----------------------------------------------


def test_ensure_system_health_server_alias_delegates_to_registry_key(monkeypatch):
    assert hasattr(server_launcher, "ensure_system_health_server")
    keys: list[str] = []
    monkeypatch.setattr(server_launcher, "ensure_web_server", keys.append)
    server_launcher.ensure_system_health_server()
    assert keys == [REGISTRY_KEY]


def test_proxy_state_map_wires_system_health_to_server_url():
    assert proxy_module._PANEL_STATE_MAP[PANEL_ID] == "system_health_server_url"


def test_the_definition_carries_the_hyphenated_panel_id():
    # Key ≠ id: the enabled-panels gate in `osprey web` resolves the panel id
    # off the definition, so the relation is declared once.
    assert FRAMEWORK_WEB_SERVERS[REGISTRY_KEY].panel_id == PANEL_ID
    assert PANEL_ID_TO_REGISTRY_KEY[PANEL_ID] == REGISTRY_KEY
    assert "_PANEL_ID_FOR_REGISTRY_KEY" not in inspect.getsource(web_cmd)


# -- consistency across the wiring sites ---------------------------------------


def test_id_and_key_consistency_across_sites():
    # The panel id is used at the builtin/proxy/frontend sites; the registry key
    # at the definition site; the definition's `panel_id` is the single bridge.
    assert PANEL_ID in BUILTIN_PANELS
    assert PANEL_ID in proxy_module._PANEL_STATE_MAP
    assert proxy_module._PANEL_STATE_MAP[PANEL_ID] == f"{REGISTRY_KEY}_server_url"
    assert REGISTRY_KEY in FRAMEWORK_WEB_SERVERS
    assert PANEL_ID_TO_REGISTRY_KEY[PANEL_ID] == REGISTRY_KEY


# -- panels config endpoint ----------------------------------------------------


def test_panels_route_exposes_system_health_server_config_endpoint():
    from osprey.interfaces.web_terminal.routes import panels as panels_module

    paths = {getattr(r, "path", None) for r in panels_module.router.routes}
    assert "/api/system-health-server" in paths


async def test_system_health_server_config_endpoint_returns_proxy_path():
    from osprey.interfaces.web_terminal.routes import panels as panels_module

    available = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(system_health_server_url="http://127.0.0.1:8094"))
    )
    result = await panels_module.system_health_server_config(available)
    assert result["available"] is True
    assert result["url"].endswith("/panel/system-health")

    unavailable = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    result = await panels_module.system_health_server_config(unavailable)
    assert result == {"url": None, "available": False}


# -- frontend descriptor -------------------------------------------------------


def test_frontend_panel_manager_registers_system_health_tab():
    """The PANELS array lives in panel-catalog.js; panel-manager.js imports it."""
    import os

    catalog_path = os.path.join(
        os.path.dirname(inspect.getfile(web_terminal_app)), "static", "js", "panel-catalog.js"
    )
    with open(catalog_path, encoding="utf-8") as fh:
        js = fh.read()
    assert "id: 'system-health'" in js
    assert "/api/system-health-server" in js
    assert "SYSTEM" in js
    # healthEndpoint MUST be the explicit '/health': an omitted/null value skips
    # polling and pins the tab healthy (panel-manager.js:461).
    assert "healthEndpoint: '/health'" in js


def test_build_chain_reads_builtins_dynamically():
    from osprey.cli import build_profile_model
    from osprey.cli.templates import manifest

    assert "BUILTIN_PANELS" in _fresh_source(build_profile_model)
    assert "BUILTIN_PANELS" in _fresh_source(manifest)


# -- collision guard -----------------------------------------------------------


def test_no_injector_registers_a_system_health_panel_id():
    """No ``_inject_*`` build step may register the ``system-health`` id.

    Builtins shadow custom panels in ``_load_panel_config`` / ``_resolve_panel_url``,
    so an injector adding a ``system-health`` custom panel would silently hijack
    the builtin tab.
    """
    source = _fresh_build_package_source()
    assert "system-health" not in source  # no injector (or literal) registers it


# -- launch chain through the real registry factory ----------------------------


def test_launch_chain_through_real_registry_factory_serves_health():
    """The registry factory builds an app whose /health answers, even unconfigured.

    Guards the silent-dead-tab failure mode: the launcher runs the factory on a
    swallowed daemon thread, so create_app must never raise and /health must be
    200 even with no config.yml resolvable.
    """
    factory = server_launcher._make_app_factory(FRAMEWORK_WEB_SERVERS[REGISTRY_KEY])
    app = factory()
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "system-health"
