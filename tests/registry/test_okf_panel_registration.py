"""Registration + wiring tests for the native ``okf`` builtin web panel.

Mirrors ``tests/registry/test_facility_knowledge_registration.py`` in intent:
prove the panel is registered as a builtin, that its ``WebServerDefinition``
constructs with the required ``config_key``, and — the DA IA-2 invariant — that
the bare string ``okf`` is used consistently across the four wiring sites
(builtin set ↔ registry key ↔ proxy state-attr map ↔ web-terminal launch gate).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from osprey.infrastructure import server_launcher
from osprey.interfaces.web_terminal import app as web_terminal_app
from osprey.interfaces.web_terminal.routes import proxy as proxy_module
from osprey.port_layout import DEFAULT_PORT_BASE, SLOTS_BY_NAME
from osprey.profiles.web_panels import BUILTIN_PANEL_LABELS, BUILTIN_PANELS
from osprey.registry.web import (
    FRAMEWORK_WEB_SERVERS,
    WebServerDefinition,
    framework_web_port_default,
)


def _fresh_source(obj) -> str:
    """Read *obj*'s source file directly from disk.

    Prefer this over ``inspect.getsource`` for architectural "this wiring line
    exists" guards. ``inspect.getsource`` slices lines using the code object's
    ``co_firstlineno`` against a ``linecache`` copy of the file; if bytecode and
    on-disk source drift (a transient ``.py``/``.pyc`` skew, which we have hit),
    it silently returns an *adjacent* definition — a flake that can false-fail or,
    worse, false-pass. Reading the whole file fresh is deterministic. For a
    module this is the entire source; behavioural assertions are used instead
    wherever a function's actual effect can be exercised directly.
    """
    path = inspect.getsourcefile(obj) or inspect.getfile(obj)
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def test_okf_is_a_builtin_panel_with_knowledge_label():
    assert "okf" in BUILTIN_PANELS
    assert BUILTIN_PANEL_LABELS["okf"] == "KNOWLEDGE"


def test_okf_web_server_definition_constructs_with_required_config_key():
    defn = FRAMEWORK_WEB_SERVERS["okf"]
    assert isinstance(defn, WebServerDefinition)
    # config_key has no default — omitting it would be a TypeError at import (DA CF-3).
    assert defn.config_key == "facility_knowledge"
    assert defn.factory_path == "osprey.interfaces.okf_panel.app:create_app"
    assert defn.require_section is True
    assert defn.factory_config_kwargs == {"bundle_path": "facility_knowledge.bundle_path"}


def test_okf_port_default_is_its_layout_slot_at_the_resolved_base():
    # The definition carries no port. The default is the ``okf`` layout slot,
    # taken at whatever base the deployment resolved — so it moves with the base
    # instead of pinning the panel to the layout's own default.
    offset = SLOTS_BY_NAME["okf"].offset
    assert framework_web_port_default("okf") == DEFAULT_PORT_BASE + offset
    assert framework_web_port_default("okf", base=20000) == 20000 + offset


def test_port_env_override_key_is_facility_knowledge():
    # The launcher derives OSPREY_{CONFIG_KEY}_PORT; assert the resulting key.
    defn = FRAMEWORK_WEB_SERVERS["okf"]
    assert f"OSPREY_{defn.config_key.upper()}_PORT" == "OSPREY_FACILITY_KNOWLEDGE_PORT"


def test_ensure_okf_server_alias_delegates_to_okf_key(monkeypatch):
    """The ensure_okf_server alias delegates to ensure_web_server("okf").

    Behavioural (was an inspect.getsource substring check): patch the delegate
    and assert the alias forwards the bare "okf" key — stronger than matching
    source text and immune to getsource line-slicing flakes.
    """
    assert hasattr(server_launcher, "ensure_okf_server")
    keys: list[str] = []
    monkeypatch.setattr(server_launcher, "ensure_web_server", keys.append)
    server_launcher.ensure_okf_server()
    assert keys == ["okf"]


def test_proxy_state_map_wires_okf_to_okf_server_url():
    assert proxy_module._PANEL_STATE_MAP["okf"] == "okf_server_url"


def test_three_way_okf_consistency():
    """The one panel id 'okf' is the registry key, the state-attr key, and the gate."""
    panel_id = "okf"
    assert panel_id in BUILTIN_PANELS
    assert panel_id in FRAMEWORK_WEB_SERVERS
    assert panel_id in proxy_module._PANEL_STATE_MAP
    # And the state attr name is derived from that same id.
    assert proxy_module._PANEL_STATE_MAP[panel_id] == f"{panel_id}_server_url"


def test_panels_route_exposes_okf_server_config_endpoint():
    """The frontend's configEndpoint /api/okf-server must exist and map to /panel/okf.

    (This site is NOT one of the plan's original five — a builtin tab silently
    fails to render without both this endpoint and the panel-manager.js entry
    below, so both are guarded here.)
    """
    from osprey.interfaces.web_terminal.routes import panels as panels_module

    paths = {getattr(r, "path", None) for r in panels_module.router.routes}
    assert "/api/okf-server" in paths


async def test_okf_server_config_endpoint_returns_proxy_path():
    """okf_server_config returns the /panel/okf reverse-proxy path when available.

    Behavioural replacement for an inspect.getsource substring check: exercise
    the handler directly for both the available and unavailable states.
    """
    from osprey.interfaces.web_terminal.routes import panels as panels_module

    available = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(okf_server_url="http://127.0.0.1:10600"))
    )
    result = await panels_module.okf_server_config(available)
    assert result == {"url": "/panel/okf", "available": True}

    unavailable = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    result = await panels_module.okf_server_config(unavailable)
    assert result == {"url": None, "available": False}


def test_frontend_panel_manager_registers_okf_tab():
    """The shipped panel catalog must include okf so the KNOWLEDGE tab renders.

    The PANELS array lives in panel-catalog.js; panel-manager.js imports it
    from there and filters it against /api/panels at init.
    """
    import os

    catalog_path = os.path.join(
        os.path.dirname(inspect.getfile(web_terminal_app)),
        "static",
        "js",
        "panel-catalog.js",
    )
    with open(catalog_path, encoding="utf-8") as fh:
        js = fh.read()
    assert "id: 'okf'" in js
    assert "/api/okf-server" in js
    assert "KNOWLEDGE" in js


def test_build_chain_reads_builtins_dynamically_no_hardcoded_okf():
    """DA IA-1: build_profile_model / manifest gate on BUILTIN_PANELS, not literals."""
    from osprey.cli import build_profile_model
    from osprey.cli.templates import manifest

    # Both import the shared set rather than hardcoding panel ids; adding "okf"
    # to BUILTIN_PANELS is therefore sufficient (no separate edit needed).
    # Fresh-file reads (not inspect.getsource) keep this guard deterministic.
    assert "BUILTIN_PANELS" in _fresh_source(build_profile_model)
    assert "BUILTIN_PANELS" in _fresh_source(manifest)
