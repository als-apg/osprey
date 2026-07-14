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

import pytest

from osprey.infrastructure import server_launcher
from osprey.interfaces.web_terminal import app as web_terminal_app
from osprey.interfaces.web_terminal.routes import proxy as proxy_module
from osprey.profiles.web_panels import BUILTIN_PANEL_LABELS, BUILTIN_PANELS
from osprey.registry.web import FRAMEWORK_WEB_SERVERS, WebServerDefinition


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
    assert defn.port_default == 8093
    assert defn.require_section is True
    assert defn.factory_config_kwargs == {"bundle_path": "facility_knowledge.bundle_path"}


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


def test_launch_okf_server_sets_proxy_state_and_invokes_launcher(monkeypatch):
    """_launch_okf_server stores the state attr the proxy reads and delegates.

    Behavioural replacement for the source-text checks that the launch fn
    references ``app.state.okf_server_url`` and ``ensure_okf_server``.
    """
    import osprey.utils.workspace as workspace

    ensure_calls: list[bool] = []
    monkeypatch.setattr(
        workspace, "load_osprey_config", lambda: {"facility_knowledge": {"bundle_path": "/b"}}
    )
    monkeypatch.setattr(server_launcher, "ensure_okf_server", lambda: ensure_calls.append(True))
    monkeypatch.delenv("OSPREY_FACILITY_KNOWLEDGE_PORT", raising=False)

    app = SimpleNamespace(state=SimpleNamespace())
    web_terminal_app._launch_okf_server(app)

    # The proxy reads app.state.okf_server_url; it must be populated (not a dead tab).
    assert getattr(app.state, "okf_server_url", None)
    # And the launch delegated to the launcher.
    assert ensure_calls == [True]


def test_lifespan_gates_okf_launch_on_enabled_panels():
    """The lifespan launches okf only when the bare "okf" panel id is enabled.

    Driving the full async lifespan here is disproportionate, so this stays a
    source guard — but reads the module source fresh from disk (deterministic)
    rather than via inspect.getsource.
    """
    assert hasattr(web_terminal_app, "_launch_okf_server")
    module_src = _fresh_source(web_terminal_app)
    assert '"okf" in enabled_panels' in module_src
    assert "_launch_okf_server(app)" in module_src


def _launch_side_port(monkeypatch, fake_config, env_value):
    """Port that _launch_okf_server stores in app.state.okf_server_url (app side)."""
    import osprey.utils.workspace as workspace
    from osprey.interfaces.web_terminal import app as wt

    monkeypatch.setattr(workspace, "load_osprey_config", lambda: fake_config)
    monkeypatch.setattr(server_launcher, "ensure_okf_server", lambda: None)  # no real launch
    if env_value is None:
        monkeypatch.delenv("OSPREY_FACILITY_KNOWLEDGE_PORT", raising=False)
    else:
        monkeypatch.setenv("OSPREY_FACILITY_KNOWLEDGE_PORT", env_value)

    app = SimpleNamespace(state=SimpleNamespace())
    wt._launch_okf_server(app)
    url = getattr(app.state, "okf_server_url", None)
    assert url, "launch fn crashed → app.state.okf_server_url is None (silent dead tab)"
    return int(url.rsplit(":", 1)[1])


def _launcher_side_port(monkeypatch, fake_config, env_value):
    """Port that ServerLauncher's config reader resolves (the port uvicorn binds)."""
    from osprey.registry.web import FRAMEWORK_WEB_SERVERS

    monkeypatch.setattr(server_launcher, "load_osprey_config", lambda: fake_config)
    if env_value is None:
        monkeypatch.delenv("OSPREY_FACILITY_KNOWLEDGE_PORT", raising=False)
    else:
        monkeypatch.setenv("OSPREY_FACILITY_KNOWLEDGE_PORT", env_value)
    _host, port = server_launcher._make_config_reader(FRAMEWORK_WEB_SERVERS["okf"])()
    return port


@pytest.mark.parametrize(
    "env_value",
    [
        None,  # no override → port_default 8093 on both sides
        "9099",  # explicit override → both sides honour it
        "",  # SET-BUT-EMPTY (compose `VAR=`) → must not crash the launch (regression)
    ],
)
def test_launch_and_launcher_agree_on_port(monkeypatch, env_value):
    """The proxied URL's port MUST equal the port uvicorn binds, for every env case.

    Regression guard for the empty-string override: int("") would raise inside
    _launch_okf_server → swallowed → okf_server_url=None → dead tab, while the
    launcher would bind 8093. Both sides must resolve the same port.
    """
    fake = {"facility_knowledge": {"bundle_path": "/some/bundle"}}
    app_port = _launch_side_port(monkeypatch, fake, env_value)
    bound_port = _launcher_side_port(monkeypatch, fake, env_value)
    assert app_port == bound_port


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
        app=SimpleNamespace(state=SimpleNamespace(okf_server_url="http://127.0.0.1:8093"))
    )
    result = await panels_module.okf_server_config(available)
    assert result == {"url": "/panel/okf", "available": True}

    unavailable = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    result = await panels_module.okf_server_config(unavailable)
    assert result == {"url": None, "available": False}


def test_frontend_panel_manager_registers_okf_tab():
    """panel-manager.js PANELS must include okf so the KNOWLEDGE tab renders."""
    import os

    pm_path = os.path.join(
        os.path.dirname(inspect.getfile(web_terminal_app)),
        "static",
        "js",
        "panel-manager.js",
    )
    with open(pm_path, encoding="utf-8") as fh:
        js = fh.read()
    assert "id: 'okf'" in js
    assert "/api/okf-server" in js
    assert "KNOWLEDGE" in js


def test_build_chain_reads_builtins_dynamically_no_hardcoded_okf():
    """DA IA-1: build_profile / manifest gate on BUILTIN_PANELS, not literals."""
    from osprey.cli import build_profile
    from osprey.cli.templates import manifest

    # Both import the shared set rather than hardcoding panel ids; adding "okf"
    # to BUILTIN_PANELS is therefore sufficient (no separate edit needed).
    # Fresh-file reads (not inspect.getsource) keep this guard deterministic.
    assert "BUILTIN_PANELS" in _fresh_source(build_profile)
    assert "BUILTIN_PANELS" in _fresh_source(manifest)
