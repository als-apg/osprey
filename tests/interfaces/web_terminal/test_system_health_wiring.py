"""App-side wiring tests for the native ``system-health`` builtin web panel.

Task 4.2 wires the SYSTEM tab into the web terminal. These guard the app-side
half of that wiring:

- the launch fn stores the proxy state attr and delegates to the launcher;
- its (host, port) resolution AGREES with what ``ServerLauncher`` binds for
  every env case, incl. the set-but-empty override (the okf regression);
- ``require_section`` is False, so the tab launches even with no ``health``
  section (the framework ships a usable default);
- the proxy state-attr map, the ``/api/system-health-server`` config endpoint,
  and the panel-manager.js descriptor (with an EXPLICIT ``healthEndpoint`` so
  the sidecar-liveness LED actually polls) are all present;
- the control-assistant preset lists the tab without disturbing the separate
  Bluesky ``health`` panel.

The registry-level registration invariants (builtin set ↔ registry key ↔
state-attr map three-way consistency, env-var derivation) live in
``tests/registry/test_system_health_panel_registration.py``.
"""

from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

import pytest
import yaml

from osprey.infrastructure import server_launcher
from osprey.interfaces.web_terminal import app as web_terminal_app
from osprey.interfaces.web_terminal.routes import proxy as proxy_module


def _fresh_source(obj) -> str:
    """Read *obj*'s source file directly from disk (deterministic).

    Preferred over ``inspect.getsource`` for "this wiring line exists" guards:
    getsource slices by ``co_firstlineno`` against a linecache copy and can
    silently return an adjacent definition under a transient ``.py``/``.pyc``
    skew. Reading the whole file fresh avoids that flake.
    """
    path = inspect.getsourcefile(obj) or inspect.getfile(obj)
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def test_launch_system_health_server_sets_proxy_state_and_invokes_launcher(monkeypatch):
    """The launch fn stores app.state.system_health_server_url and delegates."""
    import osprey.utils.workspace as workspace

    ensure_calls: list[bool] = []
    monkeypatch.setattr(
        workspace, "load_osprey_config", lambda: {"health": {"web": {"port": 8094}}}
    )
    monkeypatch.setattr(
        server_launcher, "ensure_system_health_server", lambda: ensure_calls.append(True)
    )
    monkeypatch.delenv("OSPREY_HEALTH_PORT", raising=False)

    app = SimpleNamespace(state=SimpleNamespace())
    web_terminal_app._launch_system_health_server(app)

    # The proxy reads app.state.system_health_server_url; it must be populated.
    assert getattr(app.state, "system_health_server_url", None)
    assert ensure_calls == [True]


def test_launch_system_health_server_launches_without_health_section(monkeypatch):
    """require_section is False: no ``health`` section still yields a live URL.

    Unlike the okf/channel-finder launchers (which early-return on a missing
    section), the health framework ships a usable default, so the panel must
    launch on the default port rather than leaving a dead tab.
    """
    import osprey.utils.workspace as workspace

    ensure_calls: list[bool] = []
    monkeypatch.setattr(workspace, "load_osprey_config", lambda: {})
    monkeypatch.setattr(
        server_launcher, "ensure_system_health_server", lambda: ensure_calls.append(True)
    )
    monkeypatch.delenv("OSPREY_HEALTH_PORT", raising=False)

    app = SimpleNamespace(state=SimpleNamespace())
    web_terminal_app._launch_system_health_server(app)

    assert app.state.system_health_server_url == "http://127.0.0.1:8094"
    assert ensure_calls == [True]


def _launch_side_port(monkeypatch, fake_config, env_value):
    """Port the launch fn stores in app.state.system_health_server_url."""
    import osprey.utils.workspace as workspace

    monkeypatch.setattr(workspace, "load_osprey_config", lambda: fake_config)
    monkeypatch.setattr(server_launcher, "ensure_system_health_server", lambda: None)
    if env_value is None:
        monkeypatch.delenv("OSPREY_HEALTH_PORT", raising=False)
    else:
        monkeypatch.setenv("OSPREY_HEALTH_PORT", env_value)

    app = SimpleNamespace(state=SimpleNamespace())
    web_terminal_app._launch_system_health_server(app)
    url = getattr(app.state, "system_health_server_url", None)
    assert url, "launch fn crashed → system_health_server_url is None (silent dead tab)"
    return int(url.rsplit(":", 1)[1])


def _launcher_side_port(monkeypatch, fake_config, env_value):
    """Port ServerLauncher's config reader resolves (the port uvicorn binds)."""
    from osprey.registry.web import FRAMEWORK_WEB_SERVERS

    monkeypatch.setattr(server_launcher, "load_osprey_config", lambda: fake_config)
    if env_value is None:
        monkeypatch.delenv("OSPREY_HEALTH_PORT", raising=False)
    else:
        monkeypatch.setenv("OSPREY_HEALTH_PORT", env_value)
    _host, port = server_launcher._make_config_reader(FRAMEWORK_WEB_SERVERS["system_health"])()
    return port


@pytest.mark.parametrize(
    "env_value",
    [
        None,  # no override → port_default 8094 on both sides
        "9099",  # explicit override → both sides honour it
        "",  # SET-BUT-EMPTY (compose `VAR=`) → must not crash the launch (regression)
    ],
)
def test_launch_and_launcher_agree_on_port(monkeypatch, env_value):
    """The proxied URL's port MUST equal the port uvicorn binds, for every env case.

    Regression guard for the empty-string override: ``int("")`` would raise
    inside the launch fn → swallowed → system_health_server_url=None → dead
    tab, while the launcher would bind 8094. Both sides must resolve the same
    port via the shared ``if env_val:`` guard.
    """
    fake = {"health": {"web": {}}}
    app_port = _launch_side_port(monkeypatch, fake, env_value)
    bound_port = _launcher_side_port(monkeypatch, fake, env_value)
    assert app_port == bound_port


def test_lifespan_gates_system_health_launch_on_enabled_panels():
    """The lifespan launches system-health only when the panel id is enabled."""
    assert hasattr(web_terminal_app, "_launch_system_health_server")
    module_src = _fresh_source(web_terminal_app)
    assert '"system-health" in enabled_panels' in module_src
    assert "_launch_system_health_server(app)" in module_src


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


def test_panel_manager_registers_system_health_tab_with_explicit_health_endpoint():
    """panel-manager.js must register the SYSTEM tab with an explicit health poll.

    Since commit 8ae9e282 an omitted/null ``healthEndpoint`` SKIPS polling and
    pins the tab healthy; the SYSTEM tab's LED must reflect real sidecar
    liveness, so the descriptor MUST carry ``healthEndpoint: '/health'``.
    """
    pm_path = os.path.join(
        os.path.dirname(inspect.getfile(web_terminal_app)),
        "static",
        "js",
        "panel-manager.js",
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
