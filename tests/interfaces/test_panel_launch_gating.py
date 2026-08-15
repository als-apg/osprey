"""Every companion panel must follow its own ``auto_launch``, not just the gallery.

Panel availability is computed from ``app.state.<key>_server_url`` alone, so
publishing a URL for a server that was never started hands the operator an
enabled tab whose iframe returns a bare 502. The artifact gallery learned that
lesson; the other five launchers were hand-rolled copies that published the URL
first and only then called an ``ensure_*`` that no-ops when auto-launch is off.

These tests are parametrized over the whole companion-server registry, so a
newly registered panel is covered the day it is registered — and so the gating
cannot be fixed for one panel and missed on the next.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from osprey.infrastructure import server_launcher
from osprey.interfaces.web_terminal import app as web_terminal_app
from osprey.registry.web import (
    FRAMEWORK_WEB_SERVERS,
    WebServerDefinition,
    panel_url_state_attr,
)
from osprey.utils import workspace

ALL_KEYS = sorted(FRAMEWORK_WEB_SERVERS)


def _launch(app: SimpleNamespace, key: str) -> None:
    """Invoke the panel launcher under test for *key*."""
    web_terminal_app._launch_panel_server(app, key)


def _address_config(definition: WebServerDefinition, **address: object) -> dict:
    """Build a config that writes *address* at the depth this server reads.

    Depth is the server's own declared fact (``config_web_subkey``); spelling it
    per-server in the test would just re-import the drift these launchers had.
    """
    if definition.config_web_subkey:
        return {definition.config_key: {definition.config_web_subkey: dict(address)}}
    return {definition.config_key: dict(address)}


def _url(app: SimpleNamespace, key: str) -> str | None:
    """The panel URL the proxy would read for *key*.

    Goes through the shared declaration rather than re-spelling the attribute
    name, so these tests assert the launcher publishes where the proxy looks —
    not merely that it publishes somewhere.
    """
    return getattr(app.state, panel_url_state_attr(key), None)


@pytest.fixture
def stub_app() -> SimpleNamespace:
    """A stand-in for the FastAPI app — the launcher only touches ``.state``."""
    return SimpleNamespace(state=SimpleNamespace())


@pytest.fixture
def launched(monkeypatch) -> list[str]:
    """Record the registry keys actually handed to ``ensure_web_server``."""
    calls: list[str] = []
    monkeypatch.setattr(server_launcher, "ensure_web_server", calls.append)
    return calls


@pytest.fixture
def config(monkeypatch) -> dict:
    """A mutable config seen by both the launcher and the address resolver."""
    cfg: dict = {}
    monkeypatch.setattr(server_launcher, "load_osprey_config", lambda: cfg)
    monkeypatch.setattr(workspace, "load_osprey_config", lambda: cfg)
    # An inherited port override would decide the port out from under the
    # config these tests set.
    for definition in FRAMEWORK_WEB_SERVERS.values():
        monkeypatch.delenv(definition.port_env_var, raising=False)
    return cfg


@pytest.mark.parametrize("key", ALL_KEYS)
class TestAutoLaunchGating:
    def test_auto_launch_false_publishes_no_url(self, key, config, stub_app, launched):
        """The defect: a suppressed server advertised as a live panel."""
        definition = FRAMEWORK_WEB_SERVERS[key]
        config.update(_address_config(definition, auto_launch=False, port=8099))

        _launch(stub_app, key)

        assert _url(stub_app, key) is None
        assert launched == []

    def test_auto_launch_true_publishes_the_resolved_url(self, key, config, stub_app, launched):
        definition = FRAMEWORK_WEB_SERVERS[key]
        config.update(_address_config(definition, auto_launch=True, host="127.0.0.1", port=8099))

        _launch(stub_app, key)

        assert _url(stub_app, key) == "http://127.0.0.1:8099"
        assert launched == [key]

    def test_published_url_honours_the_port_env_override(
        self, key, config, stub_app, launched, monkeypatch
    ):
        """Multi-user compose moves every panel's port via this env var.

        The URL published here and the port uvicorn binds come from the same
        resolver, so a per-user deployment cannot end up with a tab pointing at
        a port nothing listens on.
        """
        definition = FRAMEWORK_WEB_SERVERS[key]
        config.update(_address_config(definition, auto_launch=True, port=8099))
        monkeypatch.setenv(definition.port_env_var, "9999")

        _launch(stub_app, key)

        assert _url(stub_app, key) == "http://127.0.0.1:9999"

    def test_set_but_empty_port_override_falls_back_to_config(
        self, key, config, stub_app, launched, monkeypatch
    ):
        """A compose file's bare ``OSPREY_..._PORT=`` must not kill the launch."""
        definition = FRAMEWORK_WEB_SERVERS[key]
        config.update(_address_config(definition, auto_launch=True, port=8099))
        monkeypatch.setenv(definition.port_env_var, "")

        _launch(stub_app, key)

        assert _url(stub_app, key) == "http://127.0.0.1:8099"

    def test_launch_failure_retracts_the_url(self, key, config, stub_app, monkeypatch):
        """A URL assigned before the launch blew up must not survive it."""
        definition = FRAMEWORK_WEB_SERVERS[key]
        config.update(_address_config(definition, auto_launch=True, port=8099))

        def _boom(_key: str) -> None:
            raise OSError("port unusable")

        monkeypatch.setattr(server_launcher, "ensure_web_server", _boom)

        _launch(stub_app, key)

        assert _url(stub_app, key) is None

    def test_unreadable_config_publishes_no_url(self, key, stub_app, launched, monkeypatch):
        def _boom() -> dict:
            raise RuntimeError("config.yml is unreadable")

        monkeypatch.setattr(server_launcher, "load_osprey_config", _boom)
        monkeypatch.setattr(workspace, "load_osprey_config", _boom)

        _launch(stub_app, key)

        assert _url(stub_app, key) is None
        assert launched == []


@pytest.mark.parametrize("key", ALL_KEYS)
@pytest.mark.parametrize(
    "env_value",
    [
        None,  # no override → the config/default port on both sides
        "9099",  # explicit override → both sides honour it
        "",  # SET-BUT-EMPTY (compose `VAR=`) → must not crash the launch
    ],
)
def test_published_port_equals_the_port_the_launcher_binds(
    key, env_value, config, stub_app, launched, monkeypatch
):
    """The advertised port and the bound port are compared, not assumed equal.

    The panel URL published on ``app.state`` is what the operator's iframe is
    pointed at; ``ServerLauncher``'s own config reader is what uvicorn binds. A
    launcher that re-derived either half by hand drifted from the other — an
    ``int("")`` on a set-but-empty override killed the launch while the server
    came up fine, leaving a tab pointing at a port nothing served.
    """
    definition = FRAMEWORK_WEB_SERVERS[key]
    config.update(_address_config(definition, auto_launch=True))
    if env_value is None:
        monkeypatch.delenv(definition.port_env_var, raising=False)
    else:
        monkeypatch.setenv(definition.port_env_var, env_value)

    _launch(stub_app, key)

    url = _url(stub_app, key)
    assert url, "the launcher crashed — no URL published (a silent dead tab)"
    _host, bound_port = server_launcher._launchers[key]._config_reader()
    assert int(url.rsplit(":", 1)[1]) == bound_port


class TestEnabledPanelSelection:
    """Which panels the lifespan launches, isolated from whether each one may.

    The lifespan used to gate each panel with its own ``if "<id>" in
    enabled_panels`` line, so a newly registered panel was launched only if
    someone remembered to add a seventh branch. These pin the derivation.
    """

    @pytest.fixture
    def launch_calls(self, monkeypatch) -> list[str]:
        calls: list[str] = []
        monkeypatch.setattr(
            web_terminal_app, "_launch_panel_server", lambda app, key: calls.append(key)
        )
        return calls

    def test_launches_the_registry_key_behind_each_enabled_panel_id(self, stub_app, launch_calls):
        web_terminal_app._launch_enabled_panel_servers(stub_app, {"okf", "artifacts"})

        assert sorted(launch_calls) == ["artifact", "okf"]

    def test_ignores_panel_ids_with_no_companion_server(self, stub_app, launch_calls):
        """Custom panels (``events``) are URL-backed — the framework serves nothing."""
        web_terminal_app._launch_enabled_panel_servers(stub_app, {"artifacts", "events"})

        assert launch_calls == ["artifact"]

    def test_launches_nothing_when_no_builtin_panel_is_enabled(self, stub_app, launch_calls):
        web_terminal_app._launch_enabled_panel_servers(stub_app, set())

        assert launch_calls == []

    def test_every_builtin_panel_is_launchable(self, stub_app, launch_calls):
        """No built-in panel may be enabled with nothing to launch behind it."""
        from osprey.profiles.web_panels import BUILTIN_PANELS

        web_terminal_app._launch_enabled_panel_servers(stub_app, set(BUILTIN_PANELS))

        assert sorted(launch_calls) == ALL_KEYS


@pytest.mark.parametrize(
    "key", sorted(k for k, d in FRAMEWORK_WEB_SERVERS.items() if d.require_section)
)
def test_missing_section_publishes_no_url(key, config, stub_app, launched):
    """``require_section`` servers stay dark when their config section is absent."""
    _launch(stub_app, key)

    assert _url(stub_app, key) is None
    assert launched == []


@pytest.mark.parametrize(
    "key", sorted(k for k, d in FRAMEWORK_WEB_SERVERS.items() if not d.require_section)
)
def test_missing_section_still_launches_when_not_required(key, config, stub_app, launched):
    """``auto_launch`` defaults on — omitting the section must not disable the tab."""
    definition = FRAMEWORK_WEB_SERVERS[key]

    _launch(stub_app, key)

    assert _url(stub_app, key) == f"http://127.0.0.1:{definition.port_default}"
    assert launched == [key]
