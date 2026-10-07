"""Tests for web panel configuration (template-driven panel filtering)."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import threading
import time
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal import app as web_terminal_app
from osprey.interfaces.web_terminal.app import (
    BUILTIN_PANELS,
    UNIVERSAL_PANELS,
    _load_panel_config,
    _load_panel_presets,
    create_app,
)
from osprey.interfaces.web_terminal.operator_session import resolve_agent_data_root
from osprey.interfaces.web_terminal.routes import panels as panels_module
from osprey.profiles.web_panels import BUILTIN_PANEL_LABELS, SIDECAR_PANELS

from .conftest import HOST_ADDRS_TARGET, StubWorkspaceWatcher

#: The real probe, bound at import time — before the autouse stub replaces the
#: module attribute — so the memoization tests can drive the implementation the
#: stub stands in for.
_real_host_interface_addresses = panels_module._host_interface_addresses


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


#: The panel id of the sidecar under test, and the dotted path of the class the
#: lifespan imports for it. Both read off the registry so this module pins the
#: launch path rather than a spelling.
SIDECAR_ID = next(iter(sorted(SIDECAR_PANELS)))
_SIDECAR_FACTORY_TARGET = SIDECAR_PANELS[SIDECAR_ID].factory_path.replace(":", ".")


class SidecarScript:
    """What a stub sidecar does when the lifespan launches it, and what it saw.

    The real class starts a server process; nothing in this module may. The
    default script fails preflight, so every client fixture here gets the
    unavailable-panel path unless it asks for a ready one. The errors may be
    changed between attempts, and ``hold`` (an event) keeps ``wait_ready``
    blocked until it is set or the sidecar is stopped. ``events`` records every
    spawn and stop in order.
    """

    def __init__(
        self,
        preflight_error="stub sidecar: not launched",
        stderr_tail="",
        wait_ready_error=None,
        exit_status=None,
    ):
        self.preflight_error = preflight_error
        self.wait_ready_error = wait_ready_error
        self.stderr_tail = stderr_tail
        self.exit_status = exit_status
        self.url = "http://127.0.0.1:9/panel/" + SIDECAR_ID
        self.auth_headers = {"authorization": "Bearer stub-token"}
        self.constructed: list[tuple] = []
        self.spawned = 0
        self.waited: list[float] = []
        self.stopped = 0
        self.hold: threading.Event | None = None
        self.events: list[str] = []


def _stub_sidecar_class(script):
    """A stand-in sidecar class bound to *script*, matching the real signature."""

    class _StubSidecar:
        def __init__(self, shared_root, outer_prefix, pinned_mode):
            script.constructed.append((shared_root, outer_prefix, pinned_mode))
            self._stopping = threading.Event()

        @property
        def stderr_tail(self):
            return script.stderr_tail

        @property
        def exit_status(self):
            return script.exit_status

        @property
        def token(self):
            return "stub-token"

        @property
        def auth_headers(self):
            return dict(script.auth_headers)

        @property
        def url(self):
            return script.url

        def preflight(self):
            if script.preflight_error:
                raise RuntimeError(script.preflight_error)

        def spawn(self):
            script.spawned += 1
            script.events.append("spawn")

        def wait_ready(self, timeout):
            script.waited.append(timeout)
            hold = script.hold
            if hold is not None:
                while not hold.wait(0.02) and not self._stopping.is_set():
                    pass
            if self._stopping.is_set():
                raise RuntimeError("stub sidecar: stopped before it was ready")
            if script.wait_ready_error:
                raise RuntimeError(script.wait_ready_error)

        def stop(self):
            self._stopping.set()
            script.stopped += 1
            script.events.append("stop")

    return _StubSidecar


def _config_reader(values):
    """A ``get_config_value`` stand-in answering *values*, else the caller's default."""

    def _get(path, default=None, _config_path=None):
        return values.get(path, default)

    return _get


def _make_client(
    workspace_dir,
    enabled_panels=None,
    custom_panels=None,
    sidecar_script=None,
    config_values: dict | None = None,
):
    """Create a TestClient with the given panel config.

    *config_values* maps dotted config keys to what ``get_config_value`` returns
    for them; every other key resolves to the default its reader passes.
    """
    if enabled_panels is None:
        enabled_panels = set(UNIVERSAL_PANELS)
    if custom_panels is None:
        custom_panels = []
    config_patch = (
        patch("osprey.utils.config.get_config_value", side_effect=_config_reader(config_values))
        if config_values is not None
        else nullcontext()
    )
    with (
        config_patch,
        patch(_SIDECAR_FACTORY_TARGET, _stub_sidecar_class(sidecar_script or SidecarScript())),
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=(enabled_panels, custom_panels, None),
        ),
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as c:
            yield c


@pytest.fixture
def client(workspace_dir):
    """Client with only universal panels (no domain panels)."""
    yield from _make_client(workspace_dir)


@pytest.fixture
def client_all_panels(workspace_dir):
    """Client with all built-in panels enabled."""
    yield from _make_client(workspace_dir, enabled_panels=set(BUILTIN_PANELS))


@pytest.fixture
def client_with_custom_panels(workspace_dir):
    """Client with custom panels added."""
    custom = [
        {"id": "my-dashboard", "label": "DASHBOARD", "url": "http://localhost:9000"},
        {"id": "grafana", "label": "GRAFANA", "url": "http://localhost:3000"},
    ]
    enabled = set(UNIVERSAL_PANELS)
    yield from _make_client(workspace_dir, enabled_panels=enabled, custom_panels=custom)


# ---- Unit tests for _load_panel_config ----


class TestLoadPanelConfig:
    @pytest.mark.parametrize(
        "config",
        [pytest.param({}, id="no-web-section"), pytest.param({"web": {"panels": {}}}, id="empty")],
    )
    def test_no_panels_declared(self, config):
        """No ``web.panels`` entries returns only the universal panels."""
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value=config,
        ):
            enabled, custom, _default = _load_panel_config()
        assert enabled == UNIVERSAL_PANELS
        assert custom == []

    def test_domain_panels_enabled(self):
        """Domain panels with enabled: true are added."""
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={
                "web": {
                    "panels": {
                        "ariel": {"enabled": True},
                        "lattice": {"enabled": True},
                    }
                }
            },
        ):
            enabled, custom, _default = _load_panel_config()
        assert "ariel" in enabled
        assert "lattice" in enabled
        assert "channel-finder" not in enabled
        # Universal panels are always present
        assert UNIVERSAL_PANELS <= enabled

    def test_domain_panel_disabled(self):
        """Domain panels with enabled: false are excluded."""
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={
                "web": {
                    "panels": {
                        "ariel": {"enabled": False},
                        "lattice": {"enabled": True},
                    }
                }
            },
        ):
            enabled, custom, _default = _load_panel_config()
        assert "ariel" not in enabled
        assert "lattice" in enabled

    def test_domain_panel_bare_true(self):
        """A bare `true` value (not dict) enables the panel."""
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={"web": {"panels": {"ariel": True}}},
        ):
            enabled, custom, _default = _load_panel_config()
        assert "ariel" in enabled

    def test_domain_panel_dict_defaults_enabled(self):
        """A dict without explicit enabled key defaults to enabled."""
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={"web": {"panels": {"ariel": {}}}},
        ):
            enabled, custom, _default = _load_panel_config()
        assert "ariel" in enabled

    def test_custom_panel_extracted(self):
        """Non-builtin panel IDs are returned as custom panels."""
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={
                "web": {
                    "panels": {
                        "my-grafana": {
                            "label": "GRAFANA",
                            "url": "http://grafana.local:3000",
                            "health_endpoint": "/api/health",
                        }
                    }
                }
            },
        ):
            enabled, custom, _default = _load_panel_config()
        assert len(custom) == 1
        assert custom[0]["id"] == "my-grafana"
        assert custom[0]["label"] == "GRAFANA"
        assert custom[0]["url"] == "http://grafana.local:3000"
        assert custom[0]["healthEndpoint"] == "/api/health"
        assert custom[0]["rewritePrefixes"] == []

    def test_custom_panel_rewrite_prefixes_are_threaded(self):
        """A panel's ``rewrite_prefixes`` reach the proxy as ``rewritePrefixes``."""
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={
                "web": {
                    "panels": {
                        "pvinfo": {
                            "label": "PV INFO",
                            "url": "http://pvinfo.local",
                            "path": "/pvinfo/",
                            "rewrite_prefixes": ["/pvinfo", "/pvinfo/"],
                        }
                    }
                }
            },
        ):
            _enabled, custom, _default = _load_panel_config()
        assert custom[0]["rewritePrefixes"] == ["/pvinfo", "/pvinfo/"]

    def test_custom_panel_disabled_is_not_served(self):
        """``enabled: false`` switches a custom panel off exactly as it does a
        builtin: the build writes it onto every block the profile does not
        select, so a persona that inherits a url-backed block for a tab it
        excluded must not get the tab."""
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={
                "web": {
                    "panels": {
                        "beam-viewer": {
                            "label": "BEAM",
                            "url": "http://localhost:10920",
                            "enabled": False,
                        },
                        "my-grafana": {"label": "GRAFANA", "url": "http://grafana.local:3000"},
                    }
                }
            },
        ):
            enabled, custom, _default = _load_panel_config()
        assert [cp["id"] for cp in custom] == ["my-grafana"]
        assert enabled == UNIVERSAL_PANELS

    def test_events_panel_is_url_backed_custom(self):
        """The control-assistant EVENTS panel is URL-backed.

        It must be emitted as a custom panel (carrying url/path/health) so the
        web terminal renders the iframe tab, not silently collapsed into the
        builtin `enabled` set where its url is discarded and no frontend tab exists.
        """
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={
                "web": {
                    "panels": {
                        "events": {
                            "label": "EVENTS",
                            "url": "http://localhost:8020",
                            "path": "/dashboard",
                            "health_endpoint": "/health",
                        }
                    }
                }
            },
        ):
            enabled, custom, _default = _load_panel_config()
        assert "events" not in enabled  # not silently dropped as a builtin
        events = [p for p in custom if p["id"] == "events"]
        assert len(events) == 1
        assert events[0]["url"] == "http://localhost:8020"
        assert events[0]["path"] == "/dashboard"
        assert events[0]["healthEndpoint"] == "/health"
        assert events[0]["label"] == "EVENTS"

    def test_config_defined_custom_panel_carries_marker(self):
        """Config-defined custom panels carry ``configDefined=True``.

        The marker is the trust boundary: only the config loader stamps it, so
        server-side credential injection and id reservation can key off panel
        *origin* rather than the id string (which a runtime registration can
        forge). See routes/proxy.py (events token gate) and routes/panels.py
        (register reservation).
        """
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={
                "web": {"panels": {"events": {"label": "EVENTS", "url": "http://localhost:8020"}}}
            },
        ):
            _enabled, custom, _default = _load_panel_config()
        assert custom[0]["configDefined"] is True

    def test_config_load_failure(self):
        """Config load failure returns universal panels only."""
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            side_effect=RuntimeError("no config"),
        ):
            enabled, custom, _default = _load_panel_config()
        assert enabled == UNIVERSAL_PANELS
        assert custom == []


class TestAPanelIdMustBeSpellable:
    """A declared id no request for the panel could carry refuses the start.

    The id is one URL path segment — the proxy's ``/panel/<id>``, a local
    bundle's ``/panel-static/<id>/`` — and it is also spliced into the
    ``x-forwarded-prefix`` value of every request the proxy forwards for that
    panel, escaped for neither. An id outside that class reaches no route at
    the front door and fails the hop on the way out, so the terminal serves
    every other tab and answers 500 for each request that panel makes — one of
    which names the id. Refusing at boot costs the container its start and
    names the config key instead.

    A dot-segment is refused for a second reason: ``.`` and ``..`` are spelled
    from characters a path may carry and still name something other than the
    panel wherever a path holding them is resolved.
    """

    #: Ids outside the class, one per way of leaving it.
    REFUSED = [
        "überblick",
        "renée",
        "beam viewer",
        "beam/viewer",
        "beam%20viewer",
        "beam\nviewer",
        ".",
        "..",
        "-beam",
    ]

    @staticmethod
    def _load(panels):
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={"web": {"panels": panels}},
        ):
            return _load_panel_config()

    @pytest.mark.parametrize("panel_id", REFUSED)
    def test_an_id_the_proxy_could_not_spell_is_refused(self, panel_id):
        with pytest.raises(ValueError):
            self._load({panel_id: {"label": "X", "url": "http://localhost:9000"}})

    def test_the_refusal_names_the_key_the_id_and_the_class(self):
        """The operator who wrote the id is the only one who can change it.

        A boot refusal that says only *some* panel id is wrong sends them
        through every block in the file, which is the position the 500s already
        left them in.
        """
        with pytest.raises(ValueError) as refusal:
            self._load({"überblick": {"url": "http://localhost:9000"}})

        message = str(refusal.value)
        assert "web.panels.überblick" in message
        assert repr("überblick") in message
        assert "[A-Za-z0-9][A-Za-z0-9._-]*" in message

    def test_a_block_switched_off_is_refused_too(self):
        """``enabled`` is a flag an operator flips, not a reason to skip the id.

        An unservable id behind ``enabled: false`` is a 500 waiting for the day
        someone turns the panel on, and that day is the worst one to read about
        it.
        """
        with pytest.raises(ValueError):
            self._load({"überblick": {"enabled": False, "url": "http://localhost:9000"}})

    @pytest.mark.parametrize("panel_id", ["my-grafana", "beam_viewer", "GRAFANA", "v1.2", "okf2"])
    def test_an_ordinary_custom_id_is_served(self, panel_id):
        _enabled, custom, _default = self._load({panel_id: {"url": "http://localhost:9000"}})
        assert [cp["id"] for cp in custom] == [panel_id]

    def test_every_builtin_id_is_served(self):
        """The framework's own ids clear the class it holds a config id to.

        A built-in named outside it would make every deployment that selects
        that tab refuse to start, and the name is chosen in this repo rather
        than by the operator who would have to read the refusal.
        """
        builtins = sorted(BUILTIN_PANELS | UNIVERSAL_PANELS)
        enabled, custom, _default = self._load({pid: {"enabled": True} for pid in builtins})
        assert enabled == set(builtins)
        assert custom == []

    def test_a_container_declaring_such_an_id_never_serves(self, workspace_dir):
        """The refusal lands in startup, before a request can reach a panel."""
        with (
            patch(
                "osprey.interfaces.web_terminal.app._load_web_config",
                return_value={"watch_dir": str(workspace_dir)},
            ),
            patch(
                "osprey.utils.workspace.load_osprey_config",
                return_value={"web": {"panels": {"überblick": {"url": "http://localhost:9000"}}}},
            ),
        ):
            app = create_app(shell_command="echo")
            with pytest.raises(ValueError, match="web.panels"), TestClient(app):
                pass


# ---- Unit tests for _load_panel_presets ----


class TestLoadPanelPresets:
    def test_fail_open_on_config_error(self):
        """Any config-read error resolves to an empty preset list (fail open)."""
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            side_effect=RuntimeError("no config"),
        ):
            assert _load_panel_presets({"artifacts"}, []) == []

    def test_no_presets_returns_empty(self):
        """A config with no web.presets yields an empty list (the default)."""
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={"web": {}},
        ):
            assert _load_panel_presets({"artifacts"}, []) == []

    def test_drops_unknown_members_keeps_known(self):
        """Unknown member ids are dropped; the known members are kept in order."""
        cfg = {"web": {"presets": {"Setup": ["artifacts", "ghost", "ariel"]}}}
        with patch("osprey.utils.workspace.load_osprey_config", return_value=cfg):
            presets = _load_panel_presets({"artifacts", "ariel"}, [])
        assert presets == [{"name": "Setup", "panels": ["artifacts", "ariel"]}]

    def test_drops_preset_that_resolves_empty(self):
        """A preset whose members are all unknown is dropped entirely."""
        cfg = {"web": {"presets": {"Empty": ["ghost"], "Good": ["artifacts"]}}}
        with patch("osprey.utils.workspace.load_osprey_config", return_value=cfg):
            presets = _load_panel_presets({"artifacts"}, [])
        assert presets == [{"name": "Good", "panels": ["artifacts"]}]

    def test_preserves_config_order(self):
        """Preset order follows config insertion order (pyyaml preserves it)."""
        cfg = {"web": {"presets": {"B": ["artifacts"], "A": ["ariel"]}}}
        with patch("osprey.utils.workspace.load_osprey_config", return_value=cfg):
            presets = _load_panel_presets({"artifacts", "ariel"}, [])
        assert [p["name"] for p in presets] == ["B", "A"]

    def test_custom_panel_ids_are_known_members(self):
        """Custom panel ids (not just built-ins) count as known preset members."""
        cfg = {"web": {"presets": {"Dash": ["my-dash", "artifacts"]}}}
        custom = [{"id": "my-dash", "label": "DASH", "url": "http://x:9000"}]
        with patch("osprey.utils.workspace.load_osprey_config", return_value=cfg):
            presets = _load_panel_presets({"artifacts"}, custom)
        assert presets == [{"name": "Dash", "panels": ["my-dash", "artifacts"]}]

    def test_non_list_preset_value_skipped(self):
        """A preset whose value is not a list is skipped, not crashed on."""
        cfg = {"web": {"presets": {"Bad": "artifacts", "Good": ["artifacts"]}}}
        with patch("osprey.utils.workspace.load_osprey_config", return_value=cfg):
            presets = _load_panel_presets({"artifacts"}, [])
        assert presets == [{"name": "Good", "panels": ["artifacts"]}]


# ---- API endpoint tests ----


class TestPanelsAPI:
    def test_panels_api_universal_only(self, client):
        """GET /api/panels returns only universal panels when no domain panels enabled."""
        resp = client.get("/api/panels")
        assert resp.status_code == 200
        data = resp.json()
        enabled_set = set(data["enabled"])
        assert enabled_set == UNIVERSAL_PANELS
        assert data["custom"] == []

    def test_panels_api_with_custom(self, client_with_custom_panels):
        """GET /api/panels returns custom panels."""
        resp = client_with_custom_panels.get("/api/panels")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["custom"]) == 2
        assert data["custom"][0]["id"] == "my-dashboard"

    def test_panels_api_reports_runtime_disabled(self, client):
        """GET /api/panels reports allow_runtime_panels=False by default.

        The frontend reads this flag to decide whether to offer the human
        'new panel from URL' input, so it must mirror the register-route gate.
        """
        data = client.get("/api/panels").json()
        assert data["allow_runtime_panels"] is False

    def test_panels_api_reports_runtime_enabled(self, client_runtime_panels):
        """GET /api/panels reports allow_runtime_panels=True when configured."""
        data = client_runtime_panels.get("/api/panels").json()
        assert data["allow_runtime_panels"] is True

    def test_panel_focus_disabled_panel(self, client):
        """Disabled panel ID returns 422."""
        resp = client.post(
            "/api/panel-focus",
            json={"panel": "ariel"},
        )
        assert resp.status_code == 422


# ---- Helpers for new feature tests ----

_LAN_ADDR = [(2, 1, 6, "", ("10.0.0.5", 0))]
_LOOPBACK_ADDR = [(2, 1, 6, "", ("127.0.0.1", 0))]
_GETADDRINFO_TARGET = "osprey.interfaces.web_terminal.routes.panels.socket.getaddrinfo"
#: Aliased from the shared conftest, where the autouse stub that neutralizes
#: this probe lives. The tests below that exercise the deploy-host check patch
#: the same target again from the inside, and that inner patch wins.
_HOST_ADDRS_TARGET = HOST_ADDRS_TARGET


def test_normalize_ip_refuses_a_non_string_address():
    """``getaddrinfo`` types ``sockaddr[0]`` as ``str | int``; a non-string is
    not an IP literal, so the address check refuses it rather than crashing."""
    assert panels_module._normalize_ip(0) is None


@pytest.mark.parametrize("run", ["first", "second"])
def test_host_interface_probe_is_memoized_and_its_cache_does_not_leak(run):
    """The own-address probe runs once per TTL window, and once per test.

    ``_host_interface_addresses`` resolves the host's own name, which has no
    timeout of its own — on a host with a slow or unreachable resolver that is
    an unbounded stall, and it sits on the panel-registration path. So it is
    memoized, and a second call inside the TTL must not re-resolve.

    Running the identical body twice under the parametrization is the leak
    guard: the cache is a module global, so if the shared conftest fixture
    stopped resetting it, the ``second`` run would find the ``first`` run's
    entry still fresh and observe no resolution at all — the failure mode where
    one test's machine picture silently answers another test's question.
    """
    # Arrange: the hostname resolves to one LAN address; the UDP-connect probes
    # are refused, which the helper treats as an expected non-answer.
    with (
        patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR) as resolve,
        patch("osprey.interfaces.web_terminal.routes.panels.socket.socket", side_effect=OSError),
    ):
        # Act
        first = _real_host_interface_addresses()
        second = _real_host_interface_addresses()

    # Assert
    assert first == second == frozenset({ipaddress.ip_address("10.0.0.5")})
    assert resolve.call_count == 1, (
        f"{run}: the probe must be memoized within its TTL and reset between tests"
    )


def _make_client_with_runtime_panels(workspace_dir, allowlist=None):
    """Yield a TestClient whose lifespan has allow_runtime_panels enabled.

    Patches both the panel-config loader and the raw osprey config so the
    lifespan sets ``app.state.allow_runtime_panels = True``.
    """
    web_cfg: dict = {"allow_runtime_panels": True}
    if allowlist is not None:
        web_cfg["runtime_panel_allowlist"] = allowlist
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=(set(UNIVERSAL_PANELS), [], None),
        ),
        patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={"web": web_cfg},
        ),
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as c:
            yield c


def _make_client_with_hidden_panel(workspace_dir, hidden_panel_id, enabled_panels):
    """Yield a TestClient where *hidden_panel_id* is enabled but hidden:true."""
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=(set(enabled_panels), [], None),
        ),
        patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={"web": {"panels": {hidden_panel_id: {"enabled": True, "hidden": True}}}},
        ),
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as c:
            yield c


def _make_client_runtime_with_config_events(workspace_dir):
    """Yield ``(app, client)`` with allow_runtime_panels AND a config-defined EVENTS panel.

    Unlike ``_make_client_with_runtime_panels``, this patches only the raw osprey
    config and lets the real ``_load_panel_config`` run — so the events entry is
    stamped with the production ``configDefined`` marker the reservation check
    relies on.
    """
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={
                "web": {
                    "allow_runtime_panels": True,
                    "panels": {
                        "events": {
                            "label": "EVENTS",
                            "url": "http://localhost:8020",
                            "path": "/dashboard",
                        }
                    },
                }
            },
        ),
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as c:
            yield app, c


@pytest.fixture
def client_runtime_panels(workspace_dir):
    """Client with allow_runtime_panels: True and no allowlist."""
    yield from _make_client_with_runtime_panels(workspace_dir)


@pytest.fixture
def client_runtime_panels_allowlist(workspace_dir):
    """Client with allow_runtime_panels: True and allowlist=['grafana.lan']."""
    yield from _make_client_with_runtime_panels(workspace_dir, allowlist=["grafana.lan"])


# ---- /api/panels payload from the real lifespan ----


class TestPanelsAPIShape:
    def test_labels_map_enabled_builtin_ids_to_display_names(self, client_all_panels):
        """labels contains BUILTIN_PANEL_LABELS entries for each enabled built-in panel."""
        # Arrange — client_all_panels enables the full BUILTIN_PANELS set

        # Act
        resp = client_all_panels.get("/api/panels")

        # Assert
        assert resp.status_code == 200
        labels = resp.json()["labels"]
        for panel_id in BUILTIN_PANELS:
            assert panel_id in labels, f"missing label for {panel_id!r}"
            assert labels[panel_id] == BUILTIN_PANEL_LABELS[panel_id]

    def test_visible_defaults_to_all_enabled_when_no_hidden_flags(self, client_all_panels):
        """visible equals the full enabled set when no hidden:true flags are configured."""
        # Arrange — no hidden flags in config (load_osprey_config returns {})

        # Act
        resp = client_all_panels.get("/api/panels")

        # Assert
        assert resp.status_code == 200
        data = resp.json()
        assert set(data["visible"]) == set(data["enabled"])

    def test_active_is_null_on_fresh_start(self, client):
        """active is null immediately after server start (no panel has been focused yet)."""
        # Arrange — fresh TestClient, no panel-focus POST issued

        # Act
        resp = client.get("/api/panels")

        # Assert
        assert resp.status_code == 200
        assert resp.json()["active"] is None


# ---- Hidden panel visibility ----


class TestHiddenPanels:
    def test_hidden_builtin_absent_from_visible_but_present_in_enabled(self, workspace_dir):
        """A panel with hidden:true is omitted from visible but kept in enabled."""
        # Arrange — ariel enabled in _load_panel_config but hidden:true in raw config
        enabled = {"artifacts", "ariel"}
        gen = _make_client_with_hidden_panel(workspace_dir, "ariel", enabled)
        client = next(gen)

        try:
            # Act
            resp = client.get("/api/panels")

            # Assert
            assert resp.status_code == 200
            data = resp.json()
            assert "ariel" in data["enabled"], "ariel should be in enabled"
            assert "ariel" not in data["visible"], "ariel should not be in visible when hidden:true"
        finally:
            try:
                next(gen)
            except StopIteration:
                pass


# ---- POST /api/panel-visibility ----


class TestBroadcastAssertionsAreDeterministic:
    """The broadcast assertions below must not share a counter with a live thread.

    ``create_app``'s lifespan wires a ``WorkspaceWatcher`` to the same
    broadcaster object these tests mock, so a filesystem event delivered on the
    observer thread lands on the mock the route assertions count. On macOS
    watchdog replays events from up to 30 s before the watch started — including
    this fixture's own ``tmp_path`` creation — so the extra call arrives on a
    timing that no test controls.

    The directory conftest stubs the observer out. This guard fails if that stub
    is removed, rather than leaving the broadcast assertions below to misfire
    intermittently on a slow runner.
    """

    def test_lifespan_wires_a_watcher_that_starts_no_observer(self, client_all_panels):
        # Arrange — fixture has run the full lifespan

        # Act
        watcher = client_all_panels.app.state.watcher

        # Assert — stubbed, but still wired exactly as production wires it
        assert isinstance(watcher, StubWorkspaceWatcher), (
            "app tests must not start a real filesystem observer; "
            "it broadcasts onto the mock these tests assert on"
        )
        assert watcher.started is True, "lifespan must still start the watcher it wires"
        assert watcher.broadcaster is client_all_panels.app.state.broadcaster


class TestPanelVisibilityAPI:
    def test_valid_panel_mutates_visible_panels_state(self, client_all_panels):
        """POST /api/panel-visibility with a known id updates app.state.visible_panels."""
        # Arrange — all panels enabled; hide ariel to get a deterministic starting state
        client_all_panels.post("/api/panel-visibility", json={"panel": "ariel", "visible": False})

        # Act — show ariel again
        resp = client_all_panels.post(
            "/api/panel-visibility", json={"panel": "ariel", "visible": True}
        )

        # Assert
        assert resp.status_code == 200
        assert resp.json()["panel"] == "ariel"
        assert resp.json()["visible"] is True
        assert "ariel" in client_all_panels.app.state.visible_panels

    def test_unknown_panel_returns_422(self, client):
        """POST /api/panel-visibility returns 422 for a panel id that is not enabled."""
        # Arrange — client has only universal panels; "ariel" is disabled

        mock_broadcast = MagicMock()
        client.app.state.broadcaster.broadcast = mock_broadcast

        # Act
        resp = client.post("/api/panel-visibility", json={"panel": "ariel", "visible": True})

        # Assert
        assert resp.status_code == 422
        mock_broadcast.assert_not_called()


# ---- POST /api/panels/register ----


class TestPanelRegisterAPI:
    def test_register_forbidden_when_runtime_panels_disabled(self, client):
        """POST /api/panels/register returns 403 when allow_runtime_panels is False."""
        # Arrange — default client has allow_runtime_panels=False

        # Act
        resp = client.post(
            "/api/panels/register",
            json={"id": "my-panel", "label": "MY", "url": "http://grafana.lan:3000"},
        )

        # Assert
        assert resp.status_code == 403

    def test_register_success_stores_raw_url_in_custom_panels(self, client_runtime_panels):
        """Successful registration stores the original (raw) URL in app.state.custom_panels."""
        # Arrange
        raw_url = "http://grafana.lan:3000"

        # Act
        with patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR):
            resp = client_runtime_panels.post(
                "/api/panels/register",
                json={"id": "grafana", "label": "GRAFANA", "url": raw_url},
            )

        # Assert
        assert resp.status_code == 200
        stored = client_runtime_panels.app.state.custom_panels
        assert len(stored) == 1
        assert stored[0]["url"] == raw_url, "raw URL must be preserved in state for proxy"

    def test_register_success_broadcasts_with_proxy_url_path_and_health(
        self, client_runtime_panels
    ):
        """Successful registration broadcasts /panel/{id} URL plus path and healthEndpoint."""
        # Arrange
        mock_broadcast = MagicMock()
        client_runtime_panels.app.state.broadcaster.broadcast = mock_broadcast

        # Act
        with patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR):
            client_runtime_panels.post(
                "/api/panels/register",
                json={
                    "id": "grafana",
                    "label": "GRAFANA",
                    "url": "http://grafana.lan:3000",
                    "path": "/d/abc",
                    "health_endpoint": "/api/health",
                },
            )

        # Assert
        mock_broadcast.assert_called_once()
        event = mock_broadcast.call_args[0][0]
        assert event["type"] == "panel_register"
        assert event["url"] == "/panel/grafana"
        assert event["path"] == "/d/abc"
        assert event["healthEndpoint"] == "/api/health"

    def test_register_duplicate_id_replaces_not_duplicates(self, client_runtime_panels):
        """Re-registering an existing id replaces the entry (no duplicates, len unchanged)."""
        # Arrange — register "grafana" once
        with patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR):
            client_runtime_panels.post(
                "/api/panels/register",
                json={"id": "grafana", "label": "GRAFANA", "url": "http://grafana.lan:3000"},
            )

        # Act — re-register "grafana" with a different URL
        with patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR):
            resp = client_runtime_panels.post(
                "/api/panels/register",
                json={"id": "grafana", "label": "GRAFANA 2", "url": "http://grafana.lan:4000"},
            )

        # Assert
        assert resp.status_code == 200
        stored = client_runtime_panels.app.state.custom_panels
        assert len(stored) == 1, "duplicate id must not create a second entry"
        assert stored[0]["label"] == "GRAFANA 2"

    def test_register_loopback_address_returns_422(self, client_runtime_panels):
        """A URL whose host resolves to a loopback address is rejected with 422."""
        # Arrange — getaddrinfo returns 127.0.0.1

        # Act
        with patch(_GETADDRINFO_TARGET, return_value=_LOOPBACK_ADDR):
            resp = client_runtime_panels.post(
                "/api/panels/register",
                json={"id": "internal", "label": "INT", "url": "http://grafana.lan:3000"},
            )

        # Assert
        assert resp.status_code == 422

    def test_register_non_http_scheme_returns_422(self, client_runtime_panels):
        """A URL with a non-http/https scheme is rejected with 422 without a DNS lookup."""
        # Arrange — ftp:// is not a valid panel URL scheme

        # Act
        resp = client_runtime_panels.post(
            "/api/panels/register",
            json={"id": "ftp-panel", "label": "FTP", "url": "ftp://files.example.com/"},
        )

        # Assert
        assert resp.status_code == 422

    def test_register_builtin_id_returns_422(self, client_runtime_panels):
        """Using a built-in panel id (e.g. 'ariel') for registration is rejected with 422."""
        # Arrange — "ariel" is in BUILTIN_PANELS

        # Act
        resp = client_runtime_panels.post(
            "/api/panels/register",
            json={"id": "ariel", "label": "ARIEL", "url": "http://grafana.lan:3000"},
        )

        # Assert
        assert resp.status_code == 422

    @pytest.mark.parametrize("panel_id", ["überblick", "..", "beam/viewer"])
    def test_register_id_the_panel_could_not_be_reached_by_returns_422(
        self, client_runtime_panels, panel_id
    ):
        """A registration arrives after the boot gate and reaches the same places.

        The id lands in the proxy's ``/panel/<id>`` paths and in the
        forwarded-prefix header of every hop made for this panel, escaped for
        neither, so an id outside that class would register cleanly and then
        fail every request the panel serves. The refusal names the id, since
        the caller chose it.
        """
        # Act — no getaddrinfo patch: the id is judged before the URL is read.
        resp = client_runtime_panels.post(
            "/api/panels/register",
            json={"id": panel_id, "label": "X", "url": "http://grafana.lan:3000"},
        )

        # Assert
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert repr(panel_id) in detail
        assert "[A-Za-z0-9][A-Za-z0-9._-]*" in detail

    def test_register_ordinary_id_still_registers(self, client_runtime_panels):
        """The class admits the ids a registration actually uses."""
        # Act
        with patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR):
            resp = client_runtime_panels.post(
                "/api/panels/register",
                json={"id": "beam_viewer.2", "label": "BEAM", "url": "http://grafana.lan:3000"},
            )

        # Assert
        assert resp.status_code == 200
        assert [cp["id"] for cp in client_runtime_panels.app.state.custom_panels] == [
            "beam_viewer.2"
        ]

    def test_register_host_not_in_allowlist_returns_422(self, client_runtime_panels_allowlist):
        """A host not in runtime_panel_allowlist is rejected with 422 even with a LAN address."""
        # Arrange — allowlist contains only "grafana.lan"; "other-host.lan" is not listed

        # Act
        with patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR):
            resp = client_runtime_panels_allowlist.post(
                "/api/panels/register",
                json={
                    "id": "other",
                    "label": "OTHER",
                    "url": "http://other-host.lan:3000",
                },
            )

        # Assert
        assert resp.status_code == 422

    def test_register_host_in_allowlist_succeeds(self, client_runtime_panels_allowlist):
        """A host present in runtime_panel_allowlist is accepted when the address is not blocked."""
        # Arrange — allowlist contains "grafana.lan"; URL host matches

        # Act
        with patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR):
            resp = client_runtime_panels_allowlist.post(
                "/api/panels/register",
                json={
                    "id": "grafana",
                    "label": "GRAFANA",
                    "url": "http://grafana.lan:3000",
                },
            )

        # Assert
        assert resp.status_code == 200

    def test_register_deploy_host_own_address_returns_422(self, client_runtime_panels):
        """A host resolving to one of THIS host's interface addresses is rejected.

        The panel proxy fetches server-side from inside the deployment, so a
        panel pointed at the deploy host's own LAN address is a route back into
        the deployment's web terminals — and under the ``open`` auth posture
        nginx injects a per-user terminal secret on every request, making that
        a route into a neighbour's terminal. Loopback alone does not catch it:
        the address is an ordinary routable LAN address.
        """
        # Arrange — the host resolves to 10.0.0.5, which is also ours.
        own = frozenset({ipaddress.ip_address("10.0.0.5")})

        # Act
        with (
            patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR),
            patch(_HOST_ADDRS_TARGET, return_value=own),
        ):
            resp = client_runtime_panels.post(
                "/api/panels/register",
                json={"id": "selfie", "label": "SELF", "url": "http://myself.lan:3000"},
            )

        # Assert
        assert resp.status_code == 422
        assert "deployment host itself" in resp.json()["detail"]

    def test_register_ordinary_private_lan_host_still_succeeds(self, client_runtime_panels):
        """A genuine private-LAN dashboard on another host is still accepted.

        The deploy-host check must not degrade into a blanket RFC1918 refusal:
        real Grafana dashboards live on 10/8, and only THIS host's own
        addresses are off limits.
        """
        # Arrange — we are 192.168.1.20; the panel host is 10.0.0.5.
        own = frozenset({ipaddress.ip_address("192.168.1.20")})

        # Act
        with (
            patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR),
            patch(_HOST_ADDRS_TARGET, return_value=own),
        ):
            resp = client_runtime_panels.post(
                "/api/panels/register",
                json={"id": "grafana", "label": "GRAFANA", "url": "http://grafana.lan:3000"},
            )

        # Assert
        assert resp.status_code == 200

    def test_register_loopback_rejected_when_interface_probe_fails_open(
        self, client_runtime_panels
    ):
        """Loopback stays refused even when the interface probe yields nothing.

        The probe fails open (an offline host with no default route and an
        unresolvable hostname returns an empty set), so the categorical
        loopback / link-local / unspecified checks must stand on their own.
        """
        # Arrange — probe returns nothing at all.

        # Act
        with (
            patch(_GETADDRINFO_TARGET, return_value=_LOOPBACK_ADDR),
            patch(_HOST_ADDRS_TARGET, return_value=frozenset()),
        ):
            resp = client_runtime_panels.post(
                "/api/panels/register",
                json={"id": "internal", "label": "INT", "url": "http://grafana.lan:3000"},
            )

        # Assert
        assert resp.status_code == 422


class TestConfigDefinedPanelReservation:
    """A config-defined panel id is reserved against runtime registration.

    Without this, an agent (via the workspace ``register_panel`` MCP tool) could
    register ``id="events"``; the remove-then-append would silently repoint the
    config-defined EVENTS panel at an attacker URL, and the proxy's server-side
    token injection would then hand ``EVENT_DISPATCHER_TOKEN`` to that URL.
    """

    @pytest.fixture
    def app_and_client(self, workspace_dir):
        yield from _make_client_runtime_with_config_events(workspace_dir)

    def test_register_config_defined_id_returns_422(self, app_and_client):
        """Registering a config-defined id (events) is rejected, like a built-in."""
        _app, client = app_and_client
        with patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR):
            resp = client.post(
                "/api/panels/register",
                json={"id": "events", "label": "PWNED", "url": "http://attacker.lan:3000"},
            )
        assert resp.status_code == 422

    def test_config_entry_survives_squat_attempt(self, app_and_client):
        """The reserved id's config entry is never mutated — url/label unchanged.

        Guards the remove-then-append: a status-only assertion would pass even if
        the entry were replaced-then-rejected, so assert the entry itself.
        """
        app, client = app_and_client
        with patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR):
            client.post(
                "/api/panels/register",
                json={"id": "events", "label": "PWNED", "url": "http://attacker.lan:3000"},
            )
        events = [cp for cp in app.state.custom_panels if cp["id"] == "events"]
        assert len(events) == 1
        assert events[0]["url"] == "http://localhost:8020"  # original, not attacker's
        assert events[0]["label"] == "EVENTS"


# ---- Sidecar panels ----


@pytest.fixture
def ready_sidecar():
    """A script whose sidecar comes up: preflight passes, readiness returns."""
    return SidecarScript(preflight_error=None)


@pytest.fixture
def client_with_sidecar(workspace_dir, ready_sidecar):
    """Client with the sidecar panel enabled and its launch succeeding."""
    yield from _make_client(
        workspace_dir,
        enabled_panels={SIDECAR_ID} | set(UNIVERSAL_PANELS),
        sidecar_script=ready_sidecar,
    )


def _stub_app():
    """The minimum app surface the two launch helpers write to."""
    return SimpleNamespace(state=SimpleNamespace(sidecars={}, panel_auth_headers={}))


class TestSidecarReadyTimeout:
    """How long a sidecar launch waits, and where that number comes from."""

    KEY = "web.sidecar_ready_timeout_s"

    def _warnings_naming_the_key(self, caplog):
        return [
            r for r in caplog.records if r.levelname == "WARNING" and self.KEY in r.getMessage()
        ]

    def _launch(self, workspace_dir, value, script):
        for client in _make_client(
            workspace_dir,
            enabled_panels={SIDECAR_ID} | set(UNIVERSAL_PANELS),
            sidecar_script=script,
            config_values={self.KEY: value},
        ):
            return getattr(client.app.state, f"{SIDECAR_ID}_server_url")

    def test_an_unset_key_waits_the_default(self, client_with_sidecar, ready_sidecar):
        assert client_with_sidecar.app.state.sidecar_ready_timeout_s == 60.0
        assert ready_sidecar.waited == [60.0]

    def test_the_configured_wait_reaches_the_sidecar(self, workspace_dir, ready_sidecar):
        url = self._launch(workspace_dir, 240, ready_sidecar)

        assert ready_sidecar.waited == [240.0]
        assert url == ready_sidecar.url

    def test_a_numeric_string_is_read_as_seconds(self, workspace_dir, ready_sidecar, caplog):
        self._launch(workspace_dir, "90", ready_sidecar)

        assert ready_sidecar.waited == [90.0]
        assert self._warnings_naming_the_key(caplog) == []

    @pytest.mark.parametrize(
        "value",
        [
            0,
            -5,
            True,
            "soon",
            float("nan"),
            float("inf"),
            [60],
            pytest.param(10**400, id="10**400"),
        ],
        ids=repr,
    )
    def test_an_unusable_value_is_refused_by_name(
        self, workspace_dir, ready_sidecar, caplog, value
    ):
        url = self._launch(workspace_dir, value, ready_sidecar)

        assert ready_sidecar.waited == [60.0]
        assert url == ready_sidecar.url
        warnings = self._warnings_naming_the_key(caplog)
        assert len(warnings) == 1
        assert repr(value) in warnings[0].getMessage()

    def test_a_null_value_waits_the_default_quietly(self, workspace_dir, ready_sidecar, caplog):
        self._launch(workspace_dir, None, ready_sidecar)

        assert ready_sidecar.waited == [60.0]
        assert self._warnings_naming_the_key(caplog) == []

    def test_a_config_read_that_raises_waits_the_default(
        self, workspace_dir, ready_sidecar, caplog
    ):
        def _reader(path, default=None, _config_path=None):
            if path == self.KEY:
                raise RuntimeError("config unreadable")
            return default

        with (
            patch("osprey.utils.config.get_config_value", side_effect=_reader),
            patch(_SIDECAR_FACTORY_TARGET, _stub_sidecar_class(ready_sidecar)),
            patch(
                "osprey.interfaces.web_terminal.app._load_web_config",
                return_value={"watch_dir": str(workspace_dir)},
            ),
            patch(
                "osprey.interfaces.web_terminal.app._load_panel_config",
                return_value=({SIDECAR_ID} | set(UNIVERSAL_PANELS), [], None),
            ),
        ):
            with TestClient(create_app(shell_command="echo")) as client:
                assert client.get("/api/panels").status_code == 200

        assert ready_sidecar.waited == [60.0]
        assert len(self._warnings_naming_the_key(caplog)) == 1

    def test_every_launch_reads_the_wait_from_app_state(self, client_with_sidecar, ready_sidecar):
        """The seam the relaunch path relies on: one attribute, read per launch."""
        app = client_with_sidecar.app
        app.state.sidecar_ready_timeout_s = 7.5

        asyncio.run(web_terminal_app._launch_sidecar(app, SIDECAR_ID))
        assert ready_sidecar.waited[-1] == 7.5

        asyncio.run(web_terminal_app._relaunch_sidecar(app, SIDECAR_ID, None))
        assert ready_sidecar.waited[-1] == 7.5


class TestSidecarLaunchRouting:
    """Which launcher each built-in panel id reaches.

    A sidecar is in no companion registry, so feeding its id to the companion
    launcher raises on the registry-key lookup rather than starting anything.
    """

    def test_no_sidecar_id_reaches_the_companion_launcher(self, monkeypatch):
        keys: list[str] = []
        monkeypatch.setattr(
            web_terminal_app, "_launch_panel_server", lambda app, key: keys.append(key)
        )

        web_terminal_app._launch_enabled_panel_servers(_stub_app(), set(BUILTIN_PANELS))

        assert len(keys) == len(BUILTIN_PANELS) - len(SIDECAR_PANELS)
        assert set(keys).isdisjoint(SIDECAR_PANELS)

    def test_every_sidecar_id_reaches_the_sidecar_launcher(self, monkeypatch):
        launched: list[str] = []

        async def _record(_app, panel_id):
            launched.append(panel_id)

        monkeypatch.setattr(web_terminal_app, "_launch_sidecar", _record)

        asyncio.run(web_terminal_app._launch_enabled_sidecars(_stub_app(), set(BUILTIN_PANELS)))

        assert launched == sorted(SIDECAR_PANELS)

    def test_a_disabled_sidecar_is_not_launched(self, monkeypatch):
        launched: list[str] = []

        async def _record(_app, panel_id):
            launched.append(panel_id)

        monkeypatch.setattr(web_terminal_app, "_launch_sidecar", _record)

        asyncio.run(web_terminal_app._launch_enabled_sidecars(_stub_app(), set(UNIVERSAL_PANELS)))

        assert launched == []


class TestSidecarPanelAvailability:
    """What a launched — or failed — sidecar publishes, and what the panel says."""

    def test_a_ready_sidecar_publishes_its_url_and_credential(
        self, client_with_sidecar, ready_sidecar
    ):
        state = client_with_sidecar.app.state

        assert getattr(state, f"{SIDECAR_ID}_server_url") == ready_sidecar.url
        assert state.panel_auth_headers[SIDECAR_ID] == ready_sidecar.auth_headers
        assert state.sidecars[SIDECAR_ID] is not None
        assert ready_sidecar.spawned == 1
        assert ready_sidecar.waited == [web_terminal_app.DEFAULT_SIDECAR_READY_TIMEOUT_S]

    def test_the_sidecar_is_built_from_the_app_s_own_root_prefix_and_theme(
        self, client_with_sidecar, ready_sidecar
    ):
        state = client_with_sidecar.app.state
        (shared_root, outer_prefix, pinned_mode) = ready_sidecar.constructed[0]

        # The sidecar shares the agent-data root every other child of this
        # server is stamped with, not a directory of its own.
        assert shared_root == Path(resolve_agent_data_root(client_with_sidecar.app))
        assert outer_prefix == ""
        assert pinned_mode == state.web_theme_mode

    def test_the_route_reports_the_proxy_url_when_the_sidecar_is_ready(self, client_with_sidecar):
        resp = client_with_sidecar.get(f"/api/{SIDECAR_ID}-server")

        assert resp.status_code == 200
        assert resp.json() == {
            "url": f"/panel/{SIDECAR_ID}",
            "available": True,
            "state": "running",
            "message": None,
        }

    def test_the_route_reports_unavailable_when_the_sidecar_did_not_start(self, client_all_panels):
        resp = client_all_panels.get(f"/api/{SIDECAR_ID}-server")

        assert resp.status_code == 200
        assert resp.json() == {
            "url": None,
            "available": False,
            "state": "failed",
            "message": "JUPYTER failed to start: stub sidecar: not launched",
        }

    def test_a_failed_launch_publishes_neither_url_nor_credential(self, client_all_panels):
        state = client_all_panels.app.state

        assert getattr(state, f"{SIDECAR_ID}_server_url") is None
        assert SIDECAR_ID not in state.panel_auth_headers
        assert SIDECAR_ID not in state.sidecars

    def test_a_failed_launch_records_its_reason_on_disk(self, client_all_panels):
        state = client_all_panels.app.state
        record = json.loads((state.panel_status_dir / f"{SIDECAR_ID}.json").read_text())

        assert record["state"] == "failed"
        assert record["reason"] == "stub sidecar: not launched"
        assert record["recorded_at"]

    def test_a_ready_launch_records_running(self, client_with_sidecar):
        state = client_with_sidecar.app.state
        record = json.loads((state.panel_status_dir / f"{SIDECAR_ID}.json").read_text())

        assert record["state"] == "running"
        assert record["reason"] is None
        assert state.sidecar_status[SIDECAR_ID].state == "running"

    def test_a_sidecar_that_never_answers_is_stopped_and_publishes_nothing(
        self, workspace_dir, caplog
    ):
        """The one path that leaves a live process behind if ``stop()`` is dropped.

        Preflight passes and the process is spawned, so a readiness failure has
        a server to reap. ``wait_ready`` quotes the stderr tail in its own
        message, which is why the warning must carry it exactly once.
        """
        tail = "OSError: address already in use"
        script = SidecarScript(
            preflight_error=None,
            stderr_tail=tail,
            wait_ready_error=f"did not answer within 60 s\n{tail}",
        )

        with caplog.at_level("WARNING", logger=web_terminal_app.__name__):
            gen = _make_client(workspace_dir, enabled_panels={SIDECAR_ID}, sidecar_script=script)
            client = next(gen)
            state = client.app.state

            assert script.spawned == 1
            assert script.stopped == 1
            assert getattr(state, f"{SIDECAR_ID}_server_url") is None
            assert SIDECAR_ID not in state.panel_auth_headers
            assert SIDECAR_ID not in state.sidecars

            with pytest.raises(StopIteration):
                next(gen)

        # Shutdown must not reap it a second time — it was never published.
        assert script.stopped == 1
        failure = [
            r.getMessage() for r in caplog.records if "sidecar failed to start" in r.getMessage()
        ]
        assert len(failure) == 1
        assert failure[0].count(tail) == 1

    def test_a_failing_preflight_is_logged_with_the_stderr_tail(self, workspace_dir, caplog):
        script = SidecarScript(preflight_error="no interpreter", stderr_tail="Traceback: boom")

        with caplog.at_level("WARNING", logger=web_terminal_app.__name__):
            for _client in _make_client(
                workspace_dir, enabled_panels={SIDECAR_ID}, sidecar_script=script
            ):
                pass

        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        failure = [m for m in warnings if "sidecar failed to start" in m]
        assert len(failure) == 1
        assert SIDECAR_ID in failure[0]
        assert "no interpreter" in failure[0]
        assert "Traceback: boom" in failure[0]

    def test_shutdown_removes_the_status_record(self, workspace_dir, ready_sidecar):
        client_gen = _make_client(
            workspace_dir, enabled_panels={SIDECAR_ID}, sidecar_script=ready_sidecar
        )
        client = next(client_gen)
        record = client.app.state.panel_status_dir / f"{SIDECAR_ID}.json"
        assert record.is_file()

        with pytest.raises(StopIteration):
            next(client_gen)

        assert not record.exists()

    def test_shutdown_stops_a_launched_sidecar(self, workspace_dir, ready_sidecar):
        client_gen = _make_client(
            workspace_dir, enabled_panels={SIDECAR_ID}, sidecar_script=ready_sidecar
        )
        next(client_gen)
        assert ready_sidecar.stopped == 0

        with pytest.raises(StopIteration):
            next(client_gen)

        assert ready_sidecar.stopped == 1


class TestSidecarCredentialThroughTheProxy:
    """The launch credential is injected on the proxy hop, not held by the browser."""

    def test_the_backend_receives_the_published_authorization_header(self, client_all_panels):
        seen: list[str | None] = []

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                seen.append(self.headers.get("authorization"))
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok": true}')

            def log_message(self, *args):
                """Keep the handler out of the test's stderr."""

        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            state = client_all_panels.app.state
            setattr(state, f"{SIDECAR_ID}_server_url", f"http://127.0.0.1:{server.server_port}")
            state.panel_auth_headers[SIDECAR_ID] = {"authorization": "Bearer stub-token"}

            resp = client_all_panels.get(f"/panel/{SIDECAR_ID}/api/status")
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        assert resp.status_code == 200
        assert seen == ["Bearer stub-token"]


class TestSidecarExitsLater:
    """A sidecar that dies after it was published is retracted and recorded as failed."""

    def test_an_exit_retracts_the_url_and_the_credential(self, client_with_sidecar, ready_sidecar):
        state = client_with_sidecar.app.state
        assert client_with_sidecar.get(f"/api/{SIDECAR_ID}-server").json()["available"] is True

        ready_sidecar.exit_status = 1
        # The real sidecar fires this from its watcher thread; the hook hands
        # the loop the retraction, so the route may need one more turn.
        threading.Thread(target=state.sidecars[SIDECAR_ID].on_exit).start()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            resp = client_with_sidecar.get(f"/api/{SIDECAR_ID}-server")
            if resp.json()["available"] is False:
                break
            time.sleep(0.05)

        assert resp.json() == {
            "url": None,
            "available": False,
            "state": "failed",
            "message": "JUPYTER failed to start: exited with status 1",
        }
        assert getattr(state, f"{SIDECAR_ID}_server_url") is None
        assert SIDECAR_ID not in state.panel_auth_headers
        # It stays registered so shutdown still removes its per-launch state.
        assert state.sidecars[SIDECAR_ID] is not None
        assert ready_sidecar.stopped == 0

    def test_an_exit_of_a_replaced_sidecar_retracts_nothing(self, client_with_sidecar):
        state = client_with_sidecar.app.state
        stale_hook = state.sidecars[SIDECAR_ID].on_exit
        state.sidecars[SIDECAR_ID] = object()

        thread = threading.Thread(target=stale_hook)
        thread.start()
        thread.join()
        # One loop turn for the handed-over callback, then a few more requests.
        for _ in range(5):
            resp = client_with_sidecar.get(f"/api/{SIDECAR_ID}-server")
            time.sleep(0.02)

        assert resp.json()["available"] is True
        assert resp.json()["state"] == "running"
        assert SIDECAR_ID in state.panel_auth_headers


def _start(client, panel_id=SIDECAR_ID):
    return client.post(f"/api/panels/{panel_id}/start")


def _settle(client, predicate, timeout=5.0):
    """Poll the sidecar's config route until *predicate* holds; return the last body."""
    deadline = time.monotonic() + timeout
    while True:
        body = client.get(f"/api/{SIDECAR_ID}-server").json()
        if predicate(body) or time.monotonic() >= deadline:
            return body
        time.sleep(0.02)


@pytest.fixture
def failed_sidecar():
    """A script whose first launch fails preflight; clear the error to let a retry succeed."""
    return SidecarScript(preflight_error="first boom")


@pytest.fixture
def client_with_failed_sidecar(workspace_dir, failed_sidecar):
    yield from _make_client(
        workspace_dir,
        enabled_panels={SIDECAR_ID} | set(UNIVERSAL_PANELS),
        sidecar_script=failed_sidecar,
    )


class TestSidecarStartsAgain:
    """A start request relaunches a failed sidecar, one attempt at a time."""

    def test_a_start_request_relaunches_a_failed_sidecar(
        self, client_with_failed_sidecar, failed_sidecar
    ):
        failed_sidecar.preflight_error = None

        resp = _start(client_with_failed_sidecar)

        assert resp.status_code == 202
        assert resp.json()["state"] == "starting"
        assert resp.json()["message"] == "JUPYTER is starting"
        body = _settle(client_with_failed_sidecar, lambda b: b["state"] != "starting")
        assert body["available"] is True
        assert body["state"] == "running"
        assert len(failed_sidecar.constructed) == 2

    def test_a_second_request_while_one_runs_starts_nothing_more(
        self, client_with_failed_sidecar, failed_sidecar
    ):
        failed_sidecar.preflight_error = None
        failed_sidecar.hold = threading.Event()

        first = _start(client_with_failed_sidecar)
        second = _start(client_with_failed_sidecar)

        assert first.status_code == 202
        assert second.status_code == 202
        assert second.json()["state"] == "starting"
        failed_sidecar.hold.set()
        body = _settle(client_with_failed_sidecar, lambda b: b["state"] == "running")
        assert body["available"] is True
        assert failed_sidecar.spawned == 1
        assert len(failed_sidecar.constructed) == 2

    def test_a_request_for_a_running_sidecar_changes_nothing(
        self, client_with_sidecar, ready_sidecar
    ):
        resp = _start(client_with_sidecar)

        assert resp.status_code == 200
        assert resp.json()["state"] == "running"
        assert resp.json()["available"] is True
        assert ready_sidecar.spawned == 1

    def test_the_retry_waits_as_long_as_the_startup(self, workspace_dir):
        script = SidecarScript(preflight_error=None, wait_ready_error="did not answer")
        for client in _make_client(
            workspace_dir, enabled_panels={SIDECAR_ID}, sidecar_script=script
        ):
            script.wait_ready_error = None
            assert _start(client).status_code == 202
            _settle(client, lambda b: b["state"] == "running")

        timeout = web_terminal_app.DEFAULT_SIDECAR_READY_TIMEOUT_S
        assert script.waited == [timeout, timeout]

    def test_the_dead_sidecar_is_stopped_before_its_replacement_starts(
        self, client_with_sidecar, ready_sidecar
    ):
        state = client_with_sidecar.app.state
        ready_sidecar.exit_status = 1
        threading.Thread(target=state.sidecars[SIDECAR_ID].on_exit).start()
        _settle(client_with_sidecar, lambda b: b["state"] == "failed")

        assert _start(client_with_sidecar).status_code == 202
        body = _settle(client_with_sidecar, lambda b: b["state"] == "running")

        assert body["available"] is True
        assert ready_sidecar.events == ["spawn", "stop", "spawn"]

    def test_a_retry_that_fails_again_records_the_new_reason(
        self, client_with_failed_sidecar, failed_sidecar
    ):
        failed_sidecar.preflight_error = "second boom"

        assert _start(client_with_failed_sidecar).status_code == 202
        body = _settle(client_with_failed_sidecar, lambda b: b["state"] == "failed")

        assert body["message"] == "JUPYTER failed to start: second boom"
        state = client_with_failed_sidecar.app.state
        record = json.loads((state.panel_status_dir / f"{SIDECAR_ID}.json").read_text())
        assert record["reason"] == "second boom"
        assert SIDECAR_ID not in state.sidecars

    def test_a_start_request_for_a_companion_panel_is_404(self, client_all_panels):
        resp = _start(client_all_panels, "ariel")

        assert resp.status_code == 404
        assert resp.json()["detail"] == "ariel is not a panel this terminal starts"

    def test_a_start_request_for_an_unknown_panel_is_404(self, client_all_panels):
        assert _start(client_all_panels, "no-such-panel").status_code == 404

    def test_a_start_request_for_a_disabled_sidecar_is_404(self, client):
        assert _start(client).status_code == 404

    def test_shutdown_during_a_start_stops_the_starting_sidecar_and_clears_the_record(
        self, workspace_dir, failed_sidecar
    ):
        client_gen = _make_client(
            workspace_dir, enabled_panels={SIDECAR_ID}, sidecar_script=failed_sidecar
        )
        client = next(client_gen)
        record = client.app.state.panel_status_dir / f"{SIDECAR_ID}.json"
        failed_sidecar.preflight_error = None
        failed_sidecar.hold = threading.Event()
        assert _start(client).status_code == 202
        deadline = time.monotonic() + 5.0
        while not failed_sidecar.waited and time.monotonic() < deadline:
            time.sleep(0.02)
        assert failed_sidecar.waited, "the retry never reached wait_ready"

        with pytest.raises(StopIteration):
            next(client_gen)

        assert failed_sidecar.stopped >= 1
        # The first attempt failed preflight and was stopped; the retry spawned.
        assert failed_sidecar.events[:3] == ["stop", "spawn", "stop"]
        assert not record.exists()
