"""Tests for the panel_tools MCP tools.

Covers:
  - list_panels returns enabled built-in and custom panels with their labels,
    the dock's full ``layout``, its ``rev`` and its freshness
  - list_panels handles web terminal being unreachable
  - every on-screen verb relays exactly one dock op through ``panel_op``
    (open_panel -> open, drop_panel -> drop, close_panel_tab -> close_tab,
    close_panel_tile -> close_tile, open_panel_set -> open_set,
    arrange_workspace -> open_arrangement or open_set as a layout), with
    ``base_rev`` only for the ops that name tile ids
  - each returns the resulting layout and rev, or raises a structured error
    whose error_type is the named refusal
  - panel-id layouts are sent with member indexes
  - open_panel refuses a url for an agent-hop panel (CANOPY), a url under
    /panel/canopy/ for any panel, the CANOPY sidecar's own origin, and a url
    the navigation guard refuses, before anything is relayed
  - add_panel_to_rail / remove_panel_from_rail validate ids before toggling
    rail membership
  - register_panel maps the register helper's outcomes to its response
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from osprey.interfaces.web_terminal.dock_layout import layout_keys
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

_MODULE = "osprey.mcp_server.workspace.tools.panel_tools"


@pytest.fixture
def _mock_web_terminal_url():
    """Pin the base URL used by the shared http.fetch_panels helper.

    The panel tools do not resolve the URL themselves — they delegate the
    ``GET /api/panels`` read to ``osprey.mcp_server.http.fetch_panels``, so the
    patch has to land there for tests that stub ``urllib.request.urlopen``.
    """
    with patch("osprey.mcp_server.http.web_terminal_url", return_value="http://127.0.0.1:10100"):
        yield


def _get_list_panels():
    from osprey.mcp_server.workspace.tools.panel_tools import list_panels

    return get_tool_fn(list_panels)


def _get_open_panel():
    from osprey.mcp_server.workspace.tools.panel_tools import open_panel

    return get_tool_fn(open_panel)


def _tool(name: str):
    from osprey.mcp_server.workspace.tools import panel_tools

    return get_tool_fn(getattr(panel_tools, name))


#: A dock-form layout as the route returns it: two tiles, the terminal in one.
_LAYOUT = {
    "split": "row",
    "items": [
        {"tile": "t1", "tabs": ["ariel", "lattice"], "showing": "lattice", "size": 0.6},
        {"tile": "t2", "tabs": ["terminal"], "showing": "terminal", "size": 0.4},
    ],
}

#: ARTIFACTS beside a LATTICE tile; ariel is on the rail but has no tile.
_OCCUPIED = {
    "split": "row",
    "items": [
        {"tile": "t1", "tabs": ["terminal"], "showing": "terminal", "size": 0.3},
        {"tile": "t2", "tabs": ["lattice"], "showing": "lattice", "size": 0.4},
        {"tile": "t3", "tabs": ["artifacts"], "showing": "artifacts", "size": 0.3},
    ],
}


def _service_keys(layout) -> list[str]:
    """The service keys of a dock ``layout`` in reading order, terminal excluded."""
    return [key for key in layout_keys(layout) if key != "terminal"]


def _applied(layout=None, rev=7) -> dict:
    """``panel_op``'s outcome when the dock applied the op."""
    return {"ok": True, "status": 200, "layout": layout or _LAYOUT, "rev": rev}


def _refused(reason: str, detail: str = "", status: int | None = 409, **extra) -> dict:
    """``panel_op``'s outcome for a named refusal."""
    return {"ok": False, "status": status, "reason": reason, "detail": detail, **extra}


class TestListPanels:
    @pytest.mark.asyncio
    async def test_returns_enabled_panels(self, _mock_web_terminal_url):
        """Built-in enabled panels are returned with correct labels."""
        fn = _get_list_panels()

        api_response = json.dumps(
            {
                "enabled": ["artifacts", "ariel", "lattice"],
                "custom": [],
                "labels": {"artifacts": "WORKSPACE", "ariel": "ARIEL", "lattice": "LATTICE"},
                "visible": ["artifacts", "lattice"],
                "active": None,
            }
        ).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = api_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = extract_response_dict(await fn())

        assert result["status"] == "success"
        assert result["active"] is None
        ids = [p["id"] for p in result["panels"]]
        assert ids == ["artifacts", "ariel", "lattice"]
        labels = {p["id"]: p["label"] for p in result["panels"]}
        assert labels["artifacts"] == "WORKSPACE"
        assert labels["ariel"] == "ARIEL"
        assert labels["lattice"] == "LATTICE"
        visible = {p["id"]: p["visible"] for p in result["panels"]}
        assert visible["artifacts"] is True
        assert visible["ariel"] is False
        assert visible["lattice"] is True

    @pytest.mark.asyncio
    async def test_includes_custom_panels(self, _mock_web_terminal_url):
        """Custom panels from config are appended to the list."""
        fn = _get_list_panels()

        api_response = json.dumps(
            {
                "enabled": ["artifacts"],
                "custom": [{"id": "my-panel", "label": "MY PANEL", "url": "/panel/my-panel"}],
                "labels": {"artifacts": "WORKSPACE"},
                "visible": ["artifacts", "my-panel"],
                "active": None,
            }
        ).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = api_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = extract_response_dict(await fn())

        assert result["status"] == "success"
        assert len(result["panels"]) == 2
        custom = result["panels"][1]
        assert custom["id"] == "my-panel"
        assert custom["label"] == "MY PANEL"

    @pytest.mark.asyncio
    async def test_custom_panel_label_fallback(self, _mock_web_terminal_url):
        """Custom panel without explicit label falls back to id.upper()."""
        fn = _get_list_panels()

        api_response = json.dumps(
            {
                "enabled": ["artifacts"],
                "custom": [{"id": "my-grafana", "url": "/panel/my-grafana"}],
                "labels": {"artifacts": "WORKSPACE"},
                "visible": ["artifacts"],
                "active": "artifacts",
            }
        ).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = api_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = extract_response_dict(await fn())

        assert result["status"] == "success"
        custom = result["panels"][1]
        assert custom["id"] == "my-grafana"
        assert custom["label"] == "MY-GRAFANA"

    @pytest.mark.asyncio
    async def test_web_terminal_unreachable(self, _mock_web_terminal_url):
        """Returns error when web terminal is not running."""
        fn = _get_list_panels()

        with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError("refused")):
            result = extract_response_dict(await fn())

        assert result["status"] == "error"
        assert "not running" in result["message"]

    @pytest.mark.asyncio
    async def test_list_panels_reports_layout_independently_of_rail_membership(
        self, _mock_web_terminal_url
    ):
        """Rail membership and on-screen occupancy are reported as separate facts."""
        fn = _get_list_panels()

        api_response = json.dumps(
            {
                "enabled": ["artifacts", "ariel", "lattice"],
                "custom": [],
                "labels": {},
                # ariel is in the rail but has no tile; lattice has a tile.
                "visible": ["artifacts", "ariel", "lattice"],
                "active": "lattice",
                "layout": _OCCUPIED,
                "rev": 2,
                "layout_age_s": 12.0,
                "layout_dock": True,
            }
        ).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = api_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = extract_response_dict(await fn())

        on_screen = _service_keys(result["layout"])
        assert on_screen == ["lattice", "artifacts"]
        assert result["layout_age_s"] == 12.0
        assert result["layout_dock"] is True
        visible = {p["id"]: p["visible"] for p in result["panels"]}
        assert visible["ariel"] is True and "ariel" not in on_screen

    @pytest.mark.asyncio
    async def test_list_panels_freshness_is_null_when_no_client_has_reported(
        self, _mock_web_terminal_url
    ):
        """Never-reported is all-null — never a layout, which would claim a known screen."""
        fn = _get_list_panels()

        api_response = json.dumps(
            {
                "enabled": ["artifacts"],
                "custom": [],
                "labels": {},
                "visible": ["artifacts"],
                "active": None,
                # The route's never-reported state: all three null.
                "layout": None,
                "layout_age_s": None,
                "layout_dock": None,
            }
        ).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = api_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = extract_response_dict(await fn())

        assert result["layout"] is None
        assert result["layout_age_s"] is None
        assert result["layout_dock"] is None

    @pytest.mark.asyncio
    async def test_list_panels_passes_through_unknown_occupancy(self, _mock_web_terminal_url):
        """A dock-less client reports unknown occupancy (null), never known-empty."""
        fn = _get_list_panels()

        api_response = json.dumps(
            {
                "enabled": ["artifacts"],
                "custom": [],
                "labels": {},
                "visible": ["artifacts"],
                "active": "artifacts",
                # Server maps a dock:false report to unknown occupancy.
                "layout": None,
                "layout_age_s": 3.0,
                "layout_dock": False,
            }
        ).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = api_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = extract_response_dict(await fn())

        assert result["layout"] is None
        assert result["layout_dock"] is False
        assert result["layout_age_s"] == 3.0


class TestListPanelsLayout:
    @pytest.mark.asyncio
    async def test_list_panels_returns_layout_and_rev(self):
        """The full dock layout, its rev and freshness pass through."""
        data = {
            "enabled": ["ariel", "lattice"],
            "custom": [],
            "visible": ["ariel", "lattice"],
            "active": "lattice",
            "layout": _LAYOUT,
            "rev": 12,
            "layout_age_s": 1.5,
            "layout_dock": True,
            "presets": [{"name": "Ops", "panels": ["ariel"]}],
        }
        with patch(f"{_MODULE}.fetch_panels", return_value=data):
            result = extract_response_dict(await _tool("list_panels")())

        assert result["layout"] == _LAYOUT
        assert result["rev"] == 12
        assert result["layout_age_s"] == 1.5
        assert result["layout_dock"] is True
        assert result["presets"] == [{"name": "Ops", "panels": ["ariel"]}]

    @pytest.mark.asyncio
    async def test_list_panels_layout_is_null_before_any_report(self):
        data = {"enabled": ["ariel"], "custom": [], "visible": [], "layout": None, "rev": 0}
        with patch(f"{_MODULE}.fetch_panels", return_value=data):
            result = extract_response_dict(await _tool("list_panels")())

        assert result["layout"] is None
        assert result["rev"] == 0


class TestOpenPanel:
    @pytest.mark.asyncio
    async def test_open_panel_relays_open_and_returns_layout_and_rev(self):
        """open_panel is one dock op ``open {member}``, member-only: no base_rev."""
        with patch(f"{_MODULE}.panel_op", return_value=_applied(rev=9)) as mock_op:
            result = extract_response_dict(await _get_open_panel()("ariel"))

        assert result == {
            "status": "success",
            "panel": "ariel",
            "layout": _LAYOUT,
            "popouts": [],
            "rev": 9,
        }
        mock_op.assert_called_once_with("open", {"member": "ariel"}, base_rev=None)

    @pytest.mark.asyncio
    async def test_open_panel_passes_url(self):
        url = "http://127.0.0.1:10300/#draft"
        with patch(f"{_MODULE}.panel_op", return_value=_applied()) as mock_op:
            result = extract_response_dict(await _get_open_panel()("ariel", url=url))

        assert result["status"] == "success"
        mock_op.assert_called_once_with("open", {"member": "ariel", "url": url}, base_rev=None)

    @pytest.mark.asyncio
    async def test_open_panel_custom_panel(self):
        with patch(f"{_MODULE}.panel_op", return_value=_applied()) as mock_op:
            result = extract_response_dict(await _get_open_panel()("my-grafana"))

        assert result["panel"] == "my-grafana"
        mock_op.assert_called_once_with("open", {"member": "my-grafana"}, base_rev=None)

    @pytest.mark.asyncio
    async def test_open_panel_route_refusal_is_named(self):
        refused = _refused("unknown_member", "unknown panel 'nope'", status=422, rev=3)
        with patch(f"{_MODULE}.panel_op", return_value=refused):
            with assert_raises_error(error_type="unknown_member") as ctx:
                await _get_open_panel()("nope")

        envelope = ctx["envelope"]
        assert "open_panel refused" in envelope["error_message"]
        assert "unknown panel 'nope'" in envelope["error_message"]
        assert envelope["details"] == {
            "op": "open",
            "reason": "unknown_member",
            "detail": "unknown panel 'nope'",
            "rev": 3,
        }

    @pytest.mark.asyncio
    async def test_open_panel_web_terminal_unreachable(self):
        down = _refused("no_web_terminal", "Web Terminal is not running.", status=None)
        with patch(f"{_MODULE}.panel_op", return_value=down):
            with assert_raises_error(error_type="no_web_terminal") as ctx:
                await _get_open_panel()("ariel")

        assert "rev" not in ctx["envelope"]["details"]


@pytest.fixture
def _pinned_deployment():
    """Pin the port block (default base 10000) and the terminal's own port."""
    with (
        patch("osprey_connectors.workspace.load_osprey_config", return_value={}),
        patch("osprey.mcp_server.http.web_terminal_url", return_value="http://127.0.0.1:10100"),
    ):
        yield


class TestOpenPanelCanopyRefusal:
    """An agent-hop panel's screen moves through its own tool, never a url.

    The refusal happens in the tool, so it can name the panel's own tool: a
    refused call must raise the structured envelope naming ``set_workspace``
    and must not reach ``panel_op`` at all.
    """

    async def _assert_refused(self, panel_id: str, url: str) -> dict:
        fn = _get_open_panel()
        with patch(f"{_MODULE}.panel_op") as mock_op:
            with assert_raises_error(error_type="agent_hop_panel_url") as ctx:
                await fn(panel_id, url=url)
        mock_op.assert_not_called()
        envelope = ctx["envelope"]
        assert "set_workspace" in " ".join(envelope["suggestions"])
        assert envelope["details"]["panel"] == panel_id
        assert "canopy" in envelope["details"]["agent_hop_panels"]
        return envelope

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "url",
        [
            "/panel/canopy/embed/drafts/beam-tune",
            "/s/GTLView",
            "https://example.org/",
        ],
    )
    async def test_open_panel_canopy_with_url_refused(self, url, _pinned_deployment):
        """Any url for the canopy panel itself is refused, even a harmless one."""
        await self._assert_refused("canopy", url)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "url",
        [
            "/panel/canopy/embed/drafts/x",
            "/panel/canopy",
            "/u/alice/panel/canopy/s/GTLView",
            "http://127.0.0.1:10100/panel/canopy/s/x",
            "https://terminal.example.org/panel/canopy/trace/",
            "\\panel\\canopy\\s\\x",
            "/panel/%63anopy/s/x",
            "/panel/%2563anopy/s/x",
            "/panel/ariel/../canopy/s/x",
            "/panel/./canopy/s/x",
            "/panel//canopy/s/x",
            "/panel/canopy;x=1/s/x",
            "/PANEL/CANOPY/s/x",
            "  /panel/canopy/s/x\t",
        ],
    )
    async def test_open_panel_other_panel_with_canopy_path_refused(self, url, _pinned_deployment):
        """A url resolving under /panel/canopy/ is refused whatever panel_id names."""
        await self._assert_refused("ariel", url)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:54321/embed/drafts/x",
            "http://localhost:49200/s/GTLView",
            "http://[::1]:61000/",
            "HTTP://127.0.0.1:54321/",
            # The spellings a browser reads as the same loopback listener.
            "http://127.1:54321/",
            "http://2130706433:54321/",
            "http://0x7f.0.0.1:54321/",
            "http://0177.0.0.1:54321/",
            "http://[::ffff:127.0.0.1]:54321/",
            "http://localhost.:54321/",
            "http://foo.localhost:54321/",
        ],
    )
    async def test_open_panel_canopy_sidecar_origin_refused(self, url, _pinned_deployment):
        """The sidecar's own ephemeral loopback origin is refused for any panel."""
        await self._assert_refused("artifacts", url)

    @pytest.mark.asyncio
    async def test_open_panel_canopy_without_url_still_focuses(self):
        """No url: one ``open`` op is the only thing sent.

        The tool reaches neither the rail nor any other op, so the CANOPY slot
        and its pending batches are untouched.
        """
        fn = _get_open_panel()
        with (
            patch(f"{_MODULE}.panel_op", return_value=_applied()) as mock_op,
            patch(f"{_MODULE}.notify_panel_visibility") as mock_visibility,
        ):
            result = extract_response_dict(await fn("canopy"))

        assert result == {
            "status": "success",
            "panel": "canopy",
            "layout": _LAYOUT,
            "rev": 7,
            "popouts": [],
        }
        mock_op.assert_called_once_with("open", {"member": "canopy"}, base_rev=None)
        mock_visibility.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "url",
        [
            "/panel/ariel/#/draft/7",
            "/panel/ariel/#/panel/canopy/s/x",
            "/panel/ariel/?next=/panel/canopy/s/x",
            "/panel/canopy-notes/x",
            "http://127.0.0.1:10300/#draft",
            "http://127.0.0.1:10100/panel/ariel/",
            "https://grafana.example.org:54321/d/x",
        ],
    )
    async def test_open_panel_canopy_unrelated_urls_pass(self, url, _pinned_deployment):
        """Deep links into other panels are unchanged by the canopy refusal."""
        fn = _get_open_panel()
        with patch(f"{_MODULE}.panel_op", return_value=_applied()) as mock_op:
            result = extract_response_dict(await fn("ariel", url=url))

        assert result["status"] == "success"
        mock_op.assert_called_once_with("open", {"member": "ariel", "url": url}, base_rev=None)

    @pytest.mark.asyncio
    async def test_open_panel_canopy_refusal_is_data_driven(self, _pinned_deployment):
        """The refused set is the agent_hop flag, not the literal 'canopy'."""
        from osprey.profiles.sidecar_panels import SIDECAR_PANELS, SidecarPanel

        hop = SidecarPanel("hop", "x:y", mcp_path="/mcp", agent_hop=True)
        no_hop = SidecarPanel("plain", "x:y", mcp_path="/mcp")
        with patch.dict(SIDECAR_PANELS, {"hop-panel": hop, "plain-panel": no_hop}):
            await self._assert_refused("hop-panel", "/x")
            await self._assert_refused("ariel", "/panel/hop-panel/x")
            fn = _get_open_panel()
            with patch(f"{_MODULE}.panel_op", return_value=_applied()) as mock_op:
                result = extract_response_dict(await fn("plain-panel", url="/panel/plain-panel/x"))
        assert result["status"] == "success"
        mock_op.assert_called_once_with(
            "open", {"member": "plain-panel", "url": "/panel/plain-panel/x"}, base_rev=None
        )


def _get_add_panel_to_rail():
    from osprey.mcp_server.workspace.tools.panel_tools import add_panel_to_rail

    return get_tool_fn(add_panel_to_rail)


def _get_remove_panel_from_rail():
    from osprey.mcp_server.workspace.tools.panel_tools import remove_panel_from_rail

    return get_tool_fn(remove_panel_from_rail)


def _get_register_panel():
    from osprey.mcp_server.workspace.tools.panel_tools import register_panel

    return get_tool_fn(register_panel)


def _make_api_mock(enabled, custom=None, visible=None, active=None, labels=None):
    """Build a urllib urlopen mock for GET /api/panels with the given fields."""
    import json
    from unittest.mock import MagicMock

    payload = {
        "enabled": list(enabled),
        "custom": custom or [],
        "visible": visible if visible is not None else list(enabled),
        "active": active,
        "labels": labels or {},
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---- add_panel_to_rail ----


class TestAddPanelToRail:
    @pytest.mark.asyncio
    async def test_add_panel_to_rail_calls_notify_visibility_with_true(
        self, _mock_web_terminal_url
    ):
        """A known id calls notify_panel_visibility(id, True) and returns success."""
        # Arrange
        fn = _get_add_panel_to_rail()
        mock_resp = _make_api_mock(enabled=["artifacts", "ariel"])

        # Act
        with patch("urllib.request.urlopen", return_value=mock_resp):
            with patch(f"{_MODULE}.notify_panel_visibility") as mock_notify:
                result = extract_response_dict(await fn("ariel"))

        # Assert
        assert result["status"] == "success"
        assert result["panel"] == "ariel"
        mock_notify.assert_called_once_with("ariel", True)

    @pytest.mark.asyncio
    async def test_add_panel_to_rail_unknown_id_returns_error_and_does_not_notify(
        self, _mock_web_terminal_url
    ):
        """An unknown panel id returns a structured error and skips notify."""
        # Arrange
        fn = _get_add_panel_to_rail()
        mock_resp = _make_api_mock(enabled=["artifacts"])

        # Act
        with patch("urllib.request.urlopen", return_value=mock_resp):
            with patch(f"{_MODULE}.notify_panel_visibility") as mock_notify:
                result = extract_response_dict(await fn("nonexistent"))

        # Assert
        assert result["status"] == "error"
        assert "nonexistent" in result["message"]
        mock_notify.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_panel_to_rail_web_terminal_unreachable_returns_error(
        self, _mock_web_terminal_url
    ):
        """An unreachable web terminal returns an error dict."""
        # Arrange
        fn = _get_add_panel_to_rail()

        # Act
        with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError("refused")):
            result = extract_response_dict(await fn("artifacts"))

        # Assert
        assert result["status"] == "error"
        assert "not running" in result["message"]


# ---- remove_panel_from_rail ----


class TestRemovePanelFromRail:
    @pytest.mark.asyncio
    async def test_remove_panel_from_rail_calls_notify_visibility_with_false(
        self, _mock_web_terminal_url
    ):
        """A known id calls notify_panel_visibility(id, False) and returns success."""
        # Arrange
        fn = _get_remove_panel_from_rail()
        mock_resp = _make_api_mock(enabled=["artifacts", "ariel"])

        # Act
        with patch("urllib.request.urlopen", return_value=mock_resp):
            with patch(f"{_MODULE}.notify_panel_visibility") as mock_notify:
                result = extract_response_dict(await fn("ariel"))

        # Assert
        assert result["status"] == "success"
        assert result["panel"] == "ariel"
        assert result["on_rail"] is False
        mock_notify.assert_called_once_with("ariel", False)

    @pytest.mark.asyncio
    async def test_remove_panel_from_rail_unknown_id_returns_error_and_does_not_notify(
        self, _mock_web_terminal_url
    ):
        """An unknown panel id returns a structured error and skips notify."""
        # Arrange
        fn = _get_remove_panel_from_rail()
        mock_resp = _make_api_mock(enabled=["artifacts"])

        # Act
        with patch("urllib.request.urlopen", return_value=mock_resp):
            with patch(f"{_MODULE}.notify_panel_visibility") as mock_notify:
                result = extract_response_dict(await fn("unknown-panel"))

        # Assert
        assert result["status"] == "error"
        mock_notify.assert_not_called()


# ---- register_panel ----


class TestRegisterPanel:
    @pytest.mark.asyncio
    async def test_register_panel_success_returns_panel_url(self):
        """register_panel returns status:success and the proxy URL when notify returns ok=True."""
        # Arrange
        fn = _get_register_panel()
        ok_result = {"ok": True, "status": 200, "data": {"url": "/panel/grafana"}}

        # Act
        with patch(f"{_MODULE}.notify_panel_register", return_value=ok_result):
            result = extract_response_dict(
                await fn("grafana", "GRAFANA", "http://grafana.lan:3000")
            )

        # Assert
        assert result["status"] == "success"
        assert result["panel"] == "grafana"
        assert result["url"] == "/panel/grafana"

    @pytest.mark.asyncio
    async def test_register_panel_disabled_returns_disabled_message(self):
        """register_panel surfaces a human-readable 'disabled' message on HTTP 403."""
        # Arrange
        fn = _get_register_panel()
        disabled_result = {
            "ok": False,
            "status": 403,
            "detail": "Runtime panel registration is disabled.",
        }

        # Act
        with patch(f"{_MODULE}.notify_panel_register", return_value=disabled_result):
            result = extract_response_dict(
                await fn("grafana", "GRAFANA", "http://grafana.lan:3000")
            )

        # Assert
        assert result["status"] == "error"
        assert "disabled" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_register_panel_validation_error_surfaces_detail(self):
        """register_panel forwards the server detail string on HTTP 422."""
        # Arrange
        fn = _get_register_panel()
        rejected_result = {
            "ok": False,
            "status": 422,
            "detail": "Resolved address '127.0.0.1' is not permitted",
        }

        # Act
        with patch(f"{_MODULE}.notify_panel_register", return_value=rejected_result):
            result = extract_response_dict(
                await fn("grafana", "GRAFANA", "http://grafana.lan:3000")
            )

        # Assert
        assert result["status"] == "error"
        assert "127.0.0.1" in result["message"]

    @pytest.mark.asyncio
    async def test_register_panel_web_terminal_down_returns_down_message(self):
        """register_panel returns a 'not running' message when notify returns status=None."""
        # Arrange
        fn = _get_register_panel()
        down_result = {
            "ok": False,
            "status": None,
            "detail": "Web Terminal is not running.",
        }

        # Act
        with patch(f"{_MODULE}.notify_panel_register", return_value=down_result):
            result = extract_response_dict(
                await fn("grafana", "GRAFANA", "http://grafana.lan:3000")
            )

        # Assert
        assert result["status"] == "error"
        assert "not running" in result["message"].lower()


# ---- arrange_workspace ----


class TestDropPanel:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("source", "target"),
        [
            ({"new": {"panel": "ariel"}}, {"tile": "t1", "zone": "centre"}),
            ({"tab": "lattice"}, {"tile": "t2", "zone": "tab", "index": 0}),
            ({"tile": "t1"}, {"tile": "t2", "edge": "left"}),
            ({"tab": "ariel"}, {"border": {"tile": "t1", "side": "right"}}),
            ({"new": {"panel": "artifacts"}}, {"outer": "bottom"}),
        ],
    )
    async def test_drop_panel_relays_drop_with_base_rev(self, source, target):
        """Every source and target form goes through as given, with base_rev."""
        with patch(f"{_MODULE}.panel_op", return_value=_applied(rev=5)) as mock_op:
            result = extract_response_dict(await _tool("drop_panel")(source, target, 4))

        assert result == {"status": "success", "layout": _LAYOUT, "popouts": [], "rev": 5}
        mock_op.assert_called_once_with("drop", {"source": source, "target": target}, base_rev=4)

    @pytest.mark.asyncio
    async def test_drop_panel_stale_base_names_the_reread(self):
        with patch(f"{_MODULE}.panel_op", return_value=_refused("stale_base", "behind", rev=8)):
            with assert_raises_error(error_type="stale_base") as ctx:
                await _tool("drop_panel")({"tab": "ariel"}, {"outer": "left"}, 2)

        envelope = ctx["envelope"]
        assert envelope["details"]["rev"] == 8
        assert "list_panels" in envelope["suggestions"][0]
        assert "drop_panel refused (stale_base)" in envelope["error_message"]

    @pytest.mark.asyncio
    async def test_drop_panel_terminal_source_refused(self):
        refused = _refused("terminal_not_agent_movable", "the terminal ...", status=422)
        with patch(f"{_MODULE}.panel_op", return_value=refused):
            with assert_raises_error(error_type="terminal_not_agent_movable"):
                await _tool("drop_panel")({"tab": "terminal"}, {"outer": "left"}, 1)


class TestClosePanelTabAndTile:
    @pytest.mark.asyncio
    async def test_close_panel_tab_relays_close_tab_without_base_rev(self):
        with patch(f"{_MODULE}.panel_op", return_value=_applied(rev=3)) as mock_op:
            result = extract_response_dict(await _tool("close_panel_tab")("ariel"))

        assert result == {
            "status": "success",
            "panel": "ariel",
            "layout": _LAYOUT,
            "popouts": [],
            "rev": 3,
        }
        mock_op.assert_called_once_with("close_tab", {"key": "ariel"}, base_rev=None)

    @pytest.mark.asyncio
    async def test_close_panel_tab_does_not_touch_rail_membership(self):
        with (
            patch(f"{_MODULE}.panel_op", return_value=_applied()),
            patch(f"{_MODULE}.notify_panel_visibility") as mock_visibility,
        ):
            await _tool("close_panel_tab")("ariel")

        mock_visibility.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_panel_tab_unknown_key_refused(self):
        with patch(f"{_MODULE}.panel_op", return_value=_refused("unknown_key", status=422)):
            with assert_raises_error(error_type="unknown_key"):
                await _tool("close_panel_tab")("nope")

    @pytest.mark.asyncio
    async def test_close_panel_tile_relays_close_tile_with_base_rev(self):
        with patch(f"{_MODULE}.panel_op", return_value=_applied(rev=6)) as mock_op:
            result = extract_response_dict(await _tool("close_panel_tile")("t1", 5))

        assert result == {
            "status": "success",
            "tile": "t1",
            "layout": _LAYOUT,
            "popouts": [],
            "rev": 6,
        }
        mock_op.assert_called_once_with("close_tile", {"tile": "t1"}, base_rev=5)

    @pytest.mark.asyncio
    async def test_close_panel_tile_no_dock_client(self):
        with patch(f"{_MODULE}.panel_op", return_value=_refused("no_dock_client")):
            with assert_raises_error(error_type="no_dock_client") as ctx:
                await _tool("close_panel_tile")("t1", 5)

        assert "Web Terminal" in ctx["envelope"]["suggestions"][0]

    def test_old_close_panel_tool_is_gone(self):
        from osprey.mcp_server.workspace.tools import panel_tools

        assert not hasattr(panel_tools, "close_panel")


class TestOpenPanelSet:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("open_as", ["tabs", "side"])
    async def test_open_panel_set_relays_open_set(self, open_as):
        with patch(f"{_MODULE}.panel_op", return_value=_applied()) as mock_op:
            result = extract_response_dict(
                await _tool("open_panel_set")(["ariel", "lattice"], open_as)
            )

        assert result["panels"] == ["ariel", "lattice"]
        assert result["rev"] == 7
        mock_op.assert_called_once_with(
            "open_set", {"members": ["ariel", "lattice"], "as": open_as}, base_rev=None
        )

    @pytest.mark.asyncio
    async def test_open_panel_set_defaults_to_side(self):
        with patch(f"{_MODULE}.panel_op", return_value=_applied()) as mock_op:
            await _tool("open_panel_set")(["ariel"])

        assert mock_op.call_args.args[1]["as"] == "side"

    def test_open_panel_set_opens_as_tabs_or_side_only(self):
        """An exact layout is arrange_workspace's: open_panel_set takes none."""
        import inspect
        import typing

        from osprey.mcp_server.workspace.tools import panel_tools

        params = inspect.signature(_tool("open_panel_set")).parameters
        assert list(params) == ["panels", "open_as"]
        assert typing.get_args(panel_tools.OpenSetAs) == ("tabs", "side")

    @pytest.mark.asyncio
    async def test_open_panel_set_too_many_for_side_refused(self):
        with patch(f"{_MODULE}.panel_op", return_value=_refused("too_many_for_side")):
            with assert_raises_error(error_type="too_many_for_side") as ctx:
                await _tool("open_panel_set")(["a", "b", "c"], "side")

        assert "tabs" in ctx["envelope"]["suggestions"][0]


class TestArrangeWorkspace:
    @pytest.mark.asyncio
    async def test_arrange_preset_relays_open_arrangement(self):
        with patch(f"{_MODULE}.panel_op", return_value=_applied(rev=11)) as mock_op:
            result = extract_response_dict(await _tool("arrange_workspace")(preset="Ops"))

        assert result == {
            "status": "success",
            "preset": "Ops",
            "layout": _LAYOUT,
            "popouts": [],
            "rev": 11,
        }
        mock_op.assert_called_once_with("open_arrangement", {"name": "Ops"}, base_rev=None)

    @pytest.mark.asyncio
    async def test_arrange_preset_with_open_as(self):
        with patch(f"{_MODULE}.panel_op", return_value=_applied()) as mock_op:
            await _tool("arrange_workspace")(preset="Ops", open_as="tabs")

        mock_op.assert_called_once_with(
            "open_arrangement", {"name": "Ops", "as": "tabs"}, base_rev=None
        )

    @pytest.mark.asyncio
    async def test_arrange_layout_relays_open_set_as_layout(self):
        """A panel-id layout becomes open_set over its keys in reading order."""
        layout = {
            "split": "column",
            "items": [
                {"tabs": ["lattice"]},
                {"tabs": ["ariel", "artifacts"], "showing": "artifacts"},
            ],
        }
        with patch(f"{_MODULE}.panel_op", return_value=_applied()) as mock_op:
            result = extract_response_dict(await _tool("arrange_workspace")(layout=layout))

        assert result["panels"] == ["lattice", "ariel", "artifacts"]
        mock_op.assert_called_once_with(
            "open_set",
            {
                "members": ["lattice", "ariel", "artifacts"],
                "as": "layout",
                "layout": {
                    "split": "column",
                    "items": [{"tabs": [0]}, {"tabs": [1, 2], "showing": 2}],
                },
            },
            base_rev=None,
        )

    @pytest.mark.asyncio
    async def test_arrange_single_tile_layout(self):
        with patch(f"{_MODULE}.panel_op", return_value=_applied()) as mock_op:
            await _tool("arrange_workspace")(layout={"tabs": ["ariel", "lattice"]})

        mock_op.assert_called_once_with(
            "open_set",
            {"members": ["ariel", "lattice"], "as": "layout", "layout": {"tabs": [0, 1]}},
            base_rev=None,
        )

    @pytest.mark.asyncio
    async def test_arrange_layout_naming_terminal_reaches_route_unindexed(self):
        """The terminal is left as a name so the route refuses terminal_in_layout."""
        refused = _refused("terminal_in_layout", status=422)
        with patch(f"{_MODULE}.panel_op", return_value=refused) as mock_op:
            with assert_raises_error(error_type="terminal_in_layout"):
                await _tool("arrange_workspace")(layout={"tabs": ["ariel", "terminal"]})

        args = mock_op.call_args.args[1]
        assert args["members"] == ["ariel", "terminal"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"preset": "Ops", "layout": {"tabs": ["ariel"]}}],
    )
    async def test_arrange_needs_exactly_one_of_preset_or_layout(self, kwargs):
        with patch(f"{_MODULE}.panel_op") as mock_op:
            with assert_raises_error(error_type="invalid_request"):
                await _tool("arrange_workspace")(**kwargs)

        mock_op.assert_not_called()

    @pytest.mark.asyncio
    async def test_arrange_layout_refuses_open_as_other_than_layout(self):
        with patch(f"{_MODULE}.panel_op") as mock_op:
            with assert_raises_error(error_type="invalid_request"):
                await _tool("arrange_workspace")(layout={"tabs": ["ariel"]}, open_as="side")

        mock_op.assert_not_called()

    @pytest.mark.asyncio
    async def test_arrange_layout_naming_no_panel_refused(self):
        with patch(f"{_MODULE}.panel_op") as mock_op:
            with assert_raises_error(error_type="invalid_layout") as ctx:
                await _tool("arrange_workspace")(layout={"split": "row", "items": []})

        mock_op.assert_not_called()
        assert "arrange_workspace refused" in ctx["envelope"]["error_message"]

    @pytest.mark.asyncio
    async def test_arrange_unknown_preset_refused(self):
        refused = _refused("unknown_arrangement", "unknown preset 'X'; presets: ['Ops']", 422)
        with patch(f"{_MODULE}.panel_op", return_value=refused):
            with assert_raises_error(error_type="unknown_arrangement") as ctx:
                await _tool("arrange_workspace")(preset="X")

        assert "['Ops']" in ctx["envelope"]["error_message"]

    @pytest.mark.asyncio
    async def test_arrange_web_terminal_unreachable(self):
        down = _refused("no_web_terminal", "Web Terminal is not running.", status=None)
        with patch(f"{_MODULE}.panel_op", return_value=down):
            with assert_raises_error(error_type="no_web_terminal") as ctx:
                await _tool("arrange_workspace")(preset="Ops")

        assert "Continue without the workspace" in ctx["envelope"]["suggestions"][0]
