"""Tests for the panel_tools MCP tools.

Covers:
  - list_panels returns enabled built-in panels with correct labels
  - list_panels includes custom panels (with and without explicit label)
  - list_panels handles web terminal being unreachable
  - switch_panel calls notify_panel_focus with correct args
  - switch_panel passes optional url through
  - switch_panel works with app-registered (custom) panel IDs
  - show_panel / hide_panel validate ids before toggling rail membership
  - register_panel maps the register helper's outcomes to its response
  - arrange_workspace reports the applied layout plus report freshness, and
    surfaces the route's validation details verbatim as structured errors
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn

_MODULE = "osprey.mcp_server.workspace.tools.panel_tools"


@pytest.fixture
def _mock_web_terminal_url():
    """Pin the base URL used by the shared http.fetch_panels helper.

    The panel tools do not resolve the URL themselves — they delegate the
    ``GET /api/panels`` read to ``osprey.mcp_server.http.fetch_panels``, so the
    patch has to land there for tests that stub ``urllib.request.urlopen``.
    """
    with patch("osprey.mcp_server.http.web_terminal_url", return_value="http://127.0.0.1:8087"):
        yield


def _get_list_panels():
    from osprey.mcp_server.workspace.tools.panel_tools import list_panels

    return get_tool_fn(list_panels)


def _get_switch_panel():
    from osprey.mcp_server.workspace.tools.panel_tools import switch_panel

    return get_tool_fn(switch_panel)


class TestListPanels:
    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_web_terminal_unreachable(self, _mock_web_terminal_url):
        """Returns error when web terminal is not running."""
        fn = _get_list_panels()

        with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError("refused")):
            result = extract_response_dict(await fn())

        assert result["status"] == "error"
        assert "not running" in result["message"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_list_panels_reports_open_tiles_independently_of_rail_membership(
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
                "open_tiles": ["lattice", "artifacts"],
                "open_tiles_age_s": 12.0,
                "open_tiles_dock": True,
            }
        ).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = api_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = extract_response_dict(await fn())

        assert result["open_tiles"] == ["lattice", "artifacts"]
        assert result["open_tiles_age_s"] == 12.0
        assert result["open_tiles_dock"] is True
        visible = {p["id"]: p["visible"] for p in result["panels"]}
        assert visible["ariel"] is True and "ariel" not in result["open_tiles"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_list_panels_freshness_is_null_when_no_client_has_reported(
        self, _mock_web_terminal_url
    ):
        """Never-reported is all-null — never [], which would claim a known-empty screen."""
        fn = _get_list_panels()

        api_response = json.dumps(
            {
                "enabled": ["artifacts"],
                "custom": [],
                "labels": {},
                "visible": ["artifacts"],
                "active": None,
                # The route's never-reported state: all three null.
                "open_tiles": None,
                "open_tiles_age_s": None,
                "open_tiles_dock": None,
            }
        ).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = api_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = extract_response_dict(await fn())

        assert result["open_tiles"] is None
        assert result["open_tiles_age_s"] is None
        assert result["open_tiles_dock"] is None

    @pytest.mark.unit
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
                "open_tiles": None,
                "open_tiles_age_s": 3.0,
                "open_tiles_dock": False,
            }
        ).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = api_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = extract_response_dict(await fn())

        assert result["open_tiles"] is None
        assert result["open_tiles_dock"] is False
        assert result["open_tiles_age_s"] == 3.0


class TestSwitchPanel:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_calls_notify(self):
        """switch_panel delegates to notify_panel_focus."""
        fn = _get_switch_panel()

        with patch(f"{_MODULE}.notify_panel_focus") as mock_focus:
            result = extract_response_dict(await fn("ariel"))

        assert result["status"] == "success"
        assert result["panel"] == "ariel"
        mock_focus.assert_called_once_with("ariel", url=None)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_passes_url(self):
        """Optional url is forwarded to notify_panel_focus."""
        fn = _get_switch_panel()

        with patch(f"{_MODULE}.notify_panel_focus") as mock_focus:
            result = extract_response_dict(await fn("ariel", url="http://127.0.0.1:8085/#draft"))

        assert result["status"] == "success"
        mock_focus.assert_called_once_with("ariel", url="http://127.0.0.1:8085/#draft")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_switch_custom_panel(self):
        """switch_panel works with app-registered (custom) panel IDs."""
        fn = _get_switch_panel()

        with patch(f"{_MODULE}.notify_panel_focus") as mock_focus:
            result = extract_response_dict(await fn("my-grafana"))

        assert result["status"] == "success"
        assert result["panel"] == "my-grafana"
        mock_focus.assert_called_once_with("my-grafana", url=None)


# ---- Helpers for show/hide/register tests ----


def _get_show_panel():
    from osprey.mcp_server.workspace.tools.panel_tools import show_panel

    return get_tool_fn(show_panel)


def _get_hide_panel():
    from osprey.mcp_server.workspace.tools.panel_tools import hide_panel

    return get_tool_fn(hide_panel)


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


# ---- show_panel ----


class TestShowPanel:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_show_panel_calls_notify_visibility_with_true(self, _mock_web_terminal_url):
        """show_panel for a known id calls notify_panel_visibility(id, True) and returns success."""
        # Arrange
        fn = _get_show_panel()
        mock_resp = _make_api_mock(enabled=["artifacts", "ariel"])

        # Act
        with patch("urllib.request.urlopen", return_value=mock_resp):
            with patch(f"{_MODULE}.notify_panel_visibility") as mock_notify:
                result = extract_response_dict(await fn("ariel"))

        # Assert
        assert result["status"] == "success"
        assert result["panel"] == "ariel"
        mock_notify.assert_called_once_with("ariel", True)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_show_panel_unknown_id_returns_error_and_does_not_notify(
        self, _mock_web_terminal_url
    ):
        """show_panel for an unknown panel id returns a structured error and skips notify."""
        # Arrange
        fn = _get_show_panel()
        mock_resp = _make_api_mock(enabled=["artifacts"])

        # Act
        with patch("urllib.request.urlopen", return_value=mock_resp):
            with patch(f"{_MODULE}.notify_panel_visibility") as mock_notify:
                result = extract_response_dict(await fn("nonexistent"))

        # Assert
        assert result["status"] == "error"
        assert "nonexistent" in result["message"]
        mock_notify.assert_not_called()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_show_panel_web_terminal_unreachable_returns_error(self, _mock_web_terminal_url):
        """show_panel returns an error dict when the web terminal is not reachable."""
        # Arrange
        fn = _get_show_panel()

        # Act
        with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError("refused")):
            result = extract_response_dict(await fn("artifacts"))

        # Assert
        assert result["status"] == "error"
        assert "not running" in result["message"]


# ---- hide_panel ----


class TestHidePanel:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_hide_panel_calls_notify_visibility_with_false(self, _mock_web_terminal_url):
        """hide_panel for a known id calls notify_panel_visibility(id, False) and returns success."""
        # Arrange
        fn = _get_hide_panel()
        mock_resp = _make_api_mock(enabled=["artifacts", "ariel"])

        # Act
        with patch("urllib.request.urlopen", return_value=mock_resp):
            with patch(f"{_MODULE}.notify_panel_visibility") as mock_notify:
                result = extract_response_dict(await fn("ariel"))

        # Assert
        assert result["status"] == "success"
        assert result["panel"] == "ariel"
        assert result["visible"] is False
        mock_notify.assert_called_once_with("ariel", False)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_hide_panel_unknown_id_returns_error_and_does_not_notify(
        self, _mock_web_terminal_url
    ):
        """hide_panel for an unknown panel id returns a structured error and skips notify."""
        # Arrange
        fn = _get_hide_panel()
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
    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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

    @pytest.mark.unit
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


def _get_arrange_workspace():
    from osprey.mcp_server.workspace.tools.panel_tools import arrange_workspace

    return get_tool_fn(arrange_workspace)


class TestArrangeWorkspace:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_arrange_tiles_returns_applied_layout_and_freshness(self):
        """A tiles arrangement echoes the server's applied layout plus freshness."""
        # Arrange
        fn = _get_arrange_workspace()
        ok_result = {
            "ok": True,
            "status": 200,
            "data": {
                "status": "ok",
                "tiles": ["artifacts", "lattice"],
                "focus": "lattice",
                "preset": None,
                "prune_rail": False,
            },
        }
        panels = {"open_tiles": ["artifacts"], "open_tiles_age_s": 2.5, "open_tiles_dock": True}

        # Act
        with (
            patch(f"{_MODULE}.notify_panel_arrange", return_value=ok_result) as mock_arrange,
            patch(f"{_MODULE}.fetch_panels", return_value=panels),
        ):
            result = extract_response_dict(
                await fn(tiles=["artifacts", "lattice"], focus="lattice")
            )

        # Assert
        assert result["status"] == "success"
        assert result["tiles"] == ["artifacts", "lattice"]
        assert result["focus"] == "lattice"
        assert result["open_tiles_age_s"] == 2.5
        assert result["open_tiles_dock"] is True
        assert "preset" not in result
        mock_arrange.assert_called_once_with(
            tiles=["artifacts", "lattice"], preset=None, focus="lattice"
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_arrange_preset_reports_preset_name(self):
        """A preset call forwards the name and reports it back."""
        # Arrange
        fn = _get_arrange_workspace()
        ok_result = {
            "ok": True,
            "status": 200,
            "data": {
                "status": "ok",
                "tiles": ["artifacts", "errors"],
                "focus": None,
                "preset": "injection",
                "prune_rail": True,
            },
        }

        # Act
        with (
            patch(f"{_MODULE}.notify_panel_arrange", return_value=ok_result) as mock_arrange,
            patch(f"{_MODULE}.fetch_panels", return_value={}),
        ):
            result = extract_response_dict(await fn(preset="injection"))

        # Assert
        assert result["preset"] == "injection"
        assert result["tiles"] == ["artifacts", "errors"]
        assert result["focus"] is None
        mock_arrange.assert_called_once_with(tiles=None, preset="injection", focus=None)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_arrange_reports_null_freshness_when_readback_fails(self):
        """An arrangement that applied is still a success when the freshness read fails."""
        # Arrange
        fn = _get_arrange_workspace()
        ok_result = {"ok": True, "status": 200, "data": {"tiles": ["artifacts"], "focus": None}}

        # Act
        with (
            patch(f"{_MODULE}.notify_panel_arrange", return_value=ok_result),
            patch(f"{_MODULE}.fetch_panels", return_value=None),
        ):
            result = extract_response_dict(await fn(tiles=["artifacts"]))

        # Assert
        assert result["status"] == "success"
        assert result["open_tiles_age_s"] is None
        assert result["open_tiles_dock"] is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_arrange_reports_nulls_when_no_client_has_reported(self):
        """Nulls from a live server mean nobody is watching — the agent should see that."""
        # Arrange
        fn = _get_arrange_workspace()
        ok_result = {"ok": True, "status": 200, "data": {"tiles": ["artifacts"], "focus": None}}
        never_reported = {
            "enabled": ["artifacts"],
            "open_tiles": None,
            "open_tiles_age_s": None,
            "open_tiles_dock": None,
        }

        # Act
        with (
            patch(f"{_MODULE}.notify_panel_arrange", return_value=ok_result),
            patch(f"{_MODULE}.fetch_panels", return_value=never_reported),
        ):
            result = extract_response_dict(await fn(tiles=["artifacts"]))

        # Assert
        assert result["status"] == "success"
        assert result["open_tiles_age_s"] is None
        assert result["open_tiles_dock"] is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_arrange_surfaces_route_detail_verbatim(self):
        """A 422 from the route reaches the agent word for word."""
        # Arrange
        fn = _get_arrange_workspace()
        detail = "Unknown panel ids: ['bogus']. Valid panel ids: ['artifacts', 'lattice']"
        rejected = {"ok": False, "status": 422, "detail": detail}

        # Act / Assert
        with patch(f"{_MODULE}.notify_panel_arrange", return_value=rejected):
            with assert_raises_error(error_type="arrange_rejected") as ctx:
                await fn(tiles=["bogus"])
        assert detail in ctx["envelope"]["error_message"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_arrange_unknown_preset_detail_lists_available(self):
        """The route's available-preset list is preserved for the agent."""
        # Arrange
        fn = _get_arrange_workspace()
        detail = "Unknown preset: 'nope'. Available presets: ['injection']"
        rejected = {"ok": False, "status": 422, "detail": detail}

        # Act / Assert
        with patch(f"{_MODULE}.notify_panel_arrange", return_value=rejected):
            with assert_raises_error(error_type="arrange_rejected") as ctx:
                await fn(preset="nope")
        assert "Available presets: ['injection']" in ctx["envelope"]["error_message"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_arrange_web_terminal_unreachable(self):
        """An unreachable web terminal is its own error type, not a rejection."""
        # Arrange
        fn = _get_arrange_workspace()
        down = {"ok": False, "status": None, "detail": "Web Terminal is not running."}

        # Act / Assert
        with patch(f"{_MODULE}.notify_panel_arrange", return_value=down):
            with assert_raises_error(error_type="web_terminal_unreachable") as ctx:
                await fn(tiles=["artifacts"])
        assert "not running" in ctx["envelope"]["error_message"].lower()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_arrange_does_not_prevalidate_tiles_xor_preset(self):
        """The route owns tiles/preset exclusivity — the tool forwards and reports it."""
        # Arrange
        fn = _get_arrange_workspace()
        rejected = {
            "ok": False,
            "status": 422,
            "detail": "Provide exactly one of 'tiles' or 'preset'",
        }

        # Act / Assert
        with patch(f"{_MODULE}.notify_panel_arrange", return_value=rejected) as mock_arrange:
            with assert_raises_error(error_type="arrange_rejected"):
                await fn(tiles=["artifacts"], preset="injection")
        mock_arrange.assert_called_once_with(tiles=["artifacts"], preset="injection", focus=None)
