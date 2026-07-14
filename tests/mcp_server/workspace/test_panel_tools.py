"""Tests for panel_tools MCP tools (list_panels, switch_panel).

Covers:
  - list_panels returns enabled built-in panels with correct labels
  - list_panels includes custom panels (with and without explicit label)
  - list_panels handles web terminal being unreachable
  - switch_panel calls notify_panel_focus with correct args
  - switch_panel passes optional url through
  - switch_panel works with app-registered (custom) panel IDs
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tests.mcp_server.conftest import extract_response_dict, get_tool_fn

_MODULE = "osprey.mcp_server.workspace.tools.panel_tools"


@pytest.fixture
def _mock_web_terminal_url():
    with patch(f"{_MODULE}.web_terminal_url", return_value="http://127.0.0.1:8087"):
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
