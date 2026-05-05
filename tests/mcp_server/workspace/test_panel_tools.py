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

from tests.mcp_server.conftest import get_tool_fn

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
            {"enabled": ["artifacts", "ariel", "tuning"], "custom": []}
        ).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = api_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = json.loads(await fn())

        assert result["status"] == "success"
        ids = [p["id"] for p in result["panels"]]
        assert ids == ["artifacts", "ariel", "tuning"]
        labels = {p["id"]: p["label"] for p in result["panels"]}
        assert labels["artifacts"] == "WORKSPACE"
        assert labels["ariel"] == "ARIEL"
        assert labels["tuning"] == "TUNING"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_includes_custom_panels(self, _mock_web_terminal_url):
        """Custom panels from config are appended to the list."""
        fn = _get_list_panels()

        api_response = json.dumps(
            {
                "enabled": ["artifacts"],
                "custom": [{"id": "my-panel", "label": "MY PANEL", "url": "/panel/my-panel"}],
            }
        ).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = api_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = json.loads(await fn())

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
            }
        ).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = api_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = json.loads(await fn())

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
            result = json.loads(await fn())

        assert result["status"] == "error"
        assert "not running" in result["message"]


class TestSwitchPanel:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_calls_notify(self):
        """switch_panel delegates to notify_panel_focus."""
        fn = _get_switch_panel()

        with patch(f"{_MODULE}.notify_panel_focus") as mock_focus:
            result = json.loads(await fn("ariel"))

        assert result["status"] == "success"
        assert result["panel"] == "ariel"
        mock_focus.assert_called_once_with("ariel", url=None)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_passes_url(self):
        """Optional url is forwarded to notify_panel_focus."""
        fn = _get_switch_panel()

        with patch(f"{_MODULE}.notify_panel_focus") as mock_focus:
            result = json.loads(await fn("ariel", url="http://127.0.0.1:8085/#draft"))

        assert result["status"] == "success"
        mock_focus.assert_called_once_with("ariel", url="http://127.0.0.1:8085/#draft")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_switch_custom_panel(self):
        """switch_panel works with app-registered (custom) panel IDs."""
        fn = _get_switch_panel()

        with patch(f"{_MODULE}.notify_panel_focus") as mock_focus:
            result = json.loads(await fn("my-grafana"))

        assert result["status"] == "success"
        assert result["panel"] == "my-grafana"
        mock_focus.assert_called_once_with("my-grafana", url=None)
