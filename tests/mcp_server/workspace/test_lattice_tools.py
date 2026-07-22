"""Tests for the lattice-dashboard MCP tools (``workspace.tools.lattice_tools``).

The tools are thin async HTTP wrappers over the lattice dashboard server. Tests
mock the HTTP boundary (``_dashboard_request`` or the httpx client) so they run
with no server and no network. They verify: URL construction with config/env
precedence; the request helper's method dispatch; each tool's happy-path JSON
shape and the request it issues; and the mapping of httpx failures onto the
``service_unavailable`` / ``lattice_error`` envelopes.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from osprey.mcp_server.workspace.tools import lattice_tools as lt
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

_MOD = "osprey.mcp_server.workspace.tools.lattice_tools"


def _fn(tool):
    return get_tool_fn(tool)


def _http_status_error(status: int = 500, text: str = "boom") -> httpx.HTTPStatusError:
    req = httpx.Request("GET", "http://dash/api")
    resp = httpx.Response(status, text=text, request=req)
    return httpx.HTTPStatusError("bad status", request=req, response=resp)


def _patch_request(return_value=None, side_effect=None) -> AsyncMock:
    mock = AsyncMock(return_value=return_value, side_effect=side_effect)
    return patch.object(lt, "_dashboard_request", mock)


# ---------------------------------------------------------------------------
# _get_dashboard_url
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_dashboard_url_defaults(monkeypatch):
    monkeypatch.delenv("OSPREY_LATTICE_DASHBOARD_PORT", raising=False)
    with patch(f"{_MOD}.load_osprey_config", return_value={}):
        assert lt._get_dashboard_url() == "http://127.0.0.1:8097"


@pytest.mark.unit
def test_dashboard_url_from_config(monkeypatch):
    monkeypatch.delenv("OSPREY_LATTICE_DASHBOARD_PORT", raising=False)
    with patch(
        f"{_MOD}.load_osprey_config", return_value={"lattice_dashboard": {"host": "h", "port": 9}}
    ):
        assert lt._get_dashboard_url() == "http://h:9"


@pytest.mark.unit
def test_dashboard_url_env_overrides_port(monkeypatch):
    monkeypatch.setenv("OSPREY_LATTICE_DASHBOARD_PORT", "5555")
    with patch(f"{_MOD}.load_osprey_config", return_value={"lattice_dashboard": {"port": 8097}}):
        assert lt._get_dashboard_url() == "http://127.0.0.1:5555"


# ---------------------------------------------------------------------------
# _dashboard_request — method dispatch
# ---------------------------------------------------------------------------


def _mock_async_client(response):
    """Build a patch of httpx.AsyncClient whose verbs return ``response``."""
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    client.post = AsyncMock(return_value=response)
    client.put = AsyncMock(return_value=response)
    client.delete = AsyncMock(return_value=response)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return patch("httpx.AsyncClient", return_value=cm), client


@pytest.mark.unit
@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "DELETE"])
async def test_dashboard_request_dispatches_method(method):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"ok": True}
    patcher, client = _mock_async_client(resp)
    with patcher, patch.object(lt, "_get_dashboard_url", return_value="http://dash"):
        out = await lt._dashboard_request(method, "/api/x", json_body={"a": 1})
    assert out == {"ok": True}
    resp.raise_for_status.assert_called_once()
    verb = getattr(client, method.lower())
    verb.assert_awaited_once()


@pytest.mark.unit
async def test_dashboard_request_rejects_unknown_method():
    resp = MagicMock()
    patcher, _client = _mock_async_client(resp)
    with patcher, patch.object(lt, "_get_dashboard_url", return_value="http://dash"):
        with pytest.raises(ValueError, match="Unsupported method"):
            await lt._dashboard_request("PATCH", "/api/x")


# ---------------------------------------------------------------------------
# Tool happy paths + the request each issues
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_lattice_init_happy():
    payload = {"summary": {"energy": 2.0}, "families": {"QF": {}, "SD": {}}}
    with _patch_request(return_value=payload) as req:
        result = await _fn(lt.lattice_init)(lattice_path="als.m")
    data = json.loads(result)
    assert data["status"] == "ok"
    assert data["summary"] == {"energy": 2.0}
    assert sorted(data["families"]) == ["QF", "SD"]
    method, path = req.call_args.args
    assert (method, path) == ("POST", "/api/state/init")
    assert req.call_args.kwargs["json_body"] == {"lattice_path": "als.m"}


@pytest.mark.unit
async def test_lattice_state_returns_raw():
    with _patch_request(return_value={"base_lattice": "als.m", "figures": {}}):
        result = await _fn(lt.lattice_state)()
    assert json.loads(result)["base_lattice"] == "als.m"


@pytest.mark.unit
async def test_lattice_set_param_happy():
    with _patch_request(return_value={}) as req:
        result = await _fn(lt.lattice_set_param)(family="QF", value=1.5)
    data = json.loads(result)
    assert data == {
        "status": "ok",
        "family": "QF",
        "value": 1.5,
        "message": data["message"],
    }
    assert req.call_args.kwargs["json_body"] == {"family": "QF", "value": 1.5}


@pytest.mark.unit
async def test_lattice_refresh_all_fast_figures():
    with _patch_request(return_value={"launched": ["optics"]}) as req:
        await _fn(lt.lattice_refresh)()
    assert req.call_args.args == ("POST", "/api/refresh")


@pytest.mark.unit
@pytest.mark.parametrize("figure", ["da", "fma"])
async def test_lattice_refresh_verification_figures(figure):
    with _patch_request(return_value={}) as req:
        await _fn(lt.lattice_refresh)(figure=figure)
    assert req.call_args.args == ("POST", "/api/verify")


@pytest.mark.unit
async def test_lattice_refresh_named_figure():
    with _patch_request(return_value={}) as req:
        await _fn(lt.lattice_refresh)(figure="optics")
    assert req.call_args.args == ("POST", "/api/refresh/optics")


@pytest.mark.unit
async def test_lattice_set_baseline_happy():
    with _patch_request(return_value={"tunes": [0.1, 0.2]}):
        result = await _fn(lt.lattice_set_baseline)()
    data = json.loads(result)
    assert data["status"] == "ok"
    assert data["baseline"] == {"tunes": [0.1, 0.2]}


@pytest.mark.unit
async def test_lattice_get_figure_happy():
    with _patch_request(return_value={"data": [], "layout": {}}) as req:
        result = await _fn(lt.lattice_get_figure)(name="optics")
    assert json.loads(result) == {"data": [], "layout": {}}
    assert req.call_args.args == ("GET", "/api/figures/optics")


@pytest.mark.unit
async def test_lattice_get_data_happy():
    with _patch_request(return_value={"s": [0, 1]}) as req:
        result = await _fn(lt.lattice_get_data)(name="optics")
    assert json.loads(result) == {"s": [0, 1]}
    assert req.call_args.args == ("GET", "/api/data/optics")


@pytest.mark.unit
async def test_lattice_get_settings_happy():
    with _patch_request(return_value={"da": {"n_angles": 25}}):
        result = await _fn(lt.lattice_get_settings)()
    assert json.loads(result)["da"]["n_angles"] == 25


@pytest.mark.unit
async def test_lattice_update_settings_happy():
    with _patch_request(return_value={"da": {"n_angles": 25}}) as req:
        result = await _fn(lt.lattice_update_settings)(settings={"da": {"n_angles": 25}})
    assert json.loads(result)["da"]["n_angles"] == 25
    method, path = req.call_args.args
    assert (method, path) == ("PUT", "/api/settings")
    assert req.call_args.kwargs["json_body"] == {"settings": {"da": {"n_angles": 25}}}


@pytest.mark.unit
async def test_lattice_clear_baseline_happy():
    with _patch_request(return_value={"cleared": True}) as req:
        result = await _fn(lt.lattice_clear_baseline)()
    assert json.loads(result) == {"cleared": True}
    assert req.call_args.args == ("DELETE", "/api/baseline")


# ---------------------------------------------------------------------------
# Error mapping — every tool wires the same httpx-failure -> envelope contract
# ---------------------------------------------------------------------------

# (tool, call-kwargs) for every tool. Error handling is per-tool boilerplate, so
# each entry proves that tool's own except-block is wired, not just one exemplar.
_ALL_TOOLS = [
    ("lattice_init", {"lattice_path": "x.m"}),
    ("lattice_state", {}),
    ("lattice_set_param", {"family": "QF", "value": 1.0}),
    ("lattice_refresh", {}),
    ("lattice_set_baseline", {}),
    ("lattice_get_figure", {"name": "optics"}),
    ("lattice_get_data", {"name": "optics"}),
    ("lattice_get_settings", {}),
    ("lattice_update_settings", {"settings": {}}),
    ("lattice_clear_baseline", {}),
]

# Tools with an explicit httpx.HTTPStatusError branch (mapped to lattice_error).
_HTTP_ERROR_TOOLS = [
    ("lattice_init", {"lattice_path": "x.m"}),
    ("lattice_set_param", {"family": "QF", "value": 1.0}),
    ("lattice_get_figure", {"name": "optics"}),
    ("lattice_get_data", {"name": "optics"}),
    ("lattice_update_settings", {"settings": {}}),
]


@pytest.mark.unit
@pytest.mark.parametrize("tool_name, kwargs", _ALL_TOOLS)
async def test_connect_error_maps_to_service_unavailable(tool_name, kwargs):
    """A dashboard that isn't running yields a service_unavailable envelope."""
    with _patch_request(side_effect=httpx.ConnectError("refused")):
        with assert_raises_error(error_type="service_unavailable"):
            await _fn(getattr(lt, tool_name))(**kwargs)


@pytest.mark.unit
@pytest.mark.parametrize("tool_name, kwargs", _ALL_TOOLS)
async def test_generic_exception_maps_to_lattice_error(tool_name, kwargs):
    """An unexpected error falls through to a lattice_error envelope."""
    with _patch_request(side_effect=RuntimeError("kaboom")):
        with assert_raises_error(error_type="lattice_error") as ctx:
            await _fn(getattr(lt, tool_name))(**kwargs)
    assert "kaboom" in ctx["envelope"]["error_message"]


@pytest.mark.unit
@pytest.mark.parametrize("tool_name, kwargs", _HTTP_ERROR_TOOLS)
async def test_http_status_error_maps_to_lattice_error(tool_name, kwargs):
    """A non-2xx dashboard response surfaces the body text in a lattice_error."""
    with _patch_request(side_effect=_http_status_error(text="server said no")):
        with assert_raises_error(error_type="lattice_error") as ctx:
            await _fn(getattr(lt, tool_name))(**kwargs)
    assert "server said no" in ctx["envelope"]["error_message"]


@pytest.mark.unit
@pytest.mark.parametrize("tool_name, kwargs", _ALL_TOOLS)
async def test_tool_error_passes_through_unwrapped(tool_name, kwargs):
    """A ToolError raised inside the request is re-raised as-is, not remapped."""
    from osprey.mcp_server.errors import make_error

    def _raise_tool_error(*_a, **_k):
        make_error("validation_error", "bad input from within")

    with _patch_request(side_effect=_raise_tool_error):
        # error_type stays validation_error -> not swallowed into lattice_error.
        with assert_raises_error(error_type="validation_error"):
            await _fn(getattr(lt, tool_name))(**kwargs)


@pytest.mark.unit
async def test_get_figure_http_error_includes_valid_names_hint():
    """The figure tool's error carries the 'call lattice_refresh first' guidance."""
    with _patch_request(side_effect=_http_status_error(status=404, text="not computed")):
        with assert_raises_error(error_type="lattice_error") as ctx:
            await _fn(lt.lattice_get_figure)(name="optics")
    assert any("lattice_refresh" in s for s in ctx["envelope"]["suggestions"])
