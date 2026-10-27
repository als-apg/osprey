"""A panel's ``read_timeout_s`` bounds the reads of its proxied requests.

``web.panels.<id>.read_timeout_s`` has three states, and the terminal tells
them apart: **absent** leaves the panel on the shared proxy client's 30 s
default, an **int** bounds that panel's reads to so many seconds, and ``null``
lifts the bound. Absent and ``null`` are different statements — the first says
nothing, the second asks for no limit — so the loader keeps the key off a panel
that did not write it rather than reading both as ``None``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from osprey.interfaces.web_terminal.app import _load_panel_config

from ._proxy_fakes import panel_app


def _load(spec: dict) -> dict:
    """The custom panel ``_load_panel_config`` derives from one ``slow`` block."""
    block = {"label": "SLOW", "url": "http://127.0.0.1:9300", **spec}
    with patch(
        "osprey_connectors.workspace.load_osprey_config",
        return_value={"web": {"panels": {"slow": block}}},
    ):
        _enabled, custom, _default = _load_panel_config()
    (panel,) = custom
    return panel


class TestPanelSpec:
    def test_absent_is_not_projected(self):
        assert "readTimeoutS" not in _load({})

    def test_an_int_is_carried(self):
        assert _load({"read_timeout_s": 120})["readTimeoutS"] == 120

    def test_null_is_carried_as_unbounded(self):
        panel = _load({"read_timeout_s": None})
        assert "readTimeoutS" in panel
        assert panel["readTimeoutS"] is None

    @pytest.mark.parametrize("value", ["120", 0, -5, True])
    def test_a_value_that_is_not_a_timeout_is_refused(self, value):
        with pytest.raises(ValueError, match=r"web\.panels\.slow\.read_timeout_s"):
            _load({"read_timeout_s": value})


def _proxied_kwargs(workspace_dir, declared: dict) -> dict:
    """The keyword arguments the proxy hands the shared client for one GET to *declared*."""
    seen: dict = {}

    async def fake_request(**kwargs):
        seen.update(kwargs)
        return httpx.Response(200, text="ok", headers={"content-type": "text/plain"})

    for app, client in panel_app(workspace_dir, [declared]):
        seen["client_timeout"] = app.state.proxy_client.timeout
        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)
        assert client.get("/panel/slow/data").status_code == 200
    return seen


def _declared(panel: dict) -> dict:
    """A config-declared ``slow`` panel carrying *panel* on top of its URL."""
    return {
        "id": "slow",
        "label": "SLOW",
        "url": "http://127.0.0.1:9300",
        **panel,
        "configDefined": True,
    }


class TestProxiedRequest:
    def test_absent_uses_the_shared_30s_client(self, workspace_dir):
        seen = _proxied_kwargs(workspace_dir, _declared({}))
        assert "timeout" not in seen
        assert seen["client_timeout"].read == 30.0

    def test_an_int_bounds_the_read(self, workspace_dir):
        seen = _proxied_kwargs(workspace_dir, _declared({"readTimeoutS": 120}))
        timeout = seen["timeout"]
        assert timeout.read == 120
        # Only the read bound is the panel's; the rest stays the shared client's.
        shared = seen["client_timeout"]
        assert (timeout.connect, timeout.write, timeout.pool) == (
            shared.connect,
            shared.write,
            shared.pool,
        )

    def test_null_is_unbounded(self, workspace_dir):
        seen = _proxied_kwargs(workspace_dir, _declared({"readTimeoutS": None}))
        assert seen["timeout"].read is None
        assert seen["timeout"].connect == seen["client_timeout"].connect

    def test_a_runtime_panel_keeps_the_shared_client(self, workspace_dir):
        """A timeout is configuration: only a config-declared panel carries one."""
        declared = {"id": "slow", "label": "SLOW", "url": "http://127.0.0.1:9300"}
        declared["readTimeoutS"] = None
        seen = _proxied_kwargs(workspace_dir, declared)
        assert "timeout" not in seen
