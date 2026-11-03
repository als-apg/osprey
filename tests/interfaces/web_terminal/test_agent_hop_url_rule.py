"""The agent-hop url rule, judged by the terminal on every op that takes a url.

CANOPY's screen moves only through its own ``set_workspace`` tool. The agent's
``open_panel`` refuses a url aimed at CANOPY with ``agent_hop_panel_url``, but
that is only the friendlier half: a caller posting to ``/api/panel-op/open``
directly, or registering a runtime panel, meets the same rule in the terminal
(:func:`osprey.interfaces.web_terminal.routes.panels.agent_hop_refusal`).

The sidecar's own address is recognised in every spelling a browser reads as
that listener. And the check after it fails closed: an op admits an http(s)
URL only on the member's own configured backend, so a loopback address the
rule does not recognise is still refused, as ``invalid_url``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.panel_urls import canonical_host, is_loopback_host, same_place
from osprey.interfaces.web_terminal.routes.panel_ops import (
    OpenArrangementOpRequest,
    OpenSetOpRequest,
)
from osprey.interfaces.web_terminal.routes.panels import router
from osprey.profiles.sidecar_panels import CANOPY_PANEL_ID
from osprey.registry.web import panel_url_state_attr

from .conftest import bare_route_app
from .test_panel_nav_guard import _LAN_ADDR, _op_app, _open

SIDECAR_PORT = 45678
SIDECAR_URL = f"http://127.0.0.1:{SIDECAR_PORT}"
ARIEL_UPSTREAM = "http://127.0.0.1:10300"

#: Every spelling a browser (or the resolver) reads as this machine's loopback.
LOOPBACK_SPELLINGS = [
    "127.1",
    "2130706433",
    "0x7f.0.0.1",
    "0177.0.0.1",
    "[::ffff:127.0.0.1]",
    "[::1]",
    "localhost.",
    "foo.localhost",
]

#: Paths that land under CANOPY's mount, as a server reads them.
CANOPY_PATHS = [
    "/panel/canopy/s/GTLView",
    "/panel/canopy/embed/drafts/beam-tune",
    "/u/alice/panel/canopy/s/GTLView",
    "/panel/CANOPY/s/x",
    "/panel/%63anopy/s/x",
    "/panel/%2563anopy/s/x",
    "/panel/ariel/../canopy/s/x",
]

_CANOPY_STATE = panel_url_state_attr(CANOPY_PANEL_ID)


def _app(**state):
    defaults = {
        "enabled_panels": {"ariel", CANOPY_PANEL_ID},
        "visible_panels": ["ariel", CANOPY_PANEL_ID],
        "ariel_server_url": ARIEL_UPSTREAM,
        _CANOPY_STATE: SIDECAR_URL,
    }
    defaults.update(state)
    return _op_app(**defaults)


def _assert_refused(app, resp, frame, reason: str) -> None:
    assert resp.status_code == 422, resp.text
    assert resp.json()["reason"] == reason
    assert frame is None
    assert app.state.active_panel is None
    assert app.state.dock_layout_rev == 3


class TestTheBrowsersReading:
    @pytest.mark.parametrize("host", LOOPBACK_SPELLINGS)
    def test_every_spelling_is_loopback(self, host):
        assert is_loopback_host(host)

    @pytest.mark.parametrize("host", LOOPBACK_SPELLINGS[:4])
    def test_the_legacy_ipv4_spellings_are_the_dotted_address(self, host):
        assert canonical_host(host) == "127.0.0.1"

    @pytest.mark.parametrize("host", ["grafana.lan", "10.0.0.5", "09.1", "1.2.3.4.5", "localhostx"])
    def test_a_name_or_a_malformed_number_is_not_loopback(self, host):
        assert not is_loopback_host(host)

    @pytest.mark.parametrize("host", LOOPBACK_SPELLINGS)
    def test_every_spelling_on_the_sidecars_port_is_the_sidecars_listener(self, host):
        assert same_place(
            ("http", canonical_host(host), SIDECAR_PORT), ("http", "127.0.0.1", SIDECAR_PORT)
        )

    def test_another_port_or_scheme_is_another_listener(self):
        sidecar = ("http", "127.0.0.1", SIDECAR_PORT)
        assert not same_place(("http", "127.0.0.1", SIDECAR_PORT + 1), sidecar)
        assert not same_place(("https", "127.0.0.1", SIDECAR_PORT), sidecar)


class TestOpenPostedDirectly:
    """``POST /api/panel-op/open`` as any caller sends it, without the MCP tool."""

    @pytest.mark.parametrize(
        "url", ["/panel/canopy/s/GTLView", "/s/GTLView", "https://example.org/"]
    )
    def test_any_url_for_canopy_itself_is_refused(self, url):
        app = _app()

        resp, frame = _open(app, url, answer=False, member=CANOPY_PANEL_ID)

        _assert_refused(app, resp, frame, "agent_hop_panel_url")
        assert "workspace tool" in resp.json()["detail"]

    @pytest.mark.parametrize("url", CANOPY_PATHS)
    def test_a_path_under_canopys_mount_is_refused_for_another_member(self, url):
        app = _app()

        resp, frame = _open(app, url, answer=False)

        _assert_refused(app, resp, frame, "agent_hop_panel_url")

    @pytest.mark.parametrize("host", ["127.0.0.1", *LOOPBACK_SPELLINGS])
    def test_the_sidecars_address_in_any_spelling_is_refused(self, host):
        app = _app()

        resp, frame = _open(app, f"http://{host}:{SIDECAR_PORT}/s/GTLView", answer=False)

        _assert_refused(app, resp, frame, "agent_hop_panel_url")
        assert "sidecar" in resp.json()["detail"]

    @pytest.mark.parametrize("host", LOOPBACK_SPELLINGS)
    def test_a_loopback_address_no_panel_is_configured_on_fails_closed(self, host):
        """The rule does not recognise it; the member guard then refuses it all the same."""
        app = _app()

        resp, frame = _open(app, f"http://{host}:10301/", answer=False)

        _assert_refused(app, resp, frame, "invalid_url")

    @pytest.mark.parametrize("host", LOOPBACK_SPELLINGS)
    def test_with_the_sidecar_down_its_address_is_still_refused(self, host):
        app = _app(**{_CANOPY_STATE: None})

        resp, frame = _open(app, f"http://{host}:{SIDECAR_PORT}/", answer=False)

        _assert_refused(app, resp, frame, "invalid_url")

    @pytest.mark.parametrize("host", LOOPBACK_SPELLINGS[:4])
    def test_another_spelling_of_the_members_own_backend_is_refused(self, host):
        """Only the configured spelling of the member's backend is its server."""
        app = _app()

        resp, frame = _open(app, f"http://{host}:10300/", answer=False)

        _assert_refused(app, resp, frame, "invalid_url")

    def test_canopy_opens_without_a_url(self):
        app = _app()

        resp, frame = _open(app, None, member=CANOPY_PANEL_ID)

        assert resp.status_code == 200, resp.text
        assert frame["args"] == {"member": CANOPY_PANEL_ID}

    def test_another_members_own_backend_still_opens(self):
        app = _app()

        resp, frame = _open(app, f"{ARIEL_UPSTREAM}/#draft")

        assert resp.status_code == 200, resp.text
        assert frame["args"]["url"] == "/panel/ariel/#draft"


class TestTheOtherOps:
    def test_open_set_and_open_arrangement_take_no_url(self):
        """They name panels by id, so there is no url for the rule to judge."""
        assert "url" not in OpenSetOpRequest.model_fields
        assert "url" not in OpenArrangementOpRequest.model_fields

    @pytest.mark.parametrize("host", ["127.0.0.1", *LOOPBACK_SPELLINGS])
    def test_a_registration_on_the_sidecars_address_is_refused_by_name(self, host):
        client = TestClient(
            bare_route_app(
                router,
                enabled_panels={"ariel", CANOPY_PANEL_ID},
                custom_panels=[],
                active_panel=None,
                visible_panels=["ariel"],
                allow_runtime_panels=True,
                runtime_panel_allowlist=None,
                **{_CANOPY_STATE: SIDECAR_URL},
            )
        )

        with patch(
            "osprey.interfaces.web_terminal.routes.panels.socket.getaddrinfo",
            return_value=_LAN_ADDR,
        ):
            resp = client.post(
                "/api/panels/register",
                json={"id": "x", "label": "X", "url": f"http://{host}:{SIDECAR_PORT}/"},
            )

        assert resp.status_code == 422, resp.text
        assert resp.json()["detail"].startswith("agent_hop_panel_url: ")
        assert client.app.state.custom_panels == []
