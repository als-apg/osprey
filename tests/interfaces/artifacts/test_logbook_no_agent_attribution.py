"""The gallery's send-to-logbook is a human gesture, not agent activity.

Composing a logbook entry is something an operator *clicks*. Announcing it
with :func:`notify_panel_focus` — an agent-source, all-clients broadcast —
would paint agent styling in every connected browser and yank each workspace
to ARIEL because one person hit Submit. These tests pin the honest shape —
navigation is sender-local over the same-origin panel-iframe postMessage
protocol — from both ends:

  - server: submitting performs no web-terminal broadcast at all, and the
    module never reaches for the agent-source notifier;
  - contract: the gallery posts ``osprey:navigate`` to its host only when it
    is actually embedded, and the host listener accepts it under the existing
    same-origin guard with no agent attribution anywhere in the payload.

The contract half asserts against JavaScript source text, so it is a
tripwire, not a behavioral test — it catches a silent revert to the broadcast
without needing a browser. The real two-context behavior (client A gestures,
client B must not move) is covered by the browser suite.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

_MODULE = "osprey.interfaces.artifacts.logbook"


def _artifacts_static(name: str) -> str:
    """Read a file from the artifacts gallery's static/js directory."""
    import osprey.interfaces.artifacts as pkg

    return (Path(pkg.__file__).parent / "static" / "js" / name).read_text()


def _web_terminal_static(name: str) -> str:
    """Read a file from the web terminal's static/js directory."""
    import osprey.interfaces.web_terminal as pkg

    return (Path(pkg.__file__).parent / "static" / "js" / name).read_text()


def _message_listener_body() -> str:
    """Return just the body of app.js's postMessage listener.

    The host's navigate handling is only safe *because* it sits after the
    same-origin bail-out inside this one listener, so the contract tests
    below must reason about a slice of the file rather than the whole file —
    matching both strings anywhere in app.js would pass even if the guard
    lived in an unrelated function.
    """
    src = _web_terminal_static("app.js")

    start = src.index("function initIframePasteBridge")
    # Sections in this file are separated by `/* ---- <name> ---- */` banners.
    end = src.index("/* ----", start + 1)
    return src[start:end]


@pytest.fixture
def app_client(tmp_path):
    from fastapi.testclient import TestClient

    from osprey.interfaces.artifacts.app import create_app

    return TestClient(create_app(workspace_root=tmp_path))


class TestSubmitDoesNotBroadcast:
    """POST /api/logbook/submit must stay silent on the web-terminal channel."""

    @pytest.mark.unit
    def test_submit_posts_nothing_to_the_web_terminal(self, app_client, tmp_path):
        """No web-terminal POST of any kind escapes the submit path.

        Patches both posters rather than one named notifier, so
        re-introducing the broadcast under any helper fails here.
        ``osprey.mcp_server.http`` has two of them and they do not share a
        call path: the fire-and-forget ``post_json`` (panel focus,
        visibility, agent activity) and ``_post_json_with_response``, which
        panel register and panel arrange use to read a status back.
        """
        with (
            patch(f"{_MODULE}.resolve_shared_data_root", return_value=tmp_path),
            patch("osprey.mcp_server.http.post_json") as mock_post,
            patch("osprey.mcp_server.http._post_json_with_response") as mock_post_rr,
        ):
            resp = app_client.post(
                "/api/logbook/submit",
                json={"subject": "Test entry", "details": "Details here."},
            )

        assert resp.status_code == 200, resp.text
        assert mock_post.call_args_list == [], (
            "submit broadcast to the web terminal; a human gallery gesture must not "
            f"reach other clients: {mock_post.call_args_list}"
        )
        assert mock_post_rr.call_args_list == [], (
            "submit broadcast to the web terminal via the with-response poster: "
            f"{mock_post_rr.call_args_list}"
        )

    @pytest.mark.unit
    def test_submit_still_returns_the_draft_url(self, app_client, tmp_path):
        """Dropping the broadcast must not drop the URL the client navigates to.

        The returned URL is now the *only* thing that gets the operator to
        their draft — the gallery hands it to its host, and a standalone
        gallery renders it as the "Open in ARIEL" link.
        """
        with (
            patch(f"{_MODULE}.resolve_shared_data_root", return_value=tmp_path),
            patch("osprey.mcp_server.http.post_json"),
        ):
            resp = app_client.post(
                "/api/logbook/submit",
                json={"subject": "Test", "details": "Details."},
            )

        data = resp.json()
        assert data["url"].startswith("/panel/ariel")
        assert data["draft_id"] in data["url"]

    @pytest.mark.unit
    def test_module_does_not_reference_the_agent_notifier(self):
        """The import and the call site are both gone, not just unreached.

        A leftover import is the shape a revert takes: the name back in
        scope is one line away from being called again. The prose comment
        naming the removed notifier is deliberately allowed — only a binding
        or a call fails here.
        """
        import osprey.interfaces.artifacts.logbook as logbook

        assert not hasattr(logbook, "notify_panel_focus")
        assert "notify_panel_focus(" not in Path(logbook.__file__).read_text()


class TestSenderLocalNavigateContract:
    """The postMessage contract that replaced the broadcast."""

    @pytest.mark.unit
    def test_gallery_posts_navigate_to_its_host(self):
        """The gallery asks its host to navigate, scoped to the same origin."""
        src = _artifacts_static("logbook.js")

        assert "osprey:navigate" in src
        assert "window.parent.postMessage" in src
        assert "window.location.origin" in src, (
            "the navigate postMessage must be origin-scoped, never targeted at '*'"
        )

    @pytest.mark.unit
    def test_gallery_stays_silent_when_not_embedded(self):
        """A standalone gallery has no host and must not post at all."""
        src = _artifacts_static("logbook.js")

        assert "window.parent !== window" in src, (
            "the navigate post must be guarded on actually being embedded; "
            "unguarded, a standalone gallery posts to itself"
        )

    @pytest.mark.unit
    def test_navigate_payload_carries_no_agent_attribution(self):
        """Nothing in the sender's navigate path claims to be the agent."""
        src = _artifacts_static("logbook.js")

        assert 'source: "agent"' not in src
        assert "source: 'agent'" not in src

    @pytest.mark.unit
    def test_host_handles_navigate_behind_the_origin_guard(self):
        """The web terminal accepts the message, same-origin only.

        Containment, not co-occurrence: both strings appearing somewhere in
        app.js would prove nothing, so this slices the message listener and
        requires the origin bail-out to sit inside it and precede the
        navigate branch.
        """
        body = _message_listener_body()

        navigate_at = body.find("osprey:navigate")
        guard_at = body.find("e.origin !== window.location.origin")

        assert navigate_at != -1, "no navigate branch inside the message listener"
        assert guard_at != -1, "no same-origin guard inside the message listener"
        assert guard_at < navigate_at, (
            "the same-origin guard must precede the navigate branch; a navigate "
            "handled ahead of the bail-out would accept cross-origin instructions"
        )

    @pytest.mark.unit
    def test_host_rejects_urls_that_are_not_root_relative(self):
        """A same-origin sender still must not choose the scheme or the host.

        The origin check is necessary but not sufficient: the sender can be
        an agent-authored artifact in a sandboxed panel, and this url reaches
        an iframe src through ``buildEmbedSrc``, which preserves whatever
        scheme it is handed. Verified against the real URL semantics —
        ``javascript:alert(1)`` survives ``buildEmbedSrc`` intact and would
        run in the HOST origin, and ``//evil.example/x`` resolves to a
        cross-origin document. So a leading-slash test alone is a hole and
        the protocol-relative case must be excluded explicitly.
        """
        body = _message_listener_body()

        assert "startsWith('/')" in body, (
            "navigate must require a root-relative url, or a javascript: scheme "
            "reaches an iframe src in the host origin"
        )
        assert "!e.data.url.startsWith('//')" in body, (
            "navigate must reject protocol-relative urls: '//evil.example/x' passes "
            "a leading-slash test and resolves cross-origin"
        )
