"""``web.terminal: false`` — a deployment that offers no terminal.

With it ``false``:

- the page carries ``<html data-terminal="off">``, which the pre-paint boot
  script and frame-params.js read, so the Agent tile's ⋯ menu has no entry to it;
- ``/ws/terminal`` — the terminal's PTY session — is refused before
  anything is spawned.

With the key absent nothing changes: the terminal is an opt-in away.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import yaml
from starlette.websockets import WebSocketDisconnect

from osprey.interfaces.web_terminal.app import TERMINAL_REFUSAL, resolve_terminal_offered
from osprey.interfaces.web_terminal.routes.websocket import WS_CLOSE_TERMINAL_OFF
from tests.interfaces.web_terminal._started_app import terminal_client


class TestResolveTerminalOffered:
    @staticmethod
    def _write(tmp_path, web):
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump({"web": web}))
        return config_file

    def test_no_config_file_offers_the_terminal(self):
        assert resolve_terminal_offered(None) is True

    def test_an_absent_key_offers_the_terminal(self, tmp_path):
        assert resolve_terminal_offered(self._write(tmp_path, {"theme": "dark"})) is True

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(False, False), ("false", False), (True, True), (["not", "a", "flag"], True)],
    )
    def test_the_key_is_read_as_a_flag(self, tmp_path, value, expected):
        assert resolve_terminal_offered(self._write(tmp_path, {"terminal": value})) is expected

    def test_an_unreadable_config_closes_the_terminal(self, tmp_path):
        config_file = tmp_path / "config.yml"
        config_file.write_text("web: [unclosed\n  terminal: {\n")
        assert resolve_terminal_offered(config_file) is False


def test_the_refusal_names_the_key():
    assert "web.terminal: false" in TERMINAL_REFUSAL


# ---- Startup: the page and the PTY socket ----


def _client(workspace_dir, web):
    """A started app whose config file says *web*."""
    config_file = workspace_dir.parent / "config.yml"
    config_file.write_text(yaml.dump({"web": dict(web)}))
    return terminal_client(workspace_dir, config_path=config_file)


def test_the_page_says_off(workspace_dir):
    with _client(workspace_dir, {"terminal": False}) as client:
        body = client.get("/").text
        payload = client.get("/api/panels").json()

    assert 'data-terminal="off"' in body
    assert "terminal" not in payload


def test_an_absent_key_changes_nothing(workspace_dir):
    with _client(workspace_dir, {"theme": "dark"}) as client:
        body = client.get("/").text

    assert "data-terminal" not in body


def test_the_pty_socket_is_refused_and_nothing_is_spawned(workspace_dir):
    def no_spawn(*args, **kwargs):
        pytest.fail("no PTY may be spawned when web.terminal is false")

    with (
        patch(
            "osprey.interfaces.web_terminal.routes.websocket._spawn_pooled_pty",
            no_spawn,
        ),
        _client(workspace_dir, {"terminal": False}) as client,
    ):
        for query in (
            "",
            "?mode=new",
            "?session_id=00000000-0000-4000-8000-000000000000&mode=resume",
        ):
            with client.websocket_connect(f"/ws/terminal{query}") as ws:
                assert ws.receive_json() == {"type": "error", "message": TERMINAL_REFUSAL}
                with pytest.raises(WebSocketDisconnect) as closed:
                    ws.receive_text()
            assert closed.value.code == WS_CLOSE_TERMINAL_OFF
