"""``web.expert_view: false`` — a deployment that offers no Expert view.

``web.ui_mode`` picks the layout a terminal opens in; ``web.expert_view`` says
whether the Expert view exists on the deployment at all. With it ``false``:

- the resolved mode is ``simple`` whatever ``web.ui_mode`` says;
- the page carries ``<html data-expert-view="off">`` (the pre-paint rung and
  frame-params.js read it) and ``GET /api/panels`` echoes ``expert_view``;
- ``/ws/terminal`` — the Expert view's PTY session — is refused before
  anything is spawned.

With the key absent nothing changes: the default is the Expert view, as before.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
import yaml
from starlette.websockets import WebSocketDisconnect

from osprey.interfaces.web_terminal.app import (
    EXPERT_VIEW_REFUSAL,
    resolve_expert_view,
    resolve_ui_mode,
)
from osprey.interfaces.web_terminal.routes.websocket import WS_CLOSE_EXPERT_VIEW_OFF
from tests.interfaces.web_terminal._started_app import terminal_client


class TestResolveUiModeWithoutExpertView:
    @pytest.mark.parametrize("configured", ["expert", "simple", None, "", "nonsense"])
    def test_the_mode_is_simple_whatever_ui_mode_says(self, configured):
        assert resolve_ui_mode(configured, expert_view=False) == "simple"

    def test_a_stated_expert_is_reported(self, caplog):
        with caplog.at_level(logging.WARNING):
            resolve_ui_mode("expert", expert_view=False)
        assert any("web.expert_view" in record.message for record in caplog.records)

    @pytest.mark.parametrize("configured", ["expert", "simple"])
    def test_the_default_leaves_ui_mode_alone(self, configured):
        assert resolve_ui_mode(configured) == configured


class TestResolveExpertView:
    @staticmethod
    def _write(tmp_path, web):
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump({"web": web}))
        return config_file

    def test_no_config_file_offers_the_expert_view(self):
        assert resolve_expert_view(None) is True

    def test_an_absent_key_offers_the_expert_view(self, tmp_path):
        assert resolve_expert_view(self._write(tmp_path, {"ui_mode": "expert"})) is True

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(False, False), ("false", False), (True, True), (["not", "a", "flag"], True)],
    )
    def test_the_key_is_read_as_a_flag(self, tmp_path, value, expected):
        assert resolve_expert_view(self._write(tmp_path, {"expert_view": value})) is expected

    def test_an_unreadable_config_closes_the_expert_view(self, tmp_path):
        config_file = tmp_path / "config.yml"
        config_file.write_text("web: [unclosed\n  expert_view: {\n")
        assert resolve_expert_view(config_file) is False


# ---- Startup: the page, the payload and the PTY socket ----


def _client(workspace_dir, web):
    """A started app whose config file and ``web:`` section both say *web*."""
    config_file = workspace_dir.parent / "config.yml"
    config_file.write_text(yaml.dump({"web": dict(web)}))
    return terminal_client(
        workspace_dir,
        patch(
            "osprey_connectors.workspace.load_osprey_config",
            return_value={"web": dict(web)},
        ),
        config_path=config_file,
    )


def test_the_page_and_payload_say_simple_with_no_expert_view(workspace_dir):
    with _client(workspace_dir, {"ui_mode": "expert", "expert_view": False}) as client:
        body = client.get("/").text
        payload = client.get("/api/panels").json()

    assert 'data-ui-mode="simple"' in body
    assert 'data-expert-view="off"' in body
    assert payload["ui_mode"] == "simple"
    assert payload["expert_view"] is False


def test_an_absent_key_changes_nothing(workspace_dir):
    with _client(workspace_dir, {"ui_mode": "expert"}) as client:
        body = client.get("/").text
        payload = client.get("/api/panels").json()

    assert 'data-ui-mode="expert"' in body
    assert "data-expert-view" not in body
    assert payload["ui_mode"] == "expert"
    assert payload["expert_view"] is True


def test_the_pty_socket_is_refused_and_nothing_is_spawned(workspace_dir):
    def no_spawn(*args, **kwargs):
        pytest.fail("no PTY may be spawned when web.expert_view is false")

    with (
        patch(
            "osprey.interfaces.web_terminal.routes.websocket._spawn_pooled_pty",
            no_spawn,
        ),
        _client(workspace_dir, {"expert_view": False}) as client,
    ):
        for query in (
            "",
            "?mode=new",
            "?session_id=00000000-0000-4000-8000-000000000000&mode=resume",
        ):
            with client.websocket_connect(f"/ws/terminal{query}") as ws:
                assert ws.receive_json() == {"type": "error", "message": EXPERT_VIEW_REFUSAL}
                with pytest.raises(WebSocketDisconnect) as closed:
                    ws.receive_text()
            assert closed.value.code == WS_CLOSE_EXPERT_VIEW_OFF
