"""The hub page says where its terminal assets come from.

terminal.js reads ``<html data-vendor>`` to pick its boot-failure sentence; the
attribute is present only in CDN mode, so an offline image's notice names the
image rather than the CDN.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.interfaces.web_terminal._started_app import terminal_client


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


def test_cdn_mode_stamps_the_vendor_source(workspace_dir, monkeypatch):
    monkeypatch.setenv("OSPREY_OFFLINE", "0")
    with terminal_client(workspace_dir) as c:
        with patch("osprey_connectors.workspace.load_osprey_config", return_value={}):
            body = c.get("/").text
    assert 'data-vendor="cdn"' in body


def test_offline_mode_leaves_the_attribute_out(workspace_dir, monkeypatch):
    monkeypatch.setenv("OSPREY_OFFLINE", "1")
    with terminal_client(workspace_dir) as c:
        body = c.get("/").text
    assert "data-vendor" not in body
