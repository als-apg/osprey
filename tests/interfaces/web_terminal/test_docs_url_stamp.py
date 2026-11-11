"""The hub page carries the docs root the tile ⋯ Help row opens under.

panel-help.js reads ``<html data-docs-url>``; the attribute is absent when
``web.docs_url`` is blank, so an air-gapped deployment shows no Help row.
"""

from __future__ import annotations

import pytest

from tests.interfaces.web_terminal._started_app import terminal_client


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


def test_the_docs_root_is_stamped_on_the_page(workspace_dir):
    with terminal_client(workspace_dir) as c:
        c.app.state.docs_url = "https://docs.example/osprey"
        body = c.get("/").text
    assert 'data-docs-url="https://docs.example/osprey"' in body


def test_a_blank_docs_root_leaves_the_attribute_out(workspace_dir):
    with terminal_client(workspace_dir) as c:
        c.app.state.docs_url = ""
        body = c.get("/").text
    assert "data-docs-url" not in body
