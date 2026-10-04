"""The ``entry_open`` MCP tool: show an entry, and optionally a picture, in the ARIEL panel.

Runs against the keyset harness: one entry with a viewable PNG and a skipped
PDF, plus an entry with no attachments. The web terminal is never reached; the
panel focus request is captured at ``notify_panel_focus``.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from osprey.services.ariel_search.attachments import attachment_id_for
from tests.mcp_server.ariel.conftest import (
    KEYSET_ENTRY_ID,
    KEYSET_SECOND_ENTRY_ID,
    get_tool_fn,
    keyset_entries,
)
from tests.mcp_server.conftest import assert_raises_error

_PNG, _PDF = keyset_entries()[0]["attachments"]
PNG_ID = attachment_id_for(KEYSET_ENTRY_ID, _PNG)
PDF_ID = attachment_id_for(KEYSET_ENTRY_ID, _PDF)


async def _open(**kwargs):
    from osprey.mcp_server.ariel.tools.entry import entry_open

    with patch("osprey.mcp_server.http.notify_panel_focus") as focus:
        raw = await get_tool_fn(entry_open)(**kwargs)
    return json.loads(raw), focus


@pytest.fixture(autouse=True)
def _panel_base(monkeypatch):
    monkeypatch.delenv("ARIEL_WEB_URL", raising=False)


async def test_opens_the_entry_and_focuses_the_panel(keyset_harness):  # noqa: ARG001
    result, focus = await _open(entry_id=KEYSET_SECOND_ENTRY_ID)

    url = f"/panel/ariel/#entry?id={KEYSET_SECOND_ENTRY_ID}"
    assert result["url"] == url
    assert result["opened"] == "entry"
    assert result["entry_id"] == KEYSET_SECOND_ENTRY_ID
    assert result["attachment_id"] is None
    assert url in result["message"]
    focus.assert_called_once_with("ariel", url=url)


async def test_opens_a_viewable_picture_enlarged(keyset_harness):  # noqa: ARG001
    result, focus = await _open(entry_id=KEYSET_ENTRY_ID, attachment_id=PNG_ID)

    url = f"/panel/ariel/#entry?id={KEYSET_ENTRY_ID}&attachment={PNG_ID}"
    assert result["url"] == url
    assert result["opened"] == "entry_and_picture"
    assert result["attachment_id"] == PNG_ID
    focus.assert_called_once_with("ariel", url=url)


async def test_a_picture_that_is_not_viewable_opens_the_entry_only(keyset_harness):  # noqa: ARG001
    result, focus = await _open(entry_id=KEYSET_ENTRY_ID, attachment_id=PDF_ID)

    url = f"/panel/ariel/#entry?id={KEYSET_ENTRY_ID}"
    assert result["url"] == url
    assert result["opened"] == "entry"
    assert "not viewable" in result["message"]
    focus.assert_called_once_with("ariel", url=url)


async def test_the_entry_id_is_url_encoded(keyset_harness, monkeypatch):
    odd = "elog/42 #1&x=2"
    entry = {**keyset_entries()[1], "entry_id": odd}

    async def get_entry(entry_id):
        return entry if entry_id == odd else None

    monkeypatch.setattr(keyset_harness.repository, "get_entry", get_entry)
    result, _ = await _open(entry_id=odd)

    assert result["url"] == "/panel/ariel/#entry?id=elog%2F42%20%231%26x%3D2"


async def test_ariel_web_url_sets_the_base(keyset_harness, monkeypatch):  # noqa: ARG001
    monkeypatch.setenv("ARIEL_WEB_URL", "https://ariel.example")
    result, _ = await _open(entry_id=KEYSET_SECOND_ENTRY_ID)

    assert result["url"] == f"https://ariel.example/#entry?id={KEYSET_SECOND_ENTRY_ID}"


async def test_without_a_web_terminal_the_result_still_carries_the_url(keyset_harness):  # noqa: ARG001
    """``notify_panel_focus`` failing is not an error: the url is the fallback."""
    from osprey.mcp_server.ariel.tools.entry import entry_open

    with patch("osprey.mcp_server.http.notify_panel_focus", side_effect=OSError("refused")):
        result = json.loads(await get_tool_fn(entry_open)(entry_id=KEYSET_SECOND_ENTRY_ID))

    assert result["url"] == f"/panel/ariel/#entry?id={KEYSET_SECOND_ENTRY_ID}"


async def test_unknown_entry_is_not_found(keyset_harness):  # noqa: ARG001
    from osprey.mcp_server.ariel.tools.entry import entry_open

    with (
        patch("osprey.mcp_server.http.notify_panel_focus") as focus,
        assert_raises_error(error_type="not_found"),
    ):
        await get_tool_fn(entry_open)(entry_id="no-such-entry")
    focus.assert_not_called()


async def test_an_attachment_of_another_entry_is_not_found(keyset_harness):  # noqa: ARG001
    from osprey.mcp_server.ariel.tools.entry import entry_open

    with (
        patch("osprey.mcp_server.http.notify_panel_focus") as focus,
        assert_raises_error(error_type="not_found"),
    ):
        await get_tool_fn(entry_open)(entry_id=KEYSET_SECOND_ENTRY_ID, attachment_id=PNG_ID)
    focus.assert_not_called()


@pytest.mark.parametrize("bad_id", ["", "att-xyz", "../att-0123456789ab", "<script>"])
async def test_malformed_attachment_id_is_validation_error_without_echo(keyset_harness, bad_id):  # noqa: ARG001
    from osprey.mcp_server.ariel.tools.entry import entry_open

    with assert_raises_error(error_type="validation_error") as ctx:
        await get_tool_fn(entry_open)(entry_id=KEYSET_ENTRY_ID, attachment_id=bad_id)
    if bad_id:
        assert bad_id not in json.dumps(ctx["envelope"])


@pytest.mark.parametrize("bad_entry", ["", "   "])
async def test_missing_entry_id_is_validation_error(keyset_harness, bad_entry):  # noqa: ARG001
    from osprey.mcp_server.ariel.tools.entry import entry_open

    with assert_raises_error(error_type="validation_error"):
        await get_tool_fn(entry_open)(entry_id=bad_entry)


@pytest.mark.parametrize("keyset_harness", [{"view_enabled": False}], indirect=True)
async def test_view_off_still_opens_a_picture_in_the_panel(keyset_harness):  # noqa: ARG001
    """The switch governs what reaches the model; the panel shows pictures either way."""
    result, _ = await _open(entry_id=KEYSET_ENTRY_ID, attachment_id=PNG_ID)

    assert result["opened"] == "entry_and_picture"


def test_registered_allowed_and_offered_to_the_main_agent():
    import inspect

    from osprey.cli.templates.claude_code import (
        _ARIEL_TOOLS_THE_MAIN_AGENT_MAY_CALL,
        _ariel_read_tools,
    )
    from osprey.mcp_server.ariel import server
    from osprey.registry.mcp import FRAMEWORK_SERVERS

    assert "entry," in inspect.getsource(server.create_server)
    assert "entry_open" in FRAMEWORK_SERVERS["ariel"].permissions_allow
    assert "entry_open" not in FRAMEWORK_SERVERS["ariel"].permissions_ask
    assert "entry_open" in _ARIEL_TOOLS_THE_MAIN_AGENT_MAY_CALL
    assert "entry_open" not in _ariel_read_tools(True)
