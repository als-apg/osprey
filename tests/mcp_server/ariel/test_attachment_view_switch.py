"""The ``ariel.attachments.view.enabled`` switch on the ARIEL MCP surface.

With the key false the agent is offered B1's surface:

* ``tools/list`` has no ``attachment_view``, and a call that reaches the tool
  anyway is refused with ``not_supported`` naming the key, before any
  repository read;
* listings (search, browse, ``entries_by_ids``) carry no ``attachments``,
  ``attachment_count`` or ``matched_attachment_ids`` and read no
  ``attachment_files`` rows; ``matched_via`` stays;
* ``entry_get`` carries the entry's stored ``attachments`` unchanged.

With the key absent or true, or with no ARIEL context to read it from, the
tool is offered.
"""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

import pytest

from osprey.ariel_attachment_view import VIEW_ENABLED_KEY
from osprey.mcp_server.ariel.server_context import initialize_ariel_context, reset_ariel_context
from tests.mcp_server.ariel.conftest import KEYSET_ENTRY_ID, get_tool_fn, keyset_entries
from tests.mcp_server.conftest import assert_raises_error

VIEW_OFF = pytest.mark.parametrize("keyset_harness", [{"view_enabled": False}], indirect=True)


def _register_tools() -> None:
    from osprey.mcp_server.ariel.tools import (  # noqa: F401
        attachment,
        browse,
        entry,
        keyword_search,
    )


async def _listed_names() -> set[str]:
    from osprey.mcp_server.ariel.server import mcp

    _register_tools()
    return {tool.name for tool in await mcp.list_tools()}


# ---------------------------------------------------------------------------
# The offer
# ---------------------------------------------------------------------------


@VIEW_OFF
async def test_view_off_hides_attachment_view_from_tools_list(keyset_harness):  # noqa: ARG001
    """With the key false ``tools/list`` lacks ``attachment_view`` and nothing else."""
    names = await _listed_names()
    assert "attachment_view" not in names
    assert {"keyword_search", "entry_get", "browse", "entries_by_ids"} <= names


async def test_view_on_lists_attachment_view(keyset_harness):  # noqa: ARG001
    """With the key at its default the tool is listed."""
    assert "attachment_view" in await _listed_names()


@VIEW_OFF
async def test_view_off_hides_attachment_view_on_the_wire(keyset_harness):  # noqa: ARG001
    """An MCP client's ``tools/list`` goes through the same filter."""
    from fastmcp import Client

    from osprey.mcp_server.ariel.server import mcp

    _register_tools()
    async with Client(mcp) as client:
        names = {tool.name for tool in await client.list_tools()}
    assert "attachment_view" not in names
    assert "entry_get" in names


@VIEW_OFF
async def test_view_off_call_is_refused_naming_the_key(keyset_harness):
    """A direct call gets ``not_supported`` with the key in ``details``, reading nothing."""
    from osprey.mcp_server.ariel.tools.attachment import attachment_view

    reads: list[str] = []

    async def _no_read(attachment_id: str) -> Any:
        reads.append(attachment_id)
        raise AssertionError("the repository must not be read with the view off")

    keyset_harness.repository.get_rendition = _no_read
    with assert_raises_error(error_type="not_supported") as ctx:
        await get_tool_fn(attachment_view)(attachment_id="att-0123456789abcdef01234567")
    envelope = ctx["envelope"]
    assert envelope["details"]["key"] == VIEW_ENABLED_KEY
    assert VIEW_ENABLED_KEY in envelope["error_message"]
    assert reads == []


@VIEW_OFF
async def test_view_off_refuses_even_a_malformed_id(keyset_harness):  # noqa: ARG001
    """The switch is checked first, so a malformed id is also answered ``not_supported``."""
    from osprey.mcp_server.ariel.tools.attachment import attachment_view

    with assert_raises_error(error_type="not_supported"):
        await get_tool_fn(attachment_view)(attachment_id="nope")


def test_offered_defaults_to_true_without_a_context():
    """No initialised context reads as the key's default."""
    from osprey.mcp_server.ariel.server import attachment_view_offered

    reset_ariel_context()
    assert attachment_view_offered() is True


def test_offered_defaults_to_true_without_an_ariel_section(tmp_path, monkeypatch):
    """A config with no ``ariel`` section (``.config`` raises) reads as the default."""
    from osprey.mcp_server.ariel.server import attachment_view_offered, mcp

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(json.dumps({"facility": {"name": "test"}}))
    initialize_ariel_context()
    assert attachment_view_offered() is True

    _register_tools()
    names = {tool.name for tool in asyncio.run(mcp.list_tools())}
    assert "attachment_view" in names


def test_list_tools_without_create_server_lists_attachment_view():
    """An import-only walker (no context at all) still lists the tool."""
    from osprey.mcp_server.ariel.server import mcp

    reset_ariel_context()
    _register_tools()
    names = {tool.name for tool in asyncio.run(mcp.list_tools())}
    assert "attachment_view" in names


# ---------------------------------------------------------------------------
# The listings and entry_get
# ---------------------------------------------------------------------------


def _ten_hits(repository: Any) -> None:
    base = keyset_entries()[0]
    hits = []
    for index in range(10):
        entry = copy.deepcopy(base)
        entry["entry_id"] = f"keyset-{index:03d}"
        hits.append((entry, 1.0 - index / 100, ["<b>injection</b>"]))

    async def keyword_search(*args: Any, **kwargs: Any) -> list[Any]:
        return copy.deepcopy(hits)

    repository.keyword_search = keyword_search


@VIEW_OFF
async def test_view_off_ten_entry_search_reads_no_attachment_rows(keyset_harness):
    """A 10-entry search issues zero ``attachment_files`` queries and emits no summaries."""
    from osprey.mcp_server.ariel.tools.keyword_search import keyword_search

    _ten_hits(keyset_harness.repository)
    payload = json.loads(await get_tool_fn(keyword_search)(query="injection", max_results=10))

    assert len(payload["entries"]) == 10
    assert keyset_harness.repository.attachment_row_calls == []
    for item in payload["entries"]:
        assert not {"attachments", "attachment_count", "matched_attachment_ids"} & set(item)


async def test_view_on_ten_entry_search_reads_rows_once(keyset_harness):
    """The contrast: with the view on the page's rows come from one query."""
    from osprey.mcp_server.ariel.tools.keyword_search import keyword_search

    _ten_hits(keyset_harness.repository)
    await get_tool_fn(keyword_search)(query="injection", max_results=10)
    assert len(keyset_harness.repository.attachment_row_calls) == 1


@VIEW_OFF
async def test_view_off_browse_and_batch_read_no_attachment_rows(keyset_harness):
    """``browse`` and ``entries_by_ids`` skip the rows read as well."""
    from osprey.mcp_server.ariel.tools.browse import browse
    from osprey.mcp_server.ariel.tools.entry import entries_by_ids

    browsed = json.loads(await get_tool_fn(browse)())
    batch = json.loads(await get_tool_fn(entries_by_ids)(entry_ids=[KEYSET_ENTRY_ID]))

    assert keyset_harness.repository.attachment_row_calls == []
    for item in browsed["entries"] + batch["entries"]:
        assert not {"attachments", "attachment_count", "matched_attachment_ids"} & set(item)


@VIEW_OFF
async def test_view_off_entry_get_emits_stored_attachments_unchanged(keyset_harness):
    """An item carrying ``caption`` and ``thumbnail_url`` comes out equal to the stored JSONB."""
    from osprey.mcp_server.ariel.tools.entry import entry_get

    stored = [
        {
            "url": "https://elog.example/files/orbit-plot.png",
            "type": "image/png",
            "filename": "orbit-plot.png",
            "caption": "Orbit after the kicker fix.",
            "thumbnail_url": "https://elog.example/files/orbit-plot.thumb.png",
        },
        {"url": "files/report.pdf", "type": "application/pdf", "filename": "report.pdf"},
    ]
    keyset_harness.repository._rows[KEYSET_ENTRY_ID]["attachments"] = copy.deepcopy(stored)

    payload = json.loads(await get_tool_fn(entry_get)(entry_id=KEYSET_ENTRY_ID))

    assert payload["attachments"] == stored
    assert "attachment_count" not in payload
    assert keyset_harness.repository.attachment_row_calls == []


async def test_serialize_entries_view_off_keeps_matched_via_only():
    """``matched_via`` belongs to the search, so it stays; the attachment keys go."""
    from osprey.mcp_server.ariel.tools.search_envelope import serialize_entries

    class _NoRows:
        async def get_attachment_rows(self, *_args: Any) -> Any:
            raise AssertionError("no attachment_files query with the view off")

    entry = keyset_entries()[0]
    entry["_matched_via"] = ["text", "caption"]
    entry["_matched_attachment_ids"] = ["att-0123456789abcdef01234567"]

    (out,) = await serialize_entries(
        [entry],
        text_limit=100,
        attachment_limit=3,
        repository=_NoRows(),
        model_id=None,
        file_source=False,
        view_enabled=False,
    )
    assert out["matched_via"] == ["text", "caption"]
    assert not {"attachments", "attachment_count", "matched_attachment_ids"} & set(out)
